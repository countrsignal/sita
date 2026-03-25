import dgl
import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Dict, List, Optional, Tuple

from .random_rotations import random_rotations


################################################################################
# functions
################################################################################

def get_batch_indices(g: dgl.DGLGraph, data_type: str = "node") -> torch.Tensor:
    if data_type == "node":
        return torch.arange(g.batch_size, device=g.device).repeat_interleave(g.batch_num_nodes())
    elif data_type == "edge":
        return torch.arange(g.batch_size, device=g.device).repeat_interleave(g.batch_num_edges())
    else:
        raise ValueError(f"Invalid data type: {data_type}. Valid types: node, edge")


def flatten_along_spatial(x: torch.Tensor, g: dgl.DGLGraph) -> torch.Tensor:
    """
    Args:
        x: Tensor of atomic coordinates, shape (total_atoms, coord_dim)
        g: DGLGraph of molecules

    Returns:
        Tensor of atomic coordinates flattened per molecule, shape (batch_size, max_num_nodes * coord_dim)
    """
    if x.dim() not in (2, 3):
        raise ValueError("Expected atom-wise tensor with shape (total_atoms, coord_dim) or (total_atoms, coord_dim, *)")

    batch_index = get_batch_indices(g)
    batch_size = g.batch_size
    num_nodes_per_graph = g.batch_num_nodes().to(x.device)
    if x.size(0) != int(num_nodes_per_graph.sum().item()):
        raise ValueError("Number of atoms in tensor does not match graph node count")

    tail_shape = x.shape[1:]
    max_num_nodes = int(num_nodes_per_graph.max().item())
    flat = x.new_zeros((batch_size, max_num_nodes) + tail_shape)

    node_ids = torch.arange(x.size(0), device=x.device)
    offsets = torch.cumsum(num_nodes_per_graph, dim=0) - num_nodes_per_graph
    local_index = node_ids - offsets[batch_index]

    flat[batch_index, local_index] = x

    return flat.reshape(batch_size, -1)


def flatten_along_batch(x: torch.Tensor, g: dgl.DGLGraph) -> torch.Tensor:
    """
    Args:
        x: Tensor of atomic coordinates, shape (batch_size, max_num_nodes * coord_dim)
        g: DGLGraph of molecules
    
    Returns:
        x_flattened: Tensor of atomic coordinates, shape (total_atoms, coord_dim)
    """
    if x.dim() not in (2, 3):
        raise ValueError("Expected input tensor with 2 or 3 dimensions")

    batch_size = g.batch_size
    num_nodes_per_graph = g.batch_num_nodes().to(x.device)
    if x.size(0) != batch_size:
        raise ValueError("Batch dimension of tensor does not match graph batch size")

    max_num_nodes = int(num_nodes_per_graph.max().item())

    if x.dim() == 2:
        if x.size(1) % max_num_nodes != 0:
            raise ValueError("Feature dimension is not divisible by max number of nodes")
        coord_dim = x.size(1) // max_num_nodes
        x_reshaped = x.reshape(batch_size, max_num_nodes, coord_dim)
    else:
        x_reshaped = x
        coord_dim = x.size(2)

    mask = (
        torch.arange(max_num_nodes, device=x.device)
        .unsqueeze(0)
        .lt(num_nodes_per_graph.unsqueeze(1))
    )

    return x_reshaped[mask]


def nodes_to_padded_tensor(x: torch.Tensor, g: dgl.DGLGraph) -> torch.Tensor:
    """
    Convert node-wise features into a batched tensor with zero padding.

    Args:
        x: Tensor of node features, shape (total_nodes, 3)
        g: Batched DGLGraph describing the molecules

    Returns:
        Tensor of shape (batch_size, max_num_nodes, 3) where shorter graphs are
        padded with zeros.
    """
    if x.dim() != 2 or x.size(-1) != 3:
        raise ValueError(f"Expected node tensor of shape (total_nodes, 3), got {tuple(x.shape)}")

    batch_size = g.batch_size
    num_nodes_per_graph = g.batch_num_nodes().to(x.device)

    if x.size(0) != int(num_nodes_per_graph.sum().item()):
        raise ValueError("Node tensor length does not match total number of graph nodes")

    max_num_nodes = int(num_nodes_per_graph.max().item())
    padded = x.new_zeros(batch_size, max_num_nodes, 3)

    batch_index = get_batch_indices(g)
    node_ids = torch.arange(x.size(0), device=x.device)
    offsets = torch.cumsum(num_nodes_per_graph, dim=0) - num_nodes_per_graph
    local_index = node_ids - offsets[batch_index]

    padded[batch_index, local_index] = x

    return padded


def edges_to_pair_tensor(
    x: torch.Tensor, g: dgl.DGLGraph
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert edge features from a batched DGLGraph into a dense pairwise tensor
    for use as an attention bias in scaled dot-product attention over nodes.

    The returned tensor has shape (batch, N_max, N_max, feat_dim) and can be
    linearly projected to (batch, N_max, N_max, num_heads), then permuted to
    (batch, num_heads, N_max, N_max) and added to the QK^T / sqrt(d) logits.

    Args:
        x: Edge features, shape (total_edges, feat_dim).
        g: Batched DGLGraph whose edges correspond row-wise to *x*.

    Returns:
        pair: Dense pair tensor of shape
              (batch_size, max_num_nodes, max_num_nodes, feat_dim),
              zero-filled where no edge exists or where nodes are padding.
        pair_padding_mask: Boolean tensor of shape
              (batch_size, max_num_nodes, max_num_nodes).
              True at positions where *either* the row or column index
              falls outside the graph's real node count (i.e. padding).
    """
    if x.dim() != 2:
        raise ValueError(
            f"Expected edge features of shape (total_edges, feat_dim), got {tuple(x.shape)}"
        )

    batch_size = g.batch_size
    num_nodes_per_graph = g.batch_num_nodes().to(x.device)
    num_edges_per_graph = g.batch_num_edges().to(x.device)

    if x.size(0) != int(num_edges_per_graph.sum().item()):
        raise ValueError(
            "Edge tensor length does not match total number of graph edges"
        )

    max_num_nodes = int(num_nodes_per_graph.max().item())
    feat_dim = x.size(1)

    pair = x.new_zeros(batch_size, max_num_nodes, max_num_nodes, feat_dim)

    src, dst = g.edges()
    src, dst = src.to(x.device), dst.to(x.device)

    edge_batch = get_batch_indices(g, data_type="edge")
    node_offsets = torch.cumsum(num_nodes_per_graph, dim=0) - num_nodes_per_graph
    local_src = src - node_offsets[edge_batch]
    local_dst = dst - node_offsets[edge_batch]

    pair[edge_batch, local_src, local_dst] = x

    node_pad = (
        torch.arange(max_num_nodes, device=x.device)[None, :]
        >= num_nodes_per_graph[:, None]
    )
    pair_padding_mask = node_pad.unsqueeze(2) | node_pad.unsqueeze(1)

    return pair, pair_padding_mask


def scatter_center_mol(xyz: torch.Tensor, g: dgl.DGLGraph) -> torch.Tensor:
    """
    Center coordinates at the origin for each molecule using torch.scatter operations.
    
    For each molecule: xyz_centered = xyz - mean(xyz)
    
    Args:
        xyz: Tensor of atomic coordinates, shape (total_atoms, 3)
        g  : DGLGraph of molecules
    
    Returns:
        xyz_centered: Coordinates centered at origin, shape (total_atoms, 3)
    """
    device = xyz.device
    num_molecules = g.batch_size

    # Expand batch_index for scatter operations on 3D coordinates
    batch_index = get_batch_indices(g)
    batch_index_expanded = batch_index.unsqueeze(-1).expand(-1, 3)  # (total_atoms, 3)
    
    # Sum coordinates per molecule using scatter_add
    sum_xyz = torch.zeros((num_molecules, 3), device=device)
    sum_xyz.scatter_add_(0, batch_index_expanded, xyz)
    
    # Count atoms per molecule using scatter_add
    counts = torch.zeros(num_molecules, device=device)
    ones = torch.ones(xyz.shape[0], device=device)
    counts.scatter_add_(0, batch_index, ones)
    
    # Calculate mean coordinates for each molecule
    mean_xyz = sum_xyz / counts.unsqueeze(-1)  # (num_molecules, 3)
    
    # Gather the mean for each atom based on its batch index
    mean_per_atom = torch.gather(
        mean_xyz, 
        0, 
        batch_index_expanded
    )  # (total_atoms, 3)
    
    # Center coordinates
    xyz_centered = xyz - mean_per_atom
    
    return xyz_centered


def dgl_nodes_to_padded_tensor(graph: dgl.DGLGraph, feat_key: str = "x") -> torch.Tensor:
    """
    Convert a batched DGLGraph's node features into a padded tensor.

    Args:
        graph: A batched DGLGraph (homogeneous) where graph.ndata[feat_key] has
               shape [batch_size * num_nodes, 3].
        feat_key: Node feature key to read (default: "x").

    Returns:
        padded: Tensor of shape [batch_size, max_num_nodes, 3], zero-padded where needed.
        padding_mask: Boolean tensor of shape [batch_size, max_num_nodes],
              where True indicates padded values.
    """
    # Node features: [total_nodes, 3]
    x = graph.ndata[feat_key]
    if x.ndim != 2 or x.size(-1) != 3:
        raise ValueError(f"Expected node features of shape [N, 3], got {tuple(x.shape)}")

    # Per-graph node counts: length == batch_size
    # (For homogeneous batched graphs, DGL provides this.)
    num_nodes_per_graph = graph.batch_num_nodes()  # Tensor[int]

    if len(num_nodes_per_graph) == 0:
        raise ValueError("Empty batch")

    # Split the concatenated node features into per-graph chunks
    chunks = torch.split(x, num_nodes_per_graph.tolist(), dim=0)  # list of [n_i, 3] tensors

    # Pad to [batch_size, max_num_nodes, 3]
    padded = pad_sequence(chunks, batch_first=True)  # zeros used for padding

    # Build mask (False for real nodes, True for padded)
    max_num_nodes = padded.size(1)
    padding_mask = torch.arange(max_num_nodes, device=x.device)[None, :] > num_nodes_per_graph[:, None]

    return padded, padding_mask


def rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count # NOTE: As long as D_max > D_count, this will > 1.0
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


################################################################################
# classes
################################################################################


class GraphAdapter:

    def __init__(self, g: dgl.DGLGraph):
        self.batch_size = g.batch_size
        self.num_nodes_per_graph = g.batch_num_nodes()
        self.num_edges_per_graph = g.batch_num_edges()
        self.batch_ids_nodes = get_batch_indices(g, data_type="node")
        self.batch_ids_edges = get_batch_indices(g, data_type="edge")
        self.edges = g.edges()
        self.device = g.device
    
    @classmethod
    def adapt_and_pad(cls, g: dgl.DGLGraph, *args, **kwargs) -> Tuple["GraphAdapter", Dict[str, torch.Tensor]]:
        adapter = cls(g)
        padded = adapter.graph_to_padded_tensor(g, *args, **kwargs)
        return adapter, padded

    @torch.no_grad()
    def apply_random_rotations(self, g: dgl.DGLGraph, node_keys: Optional[List[str]] = None) -> dgl.DGLGraph:
        """
        Apply random rotations to the coordinates of the graph.

        Generates one random SO(3) rotation per molecule in the batch and
        applies it to every 3-D vector node feature (any ndata with shape
        (total_atoms, 3)), e.g. noisy coordinates ``xt``.
        """
        if node_keys is None:
            node_keys = ["xt"]

        coord_keys = [k for k in node_keys if k in g.ndata and g.ndata[k].dim() == 2 and g.ndata[k].size(-1) == 3]
        if not coord_keys:
            return g

        device = g.device
        dtype = g.ndata[coord_keys[0]].dtype
        R = random_rotations(self.batch_size, dtype=dtype, device=device)
        R_per_atom = R[self.batch_ids_nodes.to(device)]

        for key in coord_keys:
            g.ndata[key] = torch.einsum('nd,nds->ns', g.ndata[key], R_per_atom)

        return g

    def compute_rbf_edge_features(
        self,
        padded_coords: torch.Tensor,
        D_min: float = 0.,
        D_max: float = 20.,
        D_count: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """
        Build RBF-encoded pairwise distance features and displacement vectors
        from an already-padded coordinate tensor.

        Args:
            padded_coords: Padded node coordinates, shape
                ``(batch_size, max_num_nodes, 3)``.
            D_min: Minimum distance for RBF centres.
            D_max: Maximum distance for RBF centres.
            D_count: Number of RBF basis functions.

        Returns:
            rbf_features: Dense pairwise RBF features, shape
                ``(batch_size, max_num_nodes, max_num_nodes, D_count)``.
                Padded positions are zeroed out.
            displacements: Unit-length pairwise displacement vectors
                ``(x_j - x_i) / ||x_j - x_i||``, shape
                ``(batch_size, max_num_nodes, max_num_nodes, 3)``.
                Padded positions are zeroed out.
            pair_mask: Boolean mask, shape
                ``(batch_size, max_num_nodes, max_num_nodes)``.
                ``True`` where either the row or column index falls outside
                the molecule's real node count (i.e. padding).
        """
        # (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N, 3)
        displacements = padded_coords.unsqueeze(2) - padded_coords.unsqueeze(1)
        distances = displacements.norm(dim=-1)  # (B, N, N)
        displacements = displacements / (distances.unsqueeze(-1) + 1e-8)

        rbf_features = rbf(distances, D_min=D_min, D_max=D_max, D_count=D_count)

        max_n = padded_coords.size(1)
        node_pad = (
            torch.arange(max_n, device=self.device).unsqueeze(0)
            >= self.num_nodes_per_graph.to(self.device).unsqueeze(1)
        )
        pair_mask = node_pad.unsqueeze(2) | node_pad.unsqueeze(1)
        pair_mask = pair_mask | torch.eye(max_n, device=self.device, dtype=torch.bool).unsqueeze(0)
        rbf_features = rbf_features.masked_fill(pair_mask.unsqueeze(-1), 0.0)
        displacements = displacements.masked_fill(pair_mask.unsqueeze(-1), 0.0)

        return rbf_features, displacements, pair_mask

    @torch.no_grad()
    def graph_to_padded_tensor(
        self,
        g: dgl.DGLGraph,
        coord_key: str = "xt",
        target_key: Optional[str] = "vt",
        feat_key_nodes: str = "attr",
        feat_key_edges: str = "attr",
        D_min: float = 0.,
        D_max: float = 20.,
        D_count: int = 16,
        use_rbf: bool = False,
        return_sigma_t: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Pad node and edge features from a batched DGLGraph into dense tensors,
        appending RBF-encoded pairwise distances to the edge features.

        Args:
            g: Batched DGLGraph.
            coord_key: Key in ``g.ndata`` for 3-D coordinates used by the
                RBF distance encoding.
            target_key: Key in ``g.ndata`` for the regression targets.
                Set to ``None`` at inference time to omit
                ``padded_targets`` from the returned tuple.
            feat_key_nodes: Key in ``g.ndata`` for the node features.
            feat_key_edges: Key in ``g.edata`` for the edge features.
            D_min: Minimum distance for RBF centres.
            D_max: Maximum distance for RBF centres.
            D_count: Number of RBF basis functions.
            use_rbf: If ``True`` (default), append RBF distance features and
                displacement vectors to the edge features.  When ``False``,
                the edge tensor keeps its original feature dimension.
            return_sigma_t: If ``True``, also return the padded noise scale
                ``sigma_t`` from ``g.ndata['sigma_t']``.
        Returns:
            padded_targets: ``(batch_size, N_max, 3)`` — regression targets
                from ``g.ndata[target_key]`` (omitted when
                ``target_key is None``).
            padded_times: ``(batch_size, N_max, 1)`` — diffusion times from
                ``g.ndata['t']``.
            padded_atom_index: ``(batch_size, N_max, 1)`` — per-atom type
                indices from ``g.ndata['atom_index']``.
            padded_nodes: ``(batch_size, N_max, node_feat_dim)``.
            padded_edges: ``(batch_size, N_max, N_max, edge_feat_dim [+ D_count + 3])``
                — when ``use_rbf=True`` the RBF distance features and
                displacement vectors are concatenated along the last axis.
            node_mask: ``(batch_size, N_max)`` — ``True`` at padding positions.
            pair_mask: ``(batch_size, N_max, N_max)`` — ``True`` at padding
                positions.
            padded_sigma_t: ``(batch_size, N_max, 1)`` — noise scale
                (only included when ``return_sigma_t=True``).
        """
        node_feats = g.ndata[feat_key_nodes]  # (total_nodes, node_feat_dim)
        max_n = int(self.num_nodes_per_graph.max().item())

        # Scatter node features into a dense (batch_size, N_max, node_feat_dim)
        # tensor. `offsets` gives the starting global index of each graph's
        # nodes, and `local_idx` converts global node indices to positions
        # within each graph (0 .. n_i-1).
        padded_nodes = node_feats.new_zeros(
            self.batch_size, max_n, node_feats.size(-1)
        )
        offsets = torch.cumsum(self.num_nodes_per_graph, dim=0) - self.num_nodes_per_graph
        local_idx = (
            torch.arange(node_feats.size(0), device=self.device)
            - offsets[self.batch_ids_nodes]
        )
        padded_nodes[self.batch_ids_nodes, local_idx] = node_feats

        # Pad per-node atom type indices into (batch_size, N_max, 1).
        atom_index = g.ndata["atom_index"]
        padded_atom_index = atom_index.new_zeros(self.batch_size, max_n, 1)
        padded_atom_index[self.batch_ids_nodes, local_idx] = atom_index.unsqueeze(-1)

        # Boolean mask marking padding positions as True so that attention
        # layers can ignore them. Shape: (batch_size, N_max).
        node_mask = (
            torch.arange(max_n, device=self.device).unsqueeze(0)
            >= self.num_nodes_per_graph.unsqueeze(1)
        )

        # Convert sparse edge features to a dense pair tensor
        # (batch_size, N_max, N_max, edge_feat_dim).
        padded_edges, pair_mask = edges_to_pair_tensor(g.edata[feat_key_edges], g)

        padded_coords = nodes_to_padded_tensor(g.ndata[coord_key], g)

        if use_rbf:
            rbf_feats, displacements, rbf_mask = self.compute_rbf_edge_features(
                padded_coords, D_min=D_min, D_max=D_max, D_count=D_count,
            )
            pair_mask = rbf_mask
            padded_edges = torch.cat([padded_edges, rbf_feats, displacements], dim=-1)

        # pair_mask from edges_to_pair_tensor only covers padding; add diagonal
        diag = torch.eye(max_n, device=self.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask | diag

        if target_key is not None:
            padded_targets = nodes_to_padded_tensor(g.ndata[target_key], g)

        # Pad per-node flow / diffusion times into (batch_size, N_max, 1), reusing the
        # same local_idx mapping computed above for the node features.
        times = g.ndata["t"]
        padded_times = times.new_zeros(self.batch_size, max_n, 1)
        padded_times[self.batch_ids_nodes, local_idx] = times

        # NOTE: the MASK TENSORS for both nodes and edges indicate
        #       padding positions as FALSE and non-padding positions as TRUE
        result = (padded_times, padded_coords, padded_nodes, padded_atom_index, padded_edges, ~node_mask, ~pair_mask)
        if target_key is not None:
            result = (padded_targets,) + result

        if return_sigma_t:
            # Pad per-node noise scale into (batch_size, N_max, 1).
            sigma_t = g.ndata["sigma_t"]
            padded_sigma_t = sigma_t.new_zeros(self.batch_size, max_n, sigma_t.size(-1))
            padded_sigma_t[self.batch_ids_nodes, local_idx] = sigma_t
            result = result + (padded_sigma_t,)

        return result
    

#################################################
# deprecated functions kept for legacy support
#################################################


    # def compute_rbf_edge_features(
    #     self,
    #     padded_coords: torch.Tensor,
    #     D_min: float = 0.,
    #     D_max: float = 20.,
    #     D_count: int = 16,
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     Build RBF-encoded pairwise distance features and displacement vectors
    #     from an already-padded coordinate tensor.

    #     Args:
    #         padded_coords: Padded node coordinates, shape
    #             ``(batch_size, max_num_nodes, 3)``.
    #         D_min: Minimum distance for RBF centres.
    #         D_max: Maximum distance for RBF centres.
    #         D_count: Number of RBF basis functions.

    #     Returns:
    #         rbf_features: Dense pairwise RBF features, shape
    #             ``(batch_size, max_num_nodes, max_num_nodes, D_count)``.
    #             Padded positions are zeroed out.
    #         displacements: Unit-length pairwise displacement vectors
    #             ``(x_j - x_i) / ||x_j - x_i||``, shape
    #             ``(batch_size, max_num_nodes, max_num_nodes, 3)``.
    #             Padded positions are zeroed out.
    #         pair_mask: Boolean mask, shape
    #             ``(batch_size, max_num_nodes, max_num_nodes)``.
    #             ``True`` where either the row or column index falls outside
    #             the molecule's real node count (i.e. padding).
    #     """
    #     # (B, N, 1, 3) - (B, 1, N, 3) -> (B, N, N, 3)
    #     displacements = padded_coords.unsqueeze(2) - padded_coords.unsqueeze(1)
    #     distances = displacements.norm(dim=-1)  # (B, N, N)
    #     displacements = displacements / (distances.unsqueeze(-1) + 1e-8)

    #     rbf_features = rbf(distances, D_min=D_min, D_max=D_max, D_count=D_count)

    #     max_n = padded_coords.size(1)
    #     node_pad = (
    #         torch.arange(max_n, device=self.device).unsqueeze(0)
    #         >= self.num_nodes_per_graph.to(self.device).unsqueeze(1)
    #     )
    #     pair_mask = node_pad.unsqueeze(2) | node_pad.unsqueeze(1)
    #     pair_mask = pair_mask | torch.eye(max_n, device=self.device, dtype=torch.bool).unsqueeze(0)
    #     rbf_features = rbf_features.masked_fill(pair_mask.unsqueeze(-1), 0.0)
    #     displacements = displacements.masked_fill(pair_mask.unsqueeze(-1), 0.0)

    #     return rbf_features, displacements, pair_mask

    # @torch.no_grad()
    # def graph_to_padded_tensor(
    #     self,
    #     g: dgl.DGLGraph,
    #     coord_key: str = "xt",
    #     target_key: Optional[str] = "vt",
    #     feat_key_nodes: str = "attr",
    #     feat_key_edges: str = "attr",
    #     D_min: float = 0.,
    #     D_max: float = 20.,
    #     D_count: int = 16,
    #     use_rbf: bool = False,
    #     return_sigma_t: bool = False,
    #     apply_random_rotations: bool = False,
    # ) -> Tuple[torch.Tensor, ...]:
    #     """
    #     Pad node and edge features from a batched DGLGraph into dense tensors,
    #     appending RBF-encoded pairwise distances to the edge features.

    #     Args:
    #         g: Batched DGLGraph.
    #         coord_key: Key in ``g.ndata`` for 3-D coordinates used by the
    #             RBF distance encoding.
    #         target_key: Key in ``g.ndata`` for the regression targets.
    #             Set to ``None`` at inference time to omit
    #             ``padded_targets`` from the returned tuple.
    #         feat_key_nodes: Key in ``g.ndata`` for the node features.
    #         feat_key_edges: Key in ``g.edata`` for the edge features.
    #         D_min: Minimum distance for RBF centres.
    #         D_max: Maximum distance for RBF centres.
    #         D_count: Number of RBF basis functions.
    #         use_rbf: If ``True`` (default), append RBF distance features and
    #             displacement vectors to the edge features.  When ``False``,
    #             the edge tensor keeps its original feature dimension.
    #         return_sigma_t: If ``True``, also return the padded noise scale
    #             ``sigma_t`` from ``g.ndata['sigma_t']``.
    #         apply_random_rotations: If ``True``, apply random rotations to the coordinates.
    #     Returns:
    #         padded_targets: ``(batch_size, N_max, 3)`` — regression targets
    #             from ``g.ndata[target_key]`` (omitted when
    #             ``target_key is None``).
    #         padded_times: ``(batch_size, N_max, 1)`` — diffusion times from
    #             ``g.ndata['t']``.
    #         padded_atom_index: ``(batch_size, N_max, 1)`` — per-atom type
    #             indices from ``g.ndata['atom_index']``.
    #         padded_nodes: ``(batch_size, N_max, node_feat_dim)``.
    #         padded_edges: ``(batch_size, N_max, N_max, edge_feat_dim [+ D_count + 3])``
    #             — when ``use_rbf=True`` the RBF distance features and
    #             displacement vectors are concatenated along the last axis.
    #         node_mask: ``(batch_size, N_max)`` — ``True`` at padding positions.
    #         pair_mask: ``(batch_size, N_max, N_max)`` — ``True`` at padding
    #             positions.
    #         padded_sigma_t: ``(batch_size, N_max, 1)`` — noise scale
    #             (only included when ``return_sigma_t=True``).
    #     """
    #     node_feats = g.ndata[feat_key_nodes]  # (total_nodes, node_feat_dim)
    #     max_n = int(self.num_nodes_per_graph.max().item())

    #     # Scatter node features into a dense (batch_size, N_max, node_feat_dim)
    #     # tensor. `offsets` gives the starting global index of each graph's
    #     # nodes, and `local_idx` converts global node indices to positions
    #     # within each graph (0 .. n_i-1).
    #     padded_nodes = node_feats.new_zeros(
    #         self.batch_size, max_n, node_feats.size(-1)
    #     )
    #     offsets = torch.cumsum(self.num_nodes_per_graph, dim=0) - self.num_nodes_per_graph
    #     local_idx = (
    #         torch.arange(node_feats.size(0), device=self.device)
    #         - offsets[self.batch_ids_nodes]
    #     )
    #     padded_nodes[self.batch_ids_nodes, local_idx] = node_feats

    #     # Pad per-node atom type indices into (batch_size, N_max, 1).
    #     atom_index = g.ndata["atom_index"]
    #     padded_atom_index = atom_index.new_zeros(self.batch_size, max_n, 1)
    #     padded_atom_index[self.batch_ids_nodes, local_idx] = atom_index.unsqueeze(-1)

    #     # Boolean mask marking padding positions as True so that attention
    #     # layers can ignore them. Shape: (batch_size, N_max).
    #     node_mask = (
    #         torch.arange(max_n, device=self.device).unsqueeze(0)
    #         >= self.num_nodes_per_graph.unsqueeze(1)
    #     )

    #     # Convert sparse edge features to a dense pair tensor
    #     # (batch_size, N_max, N_max, edge_feat_dim).
    #     padded_edges, pair_mask = edges_to_pair_tensor(g.edata[feat_key_edges], g)

    #     padded_coords = nodes_to_padded_tensor(g.ndata[coord_key], g)
    #     if apply_random_rotations:
    #         padded_coords = randomly_rotate(padded_coords)

    #     if use_rbf:
    #         rbf_feats, displacements, rbf_mask = self.compute_rbf_edge_features(
    #             padded_coords, D_min=D_min, D_max=D_max, D_count=D_count,
    #         )
    #         pair_mask = rbf_mask
    #         padded_edges = torch.cat([padded_edges, rbf_feats, displacements], dim=-1)

    #     # pair_mask from edges_to_pair_tensor only covers padding; add diagonal
    #     diag = torch.eye(max_n, device=self.device, dtype=torch.bool).unsqueeze(0)
    #     pair_mask = pair_mask | diag

    #     if target_key is not None:
    #         padded_targets = nodes_to_padded_tensor(g.ndata[target_key], g)

    #     # Pad per-node flow / diffusion times into (batch_size, N_max, 1), reusing the
    #     # same local_idx mapping computed above for the node features.
    #     times = g.ndata["t"]
    #     padded_times = times.new_zeros(self.batch_size, max_n, 1)
    #     padded_times[self.batch_ids_nodes, local_idx] = times

    #     # NOTE: the MASK TENSORS for both nodes and edges indicate
    #     #       padding positions as FALSE and non-padding positions as TRUE
    #     result = (padded_times, padded_coords, padded_nodes, padded_atom_index, padded_edges, ~node_mask, ~pair_mask)
    #     if target_key is not None:
    #         result = (padded_targets,) + result

    #     if return_sigma_t:
    #         # Pad per-node noise scale into (batch_size, N_max, 1).
    #         sigma_t = g.ndata["sigma_t"]
    #         padded_sigma_t = sigma_t.new_zeros(self.batch_size, max_n, sigma_t.size(-1))
    #         padded_sigma_t[self.batch_ids_nodes, local_idx] = sigma_t
    #         result = result + (padded_sigma_t,)

    #     return result