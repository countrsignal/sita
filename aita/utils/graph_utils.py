import dgl
import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Tuple, Optional


def get_batch_indices(g: dgl.DGLGraph) -> torch.Tensor:
    return torch.arange(g.batch_size, device=g.device).repeat_interleave(g.batch_num_nodes())


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