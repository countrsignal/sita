import dgl
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer

from typing import Tuple, Union, Dict

from .modules.encoders import AtomEncoder
from .layers.gvp_opt import (
    OptimizedGVPConv,
    OptimizedEdgeUpdate,
    OptimizedNodePositionUpdate,
)
from .layers.gvp import _rbf
from .layers.swish import SwishBeta
from ..utils.graph_utils import GraphAdapter
from .layers.attention_block import AttentionBlock
from .layers.primitives import LayerNormEps, LinearNoBias


def _build_opt_edge_mlp(edge_feat_size: int, n_hidden_edge: int) -> nn.Module:
    mlp = nn.Sequential(
            nn.Linear(edge_feat_size, n_hidden_edge),
            nn.SiLU(),
            nn.Linear(n_hidden_edge, n_hidden_edge),
            nn.SiLU(),
            LayerNormEps(n_hidden_edge),
        )
    return torch.compile(mlp)


@torch.compile
def _compute_edge_features(
    node_positions: Tensor,
    src_idx: Tensor,
    dst_idx: Tensor,
    rbf_dmax: float,
    rbf_dim: int,
) -> Tuple[Tensor, Tensor]:
    """Compiled helper to compute normalized edge vectors and RBF distances."""
    x_src = node_positions.index_select(0, src_idx)
    x_dst = node_positions.index_select(0, dst_idx)

    x_diff = x_src - x_dst + 1e-8
    dij = torch.square(x_diff).sum(dim=-1, keepdim=True).sqrt() + 1e-8
    x_diff = x_diff / dij
    d = _rbf(dij.squeeze(-1), D_max=rbf_dmax, D_count=rbf_dim)
    return x_diff, d


class MolecularEncoder(nn.Module):

    def __init__(
        self,
        node_feats_in: int,
        edge_feats_in: int,
        c_atoms: int,
        c_pairs: int,
        n_vecs: int = 16,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
        message_norm: Union[float, str] = "sum",
        use_dst_feats: bool = False,
        vector_gating: bool = True,
    ) -> None:
        super().__init__()

        # Bookkeeping
        self.node_feats_in = node_feats_in
        self.edge_feats_in = edge_feats_in
        self.n_vecs = n_vecs
        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.message_norm = message_norm
        self.use_dst_feats = use_dst_feats
        self.vector_gating = vector_gating

        # Initial embedding layers
        self.node_embedder = AtomEncoder(node_feats_in, c_atoms)
        self.edge_embedder = _build_opt_edge_mlp(edge_feats_in, c_pairs)

        # Single GVP convolution layer
        self.edge_update = OptimizedEdgeUpdate(
            n_node_scalars=c_atoms,
            n_edge_feats=c_pairs,
            update_edge_w_distance=True,
            rbf_dim=rbf_dim,
        )
        self.gvp_conv = OptimizedGVPConv(
            scalar_size=c_atoms,
            vector_size=n_vecs,
            n_message_gvps=n_message_gvps,
            n_update_gvps=n_update_gvps,
            use_dst_feats=use_dst_feats,
            rbf_dmax=rbf_dmax,
            rbf_dim=rbf_dim,
            edge_feat_size=c_pairs,
            coords_range=10.0,
            message_norm=message_norm,
            vector_gating=vector_gating,
            scalar_activation=SwishBeta,
            vector_activation=nn.Sigmoid,
        )

    def precompute_distances(self, g: dgl.DGLGraph, node_positions: Tensor):
        """Precompute normalized displacement vectors and RBF embeddings."""
        src_idx, dst_idx = g.edges()
        src_idx = src_idx.to(node_positions.device)
        dst_idx = dst_idx.to(node_positions.device)
        return _compute_edge_features(
            node_positions,
            src_idx,
            dst_idx,
            self.rbf_dmax,
            self.rbf_dim,
        )
    
    def forward(self, graph: dgl.DGLGraph):
        # initialize the coordinates and velocities
        x_init = graph.ndata["xt"].clone()  # (num_nodes, 3)
        device = x_init.device
        v_init = torch.zeros(
            graph.num_nodes(),
            self.n_vecs,
            3,
            device=device,
        )  # (num_nodes, n_vec_channels, 3)

        # encode the atom features
        node_repr = self.node_embedder(
            time=graph.ndata["t"].view(-1),
            attr=graph.ndata["attr"],
            atom_index=graph.ndata["atom_index"].view(-1),
        )
        # node_repr: (num_nodes, n_hidden)

        # embed the edge features
        edge_mask = ~torch.all(graph.edata["attr"] == 0, dim=-1).unsqueeze(dim=-1)
        edge_repr = self.edge_embedder(graph.edata["attr"]) * edge_mask
        # edge_repr: (num_edges, n_hidden_edge)

        # update the edge features
        x_diff, d = self.precompute_distances(graph, x_init)
        edge_repr = self.edge_update(graph, node_repr, edge_repr, d=d) * edge_mask
        # edge_repr: (num_edges, n_hidden_edge)

        # update the node features
        node_repr, vs, xs = self.gvp_conv(
            g=graph,
            scalar_feats=node_repr,
            coord_feats=x_init,
            vec_feats=v_init,
            edge_feats=edge_repr,
            x_diff=x_diff,
            d=d,
        )
        # vs: (num_nodes, n_vec_channels, 3)
        # node_repr: (num_nodes, n_hidden)
        return vs, node_repr, edge_repr, xs


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        c_pairs: int,
        n_heads: int,
        n_layers: int = 4,
        dropout_prob: float = 0.1,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
    ):
        super().__init__()
        self.n_vecs = n_vecs
        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax

        self.in_proj = LinearNoBias(c_atoms + n_vecs * 3 + 3, c_atoms)

        self.rbf_proj = nn.Sequential(
            LinearNoBias(c_pairs + rbf_dim, c_pairs),
            LayerNormEps(c_pairs),
        )

        self.attn_w_bias = AttentionBlock(
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            n_heads=n_heads,
            dropout_prob=dropout_prob,
            bias=False,
            initial_norm=True,
        )

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(TransformerEncoderLayer(
                d_model=c_atoms,
                nhead=n_heads,
                dim_feedforward=c_atoms * 4, # NOTE: hardcoded for now
                dropout=dropout_prob,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            ))
        
        self.out_proj = nn.Sequential(
            nn.Linear(c_atoms, c_atoms),
            nn.SiLU(),
            LinearNoBias(c_atoms, 3),
        )
    
    def forward(
        self,
        adapter: GraphAdapter,
        graph: dgl.DGLGraph,
        xs: Tensor,
        vs: Tensor,
        node_repr: Tensor,
        edge_repr: Tensor,
    ) -> Dict[str, Tensor]:

        # node keys to identify which variables to convert to padded tensors
        n_keys = ["xs", "vs", "node_repr"]
        if self.training:
            n_keys.append("vt")

        # convert the DGL graph data to padded tensors
        with graph.local_scope():
            graph.ndata["xs"] = xs
            graph.ndata["vs"] = vs.flatten(-2, -1) # (num_nodes, n_vecs * 3)
            graph.ndata["node_repr"] = node_repr
            graph.edata["edge_repr"] = edge_repr

            tensor_dict = adapter.graph_to_padded_tensor(
                graph,
                node_keys=n_keys,
                edge_keys=["edge_repr"],
            )

        # unpack
        xs = tensor_dict["xs"]
        vs = tensor_dict["vs"]
        node_repr = tensor_dict["node_repr"]
        edge_repr = tensor_dict["edge_repr"]
        node_mask = tensor_dict["node_mask"]
        pair_mask = tensor_dict["pair_mask"]
        # xs: (batch_size, num_nodes, 3)
        # vs: (batch_size, num_nodes, n_vecs * 3)
        # node_repr: (batch_size, num_nodes, c_atoms)
        # edge_repr: (batch_size, num_nodes, num_nodes, c_pairs)
        # node_mask: (batch_size, num_nodes)
        # pair_mask: (batch_size, num_nodes, num_nodes)

        # inital project of node (atom) features
        node_repr = self.in_proj(torch.cat([node_repr, vs, xs], dim=-1))
        node_repr = node_repr * node_mask.unsqueeze(-1)
        # node_repr: (batch_size, num_nodes, c_atoms)

        # inject pairwise distances into pair bias via RBFs
        d = _rbf(
            torch.cdist(xs, xs, p=2.0),
            D_max=self.rbf_dmax,
            D_count=self.rbf_dim,
        )
        edge_repr = self.rbf_proj(torch.cat([edge_repr, d], dim=-1))
        edge_repr = edge_repr * pair_mask.unsqueeze(-1)
        # edge_repr: (batch_size, num_nodes, num_nodes, c_pairs)

        # apply the attention with pair bias
        node_repr = self.attn_w_bias(
            x=node_repr,
            mask=node_mask,
            edge_repr=edge_repr,
        )
        # node_repr: (batch_size, num_nodes, c_atoms)

        # apply standard transformer layers
        for layer in self.layers:
            # NOTE: PyTorch TransformerEncoderLayer expects the mask to be True for padding tokens
            node_repr = layer(node_repr, src_key_padding_mask=~node_mask)
            node_repr = node_repr * node_mask.unsqueeze(-1)
            # node_repr: (batch_size, num_nodes, c_atoms)
        
        # predict the velocities
        velocity = self.out_proj(node_repr)
        velocity = velocity * node_mask.unsqueeze(-1)
        # velocity: (batch_size, num_nodes, 3)

        # package into existing tensor dictionary
        tensor_dict["velocity"] = velocity
        tensor_dict["node_repr"] = node_repr
        tensor_dict["edge_repr"] = edge_repr
        return tensor_dict


class HybridModel(nn.Module):

    def __init__(
        self,
        node_feats_in: int,
        edge_feats_in: int,
        c_atoms: int,
        c_pairs: int,
        n_vecs: int = 16,
        n_message_gvps: int = 3,
        n_update_gvps: int = 3,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
        message_norm: Union[float, str] = "sum",
        use_dst_feats: bool = False,
        vector_gating: bool = True,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.encoder = MolecularEncoder(
            node_feats_in=node_feats_in,
            edge_feats_in=edge_feats_in,
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            n_vecs=n_vecs,
            n_message_gvps=n_message_gvps,
            n_update_gvps=n_update_gvps,
            rbf_dim=rbf_dim,
            rbf_dmax=rbf_dmax,
            message_norm=message_norm,
            use_dst_feats=use_dst_feats,
            vector_gating=vector_gating,
        )
        self.decoder = TransformerDecoder(
            n_vecs=n_vecs,
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_prob=dropout_prob,
            rbf_dim=rbf_dim,
            rbf_dmax=rbf_dmax,
        )
    
    def forward(
        self,
        graph: dgl.DGLGraph,
        adapter: GraphAdapter,
    ) -> Dict[str, Tensor]:
        # encode the graph
        vs, node_repr, edge_repr, xs = self.encoder(graph)
        # vs: (num_nodes, n_vecs, 3)
        # node_repr: (num_nodes, c_atoms)
        # edge_repr: (num_edges, c_pairs)
        # xs: (num_nodes, 3)

        # decode the graph
        tensor_dict = self.decoder(adapter, graph, xs, vs, node_repr, edge_repr)
        return tensor_dict
    
    def compute_loss(self, tensor_dict: Dict[str, Tensor]) -> Tensor:
        # check that node mask at least contains 1 atom per molecule
        assert torch.all(tensor_dict["node_mask"].sum(dim=-1) > 0), "Node mask must contain at least 1 atom per molecule"

        velocity = tensor_dict["velocity"]   # (B, N, 3)
        target = tensor_dict["vt"]           # (B, N, 3)
        node_mask = tensor_dict["node_mask"] # (B, N)

        per_atom_mse = (velocity - target).square().mean(dim=-1)  # (B, N)
        per_atom_mse = per_atom_mse * node_mask                  # zero out padding

        atoms_per_mol = node_mask.sum(dim=-1)                    # (B,)
        per_mol_loss = per_atom_mse.sum(dim=-1) / atoms_per_mol  # (B,)

        return per_mol_loss.mean()