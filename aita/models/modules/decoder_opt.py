import dgl
import torch
from torch import nn, Tensor
from typing import Union, Tuple

from ..layers.swish import SwishBeta
from ..layers.gvp import _rbf
from ..layers.gvp_opt import (
    OptimizedGVPConv,
    OptimizedEdgeUpdate,
    OptimizedNodePositionUpdate,
)
from ..layers.spatial_opt import (
    VelocityProjection,
    VelocityUpdate,
)
from ..layers.attention_block_opt import CompiledAttentionBlock
from ..layers.primitives import LinearNoBias


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


class OptimizedGVPDecoder(nn.Module):
    """GVP decoder that leverages the optimized GVP blocks and torch.compile."""

    def __init__(
        self,
        n_vec: int = 16,
        n_layers: int = 5,
        n_hidden_nodes: int = 64,
        n_hidden_edge: int = 32,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        n_coord_gvps: int = 1,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
        message_norm: Union[float, str] = "sum",
        use_dst_feats: bool = False,
        vector_gating: bool = True,
    ) -> None:
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.n_layers = n_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_edge = n_hidden_edge

        self.convs = nn.ModuleList([])
        self.edge_updater = nn.ModuleList([])
        for _ in range(n_layers):
            self.convs.append(
                OptimizedGVPConv(
                    scalar_size=n_hidden_nodes,
                    vector_size=n_vec,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    use_dst_feats=use_dst_feats,
                    rbf_dmax=rbf_dmax,
                    rbf_dim=rbf_dim,
                    edge_feat_size=n_hidden_edge,
                    coords_range=10.0,
                    message_norm=message_norm,
                    vector_gating=vector_gating,
                    scalar_activation=SwishBeta,
                    vector_activation=nn.Sigmoid,
                )
            )
            self.edge_updater.append(
                OptimizedEdgeUpdate(
                    n_node_scalars=n_hidden_nodes,
                    n_edge_feats=n_hidden_edge,
                    update_edge_w_distance=True,
                    rbf_dim=rbf_dim,
                )
            )

        self.position_updater = OptimizedNodePositionUpdate(
            n_scalars=n_hidden_nodes,
            n_vec_channels=n_vec,
            n_gvps=n_coord_gvps,
            vector_gating=vector_gating,
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

    def forward(
        self,
        h: Tensor,
        v_init: Tensor,
        x_init: Tensor,
        edge_repr: Tensor,
        edge_mask: Tensor,
        graph: dgl.DGLGraph,
    ):

        hs, vs, xs = h, v_init, x_init
        for conv, edge_nn in zip(self.convs, self.edge_updater):
            x_diff, d = self.precompute_distances(graph, xs)
            edge_repr = edge_nn(graph, hs, edge_repr, d=d) * edge_mask
            hs, vs, xs = conv(
                graph,
                hs,
                xs,
                vs,
                edge_feats=edge_repr,
                x_diff=x_diff,
                d=d,
            )

        vector_field = self.position_updater(hs, vs)
        return vector_field, hs, vs, edge_repr


class OptimizedAtomicDecoder(nn.Module):

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        c_pairs: int,
        n_heads: int = 8,
        n_layers: int = 5,
        dropout_prob: float = 0.0,
        bias: bool = False,
        initial_norm: bool = True,
    ) -> None:
        super().__init__()

        self.n_vecs = n_vecs
        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.bias = bias
        self.initial_norm = initial_norm

        self.velocity_projection = VelocityProjection(n_vecs=n_vecs, c_atoms=c_atoms)

        # Fused pair-bias projection: one shared LayerNorm + one Linear
        # instead of n_layers separate (LayerNorm + Linear) modules
        self.pair_bias_norm = nn.LayerNorm(c_pairs)
        self.pair_bias_proj = LinearNoBias(c_pairs, n_heads * n_layers)

        self.attention_blocks = nn.ModuleList([])
        self.velocity_updates = nn.ModuleList([])
        for idx in range(n_layers):
            self.attention_blocks.append(
                CompiledAttentionBlock(
                    c_atoms=c_atoms,
                    c_pairs=c_pairs,
                    n_heads=n_heads,
                    dropout_prob=dropout_prob,
                    bias=bias,
                    initial_norm=initial_norm,
                )
            )
            self.velocity_updates.append(
                VelocityUpdate(
                    n_vecs=n_vecs,
                    c_atoms=c_atoms,
                    n_vecs_out=1 if idx == n_layers - 1 else n_vecs,
                    residual=False if idx == n_layers - 1 else True, # NOTE: Residual connections only used for intermediate layers
                )
            )
    
    def forward(
        self,
        x_h: Tensor,
        pair_repr: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        vfs = self.velocity_projection(x_h=x_h, atom_mask=atom_mask)
        # vfs: (..., n_vecs, 3)
        # x_h: (..., c_atoms)
        # atom_mask: (...)

        # Fused pair-bias: one LayerNorm + one Linear for all layers at once
        B, N = pair_repr.shape[:2]
        all_biases = self.pair_bias_proj(self.pair_bias_norm(pair_repr))
        # (B, N, N, n_heads * n_layers) -> (n_layers, B, n_heads, N, N)
        all_biases = (
            all_biases
            .view(B, N, N, self.n_layers, self.n_heads)
            .permute(3, 0, 4, 1, 2)
        )
        pad_mask = torch.where(
            atom_mask[:, None, None, :], 0.0, torch.finfo(x_h.dtype).min,
        )
        all_biases = all_biases + pad_mask.unsqueeze(0)

        # Update superposition of velocity field vectors and atom representations
        for idx in range(self.n_layers - 1):
            x_h = self.attention_blocks[idx](
                x=x_h, mask=atom_mask, edge_repr=pair_repr,
                attn_bias=all_biases[idx],
            )
            vfs, x_h = self.velocity_updates[idx](vfs=vfs, x_h=x_h, atom_mask=atom_mask)
        
        # Final prediction
        x_h = self.attention_blocks[-1](
            x=x_h, mask=atom_mask, edge_repr=pair_repr,
            attn_bias=all_biases[-1],
        )
        velocity, x_h = self.velocity_updates[-1](vfs=vfs, x_h=x_h, atom_mask=atom_mask)
        return velocity.squeeze(-2), x_h