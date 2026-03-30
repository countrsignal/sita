import dgl
import torch
from torch import nn, Tensor
import torch.nn.functional as F
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
from ..layers.attention_block import AttentionBlock
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


class VelocityPredictionHead(nn.Module):
    def __init__(self, c_atoms: int):
        super().__init__()
        self.c_atoms = c_atoms
        self.w1 = nn.Linear(c_atoms, c_atoms, bias=True)
        self.w2 = nn.Linear(c_atoms, 3, bias=False)
        
    def forward(self, x_h: Tensor, atom_mask: Tensor) -> Tensor:
        v_t = self.w2(F.silu(self.w1(x_h)))
        v_t = v_t * atom_mask.unsqueeze(-1)
        return v_t



class RadialKernel(nn.Module):

    def __init__(
        self,
        dim: int,
        d_min: float = 0.5,
        d_max: float = 20.0,
        n_rbf_channels: int = 16,
        eps: float = 1e-8,
        normalize_kernel_values: bool = True,
    ):

        super(RadialKernel, self).__init__()

        self.eps = eps
        self.dim = dim
        self.d_min = d_min
        self.d_max = d_max
        self.n_rbf_channels = n_rbf_channels
        self.register_buffer(
            "length_scales",
            torch.linspace(d_min, d_max, n_rbf_channels, dtype=torch.float32),
            persistent=True,
        )
        self.normalize_kernel_values = normalize_kernel_values

        self.mlp_in  = nn.Linear(n_rbf_channels, n_rbf_channels, bias=True)
        self.mlp_out = nn.Linear(n_rbf_channels, dim, bias=True)

    def compute_length_scales(self) -> torch.Tensor:
        assert isinstance(self.length_scales, torch.Tensor), "`length_scales` needs to be a PyTorch Tensor."
        return self.length_scales

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # NOTE: Adding epsilons TWICE is CRITICAL for numerical stability!!!
        rel_pos = coords[:, :, None, :] - coords[:, None, :, :] + self.eps
        dists = torch.square(rel_pos).sum(dim=-1).sqrt() + self.eps

        # [B, L, L, C]
        lengthscales = self.compute_length_scales()
        # > Number of length scales is the number of channels
        dists = dists.unsqueeze(-1).expand(-1, -1, -1, len(lengthscales)) + self.eps
        dists = (
            dists / (lengthscales[None, None, None, :] ** 2)
        )
        # Kernel scores shape: [B, L, L, C]
        scores = torch.exp(-dists)
        if self.normalize_kernel_values:
            scores = scores / (
                torch.abs(scores).sum(dim=-1, keepdim=True) + self.eps
            )
        # Return pair representation to bias self-attention layers
        return self.mlp_out(F.silu(self.mlp_in(scores)))


class OptimizedAtomicDecoder(nn.Module):

    def __init__(
        self,
        c_atoms: int,
        c_pairs: int,
        n_heads: int = 8,
        n_layers: int = 5,
        dropout_prob: float = 0.0,
        bias: bool = False,
        initial_norm: bool = True,
    ) -> None:
        super().__init__()

        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_prob = dropout_prob
        self.bias = bias
        self.initial_norm = initial_norm

        # self.velocity_projection = VelocityProjection(n_vecs=n_vecs, c_atoms=c_atoms)
        self.velocity_out = VelocityPredictionHead(c_atoms=c_atoms)

        # Fused pair-bias projection: one shared LayerNorm + one Linear
        # instead of n_layers separate (LayerNorm + Linear) modules
        # self.pair_bias_norm = nn.LayerNorm(c_pairs)
        # self.pair_bias_proj = LinearNoBias(c_pairs, n_heads * n_layers)

        # self.velocity_updates = nn.ModuleList([])
        self.attention_blocks = nn.ModuleList([])
        for idx in range(n_layers):
            self.attention_blocks.append(
                CompiledAttentionBlock(
                    c_atoms=c_atoms,
                    n_heads=n_heads,
                    dropout_prob=dropout_prob,
                    bias=bias,
                    initial_norm=initial_norm,
                )
            )
            # self.velocity_updates.append(
            #     VelocityUpdate(
            #         n_vecs=n_vecs,
            #         c_atoms=c_atoms,
            #         n_vecs_out=1 if idx == n_layers - 1 else n_vecs,
            #         residual=False if idx == n_layers - 1 else True, # NOTE: Residual connections only used for intermediate layers
            #     )
            # )
    
    def forward(
        self,
        x_h: Tensor,
        pair_repr: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # vfs = self.velocity_projection(x_h=x_h, atom_mask=atom_mask)
        # vfs: (..., n_vecs, 3)
        # x_h: (..., c_atoms)
        # atom_mask: (...)

        # Fused pair-bias: one LayerNorm + one Linear for all layers at once
        # B, N = pair_repr.shape[:2]
        # all_biases = self.pair_bias_proj(self.pair_bias_norm(pair_repr))
        # (B, N, N, n_heads * n_layers) -> (n_layers, B, n_heads, N, N)
        # all_biases = (
        #     all_biases
        #     .view(B, N, N, self.n_layers, self.n_heads)
        #     .permute(3, 0, 4, 1, 2)
        # )
        # pad_mask = torch.where(
        #     atom_mask[:, None, None, :], 0.0, torch.finfo(x_h.dtype).min,
        # )
        # all_biases = all_biases + pad_mask.unsqueeze(0)

        # Update superposition of velocity field vectors and atom representations
        for idx in range(self.n_layers):
            x_h = self.attention_blocks[idx](
                x=x_h, mask=atom_mask, residual=x_h,
                # attn_bias=all_biases[idx],
            )
            # vfs, x_h = self.velocity_updates[idx](vfs=vfs, x_h=x_h, atom_mask=atom_mask)
        
        # Final prediction
        # x_h = self.attention_blocks[-1](
        #     x=x_h, mask=atom_mask,
        #     # attn_bias=all_biases[-1],
        #     attn_bias=None,
        # )
        # velocity, x_h = self.velocity_updates[-1](vfs=vfs, x_h=x_h, atom_mask=atom_mask)
        velocity = self.velocity_out(x_h=x_h, atom_mask=atom_mask)
        return velocity, x_h