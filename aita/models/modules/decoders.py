import math

import dgl
import dgl.function as fn

import torch
from torch import nn, Tensor

from typing import Union, Tuple

from ..layers.swish import SwishBeta
from ..layers.gvp import _norm_no_nan
from ..layers.primitives import LinearNoBias
from ..layers.attention_block import AttentionBlock
from ..layers.gvp import GVPConv, NodePositionUpdate, EdgeUpdate, _rbf
from ..layers.spatial import (
    VelocityProjection,
    VelocityUpdate,
    VelocityLayerNorm,
    VelocityProjectionV2,
    VelocityUpdateV2,
    PairTransition,
)

class GVP_Decoder(nn.Module):

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
                GVPConv(
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
                EdgeUpdate(
                    n_node_scalars=n_hidden_nodes,
                    n_edge_feats=n_hidden_edge,
                    update_edge_w_distance=True,
                    rbf_dim=rbf_dim,
                )
            )

        self.position_updater = NodePositionUpdate(
            n_scalars=n_hidden_nodes,
            n_vec_channels=n_vec,
            n_gvps=n_coord_gvps,
        )

    def precompute_distances(self, g: dgl.DGLGraph, node_positions: Tensor):
        """Precompute the pairwise distances between all nodes in the graph."""

        with g.local_scope():

            g.ndata['x_d'] = node_positions

            g.apply_edges(fn.u_sub_v("x_d", "x_d", "x_diff"))
            dij = _norm_no_nan(g.edata['x_diff'], keepdims=True) + 1e-8
            x_diff = g.edata['x_diff'] / dij
            d = _rbf(dij.squeeze(1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        
        return x_diff, d

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
            hs, vs, xs = conv(graph, hs, xs, vs, edge_feats=edge_repr, x_diff=x_diff, d=d)
            # hs: (num_nodes, n_hidden)
            # vs: (num_nodes, n_vec_channels, 3)
            # xs: (num_nodes, 3)

        vector_field = self.position_updater(hs, vs)
        # vector_field: (num_nodes, 3)
        return vector_field


class AtomicDecoder(nn.Module):

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

        self.velocity_projection = VelocityProjection(
            n_vecs=n_vecs,
            c_atoms=c_atoms,
        )

        self.attention_blocks = nn.ModuleList([])
        self.velocity_updates = nn.ModuleList([])
        self.velocity_layernorms = nn.ModuleList([])
        for idx in range(n_layers):

            self.attention_blocks.append(
                AttentionBlock(
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
                )
            )

            if idx < n_layers - 1:
                self.velocity_layernorms.append(
                    VelocityLayerNorm(eps=1e-5),
                )
    
    def forward(
        self,
        x_h: Tensor,
        pair_repr: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
    
        # NOTE: 'vfs' stands for velocity field superpositions.
        vfs = self.velocity_projection(x_h=x_h, atom_mask=atom_mask)
        # vfs: (..., n_vecs, 3)
        # x_h: (..., c_atoms)
        # atom_mask: (...)


        for idx in range(self.n_layers - 1):
            x_h = self.attention_blocks[idx](x=x_h, mask=atom_mask, edge_repr=pair_repr)
            # x_h: (..., c_atoms)

            vfs_update, x_h = self.velocity_updates[idx](vfs=vfs, x_h=x_h, atom_mask=atom_mask)
            # vfs_update: (..., n_vecs, 3)
            
            vfs = vfs + vfs_update
            vfs = self.velocity_layernorms[idx](vectors=vfs, atom_mask=atom_mask)

        x_h = self.attention_blocks[-1](x=x_h, mask=atom_mask, edge_repr=pair_repr)
        velocity, x_h  = self.velocity_updates[-1](vfs=vfs, x_h=x_h, atom_mask=atom_mask)
        # velocity: (..., 3)

        return velocity.squeeze(-2), x_h


class AtomicDecoderEBM(nn.Module):
    """Atomic decoder for EBM models."""

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

        self.pair_bias_norm = nn.LayerNorm(c_pairs)
        self.pair_bias_proj = LinearNoBias(c_pairs, n_heads * n_layers)

        self.attention_blocks = nn.ModuleList([])
        for idx in range(n_layers):
            self.attention_blocks.append(
                AttentionBlock(
                    c_atoms=c_atoms,
                    c_pairs=c_pairs,
                    n_heads=n_heads,
                    dropout_prob=dropout_prob,
                    bias=bias,
                    initial_norm=initial_norm,
                )
            )

    def forward(
        self,
        x_h: Tensor,
        pair_repr: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:

        B, N = pair_repr.shape[:2]
        all_biases = self.pair_bias_proj(self.pair_bias_norm(pair_repr))
        all_biases = (
            all_biases
            .view(B, N, N, self.n_layers, self.n_heads)
            .permute(3, 0, 4, 1, 2)
        )
        pad_mask = torch.where(
            atom_mask[:, None, None, :], 0.0, torch.finfo(x_h.dtype).min,
        )
        all_biases = all_biases + pad_mask.unsqueeze(0)

        for idx in range(self.n_layers):
            x_h = self.attention_blocks[idx](
                x=x_h, mask=atom_mask, edge_repr=pair_repr,
                attn_bias=all_biases[idx],
            )
        return x_h