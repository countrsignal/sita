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
from ..layers.gvp import GVPConv, NodePositionUpdate, EdgeUpdate, _norm_no_nan, _rbf


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


class VelocityLayerNorm(nn.Module):
    """Nontrainable norm for vector features, following GVPLayerNorm.

    Padded atoms (identified by atom_mask) are excluded from the
    normalization and guaranteed to remain zero.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, vectors: Tensor, atom_mask: Tensor) -> Tensor:
        # vectors: (batch_size, n_atoms, n_vecs, 3)
        # atom_mask: (batch_size, n_atoms)
        vn = _norm_no_nan(vectors, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True) + self.eps) + self.eps
        return (vectors / vn) * atom_mask[..., None, None]


class VelocityProjection(nn.Module):
    """Projects atom features into an initial set of velocity vectors.

    Uses a two-layer MLP (without bias) to map each atom's
    representation to n_vecs 3D vectors, initializing the velocity
    superposition for downstream equivariant updates.

    Args:
        n_vecs: Number of velocity vectors per atom.
        c_atoms: Dimensionality of the atom features (node embeddings).
    """

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
    ) -> None:
        super().__init__()

        self.n_vecs = n_vecs
        self.c_atoms = c_atoms
        self.velocity_proj = nn.Sequential(
            LinearNoBias(c_atoms, c_atoms),
            nn.SiLU(),
            LinearNoBias(c_atoms, n_vecs * 3),
        )
        self.vec_norm = VelocityLayerNorm(eps=1e-5)
    
    def forward(
        self,
        x_h: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:

        vfs = self.velocity_proj(x_h)
        vfs = vfs * atom_mask.unsqueeze(-1)
        vfs = vfs.view(x_h.shape[0], -1, self.n_vecs, 3)
        vfs = self.vec_norm(vfs, atom_mask)
        return vfs


class VelocityUpdate(nn.Module):
    """Stripped-down GVP for equivariant velocity vector updates.

    Maintains a superposition of n_vecs velocity vectors per atom. Vector
    norms are fed back into the atom features to provide a feedback
    mechanism between geometric 3D vectors and the atom representations.
    Set n_vecs_out=1 in the final layer to project down to a single
    velocity vector.
    """

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        n_vecs_out: int = None,
        vectors_activation: nn.Module = nn.Sigmoid(),
        vector_gating: bool = True,
    ) -> None:
        super().__init__()

        self.n_vecs = n_vecs
        self.n_vecs_out = n_vecs if n_vecs_out is None else n_vecs_out
        self.c_atoms = c_atoms

        dim_h = max(n_vecs, self.n_vecs_out)
        self.dim_h = dim_h

        wh_k = 1 / math.sqrt(n_vecs)
        self.Wh = nn.Parameter(
            torch.zeros(n_vecs, dim_h).uniform_(-wh_k, wh_k)
        )

        wu_k = 1 / math.sqrt(dim_h)
        self.Wu = nn.Parameter(
            torch.zeros(dim_h, self.n_vecs_out).uniform_(-wu_k, wu_k)
        )

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(c_atoms + dim_h, c_atoms),
            nn.SiLU(),
        )

        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(c_atoms, self.n_vecs_out)
        else:
            self.scalar_to_vector_gates = None

    def forward(
        self,
        vfs: Tensor,
        x_h: Tensor,
        atom_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # vfs: (..., n_vecs, 3)
        # x_h: (..., c_atoms)
        # atom_mask: (...)

        Vh = torch.einsum('... v c, v h -> ... h c', vfs, self.Wh)
        Vu = torch.einsum('... h c, h u -> ... u c', Vh, self.Wu)

        sh = _norm_no_nan(Vh, axis=-1)
        s = torch.cat((x_h, sh), dim=-1)
        feats_out = self.to_feats_out(s)

        if self.scalar_to_vector_gates is not None:
            gating = self.scalar_to_vector_gates(feats_out).unsqueeze(-1)
        else:
            gating = _norm_no_nan(Vu, axis=-1).unsqueeze(-1)

        if self.n_vecs_out == 1:
            vector_norms = _norm_no_nan(Vu, axis=-1).unsqueeze(-1)
            Vu = Vu / vector_norms

        vectors_out = self.vectors_activation(gating) * Vu
        vectors_out = vectors_out * atom_mask[..., None, None]
        feats_out = feats_out * atom_mask.unsqueeze(-1)

        return vectors_out, feats_out


class AtomicDecoder(nn.Module):

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        n_heads: int = 8,
        n_layers: int = 5,
        dropout_prob: float = 0.0,
        bias: bool = False,
        initial_norm: bool = True,
    ) -> None:
        super().__init__()

        self.n_vecs = n_vecs
        self.c_atoms = c_atoms
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
                    vectors_activation=nn.Sigmoid,
                    vector_gating=True,
                )
            )

            if idx < n_layers - 1:
                self.velocity_layernorms.append(
                    VelocityLayerNorm(eps=1e-5),
                )
    
    def forward(
        self,
        x_h: Tensor,
        atom_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
    
        # NOTE: 'vfs' stands for velocity field superpositions.
        vfs = self.velocity_projection(x_h, atom_mask)
        # vfs: (..., n_vecs, 3)
        # x_h: (..., c_atoms)
        # atom_mask: (...)


        for idx in range(self.n_layers - 1):
            x_h = self.attention_blocks[idx](x_h, atom_mask)
            vfs_update, x_h = self.velocity_updates[idx](vfs, x_h, atom_mask)
            # vfs_update: (..., n_vecs, 3)
            
            vfs = vfs + vfs_update
            vfs = self.velocity_layernorms[idx](vfs)

        x_h = self.attention_blocks[-1](x_h, atom_mask)
        velocity, x_h = self.velocity_updates[-1](vfs, x_h, atom_mask)
        # velocity: (..., 3)

        return velocity, x_h