import torch
from torch import nn, Tensor

from .gvp import _rbf
from .spatial import SpatialMLP, VPU
from .transition import ResidualTransition


class PairTransition(nn.Module):

    def __init__(
        self,
        c_pairs: int,
        pair_n_vecs: int,
        node_n_vecs: int,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
        eps: float = 1e-8,
    ):

        super().__init__()
        self.c_pairs = c_pairs
        self.pair_n_vecs = pair_n_vecs
        self.node_n_vecs = node_n_vecs
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.eps = eps

        self.vpu = VPU(n_vecs=node_n_vecs)
        self.pair_update = SpatialMLP(
            n_feats_in=rbf_dim,
            n_feats_out=c_pairs,
            n_vecs_in=1,
            n_vecs_out=pair_n_vecs,
        )
        self.pair_transition = ResidualTransition(
            dim=c_pairs,
            hidden=c_pairs,
            dropout_prob=0.0,
        )
    
    def compute_pairwise_features(self, xs: Tensor):
        # NOTE: Adding epsilons TWICE is CRITICAL for numerical stability!!!
        rel_pos = xs[:, :, None, :] - xs[:, None, :, :] + self.eps
        dists = torch.square(rel_pos).sum(dim=-1).sqrt() + self.eps
        d = _rbf(dists, D_min=0.0, D_max=self.rbf_dmax, D_count=self.rbf_dim)
        return d, rel_pos
    
    def forward(
        self,
        xs: Tensor,
        vfs: Tensor,
        atom_mask: Tensor,
        pair_repr: Tensor,
        pair_mask: Tensor,
    ):
        # xs: (batch_size, n_atoms, 3)
        # vfs: (batch_size, n_atoms, node_n_vecs, 3)
        # atom_mask: (batch_size, n_atoms)
        # pair_repr: (batch_size, n_atoms, n_atoms, c_pairs)
        # pair_mask: (batch_size, n_atoms, n_atoms)

        xs = self.vpu(vfs, xs, atom_mask)
        # xs: (batch_size, n_atoms, 3)

        d, rel_pos = self.compute_pairwise_features(xs)
        # d: (batch_size, n_atoms, n_atoms, rbf_dim)    
        # rel_pos: (batch_size, n_atoms, n_atoms, 3)

        pair_update = self.pair_update(d, rel_pos.unsqueeze(-2))
        # pair_update: (batch_size, n_atoms, n_atoms, c_pairs)

        pair_repr = self.pair_transition(pair_repr, pair_update)
        # pair_repr: (batch_size, n_atoms, n_atoms, c_pairs)

        return pair_repr, xs