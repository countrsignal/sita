import math
from typing import Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .gvp import _norm_no_nan, _rbf
from .pair_dropout import get_dropout_mask
from .transition import ResidualTransition
from .primitives import LinearNoBias, LayerNormEps
from .triangular_mult import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing


def _vec_norm(vectors: Tensor, mask: Tensor, eps: float = 1e-5) -> Tensor:
    """GVP-style non-trainable vector normalisation (non-compiled)."""
    vn = vectors.square().sum(-1, keepdim=True)
    vn = torch.sqrt(vn.mean(-2, keepdim=True) + eps) + eps
    return (vectors / vn) * mask[..., None, None]


def _norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> Tensor:
    """L2 norm clamped above *eps*."""
    return torch.sqrt(x.square().sum(dim, keepdim=keepdim).clamp(min=eps))


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


class VelocityProjectionV2(nn.Module):
    """Non-compiled VelocityProjection with state-dict keys matching the
    optimized version (``w1.weight``, ``w2.weight``)."""

    __constants__ = ["n_vecs"]

    def __init__(self, n_vecs: int, c_atoms: int) -> None:
        super().__init__()
        self.n_vecs = n_vecs
        self.w1 = nn.Linear(c_atoms, c_atoms, bias=False)
        self.w2 = nn.Linear(c_atoms, n_vecs * 3, bias=False)

    def forward(self, x_h: Tensor, atom_mask: Tensor) -> Tensor:
        vfs = self.w2(F.silu(self.w1(x_h)))
        vfs = vfs * atom_mask.unsqueeze(-1)
        vfs = vfs.unflatten(-1, (self.n_vecs, 3))
        return _vec_norm(vfs, atom_mask)


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
        n_vecs_out: Optional[int] = None,
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

        self.vectors_activation = nn.Identity() if self.n_vecs_out == 1 else nn.Sigmoid()

        self.to_feats_out = nn.Sequential(
            nn.Linear(c_atoms + dim_h, c_atoms),
            nn.SiLU(),
        )

        if self.n_vecs_out > 1:
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

        if self.n_vecs_out > 1:
            gating = self.scalar_to_vector_gates(feats_out).unsqueeze(-1)
        else:
            gating = _norm_no_nan(Vu, axis=-1).unsqueeze(-1)
            Vu = Vu / gating

        vectors_out = self.vectors_activation(gating) * Vu

        vectors_out = vectors_out * atom_mask[..., None, None]
        feats_out = feats_out * atom_mask.unsqueeze(-1)

        return vectors_out, feats_out


class VelocityUpdateV2(nn.Module):
    """Non-compiled VelocityUpdate with state-dict keys matching the
    optimized version (``Wh``, ``Wu``, ``feats_linear``, ``gate_linear``).

    Weight matrices are stored in ``(out, in)`` layout so the forward pass
    uses plain ``torch.matmul`` with no runtime transpose.
    """

    __constants__ = [
        "n_vecs", "n_vecs_out", "c_atoms", "dim_h",
        "_multi_vec", "_residual",
    ]

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        n_vecs_out: Optional[int] = None,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.n_vecs = n_vecs
        self.n_vecs_out = n_vecs if n_vecs_out is None else n_vecs_out
        self.c_atoms = c_atoms
        self._multi_vec = self.n_vecs_out > 1
        self._residual = residual and (self.n_vecs_out == n_vecs)

        dim_h = max(n_vecs, self.n_vecs_out)
        self.dim_h = dim_h

        wh_k = 1.0 / math.sqrt(n_vecs)
        self.Wh = nn.Parameter(
            torch.zeros(dim_h, n_vecs).uniform_(-wh_k, wh_k),
        )

        wu_k = 1.0 / math.sqrt(dim_h)
        self.Wu = nn.Parameter(
            torch.zeros(self.n_vecs_out, dim_h).uniform_(-wu_k, wu_k),
        )

        self.feats_linear = nn.Linear(c_atoms + dim_h, c_atoms)

        if self._multi_vec:
            self.gate_linear = nn.Linear(c_atoms, self.n_vecs_out)
        else:
            self.gate_linear = None

    def forward(self, vfs: Tensor, x_h: Tensor, atom_mask: Tensor) -> Tuple[Tensor, Tensor]:
        Vh = torch.matmul(self.Wh, vfs)   # (H,V) @ (...,V,3) -> (...,H,3)
        Vu = torch.matmul(self.Wu, Vh)     # (U,H) @ (...,H,3) -> (...,U,3)

        sh = _norm(Vh, dim=-1)             # (..., H)
        feats_out = F.silu(
            self.feats_linear(torch.cat((x_h, sh), dim=-1))
        )

        if self._multi_vec:
            gate = torch.sigmoid(self.gate_linear(feats_out).unsqueeze(-1))
            vectors_out = gate * Vu
        else:
            vectors_out = Vu

        vectors_out = vectors_out * atom_mask[..., None, None]
        feats_out = feats_out * atom_mask.unsqueeze(-1)

        if self._residual:
            vectors_out = _vec_norm(vfs + vectors_out, atom_mask)

        return vectors_out, feats_out


class VirtualPositionUpdate(nn.Module):
    """
    Update the virtual positions of the atoms.

    Args:
        n_vecs: Number of velocity vectors per atom.
        c_atoms: Dimensionality of the atom features (node embeddings).
        c_pairs: Dimensionality of the pair features (edge embeddings).
        dropout_prob: Dropout probability.
        coords_range: Range of the coordinates.
    """

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        coords_range: float = 10.0,
    ) -> None:
        super().__init__()

        self.n_vecs = n_vecs
        self.n_vecs_out = 1
        self.c_atoms = c_atoms
        self.coords_range = coords_range

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

        self.vectors_activation = nn.Tanh()

        self.to_feats_out = nn.Sequential(
            nn.Linear(c_atoms + dim_h, c_atoms),
            nn.SiLU(),
        )
        self.scalar_to_vector_gates = nn.Linear(c_atoms, 1)


    def forward(
        self,
        xs: Tensor,
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

        gating = self.scalar_to_vector_gates(feats_out).unsqueeze(-1)

        vectors_out = self.vectors_activation(gating) * Vu
        vectors_out = self.coords_range * vectors_out * atom_mask[..., None, None]

        return xs + vectors_out.squeeze(-2)



class PairTransition(nn.Module):

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        c_pairs: int,
        rbf_dim: int = 16,
        rbf_dmax: float = 20,
        dropout_prob: float = 0.0,
        coords_range: float = 10.0,
    ) -> None:
        super().__init__()

        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.dropout_prob = dropout_prob

        self.vpu = VirtualPositionUpdate(
            n_vecs=n_vecs,
            c_atoms=c_atoms,
            coords_range=coords_range,
        )

        self.pair_update = nn.Sequential(
            LinearNoBias(c_pairs + rbf_dim, c_pairs),
            nn.SiLU(),
            LinearNoBias(c_pairs, c_pairs),
        )

        self.residual_update = ResidualTransition(c_pairs, hidden=c_pairs, dropout_prob=0.0)
    
    def forward(
        self,
        xs: Tensor,
        vfs: Tensor,
        x_h: Tensor,
        atom_mask: Tensor,
        pair_repr: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        # xs: (..., n_atoms, 3)
        # vfs: (..., n_atoms, n_vecs, 3)
        # x_h: (..., c_atoms)
        # atom_mask: (...)
        # pair_repr: (..., n_atoms, n_atoms, c_pairs)
        # pair_mask: (..., n_atoms, n_atoms)

        xs = self.vpu(xs=xs, vfs=vfs, x_h=x_h, atom_mask=atom_mask)

        # compute the pairwise distances
        d = _rbf(
            torch.cdist(xs, xs, p=2.0),
            D_max=self.rbf_dmax,
            D_count=self.rbf_dim,
        )
        pair_repr = self.pair_update(torch.cat([pair_repr, d], dim=-1))
        pair_repr = self.residual_update(x=pair_repr, attn_out=pair_repr)
        pair_repr = pair_repr * pair_mask.unsqueeze(-1)
        return xs, pair_repr