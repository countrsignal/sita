import math
from typing import Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .gvp import _norm_no_nan
from .primitives import LinearNoBias


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


class SpatialMLP(nn.Module):

    def __init__(
        self,
        n_feats_in: int,
        n_feats_out: int,
        n_vecs_in: int,
        n_vecs_out: int,
        vectors_activation: nn.Module = nn.Sigmoid(),
        vector_gating: bool = True,
    ):
        super().__init__()
        self.n_vecs_in = n_vecs_in
        self.n_vecs_out = n_vecs_out
        self.n_feats_in = n_feats_in
        self.n_feats_out = n_feats_out

        self.feats_mlp = nn.Sequential(
            nn.Linear(n_feats_in, n_feats_out),
            nn.SiLU(),
            nn.Linear(n_feats_out, n_feats_out),
        )

        dim_h = max(n_vecs_in, n_vecs_out)
        self.dim_h = dim_h

        wh_k = 1 / math.sqrt(n_vecs_in)
        self.Wh = nn.Parameter(
            torch.zeros(n_vecs_in, dim_h).uniform_(-wh_k, wh_k)
        )

        wu_k = 1 / math.sqrt(dim_h)
        self.Wu = nn.Parameter(
            torch.zeros(dim_h, self.n_vecs_out).uniform_(-wu_k, wu_k)
        )

        self.vectors_activation = vectors_activation

        if vector_gating:
            self.scalar_to_vector_gates = nn.Linear(n_feats_out + dim_h, self.n_vecs_out)
        else:
            self.scalar_to_vector_gates = None        

        self.fuse_mlp = nn.Sequential(
            nn.Linear(n_feats_out + 3 * self.n_vecs_out, n_feats_out),
            nn.SiLU(),
        )

    def forward(
        self,
        feats_in: Tensor,
        vecs_in: Tensor,
    ) -> Tensor:

        feats_h = self.feats_mlp(feats_in)
        vecs_h = torch.einsum('... v c, v h -> ... h c', vecs_in, self.Wh)
        vecs_h = torch.einsum('... h c, h u -> ... u c', vecs_h, self.Wu)

        sh = _norm_no_nan(vecs_h, axis=-1)
        s = torch.cat((feats_h, sh), dim=-1)

        if self.scalar_to_vector_gates is not None:
            gating = self.scalar_to_vector_gates(s).unsqueeze(-1)
        else:
            gating = _norm_no_nan(vecs_h, axis=-1).unsqueeze(-1)
        vecs_out = self.vectors_activation(gating) * vecs_h

        s = torch.cat((feats_h, vecs_out.flatten(-2, -1)), dim=-1)
        feats_out = self.fuse_mlp(s)
        return feats_out


class VPU(nn.Module):
    """
    Virtual Position Update

    VPU layers take the superposition of velocity vectors and update the virtual position of the atoms.
    The idea is to allow the model to simulate how the atoms would move if we applied the velocity vectors to the atoms.

    """

    def __init__(
        self,
        n_vecs: int
    ):
        super().__init__()
        self.n_vecs = n_vecs

        wh_k = 1 / math.sqrt(n_vecs)
        self.Wh = nn.Parameter(
            torch.zeros(n_vecs, n_vecs).uniform_(-wh_k, wh_k)
        )

        wu_k = 1 / math.sqrt(n_vecs)
        self.Wu = nn.Parameter(
            torch.zeros(n_vecs, 1).uniform_(-wu_k, wu_k)
        )

    def forward(
        self,
        vfs: Tensor,
        positions: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        # vfs: (..., n_vecs, 3)
        # positions: (..., 3)
        # atom_mask: (...)
        Vh = torch.einsum('... v c, v h -> ... h c', vfs, self.Wh)
        displacement = torch.einsum('... h c, h u -> ... u c', Vh, self.Wu).squeeze(-2)
        return positions + displacement * atom_mask[..., None]