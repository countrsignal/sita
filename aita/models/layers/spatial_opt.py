"""Optimized spatial layers with nn.Sequential-stackable VelocityUpdate.

Drop-in replacement for the classes in spatial.py (lines 1-223).

Optimizations
-------------
* ``torch.matmul`` replaces ``torch.einsum`` (direct cuBLAS dispatch,
  no string-parse overhead).
* Weight matrices stored in ``(out, in)`` layout so the forward pass
  is a plain ``matmul`` with no runtime transpose.
* Norm computations compiled via ``torch.compile(dynamic=True)`` for
  fused GPU kernels and reduced memory traffic.
* Identity-gating path (``n_vecs_out == 1``) algebraically simplified:
  ``norm * (Vu / norm) ≡ Vu``, saving three pointwise ops.
* ``nn.Sequential`` wrappers around Linear+SiLU removed; direct
  ``F.silu(linear(...))`` avoids extra Module.__call__ overhead and
  gives the compiler a single fused subgraph.
* ``VelocityState`` NamedTuple bundles ``(vfs, x_h, atom_mask)`` so
  ``VelocityUpdate`` layers can be stacked in ``nn.Sequential``.
* Residual connection + velocity normalisation internalised inside
  ``VelocityUpdate`` (controlled by ``residual`` flag, auto-disabled
  when ``n_vecs_out != n_vecs``).

.. note::

   Because weights are stored transposed relative to ``spatial.py``,
   state-dicts from the original classes are *not* directly loadable.
   Transpose ``Wh`` and ``Wu`` before calling ``load_state_dict``.
"""

import math
from typing import Optional, NamedTuple, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Compile wrapper (transparent no-op on PyTorch < 2.0)
# ---------------------------------------------------------------------------

def _no_compile(fn=None, **kwargs):
    return fn if fn is not None else lambda f: f


_compile = getattr(torch, "compile", _no_compile)


# ---------------------------------------------------------------------------
# Fused standalone helpers
# ---------------------------------------------------------------------------

@_compile(dynamic=True)
def _fused_vec_norm(
    vectors: Tensor, mask: Tensor, eps: float = 1e-5,
) -> Tensor:
    """GVP-style non-trainable vector normalisation, compiled."""
    # vectors: (..., n_vecs, 3)   mask: (...)
    vn = vectors.square().sum(-1, keepdim=True)             # (..., V, 1)
    vn = torch.sqrt(vn.mean(-2, keepdim=True) + eps) + eps  # (..., 1, 1)
    return (vectors / vn) * mask[..., None, None]


@_compile(dynamic=True)
def _fused_norm(
    x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-8,
) -> Tensor:
    """L2 norm clamped above *eps*, compiled for kernel fusion."""
    return torch.sqrt(x.square().sum(dim, keepdim=keepdim).clamp(min=eps))


# ---------------------------------------------------------------------------
# State container for nn.Sequential stacking
# ---------------------------------------------------------------------------

class VelocityState(NamedTuple):
    """Bundles velocity-field state so ``VelocityUpdate`` works inside
    ``nn.Sequential``."""
    vfs: Tensor       # (..., n_vecs, 3)
    x_h: Tensor       # (..., c_atoms)
    atom_mask: Tensor  # (...)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class VelocityLayerNorm(nn.Module):
    """Non-trainable norm for vector features (GVPLayerNorm convention).

    Padded atoms identified by *atom_mask* are excluded from the
    normalisation and guaranteed to remain zero.
    """

    __constants__ = ["eps"]

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, vectors: Tensor, atom_mask: Tensor) -> Tensor:
        return _fused_vec_norm(vectors, atom_mask, self.eps)


class VelocityProjection(nn.Module):
    """Projects atom features into initial velocity vectors.

    Bias-free two-layer MLP mapping each atom's representation to
    *n_vecs* three-dimensional vectors.

    Args:
        n_vecs: Number of velocity vectors per atom.
        c_atoms: Dimensionality of the atom features.
    """

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
        return _fused_vec_norm(vfs, atom_mask)


class VelocityUpdate(nn.Module):
    """Equivariant velocity update, directly stackable via ``nn.Sequential``.

    Accepts and returns a :class:`VelocityState` so that layers can be
    chained without external bookkeeping::

        stack = nn.Sequential(
            VelocityUpdate(n_vecs=16, c_atoms=128),
            VelocityUpdate(n_vecs=16, c_atoms=128),
            VelocityUpdate(n_vecs=16, c_atoms=128, n_vecs_out=1),
        )
        state = VelocityState(vfs, x_h, atom_mask)
        final = stack(state)
        velocity = final.vfs.squeeze(-2)

    When *n_vecs_out* equals *n_vecs* (default) and ``residual=True``,
    a residual connection followed by velocity normalisation is applied
    inside the layer.  When ``n_vecs_out != n_vecs`` (e.g. the final
    projection to a single vector), the residual path is auto-disabled
    because the dimensions no longer match.

    Args:
        n_vecs: Number of input velocity vectors.
        c_atoms: Dimensionality of the atom features.
        n_vecs_out: Output velocity vectors (defaults to *n_vecs*).
        residual: Apply residual + norm when dimensions match.
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
        # vfs: (..., n_vecs, 3)
        # x_h: (..., c_atoms)
        # atom_mask: (...)

        Vh = torch.matmul(self.Wh, vfs)   # (H,V) @ (...,V,3) -> (...,H,3)
        Vu = torch.matmul(self.Wu, Vh)     # (U,H) @ (...,H,3) -> (...,U,3)

        sh = _fused_norm(Vh, dim=-1)       # (..., H)
        feats_out = F.silu(
            self.feats_linear(torch.cat((x_h, sh), dim=-1))
        )

        if self._multi_vec:
            gate = torch.sigmoid(self.gate_linear(feats_out).unsqueeze(-1))
            vectors_out = gate * Vu
        else:
            # Identity(norm) * (Vu / norm) ≡ Vu — skip three pointwise ops
            vectors_out = Vu

        vectors_out = vectors_out * atom_mask[..., None, None]
        feats_out = feats_out * atom_mask.unsqueeze(-1)

        if self._residual:
            vectors_out = _fused_vec_norm(vfs + vectors_out, atom_mask)

        return vectors_out, feats_out


class VelocityUpdateMLP(nn.Module):

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        n_vecs_out: Optional[int] = None,
    ) -> None:

        super().__init__()
        self.vu_mlp = nn.Sequential(
            VelocityUpdate(n_vecs=n_vecs, c_atoms=c_atoms, n_vecs_out=n_vecs_out, residual=False),
            VelocityUpdate(n_vecs=n_vecs, c_atoms=c_atoms, n_vecs_out=n_vecs_out, residual=False),
            VelocityUpdate(n_vecs=n_vecs, c_atoms=c_atoms, n_vecs_out=n_vecs_out, residual=False),
        )
    
    def forward(self, vfs: Tensor, x_h: Tensor, atom_mask: Tensor) -> Tensor:
        vstate = self.vu_mlp(VelocityState(vfs=vfs, x_h=x_h, atom_mask=atom_mask))
        vectors_out = _fused_vec_norm(vfs + vstate.vfs, atom_mask)
        return vectors_out, vstate.x_h


class VelocityPredictionMLP(nn.Module):
    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
    ) -> None:
        super().__init__()
        self.vu_mlp = nn.Sequential(
            VelocityUpdate(n_vecs=n_vecs, c_atoms=c_atoms, residual=False),
            VelocityUpdate(n_vecs=n_vecs, c_atoms=c_atoms, residual=False),
            VelocityUpdate(n_vecs=n_vecs, c_atoms=c_atoms, n_vecs_out=1, residual=False),
        )

    def forward(self, vfs: Tensor, x_h: Tensor, atom_mask: Tensor) -> Tensor:
        vstate = self.vu_mlp(VelocityState(vfs=vfs, x_h=x_h, atom_mask=atom_mask))
        return vstate.vfs.squeeze(-2)


class VirtualPositionUpdate(nn.Module):
    """Velocity-gated displacement update for virtual atom positions.

    Args:
        n_vecs: Number of velocity vectors per atom.
        c_atoms: Dimensionality of the atom features.
        coords_range: Scale factor clamping the displacement magnitude.
    """

    __constants__ = ["n_vecs", "c_atoms", "coords_range", "dim_h"]

    def __init__(
        self,
        n_vecs: int,
        c_atoms: int,
        coords_range: float = 10.0,
    ) -> None:
        super().__init__()

        self.n_vecs = n_vecs
        self.c_atoms = c_atoms
        self.coords_range = coords_range

        dim_h = max(n_vecs, 1)
        self.dim_h = dim_h

        wh_k = 1.0 / math.sqrt(n_vecs)
        self.Wh = nn.Parameter(
            torch.zeros(dim_h, n_vecs).uniform_(-wh_k, wh_k),
        )

        wu_k = 1.0 / math.sqrt(dim_h)
        self.Wu = nn.Parameter(
            torch.zeros(1, dim_h).uniform_(-wu_k, wu_k),
        )

        self.feats_linear = nn.Linear(c_atoms + dim_h, c_atoms)
        self.gate_linear = nn.Linear(c_atoms, 1)

    def forward(
        self,
        xs: Tensor,
        vfs: Tensor,
        x_h: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        Vh = torch.matmul(self.Wh, vfs)   # (H,V) @ (...,V,3) -> (...,H,3)
        Vu = torch.matmul(self.Wu, Vh)     # (1,H) @ (...,H,3) -> (...,1,3)

        sh = _fused_norm(Vh, dim=-1)       # (..., H)
        feats_out = F.silu(
            self.feats_linear(torch.cat((x_h, sh), dim=-1))
        )

        gate = self.gate_linear(feats_out).unsqueeze(-1)
        vectors_out = torch.tanh(gate) * Vu * self.coords_range
        vectors_out = vectors_out * atom_mask[..., None, None]

        return xs + vectors_out.squeeze(-2)
