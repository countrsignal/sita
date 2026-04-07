"""Optimized embedding and pair-representation layers.

Drop-in replacements for FourierEmbedding, PositionalEncoding, and
PairEmbedding from embeddings.py, plus an optimized AdaLN from
layernorm.py.

Optimizations
-------------
* ``FourierEmbeddingOpt``: ``torch.unsqueeze`` replaces ``einops.rearrange``,
  removing string-parsing overhead.
* ``PositionalEncodingOpt``: frequency bands precomputed once in ``__init__``
  and stored as a non-learnable buffer, avoiding ``torch.log``,
  ``torch.arange``, and ``torch.exp`` every forward call.
* ``AdaLNOpt``: ``s_scale`` and ``s_bias`` fused into a single
  ``Linear(dim_cond, 2 * dim, bias=False)`` + chunk, halving GEMM kernel
  launches.  A separate ``s_scale_bias`` parameter preserves the scale-path
  bias from the original ``s_scale``.
* ``TriangleMultiplicationIncomingOpt`` / ``TriangleMultiplicationOutgoingOpt``:
  ``proj_in`` and ``gate_in`` fused into one ``Linear(P, 4P)``, halving GEMM
  launches on (B, N^2) data.  ``torch.matmul`` replaces ``torch.einsum`` for
  direct cuBLAS batched-GEMM dispatch.  Forward compiled via
  ``torch.compile`` to fuse SafeLayerNorm reparameterization, sigmoid
  gating, and output gating into fewer GPU kernels.
* ``PairEmbeddingOpt``: ``nn.Sequential`` removed from MLP (direct
  ``F.silu`` call).

.. note::

   State-dict keys differ from the original classes due to fused
   projections (``fused_in`` replaces ``proj_in`` + ``gate_in``,
   ``s_proj`` + ``s_scale_bias`` replace ``s_scale`` + ``s_bias``).
"""

import math
from typing import Callable

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .primitives import LinearNoBias, LayerNormEps
from .transition import ResidualTransition
from .pair_dropout import get_dropout_mask
from .initialize import (
    lecun_normal_init_,
    gating_init_,
    final_init_,
)


# ---------------------------------------------------------------------------
# Optimized embeddings
# ---------------------------------------------------------------------------


class FourierEmbeddingOpt(nn.Module):
    """Fourier embedding without einops dependency."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(1, dim)
        torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
        torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(self, times: Tensor) -> Tensor:
        if times.dim() == 1:
            times = times.unsqueeze(-1)
        return torch.cos(2 * math.pi * self.proj(times))


class PositionalEncodingOpt(nn.Module):
    """Sinusoidal positional encoding with cached frequency buffer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        base = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -base)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, h: Tensor) -> Tensor:
        embeddings = h.unsqueeze(-1) * self.freqs
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


# ---------------------------------------------------------------------------
# Optimized AdaLN
# ---------------------------------------------------------------------------


class AdaLNOpt(nn.Module):
    """Adaptive Layer Normalization with fused scale+bias projection.

    Fuses the original ``s_scale`` and ``s_bias`` linears into a single
    ``s_proj`` (bias-free) plus a learnable ``s_scale_bias`` vector that
    replicates the bias term of the original scale projection.
    """

    def __init__(self, dim: int, dim_single_cond: int, activation_fn: Callable = F.sigmoid) -> None:
        super().__init__()
        self.dim = dim
        self.activation_fn = activation_fn
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_proj = nn.Linear(dim_single_cond, 2 * dim, bias=False)

        # Match original nn.Linear bias initialization
        bound = 1.0 / math.sqrt(dim_single_cond)
        self.s_scale_bias = nn.Parameter(torch.empty(dim).uniform_(-bound, bound))
        
        # NOTE: this has a HUGE impact on sample quality
        # self.s_scale_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        a = self.a_norm(a)
        s = self.s_norm(s)
        scale, bias = self.s_proj(s).chunk(2, dim=-1)
        return self.activation_fn(scale + self.s_scale_bias) * a + bias


# ---------------------------------------------------------------------------
# Optimized triangle multiplications
# ---------------------------------------------------------------------------


class TriangleMultiplicationIncomingOpt(nn.Module):
    """Compiled triangle multiplication (incoming) with fused projections.

    Replaces separate ``proj_in`` and ``gate_in`` with a single
    ``fused_in`` linear (P -> 4P), and uses ``torch.matmul`` instead
    of ``torch.einsum`` for the triangular contraction.
    """

    def __init__(self, c_pairs: int) -> None:
        super().__init__()
        self.c_pairs = c_pairs

        self.norm_in = LayerNormEps(c_pairs)
        self.fused_in = LinearNoBias(c_pairs, 4 * c_pairs)

        lecun_normal_init_(self.fused_in.weight[:2 * c_pairs])
        gating_init_(self.fused_in.weight[2 * c_pairs:])

        self.norm_out = LayerNormEps(c_pairs)
        self.gate_out = LinearNoBias(c_pairs, c_pairs)
        self.proj_out = LinearNoBias(c_pairs, c_pairs)

        gating_init_(self.gate_out.weight)
        final_init_(self.proj_out.weight)

        self.forward = torch.compile(self.forward)

    def forward(self, pair_repr: Tensor) -> Tensor:
        x = self.norm_in(pair_repr)
        x_in = x

        proj, gate = self.fused_in(x).chunk(2, dim=-1)
        x = proj * gate.sigmoid()
        a, b = x.chunk(2, dim=-1)

        # "bkid,bkjd->bijd" via batched matmul over (B, D) dims
        x = torch.matmul(
            a.permute(0, 3, 2, 1),
            b.permute(0, 3, 1, 2),
        ).permute(0, 2, 3, 1)

        return self.proj_out(self.norm_out(x)) * self.gate_out(x_in).sigmoid()


class TriangleMultiplicationOutgoingOpt(nn.Module):
    """Compiled triangle multiplication (outgoing) with fused projections.

    Same optimizations as the incoming variant; only the contraction
    indices differ (``bikd,bjkd->bijd`` instead of ``bkid,bkjd->bijd``).
    """

    def __init__(self, c_pairs: int) -> None:
        super().__init__()
        self.c_pairs = c_pairs

        self.norm_in = LayerNormEps(c_pairs)
        self.fused_in = LinearNoBias(c_pairs, 4 * c_pairs)

        lecun_normal_init_(self.fused_in.weight[:2 * c_pairs])
        gating_init_(self.fused_in.weight[2 * c_pairs:])

        self.norm_out = LayerNormEps(c_pairs)
        self.gate_out = LinearNoBias(c_pairs, c_pairs)
        self.proj_out = LinearNoBias(c_pairs, c_pairs)

        gating_init_(self.gate_out.weight)
        final_init_(self.proj_out.weight)

        self.forward = torch.compile(self.forward)

    def forward(self, pair_repr: Tensor) -> Tensor:
        x = self.norm_in(pair_repr)
        x_in = x

        proj, gate = self.fused_in(x).chunk(2, dim=-1)
        x = proj * gate.sigmoid()
        a, b = x.chunk(2, dim=-1)

        # "bikd,bjkd->bijd" via batched matmul over (B, D) dims
        x = torch.matmul(
            a.permute(0, 3, 1, 2),
            b.permute(0, 3, 2, 1),
        ).permute(0, 2, 3, 1)

        return self.proj_out(self.norm_out(x)) * self.gate_out(x_in).sigmoid()


# ---------------------------------------------------------------------------
# Optimized pair embedding
# ---------------------------------------------------------------------------


class PairEmbeddingOpt(nn.Module):
    """Pair embedding with compiled triangle mults and optimized MLP.

    Removes ``nn.Sequential`` from the MLP path.
    """

    def __init__(
        self,
        edge_feats_in: int,
        edge_feats_out: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()

        self.dropout_prob = dropout_prob
        self.mlp_w1 = nn.Linear(edge_feats_in, edge_feats_out, bias=True)
        self.mlp_w2 = nn.Linear(edge_feats_out, edge_feats_out, bias=True)

        self.tri_mul_in = TriangleMultiplicationIncomingOpt(c_pairs=edge_feats_out)
        self.tri_mul_out = TriangleMultiplicationOutgoingOpt(c_pairs=edge_feats_out)

        self.residual_update = ResidualTransition(
            edge_feats_out, hidden=edge_feats_out, dropout_prob=0.0,
        )

    def forward(self, pair_features: Tensor, pair_mask: Tensor) -> Tensor:
        pair_repr = self.mlp_w2(F.silu(self.mlp_w1(pair_features)))
        pair_repr = pair_repr * pair_mask.unsqueeze(-1)

        tmi = self.tri_mul_in(pair_repr=pair_repr)
        tmo = self.tri_mul_out(pair_repr=pair_repr)

        drop_row = get_dropout_mask(
            self.dropout_prob, tmi, self.training, columnwise=False,
        )
        tmi = drop_row * tmi

        drop_row = get_dropout_mask(
            self.dropout_prob, tmo, self.training, columnwise=False,
        )
        tmo = drop_row * tmo

        return self.residual_update(x=pair_repr, attn_out=tmi + tmo)
