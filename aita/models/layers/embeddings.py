import math
from math import pi
from typing import Callable

import dgl
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

from .primitives import LayerNormEps
from .transition import ResidualTransition
from .pair_dropout import get_dropout_mask
from .triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncomingFused,
    TriangleMultiplicationOutgoingFused,
)


class FourierEmbedding(nn.Module):
    """Fourier embedding layer."""

    def __init__(self, dim):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the embeddings.

        """
        super().__init__()
        self.proj = nn.Linear(1, dim)
        torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
        torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times: torch.Tensor,
    ):
        if times.dim() == 1:
            times = rearrange(times, "b -> b 1")

        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)


class PositionalEncoding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        device = h.device
        half_dim = self.dim // 2
        base = torch.log(torch.tensor(10000.0, device=device, dtype=h.dtype)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=h.dtype) * -base)
        embeddings = h[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PairEmbedding(nn.Module):
    
    def __init__(
        self,
        edge_feats_in: int,
        edge_feats_out: int,
        dropout_prob: float,
    ):
        super(PairEmbedding, self).__init__()

        self.dropout_prob = dropout_prob
        self.mlp = nn.Sequential(
            nn.Linear(edge_feats_in, edge_feats_out, bias=False),
            nn.SiLU(),
            nn.Linear(edge_feats_out, edge_feats_out, bias=False)
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_pairs=edge_feats_out,
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_pairs=edge_feats_out,
        )
        self.residual_update = ResidualTransition(edge_feats_out, hidden=edge_feats_out, dropout_prob=0.0)
    
    def forward(self, pair_features: Tensor, pair_mask: Tensor):

        # MLP for edge features
        pair_repr = self.mlp(pair_features) * pair_mask.unsqueeze(-1)
        # pair_repr: (batch_size, num_nodes, num_nodes, edge_feats_out)

        # Triangular multiplication for incoming and outgoing messages
        tmi = self.tri_mul_in(pair_repr=pair_repr)
        tmo = self.tri_mul_out(pair_repr=pair_repr)

        # Apply dropout to both updates
        drop_row = get_dropout_mask(self.dropout_prob, tmi, self.training, columnwise=False)
        tmi = drop_row * tmi

        drop_row = get_dropout_mask(self.dropout_prob, tmo, self.training, columnwise=False)
        tmo = drop_row * tmo

        # Final residual update combining the original representation with the transformed update.
        return self.residual_update(x=pair_repr, attn_out=tmi + tmo)


class PositionalEncodingCached(nn.Module):
    """Sinusoidal positional encoding with cached frequency buffer.

    Mirrors ``PositionalEncodingOpt`` exactly: precomputes frequency bands
    once in ``__init__`` and stores them as a non-persistent buffer,
    avoiding ``torch.log``, ``torch.arange``, and ``torch.exp`` every
    forward call. No ``torch.compile``.
    """

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


class AdaLNFused(nn.Module):
    """Adaptive Layer Normalization with fused scale+bias projection.

    Mirrors ``AdaLNOpt`` exactly (same parameter names, shapes, and init)
    but without any ``torch.compile`` dependency, making it compatible with
    ``torch.autograd.grad(..., create_graph=True)``.

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

        bound = 1.0 / math.sqrt(dim_single_cond)
        self.s_scale_bias = nn.Parameter(torch.empty(dim).uniform_(-bound, bound))

    def forward(self, a: Tensor, s: Tensor) -> Tensor:
        a = self.a_norm(a)
        s = self.s_norm(s)
        scale, bias = self.s_proj(s).chunk(2, dim=-1)
        return self.activation_fn(scale + self.s_scale_bias) * a + bias


class PairEmbeddingFused(nn.Module):
    """Pair embedding with non-compiled fused triangle multiplications.

    Mirrors ``PairEmbeddingOpt`` exactly (same parameter names, shapes,
    and init) but uses ``TriangleMultiplicationIncomingFused`` /
    ``TriangleMultiplicationOutgoingFused`` instead of their compiled
    counterparts, making it compatible with
    ``torch.autograd.grad(..., create_graph=True)``.
    """

    def __init__(
        self,
        edge_feats_in: int,
        edge_feats_out: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()

        self.dropout_prob = dropout_prob
        self.mlp_w1 = nn.Linear(edge_feats_in, edge_feats_out, bias=False)
        self.mlp_w2 = nn.Linear(edge_feats_out, edge_feats_out, bias=False)

        self.tri_mul_in = TriangleMultiplicationIncomingFused(c_pairs=edge_feats_out)
        self.tri_mul_out = TriangleMultiplicationOutgoingFused(c_pairs=edge_feats_out)

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