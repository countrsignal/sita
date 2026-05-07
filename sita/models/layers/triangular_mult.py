import math

import torch
from torch import nn, Tensor

from .primitives import LinearNoBias, LayerNormEps
from .initialize import (
    lecun_normal_init_,
    bias_init_one_,
    bias_init_zero_,
    gating_init_,
    final_init_,
)


class TriangleMultiplicationIncoming(nn.Module):
    
    def __init__(
        self,
        c_pairs: int,
    ):
        super(TriangleMultiplicationIncoming, self).__init__()
        self.c_pairs = c_pairs
        
        self.norm_in = LayerNormEps(c_pairs)
        self.proj_in = LinearNoBias(c_pairs, 2 * c_pairs)
        self.gate_in = LinearNoBias(c_pairs, 2 * c_pairs)


        lecun_normal_init_(self.proj_in.weight)
        gating_init_(self.gate_in.weight)
        
        self.norm_out = LayerNormEps(c_pairs)
        self.gate_out = LinearNoBias(c_pairs, c_pairs)
        self.proj_out = LinearNoBias(c_pairs, c_pairs)


        gating_init_(self.gate_out.weight)
        final_init_(self.proj_out.weight)
    
    def forward(self, pair_repr: Tensor):
        # Input gating: D -> D
        x = self.norm_in(pair_repr)
        x_in = x
        x = self.proj_in(x) * self.gate_in(x).sigmoid()

        # Split input and cast to float
        a, b = torch.chunk(x, 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bkid,bkjd->bijd", a, b)

        # Output gating
        x = self.proj_out(self.norm_out(x)) * self.gate_out(x_in).sigmoid()
        return x


class TriangleMultiplicationOutgoing(nn.Module):
    
    def __init__(
        self,
        c_pairs: int,
    ):
        super(TriangleMultiplicationOutgoing, self).__init__()
        self.c_pairs = c_pairs
        
        self.norm_in = LayerNormEps(c_pairs)
        self.proj_in = LinearNoBias(c_pairs,2 * c_pairs)
        self.gate_in = LinearNoBias(c_pairs, 2 * c_pairs)

        lecun_normal_init_(self.proj_in.weight)
        gating_init_(self.gate_in.weight)
        
        self.norm_out = LayerNormEps(c_pairs)
        self.gate_out = LinearNoBias(c_pairs, c_pairs)
        self.proj_out = LinearNoBias(c_pairs, c_pairs)

        gating_init_(self.gate_out.weight)
        final_init_(self.proj_out.weight)
    
    
    def forward(self, pair_repr: Tensor):
        # Input gating: D -> D
        x = self.norm_in(pair_repr)
        x_in = x
        x = self.proj_in(x) * self.gate_in(x).sigmoid()

        # Split input and cast to float
        a, b = torch.chunk(x, 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bikd,bjkd->bijd", a, b)

        # Output gating
        x = self.proj_out(self.norm_out(x)) * self.gate_out(x_in).sigmoid()
        return x


class TriangleMultiplicationIncomingFused(nn.Module):
    """Fused triangle multiplication (incoming) with fused projections.

    Mirrors ``TriangleMultiplicationIncomingOpt`` exactly (same parameter
    names, shapes, and init) but without ``torch.compile``, making it
    compatible with ``torch.autograd.grad(..., create_graph=True)``.
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


class TriangleMultiplicationOutgoingFused(nn.Module):
    """Fused triangle multiplication (outgoing) with fused projections.

    Mirrors ``TriangleMultiplicationOutgoingOpt`` exactly (same parameter
    names, shapes, and init) but without ``torch.compile``, making it
    compatible with ``torch.autograd.grad(..., create_graph=True)``.
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