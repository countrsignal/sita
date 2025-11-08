import math

import torch
from torch import nn, Tensor

from .primitives import LinearNoBias, LayerNormEps
from .initialize import (
    lecun_normal_init_,
    gating_init_,
    final_init_,
)


class TriangleMultiplicationIncoming(nn.Module):
    
    def __init__(
        self,
        dim_pairs: int,
    ):
        super(TriangleMultiplicationIncoming, self).__init__()
        self.dim_pairs = dim_pairs
        
        self.norm_in = LayerNormEps(dim_pairs)
        self.proj_in = LinearNoBias(dim_pairs, 2 * dim_pairs)
        self.gate_in = LinearNoBias(dim_pairs, 2 * dim_pairs)


        lecun_normal_init_(self.proj_in.weight)
        gating_init_(self.gate_in.weight)
        
        self.norm_out = LayerNormEps(dim_pairs)
        self.gate_out = LinearNoBias(dim_pairs, dim_pairs)
        self.proj_out = LinearNoBias(dim_pairs, dim_pairs)


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
        dim_pairs: int,
    ):
        super(TriangleMultiplicationOutgoing, self).__init__()
        self.dim_pairs = dim_pairs
        
        self.norm_in = LayerNormEps(dim_pairs)
        self.proj_in = LinearNoBias(dim_pairs,2 * dim_pairs)
        self.gate_in = LinearNoBias(dim_pairs, 2 * dim_pairs)

        lecun_normal_init_(self.proj_in.weight)
        gating_init_(self.gate_in.weight)
        
        self.norm_out = LayerNormEps(dim_pairs)
        self.gate_out = LinearNoBias(dim_pairs, dim_pairs)
        self.proj_out = LinearNoBias(dim_pairs, dim_pairs)

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