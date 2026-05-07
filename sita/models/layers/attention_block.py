import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Optional

from .primitives import LayerNormEps, LinearNoBias
from .transition import ResidualTransition
from .initialize import lecun_normal_init_, final_init_


class AttentionBlock(nn.Module):

    def __init__(
        self,
        c_atoms: int,
        n_heads: int,
        dropout_prob: float = 0.0,
        bias: bool = False,
        initial_norm: bool = True,
    ) -> None:
        super(AttentionBlock, self).__init__()
        assert c_atoms % n_heads == 0, "c_atoms must be divisible by n_heads"
        self.c_atoms = c_atoms
        self.n_heads = n_heads
        self.head_dim = c_atoms // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_prob = dropout_prob

        self.norm = LayerNormEps(c_atoms) if initial_norm else nn.Identity()

        self.qkv_proj = nn.Linear(c_atoms, 3 * n_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(n_heads * self.head_dim, c_atoms, bias=bias)

        self.residual = ResidualTransition(dim=c_atoms, hidden=c_atoms, dropout_prob=dropout_prob)

        self._init_weights()

    def _init_weights(self) -> None:
        for w in self.qkv_proj.weight.chunk(3, dim=0):
            lecun_normal_init_(w)
        final_init_(self.out_proj.weight)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        residual: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        # x:         (B, N, c_atoms)
        # mask:      (B, N)            — True = valid token
        residual = x if residual is None else residual
        x = self.norm(x)
        B, N, _ = x.shape

        qkv = self.qkv_proj(x).view(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # (B, n_heads, N, head_dim)


        # attn_mask = mask[:, None, None, :]
        # NOTE: mask values are TRUE for valid tokens and FALSE for padding tokens
        # (B, 1, 1, N)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout_prob if self.training else 0.0,
            scale=self.scale,
        )
        # (B, n_heads, N, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.n_heads * self.head_dim)
        attn_out = self.out_proj(attn_out)

        x = self.residual(residual, attn_out)
        x = x * mask.unsqueeze(-1)
        return x