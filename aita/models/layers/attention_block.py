import torch
from torch import nn, Tensor
from typing import Optional

from .primitives import LayerNormEps
from .transition import ResidualTransition



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
        self.c_atoms = c_atoms
        self.n_heads = n_heads
        self.dropout_prob = dropout_prob
        self.bias = bias
        self.initial_norm = initial_norm

        self.norm = LayerNormEps(c_atoms) if initial_norm else nn.Identity()
        self.mha = nn.MultiheadAttention(
            embed_dim=c_atoms,
            num_heads=n_heads,
            dropout=dropout_prob,
            bias=bias,
            add_bias_kv=bias,
            batch_first=True,
        )
        self.residual = ResidualTransition(dim=c_atoms, hidden=c_atoms, dropout_prob=dropout_prob)
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.norm(x)
        # x: (..., c_atoms)
        # mask: (..., n_atoms)

        attn_out, _ = self.mha(query=x, key=x, value=x, key_padding_mask=~mask)
        # NOTE: We invert the mask to convert True=pad → ignore & False=no pad → use.
        
        x = self.residual(x, attn_out)
        x = x * mask.unsqueeze(-1)
        return x