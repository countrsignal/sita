import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .primitives import LayerNormEps, LinearNoBias
from .transition import ResidualTransition
from .initialize import lecun_normal_init_, final_init_


class AttentionBlock(nn.Module):

    def __init__(
        self,
        c_atoms: int,
        c_pairs: int,
        n_heads: int,
        dropout_prob: float = 0.0,
        bias: bool = False,
        initial_norm: bool = True,
    ) -> None:
        super(AttentionBlock, self).__init__()
        assert c_atoms % n_heads == 0, "c_atoms must be divisible by n_heads"
        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.n_heads = n_heads
        self.head_dim = c_atoms // n_heads
        self.dropout_prob = dropout_prob

        self.norm = LayerNormEps(c_atoms) if initial_norm else nn.Identity()

        self.q_proj = nn.Linear(c_atoms, c_atoms, bias=bias)
        self.k_proj = nn.Linear(c_atoms, c_atoms, bias=bias)
        self.v_proj = nn.Linear(c_atoms, c_atoms, bias=bias)
        self.out_proj = nn.Linear(c_atoms, c_atoms, bias=bias)

        self.bias_proj = LinearNoBias(c_pairs, n_heads)

        self.residual = ResidualTransition(dim=c_atoms, hidden=c_atoms, dropout_prob=dropout_prob)

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            lecun_normal_init_(proj.weight)
        final_init_(self.out_proj.weight)
        final_init_(self.bias_proj.weight)

    def forward(self, x: Tensor, mask: Tensor, edge_repr: Tensor) -> Tensor:
        # x:         (B, N, c_atoms)
        # mask:      (B, N)            — True = valid token
        # edge_repr: (B, N, N, c_pairs)
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        # (B, n_heads, N, head_dim)

        # Learned attention bias from pair representation
        attn_bias = self.bias_proj(edge_repr).permute(0, 3, 1, 2)
        # (B, n_heads, N, N)

        # Additive mask: -inf for padding positions so they get zeroed by softmax
        pad_mask = torch.where(mask[:, None, None, :], 0.0, torch.finfo(q.dtype).min)
        attn_bias = attn_bias + pad_mask

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.dropout_prob if self.training else 0.0,
        )
        # (B, n_heads, N, head_dim)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.c_atoms)
        attn_out = self.out_proj(attn_out)

        x = self.residual(residual, attn_out)
        x = x * mask.unsqueeze(-1)
        return x