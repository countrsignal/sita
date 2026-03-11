from math import pi
import dgl
import torch
from torch import nn, Tensor
from einops import rearrange

from .primitives import LayerNormEps
from .transition import ResidualTransition
from .pair_dropout import get_dropout_mask
from .triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
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