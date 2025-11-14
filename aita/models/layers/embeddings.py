from math import pi
import dgl
import torch
from torch import nn, Tensor
from einops import rearrange


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