from math import pi
import dgl
import torch
from torch import nn, Tensor
from einops import rearrange

from .pair_dropout import get_dropout_mask
from .transition import ResidualTransition
from .triangular_mul import TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing
from ...utils.graph_utils import dgl_nodes_to_padded_tensor


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


class PairEmbedding(nn.Module):
    
    def __init__(
        self,
        pair_dim_hidden: int,
        dropout_prob: float,
    ):
        super(PairEmbedding, self).__init__()
        self.dropout_prob = dropout_prob
        # MLP for distance embedding: transforms a distance scalar to an embed_dim-dimensional vector
        self.dist_mlp = nn.Sequential(
            nn.Linear(1, pair_dim_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(pair_dim_hidden, pair_dim_hidden, bias=True)
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            dim_pairs=pair_dim_hidden,
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            dim_pairs=pair_dim_hidden,
        )
        self.residual_update = ResidualTransition(pair_dim_hidden, hidden=pair_dim_hidden, dropout_prob=0.0)
    
    def forward(self, dists: Tensor, padding_mask: Tensor, self_mask: Tensor) -> Tensor:
        # Distance-based feature:
        # Apply MLP to each pairwise distance. dist_matrix.unsqueeze(-1) makes it shape [B, N, N, 1] for the MLP.
        pair_repr = self.dist_mlp(dists)

        # mask out padding entries
        pair_repr = pair_repr.masked_fill(padding_mask[:, :, None] | padding_mask[:, None, :], 0.0)
        # mask out self-to-self contributions
        pair_repr = pair_repr.masked_fill(self_mask, 0.0)

        # Compute both triangular multiplication updates
        tmi = self.tri_mul_in(pair_repr=pair_repr)
        tmo = self.tri_mul_out(pair_repr=pair_repr)

        # Apply dropout to both updates
        drop_row = get_dropout_mask(self.dropout_prob, tmi, self.training, columnwise=False)
        tmi = drop_row * tmi

        drop_row = get_dropout_mask(self.dropout_prob, tmo, self.training, columnwise=False)
        tmo = drop_row * tmo

        # Final residual update combining the original representation with the transformed update.
        return self.residual_update(x=pair_repr, y=tmi + tmo)
    
    @staticmethod
    @torch.no_grad()
    def pwd_from_dgl_graph(graph: dgl.DGLGraph) -> Tensor:
        # NOTE: Coords tensor is of shape (batch_size, max_num_nodes, 3)
        # NOTE: Padding mask is of shape (batch_size, max_num_nodes)
        # NOTE: DGLGraph (and its tensors) is assumed to be on the accelerator device
        coords, padding_mask = dgl_nodes_to_padded_tensor(graph, feat_key="xt")

        # compute pairwise distances
        dists = torch.square(coords[:, :, None, :] - coords[:, None, :, :]).sum(dim=-1).sqrt().unsqueeze(-1)

        # create self-mask
        max_num_atoms = coords.size(1)
        self_mask = torch.eye(max_num_atoms, dtype=torch.bool, device=coords.device).unsqueeze(-1)

        # create padding mask
        padding_mask = padding_mask.unsqueeze(-1)

        # mask out padding entries
        dists = dists.masked_fill(padding_mask[:, :, None] | padding_mask[:, None, :], 0.0)
        # mask out self-to-self contributions
        dists = dists.masked_fill(self_mask, 0.0)
        return dists, padding_mask, self_mask