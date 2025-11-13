import torch
from torch import nn, Tensor

from ..layers.layernorm import AdaLN
from ..layers.embeddings import FourierEmbedding, PositionalEncoding


class AtomEncoder(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
    ) -> None:
        super().__init__()

        self.temporal_embedding   = FourierEmbedding(n_hidden)
        self.positional_embedding = PositionalEncoding(n_hidden)
        self.initial_embedding = nn.Sequential(
            nn.Linear(n_features + n_hidden, n_hidden),
            nn.SiLU(),
        )
        self.time_to_attr = AdaLN(n_hidden, n_hidden)
    
    def forward(self, time: Tensor, attr: Tensor, atom_index: Tensor) -> Tensor:
        # NOTE: we expect the atom index and time to be a 1D tensors of shape (N,)
        #       and the attribute tensor to be a 2D tensor of shape (N, n_features)
        th = self.temporal_embedding(time)
        # th: (N, n_hidden)
        ph = self.positional_embedding(atom_index)
        # ph: (N, n_hidden)
        z_init = torch.cat([attr, ph], dim=1)
        # z_init: (N, n_features + n_hidden)
        zs = self.initial_embedding(z_init)
        # zs: (N, n_hidden)
        zs = self.time_to_attr(zs, th)
        # zs: (N, n_hidden)
        return zs