import torch
from torch import nn, Tensor
from typing import Tuple

from ..layers.layernorm import AdaLN
from ..layers.primitives import LinearNoBias
from ..layers.transition import ResidualTransition
from ..layers.embeddings import FourierEmbedding, PositionalEncoding, PairEmbedding


@torch.compile
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


@torch.compile
class AtomEncoderTempered(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
    ) -> None:
        super().__init__()

        self.temporal_embedding   = FourierEmbedding(n_hidden)
        self.positional_embedding = PositionalEncoding(n_hidden)
        self.temperature_embedding   = FourierEmbedding(n_hidden)
        self.initial_embedding = nn.Sequential(
            nn.Linear(n_features + n_hidden, n_hidden),
            nn.SiLU(),
        )
        self.time_to_attr = AdaLN(n_hidden, n_hidden)
        self.temperature_to_attr = AdaLN(n_hidden, n_hidden)
    
    def forward(self, time: Tensor, attr: Tensor, atom_index: Tensor, temperature: Tensor) -> Tensor:
        # NOTE: we expect the atom index and time to be a 1D tensors of shape (N,)
        #       and the attribute tensor to be a 2D tensor of shape (N, n_features)
        #       and the temperature tensor to be a 1D tensor of shape (N,)
        bh = self.temperature_embedding(temperature)
        # bh: (N, n_hidden)
        th = self.temporal_embedding(time)
        # th: (N, n_hidden)
        ph = self.positional_embedding(atom_index)
        # ph: (N, n_hidden)
        z_init = torch.cat([attr, ph], dim=1)
        # z_init: (N, n_features + n_hidden)
        zs = self.initial_embedding(z_init)
        # zs: (N, n_hidden)
        zs = self.time_to_attr(zs, th) + self.temperature_to_attr(zs, bh)
        # zs: (N, n_hidden)
        return zs


class AttributeEncoder(nn.Module):

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
        #       and the attribute tensor to be a 3D tensor of shape (B, L, n_features)
        th = self.temporal_embedding(time)
        # th: (B, L, n_hidden)
        ph = self.positional_embedding(atom_index.view(-1)).reshape(atom_index.size(0), atom_index.size(1), -1)
        # ph: (B, L, n_hidden)
        z_init = torch.cat([attr, ph], dim=-1)
        # z_init: (B, L, n_features + n_hidden)
        zs = self.initial_embedding(z_init)
        # zs: (B, L, n_hidden)
        zs = self.time_to_attr(zs, th)
        # zs: (B, L, n_hidden)
        return zs


class AtomicEncoder(nn.Module):

    def __init__(
        self,
        node_feats_in: int,
        edge_feats_in: int,
        c_atoms: int,
        c_pairs: int,
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        self.node_feats_in = node_feats_in
        self.edge_feats_in = edge_feats_in
        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.dropout_prob = dropout_prob

        self.xyz_embedder  = LinearNoBias(3, c_atoms)
        self.attr_encoder = AttributeEncoder(
            n_features=node_feats_in,
            n_hidden=c_atoms,
        )
        self.atom_embedder = nn.Sequential(
            LinearNoBias(2 * c_atoms, c_atoms),
            nn.SiLU(),
        )
        self.pair_embedder = PairEmbedding(
            edge_feats_in=edge_feats_in,
            edge_feats_out=c_pairs,
            dropout_prob=dropout_prob,
        )
        self.message_proj = LinearNoBias(c_pairs, c_atoms)
        self.interaction_residual = ResidualTransition(dim=c_atoms, hidden=c_atoms, dropout_prob=dropout_prob)
    
    def forward(
        self,
        x_t: Tensor,
        time: Tensor,
        attr: Tensor,
        atom_index: Tensor,
        pair_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # Latent vectors based on initial coordinates
        x_h = self.xyz_embedder(x_t)
        # x_h: (batch_size, n_atoms, c_atoms)

        # Latent vectors based on atom attributes
        attr_init = self.attr_encoder(
            time=time,
            attr=attr,
            atom_index=atom_index,
        )
        # attr_init: (batch_size, n_atoms, c_atoms)

        # Combine all node features
        x_h = self.atom_embedder(torch.cat([x_h, attr_init], dim=-1))
        # x_h: (batch_size, n_atoms, c_atoms)

        # Apply atom mask
        x_h = x_h * atom_mask.unsqueeze(-1)

        # Embed the edge features
        edge_repr = self.pair_embedder(pair_features=pair_feats, pair_mask=pair_mask)
        # edge_repr: (batch_size, n_atoms, n_atoms, c_pairs)

        # Project the edge features to the atom features
        msgs = self.message_proj(edge_repr.sum(dim=-2))
        # msgs: (batch_size, n_atoms, c_atoms)

        # Aggregate the edge features to the atom features
        x_h = self.interaction_residual(x_h, msgs)
        # x_h: (batch_size, n_atoms, c_atoms)

        # Apply node mask
        x_h = x_h * atom_mask.unsqueeze(-1)

        return x_h, edge_repr