import dgl
import torch
from torch import nn, Tensor
from typing import Optional

from ..layers.layernorm import AdaLN
from ..layers.transition import Transition
from ..layers.primitives import LinearNoBias
from ..layers.embeddings import FourierEmbedding, PositionalEncoding, PairEmbedding
from ...utils.logging import RankedLogger


log = RankedLogger(__name__, on_rank_zero=True)


class AtomConditioning(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_transitions: int,
    ) -> None:
        super().__init__()

        self.temporal_embedding   = FourierEmbedding(n_hidden)
        self.positional_embedding = PositionalEncoding(n_hidden)
        self.initial_embedding = nn.Sequential(
            nn.Linear(n_features + n_hidden, n_hidden),
            nn.SiLU(),
        )
        self.time_to_attr =  AdaLN(n_hidden, n_hidden)
    
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


class AtomEncoder(nn.Module):

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_transitions: int,
        pair_dropout: float,
        pair_dim_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()

        if pair_dim_hidden is None:
            pair_dim_hidden = n_hidden

        self.n_hidden = n_hidden
        self.atom_conditioning = AtomConditioning(n_features, n_hidden, n_transitions)
        self.pair_embedding = PairEmbedding(pair_dim_hidden, pair_dropout)
        self.pair_to_single = LinearNoBias(pair_dim_hidden, n_hidden)
        self.adaln = AdaLN(n_hidden, n_hidden)
    
    def forward(self, graph: dgl.DGLGraph) -> Tensor:
        """Run the vector field forward pass.

        Args:
            t: Diffusion timestep of shape `(batch_size,)`.
            graph: Batched DGL graph with node features `ndata['h']` `(num_nodes, n_features)`
                and coordinates `ndata['x']` `(num_nodes, 3)`.
        Returns:
            Vector field predictions of shape `(num_nodes, 3)`.
        """
        # Compute the node representation
        node_repr = self.atom_conditioning(
            time=graph.ndata["t"].view(-1),
            attr=graph.ndata["attr"],
            atom_index=graph.ndata["atom_index"].view(-1),
        )
        # node_repr: (num_nodes, n_hidden)

        # Gather pairwise distances and masks
        dists, padding_mask, self_mask = PairEmbedding.pwd_from_dgl_graph(graph)

        # Compute the pair representation
        pair_repr = self.pair_embedding(
            dists=dists,
            padding_mask=padding_mask,
            self_mask=self_mask,
        )
        # pair_repr: (num_nodes, num_nodes, pair_dim_hidden)

        # Aggregate the pair representation to the node representation
        msg = self.pair_to_single(pair_repr.mean(dim=2)).view(-1, self.n_hidden)
        node_repr = self.adaln(node_repr, msg)
        # node_repr: (num_nodes, n_hidden)

        return node_repr, pair_repr
