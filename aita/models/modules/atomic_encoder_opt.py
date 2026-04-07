"""Optimized AtomicEncoder and AttributeEncoder modules.

Drop-in replacements for the classes in encoders.py (lines 83-199).

Optimizations
-------------
* ``AttributeEncoderOpt``: ``nn.Sequential`` removed from the initial
  embedding; ``torch.compile`` applied to the full forward for kernel
  fusion of embedding lookups, concatenation, initial MLP, and adaptive
  layer normalization.  Uses ``FourierEmbeddingOpt``,
  ``PositionalEncodingOpt``, and ``AdaLNOpt`` from ``embeddings_opt.py``.
* ``AtomicEncoderOpt``: ``nn.Sequential`` removed from atom embedder
  (replaced by explicit ``atom_w1`` / ``atom_w2`` linears with
  ``F.silu``).  Message aggregation path (``edge_repr.sum`` ->
  ``message_proj`` -> ``residual transition`` -> masking) compiled as
  a standalone ``_aggregate_and_update`` function to reduce
  kernel-launch overhead.  Uses ``PairEmbeddingOpt`` for edge
  embeddings.

.. note::

   State-dict keys differ from the originals due to structural changes
   (e.g. ``atom_w1`` / ``atom_w2`` replace Sequential indices,
   ``initial_w`` replaces Sequential embedding, sub-layers use fused
   projections).
"""

import torch
from torch import nn, Tensor
from typing import Tuple
import torch.nn.functional as F

from ..layers.primitives import LinearNoBias
from ..layers.transition import ResidualTransition
from ..layers.embeddings_opt import (
    FourierEmbeddingOpt,
    PositionalEncodingOpt,
    AdaLNOpt,
    PairEmbeddingOpt,
)


# ---------------------------------------------------------------------------
# Compiled helper
# ---------------------------------------------------------------------------


@torch.compile
def _aggregate_and_update(
    edge_repr: Tensor,
    msg_weight: Tensor,
    x_h: Tensor,
    interaction_residual: nn.Module,
    atom_mask: Tensor,
) -> Tensor:
    """Fused message aggregation: sum -> project -> residual MLP -> mask."""
    msgs = F.linear(edge_repr.sum(dim=-2), msg_weight)
    x_h = interaction_residual(x_h, msgs)
    return x_h * atom_mask.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class AttributeEncoderOpt(nn.Module):
    """Compiled attribute encoder with optimized sub-layers."""

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
    ) -> None:
        super().__init__()

        self.temporal_embedding = FourierEmbeddingOpt(n_hidden)
        self.positional_embedding = PositionalEncodingOpt(n_hidden)
        self.initial_w = nn.Linear(n_features + n_hidden, n_hidden)
        self.time_to_attr = AdaLNOpt(n_hidden, n_hidden)

        self.forward = torch.compile(self.forward)

    def forward(self, time: Tensor, attr: Tensor, atom_index: Tensor) -> Tensor:
        th = self.temporal_embedding(time)
        ph = self.positional_embedding(
            atom_index.reshape(-1),
        ).reshape(atom_index.size(0), atom_index.size(1), -1)
        zs = F.silu(self.initial_w(torch.cat([attr, ph], dim=-1)))
        return self.time_to_attr(zs, th)


class OptimizedAtomicEncoder(nn.Module):
    """Optimized atomic encoder with compiled sub-paths."""

    def __init__(
        self,
        node_feats_in: int,
        edge_feats_in: int,
        c_atoms: int,
        c_pairs: int,
        n_heads: int = 8,
        n_layers: int = 5,
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        self.node_feats_in = node_feats_in
        self.edge_feats_in = edge_feats_in
        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.dropout_prob = dropout_prob
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.xyz_embedder = LinearNoBias(3, c_atoms)
        self.attr_encoder = AttributeEncoderOpt(
            n_features=node_feats_in,
            n_hidden=c_atoms,
        )

        self.pair_embedder = PairEmbeddingOpt(
            edge_feats_in=edge_feats_in,
            edge_feats_out=c_pairs,
            dropout_prob=dropout_prob,
        )

        self.message_proj = LinearNoBias(c_pairs, c_atoms)
        self.interaction_residual = ResidualTransition(
            dim=c_atoms, hidden=c_atoms, dropout_prob=dropout_prob,
        )

        self.enc_attn_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.enc_attn_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=c_atoms,
                    nhead=n_heads,
                    dropout=dropout_prob,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                    bias=False,
                )
            )

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

        x_h = self.xyz_embedder(x_t)

        attr_init = self.attr_encoder(
            time=time, attr=attr, atom_index=atom_index,
        )

        x_h = x_h + attr_init
        x_h = x_h * atom_mask.unsqueeze(-1)

        edge_repr = self.pair_embedder(
            pair_features=pair_feats, pair_mask=pair_mask,
        )

        x_h = _aggregate_and_update(
            edge_repr=edge_repr,
            msg_weight=self.message_proj.weight,
            x_h=x_h,
            interaction_residual=self.interaction_residual,
            atom_mask=atom_mask,
        )

        for layer in self.enc_attn_layers:
            x_h = layer(x_h, src_key_padding_mask=~atom_mask)
            x_h = x_h * atom_mask.unsqueeze(-1)

        return x_h, edge_repr
