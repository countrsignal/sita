import torch
from torch import nn, Tensor
from typing import Tuple

from .modules.encoders import AtomicEncoder
from .modules.decoders import AtomicDecoder


class AtomicTransformer(nn.Module):
    def __init__(
        self,
        node_feats_in: int,
        edge_feats_in: int,
        n_vecs: int,
        c_atoms: int,
        c_pairs: int,
        n_heads: int = 8,
        n_layers: int = 5,
        dropout_prob: float = 0.0,
        bias: bool = False,
        initial_norm: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = AtomicEncoder(
            node_feats_in=node_feats_in,
            edge_feats_in=edge_feats_in,
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            dropout_prob=dropout_prob,
        )
        self.decoder = AtomicDecoder(
            n_vecs=n_vecs,
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_prob=dropout_prob,
            bias=bias,
            initial_norm=initial_norm,
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x_h, pair_repr = self.encoder(
            x_t=x_t,
            time=time,
            attr=attr,
            atom_index=atom_index,
            pair_feats=pair_feats,
            atom_mask=atom_mask,
            pair_mask=pair_mask,
        )
        velocity, x_h = self.decoder(
            x_h=x_h,
            pair_repr=pair_repr,
            atom_mask=atom_mask,
        )
        return velocity, x_h, pair_repr

    def compute_loss(
        self,
        preds: Tensor,
        target: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        # check that node mask at least contains 1 atom per molecule
        assert torch.all(atom_mask.sum(dim=-1) > 0), "Node mask must contain at least 1 atom per molecule"

        per_atom_mse = (preds - target).square().mean(dim=-1)  # (B, N)
        per_atom_mse = per_atom_mse * atom_mask                  # zero out padding

        atoms_per_mol = atom_mask.sum(dim=-1)                    # (B,)
        per_mol_loss = per_atom_mse.sum(dim=-1) / atoms_per_mol  # (B,)

        return per_mol_loss.mean()