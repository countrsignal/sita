import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import repeat

from typing import Tuple, Dict, Union

from aita.utils.graph_utils import rbf
from .modules.encoders import AtomicEncoderV2
from .modules.decoders import AtomicDecoderEBM
from .layers.primitives import LinearNoBias, LayerNormEps


################################################################################
# functions: loss
################################################################################

def compute_score_field_loss(
    preds: Tensor,
    target: Tensor, # NOTE: for score-matching, this is the noise vector (z)
    sigma_t: Tensor,
    atom_mask: Tensor,
) -> Tensor:
    # check that node mask at least contains 1 atom per molecule
    assert torch.all(atom_mask.sum(dim=-1) > 0), "Atom (node) mask must contain at least 1 atom per molecule"

    per_atom_mse = (sigma_t * preds + target).square().mean(dim=-1)  # (B, N)
    per_atom_mse = per_atom_mse * atom_mask                  # zero out padding

    atoms_per_mol = atom_mask.sum(dim=-1)                    # (B,)
    per_mol_loss = per_atom_mse.sum(dim=-1) / atoms_per_mol  # (B,)

    return per_mol_loss.mean()


def compute_nce_loss(positive_energies: Tensor, negative_energies: Tensor) -> Tensor:
    return -torch.mean(
        positive_energies - torch.logsumexp(
            torch.stack([positive_energies, negative_energies], dim=-1),
            dim=-1,
        )
    )


################################################################################
# classes: AtomicTransformerEBM
################################################################################


class AtomicTransformerEBM(nn.Module):
    """Energy-based model built on the same encoder/decoder backbone as
    ``AtomicTransformerFlow`` but using non-compiled layers so that
    ``torch.autograd.grad(..., create_graph=True)`` works correctly.

    State-dict keys for ``encoder`` and ``decoder`` are identical to
    ``AtomicTransformerFlow``, allowing direct checkpoint loading via
    :meth:`load_backbone_from_flow`.
    """

    def __init__(
        self,
        node_feats_in: int,
        edge_feats_in: int,
        c_atoms: int,
        c_pairs: int,
        n_heads: int = 8,
        n_layers: int = 5,
        dropout_prob: float = 0.0,
        bias: bool = False,
        initial_norm: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = AtomicEncoderV2(
            node_feats_in=node_feats_in,
            edge_feats_in=edge_feats_in,
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            dropout_prob=dropout_prob,
        )
        self.decoder = AtomicDecoderEBM(
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_prob=dropout_prob,
            bias=bias,
            initial_norm=initial_norm,
        )
        self.final_ln = LayerNormEps(c_atoms)
        self.energy_mlp_w1 = LinearNoBias(c_atoms, c_atoms)
        self.energy_mlp_w2 = LinearNoBias(c_atoms, 1)
    
    def compute_rbf_features(
        self,
        x_t: Tensor,
        pair_feats: Tensor,
        pair_mask: Tensor,
        D_min: float = 0.,
        D_max: float = 20.,
        D_count: int = 16,
        eps: float = 1e-8,
    ) -> Tensor:
        x_diff = x_t[:, :, None, :] - x_t[:, None, :, :] + eps
        dij = torch.square(x_diff).sum(dim=-1, keepdim=True).sqrt() + eps
        d = rbf(dij.squeeze(-1), D_min=D_min, D_max=D_max, D_count=D_count)
        pair_feats = torch.cat([pair_feats, d], dim=-1)
        return pair_feats * pair_mask.unsqueeze(-1)
    
    def energy_forward(
        self,
        x_t: Tensor,
        time: Tensor,
        attr: Tensor,
        atom_index: Tensor,
        pair_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        """Run encoder -> decoder -> energy MLP and return per-molecule
        scalar energies (B,)."""
        atom_repr, pair_repr = self.encoder(
            x_t=x_t, time=time, attr=attr, atom_index=atom_index,
            pair_feats=pair_feats, atom_mask=atom_mask, pair_mask=pair_mask,
        )
        # _velocity, 
        x_h = self.decoder(
            x_h=atom_repr, pair_repr=pair_repr,
            atom_mask=atom_mask, pair_mask=pair_mask,
        )
        energy = self.energy_mlp_w2(F.silu(self.energy_mlp_w1(self.final_ln(x_h))))
        return (energy.squeeze(-1) * atom_mask).sum(dim=-1)

    def forward(
        self,
        x_t: Tensor,
        time: Tensor,
        attr: Tensor,
        atom_index: Tensor,
        pair_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
        return_logprob: bool = False,
        require_grad: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Compute per-atom forces and per-molecule energies.

        Returns ``(forces, energies)`` where forces are the negative
        gradient of the summed energy w.r.t. ``x_t``.
        """
        torch_grad = self.training or require_grad
        if torch_grad:
            x_t = x_t.requires_grad_()

            # NOTE: must include gradient tracking through RBF features during training
            #       At test-time, it is expected RBF features are pre-computed.
            pair_feats = self.compute_rbf_features(
                x_t=x_t, pair_feats=pair_feats, pair_mask=pair_mask,
            )

        with torch.set_grad_enabled(torch_grad):
            energies = self.energy_forward(
                x_t=x_t, time=time, attr=attr, atom_index=atom_index,
                pair_feats=pair_feats, atom_mask=atom_mask, pair_mask=pair_mask,
            )

        if return_logprob:
            return energies

        forces = torch.autograd.grad(
            energies.sum(), x_t, create_graph=True,
        )[0]
        return forces, energies

    def training_step(
        self,
        x_t: Tensor,
        time: Tensor,
        attr: Tensor,
        atom_index: Tensor,
        pair_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
        z: Tensor,
        sigma_t: Tensor,
    ) -> Dict[str, Tensor]:
        forces, energies = self.forward(
            x_t=x_t, time=time, attr=attr, atom_index=atom_index,
            pair_feats=pair_feats, atom_mask=atom_mask, pair_mask=pair_mask,
            return_logprob=False, require_grad=True,
        )

        ####################################################################################
        # Score matching loss
        ####################################################################################
        score_loss = compute_score_field_loss(
            preds=forces, target=z,
            sigma_t=sigma_t, atom_mask=atom_mask,
        )
        ####################################################################################
        # NCE loss
        ####################################################################################
        batch_size, n_atoms = time.size(0), time.size(1)
        perturb = torch.randn(batch_size, device=time.device) * 0.025
        negative_time = time + repeat(perturb, "b -> b n 1", n=n_atoms)
        negative_time = torch.clamp(negative_time, 0.0, 1.0)

        # NOTE: must re-compute RBF features for the negative samples
        with torch.no_grad():
            negative_pair_feats = self.compute_rbf_features(
                x_t=x_t.detach(), pair_feats=pair_feats, pair_mask=pair_mask,
            )
        negative_energies = self.energy_forward(
            x_t=x_t.detach(), time=negative_time, attr=attr,
            atom_index=atom_index, pair_feats=negative_pair_feats,
            atom_mask=atom_mask, pair_mask=pair_mask,
        )

        nce_loss = compute_nce_loss(
            positive_energies=energies,
            negative_energies=negative_energies,
        )

        loss = score_loss + nce_loss
        return {
            "loss": loss,
            "score_loss": score_loss,
            "nce_loss": nce_loss,
        }

