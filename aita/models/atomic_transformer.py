import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import repeat

from itertools import chain
from typing import Tuple, Any, Iterator, Dict, Union

from .modules.encoders import AtomicEncoder
from .modules.decoders import AtomicDecoder
from .modules.decoder_opt import OptimizedAtomicDecoder
from .modules.atomic_encoder_opt import OptimizedAtomicEncoder

from ..utils.graph_utils import rbf
from .layers.primitives import LinearNoBias
from .layers.transition import ResidualTransition
from .layers.embeddings_opt import FourierEmbeddingOpt, AdaLNOpt


################################################################################
# constants: model types
################################################################################

MODEL_STATES = ["flow", "ebm", "inference"]


################################################################################
# functions: loss
################################################################################

def compute_velocity_field_loss(
    preds: Tensor,
    target: Tensor,
    atom_mask: Tensor,
) -> Tensor:
    # check that node mask at least contains 1 atom per molecule
    assert torch.all(atom_mask.sum(dim=-1) > 0), "Atom (node) mask must contain at least 1 atom per molecule"

    per_atom_mse = (preds - target).square().mean(dim=-1)  # (B, N)
    per_atom_mse = per_atom_mse * atom_mask                  # zero out padding

    atoms_per_mol = atom_mask.sum(dim=-1)                    # (B,)
    per_mol_loss = per_atom_mse.sum(dim=-1) / atoms_per_mol  # (B,)

    return per_mol_loss.mean()


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


################################################################################
# classes: EnergyEstimator, AtomicTransformer
################################################################################

class EnergyEstimator(nn.Module):
    """Energy estimator for the Atomic Transformer."""

    def __init__(
        self,
        c_atoms: int,
        c_pairs: int,
        rbf_dim: int = 16,
        rbf_dmax: float = 20.0,
        dropout_prob: float = 0.0,
        time_scale: float = 10.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.c_atoms = c_atoms
        self.c_pairs = c_pairs
        self.rbf_dim = rbf_dim
        self.rbf_dmax = rbf_dmax
        self.dropout_prob = dropout_prob
        self.time_scale = time_scale
        self.eps = eps
        
        # time conditioning
        self.temporal_embedding = FourierEmbeddingOpt(c_atoms)
        self.temporal_condition = AdaLNOpt(c_atoms, c_atoms, activation_fn=F.softplus)

        # MLP for atom embeddings
        self.atom_mlp_w1 = LinearNoBias(c_atoms, c_atoms)
        self.atom_mlp_w2 = LinearNoBias(c_atoms, c_atoms)

        # MLP for pair embeddings
        self.pair_norm   = nn.LayerNorm(c_pairs)
        self.pair_mlp_w1 = nn.Linear(c_pairs, c_pairs, bias=True)
        self.pair_mlp_w2 = nn.Linear(c_pairs, c_pairs, bias=True)

        # MLP for RBF features
        self.rbf_mlp_w1 = LinearNoBias(rbf_dim, c_pairs)
        self.rbf_mlp_w2 = LinearNoBias(c_pairs, c_pairs)

        # MLP for energy estimation
        self.energy_mlp_w1 = LinearNoBias(c_atoms, c_atoms)
        self.energy_mlp_w2 = LinearNoBias(c_atoms, 1)

        # message projection
        self.message_proj = LinearNoBias(c_pairs, c_atoms)

        # interaction residual
        self.interaction_residual = ResidualTransition(dim=c_atoms, hidden=c_atoms, dropout_prob=dropout_prob)

    def energy_forward(
        self,
        time: Tensor,
        x_t: Tensor,
        atom_repr: Tensor,
        pair_repr: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
    ) -> Tensor:
        # time: (B, N)
        # x_t: (B, N, 3)
        # atom_repr: (B, N, c_atoms)
        # pair_repr: (B, N, N, c_pairs)
        # atom_mask: (B, N)
        # pair_mask: (B, N, N)

        # encode the time
        time_emb = self.temporal_embedding(self.time_scale * time)
        # time_emb: (B, N, c_atoms)

        # mlp atoms
        atom_repr = self.atom_mlp_w2(F.silu(self.atom_mlp_w1(atom_repr)))
        atom_repr = atom_repr * atom_mask.unsqueeze(-1)

        # mlp pairs
        pair_repr = self.pair_mlp_w2(F.silu(self.pair_mlp_w1(self.pair_norm(pair_repr))))
        pair_repr = pair_repr * pair_mask.unsqueeze(-1)

        # compute RBF features
        x_diff = x_t[:, :, None, :] - x_t[:, None, :, :] + self.eps
        dij = torch.square(x_diff).sum(dim=-1, keepdim=True).sqrt() + self.eps
        d = rbf(dij.squeeze(-1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        # d: (B, N, N, rbf_dim)

        # mlp RBF
        rbf_emb = self.rbf_mlp_w2(F.silu(self.rbf_mlp_w1(d)))
        rbf_emb = rbf_emb * pair_mask.unsqueeze(-1)
        # rbf_emb: (B, N, N, c_pairs)

        # gate RBF with pair embeddings
        rbf_emb = F.tanh(pair_repr) * rbf_emb

        # message projection
        msgs = self.message_proj(rbf_emb.sum(dim=-2))
        # msgs: (B, N, c_atoms)

        # update atom embeddings via interaction residual
        atom_repr = self.interaction_residual(atom_repr, msgs)
        atom_repr = atom_repr * atom_mask.unsqueeze(-1)
        # atom_repr: (B, N, c_atoms)

        # condition atom embeddings on time
        atom_repr = self.temporal_condition(atom_repr, time_emb)
        atom_repr = atom_repr * atom_mask.unsqueeze(-1)
        # atom_repr: (B, N, c_atoms)

        # energy estimation
        energy = self.energy_mlp_w2(F.silu(self.energy_mlp_w1(atom_repr)))
        # energy: (B, N, 1)

        return (energy.squeeze(-1) * atom_mask).sum(dim=-1) # (B,)

    def forward(
        self,
        time: Tensor,
        x_t: Tensor,
        atom_repr: Tensor,
        pair_repr: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
        return_logprob: bool = False,
        require_grad: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # enable gradients with respect to x_t
        torch_grad = self.training or require_grad
        if torch_grad:
            x_t = x_t.requires_grad_()

        # compute energies
        with torch.set_grad_enabled(torch_grad):
            energies = self.energy_forward(
                time=time,
                x_t=x_t,
                atom_repr=atom_repr,
                pair_repr=pair_repr,
                atom_mask=atom_mask,
                pair_mask=pair_mask,
            )
        # energies: (B,)

        # return log probabilities ( energies )
        if return_logprob:
            return energies

        # compute position gradients
        position_grads = torch.autograd.grad(
            energies.sum(), x_t, create_graph=True
        )[0]
        # position_grads: (B, N, 3)
        return position_grads, energies


    def training_step(
        self,
        time: Tensor,
        x_t: Tensor,
        atom_repr: Tensor,
        pair_repr: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
        z: Tensor,
        sigma_t: Tensor,
    ):
        ####################################################################################
        # prepare inputs
        ####################################################################################
        # enable gradients with respect to x_t
        x_t = x_t.requires_grad_()

        ####################################################################################
        # energy estimation
        ####################################################################################
        # encode the time
        time_emb = self.temporal_embedding(self.time_scale * time)
        # time_emb: (B, N, c_atoms)

        # mlp atoms
        atom_repr = self.atom_mlp_w2(F.silu(self.atom_mlp_w1(atom_repr)))
        atom_repr = atom_repr * atom_mask.unsqueeze(-1)

        # mlp pairs
        pair_repr = self.pair_mlp_w2(F.silu(self.pair_mlp_w1(self.pair_norm(pair_repr))))
        pair_repr = pair_repr * pair_mask.unsqueeze(-1)

        # compute RBF features
        x_diff = x_t[:, :, None, :] - x_t[:, None, :, :] + self.eps
        dij = torch.square(x_diff).sum(dim=-1, keepdim=True).sqrt() + self.eps
        d = rbf(dij.squeeze(-1), D_max=self.rbf_dmax, D_count=self.rbf_dim)
        # d: (B, N, N, rbf_dim)

        # mlp RBF
        rbf_emb = self.rbf_mlp_w2(F.silu(self.rbf_mlp_w1(d)))
        rbf_emb = rbf_emb * pair_mask.unsqueeze(-1)
        # rbf_emb: (B, N, N, c_pairs)

        # gate RBF with pair embeddings
        rbf_emb = F.tanh(pair_repr) * rbf_emb

        # message projection
        msgs = self.message_proj(rbf_emb.sum(dim=-2))
        # msgs: (B, N, c_atoms)

        # update atom embeddings via interaction residual
        atom_repr = self.interaction_residual(atom_repr, msgs)
        atom_repr = atom_repr * atom_mask.unsqueeze(-1)
        # atom_repr: (B, N, c_atoms)

        # condition atom embeddings on time
        # NOTE: the POSITIVE vs NEGATIVE distinction is key here because
        #       in a few lines down, we will create a NEGATIVE sample for the NCE loss
        atom_repr_positive = self.temporal_condition(atom_repr, time_emb)
        atom_repr_positive = atom_repr_positive * atom_mask.unsqueeze(-1)
        # atom_repr_positive: (B, N, c_atoms)

        # energy estimation
        energies = self.energy_mlp_w2(F.silu(self.energy_mlp_w1(atom_repr_positive))) # (B, N, 1)
        energies = (energies.squeeze(-1) * atom_mask).sum(dim=-1) # (B,)
        # energies: (B,)

        # compute position gradients
        position_grads = torch.autograd.grad(
            energies.sum(), x_t, create_graph=True
        )[0]
        # position_grads: (B, N, 3)

        ####################################################################################
        # score matching loss
        ####################################################################################
        score_loss = compute_score_field_loss(
            preds=position_grads,
            target=z,
            sigma_t=sigma_t,
            atom_mask=atom_mask,
        )
        # score_loss: (1,)

        ####################################################################################
        # NCE loss
        ####################################################################################
        # sample negative time
        batch_size, n_atoms = time.size(0), time.size(1)
        perturb = torch.randn(batch_size, device=time.device) * 0.025
        negative_time = time + repeat(perturb, "b -> b n 1", n=n_atoms)
        negative_time = torch.clamp(negative_time, 0.0, 1.0)

        # run negative time through the energy estimator
        negative_time_emb = self.temporal_embedding(self.time_scale * negative_time)
        # negative_time_emb: (B, N, c_atoms)
        negative_atom_repr = self.temporal_condition(atom_repr, negative_time_emb)
        negative_atom_repr = negative_atom_repr * atom_mask.unsqueeze(-1)
        # negative_atom_repr: (B, N, c_atoms)
        negative_energies = self.energy_mlp_w2(F.silu(self.energy_mlp_w1(negative_atom_repr))) # (B, N, 1)
        negative_energies = (negative_energies.squeeze(-1) * atom_mask).sum(dim=-1) # (B,)
        # NCE loss
        nce_loss = -torch.mean(
            energies - torch.logsumexp(
                torch.stack([energies, negative_energies], dim=-1),
                dim=-1,
            )
        )
        # nce_loss: (1,)
        # total loss
        loss = score_loss + nce_loss
        # loss: (1,)

        return {
            "loss": loss,
            "score_loss": score_loss,
            "nce_loss": nce_loss,
        }


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
        rbf_dim: int = 16,
        rbf_dmax: float = 20.0,
        time_scale: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.encoder = OptimizedAtomicEncoder(
            node_feats_in=node_feats_in,
            edge_feats_in=edge_feats_in,
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            dropout_prob=dropout_prob,
        )
        self.decoder = OptimizedAtomicDecoder(
            n_vecs=n_vecs,
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout_prob=dropout_prob,
            bias=bias,
            initial_norm=initial_norm,
        )
        self.ebm = EnergyEstimator(
            c_atoms=c_atoms,
            c_pairs=c_pairs,
            rbf_dim=rbf_dim,
            rbf_dmax=rbf_dmax,
            dropout_prob=dropout_prob,
            time_scale=time_scale,
            eps=eps,
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
        atom_repr, pair_repr = self.encoder(
            x_t=x_t,
            time=time,
            attr=attr,
            atom_index=atom_index,
            pair_feats=pair_feats,
            atom_mask=atom_mask,
            pair_mask=pair_mask,
        )
        velocity, x_h = self.decoder(
            x_h=atom_repr,
            pair_repr=pair_repr,
            atom_mask=atom_mask,
            pair_mask=pair_mask,
        )
        return velocity, x_h, pair_repr
    
    def training_step(
        self,
        x_t: Tensor,
        time: Tensor,
        attr: Tensor,
        atom_index: Tensor,
        pair_feats: Tensor,
        atom_mask: Tensor,
        pair_mask: Tensor,
        target_velocity: Tensor,
    ) -> Dict[str, Tensor]:
        velocity, x_h, pair_repr = self.forward_flow(
            x_t=x_t,
            time=time,
            attr=attr,
            atom_index=atom_index,
            pair_feats=pair_feats,
            atom_mask=atom_mask,
            pair_mask=pair_mask,
        )
        loss = compute_velocity_field_loss(
            preds=velocity,
            target=target_velocity,
            atom_mask=atom_mask,
        )
        return {"loss": loss}