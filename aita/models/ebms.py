from typing import Optional, Tuple, Dict, Any

import torch
from torch import nn
from einops import repeat


class EBM(nn.Module):
    """Energy-based model powered by Graphormer3D."""

    def __init__(
        self,
        net: nn.Module,
    ) -> None:
        super().__init__()
        self.net = net

    def load_from_checkpoint(self, checkpoint_path: str, **kwargs: Any) -> "EBM":
        checkpoint = torch.load(checkpoint_path, **kwargs)
        self.load_state_dict(checkpoint)
        return self

    def forward(
        self,
        time: torch.Tensor,
        features: torch.Tensor,
        coordinates: torch.Tensor,
        padding_mask: torch.Tensor,
        return_logprob: bool = False,
        require_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Graphormer-based energy prediction.

        Args:
            time: Diffusion timestep tensor `(batch_size, 1)`.
                    dtype: float32
            features: Categorical features tensor `(batch_size, num_nodes)`.
                    dtype: int64
            coordinates: Atomic coordinates tensor `(batch_size, num_nodes, 3)`.
                    dtype: float32
            padding_mask: Padding mask tensor `(batch_size, num_nodes)`.
                    dtype: bool
            return_logprob: If `True`, return energies only.
            require_grad: Force gradient tracking even in eval mode.

        Returns:
            `(position_grad, energy)` or energy only when `return_logprob` is `True`.
        """
        torch_grad = self.training or require_grad

        if torch_grad:
            coordinates = coordinates.requires_grad_()

        with torch.set_grad_enabled(torch_grad):
            energy, padding_mask = self.net(
                time, features, coordinates, padding_mask
            )
            # energy: (batch_size, 1)
            # padding_mask: (batch_size, max_nodes)

            if return_logprob:
                return energy

            position_grad = torch.autograd.grad(
                energy.sum(), coordinates, create_graph=True
            )[0]
            # position_grad: (num_nodes, 3)
            return position_grad, energy


    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        # unpack batch
        t = batch["t"]
        z = batch["z"]
        x_t = batch["xt"]
        sigma_t = batch["sigma_t"]
        features = batch["features"]
        padding_mask = batch["padding_mask"]

        # predict energy
        grads, energies = self(
            time=t,
            features=features,
            coordinates=x_t,
            padding_mask=padding_mask,
            return_logprob=False,
            require_grad=True,
        )

        # score matching loss
        squared_errors = torch.square(sigma_t * grads + z).mean(dim=-1) # (B, L)
        # account for padding while computing mean
        score_loss = (
            squared_errors * ~padding_mask
        ).sum(dim=1) / (~padding_mask).sum(dim=1) # (B,)
        score_loss = score_loss.mean()

        # nce loss
        batch_size, n_atoms = t.size(0), t.size(1)
        perturb = torch.randn(batch_size, device=t.device) * 0.025
        negative_t = t + repeat(perturb, "b -> b n 1", n=n_atoms)
        negative_t = torch.clamp(negative_t, 0, 1)
        negative_energies = self(
            time=negative_t,
            features=features,
            coordinates=x_t,
            padding_mask=padding_mask,
            return_logprob=True,
            require_grad=False, # NOTE: no gradient tracking for the NCE loss
        )
        nce_loss = -torch.mean(
            energies - torch.logsumexp(
                torch.cat([energies, negative_energies], dim=-1),
                dim=-1,
                keepdim=True,
            )
        )

        # total loss
        loss = score_loss + nce_loss

        return {
            "loss": loss,
            "score_loss": score_loss,
            "nce_loss": nce_loss,
        }