from typing import Optional, Tuple

import torch
from torch import nn


class EBM(nn.Module):
    """Energy-based model powered by Graphormer3D."""

    def __init__(
        self,
        net: nn.Module,
    ) -> None:
        super().__init__()
        self.net = net

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
            time: Diffusion timestep tensor `(batch_size,)`.
            features: Categorical features tensor `(batch_size, num_nodes, num_features)`.
            coordinates: Atomic coordinates tensor `(batch_size, num_nodes, 3)`.
            padding_mask: Padding mask tensor `(batch_size, num_nodes)`.
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