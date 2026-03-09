from typing import Optional

from torch import Tensor, nn


from .primitives import LinearNoBias, LayerNormEps
from .initialize import (
    bias_init_one_,
    bias_init_zero_,
    final_init_,
    lecun_normal_init_,
)


# Adapted from: https://github.com/jwohlwend/boltz.git
class SwiGluTransition(nn.Module):
    """Perform a two-layer MLP."""
    def __init__(
        self,
        dim: int,
        hidden: int,
        out_dim: Optional[int] = None,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the hidden, default 512
        out_dim: Optional[int]
            The dimension of the output, default None

        """
        super(SwiGluTransition, self).__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = LayerNormEps(dim)        
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.silu = nn.SiLU()
        self.hidden = hidden

        lecun_normal_init_(self.fc1.weight)
        lecun_normal_init_(self.fc2.weight)
        final_init_(self.fc3.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class ResidualTransition(nn.Module):
    """Perform residual update with a two-layer MLP."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        dropout_prob: float = 0.0,
        out_dim: Optional[int] = None,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the hidden, default 512
        out_dim: Optional[int]
            The dimension of the output, default None

        """
        super(ResidualTransition, self).__init__()
        if out_dim is None:
            out_dim = dim
        self.dim = dim
        self.hidden = hidden
        
        self.norm = LayerNormEps(dim)
        self.drop1 = nn.Dropout(dropout_prob)
        self.drop2 = nn.Dropout(dropout_prob)
        
        self.fc1 = LinearNoBias(dim, hidden)
        self.fc2 = LinearNoBias(dim, hidden)
        self.fc3 = LinearNoBias(hidden, out_dim)
        self.silu = nn.SiLU()

        lecun_normal_init_(self.fc1.weight)
        lecun_normal_init_(self.fc2.weight)
        final_init_(self.fc3.weight)

    def forward(self, x: Tensor, attn_out: Tensor) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        h = self.norm(x + self.drop1(attn_out))
        h = self.silu(self.fc1(h)) * self.fc2(h)
        x = x + self.fc3(self.drop2(h))
        return x