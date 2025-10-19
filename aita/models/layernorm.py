import torch
from torch import nn, sigmoid
from torch.nn import (
    LayerNorm,
    Linear,
)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization"""

    def __init__(self, dim, dim_single_cond):
        """Initialize the adaptive layer normalization.

        Parameters
        ----------
        dim : int
            The input dimension.
        dim_single_cond : int
            The single condition dimension.

        """
        super().__init__()
        self.a_norm = LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = LayerNorm(dim_single_cond, bias=False)
        self.s_scale = Linear(dim_single_cond, dim)
        self.s_bias = Linear(dim_single_cond, dim, bias=False)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a