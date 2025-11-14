import torch
import torch.nn.functional as F
from torch import nn, sigmoid


class SafeLayerNorm(nn.Module):
    """
    A reparameterized version of LayerNorm that works better with weight decay.
    
    In standard LayerNorm, the scale parameter (gamma) is initialized to ones,
    which can cause issues when applying weight decay, as it biases the scale
    parameter towards smaller values.
    
    This implementation reparameterizes the scale as `scale = 1 + gamma`, where
    gamma is now initialized to zeros. This allows gamma to be centered around zero,
    making it more compatible with weight decay while preserving the initial
    scale of 1 for the LayerNorm operation.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, bias=True):
        super(SafeLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            # Initialize gamma to zeros (instead of ones in standard LayerNorm)
            # This ensures that scale = 1 + gamma starts at 1
            self.gamma = nn.Parameter(torch.zeros(normalized_shape))
            # Beta is still initialized to zeros as in standard LayerNorm
            if bias:
                self.beta = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('beta', None)
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        """
        Apply layer normalization with the reparameterized scale.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Compute mean and variance for normalization
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Normalize the input
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            # Apply the reparameterized scale (1 + gamma) and shift (beta)
            scale = 1.0 + self.gamma
            x_normalized = x_normalized * scale + self.beta
            
        return x_normalized


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
        self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
        self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
        self.s_scale = nn.Linear(dim_single_cond, dim)
        self.s_bias = nn.Linear(dim_single_cond, dim, bias=False)

    def forward(self, a, s):
        a = self.a_norm(a)
        s = self.s_norm(s)
        a = sigmoid(self.s_scale(s)) * a + self.s_bias(s)
        return a