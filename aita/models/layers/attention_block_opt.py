"""Compiled AttentionBlock for reduced kernel launch overhead.

Wraps AttentionBlock.forward with torch.compile to fuse pointwise ops
(LayerNorm weight reparameterization, masking, reshapes, SwiGLU
activation) between the large matmul and SDPA kernels.

State-dict keys are identical to the base AttentionBlock, so checkpoints
are interchangeable.
"""

import torch
from .attention_block import AttentionBlock


class CompiledAttentionBlock(AttentionBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward = torch.compile(self.forward)
