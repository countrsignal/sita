from torch import nn
import torch.nn.functional as F

from functools import partial

from .layer_norms import SafeLayerNorm


sdpa = F.scaled_dot_product_attention
LinearNoBias = partial(nn.Linear, bias=False)
LayerNormEps = partial(SafeLayerNorm, eps=1e-5)
