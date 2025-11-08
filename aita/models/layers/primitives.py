from torch import nn
from functools import partial


LinearNoBias = partial(nn.Linear, bias=False)
LayerNormEps = partial(nn.LayerNorm, eps=1e-5)