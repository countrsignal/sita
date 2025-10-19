import dgl
import torch
from typing import List, Callable


class GeometryDistortion:

    def __init__(self, t_distort: float = 0.5, p_distort: float = 0.2, sigma_distort: float = 0.5):
        self.t_distort = t_distort
        self.p_distort = p_distort
        self.sigma_distort = sigma_distort

    def __call__(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """Distort the geometry of the molecule.
        Args:
            g: DGLGraph of the molecule
        Returns:
            DGLGraph of the molecule with distorted coordinates
        """
        t = g.ndata["t"]
        xt = g.ndata.pop("xt")
        # only apply distortion is time is greater than or equal to the distortion threshold
        t_mask  = t >= self.t_distort
        # NOTE: this returns a binary mask of the same shape as t, i.e. (batch_size * num_atoms, 1)
        #       this will have the effect of zeroing out entire channels
        x_mask  = torch.bernoulli(t, p=self.p_distort)
        # random noise with shape (batch_size * num_atoms, 3)
        distort = torch.randn_like(xt) * self.sigma_distort
        g.ndata["xt"] = xt + t_mask * x_mask * distort
        return g


class Pipeline:

    def __init__(self, data_augs: List[Callable[[dgl.DGLGraph], dgl.DGLGraph]]):
        self.data_augs = data_augs

    @torch.no_grad()
    def __call__(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        for aug in self.data_augs:
            g = aug(g)
        return g

    def __len__(self) -> int:
        return len(self.data_augs)

    def __getitem__(self, idx: int) -> Callable[[dgl.DGLGraph], dgl.DGLGraph]:
        return self.data_augs[idx]