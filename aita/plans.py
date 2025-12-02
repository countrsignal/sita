import dgl
import torch
from einops import repeat

from typing import Tuple, Dict
from abc import ABC, abstractmethod
from scipy.optimize import linear_sum_assignment

from .utils.data_utils import nm_to_angstrom
from .utils.couplings import ot_coupling, bacthed_kabsch_umeyama
from .utils.graph_utils import (
    scatter_center_mol,
    flatten_along_spatial,
    flatten_along_batch,
    nodes_to_padded_tensor,
)


###################################
# functions
###################################

def expand_t_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand the timestep tensor to match the dimensions of the input tensor."""
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


def ot_coupling(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Couple the initial and target structures using optimal transport theory."""
    C = torch.cdist(x, y)
    C = C**2
    C = C / C.max()
    C = C.numpy() # we assume x and y are on CPU
    row_ind, col_ind = linear_sum_assignment(C)
    return x[row_ind], y[col_ind]


###################################
# classes
###################################

class PlanLite(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def sample_times(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Plan(ABC):

    def __init__(self, coupling_plan: str = "ic"):
        # coupling_plan: Specifies how to align or couple the initial and target structures:
        #   - "ic": "independent coupling" (no matching between atoms, noise added independently),
        #   - "ot": "optimal transport" (matches atoms between structures using optimal transport theory),
        #   - "ku": "Kabsch-Umeyama" (aligns structures using the Kabsch-Umeyama algorithm for optimal rotation/translation).
        assert coupling_plan in ["ic", "ot", "ku"], f"Invalid coupling plan: {coupling_plan}. Valid plans: ic, ot, ku"
        self.coupling_plan = coupling_plan
    
    def compute_coupling(self, x: torch.Tensor, y: torch.Tensor, g: dgl.DGLGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Couple the initial and target structures using the specified coupling plan.
        Args:
            x: target coordinates with shape (batch_size, num_atoms, 3)
            y: gaussian noise with shape (batch_size, num_atoms, 3)
            g: DGLGraph of the molecule
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: coupled coordinates with shape (batch_size, num_atoms, 3)
        """
        if self.coupling_plan == "ot":
            x = flatten_along_spatial(x, g)
            y = flatten_along_spatial(y, g)
            x, y = ot_coupling(x, y)
            x = flatten_along_batch(x, g)
            y = flatten_along_batch(y, g)
            return x, y
        elif self.coupling_plan == "ku":
            x = nodes_to_padded_tensor(x, g)
            y = nodes_to_padded_tensor(y, g)
            mask = (
                torch.arange(g.batch_num_nodes().max())
                .unsqueeze(0)
                .lt(g.batch_num_nodes().unsqueeze(1))
            )
            y = bacthed_kabsch_umeyama(ref=x, pivot=y, mask=mask)
            x = flatten_along_batch(x.view(g.batch_size, g.batch_num_nodes().max() * 3), g)
            y = flatten_along_batch(y.view(g.batch_size, g.batch_num_nodes().max() * 3), g)
            return x, y
        else:
            return x, y

    @abstractmethod
    def __call__(self, g: dgl.DGLGraph) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_times(self, g: dgl.DGLGraph) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def compute_volatility(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_score_from_velocity(self, t: torch.Tensor, g: dgl.DGLGraph, velocity: torch.Tensor) -> torch.Tensor:
        pass


class TrigPlanLite(PlanLite):

    def __init__(self):
        super().__init__()
        self.alpha_t = lambda t: torch.sin(t * torch.pi / 2)
        self.sigma_t = lambda t: torch.cos(t * torch.pi / 2)
        self.d_alpha_t = lambda t: torch.pi / 2 * torch.cos(t * torch.pi / 2)
        self.d_sigma_t = lambda t: -torch.pi / 2 * torch.sin(t * torch.pi / 2)
        self.d_alpha_alpha_ratio_t = lambda t: torch.pi / (2 * torch.tan(t * torch.pi / 2))

    @torch.no_grad()
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # sample times
        t = self.sample_times(batch["samples"]) # (batch_size, num_atoms, 1)
        z = torch.randn_like(batch["samples"]) # (batch_size, num_atoms, 3)

        # remove center of mass from noise
        mask = ~batch["padding_mask"].unsqueeze(-1) #(b, n_atoms, 1)
        z = z * mask
        mean = z.sum(dim=1, keepdim=True) / (mask).sum(dim=1, keepdim=True)
        z = (z - mean) * mask

        # sample x(t)
        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)
        xt = alpha_t * batch["samples"] + sigma_t * z

        # package in batch
        batch["t"] = t
        batch["z"] = z
        batch["xt"] = xt
        batch["sigma_t"] = sigma_t
        return batch

    def sample_times(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms = x.size(0), x.size(1)
        t = torch.rand(batch_size, device=x.device)
        return repeat(t, "b -> b n 1", n=num_atoms)


class TrigPlan(Plan):

    def __init__(self, coupling_plan: str):
        super().__init__(coupling_plan)
        self.alpha_t = lambda t: torch.sin(t * torch.pi / 2)
        self.sigma_t = lambda t: torch.cos(t * torch.pi / 2)
        self.d_alpha_t = lambda t: torch.pi / 2 * torch.cos(t * torch.pi / 2)
        self.d_sigma_t = lambda t: -torch.pi / 2 * torch.sin(t * torch.pi / 2)
        self.d_alpha_alpha_ratio_t = lambda t: torch.pi / (2 * torch.tan(t * torch.pi / 2))

    ############################################################################################################################
    # methods for training time utils 
    ############################################################################################################################

    @torch.no_grad()
    def __call__(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        # sample times and noise
        x = g.ndata.pop("x1") # NOTE: this is the clean data point x1 (without noise)
        t = self.sample_times(g)
        t = expand_t_like(t, x)
        z = torch.randn_like(x)
        
        # remove center of mass
        z = scatter_center_mol(z, g)

        # compute coupling
        x, z = self.compute_coupling(x, z, g)
        
        # convert to angstroms
        # NOTE: data is in nanometers by default, so we convert to angstroms
        x = nm_to_angstrom(x)
        
        # sample x(t)
        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)
        xt = alpha_t * x + sigma_t * z
        
        # sample velocity(t, x)
        d_alpha_t = self.d_alpha_t(t)
        d_sigma_t = self.d_sigma_t(t)
        vt = d_alpha_t * x + d_sigma_t * z
        
        # package in DGLGraph
        g.ndata["t"]  = t
        g.ndata["z"]  = z
        g.ndata["x1"] = x
        g.ndata["xt"] = xt
        g.ndata["vt"] = vt
        g.ndata["sigma_t"] = sigma_t
        return g

    def sample_times(self, g: dgl.DGLGraph) -> torch.Tensor:
        t = torch.rand(g.batch_size, device=g.device)
        return t.repeat_interleave(g.batch_num_nodes())

    #############################################################################################################################
    # methods for test-time sampling utils
    #############################################################################################################################

    def compute_drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = expand_t_like(t, x)
        alpha_ratio = self.d_alpha_alpha_ratio_t(t)
        return alpha_ratio * x

    def compute_volatility(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = expand_t_like(t, x)
        sigma_t = self.sigma_t(t)
        d_sigma_t = self.d_sigma_t(t)
        alpha_ratio = self.d_alpha_alpha_ratio_t(t)
        return alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t

    def get_score_from_velocity(self, t: torch.Tensor, x: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            t: [batch_dim,] time tensor
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
        
        NOTE: this function blows up when t=1.0 !!!

        """
        # NOTE: at inference time t could be a tensor with no dimension
        if t.dim() == 0:
            t = t.view(1)

        t = expand_t_like(t, x)
        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)
        d_alpha_t = self.d_alpha_t(t)
        d_sigma_t = self.d_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score
    
    def compute_tsr(self, t: torch.Tensor, rho: float, k: float) -> torch.Tensor:
        half_pi = torch.pi / 2
        tan_t = torch.tan(t * half_pi) # TODO: clamp
        cot_t = 1 / tan_t
        snr = tan_t ** 2
        r2 = rho ** 2
        r2_over_k = r2 / k
        tsr = (snr * r2 + 1) / (snr * r2_over_k + 1)
        return tsr

    def temporal_score_rescale(
        self,
        k: float,
        rho: float,
        t: torch.Tensor,
        x: torch.Tensor,
        velocity: torch.Tensor,
    )  -> torch.Tensor:
        # NOTE: at inference time t could be a tensor with no dimension
        if t.dim() == 0:
            t = t.view(1)

        t = expand_t_like(t, x)

        half_pi = torch.pi / 2
        tan_t = torch.tan(t * half_pi) # TODO: clamp
        cot_t = 1 / torch.clamp_min(tan_t, min=1e-3)
        snr = tan_t ** 2

        r2 = rho ** 2
        r2_over_k = r2 / k
        tsr = (snr * r2 + 1) / (snr * r2_over_k + 1)

        return tsr * velocity + half_pi * cot_t * (1 - tsr) * x
