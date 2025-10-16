import dgl
import torch
from torchdiffeq import odeint_adjoint

from typing import Union
from functools import partial
from abc import ABC, abstractmethod

from .utils.logging import RankedLogger
from .utils.graph_utils import fully_connected_edges, get_batch_indices, scatter_center_mol


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# functions
###################################

def expand_t_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand the timestep tensor to match the dimensions of the input tensor."""
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


###################################
# classes
###################################

class Plan(ABC):

    def __init__(self):
        pass

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


class TrigPlan(Plan):

    def __init__(self):
        super().__init__()
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
        t = self.sample_times(g)
        t = expand_t_like(t, g.ndata["x"])
        z = torch.randn_like(g.ndata["x"])
        
        # remove center of mass
        batch_index = get_batch_indices(g)
        z = scatter_center_mol(z, batch_index)
        
        # sample x(t)
        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)
        xt = alpha_t * g.ndata["x"] + sigma_t * z
        
        # sample velocity(t, x)
        d_alpha_t = self.d_alpha_t(t)
        d_sigma_t = self.d_sigma_t(t)
        vt = d_alpha_t * g.ndata["x"] + d_sigma_t * z
        
        # package in DGLGraph
        g.ndata["t"]  = t
        g.ndata["z"]  = z
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

    def get_score_from_velocity(self, t: torch.Tensor, g: dgl.DGLGraph, velocity: torch.Tensor) -> torch.Tensor:
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            t: [batch_dim,] time tensor
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
        
        NOTE: this function blows up when t=1.0 !!!

        """
        t = expand_t_like(t, g.ndata["x"])
        alpha_t = self.alpha_t(t)
        sigma_t = self.sigma_t(t)
        d_alpha_t = self.d_alpha_t(t)
        d_sigma_t = self.d_sigma_t(t)
        mean = g.ndata["x"]
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score


class Interpolant:

    def __init__(
        self,
        plan: Plan = TrigPlan(),
        integrator: str = "ode-dopri5",
        n_timesteps: int = 100,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        VALID_INTEGRATORS = ["ode-dopri5", "ode-euler", "sde-em"]
        assert integrator in VALID_INTEGRATORS, f"Invalid integrator: {integrator}.Valid integrators: {', '.join(VALID_INTEGRATORS)}"
        self.plan = plan
        self.rtol = rtol
        self.atol = atol
        self.n_timesteps = n_timesteps
        int_type, method = integrator.split("-")
        self.int_type = int_type
        self.method = method

    def ode_forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, g: dgl.DGLGraph, model: torch.nn.Module) -> torch.Tensor:
        # NOTE: this function is intended to called inside the torchdiffeq.odeint function
        # NOTE: instide the torchdiffeq.odeint function, x is a 2D tensor with shape (batch_size, num_nodes * 3)
        # NOTE: we assume the provided dgl graph already contains the categorical features
        g.ndata["x"] = x.view(g.num_nodes(), 3) # [batch_size * num_nodes, 3]
        g.ndata["t"] = t * torch.ones(g.num_nodes(), device=g.device)
        velocity = model(g)
        n_paticles = g.num_nodes() // g.batch_size # it is expected that we only generate conformers for one molecular species at a time
        return velocity.view(g.batch_size, n_paticles * 3)
    
    @torch.no_grad()
    def ode_integrate(self, batch_size: int, categorical_features: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """
        Integrate the ODE to generate conformers for a given molecule defined by the categorical features.
        Args:
            batch_size: number of conformers of s single molecule to generate
            categorical_features: categorical features of the molecule with shape (num_atoms, num_features)
            model: model to use for velocity prediction
        Returns:
            torch.Tensor: integrated conformers
        """
        assert self.int_type == "ode", f"The integrator type must be 'ode' for this method. Got {self.int_type}."
        # model device
        device = next(model.parameters()).device

        # create a graph with the given categorical features
        # we should have a batch size of batch_size * num_nodes
        n_atoms = categorical_features.size(0)
        src, dst = fully_connected_edges(n_atoms)             # edges for one molecule
        per_graph = src.numel()

        offset = torch.arange(batch_size) * n_atoms
        src = src.repeat(batch_size) + offset.repeat_interleave(per_graph)
        dst = dst.repeat(batch_size) + offset.repeat_interleave(per_graph)
        g = dgl.graph((src, dst), num_nodes=batch_size * n_atoms)

        # NOTE: dgl.graph always creates a single-graph object,
        # so batch_size defaults to 1 unless you tell DGL how many graphs you batched together
        g.set_batch_num_nodes(torch.full((batch_size,), n_atoms, dtype=torch.int64))
        g.set_batch_num_edges(torch.full((batch_size,), per_graph, dtype=torch.int64))
        g.ndata["h"] = categorical_features.repeat(batch_size, 1)
        batch_index = get_batch_indices(g)

        # Move data to device
        g = g.to(device)

        # Prepare the ODE integrator
        x_init = torch.randn((batch_size * n_atoms, 3))
        x_init = scatter_center_mol(x_init, batch_index)
        x_init = x_init.view(batch_size, n_atoms * 3).to(device)
        time_span = torch.linspace(0.0, 1.0, self.n_timesteps + 1, device=device)
        forward_fn = partial(self.ode_forward, g=g, model=model)

        # integrate the ODE
        xs = odeint_adjoint(forward_fn, x_init, time_span, method=self.method, rtol=self.rtol, atol=self.atol, adjoint_params=())
        return xs[-1].view(batch_size, n_atoms, 3)

        
