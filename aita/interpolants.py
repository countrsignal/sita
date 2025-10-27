import dgl
import torch
from torchdiffeq import odeint_adjoint

from functools import partial
from typing import Union, Tuple

from .plans import Plan, PlanLite
from .utils.logging import RankedLogger
from .utils.graph_utils import (
    scatter_center_mol,
    inference_graph_setup,
)


log = RankedLogger(__name__, on_rank_zero=True)


class Interpolant:

    def __init__(
        self,
        plan: Union[Plan, PlanLite],
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        self.plan = plan
        self.rtol = rtol
        self.atol = atol

    #############################################################################################################################
    # ODE mechanics
    #############################################################################################################################
    def ode_forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, g: dgl.DGLGraph, model: torch.nn.Module) -> torch.Tensor:
        # NOTE: this function is intended to called inside the torchdiffeq.odeint function
        # NOTE: instide the torchdiffeq.odeint function, x is a 2D tensor with shape (batch_size, num_nodes * 3)
        # NOTE: we assume the provided dgl graph already contains the categorical features
        g.ndata["xt"] = x.view(g.num_nodes(), 3) # [batch_size * num_nodes, 3]
        g.ndata["t"] = t * torch.ones((g.num_nodes(), 1), device=g.device) # [batch_size * num_nodes, 1]
        velocity = model(g)
        n_paticles = g.num_nodes() // g.batch_size # it is expected that we only generate conformers for one molecular species at a time
        return velocity.view(g.batch_size, n_paticles * 3)
    
    @torch.no_grad()
    def ode_integrate(
        self,
        batch_size: int,
        n_timesteps: int,
        categorical_features: torch.Tensor,
        model: torch.nn.Module,
        method: str = "dopri5",
    ) -> torch.Tensor:
        """
        Integrate the ODE to generate conformers for a given molecule defined by the categorical features.
        Args:
            batch_size: number of conformers of s single molecule to generate
            categorical_features: categorical features of the molecule with shape (num_atoms, num_features)
            model: model to use for velocity prediction
        Returns:
            torch.Tensor: integrated conformers
        """

        # model device
        device = next(model.parameters()).device

        # create a graph with the given categorical features
        # we should have a batch size of batch_size * num_nodes
        n_atoms = categorical_features.size(0)
        g = inference_graph_setup(n_atoms, batch_size, categorical_features)

        # Prepare state and time variables
        x_init = torch.randn((batch_size * n_atoms, 3))
        x_init = scatter_center_mol(x_init, g)
        x_init = x_init.view(batch_size, n_atoms * 3)
        time_span = torch.linspace(0.0, 1.0, n_timesteps + 1)

        # move data to device
        g = g.to(device)
        x_init = x_init.to(device)
        time_span = time_span.to(device)

        # create the forward function
        forward_fn = partial(self.ode_forward, g=g, model=model)

        # integrate the ODE
        xs = odeint_adjoint(forward_fn, x_init, time_span, method=method, rtol=self.rtol, atol=self.atol, adjoint_params=())
        return xs[-1].view(batch_size, n_atoms, 3)

    #############################################################################################################################
    # SDE mechanics
    #############################################################################################################################
    def sde_forward(self, t: Union[float, torch.Tensor], x: torch.Tensor, g: dgl.DGLGraph, model: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        velocity = self.ode_forward(t, x, g, model)
        score = self.plan.get_score_from_velocity(t, x, velocity)
        return velocity, score

    def _eurler_maruyama_step(self, dt: float, t: Union[float, torch.Tensor], x: torch.Tensor, g: dgl.DGLGraph, model: torch.nn.Module) -> torch.Tensor:
        diffusion_ceoff = self.plan.sigma_t(t).view(-1, 1)
        velocity, score = self.sde_forward(t, x, g, model)

        # Brownian motion
        w_cur = torch.randn_like(x).view(g.num_nodes(), 3)
        w_cur = scatter_center_mol(w_cur, g) # center noise at origin
        dw = w_cur * (dt ** 0.5)
        dw = dw.view(x.shape)

        # Euler-Maruyama update
        drift  = velocity + diffusion_ceoff * score
        mean_x = x + drift * dt
        return mean_x + torch.sqrt(diffusion_ceoff) * dw

    @torch.no_grad()
    def sde_integrate(
        self,
        batch_size: int,
        n_timesteps: int,
        categorical_features: torch.Tensor,
        model: torch.nn.Module,
        method: str = "em",
    ) -> torch.Tensor:
        """
        Integrate the SDE to generate conformers for a given molecule defined by the categorical features.
        Args:
            batch_size: number of conformers of s single molecule to generate
            categorical_features: categorical features of the molecule with shape (num_atoms, num_features)
            model: model to use for velocity prediction
        Returns:
            torch.Tensor: integrated conformers
        """
        
        if method not in ["em", "euler"]:
            raise NotImplementedError(f"The method {method} is not implemented for SDE integration.")
        
        # model device
        device = next(model.parameters()).device
        
        # create a graph with the given categorical features
        n_atoms = categorical_features.size(0)
        g = inference_graph_setup(n_atoms, batch_size, categorical_features)
        
        # Prepare state and time variables
        x_init = torch.randn((batch_size * n_atoms, 3))
        x_init = scatter_center_mol(x_init, g)
        x_init = x_init.view(batch_size, n_atoms * 3)
        time_span = torch.linspace(0.0, 1.0, n_timesteps + 1)[:-1] # NOTE: we exclude the final time step to avoid numerical instability
        dt = (time_span[1] - time_span[0]).abs().item() # NOTE: we convert to a scalar for convenience
        
        # move data to device
        g = g.to(device)
        x_init = x_init.to(device)
        time_span = time_span.to(device)
        
        # integrate the SDE
        x_t = x_init
        for t in time_span:
            x_t = self._eurler_maruyama_step(dt, t, x_t, g, model)
        # NOTE: the last step of the SDE is undefined due a singularity at t=1.0 in the interpolants
        #       so we perform an ode step at t=1.0 to get the final conformer
        x_t = x_t + self.ode_forward(1.0, x_t, g, model) * dt
        return x_t