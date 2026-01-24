import dgl
import torch
from torch import Tensor

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, Callable

from .plans import Plan
from .data.molecule import Molecule
from .utils.logging import RankedLogger
from .utils.graph_utils import scatter_center_mol
from .utils.data_utils import angstrom_to_nm, nm_to_angstrom
from .energies.base_molecule_energy_function import BaseMoleculeEnergy
from aita.data import molecule


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# functions
###################################

def mala_transition_kernel_step(
    step_size: float,
    x: Tensor,
    force_field: BaseMoleculeEnergy,
):
    # (a) Compute ∇ log π(x)
    log_prob_x, forces_x = force_field(x, return_force=True)
    # log_prob_x: (batch_size, n_particles * 3)
    # forces_x: (batch_size, n_particles * 3)

    # (b) Propose x' = x + (δ/2) * force  +  sqrt(δ) * ξ
    mean_fwd = x + 0.5 * step_size * forces_x

    # (c) Compute ξ ~ N(0, I)
    noise = torch.randn_like(x)
    # noise: (batch_size, n_particles * 3)

    # > remove center of mass from noise
    # NOTE: centered-noise constants cancel between forward/backward log ratios
    noise = noise.view(-1, force_field.n_particles, 3)
    noise = noise - noise.mean(dim=1, keepdim=True)
    noise = noise.view(-1, force_field.n_particles * 3)

    # (d) Compute forward proposal q(x' | x)
    proposal_x = mean_fwd + torch.sqrt(step_size) * noise
    log_q_fwd  = - 0.5 * torch.square(proposal_x - mean_fwd).sum(dim=-1) / step_size
    # proposal_x: (batch_size, n_particles * 3)
    # log_q_fwd: (batch_size, )

    # (e) Backward proposal q(x | x')
    log_prob_xp, forces_xp = force_field(proposal_x, return_force=True)
    mean_bwd  = proposal_x + 0.5 * step_size * forces_xp
    log_q_bwd = - 0.5 * torch.square(x.detach() - mean_bwd).sum(dim=-1) / step_size
    # log_q_bwd: (batch_size, )

    # (f) Compute acceptance probability
    log_alpha = log_prob_xp + log_q_bwd - log_prob_x - log_q_fwd
    log_alpha = log_alpha.clamp(max=0.0) # ensure α ≤ 1
    alpha     = torch.exp(log_alpha)
    alpha     = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

    u = torch.rand_like(alpha)                 # uniform [0,1] per batch‐element
    accept_mask = (u < alpha)                  # boolean mask [batch]

    # update x: if accepted, take x_proposed, else keep x
    x = torch.where(accept_mask.unsqueeze(-1), proposal_x, x)
    # x: (batch_size, n_particles * 3)

    return x


def transition_log_prob(
    x: Tensor,
    mean_x: Tensor,
    variance: Tensor,
    n_particles: int,
):
    quad = -0.5 * torch.square(x - mean_x).sum(dim=-1) / variance
    norm = -0.5 * 3 * (n_particles - 1) * torch.log(2 * torch.pi * variance)
    return quad + norm


def generative_transition_kernel_step(
    dt: float,
    t: Tensor,
    x: Tensor,
    g: dgl.DGLGraph,
    plan: Plan,
    model: torch.nn.Module,
    beta: float = 1.0,
    zeta: float = 1.0,
) -> Tuple[Tensor, Tensor]:

    assert zeta > 0.0, "zeta must be strictly positive or else numerical instability will occur"

    # > expand time variable to match the number of nodes for each molecule in the batch
    # NOTE: Expected that all molecules in the batch have the same number of atoms (i.e. same molecule for all graphs in the batch)
    # NOTE: it is expected that (t) has shape (batch_size, )
    n_particles = g.num_nodes() // g.batch_size
    t_per_node = t.repeat_interleave(n_particles) # [batch_size * n_particles, ]

    # > compute velocity
    g.ndata["xt"] = x.view(g.num_nodes(), 3) # [batch_size * n_particles, 3]
    g.ndata["t"] = t_per_node.view(-1, 1) # [batch_size * n_particles, 1]
    velocity = model(g)
    # velocity: (batch_size * n_particles, 3)
    # x: (batch_size, n_particles * 3)
    # t: (batch_size * n_particles, )

    # > compute score
    velocity = velocity.view(g.batch_size, n_particles * 3)
    score = plan.get_score_from_velocity(t, x, velocity)
    # velocity: (batch_size, n_particles * 3)
    # score: (batch_size, n_particles * 3)

    # > compute coefficients
    eta = beta + (1 - beta) * zeta
    diffusion_coeff = plan.sigma_t(t).view(-1, 1) ** 2
    xi = diffusion_coeff * (beta + (1 - beta) * 2.0 * zeta) / beta
    # diffusion_coeff: (batch_size, 1)
    # eta: (batch_size, 1)
    # xi: (batch_size, 1)

    # > brownian motion
    w_cur = torch.randn_like(x).view(g.num_nodes(), 3)
    w_cur = scatter_center_mol(w_cur, g) # NOTE: center noise at origin removes a degree of freedom!
    dw = w_cur * (dt ** 0.5)
    dw = dw.view(x.shape)
    # dw: (batch_size, n_particles * 3)

    # > compute forward mean
    fwd_drift  = velocity + 0.5 * eta * diffusion_coeff * score
    fwd_mean_x = x + fwd_drift * dt
    # fwd_mean_x: (batch_size, n_particles * 3)

    # > update x
    x_next = fwd_mean_x + torch.sqrt(xi) * dw
    # x_next: (batch_size, n_particles * 3)

    # > compute backward mean
    bwd_drift  = velocity - 0.5 * eta * diffusion_coeff * score
    bwd_mean_x = x_next + bwd_drift * dt
    # bwd_mean_x: (batch_size, n_particles * 3)

    # > compute variance
    variance = (xi * dt).squeeze(-1)
    # variance: (batch_size, )

    # > log probability (centroid-centered noise)
    fwd_log_prob = transition_log_prob(x_next, fwd_mean_x, variance, n_particles)
    bwd_log_prob = transition_log_prob(x, bwd_mean_x, variance, n_particles)
    # fwd_log_prob: (batch_size, )
    # bwd_log_prob: (batch_size, )

    return x_next, fwd_log_prob, bwd_log_prob


def noising_transition_kernel_step(
    dt: float,
    t: Tensor,
    x: Tensor,
    g: dgl.DGLGraph,
    plan: Plan,
    model: torch.nn.Module,
    beta: float = 1.0,
    zeta: float = 1.0,
) -> Tuple[Tensor, Tensor]:

    assert zeta > 0.0, "zeta must be strictly positive or else numerical instability will occur"

    # > expand time variable to match the number of nodes for each molecule in the batch
    # NOTE: Expected that all molecules in the batch have the same number of atoms (i.e. same molecule for all graphs in the batch)
    # NOTE: it is expected that (t) has shape (batch_size, )
    n_particles = g.num_nodes() // g.batch_size
    t_per_node = t.repeat_interleave(n_particles) # [batch_size * n_particles, ]

    ###############################################
    # Noising process direction
    ###############################################
    # > compute velocity
    g.ndata["xt"] = x.view(g.num_nodes(), 3) # [batch_size * n_particles, 3]
    g.ndata["t"] = t_per_node.view(-1, 1) # [batch_size * n_particles, 1]
    velocity = model(g)
    # velocity: (batch_size * n_particles, 3)
    # x: (batch_size, n_particles * 3)
    # t: (batch_size * n_particles, )

    # > compute score
    velocity = velocity.view(g.batch_size, n_particles * 3)
    score = plan.get_score_from_velocity(t_per_node, x, velocity)
    # velocity: (batch_size, n_particles * 3)
    # score: (batch_size, n_particles * 3)

    # > compute coefficients
    eta = beta + (1 - beta) * zeta
    diffusion_coeff = plan.sigma_t(t_per_node).view(-1, 1) ** 2
    xi = diffusion_coeff * (beta + (1 - beta) * 2.0 * zeta) / beta
    # diffusion_coeff: (batch_size, 1)
    # eta: (batch_size, 1)
    # xi: (batch_size, 1)

    # > brownian motion
    w_cur = torch.randn_like(x).view(g.num_nodes(), 3)
    w_cur = scatter_center_mol(w_cur, g) # NOTE: center noise at origin removes a degree of freedom!
    dw = w_cur * (dt ** 0.5)
    dw = dw.view(x.shape)
    # dw: (batch_size, n_particles * 3)

    # > compute backward mean
    bwd_drift  = velocity - 0.5 * eta * diffusion_coeff * score
    bwd_mean_x = x + bwd_drift * dt
    # bwd_mean_x: (batch_size, n_particles * 3)

    ###############################################
    # Proposal step
    ###############################################
    # > update x
    x_prev = bwd_mean_x + torch.sqrt(xi) * dw
    # x_prev: (batch_size, n_particles * 3)

    ###############################################
    # Generative process direction
    ###############################################
    # > compute velocity
    g.ndata["xt"] = x_prev.view(g.num_nodes(), 3) # [batch_size * n_particles, 3]
    velocity = model(g)
    # velocity: (batch_size * n_particles, 3)
    # x_prev: (batch_size, n_particles * 3)

    # > compute score
    velocity = velocity.view(g.batch_size, n_particles * 3)
    score = plan.get_score_from_velocity(t_per_node, x_prev, velocity)
    # velocity: (batch_size, n_particles * 3)
    # score: (batch_size, n_particles * 3)

    # > compute forward mean
    fwd_drift  = velocity + 0.5 * eta * diffusion_coeff * score
    fwd_mean_x = x_prev + fwd_drift * dt
    # fwd_mean_x: (batch_size, n_particles * 3)

    ###############################################
    # Log Probabilities
    ###############################################
    # > compute variance
    variance = (xi * dt).squeeze(-1)
    # variance: (batch_size, )

    # > log probability (centroid-centered noise)
    fwd_log_prob = transition_log_prob(x, fwd_mean_x, variance, n_particles)
    bwd_log_prob = transition_log_prob(x_prev, bwd_mean_x, variance, n_particles)
    # fwd_log_prob: (batch_size, )
    # bwd_log_prob: (batch_size, )

    return x_prev, fwd_log_prob, bwd_log_prob


def accept_reject(log_alpha: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    # Ensure α ≤ 1 and map NaNs to probability zero
    log_alpha = torch.nan_to_num(log_alpha, nan=-torch.inf)
    log_alpha = log_alpha.clamp(max=0.0)

    # Log acceptance probability for reporting (avoid exp under/overflow)
    log_accept_prob = log_alpha.detach()
    log_eps = torch.log(torch.tensor(1e-6, device=log_alpha.device, dtype=log_alpha.dtype))
    log_acceptance_rate = torch.logaddexp(log_accept_prob, log_eps).mean()

    # Accept proposals using a log-space comparison: log u < log α
    tiny = torch.finfo(log_alpha.dtype).tiny
    log_uniform = torch.log(torch.rand_like(log_alpha) + tiny)
    accept_mask = log_uniform < log_alpha
    return log_acceptance_rate, accept_mask

###################################
# Classes
###################################

@dataclass
class Context:
    mol: Molecule
    chains: Tensor
    states: Tensor = field(init=False, default_factory=lambda: torch.empty([]))
    n_chains: int = field(init=False, default_factory=lambda: 0)

    def __post_init__(self):
        self.n_chains = self.chains.size(0)
        self.states = torch.zeros((self.chains.size(0),), dtype=torch.float32, device=self.chains.device)

    def __len__(self):
        return self.n_chains

    def dispatch(self, device: torch.device) -> None:
        self.chains = self.chains.to(device)
        self.states = self.states.to(device)

    def adjust_acceptance_mask(self, acceptance_mask: Tensor) -> Tensor:
        # Change the incoming acceptance mask so that
        # chains that have already been accepted are set to FALSE
        adjusted_acceptance_mask = acceptance_mask.clone()
        adjusted_acceptance_mask[self.states] = False
        return adjusted_acceptance_mask
    
    def update_chains(self, adjusted_acceptance_mask: Tensor, proposals: Tensor) -> None:
        self.chains = torch.where(adjusted_acceptance_mask.unsqueeze(-1), proposals, self.chains)
    
    def update_states(self, adjusted_acceptance_mask: Tensor) -> None:
        # if a state is currently True, it must remain true regardless of the acceptance mask being TRUE or FALSE for that chain
        # if a state is currently False, it must be set to True if the acceptance mask is TRUE for that chain
        self.states = torch.where(adjusted_acceptance_mask & ~self.states, True, self.states)


class NonequilibriumCandidateMonteCarlo:

    def __init__(
        self,
        n_mala_steps: int,
        n_ncmc_steps: int,
        zeta: float = 1.0,
        t_bounds: Tuple[float, float] = (0.0, 0.5),
        temp_levels: Tuple[float, float] = (755.55, 1200.0),
        mala_step_size: float = 0.01,
    ) -> None:

        self.n_mala_steps = n_mala_steps
        self.n_ncmc_steps = n_ncmc_steps
        self.zeta = zeta
        self.mala_step_size = mala_step_size
        self.t_low, self.t_high = t_bounds
        self.temp_low, self.temp_high = temp_levels
        self._context = None

    def init_context(self, mol: Molecule, chains: Tensor) -> None:
        self._context = Context(mol, chains)
    
    def sample_times(self) -> Tensor:
        return torch.rand((len(self._context),)) * (self.t_high - self.t_low) + self.t_low
    
    def run(
        self,
        plan: Plan,
        model: torch.nn.Module,
        force_field_partial: Callable[[Tensor], Tensor],
    ) -> None:

        # ensure that the context is initialized
        assert self._context is not None, "Context must be initialized before running NCMC"
        
        # model device
        device = next(model.parameters()).device
        
        # system variables
        dt = 1 / self.n_ncmc_steps
        beta = self.temp_high / self.temp_low
        n_atoms = self._context.mol.n_atoms
        n_chains = len(self._context)

        # create a graph with the given categorical features
        graphs = self._context.mol.inference_graph_setup(len(self._context))

        # initialize force fields at the two temperature levels
        force_field_high = force_field_partial(temperature=self.temp_high)
        force_field_low = force_field_partial(temperature=self.temp_low)

        # being loop
        graphs = graphs.to(device)
        self._context.dispatch(device)
        log.info(f"Running NCMC from {self.temp_high}K to {self.temp_low}K ...")
        while torch.all(~self._context.states):

            # NOTE: chains with state that are set to FALSE are at the high temperature
            #       and chains with state that are set to TRUE are at the low temperature

            #########################################################################################################################
            # HIGH TEMPERATURE MALA
            #########################################################################################################################
            if torch.any(~self._context.states):
                # run local MALA steps at the high temperature
                # > find the high temperature chains and run MALA steps
                hot_chains = self._context.chains[~self._context.states]
                for _ in tqdm(range(self.n_mala_steps), desc=f"High Temperature ({self.temp_high}K) MALA Steps"):
                    hot_chains = mala_transition_kernel_step(
                        step_size=self.mala_step_size,
                        x=hot_chains,
                        force_field=force_field_high,
                    )
                self._context.update_chains(~self._context.states, hot_chains)
            
            #########################################################################################################################
            # Non-Equilibrium Switches
            #########################################################################################################################

            # (1) Sample the protocol times
            protocol_times = self.sample_times()
            # protocol_times: (n_chains, )

            # (2) Sample the protocol noise
            n_hot_chains = hot_chains.size(0)
            z = torch.randn((n_hot_chains * n_atoms, 3), device=device)
            z = scatter_center_mol(z, graphs).view(n_hot_chains, n_atoms * 3)
            # z: (n_chains, n_atoms * 3)

            # (3) draw x(t)
            alpha_t = plan.alpha_t(protocol_times).view(-1, 1)
            sigma_t = plan.sigma_t(protocol_times).view(-1, 1)
            x_t = alpha_t * hot_chains + sigma_t * z
            # x_t: (n_chains, n_atoms * 3)


            #########################################################################################################################
            # LOW TEMPERATURE MALA
            #########################################################################################################################
            if torch.any(self._context.states):
                # run local MALA steps at the low temperature
                # > find the low temperature chains and run MALA steps
                low_temperature_chains = self._context.chains[self._context.states]
                for _ in tqdm(range(self.n_mala_steps), desc=f"Low Temperature ({self.temp_low}K) MALA Steps"):
                    low_temperature_chains = mala_transition_kernel_step(
                        step_size=self.mala_step_size,
                        x=low_temperature_chains,
                        force_field=force_field_low,
                    )
                self._context.update_chains(self._context.states, low_temperature_chains)