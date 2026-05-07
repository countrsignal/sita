import dgl
import torch
from torch import Tensor
from torch.distributions import constraints

import gc
import math
from tqdm import tqdm
from typing import Dict, Union, Tuple, Optional, Callable, List

from .plans import Plan
from .data.molecule import Molecule
from .data.datasets import NCMCSingleMoleculeDataset
from .utils.logging import RankedLogger
from .utils.graph_utils import scatter_center_mol
from .utils.data_utils import angstrom_to_nm, nm_to_angstrom
from .energies.base_molecule_energy_function import BaseMoleculeEnergy


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# functions
###################################

def transition_log_prob(
    x_next: Tensor,
    mean_x: Tensor,
    variance: Tensor,
    n_particles: int,
):
    quad = -0.5 * torch.square(x_next - mean_x).sum(dim=-1) / variance
    norm = -0.5 * 3 * (n_particles - 1) * torch.log(2 * torch.pi * variance)
    return quad + norm


@torch.no_grad()
def simulate_ou_forward_reverse(
    z_init: torch.Tensor,
    n_steps: int,
    dt: float,
    cold_prior: torch.distributions.Distribution,
    device: str = "cpu",
):
    """
    Simulate forward OU process from N(0, I) toward N(0, σ²I),
    accumulating forward and reverse path log-probabilities.

    Path measures:
        log_p_fwd = log p_0(z_0) + Σ log p(z_{n+1} | z_n)
        log_p_bwd = log π(z_T)  + Σ log p(z_n | z_{n+1})

    where π is the cold prior (target).

    Returns:
        z_T:       [n_samples, dim]
        log_p_fwd: [n_samples]
        log_p_bwd: [n_samples]
    """
    sigma2 = cold_prior.scale ** 2
    alpha = math.exp(-dt / sigma2)
    tau2 = sigma2 * (1.0 - alpha ** 2)
    dof = (cold_prior.n_particles - 1) * cold_prior.spatial_dim
    dim = cold_prior.dim
    n_samples = z_init.size(0)

    # ---- Enforce z_0 ~ N(0, I) on mean-free subspace ----
    z = cold_prior.mean_free(z_init)

    # log p_0(z_0): mean-free Gaussian, unit variance, dof dimensions
    log_p_fwd = -0.5 * dof * math.log(2 * math.pi) - 0.5 * z.pow(2).sum(-1)
    log_p_bwd = torch.zeros(n_samples, device=device)

    alpha_n = 1.0  # tracks α^n = e^{-n Δt / σ²}

    for _ in range(n_steps):
        v_n = alpha_n ** 2 * (1.0 - sigma2) + sigma2

        # --- Forward step (exact OU transition) ---
        eps = torch.randn_like(z)
        eps = cold_prior.mean_free(eps)
        z_new = alpha * z + math.sqrt(tau2) * eps
        z_new = cold_prior.mean_free(z_new)

        # Update schedule
        alpha_n *= alpha
        v_n1 = alpha_n ** 2 * (1.0 - sigma2) + sigma2

        # --- Accumulate forward log p(z_{n+1} | z_n) ---
        res_fwd = z_new - alpha * z
        log_p_fwd += (
            -0.5 * dof * math.log(2 * math.pi * tau2)
            - 0.5 * res_fwd.pow(2).sum(-1) / tau2
        )

        # --- Accumulate reverse log p(z_n | z_{n+1}) ---
        rev_var = tau2 * v_n / v_n1
        rev_coeff = alpha * v_n / v_n1
        res_bwd = z - rev_coeff * z_new
        log_p_bwd += (
            -0.5 * dof * math.log(2 * math.pi * rev_var)
            - 0.5 * res_bwd.pow(2).sum(-1) / rev_var
        )

        z = z_new

    # ---- Terminal: log π(z_T) under cold prior ----
    log_p_bwd += cold_prior.log_prob(z)

    return z, log_p_fwd, log_p_bwd


@torch.no_grad()
def sde_integrate_heun(
    x_init: torch.Tensor,
    times: torch.Tensor,
    graph: dgl.DGLGraph,
    vector_field: torch.nn.Module,
    plan: Plan,
    simulate_generative: bool = True,
    beta: float = 1.0,
    zeta: float = 1.0,
):
    """
    SDE integrator with Heun predictor-corrector, trapezoidal backward
    mean, and midpoint variance for tighter forward/backward log-probability
    agreement and improved NCMC acceptance rates.

    Three improvements over the basic Euler-Maruyama integrator:

    1. **Heun predictor-corrector** -- The forward drift is the trapezoidal
       average  0.5*(f_curr + f_next)*dt  instead of the Euler  f_curr*dt.
       The predictor is a standard EM step; the model is then evaluated at the
       predicted point to obtain f_next; the corrector reuses the *same noise*
       with the averaged drift.  Cost: still one model evaluation per step
       (at the predicted point; the departure-point evaluation is carried
       forward from the previous step).

    2. **Trapezoidal backward mean** -- The backward (virtual reverse) mean
       uses  0.5*(g_curr + g_next)*dt  where g_curr and g_next are the
       backward drifts evaluated at departure and arrival, respectively.
       This is free: both sets of (v, s) are already available.

    3. **Midpoint variance** -- A shared  xi_mid = (xi_curr + xi_next) / 2
       is used for both forward and backward log-probs.  Because the same
       variance appears in both Gaussian kernels, the normalisation constants
       cancel *exactly* in the per-step Jarzynski weight, eliminating any
       log-determinant mismatch.

    Together these remove the  sign*(v_{i+1} - v_i)  velocity-difference term
    from the per-step work and symmetrise the evaluation, leaving only the
    irreducible  xi*s  residual (from f_fwd + f_bwd = xi*s) in the backward
    quadratic form.

    Args:
        x_init:       Initial samples, shape (B, n_particles * 3).
        times:        1-D time grid.  Ascending for generative, descending
                      for noising.
        graph:        DGL graph with molecular features.
        vector_field: Trained velocity-field model.
        plan:         Stochastic-interpolant plan.
        forward:      True = generative (noise->data),
                      False = noising   (data->noise).
        beta:         Inverse-temperature ratio  (default 1.0).
        zeta:         Diffusion annealing parameter (must be > 0, default 1.0).

    Returns:
        x:       Final samples  (B, D),  CPU, float32.
        log_fwd: Simulated-path log-prob  (B,),  CPU, float64.
        log_bwd: Virtual-reverse log-prob (B,),  CPU, float64.
    """

    # ── Validate ─────────────────────────────────────────────────────
    assert zeta > 0.0, "zeta must be > 0 for numerical stability"

    # ── Integration constants ────────────────────────────────────────
    x        = x_init.clone()
    dt       = (times[1] - times[0]).abs().item()
    eta      = beta + (1 - beta) * zeta
    sign     = 1.0 if simulate_generative else -1.0
    xi_scale = (beta + (1 - beta) * 2.0 * zeta) / beta   # xi = sigma^2 * xi_scale

    t_eps = max(1e-4, 0.5 * dt)
    times = times.clamp(t_eps, 1.0 - t_eps)

    # ── Batch / graph dimensions ─────────────────────────────────────
    batch_size  = graph.batch_size
    n_particles = graph.num_nodes() // batch_size
    device      = next(vector_field.parameters()).device

    log_fwd = torch.zeros(batch_size, device=device, dtype=torch.float64)
    log_bwd = torch.zeros(batch_size, device=device, dtype=torch.float64)

    # ── Move data to device ──────────────────────────────────────────
    graph.ndata["xt"] = x_init.view(graph.num_nodes(), 3)
    graph.ndata["t"]  = times[0].expand(batch_size * n_particles).view(-1, 1)
    graph = graph.to(device)
    times = times.to(device)
    x     = x.to(device)

    # ── Initial velocity & score ─────────────────────────────────────
    v = vector_field(graph).view(batch_size, n_particles * 3)
    s = plan.get_score_from_velocity(times[0], x, v)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Main integration loop ────────────────────────────────────────
    for i in range(len(times) - 1):

        t_curr = times[i]
        t_next = times[i + 1]

        # ── Diffusion coefficients at BOTH endpoints ─────────────────
        sigma2_curr = plan.sigma_t(t_curr).view(-1, 1) ** 2
        sigma2_next = plan.sigma_t(t_next).view(-1, 1) ** 2
        xi_curr     = sigma2_curr * xi_scale
        xi_next     = sigma2_next * xi_scale

        # Midpoint variance (shared by forward & backward kernels)
        xi_mid = 0.5 * (xi_curr + xi_next)

        # Backward score coefficients at each endpoint
        #   bwd_coeff = xi - 0.5 * eta * sigma^2
        bwd_coeff_curr = xi_curr - 0.5 * eta * sigma2_curr
        bwd_coeff_next = xi_next - 0.5 * eta * sigma2_next

        # ── Centroid-centred Brownian increment (sampled ONCE) ────────
        w = torch.randn_like(x).view(graph.num_nodes(), 3)
        w = scatter_center_mol(w, graph)
        dw = (w * dt**0.5).view(x.shape)

        # ── (1) Forward drift at DEPARTURE ───────────────────────────
        f_curr = sign * v + 0.5 * eta * sigma2_curr * s

        # ── (2) EM predictor (uses midpoint noise) ───────────────────
        x_pred = x + f_curr * dt + torch.sqrt(xi_mid) * dw

        # ── (3) Model evaluation at PREDICTED arrival ────────────────
        graph.ndata["xt"] = x_pred.view(graph.num_nodes(), 3)
        graph.ndata["t"]  = t_next.expand(batch_size * n_particles).view(-1, 1)

        v_new = vector_field(graph).view(batch_size, n_particles * 3)
        s_new = plan.get_score_from_velocity(t_next, x_pred, v_new)
        s_new = torch.nan_to_num(s_new, nan=0.0, posinf=0.0, neginf=0.0)

        # ── (4) Forward drift at ARRIVAL (predicted point) ───────────
        f_next = sign * v_new + 0.5 * eta * sigma2_next * s_new

        # ── (5) Heun corrector: trapezoidal forward mean ─────────────
        mu_fwd = x + 0.5 * (f_curr + f_next) * dt
        x_new  = mu_fwd + torch.sqrt(xi_mid) * dw        # SAME noise

        # ── (6) Forward log-prob (midpoint variance) ─────────────────
        variance = (xi_mid * dt).squeeze(-1)
        log_fwd += transition_log_prob(
            x_new, mu_fwd, variance, n_particles
        ).to(torch.float64)

        # ── (7) Backward drifts at BOTH endpoints ────────────────────
        g_curr = -sign * v     + bwd_coeff_curr * s
        g_next = -sign * v_new + bwd_coeff_next * s_new

        # ── (8) Trapezoidal backward mean ────────────────────────────
        mu_bwd = x_new + 0.5 * (g_curr + g_next) * dt

        # ── (9) Backward log-prob (SAME midpoint variance) ───────────
        log_bwd += transition_log_prob(
            x, mu_bwd, variance, n_particles
        ).to(torch.float64)

        # Advance state
        x, v, s = x_new, v_new, s_new

    return x.cpu(), log_fwd.cpu(), log_bwd.cpu()


@torch.no_grad()
def ncmc_accept_reject(
    x_current: Tensor,
    x_proposed: Tensor,
    log_pi_current: Tensor,
    log_pi_proposed: Tensor,
    log_fwd_noise: Tensor,
    log_bwd_noise: Tensor,
    log_fwd_latents: Tensor,
    log_bwd_latents: Tensor,
    log_fwd_gen: Tensor,
    log_bwd_gen: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    NCMC Metropolis-Hastings accept/reject for the three-leg proposal:

        x --(noise)--> z0 --(OU bridge)--> z_k --(generate)--> x'

    The log acceptance ratio is:

        log w = [log pi(x') - log pi(x)]
              + [log_bwd_noise + log_bwd_latents + log_bwd_gen]
              - [log_fwd_noise + log_fwd_latents + log_fwd_gen]

    Args:
        x_current:       Current samples,                   shape (B, D).
        x_proposed:      Proposed samples,                  shape (B, D).
        log_pi_current:  log pi(x) target density,          shape (B,).
        log_pi_proposed: log pi(x') target density,         shape (B,).
        log_fwd_noise:   Forward path measure (noising),    shape (B,).
        log_bwd_noise:   Backward path measure (noising),   shape (B,).
        log_fwd_latents: Forward path measure (OU bridge),  shape (B,).
        log_bwd_latents: Backward path measure (OU bridge), shape (B,).
        log_fwd_gen:     Forward path measure (generative), shape (B,).
        log_bwd_gen:     Backward path measure (generative),shape (B,).

    Returns:
        x_out:  Accepted samples, shape (B, D).
        accept: Boolean acceptance mask, shape (B,).
        log_w:  Log acceptance ratios, shape (B,).
    """

    # Total forward path log-probability: x -> z0 -> z_k -> x'
    log_fwd = log_fwd_noise + log_fwd_latents + log_fwd_gen

    # Total backward path log-probability: x' -> z_k -> z0 -> x
    log_bwd = log_bwd_noise + log_bwd_latents + log_bwd_gen

    # NCMC log acceptance ratio (Metropolis-Hastings)
    log_alpha = (log_pi_proposed - log_pi_current) + (log_bwd - log_fwd)

    # ── Sanitise ─────────────────────────────────────────────────────
    # Map any NaN / +inf to rejection  (log_alpha = -inf => alpha = 0)
    log_alpha = torch.nan_to_num(log_alpha, nan=-torch.inf, posinf=0.0, neginf=-torch.inf)
    # Clamp: alpha <= 1  =>  log_alpha <= 0
    log_alpha = log_alpha.clamp(max=0.0)

    # ── Accept / reject in log space ─────────────────────────────────
    # accept iff  log(u) < log(alpha),   u ~ Uniform(0, 1)
    tiny      = torch.finfo(log_alpha.dtype).tiny          # avoid log(0)
    log_u     = torch.log(torch.rand(log_alpha.shape, dtype=log_alpha.dtype,
                                     device=log_alpha.device) + tiny)
    accept_mask = log_u < log_alpha                        # (B,)  bool

    return accept_mask, log_alpha.to(torch.float32)


###################################
# Classes
###################################


class MeanFreePrior(torch.distributions.Distribution):
    arg_constraints: Dict[str, constraints.Constraint] = {}

    def __init__(self, n_particles, spatial_dim, scale, device="cpu"):
        super().__init__()
        self.n_particles = n_particles
        self.spatial_dim = spatial_dim
        self.dim = n_particles * spatial_dim
        self.scale = scale
        self.device = device

    def log_prob(self, x):
        x = x.reshape(-1, self.n_particles, self.spatial_dim)
        N, D = x.shape[-2:]

        # r is invariant to a basis change in the relevant hyperplane.
        r2 = torch.sum(x**2, dim=(-1, -2)) / self.scale**2

        # The relevant hyperplane is (N-1) * D dimensional.
        degrees_of_freedom = (N - 1) * D

        # Normalizing constant and logpx are computed:
        log_normalizing_constant = (
            -0.5 * degrees_of_freedom * math.log(2 * torch.pi * self.scale**2)
        )
        log_px = -0.5 * r2 + log_normalizing_constant
        return log_px

    def sample(self, n_samples):
        if isinstance(n_samples, int):
            n_samples = torch.Size([n_samples])
        samples = torch.randn(*n_samples, self.dim, device=self.device) * self.scale
        samples = samples.reshape(-1, self.n_particles, self.spatial_dim)
        samples = samples - samples.mean(-2, keepdims=True)
        return samples.reshape(-1, self.n_particles * self.spatial_dim)

    def score(self, x):
        """Analytical score function: ∇_x log p(x) = -x / scale^2"""
        return -x / self.scale**2

    def mean_free(self, x):
        x = x.reshape(-1, self.n_particles, self.spatial_dim)
        x = x - x.mean(dim=1, keepdim=True)
        x = x.reshape(-1, self.n_particles * self.spatial_dim)
        return x


class NonequilibriumCandidateMonteCarlo:

    def __init__(
        self,
        n_samples: int,
        batch_size: int,
        n_timesteps: int,
        temperature_ladder: List[List[float]],
    ):
        # bookkeeping
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        self.temperature_ladder = temperature_ladder

        # track ladder index
        self._temperature_index = 0
        
    def run(
        self,
        plan: Plan,
        model: torch.nn.Module,
        dataset: NCMCSingleMoleculeDataset,
        forcefield_partial: Callable,
    ):
        """
        Run NCMC pipeline.

        Args:
            plan: Plan object.
            model: Model object.
            dataset: NCMCSingleMoleculeDataset object.
            forcefield_partial: Partially instantiated forcefield object.

        Returns:
            accepted_energies: Tensor of accepted energies.
        """

        # NCMC workflow variables
        accepted_samples = []
        accepted_energies = []
        rolling_batch_size = self.batch_size

        # NCMC simulation times
        rev_times = torch.linspace(1.0, 0.0, self.n_timesteps)
        gen_times = torch.linspace(0.0, 1.0, self.n_timesteps)

        # NCMC energy functions
        T_high, T_low = self.temperature_ladder[self._temperature_index]
        u_low = forcefield_partial(temperature=T_low)

        # Initialize the cold prior
        cold_prior = MeanFreePrior(
            n_particles=u_low.n_particles,
            spatial_dim=3,
            scale=(T_low / T_high) ** 0.5,
        )

        # get molecule object
        mol = dataset.molecules[dataset.pdb_id]

        # materialize the dataset
        starting_samples, starting_energies = dataset.materialize_tensors()
        # > convert to angstroms
        starting_samples = nm_to_angstrom(starting_samples)
        # > remove center of mass
        starting_samples = cold_prior.mean_free(starting_samples)
        # starting_samples: (n_samples, n_particles * 3)
        # starting_energies: (n_samples,)

        # run NCMC pipeline
        log.info(f"Running NCMC for {self.n_samples} samples")
        while len(accepted_samples) < self.n_samples:

            # randomly draw index values to slice the dataset
            indices = torch.randint(0, len(dataset), (rolling_batch_size,))

            # get the samples and energies
            ref_samples = starting_samples[indices]
            ref_energies = starting_energies[indices]

            # get graphs
            graph = mol.inference_graph_setup(len(ref_samples))

            # (1) Simulate the Reverse Process
            z, log_p_bwd_noise, log_p_fwd_noise = sde_integrate_heun(
                x_init=ref_samples,
                times=rev_times,
                graph=graph,
                vector_field=model,
                plan=plan,
                simulate_generative=False, # NOTE: NOT GENERATIVE --- FWD & BWD ARE SWAPPED!!!
                beta=1.0,
                zeta=1.0,
            )

            # (2) Simulat OU Bridge
            z_prime, log_q_fwd_latents, log_q_bwd_latents = simulate_ou_forward_reverse(
                z_init=z,
                n_steps=10,
                dt=0.1,
                cold_prior=cold_prior,
            )

            # (3) Simulate the Annealed Generative Process
            x_prime, log_p_fwd_gen, log_p_bwd_gen = sde_integrate_heun(
                x_init=z_prime,
                times=gen_times,
                graph=graph,
                vector_field=model,
                plan=plan,
                simulate_generative=True,
                beta=(T_high / T_low),
                zeta=1.0,
            )

            # (4) Evaluate the energies of the proposed samples
            x_prime_energies = -u_low(angstrom_to_nm(x_prime), return_force=False)

            # (5) Accept/reject the proposed samples
            accept_mask, log_alpha = ncmc_accept_reject(
                x_current=ref_samples,
                x_proposed=x_prime,
                log_pi_current=-ref_energies,
                log_pi_proposed=-x_prime_energies,
                log_fwd_noise=log_p_fwd_noise,
                log_bwd_noise=log_p_bwd_noise,
                log_fwd_latents=log_q_fwd_latents,
                log_bwd_latents=log_q_bwd_latents,
                log_fwd_gen=log_p_fwd_gen,
                log_bwd_gen=log_p_bwd_gen,
            )

            # Log number of samples accepted
            num_accepted = accept_mask.sum().item()
            log.info(f"Accepted {num_accepted} out of {rolling_batch_size} samples")

            # Continue if we did not accept any samples
            if num_accepted == 0:
                continue

            # Update the accepted samples and energies
            accepted_samples.extend(angstrom_to_nm(x_prime[accept_mask]).split(1, dim=0))
            accepted_energies.extend(x_prime_energies[accept_mask].split(1, dim=0))

            # Update the rolling batch size if we are not going to fill 
            if (self.n_samples - len(accepted_samples)) / rolling_batch_size < 1.0:
                rolling_batch_size = self.n_samples - len(accepted_samples)
        
        # Clear dataset cache and update the dataset
        dataset.clear_cache()
        dataset.update_dataset(
            mol_id=dataset.pdb_id,
            samples=accepted_samples,
            energies=accepted_energies,
        )

        # update temperature index
        self._temperature_index += 1

        # clean up
        del(accepted_samples, rolling_batch_size, rev_times, gen_times, T_high, T_low, u_low, cold_prior, mol, starting_samples, starting_energies, graph, z, log_p_bwd_noise, log_p_fwd_noise, z_prime, log_q_fwd_latents, log_q_bwd_latents, x_prime, log_p_fwd_gen, log_p_bwd_gen, x_prime_energies, accept_mask, log_alpha)
        torch.cuda.empty_cache()
        gc.collect()

        # return energies for plotting
        return torch.cat(accepted_energies, dim=0)