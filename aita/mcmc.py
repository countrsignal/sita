import torch
from torch import nn, Tensor

import hydra 
from omegaconf import DictConfig

import gc
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Tuple, Optional, Any, Dict, Union

from .interpolants import Interpolant
from .utils.logging import RankedLogger
from .data.molecule import Molecule, DEBUG_MOLECULES
from .utils.data_utils import nm_to_angstrom, angstrom_to_nm
from .energies.base_molecule_energy_function import BaseMoleculeEnergy


log = RankedLogger(__name__, on_rank_zero=True)


#############################################################################################################################
# Functions
#############################################################################################################################

def has_tensor_cores() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap >= (8, 0)  # Ampere+ devices expose FP32 tensor cores


def evaluate_log_p_and_log_q(
    x: Tensor,
    mol: Molecule,
    ebm: nn.Module,
    forcefield: BaseMoleculeEnergy,
) -> Tuple[Tensor, Tensor]:
    """
    Evaluate the log probabilities of the EBM and the forcefield.
    """


    # Evaluate EBM log probabilities
    # > prepare features, padding mask, and times
    features = mol.atom_types.argmax(dim=1).unsqueeze(0).repeat(x.size(0), 1)
    padding_mask = torch.zeros_like(features, dtype=torch.bool)
    times = torch.ones((x.size(0), x.size(1), 1))

    # move to device
    device = next(ebm.parameters()).device
    features = features.to(device)
    padding_mask = padding_mask.to(device)
    times = times.to(device)
    if x.device != device:
        x = x.to(device)

    # Evaluate EBM log probabilities
    with torch.inference_mode():
        log_q = ebm(
            time=times,
            features=features,
            coordinates=x,
            padding_mask=padding_mask,
            return_logprob=True,
            require_grad=False,
        ).flatten()

    # Evaluate forcefield log probabilities
    log_p = forcefield(angstrom_to_nm(x.reshape(-1, mol.n_atoms * 3)), return_force=False)
    
    # Clean up memory
    del(features, padding_mask, times)
    torch.cuda.empty_cache()
    gc.collect()

    # Return log probabilities
    return log_p, log_q


#############################################################################################################################
# Classes
#############################################################################################################################

@dataclass
class ProposalState:
    y: Tensor
    log_p: Tensor
    log_q: Tensor
    log_rnd: Optional[Tensor] = field(default_factory=lambda: None)

    def __len__(self):
        return len(self.y)
    
    def to(self, device: torch.device) -> "ProposalState":
        self.y = self.y.to(device)
        self.log_p = self.log_p.to(device)
        self.log_q = self.log_q.to(device)
        if self.log_rnd is not None:
            self.log_rnd = self.log_rnd.to(device)
        return self
    
    def cpu(self) -> "ProposalState":
        return self.to(torch.device("cpu"))

    def log_w(self) -> Tensor:
        return self.log_p - self.log_q


@dataclass
class ChainState:
    x: Tensor
    log_p: Tensor
    log_q: Tensor
    log_rnd: Optional[Tensor] = field(default_factory=lambda: None)

    def __len__(self):
        return len(self.x)
    
    def to(self, device: torch.device) -> "ChainState":
        self.x = self.x.to(device)
        self.log_p = self.log_p.to(device)
        self.log_q = self.log_q.to(device)
        if self.log_rnd is not None:
            self.log_rnd = self.log_rnd.to(device)
        return self
    
    def cpu(self) -> "ChainState":
        return self.to(torch.device("cpu"))
    
    def log_w(self) -> Tensor:
        return self.log_p - self.log_q
    
    def prospective_log_w(self, proposal_log_w: Tensor, accept_mask: Tensor) -> Tensor:
        log_w = self.log_w().clone()
        log_w[accept_mask] = proposal_log_w[accept_mask]
        return log_w
    
    def update_chain(self, proposal: ProposalState, accept_mask: Tensor) -> "ChainState":
        self.x[accept_mask] = proposal.y[accept_mask]
        self.log_p[accept_mask] = proposal.log_p[accept_mask]
        self.log_q[accept_mask] = proposal.log_q[accept_mask]
        if proposal.log_rnd is not None:
            self.log_rnd[accept_mask] = proposal.log_rnd[accept_mask]
        return self


class MCMC(ABC):

    def __init__(
        self,
        pdb_path: str,
        n_chains: int,
        n_timesteps: int,
        sampling_kwargs: Union[DictConfig, Dict[str, Any]],
    ):
        self.pdb_path = Path(pdb_path)
        self.n_chains = n_chains
        self.n_timesteps = n_timesteps
        self.sampling_kwargs = sampling_kwargs
        # parse and load molecule
        if self.pdb_path.stem in DEBUG_MOLECULES:
            self.molecule = DEBUG_MOLECULES[self.pdb_path.stem].from_pdb(self.pdb_path)
        else:
            self.molecule = Molecule.from_pdb(self.pdb_path)
    
    @abstractmethod
    def init_chains(self, interpolant: Interpolant, forcefield: Callable, flow: nn.Module, ebm: nn.Module) -> ChainState:
        pass

    @abstractmethod
    def propose(self, chain: ChainState, interpolant: Interpolant, forcefield: Callable, flow: nn.Module, ebm: nn.Module) -> ProposalState:
        pass

    @abstractmethod
    def log_acceptance_ratio(self, chain: ChainState, proposal: ProposalState) -> Tensor:
        pass

    @torch.no_grad()
    def accept_reject(self, log_alpha: Tensor) -> Tensor:
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
        # --- Log acceptance ratio & number of accepted samples ---
        log.info(f"Log Acc. Rate: {log_alpha.mean().item():.4f} | N. Accepted: {accept_mask.sum().item()}")
        # --- Return accept mask and log alpha ---
        return accept_mask


class IMH(MCMC):

    def __init__(self, pdb_path: str, n_chains: int, n_timesteps: int, sampling_kwargs: Union[DictConfig, Dict[str, Any]]):
        super().__init__(pdb_path, n_chains, n_timesteps, sampling_kwargs)
    
    def init_chains(self, interpolant: Interpolant, forcefield: Callable, flow: nn.Module, ebm: nn.Module) -> ChainState:
        
        # Generate initial samples
        x_init = interpolant.ode_integrate(
            mol=self.molecule,
            batch_size=self.n_chains,
            n_timesteps=self.n_timesteps,
            model=flow,
            **self.sampling_kwargs,
        )
        # x_init: (n_chains, n_atoms, 3)
        # NOTE: `x_init` should still be on the model device

        # Evaluate log probabilities
        log_p, log_q = evaluate_log_p_and_log_q(
            x=x_init,
            mol=self.molecule,
            ebm=ebm,
            forcefield=forcefield,
        )
        # log_p: (n_chains,)
        # log_q: (n_chains,)

        # Return chain state on CPU
        return ChainState(
            x=x_init,
            log_p=log_p,
            log_q=log_q,
        ).cpu()
    
    @torch.inference_mode()
    def propose(self, chain: ChainState, interpolant: Interpolant, forcefield: Callable, flow: nn.Module, ebm: nn.Module) -> ProposalState:
        # NOTE: IMH is an independent Metropolis-Hastings sampler,
        #   so the current state of the chain is not used to generate the proposal

        # Generate proposal samples
        proposal = interpolant.ode_integrate(
            mol=self.molecule,
            batch_size=self.n_chains,
            n_timesteps=self.n_timesteps,
            model=flow,
            **self.sampling_kwargs,
        )
        # proposal: (n_chains, n_atoms, 3)
        # NOTE: `proposal` should still be on the model device

        # Evaluate log probabilities
        log_p, log_q = evaluate_log_p_and_log_q(
            x=proposal,
            mol=self.molecule,
            ebm=ebm,
            forcefield=forcefield,
        )
        # log_p: (n_chains,)
        # log_q: (n_chains,)

        # Return proposal state on CPU
        return ProposalState(
            y=proposal,
            log_p=log_p,
            log_q=log_q,
        ).cpu()

    @torch.no_grad()
    def log_acceptance_ratio(self, chain: ChainState, proposal: ProposalState) -> Tensor:
        # Calculate log acceptance ratio
        log_alpha = proposal.log_w() - chain.log_w()
        # Return log acceptance ratio
        return log_alpha


class MonteCarloAnnealing:

    def __init__(self, config: DictConfig):
        self.config = config
        self._is_setup = False

        # setup at init
        log.log(20, "Setting up Monte Carlo Annealing...")
        self.setup()

        # diagnostics dictionary to track progress
        self.diagnostics = {
            "log_acceptance_rate": [],
            "log_w_variance": [],
            "log_w_max": [],
            "weights_variance": [],
            "weights_max": [],
        }

    def setup(self) -> None:
        if self._is_setup:
            return
        else:
            self._is_setup = True
        
        # check if tensor cores are available
        if has_tensor_cores():
            log.log(20, "Tensor cores are available. Setting float32 matmul precision to 'high'...")
            torch.set_float32_matmul_precision('high')

        # instantiate interpolant
        log.log(20, "Instantiating interpolant...")
        self.interpolant = hydra.utils.instantiate(self.config.interpolant)
        # instantiate forcefield
        log.log(20, "Instantiating (partial) forcefield...")
        # NOTE: Temperature is not yet specified, so we instantiate a partial forcefield
        self.forcefield_partial = hydra.utils.instantiate(self.config.energy)
        # instantiate flow
        log.log(20, "Instantiating flow...")
        self.flow = hydra.utils.instantiate(self.config.flow)
        self.flow = self.flow.load_from_checkpoint(self.config.flow_model_ckpt, weights_only=True, map_location="cpu")
        self.flow.eval();
        # instantiate ebm
        log.log(20, "Instantiating ebm...")
        self.ebm = hydra.utils.instantiate(self.config.ebm)
        self.ebm = self.ebm.load_from_checkpoint(self.config.ebm_model_ckpt, weights_only=True, map_location="cpu")
        self.ebm.eval();
        # instantiate mcmc
        log.log(20, "Instantiating placeholder for mcmc...")
        # NOTE: MCMC class is instantiated inside the run method
        #       It is re-instantiated for each rung on the temperature ladder
        self.mcmc = None
        log.log(20, "Monte Carlo Annealing setup complete.")

    def dispatch_models(self, device: torch.device) -> None:
        self.flow.to(device)
        self.ebm.to(device)

    def run(self, annealing_index: int) -> None:
        # instantiate mcmc
        # > compute value prior beta
        T_high, T_low = self.config.temperature_ladder[annealing_index]
        self.config.mcmc.sampling_kwargs.prior_beta = (T_low / T_high) ** 0.5
        # > instantiate full forcefield
        forcefield = self.forcefield_partial(temperature=T_low)
        # > initialize mcmc
        self.mcmc = hydra.utils.instantiate(self.config.mcmc)
        # initialize chains
        log.log(20, "Initializing chains...")
        chains = self.mcmc.init_chains(
            interpolant=self.interpolant,
            forcefield=forcefield,
            flow=self.flow,
            ebm=self.ebm,
        )
        log.log(20, "Chains initialized.")
        # run mcmc
        log.log(20, "Running Monte Carlo Annealing...")
        for _ in tqdm(range(self.config.n_steps), desc="Running Monte Carlo Annealing"):
            # propose new state
            proposal = self.mcmc.propose(chains, self.interpolant, forcefield, self.flow, self.ebm)
            # calculate log acceptance ratio
            log_alpha = self.mcmc.log_acceptance_ratio(chains, proposal)
            # accept/reject
            accept_mask = self.mcmc.accept_reject(log_alpha)

            # inspect whether acceptance will increase the log-weight variance
            # log_w = chains.log_w().var()
            # log_w_virtual = chains.prospective_log_w(proposal.log_w(), accept_mask).var()
            # if log_w_virtual > log_w:
            #     # If variance increase, we terminate the chain and return the current chains
            #     log.log(20, "Log-weight variance increased. Terminating chain.")
            #     return chains

            # update chains
            chains = chains.update_chain(proposal, accept_mask)
            # track diagnostics
            if self.config.run_diagnostics:
                self.diagnostics["log_acceptance_rate"].append(log_alpha.mean().item())
                self.diagnostics["log_w_variance"].append(chains.log_w().var().item())
                self.diagnostics["log_w_max"].append(chains.log_w().max().item())
                self.diagnostics["weights_variance"].append(torch.softmax(chains.log_w(), dim=0).var().item())
                self.diagnostics["weights_max"].append(torch.softmax(chains.log_w(), dim=0).max().item())
        # return chains
        log.log(20, "Monte Carlo Annealing completed.")
        return chains