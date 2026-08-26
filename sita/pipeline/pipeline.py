import dgl
import torch
from typing import Optional

import hydra
from omegaconf import DictConfig

import gc
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple

from ..plans import Plan
from ..data.molecule import Molecule
from ..interpolants import Interpolant
from ..utils.logging import RankedLogger
from ..utils.data_utils import angstrom_to_nm
from ..diffusion import sde_integrate_heun, MeanFreePrior
from ..energies.base_molecule_energy_function import BaseMoleculeEnergy


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# Classes
###################################

class Protocol(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        pass


class Pipeline:

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        # Ensure that setup() is only called once
        self._is_setup = False
        self.setup()
    
    def setup(self) -> None:
        if self._is_setup:
            return
        else:
            self._is_setup = True
        
        # Setup flow protocol
        if "flow" not in self.config:
            self.flow_train = []
            self.flow_inference = []
        else:
            if "train" in self.config.flow:
                self.flow_train = hydra.utils.instantiate(self.config.flow.train)
            else:
                self.flow_train = []

            if "inference" in self.config.flow:
                self.flow_inference = hydra.utils.instantiate(self.config.flow.inference)
            else:
                self.flow_inference = []
        
        # Setup ebm protocol
        if "ebm" not in self.config:
            self.ebm_train = []
            self.ebm_inference = []
        else:
            if "train" in self.config.ebm:
                self.ebm_train = hydra.utils.instantiate(self.config.ebm.train)
            else:
                self.ebm_train = []

            if "inference" in self.config.ebm:
                self.ebm_inference = hydra.utils.instantiate(self.config.ebm.inference)
            else:
                self.ebm_inference = []
    
    def run_flow(self, graph: dgl.DGLGraph, is_training: bool = True) -> dgl.DGLGraph:
        if is_training:
            if len(self.flow_train) == 0:
                return graph

            for protocol in self.flow_train:
                graph = protocol(graph)
        else:
            if len(self.flow_inference) == 0:
                return graph

            for protocol in self.flow_inference:
                graph = protocol(graph)
        return graph
    
    def run_ebm(self, inputs: Dict[str, torch.Tensor], is_training: bool = True) -> Dict[str, torch.Tensor]:
        if is_training:
            if len(self.ebm_train) == 0:
                return inputs

            for protocol in self.ebm_train:
                inputs = protocol(inputs)
        else:
            if len(self.ebm_inference) == 0:
                return inputs

            for protocol in self.ebm_inference:
                inputs = protocol(inputs)
        return inputs

    @staticmethod
    def generate_from_flow(
        n_samples: int,
        samples_per_batch: int,
        n_timesteps: int,
        molecules: List[Molecule],
        flow_model: torch.nn.Module,
        interpolant: Interpolant,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate samples from the flow model.
        """
        assert n_samples % samples_per_batch == 0, "`n_samples` must be divisible by `samples_per_batch`"
        n_batches = n_samples // samples_per_batch

        samples_th = []
        results_dict: Dict[str, Dict[str, torch.Tensor]] = {}
        for mol in molecules:
            for _ in tqdm(range(n_batches), desc=f"Generating samples for {mol.name}"):
                batch_samples = interpolant.ode_integrate(
                    mol=mol,
                    model=flow_model,
                    batch_size=samples_per_batch,
                    n_timesteps=n_timesteps,
                    **kwargs,
                )
                samples_th.append(batch_samples.detach().cpu())
        results_dict[mol.name] = {"samples": torch.cat(samples_th, dim=0)}
        return results_dict

    @staticmethod
    def generate_from_flow_sde(
        n_samples: int,
        samples_per_batch: int,
        n_timesteps: int,
        molecule: Molecule,
        flow_model: torch.nn.Module,
        plan: Plan,
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        """
        Generate samples by converting the flow model into a diffusion model and then sampling the SDE.
        """
        assert n_samples % samples_per_batch == 0, "`n_samples` must be divisible by `samples_per_batch`"
        n_batches = n_samples // samples_per_batch

        # model device
        device = next(flow_model.parameters()).device

        # create a graph with the given categorical features
        # we should have a batch size of batch_size * num_nodes
        g = molecule.inference_graph_setup(samples_per_batch)

        # create a mean free prior
        prior = MeanFreePrior(
            n_particles=molecule.n_atoms,
            spatial_dim=3,
            scale=1.0,
            device="cpu",
        )

        # prepare time variables
        time_span = torch.linspace(0.0, 1.0, n_timesteps + 1)

        # move data to device
        g = g.to(device)
        time_span = time_span.to(device)

        # initialize list of samples
        samples_th = []
        results_dict: Dict[str, torch.Tensor] = {}
        for _ in tqdm(range(n_batches), desc=f"Generating samples for {molecule.name}"):
            
            # prepare initial state
            x_init = prior.sample(samples_per_batch)
            x_init = x_init.to(device)

            # integrate the SDE
            batch_samples, *_ = sde_integrate_heun(
                x_init=x_init,
                times=time_span,
                graph=g,
                vector_field=flow_model,
                plan=plan,
                simulate_generative=True,
                beta=1.0,
                zeta=1.0,
            )
            samples_th.append(batch_samples)

        results_dict[molecule.name] = {"samples": torch.cat(samples_th, dim=0).view(n_samples, molecule.n_atoms, 3)}
        return results_dict

    @staticmethod
    def evaluate_log_probs(
        batch_size: int,
        mol: Molecule,
        samples: torch.Tensor,
        ebm: torch.nn.Module,
        forcefield: Optional[BaseMoleculeEnergy] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probabilities of both the EBM and the forcefield.

        Args:
            ebm: EBM model.
            loader: Data loader.
            device: Device.
            forcefield: Forcefield.

        Returns:
            log_probs: Log probabilities of the samples.
            energies: Energies of the samples.
        """

        assert (samples.size(0) % batch_size) == 0, "Total number of samples must be divisible by `batch_size`"

        # set model to evaluation mode
        if ebm.training:
            ebm.eval();
        
        # reshape the samples to (batch_size, num_atoms, 3)
        if samples.dim() == 2:
            samples = samples.reshape(-1, mol.n_atoms, 3)

        # get device id from EBM model
        device = next(ebm.parameters()).device

        # prepare features, padding mask, and times
        features = mol.atom_types.argmax(dim=1).unsqueeze(0).repeat(batch_size, 1)
        padding_mask = torch.zeros_like(features, dtype=torch.bool)
        times = torch.ones((batch_size, samples.size(1), 1))

        # move to device
        features = features.to(device)
        padding_mask = padding_mask.to(device)
        times = times.to(device)

        # break samples into batches
        batches = samples.split(batch_size, dim=0) # DO NOT USE CHUNK HERE!

        # evaluate EBM
        log_probs_ebm: List[torch.Tensor] = []
        log_probs_ffd: List[torch.Tensor] = []
        for batch in tqdm(batches, desc="Evaluating log-probabilities"):

            # move batch to device
            batch = batch.to(device)

            with torch.no_grad():
                # evaluate EBM
                log_p_ebm = ebm(
                    time=times,
                    features=features,
                    coordinates=batch,
                    padding_mask=padding_mask,
                    return_logprob=True,
                    require_grad=False,
                )

                # evaluate forcefield
                log_p_ffd = forcefield(angstrom_to_nm(batch.reshape(-1, mol.n_atoms * 3)), return_force=False)

            # append log probabilities
            log_probs_ebm.append(log_p_ebm.cpu().flatten())
            log_probs_ffd.append(log_p_ffd.cpu().flatten())

        # concatenate log probabilities
        log_probs_ebm = torch.cat(log_probs_ebm, dim=-1)
        log_probs_ffd = torch.cat(log_probs_ffd, dim=-1)

        # clean up memory
        del(batch, features, padding_mask, times, log_p_ebm, log_p_ffd)
        torch.cuda.empty_cache()
        gc.collect()

        # return log probabilities
        return log_probs_ebm, log_probs_ffd