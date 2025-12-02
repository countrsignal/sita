import dgl
import torch
import numpy as np

import hydra
from omegaconf import DictConfig

import gc
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

from ..data.molecule import Molecule
from ..interpolants import Interpolant
from ..utils.logging import RankedLogger


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
        method: str = "dopri5",
        tsr_params: Optional[Dict[str, float]] = None,
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
                    method=method,
                    tsr_params=tsr_params,
                )
                samples_th.append(batch_samples.detach().cpu())
        results_dict[mol.name] = {"samples": torch.cat(samples_th, dim=0)}
        return results_dict