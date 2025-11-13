import dgl
import torch
import numpy as np

import hydra
from omegaconf import DictConfig

import gc
from pathlib import Path
from typing import Dict, List, Any
from abc import ABC, abstractmethod

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