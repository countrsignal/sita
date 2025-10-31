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
from ..data.features import DEBUG_FEATURIZERS, feats_from_pdb

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
    
    def inference_prep(
        self,
        data_dir: str,
        model_type: str,
        eval_molecules: List[str],
    ) -> List[torch.Tensor]:

        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
        
        assert model_type in ["ebm", "flow"], "Model type must be either 'ebm' or 'flow'."
        
        if len(eval_molecules) == 0:
            pdb_files = list((data_dir / "pdbs").glob("*.pdb"))
        else:
            pdb_files = [data_dir / "pdbs" / f"{molecule}.pdb" for molecule in eval_molecules]
        
        features = []
        is_flow = model_type == "flow"
        has_debug_mols = [pdb.stem in self.config.debug_molecules for pdb in pdb_files]
        if any(has_debug_mols):
            assert all(has_debug_mols), "Your list of eval molecules CANNOT contain a mix of debug and non-debug molecules."
            for pdb in pdb_files:
                features.append(
                    DEBUG_FEATURIZERS[pdb.stem](return_concat=is_flow)
                )
        else:
            atom_types_encoding = np.load(data_dir / "atom_types_encoding.npy", allow_pickle=True).item()
            for pdb in pdb_files:
                features.append(
                    feats_from_pdb(pdb, atom_types_encoding, return_concat=is_flow)
                )
            # free up memory
            del(atom_types_encoding)
        # free up memory
        del(pdb_files, has_debug_mols, is_flow, data_dir, model_type, eval_molecules)
        gc.collect()
        return features