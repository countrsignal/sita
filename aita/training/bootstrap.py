import hydra
from omegaconf import DictConfig

import dgl
import torch
from torch import Tensor

from lightning import LightningModule

from typing import Optional

from ..utils.logging import RankedLogger
from ..pipeline.pipeline import Pipeline


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# Classes
###################################

class LitBootstrap(LightningModule):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        # Passing in config expands it one level, so can accessed
        # by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)
        # Ensure that setup() is only called once
        self._is_setup = False
        # Setup at init
        log.log(20, "Setting up bootstrap modules...")
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup:
            return
        else:
            self._is_setup = True

        # Setup dataset and dataloader
        self.dataset = hydra.utils.instantiate(self.hparams.dataset)
        log.log(20, "Dataset Initialized.")
        
        # Setup pipeline
        pass