import hydra
from omegaconf import DictConfig

import dgl
import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger

from typing import Optional, List

from .common import fetch_wandb_logger
from ..utils.logging import RankedLogger


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# Classes
###################################

class PreTrainerFlow(LightningModule):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        # Passing in config expands it one level, so can accessed
        # by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)
        # Ensure that setpu() is only called once
        self._is_setup = False
        # Setup at init
        log.log(20, "Setting up pre-training modules for flow model...")
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup:
            return
        else:
            self._is_setup = True

        # Setup dataset and dataloader
        self.dataset = hydra.utils.instantiate(self.hparams.dataset)
        log.log(20, "Dataset Initialized.")

        # Setup interpolant object
        self.interpolant = hydra.utils.instantiate(self.hparams.interpolant)
        log.log(20, "Interpolant Initialized.")

        # Setup model
        self.flow = hydra.utils.instantiate(self.hparams.flow)
        log.log(20, "Model Initialized.")

        # Exponential moving average
        if "ema" in self.hparams:
            self.ema = hydra.utils.instantiate(self.hparams.ema, model=self.flow)
            log.log(20, "Training with EMA.")
        else:
            self.ema = None
            log.log(20, "Training without EMA.")
    
    def train_dataloader(self):
        return self.dataset.get_train_dataloader(self.hparams.loader.batch_size, self.hparams.loader.num_workers, self.hparams.loader.pin_memory)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.flow.parameters())
        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "pretrain/flow/loss",  # make sure this matches exactly what you log
                "interval": "epoch",              # optional: how often to step the scheduler
                "frequency": 1,
            },
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.ema is not None:
            if not self.ema.allow_different_devices:
                # if allow_different_devices is False
                # # then the ema model must be on the same device as the model
                self.ema.to(self.device)
    
    def training_step(self, batch: dgl.DGLGraph, batch_idx: int) -> Tensor:
        # sample interpolants plan
        batch = self.interpolant.plan(batch)
        # predict velocity
        velocity = self.flow(batch)
        # compute loss
        batch.ndata["loss_per_node"] = torch.square(velocity - batch.ndata["vt"]).mean(dim=-1)
        loss_per_mol = dgl.mean_nodes(batch, "loss_per_node")
        loss = loss_per_mol.mean()
        # log loss
        self.log(
            "pretrain/flow/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.batch_size,
            )
        return loss
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update()