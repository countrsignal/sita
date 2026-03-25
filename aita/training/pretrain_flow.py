import hydra
from omegaconf import DictConfig

import dgl
import torch
from torch import Tensor

from lightning import LightningModule

from typing import Optional

from ..pipeline.pipeline import Pipeline
from ..utils.logging import RankedLogger
from ..utils.graph_utils import GraphAdapter


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
        # Ensure that setup() is only called once
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

        # DEBUG
        log.debug(f"Interpolant: {type(self.interpolant)}")
        log.debug(f"Interpolant plan: {type(self.interpolant.plan)}")

        # Setup pipeline
        if "pipeline" in self.hparams and self.hparams.pipeline is not None:
            self.pipeline = Pipeline(self.hparams.pipeline)
            log.log(20, "Pipeline Initialized.")
        else:
            self.pipeline = None
            log.log(20, "Pipeline not found. Skipping pipeline.")

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
        if "scheduler" in self.hparams:
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
        else:
            return {"optimizer": optimizer}

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self.ema is not None:
            if not self.ema.allow_different_devices:
                # if allow_different_devices is False
                # # then the ema model must be on the same device as the model
                self.ema.to(self.device)

    def on_before_batch_transfer(self, batch: dgl.DGLGraph, dataloader_idx: int) -> dgl.DGLGraph:
        # NOTE: we perform all the necessary data transformations here on CPU
        # > sample interpolants plan
        batch = self.interpolant.plan(batch)
        # apply data pipeline
        if self.pipeline is not None:
            batch = self.pipeline.run_flow(batch, is_training=True)

        return batch

    def training_step(self, batch: dgl.DGLGraph, batch_idx: int) -> Tensor:
        # training step
        loss_dict = self.flow.training_step(batch)

        # log loss
        for key, value in loss_dict.items():
            self.log(
                f"pretrain/flow/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch.batch_size,
                )
        return loss_dict["loss"]
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update()


class PreTrainerATFlow(PreTrainerFlow):
    """
        Same as PreTrainerFlow, but for Atomic Transformer Flow models.
        We only need to override the training step to use the new model.
    """

    def __init__(self, config: DictConfig) -> None:
        super(PreTrainerATFlow, self).__init__(config)
    
    def training_step(self, batch: dgl.DGLGraph, batch_idx: int) -> Tensor:
        # adapt the dgl graph 
        adapter = GraphAdapter(batch)

        # randomly rotate input coordinates and target velocity fields
        batch = adapter.apply_random_rotations(batch, node_keys=["xt", "vt"])

        # convert to padded tensors
        data = adapter.graph_to_padded_tensor(batch, use_rbf=True)

        # unpack the training data
        time, x_t, attr, atom_index, pair_feats, atom_mask, pair_mask, target_velocity = data

        # forward pass
        loss_dict = self.flow.training_step(
            x_t=x_t,
            time=time,
            attr=attr,
            atom_index=atom_index,
            pair_feats=pair_feats,
            atom_mask=atom_mask,
            pair_mask=pair_mask,
            target_velocity=target_velocity,
        )

        # log loss
        for key, value in loss_dict.items():
            self.log(
                f"pretrain/flow/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch.batch_size,
            )
        return loss_dict["loss"]