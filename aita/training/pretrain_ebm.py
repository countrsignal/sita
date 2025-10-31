import gc
from tqdm import trange

import hydra
from omegaconf import DictConfig

import torch
from torch import Tensor

from einops import repeat

from lightning import LightningModule

from typing import Optional, Dict

from ..utils.logging import RankedLogger
from ..pipeline.pipeline import Pipeline


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# Classes
###################################

class PreTrainerEBM(LightningModule):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        # Passing in config expands it one level, so can accessed
        # by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)
        # Ensure that setup() is only called once
        self._is_setup = False
        # Setup at init
        log.log(20, "Setting up pre-training modules for EBM...")
        self.setup()
    
    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup:
            return
        else:
            self._is_setup = True

        # Setup dataset and dataloader
        self.dataset = hydra.utils.instantiate(self.hparams.dataset)
        log.log(20, "Dataset Class Initialized.")

        # setup interpolant object
        ebm_plan = hydra.utils.instantiate(self.hparams.plans.ebm_plan)
        self.interpolant = hydra.utils.instantiate(self.hparams.interpolant)(plan=ebm_plan)
        log.log(20, "Interpolant Initialized.")

        # setup pipeline
        if "pipeline" in self.hparams and self.hparams.pipeline is not None:
            self.pipeline = Pipeline(self.hparams.pipeline)
            log.log(20, "Pipeline Initialized.")
        else:
            self.pipeline = None
            log.log(20, "Pipeline not found. Skipping pipeline.")
        
        # setup model
        self.ebm = hydra.utils.instantiate(self.hparams.ebm)
        log.log(20, "EBM Initialized.")

        # exponential moving average
        if "ema" in self.hparams:
            self.ema = hydra.utils.instantiate(self.hparams.ema, model=self.ebm)
            log.log(20, "Training with EMA.")
        else:
            self.ema = None
            log.log(20, "Training without EMA.")
    
    def train_dataloader(self):
        return self.dataset.get_train_dataloader(self.hparams.loader.batch_size, self.hparams.loader.num_workers, self.hparams.loader.pin_memory)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.ebm.parameters())
        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "pretrain/ebm/loss",  # make sure this matches exactly what you log
                "interval": "epoch",              # optional: how often to step the scheduler
                "frequency": 1,
            },
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        # setup EMA
        if self.ema is not None:
            if not self.ema.allow_different_devices:
                self.ema.to(self.device)

        #################################################################################
        # Generate training data from the flow model
        #################################################################################
        # Building dataset object
        log.log(20, "Loading pre-trained flow model...")
        flow_model = hydra.utils.instantiate(self.hparams.flow)
        flow_model.load_state_dict(torch.load(self.hparams.flow_model_ckpt, weights_only=True, map_location="cpu"))
        flow_model = flow_model.to(self.device)
        flow_model.eval()

        # NOTE: for EBM pre-training we use partial initalization
        flow_plan = hydra.utils.instantiate(self.hparams.plans.flow_plan)
        interpolant = hydra.utils.instantiate(self.hparams.interpolant)(plan=flow_plan)

        log.log(20, "Generating synthetic data from the flow model...")
        for molecule, one_hots_tuple in self.dataset.molecule_features.items():
            for _ in trange(self.hparams.sample_from_flow.n_batches, desc=f"Sampling conformers for {molecule}"):
                # sample from the flow model
                samples = interpolant.ode_integrate(
                    batch_size=self.hparams.sample_from_flow.samples_per_batch,
                    n_timesteps=self.hparams.sample_from_flow.n_timesteps,
                    categorical_features=torch.cat(one_hots_tuple, dim=1),
                    model=flow_model,
                ).cpu()
                # update the dataset with the new samples
                self.dataset.update_dataset(
                    mol_id=molecule,
                    samples=samples.chunk(self.hparams.sample_from_flow.samples_per_batch, dim=0),
                )
        log.log(20, "Data generation completed.")
        # clear memory
        del(samples, one_hots_tuple, molecule, interpolant, flow_model, flow_plan)
        gc.collect()

    def on_before_batch_transfer(self, batch: Dict[str, Tensor], dataloader_idx: int) -> Dict[str, Tensor]:
        # NOTE: we perform all the necessary data transformations here on CPU
        # > sample interpolants plan
        batch = self.interpolant.plan(batch)
        # apply data pipeline
        if self.pipeline is not None:
            batch = self.pipeline.run_ebm(batch, is_training=True)
        return batch
    
    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        # transfer batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # unpack batch
        t = batch["t"]
        z = batch["z"]
        x_t = batch["xt"]
        sigma_t = batch["sigma_t"]
        features = batch["features"]
        padding_mask = batch["padding_mask"]

        # predict energy
        grads, energies = self.ebm(
            time=t,
            features=features,
            coordinates=x_t,
            padding_mask=padding_mask,
            return_logprob=False,
            require_grad=True,
        )

        # score matching loss
        squared_errors = torch.square(sigma_t * grads + z).mean(dim=-1) # (B, L)
        # account for padding while computing mean
        score_loss = (
            squared_errors * ~padding_mask
        ).sum(dim=1) / (~padding_mask).sum(dim=1) # (B,)
        score_loss = score_loss.mean()

        # nce loss
        batch_size, n_atoms = t.size(0), t.size(1)
        perturb = torch.randn(batch_size, device=t.device) * 0.025
        negative_t = t + repeat(perturb, "b -> b n 1", n=n_atoms)
        negative_t = torch.clamp(negative_t, 0, 1)
        negative_energies = self.ebm(
            time=negative_t,
            features=features,
            coordinates=x_t,
            padding_mask=padding_mask,
            return_logprob=True,
            require_grad=False, # NOTE: no gradient tracking for the NCE loss
        )
        loss_nce = -torch.mean(
            energies - torch.logsumexp(
                torch.cat([energies, negative_energies], dim=-1),
                dim=-1,
                keepdim=True,
            )
        )

        # total loss
        loss = score_loss + loss_nce

        # log loss
        self.log("pretrain/ebm/score_loss", score_loss, prog_bar=True, logger=True)
        self.log("pretrain/ebm/nce_loss", loss_nce, prog_bar=True, logger=True)
        self.log("pretrain/ebm/loss", loss, prog_bar=True, logger=True)
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update()