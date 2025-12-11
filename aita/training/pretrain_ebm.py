import gc
import numpy as np
from tqdm import trange

import hydra
from omegaconf import DictConfig

import torch
from torch import Tensor

from einops import repeat

from lightning import LightningModule

from typing import Optional, Dict

from ..data.molecule import Molecule
from ..pipeline.pipeline import Pipeline
from ..utils.logging import RankedLogger
from ..utils.plotting import plot_ebm_histogram
from .common import fetch_wandb_logger, eval_ebm_single_molecule


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
        
        # setup evaluation dataloader
        self.eval_dl = self.dataset.get_eval_dataloader(
            self.hparams.loader.batch_size,
            num_workers=0,
            pin_memory=False,
        )
        
        # variable trackers
        self._epoch = 0
    
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
        # Initialize flow model
        log.log(20, "Loading pre-trained flow model...")
        flow_model = hydra.utils.instantiate(self.hparams.flow)
        flow_model = flow_model.load_from_checkpoint(self.hparams.flow_model_ckpt, weights_only=True, map_location="cpu")
        
        # compile model for faster inference
        flow_model.edge_embedding = torch.compile(flow_model.edge_embedding)
        for idx in range(flow_model.gvp_decoder.n_layers):
            flow_model.gvp_decoder.edge_updater[idx] = torch.compile(flow_model.gvp_decoder.edge_updater[idx])
        flow_model.gvp_decoder.position_updater = torch.compile(flow_model.gvp_decoder.position_updater)
        
        # move to device
        flow_model = flow_model.to(self.device)
        flow_model.eval()

        # NOTE: for EBM pre-training we use partial initalization
        flow_plan = hydra.utils.instantiate(self.hparams.plans.flow_plan)
        interpolant = hydra.utils.instantiate(self.hparams.interpolant)(plan=flow_plan)

        log.log(20, "Generating synthetic data from the flow model...")
        mol = self.dataset.molecules[self.dataset.pdb_id]
        res_dict = Pipeline.generate_from_flow(
            n_samples=self.hparams.sample_from_flow.n_samples,
            samples_per_batch=self.hparams.sample_from_flow.samples_per_batch,
            n_timesteps=self.hparams.sample_from_flow.n_timesteps,
            molecules=[mol],
            flow_model=flow_model,
            interpolant=interpolant,
            method=self.hparams.sample_from_flow.method,
            tsr_params=self.hparams.sample_from_flow.tsr_params,
        )
        log.log(20, "Data generation completed.")

        # update dataset with the new samples
        self.dataset.update_dataset(
            mol_id=mol.name,
            samples=res_dict[mol.name]["samples"].chunk(self.hparams.sample_from_flow.n_samples, dim=0),
        )

        # save generated samples in numpy file
        np.save(
            self.hparams.sample_from_flow.output_path,
            res_dict[mol.name]["samples"].numpy(),
            allow_pickle=True,
        )

        # clear memory
        del(mol, res_dict, interpolant, flow_model, flow_plan)
        torch.cuda.empty_cache()
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
    
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        self._epoch += 1
        if self._epoch % self.hparams.ebm_inspection_interval == 0:
            wandb_logger = fetch_wandb_logger(self.loggers)
            if wandb_logger is not None:
                log_probs = eval_ebm_single_molecule(self.ebm, self.eval_dl, self.device)
                plot_ebm_histogram(
                    log_probs,
                    prefix="media/",
                    wandb_logger=wandb_logger,
                )
            else:
                log.log(20, "No wandb logger found. Skipping EBM inspection.")


    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update()