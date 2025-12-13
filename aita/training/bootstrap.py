import gc
import os
import numpy as np
import mdtraj as md
from typing import Optional, List, Dict, Union

import hydra
from omegaconf import DictConfig

import dgl
import torch
from torch import Tensor

from lightning import LightningModule

from ..models.vector_field_v2 import VFV2
from ..utils.logging import RankedLogger
from ..pipeline.pipeline import Pipeline
from ..data.datasets import GenerativeDatasetSingleMolecule
from .common import fetch_wandb_logger, eval_ebm_single_molecule
from ..utils.plotting import plot_ebm_histogram, plot_energy_histograms, adp_ramachandran_plot, adp_free_energy_profile
from ..utils.inference_utils import calc_log_w, quantile_clip, quantile_filter, normalize_log_w, calc_ess, importance_weighted_resample


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# Types
###################################

BATCH_FLOW = dgl.DGLGraph
BATCH_EBM = Dict[str, Tensor]
BATCH = Union[BATCH_FLOW, BATCH_EBM]

###################################
# functions
###################################

def load_reference_md_data(dcd_path: str, pdb_path: str) -> md.Trajectory:


###################################
# Classes
###################################

class AnnealBootstrap(LightningModule):

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
        self.dataset: GenerativeDatasetSingleMolecule = hydra.utils.instantiate(self.hparams.dataset)
        log.log(20, "Dataset Initialized.")

        # setup MD forcefield
        # NOTE: We partial instantiate the forcefield as we will be instantiating a full forcefield for each temperature level
        self.forcefield_partial = hydra.utils.instantiate(self.hparams.forcefield)
        log.log(20, "Forcefield Initialized.")

        # setup interpolant object
        ebm_plan = hydra.utils.instantiate(self.hparams.plans.ebm_plan)
        flow_plan = hydra.utils.instantiate(self.hparams.plans.flow_plan)
        self.ebm_interpolant = hydra.utils.instantiate(self.hparams.interpolant)(plan=ebm_plan)
        self.flow_interpolant = hydra.utils.instantiate(self.hparams.interpolant)(plan=flow_plan)
        log.log(20, "Interpolants Initialized.")

        # Setup pipeline
        if "pipeline" in self.hparams and self.hparams.pipeline is not None:
            self.pipeline = Pipeline(self.hparams.pipeline)
            log.log(20, "Pipeline Initialized.")
        else:
            self.pipeline = None
            log.log(20, "Pipeline not found. Skipping pipeline.")
        
        # setup models
        # > setup EBM
        self.ebm = hydra.utils.instantiate(self.hparams.ebm)
        log.log(20, "EBM Initialized.")
        # > load pre-trained flow model
        self.flow: VFV2 = hydra.utils.instantiate(self.hparams.flow)
        self.flow = self.flow.load_from_checkpoint(self.hparams.flow_model_ckpt, weights_only=True, map_location="cpu")
        log.log(20, "Pre-trained Flow Model Loaded.")

        # exponential moving average
        if "ema" in self.hparams:
            self.ema = hydra.utils.instantiate(self.hparams.ema, model=self.ebm)
            log.log(20, "Training with EMA.")
        else:
            self.ema = None
            log.log(20, "Training without EMA.")
        
        # setup evaluation dataloader as null (gets initialized in on_fit_start)
        self.eval_dl = None
        
        # variable trackers
        self._epoch: int = 0
        self._training_era: str = "ebm"
        self._temperature_index: int = 0
        self._temperature_ladder: List[int] = self.hparams.temperature_ladder
    
    @property
    def training_era(self) -> str:
        return self._training_era
    
    @training_era.setter
    def training_era(self, value: str) -> None:
        if value not in ["ebm", "flow"]:
            raise ValueError(f"Invalid training era: {value}")
        self._training_era = value
    
    def training_model_swap(self, era: str) -> None:
        """Swap which model is being trained. Also re-initializes EMA if using."""

        assert self.training_era != era, f"Next era cannot be the same as the current era."

        if era == "ebm":
            # train EBM
            self.ebm.train();
            self.flow.eval();
            # re-initialize EMA if using
            if self.ema is not None:
                # > save Flow EMA model weights from previous era
                temp = self._temperature_ladder[self._temperature_index]
                torch.save(self.ema.ema_model.state_dict(), os.path.join(self.hparams.era_ckpt_dir, f"flow_ema_model_{temp}K.pth"))
                # > re-initialize EMA
                self.ema = hydra.utils.instantiate(self.hparams.ema, model=self.ebm)
        else:
            # train Flow
            self.ebm.eval();
            self.flow.train();
            # re-initialize EMA if using
            if self.ema is not None:
                # > save EBM ema model weights from previous era
                temp = self._temperature_ladder[self._temperature_index]
                torch.save(self.ema.ema_model.state_dict(), os.path.join(self.hparams.era_ckpt_dir, f"ebm_ema_model_{temp}K.pth"))
                # > re-initialize EMA
                self.ema = hydra.utils.instantiate(self.hparams.ema, model=self.flow)
        
        self.training_era = era
        self._temperature_index += 1
    
    def train_dataloader(self):
        return self.dataset.get_train_dataloader(self.hparams.loader.batch_size, self.hparams.loader.num_workers, self.hparams.loader.pin_memory)

    def configure_optimizers(self):
        if self.training_era == "ebm":
            optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.ebm.parameters())
        else:
            optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.flow.parameters())
        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"{self.training_era}/loss",  # make sure this matches exactly what you log
                "interval": "epoch",              # optional: how often to step the scheduler
                "frequency": 1,
            },
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        if self.ema is not None:
            if not self.ema.allow_different_devices:
                self.ema.to(self.device)

        ################################################################################
        # EBM Training Era
        ################################################################################
        if self.training_era == "ebm":
            # move flow model to GPU device
            self.flow = self.flow.to(self.device)
            self.flow.eval();
            
            if self._temperature_index == 0:
                log.log(20, "Generating data from the pre-trained flow model at 1200 Kelvin ...")
            else:
                temp = self._temperature_ladder[self._temperature_index - 1]
                log.log(20, f"Generating annealed data from the flow model at {temp} Kelvin ...")

            # generate data from the flow model
            mol = self.dataset.molecules[self.dataset.pdb_id]
            res_dict = Pipeline.generate_from_flow(
                n_samples=self.hparams.sample_from_flow.n_samples,
                samples_per_batch=self.hparams.sample_from_flow.samples_per_batch,
                n_timesteps=self.hparams.sample_from_flow.n_timesteps,
                molecules=[mol],
                flow_model=self.flow,
                interpolant=self.flow_interpolant,
                method=self.hparams.sample_from_flow.method,
                tsr_params=self.hparams.sample_from_flow.tsr_params,
            )
            log.log(20, "Data generation completed.")

            # Reset and update dataset with the new samples
            self.dataset.clear_cache()
            self.dataset.update_dataset(
                mol_id=mol.name,
                samples=res_dict[mol.name]["samples"].chunk(self.hparams.sample_from_flow.n_samples, dim=0),
            )
            self.dataset.training_sampler = False

            # save generated samples in numpy file
            np.save(
                os.path.join(self.hparams.era_ckpt_dir, f"flow_samples_{temp}K.npy"),
                res_dict[mol.name]["samples"].numpy(),
                allow_pickle=True,
            )

            # setup evaluation dataloader
            self.eval_dl = self.dataset.get_eval_dataloader(
                self.hparams.loader.batch_size,
                num_workers=0,
                pin_memory=False,
            )

            # move flow back to cpu
            self.flow = self.flow.cpu()

            # clear memory
            del(mol, res_dict)
            torch.cuda.empty_cache()
            gc.collect()

        ################################################################################
        # Flow Training Era
        ################################################################################
        else:
            # move ebm to GPU device
            self.ebm = self.ebm.to(self.device)

            if self._temperature_index == 0:
                next_temp = self._temperature_ladder[self._temperature_index]
                log.log(20, f"Annealing from 1200K -> {next_temp}K ...")
            else:
                prev_temp = self._temperature_ladder[self._temperature_index - 1]
                next_temp = self._temperature_ladder[self._temperature_index]
                log.log(20, f"Annealing from {prev_temp}K -> {next_temp}K ...")

            # Instantiate full forcefield at current temperature
            forcefield = self.forcefield_partial.instantiate(temperature=next_temp)
            log.log(20, f"Forcefield Instantiated at {next_temp}K.")

            # load reference MD data and compute energies
            log.log(20, "Loading reference MD data and computing energies...")
            dof = self.dataset.molecules[self.dataset.pdb_id].n_atoms * 3
            dcd_dir  = self.dataset.data_path / "mds" / "temperature"
            mol_name = self.dataset.molecules[self.dataset.pdb_id].name
            dcd_path = list(dcd_dir.glob(f"{mol_name}_{next_temp}K*.dcd"))[0]
            ref_traj = md.load(dcd_path, top=self.dataset.pdb_path)
            ref_coords = torch.from_numpy(ref_traj.xyz).reshape(-1, dof)
            ref_energies = forcefield(ref_coords, return_force=False)
            log.log(20, "Reference energies computed.")

            # Evaluate surrogate density model (EBM)
            log.log(20, "Evaluating surrogate density model (EBM)...")
            wandb_logger = fetch_wandb_logger(self.loggers)
            log_probs, energies = eval_ebm_single_molecule(
                self.ebm,
                self.eval_dl,
                self.device,
                forcefield=forcefield,
            )
            log.log(20, "Log probabilities and energies computed.")

            # Calculate log weights
            log.log(20, "Calculating log weights...")
            log_w = calc_log_w(log_probs, energies)
            log.log(20, "Log weights computed.")

            # search for optimal quantile clipping threshold
            log.log(20, "Searching for optimal quantile clipping threshold...")
            quantiles = torch.linspace(0.9, 1.0, 11)[:-1]
            effective_sample_sizes = []
            for quantile in quantiles:
                trial_log_w = log_w[quantile_clip(log_w, quantile)]
                trial_log_w_normalized = normalize_log_w(trial_log_w)
                effective_sample_sizes.append(calc_ess(trial_log_w_normalized))
            effective_sample_sizes = torch.tensor(effective_sample_sizes)
            optimal_quantile = quantiles[effective_sample_sizes.argmax()]
            log.log(20, f"Optimal quantile clipping threshold: {optimal_quantile.item()} with ESS: {effective_sample_sizes.max().item():.4f}",)

            # clip log weights using optimal quantile
            samples = torch.cat(self.dataset.cache, dim=-1)
            samples, energies, log_w = quantile_filter(samples, energies, log_w, optimal_quantile)
            normalized_log_w = normalize_log_w(log_w)

            # plot re-weighted energy histograms
            plot_energy_histograms(
                ode=energies.numpy(),
                sim=ref_energies.numpy(),
                weights=normalized_log_w.numpy(),
                bins=100,
                xlabel=r"$U(x) / k_{B}T$" + f" ({next_temp}K)",
                figsize=(11, 9),
                prefix="media/",
                wandb_logger=wandb_logger,
            )

            # Importance weighted resampling
            samples = importance_weighted_resample(samples, normalized_log_w)

            # Reset and update dataset with the new samples
            self.dataset.clear_cache()
            self.dataset.update_dataset(
                mol_id=mol.name,
                samples=samples.chunk(samples.size(0), dim=0),
            )
            self.dataset.training_sampler = True

            # move ebm back to cpu
            self.ebm = self.ebm.cpu()

            # clear memory
            del(forcefield, ref_traj, ref_coords, ref_energies, log_probs, energies, log_w, normalized_log_w)
            torch.cuda.empty_cache()
            gc.collect()

    def on_before_batch_transfer(self, batch: BATCH, dataloader_idx: int) -> BATCH:
        if self.training_era == "ebm":
            batch = self.ebm_interpolant.plan(batch)
            if self.pipeline is not None:
                batch = self.pipeline.run_ebm(batch, is_training=True)
        else:
            batch = self.flow_interpolant.plan(batch)
            if self.pipeline is not None:
                batch = self.pipeline.run_flow(batch, is_training=True)
        return batch

    def training_step(self, batch: BATCH, batch_idx: int) -> Tensor:
        if self.training_era == "ebm":
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss_dict = self.ebm.training_step(batch)
        else:
            batch = batch.to(self.device)
            loss_dict = self.flow.training_step(batch)

        # log loss
        for key, value in loss_dict.items():
            self.log(
                f"{self.training_era}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch.batch_size,
            )

        return loss_dict["loss"]

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        self._epoch += 1
        if self.training_era == "ebm":
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