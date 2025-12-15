import gc
import os
import numpy as np
import mdtraj as md
from tqdm import tqdm
from typing import Optional, List, Dict, Union

import hydra
from omegaconf import DictConfig

import dgl
import torch
from torch import Tensor

from lightning import LightningModule

from ..utils.logging import RankedLogger
from ..pipeline.pipeline import Pipeline
from ..models.vector_field_v2 import VFV2
from ..utils.data_utils import angstrom_to_nm
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


###################################
# Classes
###################################

class AnnealerADP(LightningModule):

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        # Passing in config expands it one level, so can accessed
        # by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)
        # Ensure that setup() is only called once
        self._is_setup = False
        # Setup at init
        log.log(20, "Setting up annealer sub-modules...")
        self.setup()

    def setup(self, stage: Optional[str] = None) -> None:
        if self._is_setup:
            return
        else:
            self._is_setup = True

        # Setup dataset and dataloader
        self.dataset: GenerativeDatasetSingleMolecule = hydra.utils.instantiate(self.hparams.dataset)
        # load pre-computed samples if path is provided
        if self.hparams.get("initial_flow_samples_path") is not None and self.hparams.initial_flow_samples_path != "":
            self.init_dataset_from_numpy(self.hparams.initial_flow_samples_path)
        log.log(20, "Dataset Initialized.")

        # setup MD forcefield
        # NOTE: We partial instantiate the forcefield as we will be instantiating a full forcefield for each temperature level
        self.forcefield_partial = hydra.utils.instantiate(self.hparams.energy)
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

        # setup placeholder for reference MD data energies
        self.ref_energies = None
        
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

    def anneal_step(self) -> None:
        """Perform one annealing step."""
        self._temperature_index += 1

    def init_dataset_from_numpy(self, file_path: str) -> None:
        """Initialize the dataset from a numpy file."""
        samples_np = np.load(file_path, allow_pickle=True)
        samples_th = torch.from_numpy(samples_np).float()
        # clear cache
        self.dataset.clear_cache()
        # update dataset
        self.dataset.update_dataset(
            mol_id=self.dataset.pdb_id,
            samples=samples_th.chunk(samples_th.size(0), dim=0),
        )
        self.dataset.training_sampler = False # NOTE: MUST BE FALSE FOR EBM TRAINING
        # clear memory
        del(samples_np, samples_th)
        gc.collect()

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
                if self._temperature_index == 0:
                    temp = 1200
                else:
                    temp = self._temperature_ladder[self._temperature_index - 1]
                torch.save(self.ema.ema_model.state_dict(), os.path.join(self.hparams.era_ckpt_dir, f"ebm_ema_model_{temp}K.pth"))
                # > re-initialize EMA
                self.ema = hydra.utils.instantiate(self.hparams.ema, model=self.flow)
        
        self.training_era = era
    
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
                temp = 1200
                log.log(20, "Generating data from the pre-trained flow model at 1200 Kelvin ...")
            else:
                temp = self._temperature_ladder[self._temperature_index]
                log.log(20, f"Generating annealed data from the flow model at {temp} Kelvin ...")

            # generate data from the flow model
            mol = self.dataset.molecules[self.dataset.pdb_id]
            res_dict = Pipeline.generate_from_flow(
                n_samples=self.hparams.generate_ebm_dataset.n_samples,
                samples_per_batch=self.hparams.generate_ebm_dataset.samples_per_batch,
                n_timesteps=self.hparams.generate_ebm_dataset.n_timesteps,
                molecules=[mol],
                flow_model=self.flow,
                interpolant=self.flow_interpolant,
                method=self.hparams.generate_ebm_dataset.method,
                tsr_params=self.hparams.generate_ebm_dataset.tsr_params,
            )
            log.log(20, "Data generation completed.")

            # Reset and update dataset with the new samples
            self.dataset.clear_cache()
            self.dataset.update_dataset(
                mol_id=mol.name,
                samples=res_dict[mol.name]["samples"].chunk(self.hparams.generate_ebm_dataset.n_samples, dim=0),
            )
            self.dataset.training_sampler = False # NOTE: MUST BE FALSE FOR EBM TRAINING

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
                curr_temp = self._temperature_ladder[self._temperature_index]
                log.log(20, f"Annealing from 1200K -> {curr_temp}K ...")
            else:
                prev_temp = self._temperature_ladder[self._temperature_index - 1]
                curr_temp = self._temperature_ladder[self._temperature_index]
                log.log(20, f"Annealing from {prev_temp}K -> {curr_temp}K ...")

            # Instantiate full forcefield at current temperature
            forcefield = self.forcefield_partial(temperature=curr_temp)
            log.log(20, f"Forcefield Instantiated at {curr_temp}K.")

            # load reference MD data and compute energies
            log.log(20, "Loading reference MD data and computing energies...")
            mol = self.dataset.molecules[self.dataset.pdb_id]
            dof = mol.n_atoms * 3
            dcd_dir  = self.dataset.data_path / "mds" / "temperature"
            dcd_path = list(dcd_dir.glob(f"{mol.name}_{curr_temp}K*.dcd"))[0]

            # DEBUG
            log.debug(f"<!>DCD path: {dcd_path}")

            ref_traj = md.load(dcd_path, top=self.dataset.pdb_path)
            ref_coords = torch.from_numpy(ref_traj.xyz).reshape(-1, dof)
            ref_energies = -forcefield(ref_coords, return_force=False)
            log.log(20, "Reference energies computed.")

            # store reference MD data
            self.ref_energies = ref_energies

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
            log_w = calc_log_w(energies=energies, log_probs=log_probs)
            log.log(20, "Log weights computed.")

            #########################################################################################################################
            # # search for optimal quantile clipping threshold
            # log.log(20, "Searching for optimal quantile clipping threshold...")
            # quantiles = torch.linspace(0.9, 1.0, 11)[:-1]
            # effective_sample_sizes = []
            # for quantile in quantiles:
            #     trial_log_w = log_w[quantile_clip(log_w=log_w, quantile=quantile)]
            #     trial_log_w_normalized = normalize_log_w(log_w=trial_log_w)
            #     effective_sample_sizes.append(calc_ess(log_w_normalized=trial_log_w_normalized))
            # effective_sample_sizes = torch.tensor(effective_sample_sizes)

            # # DEBUG
            # log.debug(f"<!>Effective sample sizes: {effective_sample_sizes}")

            # optimal_quantile = quantiles[effective_sample_sizes.argmax()]
            # ness = (effective_sample_sizes.max() / len(self.dataset))
            # log.log(20, f"Optimal quantile clipping threshold: {optimal_quantile.item()} with ESS: {ness:.4f}")
            #########################################################################################################################

            # clip log weights using optimal quantile
            samples = torch.cat(self.dataset.cache, dim=0)
            samples, energies, log_w = quantile_filter(
                samples=samples,
                energies=energies,
                log_w=log_w,
                quantile=0.99,
            )
            normalized_log_w = normalize_log_w(log_w)

            # plot re-weighted energy histograms
            plot_energy_histograms(
                ode=energies.numpy(),
                sim=ref_energies.numpy(),
                weights=normalized_log_w.numpy(),
                bins=100,
                xlabel=r"$U(x) / k_{B}T$" + f" ({curr_temp}K)",
                figsize=(11, 9),
                prefix="anneal_step",
                wandb_logger=wandb_logger,
            )

            # Importance weighted resampling
            samples, _ = importance_weighted_resample(samples, normalized_log_w)
            
            # > save samples to numpy file for debugging
            np.save(
                os.path.join(self.hparams.era_ckpt_dir, f"flow_debug_samples_{curr_temp}K.npy"),
                samples.numpy(),
                allow_pickle=True,
            )

            # Reset and update dataset with the new samples
            # NOTE: MUST CONVERT SAMPLES BACK TO NANOMETERS THE DATASET UPDATES
            #       > samples will be converted back to angstroms in `interpolant.plan()`
            self.dataset.clear_cache()
            self.dataset.update_dataset(
                mol_id=mol.name,
                samples=angstrom_to_nm(samples).chunk(samples.size(0), dim=0),
            )
            self.dataset.training_sampler = True # NOTE: MUST BE TRUE FOR FLOW TRAINING

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
            batch_size = batch["xt"].size(0)
            loss_dict = self.ebm.training_step(batch)
        else:
            batch = batch.to(self.device)
            batch_size = batch.batch_size
            loss_dict = self.flow.training_step(batch)

        # log loss
        for key, value in loss_dict.items():
            self.log(
                f"{self.training_era}/{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        return loss_dict["loss"]

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        self._epoch += 1

        ################################################################################
        # EBM Training Epoch End
        ################################################################################
        if self.training_era == "ebm":
            if self._epoch % self.hparams.ebm_inspection_interval == 0:
                wandb_logger = fetch_wandb_logger(self.loggers)
                if wandb_logger is not None:
                    # evaluate EBM on evaluation dataset
                    log_probs = eval_ebm_single_molecule(self.ebm, self.eval_dl, self.device)

                    # plot EBM histogram
                    plot_ebm_histogram(
                        log_probs,
                        prefix="inspect/ebm/",
                        wandb_logger=wandb_logger,
                    )

                    # clear memory
                    del(log_probs)
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    log.log(20, "No wandb logger found. Skipping EBM inspection.")

        ################################################################################
        # Flow Training Epoch End
        ################################################################################
        else:
            if self._epoch % self.hparams.flow_inspection_interval == 0:
                wandb_logger = fetch_wandb_logger(self.loggers)
                if wandb_logger is not None:
                    # Instantiate full forcefield at current temperature
                    current_temp = self._temperature_ladder[self._temperature_index]
                    forcefield = self.forcefield_partial(temperature=current_temp)
                    log.log(20, f"Forcefield Instantiated at {current_temp}K.")

                    # generate data from the flow model
                    log.log(20, "Generating data from the flow model...")
                    mol = self.dataset.molecules[self.dataset.pdb_id]
                    res_dict = Pipeline.generate_from_flow(
                        n_samples=self.hparams.generate_flow_samples.n_samples,
                        samples_per_batch=self.hparams.generate_flow_samples.samples_per_batch,
                        n_timesteps=self.hparams.generate_flow_samples.n_timesteps,
                        molecules=[mol],
                        flow_model=self.flow,
                        interpolant=self.flow_interpolant,
                        method=self.hparams.generate_flow_samples.method,
                        tsr_params=None,
                    )

                    # evaluate forcefield on generated samples
                    gen_samples = res_dict[mol.name]["samples"].reshape(-1, mol.n_atoms * 3)
                    gen_energies = -forcefield(angstrom_to_nm(gen_samples), return_force=False)

                    # plot energy histograms
                    plot_energy_histograms(
                        ode=gen_energies.numpy(),
                        sim=self.ref_energies.numpy(),
                        weights=None,
                        bins=100,
                        xlabel=r"$U(x) / k_{B}T$" + f" ({current_temp}K)",
                        figsize=(11, 9),
                        prefix="inspect/flow/",
                        wandb_logger=wandb_logger,
                    )

                    # clear memory
                    del(forcefield, res_dict, gen_samples, gen_energies, mol)
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    log.log(20, "No wandb logger found. Skipping Flow inspection.")

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema is not None:
            self.ema.update()