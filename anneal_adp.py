import os
from typing import Any, Dict, List, Optional, Tuple, Union


os.environ["HYDRA_FULL_ERROR"] = "1"


import hydra
import rootutils
from omegaconf import DictConfig, OmegaConf

import torch
import lightning as L
from lightning.pytorch.loggers import Logger
from lightning import Callback, Trainer


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from aita.training.annealers import AnnealerADP
from aita.training.common import fetch_wandb_logger
from aita.utils.logging import RankedLogger
from aita.utils.configs import print_config
from aita.utils.training import (
    task_wrapper,
    instantiate_loggers,
    instantiate_callbacks,
    log_hyperparameters,
)


log = RankedLogger(__name__, on_rank_zero=True)

# register lightweight OmegaConf resolvers
OmegaConf.register_new_resolver("div", lambda a, b: float(a) / float(b))
OmegaConf.register_new_resolver("sub", lambda a, b: float(a) - float(b))

###################################
# functions
###################################

def has_tensor_cores() -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap >= (8, 0)  # Ampere+ devices expose FP32 tensor cores


def reset_trainer(cfg: DictConfig, change_max_epochs: int, **kwargs) -> Trainer:
    """Reset the trainer.
    
    :param cfg: A DictConfig configuration composed by Hydra.
    :param change_max_epochs: The number of epochs to change the max epochs to.
    :param kwargs: Additional keyword arguments to pass to the trainer.
    :return: A Trainer object.
    """
    cfg.trainer.max_epochs = change_max_epochs
    return hydra.utils.instantiate(cfg.trainer, **kwargs)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    # set float32 matmul precision to high if tensor cores are available
    if has_tensor_cores():
        torch.set_float32_matmul_precision("high")

    log.log(20, f"Instantiating AnnealerADP module...")
    model: AnnealerADP = AnnealerADP(cfg)

    log.log(20, "Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.log(20, "Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    wandb_logger = fetch_wandb_logger(logger)
    if wandb_logger is not None:
        run = wandb_logger.experiment  # creates the run if it hasn’t started yet
        if run is not None:
            log.log(20, f"W&B run name: {run.name}")

    log.log(20, f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "model": {"flow": model.flow, "ebm": model.ebm},
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.log(20, "Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        # create directory for era checkpoints
        os.makedirs(cfg.era_ckpt_dir, exist_ok=True)

        # load pre-trained EBM model
        log.log(20, "Loading pre-trained EBM model...")
        model.ebm = model.ebm.load_from_checkpoint(cfg.ebm_model_ckpt, weights_only=True, map_location="cpu")
        log.log(20, "Pre-trained EBM model loaded!")

        # NOTE: we must populate the dataset with samples from the pre-trained flow model
        log.log(20, "Populating the dataset with samples from the pre-trained flow model...")
        model = model.cuda()
        model.on_fit_start() # NOTE: this is safe as the training_era is set to "ebm" at initialization
        log.log(20, "Dataset populated with samples from the pre-trained flow model!")

        # Annealing process
        log.log(20, "Starting the annealing process!")
        ladder_length = len(cfg.temperature_ladder) * 2
        for i in range(1, ladder_length + 1):
            # Determine which model will be trained in current era
            if (i % 2 == 0):
                current_era = "ebm"
            else:
                current_era = "flow"

            # Reset the trainer
            del(trainer)
            n_epochs_finetune = cfg.get("n_epochs_finetune_ebm") if current_era == "ebm" else cfg.get("n_epochs_finetune_flow")
            trainer = reset_trainer(
                cfg,
                change_max_epochs=n_epochs_finetune[model._temperature_index],
                callbacks=callbacks,
                logger=logger,
            )

            # NOTE: Swap the models, save the EMA weights from the model being swapped out,
            #       and re-initializes EMA for the new model
            model.training_model_swap(current_era)

            # Train the newly swapped-in model
            current_temperature = cfg.temperature_ladder[model._temperature_index]
            log.log(20, f"> Training {current_era} at {current_temperature}K!")
            trainer.fit(model=model, ckpt_path=None)
            torch.cuda.empty_cache()

            # save EMA model weights
            model.save_ema_model(current_era)

            # increment temperature index for the next era
            if (i % 2 == 0):
                model.anneal_step()
    
        # NOTE: we save the very last EBM EMA model as there is no swap after the last era
        log.log(20, "Saving the very last EBM EMA model...")
        torch.save(model.ema.ema_model.state_dict(), os.path.join(cfg.era_ckpt_dir, f"ebm_last_ema_model_{current_temperature}K.pth"))
        log.log(20, "Annealing process complete!")

    return None


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # print config
    print_config(cfg)

    # train the model
    train(cfg)


if __name__ == "__main__":
    main()