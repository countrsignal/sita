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


from aita.training.pretrain_ebm import PreTrainerEBM
from aita.training.pretrain_flow import PreTrainerFlow
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

    task_name = cfg.get("task_name")
    is_flow = task_name.endswith("flow")

    log.log(20, f"Instantiating LitBootstrap module...")
    model: Union[PreTrainerFlow, PreTrainerEBM] = PreTrainerFlow(cfg) if is_flow else PreTrainerEBM(cfg)

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
        "model": {"flow": model.flow} if is_flow else {"ebm": model.ebm},
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.log(20, "Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.log(20, "Starting training!")
        trainer.fit(model=model, ckpt_path=cfg.get("ckpt_path"))
        log.log(20, "Training complete!")
    
        # NOTE: we save both the model and the ema model
        log.log(20, "Saving model checkpoints...")
        # > save the model weights
        if is_flow:
            torch.save(model.flow.state_dict(), os.path.join(cfg.paths.output_dir, "model.pth"))
        else:
            torch.save(model.ebm.state_dict(), os.path.join(cfg.paths.output_dir, "model.pth"))
        # > save the ema model weights
        if model.ema is not None:
            torch.save(model.ema.ema_model.state_dict(), os.path.join(cfg.paths.output_dir, "ema_model.pth"))

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