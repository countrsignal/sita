from lightning.pytorch.loggers import WandbLogger



def fetch_wandb_logger(loggers) -> WandbLogger:
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger