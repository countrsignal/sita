import gc
from tqdm import tqdm
from typing import List, Optional

import torch
from lightning.pytorch.loggers import WandbLogger

from ..utils.data_utils import angstrom_to_nm
from ..energies.base_molecule_energy_function import BaseMoleculeEnergy


def fetch_wandb_logger(loggers) -> WandbLogger:
    wandb_logger = None
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            wandb_logger = logger
            break

    return wandb_logger


def eval_ebm_single_molecule(
    ebm: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    forcefield: Optional[BaseMoleculeEnergy] = None,
):
    assert len(loader.dataset.molecules) == 1, "Only one molecule evaluation is supported."

    # set model to evaluation mode
    ebm.eval();
    
    # evaluation loop
    log_probs: List[torch.Tensor] = []

    if forcefield is not None:
        energies: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating EBM"):
            # transfer batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # unpack batch
            features = batch["features"]
            samples  = batch["samples"]
            padding_mask = batch["padding_mask"]
            times = torch.ones((samples.size(0), samples.size(1), 1), device=samples.device)

            # evaluate EBM
            values = ebm(
                time=times,
                features=features,
                coordinates=samples,
                padding_mask=padding_mask,
                return_logprob=True,
                require_grad=False,
            )

            if forcefield is not None:
                dof = samples.size(1) * 3
                ff_energies = forcefield(angstrom_to_nm(samples.reshape(-1, dof)), return_force=False)
                energies.append(ff_energies.cpu().flatten())

            log_probs.append(
                values.cpu().flatten()
            )
    
    # clean up memory
    if forcefield is not None:
        del(ff_energies)
    del(batch, features, samples, padding_mask, times, values)
    torch.cuda.empty_cache()
    gc.collect()

    # return model to training mode
    ebm.train();

    # return log probabilities
    if forcefield is not None:
        return torch.cat(log_probs, dim=-1), torch.cat(energies, dim=-1)
    else:
        return torch.cat(log_probs, dim=-1)