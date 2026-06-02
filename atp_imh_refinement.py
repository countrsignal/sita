import dgl
import torch
import rootutils
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

torch.set_float32_matmul_precision('high')
rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

import lightning as L
from sita.pipeline import Pipeline
from sita.utils.interactive import (
    initialize_config,
    hyrda_init,
)
from sita.plans import TrigPlan
from sita.data.molecule import ATP
from sita.utils.logging import RankedLogger


log = RankedLogger(__name__, on_rank_zero=True)


# GLOBALS
CKPT_DIR = "/net/galaxy/home/koes/dap181/labspace/sita/checkpoints/atp/"
EBM_CKPT = f"{CKPT_DIR}/ebm_ema_model_300K.pth"
FLOW_CKPT = f"{CKPT_DIR}/flow_ema_model_300K.pth"
PDB_PATH = "/net/galaxy/home/koes/dap181/labspace/sita/data/debug/alanine_tripeptide.pdb"
CONFIG_DIR = "/net/galaxy/home/koes/dap181/labspace/sita/configs/"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def var_log_w(log_w: torch.Tensor) -> torch.Tensor:
    return torch.square(log_w).mean() - torch.square(log_w.mean())


def main(args):
    L.seed_everything(args.seed)

    # initialize stuff
    log.info(f"Initializing variables and models...")
    flow_plan = TrigPlan(coupling_plan="ot")
    atp = ATP.from_pdb(PDB_PATH)

    cfg = initialize_config(
        config_dir=CONFIG_DIR,
        overrides=["experiment=anneal_atp_jump_prior"],
    )
    interpolant = hyrda_init("interpolant", cfg)(plan=flow_plan)
    forcefield_partial = hyrda_init("energy", cfg)
    u  = forcefield_partial(temperature=300.00)

    # load models
    flow = hyrda_init("flow", cfg)
    flow = flow.load_from_checkpoint(
        f"{CKPT_DIR}/flow_ema_model_300K.pth",
        weights_only=True,
        map_location="cpu",
    )
    flow = flow.to("cuda:0")
    flow.eval();
    ebm = hyrda_init("ebm", cfg)
    ebm = ebm.load_from_checkpoint(
        f"{CKPT_DIR}/ebm_finetuned_300K.pth", # NOTE: must used the post-bootstrap, finetuned EBM!!!!
        weights_only=True,
        map_location="cpu",
    )
    ebm = ebm.to("cuda:0")
    ebm.eval();
    log.info(f"Models loaded successfully.")

    # setup IMH variables
    k_rounds = 100
    n_samples = 10_000

    # generate initial samples
    log.info(f"Generating {n_samples} initial samples for the IMH chains...")
    res = Pipeline.generate_from_flow(
        n_samples=n_samples,
        samples_per_batch=1_000,
        n_timesteps=100,
        molecules=[atp],
        flow_model=flow,
        interpolant=interpolant,
        prior_beta=1.0,
        method="dopri5",
    )
    x = res[atp.name]["samples"].view(-1, atp.n_atoms * 3)

    # evaluate initial log probabilities
    log.info(f"Evaluating initial log probabilities...")
    x_log_rho, x_log_pi = Pipeline.evaluate_log_probs(
        batch_size=500,
        mol=atp,
        samples=x,
        ebm=ebm,
        forcefield=u,
    )

    # run IMH chains
    log.info(f"Running {k_rounds} IMH rounds...")
    n_accepted  = []
    mcmc_states = [ x.clone().numpy() ]
    log_weights = [ (x_log_pi - x_log_rho).numpy() ]
    var_w = [ var_log_w(x_log_pi - x_log_rho).item() ]

    torch.cuda.empty_cache()
    for idx in tqdm(range(k_rounds), desc="IMH round"):
        # Proposal step ----------------------------------------------------
        # > draw samples from the flow
        res = Pipeline.generate_from_flow(
            n_samples=n_samples,
            samples_per_batch=1_000,
            n_timesteps=100,
            molecules=[atp],
            flow_model=flow,
            interpolant=interpolant,
            prior_beta=1.0,
            method="dopri5",
        )
        y = res[atp.name]["samples"].view(-1, atp.n_atoms * 3)

        # Accept/reject step ------------------------------------------------
        # > evaluate log probabilities
        y_log_rho, y_log_pi = Pipeline.evaluate_log_probs(
            batch_size=500,
            mol=atp,
            samples=y,
            ebm=ebm,
            forcefield=u,
        )

        # Acceptance ratio ------------------------------------------------
        log_alpha = (y_log_pi - y_log_rho) - (x_log_pi - x_log_rho)
        
        # ── Sanitise ─────────────────────────────────────────────────────
        # Map any NaN / +inf to rejection  (log_alpha = -inf => alpha = 0)
        log_alpha = torch.nan_to_num(log_alpha, nan=-torch.inf, posinf=0.0, neginf=-torch.inf)
        # Clamp: alpha <= 1  =>  log_alpha <= 0
        log_alpha = log_alpha.clamp(max=0.0)
        
        # ── Accept / reject in log space ─────────────────────────────────
        # accept iff  log(u) < log(alpha),   u ~ Uniform(0, 1)
        tiny      = torch.finfo(log_alpha.dtype).tiny          # avoid log(0)
        log_u     = torch.log(torch.rand(log_alpha.shape, dtype=log_alpha.dtype,
                                        device=log_alpha.device) + tiny)
        accept_mask = log_u < log_alpha                        # (B,)  bool
        x[accept_mask] = y[accept_mask].clone()
        x_log_rho[accept_mask] = y_log_rho[accept_mask].clone()
        x_log_pi[accept_mask] = y_log_pi[accept_mask].clone()
        log_w = x_log_pi - x_log_rho

        # record diagnostics
        n_accepted.append( sum(accept_mask) )
        mcmc_states.append( x.clone().numpy() )
        var_w.append( var_log_w(log_w).item() )
        log_weights.append( log_w.clone().numpy() )
        log.info(f"Log Acc. Rate: {log_alpha.mean().item():.4f} | N. Accepted: {sum(accept_mask)}")

    # save results
    log.info(f"Saving results...")
    filename = f"{CKPT_DIR}/atp_imh_refinement_seed_{args.seed}.pkl"
    with open(filename, "wb") as f:
        pkl.dump({
            "n_accepted": n_accepted,
            "mcmc_states": mcmc_states,
            "log_weights": log_weights,
            "var_w": var_w,
        }, f)
    log.info(f"Results saved to {filename}")
    exit(0)


if __name__ == "__main__":
    args = parse_args()
    main(args)