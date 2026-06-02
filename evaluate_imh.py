import dgl
import torch
import math
import rootutils
import ot as pot
import numpy as np
import pandas as pd
import mdtraj as md
import pickle as pkl
from functools import partial
from typing import Optional
from tqdm import tqdm
import deeptime as dt
import matplotlib.pyplot as plt
from argparse import ArgumentParser

torch.set_float32_matmul_precision('high')
rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

import lightning as L
from sita.data.molecule import ATP
from sita.utils.logging import RankedLogger
from sita.analysis import process_generated_samples
from sita.utils.interactive import (
    initialize_config,
    hyrda_init,
)
from sita.utils.inference_utils import calc_energy_w1, calc_energy_w2
from sita.utils.inference_utils import (
    quantile_filter,
    normalize_log_w,
    importance_weighted_resample,
    calc_ess,
)


log = RankedLogger(__name__, on_rank_zero=True)

# GLOBALS
CKPT_DIR = "/net/galaxy/home/koes/dap181/labspace/sita/checkpoints/atp/"
PDB_PATH = "/net/galaxy/home/koes/dap181/labspace/sita/data/debug/alanine_tripeptide.pdb"
REF_DCD_PATH = "/net/galaxy/home/koes/dap181/labspace/sita/sim_scripts/output/atp/trajectory.dcd"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mid_idx", type=int, default=50)
    return parser.parse_args()


def eval_rama_kl(samples_th: torch.Tensor, test_md: md.Trajectory) -> float:
    nbins = 200
    eps = 1e-10
    nbins_ram = 64
    eps_ram = 1e-10

    # reference samples
    psi_d = md.compute_psi(test_md)[1].reshape(-1)
    phi_d = md.compute_phi(test_md)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi_d), np.isnan(phi_d))
    not_nan = np.logical_not(is_nan)
    psi_d = psi_d[not_nan]
    phi_d = phi_d[not_nan]

    # generated samples
    sampled_traj = md.Trajectory(
        samples_th.numpy().reshape(-1, test_md.xyz.shape[1], 3),
        topology=test_md.topology,
    )
    psi = md.compute_psi(sampled_traj)[1].reshape(-1)
    phi = md.compute_phi(sampled_traj)[1].reshape(-1)
    is_nan = np.logical_or(np.isnan(psi), np.isnan(phi))
    not_nan = np.logical_not(is_nan)
    psi = psi[not_nan]
    phi = phi[not_nan]

    # compute Ramachandran KL
    hist_ram_test = np.histogram2d(
        phi_d, psi_d, nbins_ram, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True
    )[0]
    hist_ram_gen = np.histogram2d(
        phi, psi, nbins_ram, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True
    )[0]
    kld_ram = (
        np.sum(hist_ram_test * np.log((hist_ram_test + eps_ram) / (hist_ram_gen + eps_ram)))
        * (2 * np.pi / nbins_ram) ** 2
    )
    return kld_ram


def get_phi_psi_vectors(samples, topology):
    samples_np = samples.numpy()
    traj_samples = md.Trajectory(samples_np, topology=topology)
    phis = md.compute_phi(traj_samples)[1]
    psis = md.compute_psi(traj_samples)[1]
    return phis, psis


def torus_wasserstein(x0, x1):
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    # weights:
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])

    # wrapped (circular) distances:
    x0 = x0[:, None]
    x1 = x1[None, :]
    dists = np.minimum(np.abs(x0 - x1), 2 * np.pi - np.abs(x0 - x1)) ** 2

    # Compute Wasserstein distance using POT
    distance_squared = pot.emd2(a, b, dists.sum(-1), numItermax=int(1e9))
    return np.sqrt(distance_squared).item()


def ramachandran_metrics(true_samples, pred_samples, topology):
    phis_true, psis_true = get_phi_psi_vectors(true_samples, topology)
    x_true = torch.cat([torch.from_numpy(phis_true), torch.from_numpy(psis_true)], dim=1)

    phis_pred, psis_pred = get_phi_psi_vectors(pred_samples, topology)
    x_pred = torch.cat([torch.from_numpy(phis_pred), torch.from_numpy(psis_pred)], dim=1)

    return torus_wasserstein(x_true, x_pred)


def bz_wrap(array):
    return (np.sin(array), np.cos(array))


def bz_distances(xyz):
    distance_matrix_ca = np.linalg.norm(
        xyz[:, None, :, :] - xyz[:, :, None, :],
        axis=-1
    )
    n_ca = distance_matrix_ca.shape[-1]
    m, n = np.triu_indices(n_ca, k=1)
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca


def bz_tica_features(
    trajectory,
    use_dihedrals=True,
    use_distances=True,
    selection="symbol == C or symbol == N or symbol == S"
):
    trajectory = trajectory.atom_slice(trajectory.top.select(selection))
    n_atoms = trajectory.xyz.shape[1]
    if use_dihedrals:
        _, phi = md.compute_phi(
            trajectory
        )
        _, psi = md.compute_psi(
            trajectory
        )
        _, omega= md.compute_omega(
            trajectory
        )
        dihedrals = np.concatenate([*bz_wrap(phi), *bz_wrap(psi), *bz_wrap(omega)], axis=-1)
    if use_distances:
        ca_distances = bz_distances(trajectory.xyz)
    if use_distances and use_dihedrals:
        return np.concatenate([ca_distances, dihedrals], axis=-1)
    elif use_distances:
        return ca_distances
    elif use_dihedrals:
        return ca_dihedrals
    else:
        return []


def bz_run_tica(trajectory, lagtime=500, dim=40):
    ca_features = bz_tica_features(trajectory)
    tica = dt.decomposition.TICA(dim=dim, lagtime=lagtime)
    koopman_estimator = dt.covariance.KoopmanWeightingEstimator(lagtime=lagtime)
    reweighting_model = koopman_estimator.fit(ca_features).fetch_model()
    tica_model = tica.fit(ca_features, reweighting_model).fetch_model()
    return tica_model


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret


def main(args):
    L.seed_everything(args.seed)

    # load molecule
    atp = ATP.from_pdb(PDB_PATH)

    # load forcefield
    # > must instantiate from hydra config
    cfg = initialize_config(config_dir="/net/galaxy/home/koes/dap181/labspace/sita/configs/", overrides=["experiment=anneal_atp_jump_prior"])
    forcefield_partial = hyrda_init("energy", cfg)
    u = forcefield_partial(temperature=300.00)
    del(forcefield_partial)

    # load reference MD trajectory
    log.info("Loading reference MD trajectory...")
    ref_traj = md.load(REF_DCD_PATH, top=PDB_PATH)
    ref_samples = torch.from_numpy(ref_traj.xyz).reshape(-1, atp.n_atoms * 3)
    ref_energies = -u(ref_samples, return_force=False)

    # load IMH results dictionary
    log.info("Loading IMH results dictionary...")
    with open(f"{CKPT_DIR}/atp_imh_refinement_seed_{args.seed}.pkl", "rb") as f:
        results = pkl.load(f)

    # > Collect samples from start, end, and argmin{ Var(log_w) } framse of the chain
    log.info("Collecting samples from start, end, and argmin{ Var(log_w) } frames of the chain...")
    x_start = torch.from_numpy(results["mcmc_states"][0])
    x_log_w = results["log_weights"][0]

    x_mid = torch.from_numpy(results["mcmc_states"][args.mid_idx])
    x_end = torch.from_numpy(results["mcmc_states"][-1])
    min_var_idx = np.argmin(results["var_w"]).item()
    x_min_var = torch.from_numpy(results["mcmc_states"][min_var_idx])

    # > Process samples
    log.info("Processing samples...")
    y_start, y_log_w = process_generated_samples(
        atp,
        samples=x_start,
        ref_samples=ref_samples,
        log_weights=x_log_w,
    )
    y_mid = process_generated_samples(
        atp,
        samples=x_mid,
        ref_samples=ref_samples,
        log_weights=None,
    )
    y_end = process_generated_samples(
        atp,
        samples=x_end,
        ref_samples=ref_samples,
        log_weights=None,
    )
    y_min_var = process_generated_samples(
        atp,
        samples=x_min_var,
        ref_samples=ref_samples,
        log_weights=None,
    )

    # > convert to torch tensors
    y_log_w_th = torch.from_numpy(y_log_w)
    y_start_th = torch.from_numpy(y_start).reshape(-1, atp.n_atoms * 3)
    y_mid_th = torch.from_numpy(y_mid).reshape(-1, atp.n_atoms * 3)
    y_end_th = torch.from_numpy(y_end).reshape(-1, atp.n_atoms * 3)
    y_min_var_th = torch.from_numpy(y_min_var).reshape(-1, atp.n_atoms * 3)

    # > evaluate energies
    log.info("Evaluating energies...")
    energies_start = -u(y_start_th, return_force=False)
    energies_mid = -u(y_mid_th, return_force=False)
    energies_end = -u(y_end_th, return_force=False)
    energies_min_var = -u(y_min_var_th, return_force=False)

    # > create the 1-step Importance Weighted Samples
    log.info("Creating 1-step Importance Weighted Samples...")
    is_y_start_th, is_energies_start, is_y_log_w_th = quantile_filter(y_start_th, energies_start, y_log_w_th, quantile=0.99)
    is_y_log_w_norm = normalize_log_w(is_y_log_w_th)
    iwr_y_start, iwr_idx = importance_weighted_resample(n_samples=len(y_start_th), samples=y_start_th, log_w_normalized=is_y_log_w_norm)
    iwr_energies_start = is_energies_start[iwr_idx]

    # > create downsample indices for comparison with reference MD trajectory
    downsample_start = torch.randint(0, len(ref_samples), size=(len(y_start_th),))
    downsample_mid = torch.randint(0, len(ref_samples), size=(len(y_mid_th),))
    downsample_end = torch.randint(0, len(ref_samples), size=(len(y_end_th),))
    downsample_min_var = torch.randint(0, len(ref_samples), size=(len(y_min_var_th),))
    downsample_iwr = torch.randint(0, len(ref_samples), size=(len(iwr_y_start),))

    # > format the samples
    samples_dict = {
        "unadjusted": {
            "samples": y_start_th,
            "energies": energies_start,
            "downsample_indices": downsample_start,
        },
        "1-step IWR": {
            "samples": iwr_y_start,
            "energies": iwr_energies_start,
            "downsample_indices": downsample_iwr,
            "log_weights": is_y_log_w_norm,
        },
        f"mcmc step-{args.mid_idx}": {
            "samples": y_mid_th,
            "energies": energies_mid,
            "downsample_indices": downsample_mid,
        },
        "end": {
            "samples": y_end_th,
            "energies": energies_end,
            "downsample_indices": downsample_end,
        },
        "min var": {
            "samples": y_min_var_th,
            "energies": energies_min_var,
            "downsample_indices": downsample_min_var,
        },
    }

    # > one dict per sample set (keys match samples_dict); metrics stored inside
    results_metrics = {k: {} for k in samples_dict}

    # Evaluate Loop #####################################################
    for key, pkg in samples_dict.items():

        # log info
        log.info(f"Evaluating {key} samples...")

        # unpack package
        samples = pkg["samples"]
        energies = pkg["energies"]
        downsample_indices = pkg["downsample_indices"]
        if "log_weights" in pkg:
            log_weights = pkg["log_weights"]
            ess = (calc_ess(log_weights) / log_weights.size(0)).item()
        else:
            log_weights = None
            ess = 1.0
        log.info(f"Effective sample size: {ess:.4f}")

        # calculate energy wasserstein distances
        log.info("Calculating energy wasserstein distances...")
        e_w1 = calc_energy_w1(energies.numpy(), ref_energies[downsample_indices].numpy())
        e_w2 = calc_energy_w2(energies.numpy(), ref_energies[downsample_indices].numpy())
        log.info(f"Energy Wasserstein-1 distance: {e_w1:.4f}")
        log.info(f"Energy Wasserstein-2 distance: {e_w2:.4f}")

        # calculate rama-kl
        log.info("Calculating rama-kl...")
        rama_kl = eval_rama_kl(samples, ref_traj[downsample_indices.numpy()])
        log.info(f"RAMA KL: {rama_kl:.4f}")

        # calculate torus-w2
        log.info("Calculating torus-w2...")
        torus_w2 = ramachandran_metrics(
            true_samples=ref_samples[downsample_indices].reshape(-1, atp.n_atoms, 3),
            pred_samples=samples.reshape(-1, atp.n_atoms, 3),
            topology=ref_traj.topology,
        )
        log.info(f"Torus Wasserstein-2 distance: {torus_w2:.4f}")

        # calculate tica metrics
        log.info("Calculating tica metrics...")
        # > fit the tica model and project downsampled reference MD trajectory
        tica_model = bz_run_tica(ref_traj, lagtime=100, dim=atp.n_atoms * 3)
        md_features = bz_tica_features(ref_traj[downsample_indices.numpy()])
        md_tics = tica_model.transform(md_features)

        # > project the generated samples
        samples_traj = md.Trajectory(samples.reshape(-1, atp.n_atoms, 3).numpy(), topology=ref_traj.topology)
        pred_features = bz_tica_features(samples_traj)
        pred_tics = tica_model.transform(pred_features)

        # > convert to tensors
        pred_tics = torch.from_numpy(pred_tics)
        md_tics = torch.from_numpy(md_tics)

        # > calculate tica wasserstein distances
        tica_w1 = wasserstein(pred_tics, md_tics, method="exact", power=1)
        tica_w2 = wasserstein(pred_tics, md_tics, method="exact", power=2)
        log.info(f"TICA Wasserstein-1 distance: {tica_w1:.4f}")
        log.info(f"TICA Wasserstein-2 distance: {tica_w2:.4f}")

        # store results
        results_metrics[key]["energy_w1"] = e_w1
        results_metrics[key]["energy_w2"] = e_w2
        results_metrics[key]["rama_kl"] = rama_kl
        results_metrics[key]["torus-w2"] = torus_w2
        results_metrics[key]["tica_w1"] = tica_w1
        results_metrics[key]["tica_w2"] = tica_w2
        results_metrics[key]["ess"] = ess

    # rows = sample set, columns = metrics
    results_df = pd.DataFrame.from_dict(results_metrics, orient="index")
    results_df = results_df[
        ["ess", "energy_w1", "energy_w2", "rama_kl", "torus-w2", "tica_w1", "tica_w2"]
    ]
    results_df.index.name = "sample_set"
    out_csv = f"{CKPT_DIR}/imh_metrics_seed_{args.seed}.csv"
    results_df.to_csv(out_csv)
    log.info(f"Wrote metrics to {out_csv}")
    log.info("Done!")
    exit()


if __name__ == "__main__":
    args = parse_args()
    main(args)