#!/usr/bin/env python
"""Alanine dipeptide (ADP) flow-vs-MD comparison analysis.

This script is a cleaned, scripted version of ``notebooks/PITA_Comparison.ipynb``.
It loads a trained flow checkpoint, draws samples for alanine dipeptide, and
compares them against a reference molecular-dynamics (MD) trajectory across a
handful of standard metrics and figures:

    * Energy distribution (W1 / W2 + overlapping histogram).
    * Ramachandran torsion distributions (torus-W2, KLD, comparison plot).
    * TICA projection (W1 / W2 + density plots).

All numerical results are written to ``metrics.json`` and every figure is saved
as a PNG inside the chosen output directory.

Example
-------
    python analysis_adp.py \\
        --checkpoint experimental_results/anneal_adp/runs/2026-01-31_00-27-17/era_ckpts/flow_ema_model_300K.pth \\
        --n-samples 10000 \\
        --output-dir analysis_outputs/adp
"""

from __future__ import annotations

import os
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

# Use a non-interactive backend so figures render and save without a display.
import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import dgl  # noqa: F401  (imported for model/graph side effects, mirrors the notebook)
import torch
import rootutils
import numpy as np
import seaborn as sns
import mdtraj as md
import ot as pot
import lightning as L
from omegaconf import OmegaConf

# Resolve the project root and add it to the path so that ``sita`` is importable.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
PROJECT_ROOT = Path(__file__).resolve().parent

from sita.plans import TrigPlan
from sita.pipeline import Pipeline
from sita.data.molecule import ADP
from sita.utils.interactive import initialize_config, hyrda_init
from sita.energies.components.tica import run_tica, tica_features
from sita.utils.inference_utils import (
    map_adp_chirality_batch,
    adp_torsion_angles,
    calc_energy_w1,
    calc_energy_w2,
)

log = logging.getLogger("analysis_adp")

# OmegaConf resolvers used by some experiment configs (registered idempotently).
OmegaConf.register_new_resolver("div", lambda a, b: float(a) / float(b), replace=True)
OmegaConf.register_new_resolver("sub", lambda a, b: float(a) - float(b), replace=True)

MOLECULE_NAME = "alanine_dipeptide"
N_ATOMS = 22
DIM = N_ATOMS * 3

# Shared matplotlib style for a consistent, publication-friendly look.
PLOT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare flow-generated ADP samples against a reference MD trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Model / config ---
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "experimental_results/anneal_adp/runs/2026-01-31_00-27-17/era_ckpts/flow_ema_model_300K.pth",
        help="Path to the trained flow EMA checkpoint (.pth).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs",
        help="Hydra config directory.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="anneal_adp_jump_prior",
        help="Hydra experiment override used to build the interpolant/energy/flow.",
    )
    parser.add_argument(
        "--coupling-plan",
        type=str,
        default="ot",
        help="Coupling plan for the TrigPlan flow plan.",
    )

    # --- Data paths ---
    parser.add_argument(
        "--pdb-path",
        type=Path,
        default=PROJECT_ROOT / "data/debug/alanine_dipeptide.pdb",
        help="Topology PDB file for alanine dipeptide.",
    )
    parser.add_argument(
        "--ref-trajectory",
        type=Path,
        default=PROJECT_ROOT / "data/mds/temperature/alanine_dipeptide_300K_trajectory.dcd",
        help="Reference MD trajectory (.dcd) used as the ground-truth distribution.",
    )
    # --- TICA reference (optional) ---
    parser.add_argument(
        "--tica-trajectory",
        type=Path,
        default=PROJECT_ROOT / "sim_scripts/output/adp/trajectory.dcd",
        help="Trajectory used to fit the TICA model.",
    )
    parser.add_argument(
        "--tica-top",
        type=Path,
        default=PROJECT_ROOT / "sim_scripts/output/adp/system.pdb",
        help="Topology for the TICA trajectory.",
    )
    parser.add_argument("--tica-lagtime", type=int, default=100, help="TICA lag time.")
    parser.add_argument("--skip-tica", action="store_true", help="Skip the TICA analysis.")

    # --- Sampling ---
    parser.add_argument("--temperature", type=float, default=300.0, help="Energy temperature (K).")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Number of samples to generate.")
    parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=10_000,
        help="Samples per batch (must divide --n-samples).",
    )
    parser.add_argument("--n-timesteps", type=int, default=100, help="Number of ODE integration steps.")
    parser.add_argument("--prior-beta", type=float, default=1.0, help="Prior inverse-temperature beta.")
    parser.add_argument("--method", type=str, default="dopri5", help="ODE solver method.")
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=10.0,
        help="Divide generated coordinates by this factor to convert Angstrom -> nm.",
    )

    # --- Runtime ---
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (e.g. cuda:0 or cpu).")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis_outputs/adp",
        help="Directory where figures and metrics are written.",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Model / data construction
# --------------------------------------------------------------------------- #
def resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to CPU when CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_components(args: argparse.Namespace, device: torch.device):
    """Instantiate the molecule, interpolant, energy and flow model."""
    log.info("Building flow plan and molecule.")
    flow_plan = TrigPlan(coupling_plan=args.coupling_plan)
    adp = ADP.from_pdb(str(args.pdb_path))

    log.info("Composing Hydra config (experiment=%s).", args.experiment)
    cfg = initialize_config(
        config_dir=str(args.config_dir),
        overrides=[f"experiment={args.experiment}"],
    )

    log.info("Instantiating interpolant and energy (T=%.2fK).", args.temperature)
    interpolant = hyrda_init("interpolant", cfg)(plan=flow_plan)
    energy = hyrda_init("energy", cfg)(temperature=args.temperature)

    log.info("Loading flow checkpoint: %s", args.checkpoint)
    flow = hyrda_init("flow", cfg)
    flow = flow.load_from_checkpoint(
        str(args.checkpoint),
        weights_only=True,
        map_location="cpu",
    )
    flow = flow.to(device)
    flow.eval()

    return adp, interpolant, energy, flow


def load_reference(args: argparse.Namespace, energy) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the reference MD trajectory and compute its energies."""
    log.info("Loading reference trajectory: %s", args.ref_trajectory)
    ref_traj = md.load(str(args.ref_trajectory), top=str(args.pdb_path))
    ref_coords = torch.from_numpy(ref_traj.xyz).reshape(-1, DIM)
    ref_energies = -energy(ref_coords, return_force=False)
    log.info(
        "Reference: %d frames | energy range [%.3f, %.3f].",
        ref_coords.shape[0],
        ref_energies.min().item(),
        ref_energies.max().item(),
    )
    return ref_coords, ref_energies


@torch.no_grad()
def generate_samples(args: argparse.Namespace, adp, interpolant, flow) -> torch.Tensor:
    """Generate flow samples and return them flattened to (-1, DIM) in Angstrom."""
    log.info("Generating %d samples (method=%s).", args.n_samples, args.method)
    res = Pipeline.generate_from_flow(
        n_samples=args.n_samples,
        samples_per_batch=args.samples_per_batch,
        n_timesteps=args.n_timesteps,
        molecules=[adp],
        flow_model=flow,
        interpolant=interpolant,
        prior_beta=args.prior_beta,
        method=args.method,
    )
    samples = res[MOLECULE_NAME]["samples"].view(-1, DIM)
    torch.cuda.empty_cache()
    return samples


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def compute_phi_psi(coords: torch.Tensor, topology) -> np.ndarray:
    """Return stacked (phi, psi) torsions for coordinates of shape (-1, N_ATOMS, 3)."""
    traj = md.Trajectory(coords.reshape(-1, N_ATOMS, 3).cpu().numpy(), topology=topology)
    phi = md.compute_phi(traj)[1]
    psi = md.compute_psi(traj)[1]
    return np.concatenate([phi, psi], axis=1)


def torus_wasserstein(x0: np.ndarray, x1: np.ndarray) -> float:
    """Wasserstein-2 distance between two sets of angles on the torus."""
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    diff = np.abs(x0[:, None] - x1[None, :])
    dists = np.minimum(diff, 2 * np.pi - diff) ** 2
    distance_squared = pot.emd2(a, b, dists.sum(-1), numItermax=int(1e9))
    return math.sqrt(distance_squared)


def euclidean_wasserstein(x0: torch.Tensor, x1: torch.Tensor, power: int = 2) -> float:
    """Exact Wasserstein-``power`` distance between two point clouds."""
    assert power in (1, 2)
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    cost = torch.cdist(x0.reshape(x0.shape[0], -1), x1.reshape(x1.shape[0], -1))
    if power == 2:
        cost = cost**2
    value = pot.emd2(a, b, cost.detach().cpu().numpy(), numItermax=int(1e7))
    return math.sqrt(value) if power == 2 else float(value)


def torsion_kld(true_angles: np.ndarray, gen_angles: np.ndarray) -> Dict[str, float]:
    """KL divergences for phi, psi and the joint Ramachandran density."""
    eps = 1e-10
    phi_d, psi_d = true_angles[:, 0], true_angles[:, 1]
    phi_g, psi_g = gen_angles[:, 0], gen_angles[:, 1]

    def kld_1d(true_vals, gen_vals, nbins=200):
        h_true, _ = np.histogram(true_vals, nbins, range=[-np.pi, np.pi], density=True)
        h_gen, _ = np.histogram(gen_vals, nbins, range=[-np.pi, np.pi], density=True)
        return float(np.sum(h_true * np.log((h_true + eps) / (h_gen + eps))) * 2 * np.pi / nbins)

    nbins_ram = 64
    h_true = np.histogram2d(
        phi_d, psi_d, nbins_ram, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True
    )[0]
    h_gen = np.histogram2d(
        phi_g, psi_g, nbins_ram, range=[[-np.pi, np.pi], [-np.pi, np.pi]], density=True
    )[0]
    kld_ram = float(
        np.sum(h_true * np.log((h_true + eps) / (h_gen + eps))) * (2 * np.pi / nbins_ram) ** 2
    )

    return {
        "kld_phi": kld_1d(phi_d, phi_g),
        "kld_psi": kld_1d(psi_d, psi_g),
        "kld_ramachandran": kld_ram,
    }


# --------------------------------------------------------------------------- #
# Plotting helpers (self-contained, save to disk)
# --------------------------------------------------------------------------- #
def plot_energy_histogram(
    gen_energies: np.ndarray,
    ref_energies: np.ndarray,
    out_path: Path,
    x_lim: Tuple[float, float] = (-55.0, -5.0),
) -> None:
    """Overlapping energy histograms for MD vs flow samples."""
    with mpl.rc_context(rc=PLOT_RC), sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(ref_energies, bins=50, color="#99d8c9", density=True, label="MD data", edgecolor="none")
        ax.hist(
            gen_energies, bins=50, color="#4575b4", alpha=0.45, density=True, label="Flow", edgecolor="none"
        )
        ax.set_xlabel(r"$E(x)$", labelpad=10, fontsize=45)
        ax.set_ylabel("Normalized Density", labelpad=10, fontsize=45)
        ax.set_xlim(x_lim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_ramachandran_comparison(
    md_angles: np.ndarray,
    gen_angles: np.ndarray,
    out_path: Path,
    bins: int = 100,
) -> None:
    """Two-panel Ramachandran comparison: MD (left) vs flow (right)."""
    plot_range = [-np.pi, np.pi]
    edges = np.linspace(plot_range[0], plot_range[1], bins + 1)

    h_md = np.histogram2d(md_angles[:, 0], md_angles[:, 1], bins=[edges, edges])[0]
    h_nn = np.histogram2d(gen_angles[:, 0], gen_angles[:, 1], bins=[edges, edges])[0]
    h_md_masked = np.ma.masked_less(h_md, 1)
    h_nn_masked = np.ma.masked_less(h_nn, 1)

    hmax = float(np.max([h_md.max(), h_nn.max(), 1.0]))
    norm = LogNorm(vmin=1, vmax=hmax)
    ticks = np.array([np.exp(-6) * hmax, np.exp(-4) * hmax, np.exp(-2) * hmax, hmax])

    with mpl.rc_context(rc=PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(18, 9), constrained_layout=True)
        X, Y = np.meshgrid(edges, edges, indexing="ij")
        axes[0].pcolormesh(X, Y, h_md_masked, norm=norm, rasterized=True)
        im = axes[1].pcolormesh(X, Y, h_nn_masked, norm=norm, rasterized=True)

        for ax, title in zip(axes, ["MD Generated", "Flow Generated"]):
            ax.set_xlim(plot_range)
            ax.set_ylim(plot_range)
            ax.set_xlabel(r"$\varphi$", fontsize=45)
            ax.set_ylabel(r"$\psi$", fontsize=45)
            ax.set_title(title, fontsize=45)
            ax.xaxis.set_tick_params(labelsize=25)
            ax.yaxis.set_tick_params(labelsize=25)

        cbar = fig.colorbar(im, ax=axes, ticks=ticks)
        cbar.ax.set_yticklabels(np.abs(-np.log(ticks / hmax)), fontsize=25)
        cbar.ax.invert_yaxis()
        cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)


def plot_tica_density(tics: np.ndarray, lims: np.ndarray, title: str, out_path: Path) -> None:
    """2D TICA density plot for TIC0 vs TIC1."""
    fig, ax = plt.subplots()
    ax.hist2d(tics[:, 0], tics[:, 1], bins=100, norm=LogNorm(), cmap="viridis", rasterized=True)
    ax.set_xlabel("TIC0", fontsize=20)
    ax.set_ylabel("TIC1", fontsize=20)
    ax.set_xlim(lims[:, 0].min(), lims[:, 0].max())
    ax.set_ylim(lims[:, 1].min(), lims[:, 1].max())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis stages
# --------------------------------------------------------------------------- #
def energy_analysis(
    samples: torch.Tensor,
    ref_energies: torch.Tensor,
    energy,
    coord_scale: float,
    out_dir: Path,
) -> Dict[str, float]:
    """Energy W1/W2 metrics and the energy histogram figure."""
    log.info("Running energy analysis.")
    gen_energies = -energy(samples / coord_scale, return_force=False)
    gen_np = gen_energies.cpu().numpy()
    ref_np = ref_energies.cpu().numpy()

    ridx = torch.randint(0, ref_np.shape[0], size=(gen_np.shape[0],))
    ref_matched = ref_np[ridx.numpy()]

    metrics = {
        "energy_w1": float(calc_energy_w1(gen_energies=gen_np, targ_energies=ref_matched)),
        "energy_w2": float(calc_energy_w2(gen_energies=gen_np, targ_energies=ref_matched)),
        "gen_energy_min": float(gen_np.min()),
        "gen_energy_max": float(gen_np.max()),
    }
    plot_energy_histogram(gen_np, ref_np, out_dir / "energy_histogram.png")
    return metrics


def ramachandran_analysis(
    samples: torch.Tensor,
    ref_coords: torch.Tensor,
    topology,
    pdb_path: Path,
    coord_scale: float,
    out_dir: Path,
) -> Dict[str, float]:
    """Ramachandran torus-W2, torsion KLD and the comparison figure."""
    log.info("Running Ramachandran analysis.")
    gen_samples = (samples.view(-1, N_ATOMS, 3) / coord_scale).cpu().numpy()
    gen_mapped = map_adp_chirality_batch(gen_samples)

    gen_angles = compute_phi_psi(torch.from_numpy(gen_mapped), topology)

    ridx = torch.randint(0, ref_coords.shape[0], size=(gen_mapped.shape[0],))
    ref_subset = ref_coords.view(-1, N_ATOMS, 3)[ridx]
    md_angles = compute_phi_psi(ref_subset, topology)

    metrics = {"torsion_w2": torus_wasserstein(md_angles, gen_angles)}
    metrics.update(torsion_kld(md_angles, gen_angles))

    plot_ramachandran_comparison(md_angles, gen_angles, out_dir / "ramachandran_comparison.png")
    return metrics


def tica_analysis(
    samples: torch.Tensor,
    args: argparse.Namespace,
    topology,
    out_dir: Path,
) -> Dict[str, float]:
    """Fit TICA on a reference trajectory and compare projected densities."""
    log.info("Running TICA analysis (lagtime=%d).", args.tica_lagtime)
    ref_md = md.load_dcd(str(args.tica_trajectory), top=str(args.tica_top))
    tica_model = run_tica(ref_md, lagtime=args.tica_lagtime)
    ref_features = tica_features(ref_md)

    gen_samples = (samples.view(-1, N_ATOMS, 3) / args.coord_scale).cpu().numpy()
    gen_mapped = map_adp_chirality_batch(gen_samples)
    gen_traj = md.Trajectory(gen_mapped, topology=topology)
    gen_features = tica_features(gen_traj)

    n = min(args.n_samples, len(ref_features), len(gen_features))
    ridx = torch.randint(0, len(ref_features), size=(len(gen_features),)).sort().values
    ref_tics = torch.as_tensor(tica_model.transform(ref_features))[ridx, 0:2]
    gen_tics = torch.as_tensor(tica_model.transform(gen_features))[:n, 0:2]

    plot_tica_density(ref_tics.numpy(), ref_tics.numpy(), "MD", out_dir / "tica_md.png")
    plot_tica_density(gen_tics.numpy(), gen_tics.numpy(), "Flow", out_dir / "tica_flow.png")

    return {
        "tica_w1": euclidean_wasserstein(gen_tics, ref_tics, power=1),
        "tica_w2": euclidean_wasserstein(gen_tics, ref_tics, power=2),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing outputs to %s", args.output_dir)

    adp, interpolant, energy, flow = build_components(args, device)
    ref_coords, ref_energies = load_reference(args, energy)
    samples = generate_samples(args, adp, interpolant, flow)
    log.info("Generated %d samples.", samples.shape[0])

    metrics: Dict[str, float] = {}
    metrics.update(energy_analysis(samples, ref_energies, energy, args.coord_scale, args.output_dir))
    metrics.update(
        ramachandran_analysis(
            samples, ref_coords, energy.topology, args.pdb_path, args.coord_scale, args.output_dir
        )
    )

    if not args.skip_tica:
        if args.tica_trajectory.exists() and args.tica_top.exists():
            metrics.update(tica_analysis(samples, args, energy.topology, args.output_dir))
        else:
            log.warning(
                "Skipping TICA analysis: trajectory (%s) or topology (%s) not found.",
                args.tica_trajectory,
                args.tica_top,
            )

    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    log.info("Metrics: %s", json.dumps(metrics, indent=2))
    log.info("Saved metrics to %s", metrics_path)
    log.info("Analysis complete.")


if __name__ == "__main__":
    main()
