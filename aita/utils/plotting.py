from token import OP
import PIL
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import gc
import functools
from typing import Tuple, Optional

import numpy as np
import mdtraj as md

import torch
from lightning.pytorch.loggers import WandbLogger

from .inference_utils import adp_torsion_angles, map_chirality_batch, estimate_fes


def clean_up_plots(func):
    """
    Decorator that cleans up memory by removing all local variables produced
    within the function that it wraps.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            plt.close("all")
            gc.collect()
    return wrapper


def fig_to_image(fig):
    fig.canvas.draw()
    return PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


@clean_up_plots
def plot_ebm_histogram(
    tensor: torch.Tensor,
    bins: int = 50,
    color: str = '#4C72B0',
    alpha: float = 0.75,
    xlabel: str = 'Value',
    ylabel: str = 'Frequency',
    figsize=(7, 5),
    grid: bool = True,
    prefix: str = '',
    wandb_logger: Optional[WandbLogger] = None,
) -> None:
    """
    Plot a histogram from a 1D PyTorch tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N,).
        bins (int): Number of histogram bins.
        color (str): Color of the histogram bars.
        alpha (float): Transparency of the bars.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        figsize (tuple): Figure size in inches.
        grid (bool): Whether to show grid lines.
    """
    if tensor.ndim != 1:
        raise ValueError("Input tensor must be 1D.")
    
    data = tensor.numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    if wandb_logger is not None:
        ebm_histogram_fig = fig_to_image(fig)
        wandb_logger.log_image(f"{prefix}/Log Probability Histogram", [ebm_histogram_fig])
        plt.close()
    else:
        plt.show()


@clean_up_plots
def plot_energy_histograms(
    ode: np.ndarray,
    sim: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    bins: int = 50,
    color_ode="#4575b4",
    color_weighted="#02818a",
    label_sim: str = "MD data",
    label_ode: str = "ODE",
    alpha_ode: float = 0.45,
    alpha_sim: float = 1.0,
    figsize: tuple = (7, 5),
    xlabel: str = None,
    ylabel: str = "Density",
    title: str = None,
    kde: bool = False,
    x_lim: Optional[Tuple[float, float]] = None,
    prefix: str = '',
    wandb_logger: Optional[WandbLogger] = None,
) -> None:
    """
    Plot aesthetically refined overlapping histograms for two 1D tensors.

    Parameters
    ----------
    ode, sde : torch.Tensor
        1D tensors containing data to be plotted.
    bins : int, optional
        Number of bins for the histogram. Default is 50.
    color_ode, color_sde : str, optional
        Hex or named colors for the ODE and SDE distributions.
    label_ode, label_sde : str, optional
        Labels for the legend.
    alpha : float, optional
        Transparency for the histograms (0 to 1).
    figsize : tuple, optional
        Figure size in inches.
    xlabel, ylabel, title : str, optional
        Axis and figure labels.
    kde : bool, optional
        If True, overlay kernel density estimates.
    prefix: str
        Prefix for the plot.
    wandb_logger: Optional[WandbLogger] = None,
        Wandb logger object.
    """
    # Setup style
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,
    }
    with mpl.rc_context(rc=rc), sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histograms
        ax.hist(
            sim,
            bins=bins,
            color="#99d8c9",
            alpha=alpha_sim,
            density=True,
            label=label_sim,
            edgecolor='none',
        )
        ax.hist(
            ode,
            bins=bins,
            color=color_ode,
            alpha=alpha_ode,
            density=True,
            label=label_ode,
            edgecolor='none',
        )

        if weights is not None:
            ax.hist(
                ode,
                bins=bins,
                color=color_weighted,
                alpha=1.0,
                density=True,
                label="Weighted Samples",
                histtype='step',
                linewidth=8,
                weights=weights,
            )

        # Optional KDE overlay
        if kde:
            sns.kdeplot(ode, color=color_ode, ax=ax, lw=2.0, label=f"{label_ode} KDE")

        # Style axes
        ax.set_xlabel(xlabel or "Value", labelpad=10, fontsize=45)
        ax.set_ylabel(ylabel, labelpad=10, fontsize=45)

        if x_lim is not None:
            ax.set_xlim(x_lim)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=25)

        plt.tight_layout()
        if wandb_logger is not None:
            energy_histograms_fig = fig_to_image(fig)
            wandb_logger.log_image(f"{prefix}/Energy Histograms", [energy_histograms_fig])
            plt.close()
        else:
            plt.show()


@clean_up_plots
def adp_ramachandran_plot(
    samples: np.ndarray,
    pdb_path: str,
    figsize: tuple = (11, 9),
    prefix: str = '',
    wandb_logger: Optional[WandbLogger] = None,
) -> None:
    """
    Plot Ramachandran plot for ADP samples.

    Parameters
    ----------
    samples: np.ndarray
        Samples array.
    energies: np.ndarray
    pdb_path: str
        Path to pdb file.
    figsize: tuple
        Figure size.
    prefix: str
        Prefix for the plot.
    """
    if len(samples.shape) == 2:
        samples = samples.reshape(-1, 22, 3)

    angles = adp_torsion_angles(samples, pdb_path)
    plot_range = [-np.pi, np.pi]

    # Ramachandran plot #########################################################
    fig, ax = plt.subplots(figsize=figsize)

    h, x_bins, y_bins, im = ax.hist2d(angles[:,0], angles[:,1], 100, norm=LogNorm(),cmin=1,  range=[plot_range,plot_range],rasterized=True)
    ticks = np.array([np.exp(-6)*h.max(), np.exp(-4)*h.max(),np.exp(-2)*h.max(), h.max()])
    ax.set_xlabel(r"$\varphi$", fontsize=45)
    ax.set_title("Ramachandran Plot", fontsize=45)
    ax.set_ylabel(r"$\psi$", fontsize=45)
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)

    cbar = fig.colorbar(im, ticks=ticks)
    cbar.ax.set_yticklabels(np.abs(-np.log(ticks/h.max())), fontsize=25)
    cbar.ax.invert_yaxis()
    cbar.ax.set_ylabel(r"Free energy / $k_B T$", fontsize=35)
    ##############################################################################

    if wandb_logger is not None:
        ramachandran_fig = fig_to_image(fig)
        wandb_logger.log_image(f"{prefix}/Ramachandran", [ramachandran_fig])
        plt.close()
    else:
        plt.show()
    

@clean_up_plots
def adp_free_energy_profile(
    samples: np.ndarray,
    log_w: np.ndarray,
    pdb_path: str,
    gt_fes_path: str,
    figsize: tuple = (11, 9),
    prefix: str = '',
    wandb_logger: Optional[WandbLogger] = None,
) -> None:
    """
    Plot free energy profile.

        Parameters
        ----------
        wandb_logger: WandbLogger
            Wandb logger object.
        samples: np.ndarray
            Samples array.
        energies: np.ndarray
        log_w: np.ndarray
            Log weights array.
        pdb_path: str
            Path to pdb file.
        gt_fes_path: str
            Path to gt fes file.
        figsize: tuple
            Figure size.
        prefix: str
            Prefix for the plot.
    """
    if len(samples.shape) == 2:
        samples = samples.reshape(-1, 22, 3)

    samples_mapped = map_chirality_batch(samples)
    traj_samples4 = md.Trajectory(samples_mapped, topology=md.load_topology(pdb_path))

    phi = md.compute_phi(traj_samples4)[1].flatten()
    phi_right = phi.copy()
    phi_left = phi.copy()
    phi_right[phi<0] += 2*np.pi
    phi_left[phi>np.pi/2] -= 2*np.pi

    gt_fes = np.load(gt_fes_path)
    f_i_mean = gt_fes["f_i_mean"]
    f_i_std = gt_fes["f_i_std"]

    grid_left, fes_left = estimate_fes(phi_left, weights=np.exp(log_w))
    grid_right, fes_right = estimate_fes(phi_right, weights=np.exp(log_w))

    middle = 1.1
    idx_left = (grid_left>=-np.pi)&(grid_left<middle)
    grid_left = grid_left[idx_left]
    fes_left = fes_left[idx_left]
    idx_right = (grid_right<=np.pi)&(grid_right>middle)
    grid_right = grid_right[idx_right]
    fes_right = fes_right[idx_right]

    # Free energy profile #########################################################
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(gt_fes["xs"], f_i_mean, linewidth=5)
    ax.fill_between(gt_fes["xs"], f_i_mean - f_i_std, f_i_mean + f_i_std, alpha=0.2)
    ax.plot(np.hstack([grid_left, grid_right]), np.hstack([fes_left, fes_right]), linewidth=5, linestyle="--")
    ax.set_xlabel(r"$\varphi$", fontsize=45)
    ax.set_ylabel(r"Free energy / $k_B T$", fontsize=45)
    ##############################################################################

    if wandb_logger is not None:
        free_energy_profile_fig = fig_to_image(fig)
        wandb_logger.log_image(f"{prefix}/Free Energy Profile", [free_energy_profile_fig])
        plt.close()
    else:
        plt.show()

    # Free energy difference #####################################################
    left = 0.
    right = 2
    hist, edges = np.histogram(phi, bins=100, density=True,weights=np.exp(log_w))
    centers = 0.5*(edges[1:] + edges[:-1])
    centers_pos = (centers > left) & (centers < right)
    free_energy_difference = -np.log(hist[centers_pos].sum() / hist[~centers_pos].sum())
    ##############################################################################

    if wandb_logger is not None:
        wandb_logger.log_scalar(f"{prefix}/Free Energy Difference", free_energy_difference)
    else:
        print(f"{prefix}/Free Energy Difference: {free_energy_difference}")
    

