import ot
import math
import torch
import numpy as np
import mdtraj as md
from scipy.stats import gaussian_kde
from typing import Tuple, Optional


################################################################################
# Stats
################################################################################

@torch.no_grad()
def calc_log_w(energies: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    """
    Log Importance Weights

    NOTE: The unnormalized EBM density is e^E(x) so the log probability is E(x).

    Args:
        energies: torch.Tensor
            Energies of the samples.
            (batch_size, )
        log_probs: torch.Tensor
            Log probabilities of the samples under the EBM
            (batch_size, )

    Returns:
        log_w: torch.Tensor
            Log importance weights.
            (batch_size, )
    """
    return -energies - log_probs


@torch.no_grad()
def quantile_clip(log_w: torch.Tensor, quantile: float = 0.999) -> torch.Tensor:
    """
    Clip log weights beyond certain quantile.
    """
    cutoff = torch.quantile(log_w, q=quantile)
    return torch.where(log_w < cutoff)[0]


@torch.no_grad()
def quantile_filter(samples: torch.Tensor, energies: torch.Tensor, log_w: torch.Tensor, quantile: float = 0.999) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter samples and log weights beyond certain quantile.
    """
    idx = quantile_clip(log_w, quantile)
    return samples[idx], energies[idx], log_w[idx]


@torch.no_grad()
def normalize_log_w(log_w: torch.Tensor) -> torch.Tensor:
    """
    Normalize log weights to sum to 1.
    """
    return log_w - torch.logsumexp(log_w, dim=0)


@torch.no_grad()
def calc_ess(log_w_normalized: torch.Tensor) -> float:
    """Effective Sample Size"""
    ess = 1.0 / torch.sum(torch.exp(log_w_normalized)**2)
    return ess

@torch.no_grad()
def importance_weighted_resample(samples: torch.Tensor, log_w_normalized: torch.Tensor) -> torch.Tensor:
    """Importance Resampling using multinomial sampling"""
    weights = torch.exp(log_w_normalized)
    indices = torch.multinomial(weights, samples.shape[0], replacement=True)
    return samples[indices], indices


################################################################################
# Helpers
################################################################################

def determine_chirality_batch(cartesian_coords_batch):
    # Convert Cartesian coordinates to numpy array
    coords_batch = np.array(cartesian_coords_batch)

    # Check if the shape of the array is (n, 4, 3), where n is the number of chirality centers
    if coords_batch.shape[-2:] != (4, 3):
        raise ValueError("Input should be a batch of four 3D Cartesian coordinates")

    # Calculate the vectors from the chirality centers to the four connected atoms
    vectors_batch = coords_batch - coords_batch[:, 0:1, :]

    # Calculate the normal vectors of the planes formed by the three vectors for each chirality center
    normal_vectors_batch = np.cross(vectors_batch[:, 1, :], vectors_batch[:, 2, :])

    # Calculate the dot products of the normal vectors and the vectors from the chirality centers to the fourth atoms
    dot_products_batch = np.einsum('...i,...i->...', normal_vectors_batch, vectors_batch[:, 3, :])

    # Determine the chirality labels based on the signs of the dot products
    chirality_labels_batch = np.where(dot_products_batch > .000, 'L', 'D')

    return chirality_labels_batch


def map_chirality_batch(samples: np.ndarray) -> np.ndarray:
    """
    Map samples to the correct chirality.

    Parameters
    ----------
    samples: np.ndarray
        Samples array.

    Returns
    -------
    samples_mapped: np.ndarray
        Mapped samples array.
    """
    if len(samples.shape) == 2:
        samples = samples.reshape(-1, 22, 3)

    # carbon atoms
    carbon_idx = np.array([ 1, 10, 18])
    carbon_samples = samples[:, carbon_idx]
    carbon_distances = np.linalg.norm(samples[:, [8]] - carbon_samples, axis=-1)
    # likely index of c beta atom
    cb_idx = np.where(carbon_distances==carbon_distances.min(1, keepdims=True))

    back_bone_samples = samples[:, np.array([8,6,14])]
    cb_samples = samples[cb_idx[0], carbon_idx[cb_idx[1]]] [:, None, :]
    chirality = determine_chirality_batch(np.concatenate([back_bone_samples, cb_samples], axis=1))
    samples_mapped = samples.copy()
    samples_mapped[chirality=="D"] *= -1

    return samples_mapped


def adp_torsion_angles(samples: np.ndarray, pdb_path: str) -> np.ndarray:
    samples_mapped = map_chirality_batch(samples)
    traj_samples = md.Trajectory(samples_mapped, topology=md.load_topology(pdb_path))
    phi_indices, psi_indices = [4, 6, 8, 14], [6, 8, 14, 16]
    angles = md.compute_dihedrals(traj_samples, [phi_indices, psi_indices])
    return angles


def estimate_fes(samples: np.ndarray, weights: Optional[np.ndarray], kBT: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """ Free Energy Surface """
    bw_method = 0.18 # bandwidth method for the kernel density estimation
    grid = np.linspace(samples.min(), samples.max(), 100)

    if weights is not None:
        samples=samples[weights!=0]
        weights=weights[weights!=0]
        samples=samples[~np.isnan(weights)]
        weights=weights[~np.isnan(weights)]
        samples=samples[~np.isinf(weights)]
        weights=weights[~np.isinf(weights)]

    fes = -kBT * gaussian_kde(samples, bw_method, weights).logpdf(grid)
    fes -= fes.min()
    return grid, fes


def phi_to_grid(phis: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi_right = phis.copy()
    phi_left = phis.copy()
    phi_right[phis<=0] += 2*np.pi
    phi_left[phis>=np.pi/2] -= 2*np.pi       

    grid_left, fes_left = estimate_fes(phi_left, weights=weights)
    grid_right, fes_right = estimate_fes(phi_right, weights=weights)
    middle = 0
    idx_left = (grid_left>=-np.pi)&(grid_left<=middle)
    grid_left_data  = grid_left[idx_left]
    fes_left_data  = fes_left[idx_left]
    idx_right = (grid_right<=np.pi)&(grid_right>=middle)
    grid_right_data  = grid_right[idx_right]
    fes_right_data  = fes_right[idx_right]    
    return grid_left_data, fes_left_data, grid_right_data, fes_right_data


def estimate_fed(phis: np.ndarray, log_w: np.ndarray) -> float:
    """ Free Energy Difference """
    left = 0.
    right = 2
    hist, edges = np.histogram(phis, bins=100, density=True,weights=np.exp(log_w))
    centers = 0.5*(edges[1:] + edges[:-1])
    centers_pos = (centers > left) & (centers < right)
    free_energy_difference = -np.log(hist[centers_pos].sum() / hist[~centers_pos].sum())
    return free_energy_difference.item()


################################################################################
# Metrics
################################################################################

def calc_energy_w1(gen_energies: np.ndarray, targ_energies: np.ndarray):
    gen_energies = gen_energies.ravel()
    targ_energies = targ_energies.ravel()
    w1 = ot.emd2_1d(gen_energies, targ_energies, metric = "euclidean")
    return w1


def calc_energy_w2(gen_energies: np.ndarray, targ_energies: np.ndarray):
    gen_energies = gen_energies.ravel()
    targ_energies = targ_energies.ravel()
    w2 = ot.emd2_1d(gen_energies, targ_energies, metric = "sqeuclidean")
    return np.sqrt(w2).item()


def calc_torsion_w2(gen_angles: np.ndarray, holdout_angles: np.ndarray) -> float:
    """calculates OT w2 Torsion angles 

    Args:
        gen_angles: np.ndarray
            np array of sidechain angles 
        holdout_angles: np.ndarray
            np array of sidechain angles

    Returns
    -------
    w2: float
        Wasserstein distance between the two distributions
    """
    dist = np.expand_dims(gen_angles,0) - np.expand_dims(holdout_angles,1)
    dist = np.sum((dist % np.pi)**2,axis = -1)
    uniform_weights = ot.unif(gen_angles.shape[0])
    W, _ = ot.emd2(uniform_weights, uniform_weights, dist, numItermax=1e9)
    return np.sqrt(W).item()


def torus_wasserstein(gen_angles: np.ndarray, holdout_angles: np.ndarray) -> float:
    # weights:
    uniform_weights = ot.unif(gen_angles.shape[0])

    # wrapped (circular) distances:
    gen_angles = gen_angles[:, None]
    holdout_angles = holdout_angles[None, :]
    dists = np.minimum(np.abs(gen_angles - holdout_angles), 2 * np.pi - np.abs(gen_angles - holdout_angles)) ** 2

    # Compute Wasserstein distance using POT
    distance_squared = ot.emd2(uniform_weights, uniform_weights, dists.sum(-1), numItermax=int(1e9))
    return np.sqrt(distance_squared).item()