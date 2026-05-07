import torch
import torch.nn.functional as F

from einops import einsum

from lightning.pytorch.loggers import WandbLogger

import numpy as np
from typing import Optional


@torch.no_grad()
def log_loss_to_wandb(
    wandb_logger: WandbLogger,
    loss: torch.Tensor,
    step: int,
    prefix: str,
    loss_id: str,
    step_id: str,
):
    loss_value = loss.mean().item()
    wandb_logger.log(
        {
            f"{prefix}/{loss_id}": loss_value,
            f"{prefix}/{step_id}": step,
        }
    )


# Adapted from: https://github.com/jwohlwend/boltz.git
def weighted_rigid_align(
    true_coords,
    pred_coords,
    weights,
    mask,
):
    """Compute weighted alignment.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates
    pred_coords: torch.Tensor
        The predicted atom coordinates
    weights: torch.Tensor
        The weights for alignment
    mask: (Optional) torch.Tensor
        The atoms mask

    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """
    assert true_coords.dim() == 3, "Expected 3D tensor for targets of shape (B, N, 3)"
    assert pred_coords.dim() == 3, "Expected 3D tensor for perdictions of shape (B, N, 3)"
    batch_size, num_points, dim = true_coords.shape
    
    # If mask is None, assume all atoms are present (no masking)
    if mask is None:
        mask = torch.ones_like(weights)
    
    # Weights shape => (B, N). Multiply elementwise by mask => (B, N)
    # Then unsqueeze to broadcast across x, y, z => final shape (B, N, 1)
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )
    aligned_coords.detach_()

    return aligned_coords


def compute_mse_loss(
    denoised_atom_coords: torch.Tensor,
    true_atom_coords: torch.Tensor,
    sigma_loss_weights: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    batch_reduce: str = "mean",
    return_aligned_coords: bool = False,
):
    assert batch_reduce in ["mean", "sum", "none"], "Invalid batch_reduce value"
    
    align_weights = denoised_atom_coords.new_ones(denoised_atom_coords.shape[:2])
    
    # If mask is None, assume all atoms are present (no masking)
    if atom_mask is None:
        atom_mask = torch.ones_like(align_weights)
    
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            true_atom_coords.detach().float(),
            denoised_atom_coords.detach().float(),
            align_weights.detach().float(),
            mask=atom_mask.detach().float(),
        )

    # Cast back
    atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
        denoised_atom_coords
    )
    # weighted MSE loss of denoised atom positions
    mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(
        dim=-1
    )
    mse_loss = torch.sum(
        mse_loss * align_weights * atom_mask, dim=-1
    ) / torch.sum(3 * align_weights * atom_mask, dim=-1)

    # weight by sigma factor
    if batch_reduce == "mean":
        mse_loss = (mse_loss * sigma_loss_weights).mean()
    elif batch_reduce == "sum":
        mse_loss = (mse_loss * sigma_loss_weights).sum()
    elif batch_reduce == "none":
        mse_loss = mse_loss * sigma_loss_weights
    else:
        raise ValueError(f"Invalid batch_reduce value: {batch_reduce}")
    
    if return_aligned_coords:
        return mse_loss, atom_coords_aligned_ground_truth
    else:
        return mse_loss