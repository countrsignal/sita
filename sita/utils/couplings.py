import dgl
import torch
from einops import einsum

from typing import Tuple
from scipy.optimize import linear_sum_assignment


def ot_coupling(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Couple the initial and target structures using optimal transport theory."""
    C = torch.cdist(x, y)
    C = C**2
    C = C / C.max()
    C = C.numpy() # we assume x and y are on CPU
    row_ind, col_ind = linear_sum_assignment(C)
    return x[row_ind], y[col_ind]


# Adapted from: https://github.com/jwohlwend/boltz.git
def bacthed_kabsch_umeyama(
    ref: torch.Tensor,
    pivot: torch.Tensor,
    mask: torch.Tensor,
):
    """Compute Kabsch-Umeyama alignment between true and predicted coordinates."""
    assert pivot.dim() == 3, "Expected 3D tensor for targets of shape (B, N, 3)"
    assert ref.dim() == 3, "Expected 3D tensor for ref of shape (B, N, 3)"
    out_shape = torch.broadcast_shapes(pivot.shape, ref.shape)
    *batch_size, num_points, dim = out_shape

    if torch.any(mask.sum(dim=-1) < (dim + 1)):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Mask shape => (B, N). Multiply elementwise by mask => (B, N)
    # Then unsqueeze to broadcast across x, y, z => final shape (B, N, 1)
    mask = mask.unsqueeze(-1)

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        mask * ref, pivot, "b n i, b n j -> b i j"
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
    ].repeat(*batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = einsum(pivot, rot_matrix, "b n i, b j i -> b n j")

    return aligned_coords.detach()