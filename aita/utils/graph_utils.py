import dgl
import torch
from typing import Tuple


def fully_connected_edges(num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.arange(num_nodes)
    edges = torch.cartesian_prod(nodes, nodes)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return edges[:, 0], edges[:, 1]


def get_batch_indices(g: dgl.DGLGraph) -> torch.Tensor:
    return torch.arange(g.batch_size, device=g.device).repeat_interleave(g.batch_num_nodes())


def scatter_center_mol(xyz: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    """
    Center coordinates at the origin for each molecule using torch.scatter operations.
    
    For each molecule: xyz_centered = xyz - mean(xyz)
    
    Args:
        xyz: Tensor of atomic coordinates, shape (total_atoms, 3)
        batch_index: Tensor of batch indices for each atom, shape (total_atoms,)
    
    Returns:
        xyz_centered: Coordinates centered at origin, shape (total_atoms, 3)
    """
    num_molecules = batch_index.max().item() + 1
    device = xyz.device
    
    # Expand batch_index for scatter operations on 3D coordinates
    batch_index_expanded = batch_index.unsqueeze(-1).expand(-1, 3)  # (total_atoms, 3)
    
    # Sum coordinates per molecule using scatter_add
    sum_xyz = torch.zeros((num_molecules, 3), device=device)
    sum_xyz.scatter_add_(0, batch_index_expanded, xyz)
    
    # Count atoms per molecule using scatter_add
    counts = torch.zeros(num_molecules, device=device)
    ones = torch.ones(xyz.shape[0], device=device)
    counts.scatter_add_(0, batch_index, ones)
    
    # Calculate mean coordinates for each molecule
    mean_xyz = sum_xyz / counts.unsqueeze(-1)  # (num_molecules, 3)
    
    # Gather the mean for each atom based on its batch index
    mean_per_atom = torch.gather(
        mean_xyz, 
        0, 
        batch_index_expanded
    )  # (total_atoms, 3)
    
    # Center coordinates
    xyz_centered = xyz - mean_per_atom
    
    return xyz_centered