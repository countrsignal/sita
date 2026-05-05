import torch

import math
import scipy
import ot as pot
import numpy as np
import mdtraj as md
from typing import Optional, Tuple

import signal
from tqdm import tqdm

import networkx as nx
import networkx.algorithms.isomorphism as iso

from .data.molecule import Molecule
from .utils.data_utils import angstrom_to_nm
from .utils.inference_utils import map_adp_chirality_batch


def as_numpy(tensor):
    """convert tensor to numpy"""
    return torch.as_tensor(tensor).detach().cpu().numpy()


def create_adjacency_list(distance_matrix, atom_types):
    adjacency_list = []

    # Iterate through the distance matrix
    num_nodes = len(distance_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid duplicate pairs
            distance = distance_matrix[i][j]
            element_i = atom_types[i]
            element_j = atom_types[j]
            if 1 in (element_i, element_j):
                distance_cutoff = 0.14
            elif 4 in (element_i, element_j):
                distance_cutoff = 0.22
            elif 0 in (element_i, element_j):
                distance_cutoff = 0.18
            else:
                # elements should not be bonded
                distance_cutoff = 0.0

            # Add edge if distance is below the cutoff
            if distance < distance_cutoff:
                adjacency_list.append([i,j])

    return adjacency_list


# chekc if chirality is the same
# if not --> mirror
# if still not --> discard
def find_chirality_centers(
    adj_list: torch.Tensor, atom_types: torch.Tensor, num_h_atoms: int = 2
) -> torch.Tensor:
    """
    Return the chirality centers for a peptide, e.g. carbon alpha atoms and their bonds.

    Args:
        adj_list: List of bonds
        atom_types: List of atom types
        num_h_atoms: If num_h_atoms or more hydrogen atoms connected to the center, it is not reportet.
            Default is 2, because in this case the mirroring is a simple permutation.

    Returns:
        chirality_centers
    """
    chirality_centers = []
    candidate_chirality_centers = torch.where(torch.unique(adj_list, return_counts=True)[1] == 4)[0]
    for center in candidate_chirality_centers:
        bond_idx, bond_pos = torch.where(adj_list == center)
        bonded_idxs = adj_list[bond_idx, (bond_pos + 1) % 2].long()
        adj_types = atom_types[bonded_idxs]
        if torch.count_nonzero(adj_types - 1) > num_h_atoms:
            chirality_centers.append([center, *bonded_idxs[:3]])
    return torch.tensor(chirality_centers).to(adj_list).long()


def compute_chirality_sign(coords: torch.Tensor, chirality_centers: torch.Tensor) -> torch.Tensor:
    """
    Compute indicator signs for a given configuration.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers

    Returns:
        Indicator signs
    """
    assert coords.dim() == 3
    # print(coords.shape, chirality_centers.shape, chirality_centers)
    direction_vectors = (
        coords[:, chirality_centers[:, 1:], :] - coords[:, chirality_centers[:, [0]], :]
    )
    perm_sign = torch.einsum(
        "ijk, ijk->ij",
        direction_vectors[:, :, 0],
        torch.cross(direction_vectors[:, :, 1], direction_vectors[:, :, 2], dim=-1),
    )
    return torch.sign(perm_sign)


def check_symmetry_change(
    coords: torch.Tensor, chirality_centers: torch.Tensor, reference_signs: torch.Tensor
) -> torch.Tensor:
    """
    Check for a batch if the chirality changed wrt to some reference reference_signs.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers
        reference_signs: List of reference sign for the chirality_centers
    Returns:
        Mask, where changes are True
    """
    perm_sign = compute_chirality_sign(coords, chirality_centers)
    return (perm_sign != reference_signs.to(coords)).any(dim=-1)


def fix_chirality(samples, adj_list, atom_types, data, dim):
    chirality_centers = find_chirality_centers(adj_list, atom_types)
    if len(chirality_centers) == 0:
        print("No chirality centers found, skipping chirality check")
        symmetry_change = np.zeros(len(samples), dtype=bool)
        return samples, symmetry_change
    reference_signs = compute_chirality_sign(torch.from_numpy(data.reshape(-1, dim//3, 3))[[1]], chirality_centers)
    symmetry_change = check_symmetry_change(torch.from_numpy(samples.reshape(-1, dim//3, 3)), chirality_centers, reference_signs)
    samples[symmetry_change] *=-1
    symmetry_change = check_symmetry_change(torch.from_numpy(samples.reshape(-1, dim//3, 3)), chirality_centers, reference_signs)
    print(f"Correct symmetry rate {(~symmetry_change).sum()/len(samples)}")
    return samples, symmetry_change


def align_topology(sample, reference, scaling, atom_types):
    sample = sample.reshape(-1, 3)
    all_dists = scipy.spatial.distance.cdist(sample, sample)
    adj_list_computed = create_adjacency_list(all_dists/scaling, atom_types)
    G_reference = nx.Graph(reference)
    G_sample = nx.Graph(adj_list_computed)
    # not same number of nodes
    if len(G_sample.nodes) != len(G_reference.nodes):
        return sample, False
    for i, atom_type in enumerate(atom_types):
        G_reference.nodes[i]['type']=atom_type
        G_sample.nodes[i]['type']=atom_type
        
    nm = iso.categorical_node_match("type", -1)
    GM = iso.GraphMatcher(G_reference, G_sample, node_match=nm)
    is_isomorphic = GM.is_isomorphic()
    # True
    GM.mapping
    initial_idx = list(GM.mapping.keys())
    final_idx = list(GM.mapping.values())
    sample[initial_idx] = sample[final_idx]
    return sample, is_isomorphic


def align_samples(samples_np, adj_list, dim, atom_types, scaling):
    def handler(signum, frame):
        raise TimeoutError("Function call took too long")

    aligned_samples = []
    aligned_idxs = []
    #for i, sample in enumerate(samples_np[(energies_np.flatten() < -52800)].reshape(-1,dim//3, 3)):
    for i, sample in tqdm(enumerate(samples_np.reshape(-1, dim//3, 3))):   
            # Set a timer for 5 seconds
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)  # Timeout set to 5 seconds

        try:
            # Call your function here
            aligned_sample, is_isomorphic = align_topology(sample, as_numpy(adj_list).tolist(), scaling, atom_types)
            if is_isomorphic:
                aligned_samples.append(aligned_sample)
                aligned_idxs.append(i)
        except TimeoutError: 
            print("Skipping iteration, function call took too long")
            continue  # Skip to the next iteration
        finally:
            signal.alarm(0)

    aligned_samples = np.array(aligned_samples)
    print(f"Correct configuration rate {len(aligned_samples)/len(samples_np)}")
    return aligned_samples, aligned_idxs


def process_generated_samples(
    mol: Molecule,
    samples: torch.Tensor,
    ref_samples: Optional[torch.Tensor] = None,
    log_weights: Optional[torch.Tensor] = None,
):
    """
    Process generated samples.

    Args:
        mol: Molecule
        samples: Generated samples
        ref_samples: Reference samples
        log_weights: Log weights of the samples
    """
    if mol.name == "alanine_dipeptide":
        return map_adp_chirality_batch(samples.numpy())
    
    # for all other molecules, we must align to reference samples
    if ref_samples is None:
        raise ValueError("Reference samples are required for molecules, EXCEPT for alanine dipeptide")
    
    topology = md.load_topology(mol.pdb_path)
    atom_dict = {"C": 0, "H":1, "N":2, "O":3, "S":4}

    atom_types = []
    for atom_name in topology.atoms:
        atom_types.append(atom_name.name[0])
    atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))
    adj_list = torch.from_numpy(np.array([(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32))

    x_ref_np = ref_samples.view(-1, mol.n_atoms, 3).numpy()
    x_np = angstrom_to_nm(samples.view(-1, mol.n_atoms, 3)).numpy()

    aligned_samples, aligned_idxs = align_samples(x_np, adj_list, mol.n_atoms * 3, atom_types, scaling=1.0)
    if log_weights is not None:
        log_weights = log_weights[aligned_idxs]

    aligned_samples, symmetry_change = fix_chirality(aligned_samples, adj_list, atom_types, x_ref_np, mol.n_atoms * 3)

    if log_weights is not None:
        return as_numpy(aligned_samples)[~symmetry_change], log_weights[~symmetry_change]
    else:
        return as_numpy(aligned_samples)[~symmetry_change]
