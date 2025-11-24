import torch
import numpy as np
import mdtraj as md

import scipy
import signal
from tqdm import tqdm
from statistics import median
from typing import Tuple

import networkx as nx
from networkx import isomorphism
import networkx.algorithms.isomorphism as iso


def atom_types_from_topology(topology: md.Topology) -> torch.Tensor:
    atom_dict = {"C": 0, "H":1, "N":2, "O":3, "S":4}
    atom_types = np.array([atom_dict[atom.name[0]] for atom in topology.atoms])
    return torch.from_numpy(atom_types)


def adjacency_list_from_topology(topology: md.Topology) -> torch.Tensor:
    bb_idx = topology.select("backbone")
    return torch.from_numpy(np.array([(b.atom1.index, b.atom2.index) for b in topology.bonds], dtype=np.int32))


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
                adjacency_list.append([i, j])

    return adjacency_list


def align_topology(sample, reference, atom_types):
    sample = sample.reshape(-1, 3)
    all_dists = scipy.spatial.distance.cdist(sample, sample)
    # sns.clustermap(all_dists)
    adj_list_computed = create_adjacency_list(all_dists, atom_types)
    G_reference = nx.Graph(reference)
    G_sample = nx.Graph(adj_list_computed)
    # not same number of nodes
    if len(G_sample.nodes) != len(G_reference.nodes):
        return sample, False
    for i, atom_type in enumerate(atom_types):
        G_reference.nodes[i]["type"] = atom_type
        G_sample.nodes[i]["type"] = atom_type

    nm = iso.categorical_node_match("type", -1)
    GM = isomorphism.GraphMatcher(G_reference, G_sample, node_match=nm)
    is_isomorphic = GM.is_isomorphic()
    initial_idx = list(GM.mapping.keys())
    final_idx = list(GM.mapping.values())
    sample[initial_idx] = sample[final_idx]
    return sample, is_isomorphic


def gather_aligned_samples(samples: torch.Tensor, adj_list: torch.Tensor, atom_types: torch.Tensor):

    def handler(signum, frame):
        raise TimeoutError("Timeout while gathering aligned samples")
    
    aligned_idx = []
    aligned_samples = []
    for i, sample in enumerate(tqdm(samples, desc="Aligning samples")):

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(5)
        try:
            aligned_sample, is_isomorphic = align_topology(sample.numpy(), adj_list.tolist(), atom_types.numpy())
        except TimeoutError:
            print(f"Timeout while aligning sample {i}")
            continue
        signal.alarm(0)

        if is_isomorphic:
            aligned_idx.append(i)
            aligned_samples.append(torch.from_numpy(aligned_sample))
    return torch.stack(aligned_samples), torch.tensor(aligned_idx)


# check if chirality is the same
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
    candidate_chirality_centers = torch.where(torch.unique(adj_list, return_counts=True)[1] == 4)[
        0
    ]
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


def rectify_chirality(
    samples: torch.Tensor,
    reference_samples: torch.Tensor,
    adj_list: torch.Tensor,
    atom_types: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rectify the chirality of a batch of samples.
    """
    chirality_centers = find_chirality_centers(adj_list, atom_types)
    reference_signs = compute_chirality_sign(reference_samples[[1]], chirality_centers)
    symmetry_change = check_symmetry_change(samples, chirality_centers, reference_signs)
    samples[symmetry_change] *= -1
    symmetry_change = check_symmetry_change(samples, chirality_centers, reference_signs)
    samples = samples[~symmetry_change]
    return samples, symmetry_change


def process_generated_samples(samples: torch.Tensor, ref_md_data: md.Trajectory):
    atom_types = atom_types_from_topology(ref_md_data.topology)
    adj_list = adjacency_list_from_topology(ref_md_data.topology)
    aligned_samples, aligned_idxs = gather_aligned_samples(samples, adj_list, atom_types)
    samples, symmetry_change = rectify_chirality(aligned_samples, ref_md_data.xyz, adj_list, atom_types)
    return samples