import torch
import numpy as np
import mdtraj as md
from typing import Mapping, List


###################################
# constants
###################################

# Residue order is fixed to ensure deterministic encoding across peptides.
AMINO_TO_INDEX = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
    "ACE": 20,
    "NME": 21,
}


###################################
# functions
###################################


def get_adp_features(return_concat: bool = True):
    """
    Get the features for the alanine dipeptide molecule.

    Args:
        return_concat: whether to return the concatenated features or not
    """
    atom_types = torch.arange(22)
    atom_types[[1, 2, 3]] = 2
    atom_types[[19, 20, 21]] = 20
    atom_types[[11, 12, 13]] = 12
    atom_types = torch.nn.functional.one_hot(atom_types)
    residue_type = torch.arange(22)
    residue_type[:6] = 0
    residue_type[6:16] = 1
    residue_type[16:] = 2
    residue_type = torch.nn.functional.one_hot(residue_type)
    if return_concat:
        return torch.cat([residue_type, atom_types], dim=1)
    else:
        return residue_type, atom_types


def _normalise_atom_name(atom_name: str, residue_code: int) -> str:
    """Apply legacy normalisation rules for atom names.

    The original dataset collapses common hydrogen (H1/H2/H3) and oxygen (OE1,
    OE2, OD1, OD2) suffixes unless the atom belongs to a subset of aromatic
    residues. Replicating the same logic keeps the encoded features compatible
    with the existing models.
    """

    if atom_name.startswith("H") and atom_name[-1] in {"1", "2", "3"}:
        aromatic_mask = {8, 13, 17, 18}
        protected_prefixes = {"HE", "HD", "HZ", "HH"}
        if residue_code not in aromatic_mask or atom_name[:2] not in protected_prefixes:
            atom_name = atom_name[:-1]

    if atom_name.startswith("OE") or atom_name.startswith("OD"):
        atom_name = atom_name[:-1]

    return atom_name


def categorical_featurizer(
    atom_types_encoding: Mapping[str, int], topology: md.Topology, return_concat: bool = True
) -> torch.Tensor:

    atom_types: List[str] = []
    residue_types: List[int] = []

    for residue_index, residue in enumerate(topology.residues):
        residue_code = AMINO_TO_INDEX[residue.name]

        for atom in residue.atoms:
            residue_types.append(residue_code)

            atom_name = _normalise_atom_name(atom.name, residue_code)
            atom_types.append(atom_name)

    atom_type_indices = np.array(
        [atom_types_encoding[atom_type] for atom_type in atom_types]
    )

    atom_one_hot = torch.nn.functional.one_hot(
        torch.tensor(atom_type_indices, dtype=torch.long),
        num_classes=len(atom_types_encoding),
    )

    residue_type_one_hot = torch.nn.functional.one_hot(
        torch.tensor(residue_types, dtype=torch.long), num_classes=20
    )

    if return_concat:
        return torch.cat(
                [residue_type_one_hot, atom_one_hot], dim=1
            )
    else:
        return residue_type_one_hot, atom_one_hot


def feats_from_pdb(pdb_path: str, atom_types_encoding: Mapping[str, int], return_concat: bool = True) -> torch.Tensor:
    """
    Get the features for a PDB file.

    Args:
        pdb_path: path to the PDB file
        atom_types_encoding: mapping from atom types to indices
        return_concat: whether to return the concatenated features or not
    """
    topology = md.load_topology(pdb_path)
    return categorical_featurizer(atom_types_encoding, topology, return_concat=return_concat)


###################################
# constants
###################################

DEBUG_FEATURIZERS = {
    "alanine_dipeptide": get_adp_features,
}