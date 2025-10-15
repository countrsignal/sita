import gc
import dgl
import torch
import numpy as np
import mdtraj as md
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Tuple, Mapping

from ..utils.data_utils import remove_mean
from ..utils.graph_utils import fully_connected_edges


###################################
# functions
###################################


def get_adp_features():
    atom_types = torch.arange(22)
    atom_types[[1, 2, 3]] = 2
    atom_types[[19, 20, 21]] = 20
    atom_types[[11, 12, 13]] = 12
    h_initial = torch.nn.functional.one_hot(atom_types)
    return h_initial


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
    atom_types_encoding: Mapping[str, int], topology: md.Topology
) -> torch.Tensor:

    # Residue order is fixed to ensure deterministic encoding across peptides.
    amino_to_index = {
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
    }

    atom_types: List[str] = []
    residue_indices: List[int] = []
    residue_types: List[int] = []

    for residue_index, residue in enumerate(topology.residues):
        residue_code = amino_to_index[residue.name]

        for atom in residue.atoms:
            residue_indices.append(residue_index)
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
    residue_index_one_hot = torch.nn.functional.one_hot(
        torch.tensor(residue_indices, dtype=torch.long),
        num_classes=topology.n_residues,
    )
    residue_type_one_hot = torch.nn.functional.one_hot(
        torch.tensor(residue_types, dtype=torch.long), num_classes=20
    )

    features = torch.cat(
        [residue_index_one_hot, residue_type_one_hot, atom_one_hot], dim=1
    )

    return features


###################################
# Classes
###################################

class MolecularGraphDataset(dgl.data.DGLDataset):

    def __init__(self, data_path: str, param: str, anneal_type: str):

        assert anneal_type in ["alchemical", "temperature"], "Anneal type must be either 'alchemical' or 'temperature'"

        self.data_path = Path(data_path)
        self.anneal_type = anneal_type
        self.param = param
        self.molecules = [
            file.stem for file in (self.data_path / "pdbs").glob("*.pdb")
        ]
        self.atom_types_encoding = np.load(
            self.data_path / "atom_types_encoding.npy",
            allow_pickle=True,
        ).item()
        self.features, self.samples, self.backmap = self.load_data()

    def load_data(self):
        features: List[torch.Tensor] = []
        samples: List[torch.Tensor] = []
        backmap: Dict[str, int] = {} # maps from sample index to molecule index
        data_dir = self.data_path / "mds" / self.anneal_type
        for mol_idx, molecule in enumerate(self.molecules):
            # previous number of samples
            prev_num_samples = len(samples)
            # find DCD file and PDB file
            dcd_file = list(data_dir.glob(f"{molecule}_{self.param}*.dcd"))[0]
            pdb_file = self.data_path / "pdbs" / f"{molecule}.pdb"
            # load MD trajectory
            traj = md.load(dcd_file, top=pdb_file)
            # process coordinates data
            coords_tensor = torch.from_numpy(traj.xyz).float()
            coords_tensor = remove_mean(coords_tensor, traj.n_atoms, 3)
            coords_tensor = torch.chunk(coords_tensor, traj.n_frames, dim=0) # this is a list of tensors, each with shape (1, n_atoms, 3)
            # process categorical features data
            # NOTE: Alanine dipeptide is our test case molecule and we featurize it differently
            if molecule != "alanine_dipeptide":
                feats_tensor = categorical_featurizer(self.atom_types_encoding, traj.topology)
            else:
                feats_tensor = get_adp_features()
            features.append(feats_tensor)
            samples.extend(coords_tensor)
            # update index_dict
            # maps from the rolling sample index to the index of the molecule
            num_samples_enrolled = len(samples)
            backmap.update({idx: mol_idx for idx in list(range(prev_num_samples, num_samples_enrolled))})
        # clear memory
        del(traj, coords_tensor, feats_tensor, dcd_file, pdb_file, num_samples_enrolled, prev_num_samples)
        gc.collect()
        return features, samples, backmap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        h, x = self.features[self.backmap[index]], self.samples[index]
        x = x.squeeze(0)
        # create edges for fully connected graph without self-connections
        edges = fully_connected_edges(x.shape[0]) # this is a tuple of two tensors
        # create graph
        g = dgl.graph(edges, num_nodes=x.shape[0])
        g.ndata["h"] = h
        g.ndata["x"] = x
        return g
    
    def get_train_dataloader(self, batch_size: int, num_workers: int = 0, pin_memory: bool = False) -> dgl.dataloading.GraphDataLoader:
        return dgl.dataloading.GraphDataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )