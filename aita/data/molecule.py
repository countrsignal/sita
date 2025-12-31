import dgl
import torch

from dgl import DGLGraph
from torch import Tensor, empty

from rdkit import Chem
from pathlib import Path

from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field

from .features import (
    ATOM_TYPES_ENCODING, 
    parse_mol_rdkit,
    categorical_featurizer,
    build_dgl_edge_features,
    DEBUG_FEATURIZERS,
)


###################################
# functions
###################################

def fully_connected_edges(num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    nodes = torch.arange(num_nodes)
    edges = torch.cartesian_prod(nodes, nodes)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return edges[:, 0], edges[:, 1]


###################################
# Classes
###################################

@dataclass
class Molecule:
    name: str
    n_atoms: int = field(init=False, default_factory=lambda: 0)
    atom_dict: Dict[str, Any] = field(default_factory=dict)
    bond_dict: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)
    res_types: Tensor = field(default_factory=lambda: empty([]))
    atom_types: Tensor = field(default_factory=lambda: empty([]))
    atom_indices: Tensor = field(default_factory=lambda: empty([]))
    bond_types: Tensor = field(init=False, default_factory=lambda: empty([]))
    bond_orders: Tensor = field(init=False, default_factory=lambda: empty([]))

    def __post_init__(self):
        self.n_atoms = len(self.atom_dict)
        self.bond_types, self.bond_orders = build_dgl_edge_features(
            self.bond_dict, fully_connected_edges(self.n_atoms)
        )
    
    def __len__(self):
        return self.n_atoms

    def to_dgl_graph(self) -> DGLGraph:
        edges = fully_connected_edges(self.n_atoms)
        graph = dgl.graph(edges, num_nodes=len(self.atom_dict))
        graph.ndata["attr"] = torch.cat([self.res_types, self.atom_types], dim=-1)
        graph.ndata["atom_index"] = self.atom_indices
        graph.edata["attr"] = torch.cat([self.bond_types, self.bond_orders], dim=-1)
        return graph

    def inference_graph_setup(self, batch_size: int) -> DGLGraph:
        """
        Builds a DGL graph for inference time generation of multiple conformers of a single molecule.

        Args:
            batch_size: number of molecules in the batch
        Returns:
            DGL graph of the molecule with shape (batch_size * n_atoms, num_features)
        """
        src, dst = fully_connected_edges(self.n_atoms)             # edges for one molecule
        per_graph = src.numel()

        offset = torch.arange(batch_size) * self.n_atoms
        src = src.repeat(batch_size) + offset.repeat_interleave(per_graph)
        dst = dst.repeat(batch_size) + offset.repeat_interleave(per_graph)
        g = dgl.graph((src, dst), num_nodes=batch_size * self.n_atoms)

        # NOTE: dgl.graph always creates a single-graph object,
        # so batch_size defaults to 1 unless you tell DGL how many graphs you batched together
        g.set_batch_num_nodes(torch.full((batch_size,), self.n_atoms, dtype=torch.int64))
        g.set_batch_num_edges(torch.full((batch_size,), per_graph, dtype=torch.int64))
        g.ndata["attr"] = torch.cat([self.res_types, self.atom_types], dim=-1).repeat(batch_size, 1)
        g.ndata["atom_index"] = self.atom_indices.repeat(batch_size)
        g.edata["attr"] = torch.cat([self.bond_types, self.bond_orders], dim=-1).repeat(batch_size, 1)
        return g

    @staticmethod
    def from_pdb(pdb_path: str) -> "Molecule":
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        assert mol is not None, f"Failed to parse PDB file {pdb_path}"
        atom_dict, bond_dict = parse_mol_rdkit(mol)
        residue_type_one_hot, atom_one_hot = categorical_featurizer(atom_dict, ATOM_TYPES_ENCODING, return_concat=False)
        return Molecule(
            name=Path(pdb_path).stem,
            atom_dict=atom_dict,
            bond_dict=bond_dict,
            res_types=residue_type_one_hot,
            atom_types=atom_one_hot,
            atom_indices=torch.arange(len(atom_dict)),
        )

    def coords_from_pdb(self, pdb_path: str) -> torch.Tensor:
        """
        Load 3D atom coordinates from a PDB file.

        Args:
            pdb_path: path to the PDB file for this molecule

        Returns:
            Tensor of shape (n_atoms, 3) with coordinates ordered by atom index.
        """
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        assert mol is not None, f"Failed to parse PDB file {pdb_path}"

        conf = mol.GetConformer()
        num_atoms = conf.GetNumAtoms()

        # Ensure coordinate count matches the molecule definition
        if self.n_atoms != 0:
            assert num_atoms == self.n_atoms, (
                f"PDB atom count ({num_atoms}) does not match molecule atom count ({self.n_atoms})"
            )

        coords = torch.empty((num_atoms, 3), dtype=torch.float32)
        for idx in range(num_atoms):
            pos = conf.GetAtomPosition(idx)
            coords[idx] = torch.tensor([pos.x, pos.y, pos.z], dtype=torch.float32)

        return coords


@dataclass
class ADP(Molecule):

    @staticmethod
    def from_pdb(pdb_path: str) -> "ADP":
        mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
        assert mol is not None, f"Failed to parse PDB file {pdb_path}"
        atom_dict, bond_dict = parse_mol_rdkit(mol)
        residue_type_one_hot, atom_one_hot = DEBUG_FEATURIZERS["alanine_dipeptide"](return_concat=False)
        return ADP(
            name=Path(pdb_path).stem,
            atom_dict=atom_dict,
            bond_dict=bond_dict,
            res_types=residue_type_one_hot,
            atom_types=atom_one_hot,
            atom_indices=torch.arange(len(atom_dict)),
        )


###################################
# Constants
###################################

DEBUG_MOLECULES = {
    "alanine_dipeptide": ADP,
}
