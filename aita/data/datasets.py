import gc
import dgl
import torch
import numpy as np
import mdtraj as md
from tqdm import tqdm
from pathlib import Path
from abc import ABC
from typing import Optional, List, Dict, Tuple

from ..utils.data_utils import remove_mean
from ..utils.graph_utils import fully_connected_edges
from ..data.features import get_adp_features, categorical_featurizer


###################################
# Classes
###################################

DEBUG_FEATURIZERS = {
    "alanine_dipeptide": get_adp_features,
}


class SimulationDataset(dgl.data.DGLDataset):
    """
    Dataset for molecular conformers generated from MD simulation.

    Args:
        data_path: path to the data directory
        param: parameter of the anneal type
        anneal_type: type of anneal
    
    NOTE: Data from MD simulation is in nanometers by default.
          We convert to angstroms for training in aita.interpolants.Interpolant class.
    """

    def __init__(self, data_path: str, param: str, anneal_type: str, debug_molecule: Optional[str] = None):
        super(SimulationDataset, self).__init__(name="SimulationDataset")

        assert anneal_type in ["alchemical", "temperature"], "Anneal type must be either 'alchemical' or 'temperature'"
        
        self.debug_molecule = debug_molecule
        self.data_path = Path(data_path)
        self.anneal_type = anneal_type
        self.param = param

        if self.debug_molecule is not None:
            self.molecules = [self.debug_molecule]
        else:
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
            if self.debug_molecule is None:
                feats_tensor = categorical_featurizer(self.atom_types_encoding, traj.topology, return_concat=True)
            else:
                # NOTE: Alanine dipeptide is our test case molecule and we featurize it differently
                feats_tensor = DEBUG_FEATURIZERS[self.debug_molecule](return_concat=True)
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
        # retrieve features and samples
        h, x = self.features[self.backmap[index]], self.samples[index]
        x = x.squeeze(0) # conver coordinates shape from (1, num_nodes, 3) -> (num_nodes, 3)
        # create edges for fully connected graph without self-connections
        edges = fully_connected_edges(x.shape[0]) # this is a tuple of two tensors
        # create graph
        g = dgl.graph(edges, num_nodes=x.shape[0])
        g.ndata["h"] = h
        g.ndata["x"] = x
        # create positional encoding
        g.ndata["atom_index"] = torch.arange(x.size(0))
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


class GenerativeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path: str, debug_molecule: Optional[str] = None):
        self.debug_molecule = debug_molecule
        self.data_path = Path(data_path)
        # load atom types encoding
        self.atom_types_encoding = np.load(
            self.data_path / "atom_types_encoding.npy",
            allow_pickle=True,
        ).item()
        # load molecule features
        self.molecule_features = {}
        if self.debug_molecule is None:
            pdb_dir = self.data_path / "pdbs"
            for file in pdb_dir.glob("*.pdb"):
                residue_type_one_hot, atom_one_hot = categorical_featurizer(self.atom_types_encoding, md.load_topology(file), return_concat=False)
                self.molecule_features[file.stem] = (residue_type_one_hot, atom_one_hot)
        else:
            # NOTE: ADP is our test case molecule and we featurize it differently
            residue_type_one_hot, atom_one_hot = DEBUG_FEATURIZERS[self.debug_molecule](return_concat=False)
            self.molecule_features[self.debug_molecule] = (residue_type_one_hot, atom_one_hot)

        # sample queue
        self.samples = []
        # dictionary for mapping from sample indices to molecule ids
        self.backmap = {}
    
    def update_dataset(self, mol_id: str, samples: List[torch.Tensor]) -> None:
        prev_num_samples = len(self.samples)
        # add samples to queue
        self.samples.extend(samples)
        # update backmap
        num_samples_enrolled = len(self.samples)
        self.backmap.update({idx: mol_id for idx in list(range(prev_num_samples, num_samples_enrolled))})
        # clear memory
        del(samples, num_samples_enrolled, prev_num_samples)
        gc.collect()
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        one_hots = self.molecule_features[self.backmap[index]][-1] # NOTE: we only need the atom one-hots
        samples = self.samples[index]
        # convert one-hots to tokens
        features = one_hots.argmax(dim=1).unsqueeze(0) # (1, num_nodes)
        # remove center of mass
        samples = samples - samples.mean(dim=1, keepdim=True) # (1, num_nodes, 3)
        return features, samples
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # unpack batch
        # NOTE: we expect features to have shape (1, num_nodes, num_features) and samples to have shape (1, num_nodes, 3)
        features, samples = zip(*batch)
        # pad features and samples
        padded_samples = []
        padded_features = []
        pad_value = len(self.atom_types_encoding) + 1
        max_num_nodes = max(f.size(0) for f in features)
        for f, s in zip(features, samples):
            if self.debug_molecule is None:
                # NOTE: when NOT in debug mode, we use the padding index as the number of atom types + 1!
                padded_samples.append(torch.nn.functional.pad(s, (0, 0, 0, max_num_nodes - s.size(0)), value=0))
                padded_features.append(torch.nn.functional.pad(f, (0, max_num_nodes - f.size(0)), value=pad_value))
            else:
                padded_samples.append(s)
                padded_features.append(f)
        padded_samples = torch.cat(padded_samples, dim=0)
        padded_features = torch.cat(padded_features, dim=0).long()
        padding_mask = (padded_features == pad_value).bool()
        return {"features": padded_features, "samples": padded_samples, "padding_mask": padding_mask}
    
    def get_train_dataloader(self, batch_size: int, num_workers: int = 0, pin_memory: bool = False) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )