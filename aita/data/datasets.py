import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data import Dataset, DataLoader

import gc
import json
import numpy as np
import mdtraj as md
from tqdm import tqdm
from pathlib import Path

from abc import ABC
from typing import Optional, List, Dict, Tuple, Union, Callable

from .features import ATOM_TYPES_ENCODING
from .molecule import Molecule, DEBUG_MOLECULES
from ..utils.logging import RankedLogger


log = RankedLogger(__name__, on_rank_zero=True)


###################################
# Classes
###################################

class SimulationDataset(Dataset):
    """
    Dataset for molecular conformers generated from MD simulation.

    Args:
        data_path: path to the data directory
        param: parameter of the anneal type
        anneal_type: type of anneal
    
    NOTE: Data from MD simulation is in nanometers by default.
          We convert to angstroms for training in aita.interpolants.Interpolant class.
    """

    def __init__(self, data_path: str, param: str, anneal_type: str, split_json_filename: Optional[str], debug_molecule: Optional[str] = None):
        super(SimulationDataset, self).__init__()

        assert anneal_type in ["alchemical", "temperature"], "Anneal type must be either 'alchemical' or 'temperature'"

        self.debug_molecule = debug_molecule
        self.data_path = Path(data_path)
        self.anneal_type = anneal_type
        self.param = param

        if self.debug_molecule is not None:
            # NOTE: if user provides a debug molecule, splits json uis ignored
            self.splits = None
            self.molecules = {
                debug_molecule: DEBUG_MOLECULES[self.debug_molecule].from_pdb(self.data_path / "debug" / f"{self.debug_molecule}.pdb")
            }
        else:
            self.splits = json.load(open(self.data_path / split_json_filename, "r"))
            self.molecules = {}
            for pdb_file in self.splits["train"]:
                pdb_path = self.data_path / "pdbs" / pdb_file
                self.molecules[pdb_path.stem] = Molecule.from_pdb(pdb_path)

        self.backmap, self.samples = self.load_data()

    def load_data(self):
        backmap:  Dict[str, int] = {} # maps from sample index to molecule index
        samples:  List[torch.Tensor] = []
        data_dir = self.data_path / "mds" / self.anneal_type
        for molecule in self.molecules:
            # previous number of samples
            prev_num_samples = len(samples)
            # find DCD file and PDB file
            dcd_file = list(data_dir.glob(f"{molecule}_{self.param}*.dcd"))[0]

            ############################################
            if self.debug_molecule is not None:
                log.log(20, f"Loading DCD file: {dcd_file}")
            ############################################

            if self.debug_molecule is not None:
                pdb_file = self.data_path / "debug" / f"{self.debug_molecule}.pdb"
            else:
                pdb_file = self.data_path / "pdbs" / f"{molecule}.pdb"
            # load MD trajectory
            traj = md.load(dcd_file, top=pdb_file)
            # process coordinates data
            coords_tensor = torch.from_numpy(traj.xyz).float()
            coords_tensor = coords_tensor - coords_tensor.mean(dim=1, keepdim=True)
            coords_tensor = torch.chunk(coords_tensor, traj.n_frames, dim=0) # this is a list of tensors, each with shape (1, n_atoms, 3)
            # cache data
            samples.extend(coords_tensor)
            # update index_dict
            # maps from the rolling sample index to the index of the molecule
            num_samples_enrolled = len(samples)
            backmap.update({idx: molecule for idx in list(range(prev_num_samples, num_samples_enrolled))})
        # clear memory
        del(traj, coords_tensor, dcd_file, pdb_file, num_samples_enrolled, prev_num_samples)
        gc.collect()
        return backmap, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # retrieve features and samples
        mol_id = self.backmap[index]
        # convert samples to DGL graph
        g = self.molecules[mol_id].to_dgl_graph()
        g.ndata["x1"] = self.samples[index].squeeze(0) # (num_nodes, 3)
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
    
    def eval_molecules(self, split: str) -> Dict[str, Molecule]:
        assert split != "train", "Train split is not allowed for evaluation"
        if self.debug_molecule is not None:
            return {}
        else:
            eval_molecules = {}
            for pdb_file in self.splits[split]:
                pdb_path = self.data_path / "pdbs" / pdb_file
                eval_molecules[pdb_path.stem] = Molecule.from_pdb(pdb_path)
            return eval_molecules


class GenerativeDataset(Dataset):

    def __init__(self, data_path: str, split_json_filename: Optional[str], debug_molecule: Optional[str] = None):
        super(GenerativeDataset, self).__init__()

        self.debug_molecule = debug_molecule
        self.data_path = Path(data_path)

        if self.debug_molecule is not None:
            # NOTE: if user provides a debug molecule, splits json is ignored
            self.splits = None
            self.molecules = {
                debug_molecule: DEBUG_MOLECULES[self.debug_molecule].from_pdb(self.data_path / "debug" / f"{self.debug_molecule}.pdb")
            }
        else:
            self.splits = json.load(open(self.data_path / split_json_filename, "r"))
            self.molecules = {}
            for pdb_file in self.splits["train"]:
                pdb_path = self.data_path / "pdbs" / pdb_file
                self.molecules[pdb_path.stem] = Molecule.from_pdb(pdb_path)

        # dictionary for mapping from sample indices to molecule ids
        self.backmap = {}
        # samples cache
        self._cache = []
        # property to distinguish training phases
        self._training_sampler = False
    
    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value: List[torch.Tensor]) -> None:
        self._cache = value
    
    def clear_cache(self) -> None:
        self._cache = []

    @property
    def training_sampler(self):
        return self._training_sampler

    @training_sampler.setter
    def training_sampler(self, value: bool) -> None:
        self._training_sampler = value
    
    def update_dataset(self, mol_id: str, samples: List[torch.Tensor]) -> None:
        prev_num_samples = len(self.cache)
        # add samples to cache
        self.cache.extend(samples)
        # update backmap
        num_samples_enrolled = len(self.cache)
        self.backmap.update({idx: mol_id for idx in list(range(prev_num_samples, num_samples_enrolled))})
        # clear memory
        del(samples, num_samples_enrolled, prev_num_samples)
        gc.collect()
    
    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        # convert one-hot atom types to tokens
        mol_id  = self.backmap[index]

        if self.training_sampler:
            # NOTE: when training the flow model
            g = self.molecules[mol_id].to_dgl_graph()
            g.ndata["x1"] = self.cache[index].squeeze(0) # (num_nodes, 3)
            return g
        else:
            # NOTE: when training the EBM model
            features = self.molecules[mol_id].atom_types.argmax(dim=1).unsqueeze(0) # (1, num_nodes)
            samples = self.cache[index]
            samples = samples - samples.mean(dim=1, keepdim=True) # (1, num_nodes, 3)
            return features, samples
    
    def _ebm_collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # unpack batch
        # NOTE: we expect features to have shape (1, num_nodes, num_features) and samples to have shape (1, num_nodes, 3)
        features, samples = zip(*batch)
        # pad features and samples
        padded_samples = []
        padded_features = []
        pad_value = ATOM_TYPES_ENCODING["PADDING_INDEX"]
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
    
    def get_train_dataloader(self, batch_size: int, num_workers: int = 0, pin_memory: bool = False) -> Union[DataLoader, GraphDataLoader]:
        if self.training_sampler:
            return dgl.dataloading.GraphDataLoader(
                self,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self._ebm_collate_fn,
            )

    def eval_molecules(self, split: str) -> Dict[str, Molecule]:
        assert split != "train", "Train split is not allowed for evaluation"
        if self.splits is None:
            return {}
        else:
            eval_molecules = {}
            for pdb_file in self.splits[split]:
                pdb_path = self.data_path / "pdbs" / pdb_file
                eval_molecules[pdb_path.stem] = Molecule.from_pdb(pdb_path)
            return eval_molecules


class GenerativeDatasetSingleMolecule(Dataset):

    def __init__(self, data_path: str, pdb_id: str):
        super(GenerativeDatasetSingleMolecule, self).__init__()

        self.data_path = Path(data_path)
        self.pdb_id = pdb_id

        # NOTE: for single molecule datasets, we do not use splits and we load the molecule from the pdb file
        if pdb_id in DEBUG_MOLECULES:
            self.pdb_path = self.data_path / "debug" / f"{pdb_id}.pdb"
            self.molecules = {
                pdb_id: DEBUG_MOLECULES[pdb_id].from_pdb(self.pdb_path)
            }
        else:
            self.pdb_path = self.data_path / "pdbs" / f"{pdb_id}.pdb"
            self.molecules = {
                pdb_id: Molecule.from_pdb(self.pdb_path)
            }

        # NOTE: the rest of the machanics should be the same as the GenerativeDataset class
        # dictionary for mapping from sample indices to molecule ids
        self.backmap = {}
        # samples cache
        self._cache = []
        # property to distinguish training phases
        self._training_sampler = False

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value: List[torch.Tensor]) -> None:
        self._cache = value

    def clear_cache(self) -> None:
        self._cache = []
        self.backmap = {}

    @property
    def training_sampler(self):
        return self._training_sampler

    @training_sampler.setter
    def training_sampler(self, value: bool) -> None:
        self._training_sampler = value

    def update_dataset(self, mol_id: str, samples: List[torch.Tensor]) -> None:
        prev_num_samples = len(self.cache)
        # add samples to cache
        self.cache.extend(samples)
        # update backmap
        num_samples_enrolled = len(self.cache)
        self.backmap.update({idx: mol_id for idx in list(range(prev_num_samples, num_samples_enrolled))})
        # clear memory
        del(samples, num_samples_enrolled, prev_num_samples)
        gc.collect()

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        # get the mol of interest
        mol_id  = self.backmap[index]

        if self.training_sampler:
            # NOTE: when training the flow model
            g = self.molecules[mol_id].to_dgl_graph()
            g.ndata["x1"] = self.cache[index].squeeze(0) # (num_nodes, 3)
            return g
        else:
            # NOTE: when training the EBM model
            features = self.molecules[mol_id].atom_types.argmax(dim=1).unsqueeze(0) # (1, num_nodes)
            samples = self.cache[index]
            samples = samples - samples.mean(dim=1, keepdim=True) # (1, num_nodes, 3)
            return features, samples
    
    def _ebm_collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # unpack batch
        # NOTE: we expect features to have shape (1, num_nodes, num_features) and samples to have shape (1, num_nodes, 3)
        features, samples = zip(*batch)
        # pad features and samples
        padded_samples = []
        padded_features = []
        pad_value = ATOM_TYPES_ENCODING["PADDING_INDEX"]
        max_num_nodes = max(f.size(0) for f in features)
        for f, s in zip(features, samples):
            if self.pdb_id not in DEBUG_MOLECULES:
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
    
    def get_train_dataloader(self, batch_size: int, num_workers: int = 0, pin_memory: bool = False) -> Union[DataLoader, GraphDataLoader]:
        if self.training_sampler:
            return dgl.dataloading.GraphDataLoader(
                self,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self._ebm_collate_fn,
            )

    def get_eval_dataloader(self, batch_size: int, num_workers: int = 0, pin_memory: bool = False) -> Union[DataLoader, GraphDataLoader]:
        if self.training_sampler:
            return dgl.dataloading.GraphDataLoader(
                self,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            return torch.utils.data.DataLoader(
                self,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=self._ebm_collate_fn,
            )

    @staticmethod
    def load_generated_data(numpy_path: str, data_path: str, pdb_id: str, training_sampler: bool = False) -> "GenerativeDatasetSingleMolecule":
        """
        Load generated data from a numpy file.

        Args:
            numpy_path: path to the numpy file
            data_path: path to the data directory
            pdb_id: pdb id of the molecule
            training_sampler: whether to use the training sampler
        """
        # load dataset
        dataset = GenerativeDatasetSingleMolecule(data_path=data_path, pdb_id=pdb_id)
        samples_np = np.load(numpy_path, allow_pickle=True)
        samples_th = torch.from_numpy(samples_np).float()
        # update dataset
        dataset.cache.extend(samples_th.chunk(len(samples_th), dim=0))
        dataset.backmap.update({idx: pdb_id for idx in list(range(len(samples_th)))})
        dataset.training_sampler = training_sampler
        # clear memory
        del(samples_np, samples_th)
        gc.collect()
        return dataset


class NCMCSingleMoleculeDataset(Dataset):

    def __init__(self, data_path: str, pdb_id: str):
        """
        Args:
            data_path: path to the data directory
            pdb_id: pdb id of the molecule
        """
        super(NCMCSingleMoleculeDataset, self).__init__()

        self.data_path = Path(data_path)
        self.pdb_id = pdb_id

        if pdb_id in DEBUG_MOLECULES:
            self.pdb_path = self.data_path / "debug" / f"{pdb_id}.pdb"
            self.molecules = {
                pdb_id: DEBUG_MOLECULES[pdb_id].from_pdb(self.pdb_path)
            }
        else:
            self.pdb_path = self.data_path / "pdbs" / f"{pdb_id}.pdb"
            self.molecules = {
                pdb_id: Molecule.from_pdb(self.pdb_path)
            }

        self.backmap = {}
        self._samples: List[torch.Tensor] = []
        self._energies: List[torch.Tensor] = []

    @property
    def samples(self) -> List[torch.Tensor]:
        return self._samples

    @samples.setter
    def samples(self, value: List[torch.Tensor]) -> None:
        self._samples = value

    @property
    def energies(self) -> List[torch.Tensor]:
        return self._energies

    @energies.setter
    def energies(self, value: List[torch.Tensor]) -> None:
        self._energies = value

    def clear_cache(self) -> None:
        self._samples = []
        self._energies = []
        self.backmap = {}
    
    def materialize_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        all_samples = torch.cat(self._samples, dim=0)
        all_energies = torch.cat(self._energies, dim=0)
        return all_samples, all_energies

    def update_dataset(self, mol_id: str, samples: List[torch.Tensor], energies: List[torch.Tensor]) -> None:
        assert len(samples) == len(energies), "Number of samples and energies must be the same"

        prev_num_samples = len(self._samples)

        self._samples.extend(samples)
        self._energies.extend(energies)

        num_samples_enrolled = len(self._samples)
        self.backmap.update({idx: mol_id for idx in range(prev_num_samples, num_samples_enrolled)})

        del(num_samples_enrolled, prev_num_samples)
        gc.collect()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        mol_id = self.backmap[index]
        g = self.molecules[mol_id].to_dgl_graph()
        g.ndata["x1"] = self._samples[index].view(-1, 3) # (num_nodes, 3)
        return g

    def get_train_dataloader(self, batch_size: int, num_workers: int = 0, pin_memory: bool = False) -> Union[DataLoader, GraphDataLoader]:
        return dgl.dataloading.GraphDataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def get_eval_dataloader(self, batch_size: int, num_workers: int = 0, pin_memory: bool = False) -> Union[DataLoader, GraphDataLoader]:
        return dgl.dataloading.GraphDataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def save_samples(self, save_dir: str, filename: str) -> None:
        """Materialize current samples and save to a numpy file."""
        all_samples, _ = self.materialize_tensors()
        np.save(
            str(Path(save_dir) / filename),
            all_samples.numpy(),
            allow_pickle=True,
        )
        del(all_samples, _)

    @staticmethod
    def init_from_dcd(
        dcd_path: str,
        data_path: str,
        pdb_id: str,
        subsample_size: int,
        forcefield: Callable,
    ) -> "NCMCSingleMoleculeDataset":

        # check if debug molecule
        if pdb_id in DEBUG_MOLECULES:
            path_to_pdb = Path(data_path) / "debug" / f"{pdb_id}.pdb"
        else:
            path_to_pdb = Path(data_path) / "pdbs" / f"{pdb_id}.pdb"

        # load trajectory
        traj = md.load(dcd_path, top=path_to_pdb)
        samples_th = torch.from_numpy(traj.xyz).float().view(-1, forcefield.n_particles * 3)

        # subsample if requested
        if subsample_size is not None:
            random_indices = torch.randint(0, samples_th.size(0), (subsample_size,))
            samples_th = samples_th[random_indices]

        # compute energies
        energies = -forcefield(samples_th, return_force=False)

        # initialize dataset
        dataset = NCMCSingleMoleculeDataset(data_path=data_path, pdb_id=pdb_id)
        dataset.update_dataset(
            mol_id=pdb_id,
            samples=samples_th.chunk(samples_th.size(0), dim=0),
            energies=energies.chunk(energies.size(0), dim=0),
        )
        del(traj, samples_th, energies)
        gc.collect()
        return dataset