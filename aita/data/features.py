import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Mapping, DefaultDict, Optional

import torch
import mdtraj as md

from rdkit import Chem
from rdkit.Chem import rdchem


###################################
# constants
###################################

# Residue order is fixed to ensure deterministic encoding across peptides.
AMINO_TO_INDEX: Dict[str, int] = {
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


ATOM_TYPES_ENCODING: Dict[str, int] = {
    'C': 0,
    'CA': 1,
    'CB': 2,
    'CD': 3,
    'CD1': 4,
    'CD2': 5,
    'CE': 6,
    'CE1': 7,
    'CE2': 8,
    'CE3': 9,
    'CG': 10,
    'CG1': 11,
    'CG2': 12,
    'CH2': 13,
    'CZ': 14,
    'CZ2': 15,
    'CZ3': 16,
    'H': 17,
    'HA': 18,
    'HB': 19,
    'HD': 20,
    'HD1': 21,
    'HD2': 22,
    'HE': 23,
    'HE1': 24,
    'HE2': 25,
    'HE3': 26,
    'HG': 27,
    'HG1': 28,
    'HG2': 29,
    'HH': 30,
    'HH1': 31,
    'HH2': 32,
    'HZ': 33,
    'HZ2': 34,
    'HZ3': 35,
    'N': 36,
    'ND1': 37,
    'ND2': 38,
    'NE': 39,
    'NE1': 40,
    'NE2': 41,
    'NH1': 42,
    'NH2': 43,
    'NZ': 44,
    'O': 45,
    'OD': 46,
    'OE': 47,
    'OG': 48,
    'OG1': 49,
    'OH': 50,
    'OXT': 51,
    'SD': 52,
    'SG': 53,
    'PADDING_INDEX': 54,
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


def parse_mol_rdkit(
    mol: Chem.Mol,
    *,
    detect_hbonds: bool = True,
    hbond_max_DA: float = 3.5,     # Å
    hbond_min_angle: float = 120.0 # degrees, angle D–H···A at H
):
    """
    Elegant, single-library (RDKit) bond analyzer.

    Returns
    -------
    per_atom : dict[int, dict[str, list[int]]]
        PDB serial -> {
            'covalent': [...],
            'peptide':  [...],        # subset of covalent C(i)-N(i+1)
            'aromatic': [...],        # aromatic bonds
            'single'|'double'|'triple': [...],  # bond orders (excl. aromatic)
            'hydrogen': [...]         # H-bonds: partner is the *other* atom (H <-> acceptor)
        }
    meta : dict[int, dict]
        Minimal atom metadata (name, element, resname, resid, chain, rd_idx, xyz).
    """
    # --- load & sanitize (assign bond orders, aromaticity, valence) ---
    if mol is None:
        raise ValueError("Provided RDKit Mol is None; cannot analyse bonds.")
    mol = Chem.Mol(mol)  # copy to avoid mutating caller's molecule
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)

    # ensure we have coordinates
    conf = mol.GetConformer()
    if not conf.Is3D():
        # PDBs are typically 3D; we only need positions, not distances to a specific unit system
        pass

    def serial_of(atom: rdchem.Atom) -> int:
        info = atom.GetPDBResidueInfo()
        return info.GetSerialNumber() if info is not None else atom.GetIdx() + 1

    def atom_meta(atom: rdchem.Atom) -> dict:
        info = atom.GetPDBResidueInfo()
        x, y, z = conf.GetAtomPosition(atom.GetIdx())
        return {
            "name": info.GetName().strip() if info else atom.GetSymbol(),
            "element": atom.GetSymbol(),
            "resname": info.GetResidueName().strip() if info else None,
            "resid": info.GetResidueNumber() if info else None,
            "chain": info.GetChainId().strip() if info else None,
            "rd_index": atom.GetIdx(),
            "xyz": (float(x), float(y), float(z)),
        }

    per_atom: DefaultDict[int, DefaultDict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    meta: Dict[int, dict] = {}

    # build metadata
    for a in mol.GetAtoms():
        s = serial_of(a)
        meta[s] = atom_meta(a)

    # --- covalent bonds + bond orders + aromaticity ---
    order_map = {
        rdchem.BondType.SINGLE: "single",
        rdchem.BondType.DOUBLE: "double",
        rdchem.BondType.TRIPLE: "triple",
        # AROMATIC is handled via bond.GetIsAromatic()
    }

    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        s1, s2 = serial_of(a1), serial_of(a2)
        # generic covalent adjacency
        per_atom[s1]["covalent"].append(s2)
        per_atom[s2]["covalent"].append(s1)

        # aromaticity
        if b.GetIsAromatic():
            per_atom[s1]["aromatic"].append(s2)
            per_atom[s2]["aromatic"].append(s1)
        else:
            # bond order class (single/double/triple)
            label = order_map.get(b.GetBondType(), None)
            if label:
                per_atom[s1][label].append(s2)
                per_atom[s2][label].append(s1)

    # --- peptide bonds (C(i) -- N(i+1) within same chain) ---
    for b in mol.GetBonds():
        a, c = b.GetBeginAtom(), b.GetEndAtom()
        ai, ci = a.GetPDBResidueInfo(), c.GetPDBResidueInfo()
        if ai is None or ci is None:
            continue

        def is_peptide_pair(x: rdchem.Atom, y: rdchem.Atom) -> bool:
            xi, yi = x.GetPDBResidueInfo(), y.GetPDBResidueInfo()
            return (
                x.GetSymbol() == "C" and y.GetSymbol() == "N"
                and xi.GetChainId() == yi.GetChainId()
                and (yi.GetResidueNumber() - xi.GetResidueNumber()) == 1
            )

        if is_peptide_pair(a, c) or is_peptide_pair(c, a):
            s1, s2 = serial_of(a), serial_of(c)
            per_atom[s1]["peptide"].append(s2)
            per_atom[s2]["peptide"].append(s1)

    # --- hydrogen bonds (geometry, using RDKit coordinates) ---
    if detect_hbonds:
        # precompute neighbors (donors: N/O/S that have at least one H; acceptors: N/O/S heavy)
        # build covalent adjacency to find attached H
        cov_adj: DefaultDict[int, List[int]] = defaultdict(list)
        for b in mol.GetBonds():
            s1 = serial_of(b.GetBeginAtom())
            s2 = serial_of(b.GetEndAtom())
            cov_adj[s1].append(s2); cov_adj[s2].append(s1)

        # donors and acceptors by serial
        donors: List[int] = []  # donor heavy atom serials
        H_by_donor: Dict[int, List[int]] = defaultdict(list)
        acceptors: List[int] = []

        for a in mol.GetAtoms():
            s = serial_of(a)
            el = a.GetSymbol().upper()
            if el in {"N", "O", "S"}:
                # hydrogens attached?
                Hs = [t for t in cov_adj.get(s, []) if meta[t]["element"].upper() == "H"]
                if Hs:
                    donors.append(s)
                    H_by_donor[s] = Hs
                # acceptor: heavy N/O/S (keep simple & general)
                acceptors.append(s)

        # quick access to positions
        def pos(serial: int):
            x, y, z = meta[serial]["xyz"]
            return x, y, z

        def dist2(p, q):
            dx, dy, dz = p[0]-q[0], p[1]-q[1], p[2]-q[2]
            return dx*dx + dy*dy + dz*dz

        def angle_deg(A, B, C):
            # angle at B for A–B–C
            v1 = (A[0]-B[0], A[1]-B[1], A[2]-B[2])
            v2 = (C[0]-B[0], C[1]-B[1], C[2]-B[2])
            n1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
            n2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
            if n1 == 0 or n2 == 0:
                return 0.0
            c = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])/(n1*n2)
            c = max(-1.0, min(1.0, c))
            return math.degrees(math.acos(c))

        DA2 = hbond_max_DA * hbond_max_DA
        acc_set = set(acceptors)
        for D in donors:
            pD = pos(D)
            for A in acc_set:
                if A == D:
                    continue
                pA = pos(A)
                if dist2(pD, pA) > DA2:
                    continue
                # check each hydrogen on donor
                for H in H_by_donor[D]:
                    if H == A:  # trivial guard
                        continue
                    ang = angle_deg(pD, pos(H), pA)
                    if ang >= hbond_min_angle:
                        # record on both ends (H <-> A)
                        per_atom[H]["hydrogen"].append(A)
                        per_atom[A]["hydrogen"].append(H)

    # tidy (sorted unique lists)
    for s in per_atom:
        for k in per_atom[s]:
            per_atom[s][k] = sorted(set(per_atom[s][k]))

    return meta, per_atom


def categorical_featurizer(
    meta: Mapping[int, Mapping[str, object]],
    atom_types_encoding: Mapping[str, int],
    return_concat: bool = True,
    serial_order: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Build the same categorical features as categorical_featurizer(), but using
    the RDKit-derived meta dictionary returned by analyze_bonds_rdkit().

    Parameters
    ----------
    atom_types_encoding : Mapping[str, int]
        Mapping from normalised atom name -> index.
    meta : Mapping[int, Mapping[str, object]]
        Per-atom metadata keyed by PDB serial. Each entry must contain:
          - 'name'   : PDB atom name (str)
          - 'resname': residue 3-letter code (str; must be in AMINO_TO_INDEX)
    return_concat : bool
        If True, returns concatenation of residue and atom one-hot features.
        If False, returns a tuple (residue_one_hot, atom_one_hot).
    serial_order : Optional[List[int]]
        If provided, specifies the exact order of PDB serials to use when
        emitting features. Otherwise atoms are ordered by ascending serial.
    """
    if not meta:
        raise ValueError("meta is empty; cannot build categorical features.")

    # Choose output ordering
    if serial_order is not None:
        serials: List[int] = list(serial_order)
    else:
        serials = sorted(meta.keys())

    atom_types: List[str] = []
    residue_types: List[int] = []

    for serial in serials:
        m = meta[serial]
        resname = m["resname"]
        if not isinstance(resname, str):
            raise ValueError(f"Invalid or missing 'resname' for serial {serial}: {resname!r}")
        residue_code = AMINO_TO_INDEX[resname]

        atom_name_raw = m["name"]
        if not isinstance(atom_name_raw, str):
            raise ValueError(f"Invalid or missing 'name' for serial {serial}: {atom_name_raw!r}")
        atom_name = _normalise_atom_name(atom_name_raw, residue_code)

        residue_types.append(residue_code)
        atom_types.append(atom_name)

    atom_type_indices = np.array([atom_types_encoding[a] for a in atom_types])

    atom_one_hot = torch.nn.functional.one_hot(
        torch.tensor(atom_type_indices, dtype=torch.long),
        num_classes=len(atom_types_encoding),
    )

    residue_type_one_hot = torch.nn.functional.one_hot(
        torch.tensor(residue_types, dtype=torch.long),
        num_classes=20,
    )

    if return_concat:
        return torch.cat([residue_type_one_hot, atom_one_hot], dim=1)
    else:
        return residue_type_one_hot, atom_one_hot


def build_dgl_edge_features(
    per_atom: Dict[int, Dict[str, List[int]]],
    edges: Tuple[torch.Tensor, torch.Tensor],
    *,
    node_serial: Optional[List[int]] = None,
):
    """
    Create one-hot edge features from per_atom/meta for provided edges.

    Parameters
    ----------
    per_atom : dict
        Maps PDB serial -> { bond_type : [partner_serials...] } as produced by analyze_bonds_rdkit().
    edges : Tuple[torch.Tensor, torch.Tensor]
        Edge endpoint node ids (shape: (E,)). Must be same length.
    """

    # Fixed category sets (kept stable across molecules)
    bond_type_labels  = ['covalent', 'peptide', 'hydrogen', 'aromatic', 'disulfide', 'salt_bridge']
    bond_order_labels = ['single', 'double', 'triple']

    # node_id -> PDB serial
    id2serial = None
    if node_serial is not None:
        if isinstance(node_serial, torch.Tensor):
            node_serial = node_serial.tolist()
        id2serial = dict(enumerate(list(node_serial)))

    # Precompute fast lookup: per serial -> key -> set(partners)
    neigh = {s: {k: set(vs) for k, vs in d.items()} for s, d in per_atom.items()}

    # Edges (ordered as provided)
    src, dst = edges
    if src.shape[0] != dst.shape[0]:
        raise ValueError("src and dst must have the same length.")
    E = src.shape[0]
    T, O = len(bond_type_labels), len(bond_order_labels)

    bt = torch.zeros((E, T), dtype=torch.float32)
    bo = torch.zeros((E, O), dtype=torch.float32)

    for eid in range(E):
        u_id = int(src[eid].item())
        v_id = int(dst[eid].item())
        su = (id2serial[u_id] if id2serial is not None else u_id + 1)
        sv = (id2serial[v_id] if id2serial is not None else v_id + 1)
        d_su = neigh.get(su, {})

        # Bond *types*
        for i, key in enumerate(bond_type_labels):
            if sv in d_su.get(key, ()):
                bt[eid, i] = 1.0

        # Bond *orders*
        for j, key in enumerate(bond_order_labels):
            if sv in d_su.get(key, ()):
                bo[eid, j] = 1.0

    return bt, bo


###################################
# constants
###################################

DEBUG_FEATURIZERS = {
    "alanine_dipeptide": get_adp_features,
}