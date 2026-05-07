import re
import py3Dmol
import numpy as np
from typing import List, Optional, Any

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig


###################################
# functions
###################################

# --- configuration  and initialization ---
def initialize_config(
    config_dir: str,
    config_name: str = "config.yaml",
    overrides: List[str] = [],
) -> DictConfig:
    """Initialize the configuration."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name=config_name, overrides=overrides)
    return config


def hyrda_init(key: str, config: DictConfig, *args, **kwargs) -> Any:
    return hydra.utils.instantiate(config[key], *args, **kwargs)


# --- visualization ---
def view_mol_3d(coords: np.ndarray,
                      pdb_path: str,
                      *,
                      sample_idx: Optional[int] = None,
                      show_h: bool = False,         # hydrogens hidden by default
                      width: int = 800,
                      height: int = 600,
                      background: str = 'white',
                      stick_radius: float = 0.18,
                      animate: bool = True):
    """
    Visualize coordinates on a PDB topology with clean bonds/colors.

    - Writes explicit CONECT bonds (H has max 1 bond; no H–H).
    - Uses a muted element palette and hides hydrogens by default.

    Args:
        coords: coordinates with accepted shape (B, N*3), (1, N*3), or (N*3)
        pdb_path: path to the PDB file
        sample_idx: index of the sample to plot.
                    Use when coords is a batch of samples (B, N*3).
                    Default is None.
        show_h: show hydrogens
                Default is False.
        width: width of the viewer
                Default is 800.
        height: height of the viewer
                Default is 600.
        background: background color
                Default is 'white'.
        stick_radius: radius of the sticks
                Default is 0.18.
        animate: animate the viewer
                Default is True.
        stick_radius: radius of the sticks
        animate: animate the viewer

    Returns:
        v: py3Dmol.view object
    """

    # check for batch dimension
    # NOTE: while the function accepts batch of coordinates,
    #       we are only capable of plotting one sample at a time
    if coords.ndim == 1:
        coords = coords[np.newaxis, :]

    # check for sample index
    # NOTE: if sample_idx is not None, we will plot the sample at the given index
    #       otherwise, we will plot the first sample
    if sample_idx is not None:
        coords = coords[sample_idx:sample_idx+1, :]  # keep 2D
    B, threeN = coords.shape


    # --- parse topology atoms (order defines atom indexing) ---
    atoms = []
    with open(pdb_path, 'r') as fh:
        for ln in fh:
            if ln.startswith(('ATOM', 'HETATM')):
                serial = int(ln[6:11])
                name = ln[12:16].strip()
                elem = (ln[76:78].strip() or re.sub('[^A-Za-z]', '', name)[:2]).capitalize()
                elem = 'H' if elem.startswith('H') else elem  # normalize
                atoms.append({'line': ln, 'serial': serial, 'elem': elem})
    n = len(atoms)

    # --- element radii for bond inference (Å; slightly generous) ---
    r = {
        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05, 'P': 1.07,
        'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
    }
    def covrad(e): return r.get(e, 0.8)

    # --- build bonds once from first frame, with strict H rules ---
    xyz0 = coords[0].reshape(n, 3)
    d = np.linalg.norm(xyz0[:, None, :] - xyz0[None, :, :], axis=-1)
    pairs = []
    # initial distance screen (no self, no too-close)
    for i in range(n):
        ei = atoms[i]['elem']
        for j in range(i + 1, n):
            ej = atoms[j]['elem']
            if ei == 'H' and ej == 'H':
                continue  # forbid H-H
            thr = 1.22 * (covrad(ei) + covrad(ej)) + 0.05
            if 0.45 < d[i, j] <= thr:
                pairs.append([i, j])

    # enforce "one bond per hydrogen": keep only the nearest heavy neighbor
    if pairs:
        nbrs = {i: [] for i in range(n)}
        for i, j in pairs:
            nbrs[i].append(j); nbrs[j].append(i)
        def keep_one_h(i):
            # pick nearest heavy neighbor if any
            cand = [j for j in nbrs[i] if atoms[j]['elem'] != 'H']
            if not cand: return set()
            j = min(cand, key=lambda k: d[i, k])
            return {(min(i, j), max(i, j))}
        keep = set()
        for i, j in pairs:
            ei, ej = atoms[i]['elem'], atoms[j]['elem']
            if ei == 'H' and ej != 'H':
                keep |= keep_one_h(i)
            elif ej == 'H' and ei != 'H':
                keep |= keep_one_h(j)
            else:
                keep.add((i, j))
        bonds = sorted(keep)
    else:
        bonds = []

    # precompute CONECT lines (per model)
    def conect_block():
        lines = []
        for i, j in bonds:
            si, sj = atoms[i]['serial'], atoms[j]['serial']
            lines.append(f"CONECT{si:5d}{sj:5d}\n")
            lines.append(f"CONECT{sj:5d}{si:5d}\n")
        return ''.join(lines)

    # --- build multi-model PDB where only coordinates change, bonds fixed ---
    def one_model_block(flat_xyz, model_idx):
        block = [f"MODEL        {model_idx}\n"]
        for k, a in enumerate(atoms):
            x, y, z = flat_xyz[3*k:3*k+3]
            ln = a['line']
            block.append(ln[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + ln[54:])
        block.append(conect_block())
        block.append("ENDMDL\n")
        return ''.join(block)

    pdb_frames = ''.join(one_model_block(coords[b].ravel(), b+1) for b in range(B))

    # --- viewer ---
    v = py3Dmol.view(width=width, height=height) # bonds now come from CONECT
    if show_h:
        v.addModelsAsFrames(pdb_frames, 'pdb', {'keepH': True, 'reassignBonds': False})
    else:
        v.addModelsAsFrames(pdb_frames)
    v.setBackgroundColor(background)
    v.setViewStyle({'style': 'outline', 'width': 0.03})

    # muted palette (heavy atoms only unless show_h=True)
    muted = {
        'C': '#7a7a7a', 'N': '#3a6ea5', 'O': '#b14a4a',
        'S': '#b58b00', 'P': '#a275a2',
        'F': '#6eaf7a', 'Cl': '#598a62', 'Br': '#7d5a5a', 'I': '#6a5a7d',
        'H': '#cfcfcf'
    }
    def style_elem(e):
        if e == 'H' and not show_h:
            return
        v.setStyle({'elem': e}, {'stick': {'radius': stick_radius, 'color': muted.get(e, '#7f7f7f')}})

    for e in {a['elem'] for a in atoms}:
        style_elem(e)

    v.zoomTo()
    if animate and B > 1:
        v.animate({'loop': 'forward', 'interval': 40})
    return v
