# Wasserstein-2 Metric Computation for Alanine Dipeptide Torsion Angles

## Overview
This document explains how the Wasserstein-2 (W2) metric is computed for the torsion angle distribution of alanine dipeptide (ADP) in this repository.

## Key Files
1. **`aita/utils/inference_utils.py`** - Contains the main implementation
2. **`aita/models/components/distribution_distances.py`** - Contains general Wasserstein distance computations
3. **`aita/utils/plotting.py`** - Uses torsion angle computation for visualization

## Step-by-Step Computation Pipeline

### Step 1: Extract Torsion Angles from Cartesian Coordinates

The computation begins with the `adp_torsion_angles()` function in `aita/utils/inference_utils.py` (lines 152-157):

```python
def adp_torsion_angles(samples: np.ndarray, pdb_path: str) -> np.ndarray:
    samples_mapped = map_chirality_batch(samples)
    traj_samples = md.Trajectory(samples_mapped, topology=md.load_topology(pdb_path))
    phi_indices, psi_indices = [4, 6, 8, 14], [6, 8, 14, 16]
    angles = md.compute_dihedrals(traj_samples, [phi_indices, psi_indices])
    return angles
```

**Process:**
1. **Chirality Mapping** (`map_chirality_batch`, lines 119-149): Ensures all samples have consistent L-chirality by:
   - Identifying carbon atoms (C_alpha) around the backbone
   - Computing chirality using the cross product method
   - Flipping D-chirality samples to L-chirality by negating coordinates

2. **Trajectory Construction**: Creates an MDTraj trajectory object with the PDB topology

3. **Dihedral Computation**: Computes phi (φ) and psi (ψ) torsion angles using atom indices:
   - φ (phi): defined by atoms [4, 6, 8, 14]
   - ψ (psi): defined by atoms [6, 8, 14, 16]

### Step 2: Compute Wasserstein-2 Distance on Torsion Space

The W2 metric for torsion angles is computed by `calc_torsion_w2()` in `aita/utils/inference_utils.py` (lines 225-243):

```python
def calc_torsion_w2(gen_angles: np.ndarray, holdout_angles: np.ndarray) -> float:
    """calculates OT w2 Torsion angles 
    
    Args:
        gen_angles: np.ndarray
            np array of sidechain angles 
        holdout_angles: np.ndarray
            np array of sidechain angles
    
    Returns
    -------
    w2: float
        Wasserstein distance between the two distributions
    """
    dist = np.expand_dims(gen_angles,0) - np.expand_dims(holdout_angles,1)
    dist = np.sum((dist % np.pi)**2,axis = -1)
    uniform_weights = ot.unif(gen_angles.shape[0])
    W, _ = ot.emd2(uniform_weights, uniform_weights, dist, numItermax=1e9)
    return np.sqrt(W).item()
```

**Key Steps:**

1. **Pairwise Distance Matrix Construction**:
   ```python
   dist = np.expand_dims(gen_angles,0) - np.expand_dims(holdout_angles,1)
   ```
   - Creates pairwise differences between all generated and holdout angle pairs
   - Shape: (1, N_gen, D) - (M_holdout, 1, D) → (N_gen, M_holdout, D)

2. **Periodic Distance (Modulo π)**:
   ```python
   dist = np.sum((dist % np.pi)**2, axis=-1)
   ```
   - Uses modulo π to handle the periodic nature of torsion angles
   - Squares the differences to compute L2 distance
   - Sums over angle dimensions (φ and ψ) to get scalar distance per pair
   - Result shape: (N_gen, M_holdout)

3. **Optimal Transport**:
   ```python
   uniform_weights = ot.unif(gen_angles.shape[0])
   W, _ = ot.emd2(uniform_weights, uniform_weights, dist, numItermax=1e9)
   ```
   - Uses the POT (Python Optimal Transport) library's `emd2` function
   - Assumes uniform distributions for both generated and holdout samples
   - Solves the discrete optimal transport problem
   - `numItermax=1e9` ensures convergence for large problems

4. **Return W2 Metric**:
   ```python
   return np.sqrt(W).item()
   ```
   - Takes square root because `emd2` returns the squared W2 distance
   - Converts to Python float with `.item()`

## Alternative Implementation: Torus Wasserstein

There's also a `torus_wasserstein()` function (lines 246-257) that handles the circular/periodic nature differently:

```python
def torus_wasserstein(gen_angles: np.ndarray, holdout_angles: np.ndarray) -> float:
    uniform_weights = ot.unif(gen_angles.shape[0])
    
    # wrapped (circular) distances:
    gen_angles = gen_angles[:, None]
    holdout_angles = holdout_angles[None, :]
    dists = np.minimum(np.abs(gen_angles - holdout_angles), 
                       2 * np.pi - np.abs(gen_angles - holdout_angles)) ** 2
    
    # Compute Wasserstein distance using POT
    distance_squared = ot.emd2(uniform_weights, uniform_weights, dists.sum(-1), numItermax=int(1e9))
    return np.sqrt(distance_squared).item()
```

**Differences from `calc_torsion_w2()`:**
- Uses proper circular/wrapped distance: `min(|θ1-θ2|, 2π-|θ1-θ2|)`
- Handles the full [−π, π] range correctly
- More appropriate for angles that wrap around

## General Wasserstein Implementation

The repository also contains a general-purpose Wasserstein function in `aita/models/components/distribution_distances.py` (lines 13-42):

```python
def wasserstein(x0: torch.Tensor, x1: torch.Tensor, method: Optional[str] = None, 
                reg: float = 0.05, power: int = 2, **kwargs) -> float:
    # Supports both W1 and W2
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret
```

This is used for general distribution comparisons but **NOT** for torsion angles, which require special periodic handling.

## Usage in the Codebase

The torsion angle W2 metric is primarily defined in `inference_utils.py` but is **not actively called** in the current training pipeline. It appears to be a utility function for:
1. Post-hoc analysis of generated samples
2. Evaluation during inference/testing
3. Comparison with reference MD simulation data

The visualization code in `plotting.py` uses `adp_torsion_angles()` to create Ramachandran plots and free energy profiles, but doesn't explicitly call `calc_torsion_w2()`.

## Mathematical Background

### Wasserstein-2 Distance
The W2 distance (also called Earth Mover's Distance) between two probability distributions P and Q is:

```
W_2(P,Q) = sqrt(min_{γ ∈ Γ(P,Q)} ∫∫ ||x-y||^2 dγ(x,y))
```

For discrete distributions with uniform weights, this becomes an optimal transport problem solved via linear programming.

### Periodic Distance Handling
For torsion angles θ ∈ [−π, π]:
- **`calc_torsion_w2`** uses: `(θ1 - θ2) mod π`
- **`torus_wasserstein`** uses: `min(|θ1-θ2|, 2π-|θ1-θ2|)`

The latter is more geometrically correct for circular spaces.

## Summary

The Wasserstein-2 metric for alanine dipeptide torsion angles is computed by:
1. Converting Cartesian coordinates to torsion angles (φ, ψ)
2. Ensuring proper chirality mapping
3. Computing pairwise periodic distances between angle distributions
4. Solving the optimal transport problem using the POT library
5. Taking the square root to get the W2 distance

The implementation accounts for the periodic nature of angles and uses efficient numerical optimization for the optimal transport computation.
