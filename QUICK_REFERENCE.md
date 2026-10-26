# Quick Reference: Using Wasserstein-2 for Alanine Dipeptide Torsion Angles

## Example Usage

### Basic Workflow

```python
import numpy as np
from aita.utils.inference_utils import adp_torsion_angles, calc_torsion_w2

# Assume you have generated samples and holdout/reference samples
# Shape: [n_samples, 22, 3] for 22 atoms with 3D coordinates
generated_samples = np.random.randn(1000, 22, 3)  # Your generated samples
reference_samples = np.random.randn(1000, 22, 3)  # Reference/MD samples

# Path to PDB topology file
pdb_path = "path/to/alanine_dipeptide.pdb"

# Step 1: Extract torsion angles
gen_angles = adp_torsion_angles(generated_samples, pdb_path)
ref_angles = adp_torsion_angles(reference_samples, pdb_path)

# gen_angles and ref_angles are now [n_samples, 2]
# Column 0: φ (phi) angles
# Column 1: ψ (psi) angles

# Step 2: Compute Wasserstein-2 distance
w2_distance = calc_torsion_w2(gen_angles, ref_angles)

print(f"Wasserstein-2 distance: {w2_distance:.6f}")
```

### Alternative: Using Torus Wasserstein (Recommended for Circular Data)

```python
from aita.utils.inference_utils import adp_torsion_angles, torus_wasserstein

# Extract angles (same as above)
gen_angles = adp_torsion_angles(generated_samples, pdb_path)
ref_angles = adp_torsion_angles(reference_samples, pdb_path)

# Compute using proper circular distance
w2_distance = torus_wasserstein(gen_angles, ref_angles)

print(f"Torus Wasserstein-2 distance: {w2_distance:.6f}")
```

## Function Signatures

### `adp_torsion_angles(samples, pdb_path)`

**Purpose**: Extract φ and ψ torsion angles from Cartesian coordinates

**Parameters**:
- `samples` (np.ndarray): Shape [n_samples, 22, 3] or [n_samples, 66]
  - 22 atoms of alanine dipeptide
  - 3D Cartesian coordinates
- `pdb_path` (str): Path to PDB topology file

**Returns**:
- `angles` (np.ndarray): Shape [n_samples, 2]
  - Column 0: φ (phi) angles in radians
  - Column 1: ψ (psi) angles in radians

**Note**: Automatically handles chirality correction

---

### `calc_torsion_w2(gen_angles, holdout_angles)`

**Purpose**: Compute Wasserstein-2 distance using modulo-π periodic distance

**Parameters**:
- `gen_angles` (np.ndarray): Shape [N, 2] - Generated torsion angles
- `holdout_angles` (np.ndarray): Shape [M, 2] - Reference torsion angles

**Returns**:
- `w2` (float): Wasserstein-2 distance

**Distance Formula**: 
```
d²(θ₁, θ₂) = Σₖ ((θ₁ₖ - θ₂ₖ) mod π)²
```

---

### `torus_wasserstein(gen_angles, holdout_angles)`

**Purpose**: Compute Wasserstein-2 distance using proper circular distance

**Parameters**:
- `gen_angles` (np.ndarray): Shape [N, 2] - Generated torsion angles
- `holdout_angles` (np.ndarray): Shape [M, 2] - Reference torsion angles

**Returns**:
- `w2` (float): Wasserstein-2 distance

**Distance Formula**:
```
d²(θ₁, θ₂) = Σₖ min(|θ₁ₖ - θ₂ₖ|, 2π - |θ₁ₖ - θ₂ₖ|)²
```

**Note**: This is the recommended method for proper handling of periodic angles

---

### `map_chirality_batch(samples)`

**Purpose**: Correct chirality of alanine dipeptide samples to L-form

**Parameters**:
- `samples` (np.ndarray): Shape [n_samples, 22, 3]

**Returns**:
- `samples_mapped` (np.ndarray): Same shape, with D-chirality samples flipped

**Process**:
1. Identifies C-beta atom from carbon atoms
2. Computes chirality using cross product
3. Flips D-form to L-form by negating coordinates

---

## Comparison: calc_torsion_w2 vs torus_wasserstein

| Feature | calc_torsion_w2 | torus_wasserstein |
|---------|----------------|-------------------|
| Distance | `(θ₁-θ₂) mod π` | `min(\|θ₁-θ₂\|, 2π-\|θ₁-θ₂\|)` |
| Periodicity | mod π | Full 2π wrapping |
| Geometric Correctness | Approximate | Exact for circular data |
| Recommended Use | Legacy/compatibility | Modern analysis |

## Dependencies

```python
import numpy as np
import mdtraj as md  # For trajectory analysis
import ot  # Python Optimal Transport library
```

## Common Pitfalls

1. **Input Shape**: Ensure samples are shape [n_samples, 22, 3]
   ```python
   # If you have flattened coordinates [n_samples, 66]
   samples = samples.reshape(-1, 22, 3)
   ```

2. **PDB Topology**: Must match the atom ordering in your samples
   ```python
   # Verify atom count
   topology = md.load_topology(pdb_path)
   assert topology.n_atoms == 22
   ```

3. **Angle Range**: MDTraj returns angles in radians [-π, π]
   - No need to convert if using the provided functions

4. **Sample Size**: For large datasets, computation can be slow
   ```python
   # The optimal transport solver can take time for large N, M
   # Consider subsampling if needed
   gen_angles_sub = gen_angles[::10]  # Every 10th sample
   ```

## Performance Notes

- **Time Complexity**: O(N³) for optimal transport with N samples
- **Memory**: O(N²) for distance matrix
- **numItermax**: Set to 1e9 to ensure convergence
  - May need adjustment for very large problems

## Visualization

After computing angles, you can visualize with:

```python
from aita.utils.plotting import adp_ramachandran_plot

# Create Ramachandran plot
adp_ramachandran_plot(
    samples=generated_samples,
    pdb_path=pdb_path,
    figsize=(11, 9),
    prefix="experiment_name",
    wandb_logger=None  # or provide WandbLogger instance
)
```

## Related Functions

For energy distributions (not angles):

```python
from aita.utils.inference_utils import calc_energy_w2

# Compute W2 for energy distributions
energy_w2 = calc_energy_w2(gen_energies, target_energies)
```

## References

- Python Optimal Transport (POT): https://pythonot.github.io/
- MDTraj: https://mdtraj.org/
- Wasserstein metric: https://en.wikipedia.org/wiki/Wasserstein_metric
