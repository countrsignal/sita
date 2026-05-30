# Wasserstein-2 Metric for Alanine Dipeptide - Documentation Index

## Overview

This directory contains comprehensive documentation on how the Wasserstein-2 (W2) metric is computed for the torsion angle distribution of alanine dipeptide (ADP) in this repository.

## Documentation Files

### 📊 [WASSERSTEIN_METRIC_ANALYSIS.md](WASSERSTEIN_METRIC_ANALYSIS.md)
**Purpose**: In-depth technical analysis  
**Best for**: Understanding the implementation details, mathematical background, and code structure

**Contents**:
- Overview of the computation pipeline
- Detailed code walkthroughs with line numbers
- Step-by-step explanation of each component
- Mathematical formulations and theory
- Comparison of different implementations
- Usage context in the codebase

**Length**: ~184 lines, 7.2 KB

---

### 🔄 [COMPUTATION_FLOW.txt](COMPUTATION_FLOW.txt)
**Purpose**: Visual flow diagram  
**Best for**: Quick understanding of the overall process and data flow

**Contents**:
- ASCII diagram showing the complete pipeline
- Input/output at each stage
- Alternative computation paths
- File locations and function references
- Key libraries used
- Mathematical formulation summary

**Length**: ~157 lines, 5.9 KB

---

### 🚀 [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
**Purpose**: Practical usage guide  
**Best for**: Implementing and using the W2 metric in your code

**Contents**:
- Ready-to-use code examples
- Function signatures and parameters
- Comparison of different methods
- Common pitfalls and solutions
- Performance considerations
- Visualization examples

**Length**: ~205 lines, 5.6 KB

---

## Quick Start

If you just want to use the metric:
1. Start with [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. Copy the example code
3. Adapt to your data

If you want to understand how it works:
1. Read [WASSERSTEIN_METRIC_ANALYSIS.md](WASSERSTEIN_METRIC_ANALYSIS.md)
2. Refer to [COMPUTATION_FLOW.txt](COMPUTATION_FLOW.txt) for the visual overview
3. Check source files in `aita/utils/inference_utils.py`

## Key Implementation Files

All implementations are located in the repository:

```
aita/
├── utils/
│   ├── inference_utils.py          # Main W2 implementation
│   │   ├── adp_torsion_angles()    # Extract φ,ψ angles from coordinates
│   │   ├── map_chirality_batch()   # Correct molecular chirality
│   │   ├── calc_torsion_w2()       # W2 with mod-π distance
│   │   └── torus_wasserstein()     # W2 with proper circular distance
│   └── plotting.py                 # Visualization utilities
│
└── models/components/
    └── distribution_distances.py   # General Wasserstein functions
```

## Summary of the Computation

### The Pipeline (6 Steps)

1. **Chirality Correction**
   - Input: Cartesian coordinates [N, 22, 3]
   - Output: L-chirality corrected coordinates
   - Function: `map_chirality_batch()`

2. **Trajectory Creation**
   - Input: Corrected coordinates + PDB topology
   - Output: MDTraj trajectory object
   - Library: MDTraj

3. **Torsion Angle Extraction**
   - Input: Trajectory object
   - Output: φ and ψ angles [N, 2]
   - Function: `md.compute_dihedrals()`
   - Atom indices: φ=[4,6,8,14], ψ=[6,8,14,16]

4. **Pairwise Distance Matrix**
   - Input: Generated angles [N,2], Reference angles [M,2]
   - Output: Distance matrix [N, M]
   - Methods:
     - `calc_torsion_w2`: d² = Σ((θ₁-θ₂) mod π)²
     - `torus_wasserstein`: d² = Σ min(|θ₁-θ₂|, 2π-|θ₁-θ₂|)²

5. **Optimal Transport**
   - Input: Distance matrix, uniform weights
   - Output: Squared W2 distance
   - Function: `ot.emd2()` from POT library
   - Solves: Earth Mover's Distance problem

6. **Final Metric**
   - Input: Squared W2
   - Output: Wasserstein-2 distance
   - Operation: W₂ = √(W²)

### Two Implementations

| Implementation | Distance Formula | Recommended? |
|---------------|------------------|--------------|
| `calc_torsion_w2()` | (θ₁-θ₂) mod π | Legacy |
| `torus_wasserstein()` | min(\|θ₁-θ₂\|, 2π-\|θ₁-θ₂\|) | ✅ Yes |

**Recommendation**: Use `torus_wasserstein()` for proper handling of periodic/circular data.

## Example Usage

```python
from aita.utils.inference_utils import adp_torsion_angles, torus_wasserstein
import numpy as np

# Your data: [n_samples, 22, 3]
generated_samples = np.load("generated.npy")
reference_samples = np.load("reference.npy")

# Extract torsion angles
gen_angles = adp_torsion_angles(generated_samples, "path/to/topology.pdb")
ref_angles = adp_torsion_angles(reference_samples, "path/to/topology.pdb")

# Compute W2 metric
w2_distance = torus_wasserstein(gen_angles, ref_angles)

print(f"Wasserstein-2 distance: {w2_distance:.6f}")
```

## Dependencies

- **NumPy**: Array operations
- **MDTraj**: Molecular dynamics trajectory analysis
- **POT**: Python Optimal Transport library
- **SciPy**: Scientific computing (for chirality calculations)

## Mathematical Foundation

The Wasserstein-2 distance between probability distributions P and Q is:

```
W₂(P,Q) = [min_{γ∈Γ(P,Q)} ∫∫ ‖x-y‖² dγ(x,y)]^(1/2)
```

For discrete uniform distributions:
```
W₂ = [min_{Γ} (1/N²) Σᵢⱼ d²(xᵢ,yⱼ) · Γᵢⱼ]^(1/2)
```

Where:
- **Γ**: Transport plan (coupling matrix)
- **d(x,y)**: Distance between torsion angle pairs
- **γ**: Coupling measure

## Common Use Cases

1. **Model Evaluation**: Compare generated samples to MD simulations
2. **Training Metrics**: Monitor distribution matching during training
3. **Ablation Studies**: Assess impact of model components
4. **Benchmark Comparison**: Compare different generative models

## Citation

If you use this implementation, please cite the appropriate papers referenced in the main repository.

## Support

For questions or issues:
1. Check the detailed documentation files listed above
2. Review the source code in `aita/utils/inference_utils.py`
3. Open an issue in the repository

---

**Last Updated**: 2026-01-29  
**Documentation Version**: 1.0
