# Paper Reproduction Task

You are given a physics paper to reproduce. Your goal is to read the paper, understand its methodology, and implement a DMRG simulation to reproduce 7 key figures.

---

## 1. Article Information

| Field               | Value                                                        |
| ------------------- | ------------------------------------------------------------ |
| **Title**           | Density-matrix algorithms for quantum renormalization groups |
| **Author**          | Steven R. White                                              |
| **DOI**             | 10.1103/PhysRevB.48.10345                                    |
| **Paper File**      | `white1993.md` (in the same task directory)                  |

---

## 2. What You Must Do

### Step 1: Read and Analyze the Paper
Read the full paper at `white1993.md`. Write a detailed analysis document at `reproduction/ANALYSIS.md` that covers:

1. **Methodology**: Describe the DMRG algorithm (both infinite and finite system versions)
2. **Key Formulas**: List and explain every formula needed for the reproduction, including:
   - The Heisenberg Hamiltonian (OBC and PBC)
   - Superblock construction
   - Density matrix and truncation
   - SVD-based truncation
   - Multi-target state density matrix
   - Measurement formulas for observables
3. **Algorithm Details**: Describe the infinite system algorithm (Table II) and finite system algorithm (Table III) step by step
4. **Physical Observables**: Explain what each figure measures and its physical significance

### Step 2: Implement the Simulation
Write all code under `reproduction/`. You need:
- Core DMRG modules (see Section 6 below)
- Individual scripts for each figure

### Step 3: Generate Data Files
Run your scripts to produce CSV data files under `data/`. See Section 5 for exact specifications.

---

## 3. Input Parameters

| Parameter     | Symbol      | Values         |
| ------------- | ----------- | -------------- |
| Spin          | $S$         | 1/2, 1         |
| Coupling      | $J$         | 0.236, 1.0     |
| Length        | $L$         | 4–100          |
| States kept   | $m$         | 4–200          |
| Target states | $k$         | 1–5            |
| $S_z$ sector  | $S_z^{tot}$ | 0, 1           |
| Boundary      | BCs         | OBC, PBC, Soft |

---

## 4. Banned Libraries & Practices

| Category              | Banned                                                        | Allowed Alternative         |
| --------------------- | ------------------------------------------------------------- | --------------------------- |
| **DMRG/TN Libraries** | `TeNPy`, `ITensor`, `ALPS`, `TenPy`, `quimb`, `tensornetwork` | Implement from scratch      |
| **ED Packages**       | `QuSpin`, `EDKit`, `exact_diag`                               | Build Hamiltonians manually |
| **GPU Libraries**     | `cupy`, `PyTorch`, `JAX`, `TensorFlow`                        | `NumPy`, `SciPy`            |
| **Symbolic Math**     | `SymPy` for numerical computation                             | Use for verification only   |
| **Auto-diff**         | `autograd`, `JAX`                                             | Manual derivatives          |

---

## 5. Data Files to Reproduce

### Figure 2: Density-Matrix Eigenvalues
**File**: `data/fig2.csv`

| Property | Value                                              |
| -------- | -------------------------------------------------- |
| X-axis   | $\alpha$ (eigenvalue index)                        |
| Y-axis   | $w_\alpha$ (log, $10^0$–$10^{-10}$)                |
| System   | $L=32$, $S_z=0$                                    |
| Series   | Open S=1/2, Open S=1, Periodic S=1/2, Periodic S=1 |

**X values**: 1, 2, 3, ..., 50

**Columns**: `alpha,"Open, S=1/2","Open, S=1","Periodic, S=1/2","Periodic, S=1"`

---

### Figure 3: Energy Convergence (S=1/2)
**File**: `data/fig3.csv`

| Property | Value                                        |
| -------- | -------------------------------------------- |
| X-axis   | $m$ (states kept)                            |
| Y-axis   | $\Delta E/\|E\|$ (log, $10^{-1}$–$10^{-11}$) |
| System   | $L=28$, $S=1/2$                              |
| Series   | Open BCs, Periodic BCs                       |

**X values**: 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 90, 100, 110, 120

**Columns**: `m,Open BCs,Periodic BCs`

---

### Figure 4: Energy Convergence (S=1)
**File**: `data/fig4.csv`

| Property | Value                                        |
| -------- | -------------------------------------------- |
| X-axis   | $m$ (states kept)                            |
| Y-axis   | $\Delta E/\|E\|$ (log, $10^{-1}$–$10^{-10}$) |
| System   | $L=16$, $S=1$                                |
| Series   | Open BCs, Periodic BCs                       |

**X values**: 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 90, 100, 110, 120

**Columns**: `m,Open BCs,Periodic BCs`

---

### Figure 5: Energy Gap vs 1/L
**File**: `data/fig5.csv`

| Property | Value                        |
| -------- | ---------------------------- |
| X-axis   | $1/L$ (inverse chain length) |
| Y-axis   | $\Delta_L$ (0–1.0)           |
| System   | PBC                          |
| Series   | S=1/2 (→0), S=1 (→0.41)      |

**X values (L)**: 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 50, 64, 80, 100

**Columns**: `L,1/L,Gap S=1/2,Gap S=1`

---

### Figure 6: Local Bond Strength (S=1/2)
**File**: `data/fig6.csv`

| Property | Value                                             |
| -------- | ------------------------------------------------- |
| X-axis   | $i$ (site index)                                  |
| Y-axis   | $\langle S_i \cdot S_{i+1}\rangle$ (−0.7 to −0.3) |
| System   | $S=1/2$, OBC                                      |
| Series   | (a) $L$=60, (b) $L$=61, (c) $L$=60 soft           |

**X values**: 1, 2, 3, ..., 59 (panels a, c); 1, 2, 3, ..., 60 (panel b)

**Columns**: `i,Panel (a),Panel (b),Panel (c)`

---

### Figure 7: Local Observables (S=1)
**File**: `data/fig7.csv`

| Property | Value                                                  |
| -------- | ------------------------------------------------------ |
| X-axis   | $i$ (site index)                                       |
| Y-axis   | Bond: −1.7 to −1.3; $S_z$: −0.4 to 0.6                 |
| System   | $L$=60, $S=1$, OBC                                     |
| Series   | (a) Bond ($S_z$=0), (b) $\langle S_z\rangle$ ($S_z$=1) |

**X values**: 1, 2, 3, ..., 59 (bond strength); 1, 2, 3, ..., 60 (local $S_z$)

**Columns**: `i,Bond Strength,Local Sz`

---

### Figure 8: Multi-Target Eigenvalues
**File**: `data/fig8.csv`

| Property | Value                               |
| -------- | ----------------------------------- |
| X-axis   | $\alpha$ (eigenvalue index)         |
| Y-axis   | $w_\alpha$ (log, $10^0$–$10^{-10}$) |
| System   | $L$=32, $S=1/2$, OBC                |
| Series   | $k$=1, 2, 3, 4, 5 targets           |

**X values**: 1, 2, 3, ..., 50

**Columns**: `alpha,1 Target,2 Targets,3 Targets,4 Targets,5 Targets`

---

## 6. Code Structure Requirements

### 6.1 Core Modules
**Path**: `reproduction/`

| File               | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `operators.py`     | Spin matrices $S^z, S^+, S^-$ for $S=1/2$ and $S=1$     |
| `block.py`         | Block class with Hamiltonian and boundary operators      |
| `superblock.py`    | Superblock construction, diagonalization, density matrix |
| `dmrg_infinite.py` | Infinite System Algorithm                                |
| `dmrg_finite.py`   | Finite System Algorithm                                  |

### 6.2 Figure Scripts
| Figure | Script Path                    |
| ------ | ------------------------------ |
| Fig 2  | `reproduction/fig2_compute.py` |
| Fig 3  | `reproduction/fig3_compute.py` |
| Fig 4  | `reproduction/fig4_compute.py` |
| Fig 5  | `reproduction/fig5_compute.py` |
| Fig 6  | `reproduction/fig6_compute.py` |
| Fig 7  | `reproduction/fig7_compute.py` |
| Fig 8  | `reproduction/fig8_compute.py` |

### 6.3 Analysis Document
**Path**: `reproduction/ANALYSIS.md`

Must contain your analysis of the paper's methodology and all formulas needed for reproduction.

### 6.4 Output Directories
- Data files: `data/`
- Code files: `reproduction/`

---

## 7. Deliverables Summary

Upon completion, the following files must exist:
1. `reproduction/ANALYSIS.md` — Your methodology and formula analysis
2. `reproduction/operators.py` through `reproduction/dmrg_finite.py` — Core DMRG modules
3. `reproduction/fig2_compute.py` through `reproduction/fig8_compute.py` — Figure scripts
4. `data/fig2.csv` through `data/fig8.csv` — Output data files (7 total)

Export data with 10 decimal places (`%.10e`).

---

## 8. Reproduction Requirements

### 8.1 Computational Resources

| Requirement          | Value        | Notes                              |
| -------------------- | ------------ | ---------------------------------- |
| **Total time limit** | ~2 - 4 hours | PBC with $m=200$ and finite sweeps |
| **Memory limit**     | 4 - 8 GB     | Superblock for $m=200$, $S=1$      |
| **CPU**              | 1 - 4 Cores  | No parallelization required        |
