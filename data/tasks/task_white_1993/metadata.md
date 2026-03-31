# Ground Truth: Density-Matrix Algorithms for Quantum Renormalization Groups

This document provides the ground truth formulas and methodology for grading the AI agent's reproduction.

---

## 1. Article Information

| Field | Value |
|-------|-------|
| **Title** | Density-matrix algorithms for quantum renormalization groups |
| **Author** | Steven R. White |
| **DOI** | 10.1103/PhysRevB.48.10345 |

---

## 2. Main Formulas

### 2.1 Heisenberg Hamiltonian
$$
H = J \sum_{i=1}^{L-1} \mathbf{S}_i \cdot \mathbf{S}_{i+1} \quad \text{(OBC)}, \qquad H = J \sum_{i=1}^{L} \mathbf{S}_i \cdot \mathbf{S}_{i+1 \mod L} \quad \text{(PBC)}
$$

**Eq. 1 (Spin interaction):**
$$
\mathbf{S}_i \cdot \mathbf{S}_{i+1} = S_i^z S_{i+1}^z + \frac{1}{2}(S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+) \tag{1}
$$

### 2.2 Block Hamiltonian

**Eq. 2 (Superblock Hamiltonian):**
$$
[H_{BB}]_{i_1 i_2; i_1' i_2'} = [H_B]_{i_1 i_1'} \delta_{i_2 i_2'} + [H_B]_{i_2 i_2'} \delta_{i_1 i_1'} + [S_r^z]_{i_1 i_1'} [S_\ell^z]_{i_2 i_2'} + \frac{1}{2}[S_r^+]_{i_1 i_1'} [S_\ell^-]_{i_2 i_2'} + \frac{1}{2}[S_r^-]_{i_1 i_1'} [S_\ell^+]_{i_2 i_2'} \tag{2}
$$

### 2.3 Truncation

**Eq. 3 (Basis transformation):**
$$
H_{B'} = O H_{BB} O^\dagger \tag{3}
$$

**Eqs. 4-6 (Operator transformation):**
$$
[\tilde{S}_\ell^z]_{i_1 i_2; i_1' i_2'} = [S_\ell^z]_{i_1 i_1'} \delta_{i_2 i_2'} \tag{4}
$$
$$
[\tilde{S}_r^z]_{i_1 i_2; i_1' i_2'} = [S_r^z]_{i_2 i_2'} \delta_{i_1 i_1'} \tag{5}
$$
$$
S_\ell^z = O \tilde{S}_\ell^z O^\dagger \tag{6}
$$

### 2.4 Density Matrix

**Eq. 7-10 (Approximation):**
$$
|\psi\rangle \approx |\bar{\psi}\rangle = \sum_{\alpha,j} a_{\alpha,j}|u^\alpha\rangle|j\rangle \tag{7}
$$
$$
\mathcal{S} = \left||\psi\rangle - |\bar{\psi}\rangle\right|^2 \tag{8}
$$
$$
|\bar{\psi}\rangle = \sum_\alpha a_\alpha |u^\alpha\rangle|v^\alpha\rangle \tag{9}
$$
$$
\mathcal{S} = \sum_{ij} \left(\psi_{ij} - \sum_{\alpha=1}^m a_\alpha u_i^\alpha v_j^\alpha\right)^2 \tag{10}
$$

**Eq. 11-13 (SVD and density matrix):**
$$
\psi = U D V^T \tag{11}
$$
$$
\rho_{ii'} = \sum_j \psi_{ij} \psi_{i'j} \tag{12}
$$
$$
\rho = U D^2 U^T \tag{13}
$$

**Eqs. 14-17 (Multiple target states):**
$$
\mathcal{S} = \sum_k W_k \sum_{ij} \left( \psi_{ij}^k - \sum_{\alpha=1}^m a_{\alpha}^k u_i^{\alpha} v_j^{k,\alpha} \right)^2 \tag{14}
$$
$$
\rho u^{\alpha} = w_{\alpha} u^{\alpha} \tag{15}
$$
$$
\rho_{i,i'} = \sum_k W_k \sum_j \psi_{ij}^k \psi_{i'j}^k \tag{16}
$$
$$
w_{\alpha} = \sum_k W_k (a_{\alpha}^k)^2 \tag{17}
$$

### 2.5 Measurements

**Eq. 19 (Single-site):**
$$
\langle \psi | S_j^z | \psi \rangle = \sum_{i_1 i_2 i_3 i_4 i_1'} \psi_{i_1 i_2 i_3 i_4}^* [S_j^z]_{i_1 i_1'} \psi_{i_1' i_2 i_3 i_4} \tag{19}
$$

**Local bond strength:**
$$
\langle \mathbf{S}_i \cdot \mathbf{S}_{i+1} \rangle = \langle S_i^z S_{i+1}^z \rangle + \frac{1}{2} \langle S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+ \rangle
$$

---

## 3. Methodology

### 3.1 Infinite System Algorithm (Table II)

| Step | Action                                                                                  |
| ---- | --------------------------------------------------------------------------------------- |
| 1    | Form superblock: single site + single site (4 sites total)                              |
| 2    | Build $H_{BB}$                                                                          |
| 3    | Diagonalize $H_{BB}$ for target state $\|\psi\rangle$                                   |
| 4    | Compute $\rho = \|\psi\rangle \langle \psi\|$                                           |
| 5    | Diagonalize $\rho$, keep $m$ largest eigenvalue eigenvectors → $O$                      |
| 6    | Transform: $H_{B'} = O H_{12} O^\dagger$, $S_{B'}^{z,\pm} = O S_{12}^{z,\pm} O^\dagger$ |
| 7    | Set block 4 = reflect(block 1), iterate                                                 |

### 3.2 Finite System Algorithm (Table III)

| Step | Action                                                                   |
| ---- | ------------------------------------------------------------------------ |
| 1    | Run infinite algorithm for $L/2 - 1$ steps, store $B_1, \ldots, B_{L/2}$ |
| 2    | Set $\ell = L/2$, form $B_\ell \bullet \bullet B_{L-\ell-2}^R$           |
| 3    | Execute Table II steps 2-7                                               |
| 4    | Store new block as $B_{\ell+1}$                                          |
| 5    | Replace right block with $B_{L-\ell-2}^R$                                |
| 6    | If $\ell < L-3$: $\ell \leftarrow \ell + 1$, goto 3                      |
| 7    | New iteration: reset $\ell = 1$, goto 3                                  |
| 8    | Stop after 2-3 iterations                                                |

### 3.3 Measurements

| Observable          | Formula                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Ground state energy | $E_0$                                                                                                                             |
| Energy gap          | $\Delta_L = E(S_z=1) - E(S_z=0)$                                                                                                  |
| Bond strength       | $\langle S_i \cdot S_{i+1}\rangle = \langle S_i^z S_{i+1}^z\rangle + \frac{1}{2}\langle S_i^+ S_{i+1}^- + S_i^- S_{i+1}^+\rangle$ |
| Local magnetization | $\langle S_i^z \rangle$                                                                                                           |

---

## 4. Expected Results

### Figure 2: Density-Matrix Eigenvalues
- Eigenvalues decay exponentially for OBC
- Slower decay for PBC (needs more states)
- S=1 has slower decay than S=1/2

### Figure 3: Energy Convergence (S=1/2, L=28)
- OBC: reaches $10^{-10}$ relative error with m~60
- PBC: reaches $10^{-5}$ relative error with m~120

### Figure 4: Energy Convergence (S=1, L=16)
- Similar pattern but slower convergence than S=1/2

### Figure 5: Energy Gap vs 1/L
- S=1/2 Heisenberg chain: gap → 0 (gapless)
- S=1 Heisenberg chain: gap → 0.41 (Haldane gap)

### Figure 6: Local Bond Strength (S=1/2)
- Panel (a): Even L, strong dimerization at edges
- Panel (b): Odd L, uniform alternation
- Panel (c): Soft BCs, reduced edge effects

### Figure 7: Local Observables (S=1)
- Panel (a): Bond strength with edge effects ($S_z=0$)
- Panel (b): Local magnetization profile ($S_z=1$)

### Figure 8: Multi-Target Eigenvalues
- More target states → slower eigenvalue decay
- k=5: eigenvalue spectrum is broadest

---

## Banned Libraries & Practices

| Category              | Banned                                                        | Allowed Alternative         |
| --------------------- | ------------------------------------------------------------- | --------------------------- |
| **DMRG/TN Libraries** | `TeNPy`, `ITensor`, `ALPS`, `TenPy`, `quimb`, `tensornetwork` | Implement from scratch      |
| **ED Packages**       | `QuSpin`, `EDKit`, `exact_diag`                               | Build Hamiltonians manually |
| **GPU Libraries**     | `cupy`, `PyTorch`, `JAX`, `TensorFlow`                        | `NumPy`, `SciPy`            |
| **Symbolic Math**     | `SymPy` for numerical computation                             | Use for verification only   |
| **Auto-diff**         | `autograd`, `JAX`                                             | Manual derivatives          |
