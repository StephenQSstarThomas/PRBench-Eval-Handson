"""
Superblock Construction and Operations for DMRG

This module provides superblock operations for density-matrix renormalization group.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section V, Table II

Module Organization:
  1. Superblock class   - DMRG superblock container (unified dense/sparse)
  2. Construction       - __init__ (join blocks with center coupling)
  3. Properties         - H, Sz_total (lazy-computed Hamiltonian and quantum numbers)
  4. DMRG operations    - diagonalize, compute_density_matrix, truncate
  5. Measurements       - measure_energy, measure_center_bond
  6. Transformations    - copy, to_sparse, to_dense

Superblock Configurations (Figure 1):
  Open boundary (OBC):
    B_l * * B_l'^R  (two free sites in middle, blocks at ends)

  Periodic boundary (PBC):
    B_l * B_l'^R *  (wrap-around coupling, Section V.C)

Basis Convention:
  Superblock basis is tensor product |i_L> (x) |i_R>:
    - Dimension: dim_total = chi_L x chi_R
    - Indexing: i = i_L * chi_R + i_R (left-index-major)
    - Sz_total[i] = Sz_L[i_L] + Sz_R[i_R]

Key Equations:
  Heisenberg interaction (Eq. 1):
    - S_i · S_j = Sz_i Sz_j + (S+_i S-_j + S-_i S+_j) / 2

  Superblock Hamiltonian (Eq. 2):
    - H = H_L (x) I_R + I_L (x) H_R + J * (S_r^L · S_l^R)
    - For PBC, add wrap-around: J * (S_l^L · S_r^R)

  Reduced density matrix (Eq. 12):
    - rho(i_L; i_L') = Sum_{i_R} psi(i_L, i_R) psi*(i_L', i_R)
    - In matrix form: rho = Psi @ Psi^T

  Kronecker product indexing (Eq. 18):
    - [A (x) B]_{ij; i'j'} = A_{ii'} B_{jj'}

Algorithm (Table II):
  Step 2: Form superblock H from enlarged blocks
  Step 3: Diagonalize H -> ground state psi
  Step 4: Form density matrix rho = Tr_R(|psi><psi|)
  Step 5: Diagonalize rho -> eigenvalues w_alpha, eigenvectors u^alpha
  Step 6: Keep m largest w_alpha, form projection O

Memory Scaling:
  - Dense:  O(dim_total^2) = O(chi^4) for symmetric blocks
  - Sparse: O(nnz) ~ O(dim_total) for Heisenberg model
  - Crossover: Sparse beneficial when dim_total > ~1000
"""

import numpy as np
from operators import eye, is_hermitian, kron
from scipy import sparse as sp
from scipy.sparse.linalg import eigsh

# =============================================================================
# SUPERBLOCK CLASS
# =============================================================================


class Superblock:
    """
    DMRG superblock container (unified dense/sparse implementation).

    Stores superblock Hamiltonian as dense ndarray or sparse CSR matrix,
    controlled by the `sparse` attribute. Default is dense.

    Attributes
    ----------
    left_block : Block
        Left enlarged block B_l* (l+1 sites, rightmost is free site).
    right_block : Block
        Right enlarged block *B_l'^R (reflected, leftmost is free site).
    J : float
        Heisenberg exchange coupling constant.
    spin : float
        Spin quantum number S of each site (0.5, 1.0, etc.).
    boundary : str
        Boundary condition: "open" (OBC) or "periodic" (PBC).
    dim_left : int
        Basis dimension of left block (chi_L).
    dim_right : int
        Basis dimension of right block (chi_R).
    dim_total : int
        Total superblock dimension (chi_L x chi_R).
    total_length : int
        Total number of sites in superblock (L).
    sparse : bool
        Storage format flag.
    """

    __slots__ = (
        "left_block",
        "right_block",
        "J",
        "spin",
        "boundary",
        "dim_left",
        "dim_right",
        "dim_total",
        "total_length",
        "sparse",
        "_H",
        "_Sz_total",
    )

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, left_block, right_block, J=1.0, boundary="open", sparse=False):
        """
        Initialize superblock from two enlarged blocks.

        Constructs superblock by joining blocks with Heisenberg coupling
        at the center bond (Table II, Step 2).

        Parameters
        ----------
        left_block : Block
            Left enlarged block B_l* (includes rightmost free site).
        right_block : Block
            Right enlarged block *B_l'^R (reflected, leftmost free site).
        J : float, optional
            Heisenberg exchange coupling (default: 1.0).
        boundary : {"open", "periodic"}, optional
            Boundary condition (default: "open").
        sparse : bool, optional
            If True, store as sparse CSR (default: False).
        """
        if boundary not in ("open", "periodic"):
            raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary}")

        self.sparse = sparse

        # Convert blocks to appropriate format
        if sparse:
            self.left_block = left_block.to_sparse()
            self.right_block = right_block.to_sparse()
        else:
            self.left_block = left_block.to_dense()
            self.right_block = right_block.to_dense()

        self.J = float(J)
        self.spin = self.left_block.spin
        self.boundary = boundary
        self.dim_left = self.left_block.bond_dim
        self.dim_right = self.right_block.bond_dim
        self.dim_total = self.dim_left * self.dim_right
        self.total_length = self.left_block.length + self.right_block.length
        self._H = None  # Lazy: built on first H access
        self._Sz_total = None  # Lazy: built on first Sz_total access

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def H(self):
        """
        Superblock Hamiltonian (cached on first access).

        Returns
        -------
        ndarray or csr_matrix, shape (dim_total, dim_total)
            Full Hamiltonian matrix (Hermitian).

        Notes
        -----
        Constructs H using Eq. (2):
          H = H_L (x) I_R + I_L (x) H_R + J * (S_r^L · S_l^R)

        The center coupling S_r^L · S_l^R expands via Eq. (1):
          S · S = Sz (x) Sz + (S+ (x) S- + S- (x) S+) / 2

        For PBC, adds wrap-around term J * (S_l^L · S_r^R).
        """
        if self._H is None:
            L, R, J = self.left_block, self.right_block, self.J
            I_L = eye(self.dim_left, sparse=self.sparse)
            I_R = eye(self.dim_right, sparse=self.sparse)

            # Block Hamiltonians: H_L(x)I_R + I_L(x)H_R
            H = kron(L.H, I_R, sparse=self.sparse) + kron(I_L, R.H, sparse=self.sparse)

            # Center coupling: S_r^L * S_l^R (group flip-flop terms)
            H += J * kron(L.Sz_r, R.Sz_l, sparse=self.sparse)
            H += (0.5 * J) * (
                kron(L.Sp_r, R.Sm_l, sparse=self.sparse)
                + kron(L.Sm_r, R.Sp_l, sparse=self.sparse)
            )

            # PBC wrap-around: S_l^L * S_r^R
            if self.boundary == "periodic":
                H += J * kron(L.Sz_l, R.Sz_r, sparse=self.sparse)
                H += (0.5 * J) * (
                    kron(L.Sp_l, R.Sm_r, sparse=self.sparse)
                    + kron(L.Sm_l, R.Sp_r, sparse=self.sparse)
                )

            self._H = H
        return self._H

    @property
    def Sz_total(self):
        """
        Total Sz quantum number for each superblock basis state (cached).

        Returns
        -------
        ndarray, shape (dim_total,)
            Total Sz for each basis state, ordered by left-index-major.

        Notes
        -----
        For the tensor product basis |i_L, i_R>:
          Sz_total[i] = Sz_L[i // chi_R] + Sz_R[i % chi_R]

        Used for block-diagonalizing H by Sz symmetry, which reduces
        computational cost from O(dim^3) to O(Sum_sector dim_sector^3).
        """
        if self._Sz_total is None:
            # Broadcasting: (chi_L, 1) + (1, chi_R) -> (chi_L, chi_R) -> ravel
            self._Sz_total = (
                self.left_block.Sz_total[:, np.newaxis]
                + self.right_block.Sz_total[np.newaxis, :]
            ).ravel()
        return self._Sz_total

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def get_sector_indices(self, Sz_target):
        """
        Find basis state indices in a given Sz sector.

        Parameters
        ----------
        Sz_target : float
            Target total Sz quantum number.

        Returns
        -------
        ndarray
            Indices i where |Sz_total[i] - Sz_target| < 1e-10.

        Notes
        -----
        The Hamiltonian is block-diagonal in Sz sectors, so restricting
        to a sector reduces the matrix dimension for diagonalization.
        """
        return np.flatnonzero(np.abs(self.Sz_total - Sz_target) < 1e-10)

    def validate(self, tol=1e-12):
        """
        Validate superblock consistency.

        Checks Hamiltonian symmetry and block validity.

        Parameters
        ----------
        tol : float, optional
            Tolerance for Hermiticity check (default: 1e-12).

        Returns
        -------
        bool
            True if all checks pass.

        Raises
        ------
        AssertionError
            If any check fails.
        """
        # Validate constituent blocks
        self.left_block.validate(tol)
        self.right_block.validate(tol)

        # Check H is Hermitian
        assert is_hermitian(self.H, tol), "Superblock H not Hermitian"

        # Check dimensions consistency
        assert self.H.shape == (self.dim_total, self.dim_total), "H shape mismatch"
        assert len(self.Sz_total) == self.dim_total, "Sz_total length mismatch"

        return True

    # -------------------------------------------------------------------------
    # DMRG Operations: Diagonalization (Table II, Step 3)
    # -------------------------------------------------------------------------

    def diagonalize(self, num_states=1, Sz_target=None, tol=1e-12):
        """
        Find lowest eigenstates of superblock Hamiltonian.

        "Using the Davidson or Lanczos method, diagonalize the superblock
        Hamiltonian to find the target state psi(i1,i2,i3,i4)." (Table II)

        Parameters
        ----------
        num_states : int, optional
            Number of lowest eigenstates to compute (default: 1).
        Sz_target : float or None, optional
            If given, restrict to this Sz sector (recommended for efficiency).
        tol : float, optional
            Convergence tolerance for iterative solver (default: 1e-12).

        Returns
        -------
        energies : ndarray, shape (num_states,)
            Eigenvalues E_0 <= E_1 <= ... in ascending order.
        states : ndarray, shape (dim_sector, num_states)
            Eigenvectors as columns (normalized).
        sector_indices : ndarray or None
            Maps sector basis -> full basis (None if Sz_target is None).

        Notes
        -----
        Algorithm selection:
          - Small dim or many states: np.linalg.eigh (O(n^3), exact)
          - Large dim, few states: eigsh/Lanczos (O(n^2 k), iterative)

        Sz conservation:
          - H is block-diagonal in Sz, so restricting to a sector gives
            the same eigenvalues faster
          - Ground state is typically in Sz=0 sector
        """
        # Extract sector or use full space
        if Sz_target is None:
            H_sector = self.H
            sector_indices = None
            dim = self.dim_total
        else:
            sector_indices = self.get_sector_indices(Sz_target)
            dim = len(sector_indices)
            if dim == 0:
                raise ValueError(f"No states with Sz_total = {Sz_target}")
            if self.sparse:
                H_sector = self.H[sector_indices, :][:, sector_indices]
            else:
                H_sector = self.H[np.ix_(sector_indices, sector_indices)]

        # Guard against requesting more states than available
        num_states = min(num_states, dim)

        # Small matrix or many states: full diagonalization
        if dim <= num_states + 1 or num_states >= dim // 4:
            if self.sparse:
                H_dense = H_sector.toarray()
            else:
                H_dense = H_sector
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            return (
                eigenvalues[:num_states],
                eigenvectors[:, :num_states].real,
                sector_indices,
            )

        # Large matrix, few states: Lanczos iteration
        if not self.sparse:
            H_sector = sp.csr_matrix(H_sector)

        ncv = min(dim - 1, max(2 * num_states + 1, 20))
        eigenvalues, eigenvectors = eigsh(
            H_sector, k=num_states, which="SA", tol=tol, ncv=ncv
        )

        # Sort ascending (eigsh order not guaranteed)
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx].real, sector_indices

    # -------------------------------------------------------------------------
    # DMRG Operations: Density Matrix (Table II, Step 4)
    # -------------------------------------------------------------------------

    def compute_density_matrix(self, psi, sector_indices=None):
        """
        Compute reduced density matrix for left block (Eq. 12).

        Traces over right block (environment) degrees of freedom:
          rho(i_L; i_L') = Sum_{i_R} psi(i_L, i_R) psi*(i_L', i_R)

        In matrix form with Psi reshaped to (chi_L, chi_R):
          rho = Psi @ Psi^T

        Parameters
        ----------
        psi : ndarray, shape (dim_sector,) or (dim_total,)
            Normalized superblock wavefunction.
        sector_indices : ndarray or None
            If psi is in sector basis, maps sector -> full basis indices.

        Returns
        -------
        rho : ndarray, shape (dim_left, dim_left)
            Reduced density matrix.

        Notes
        -----
        Properties of rho:
          - Hermitian: rho = rho^T
          - Unit trace: Tr(rho) = 1
          - Positive semi-definite: eigenvalues w_alpha >= 0

        The eigenvalues w_alpha represent probabilities for each reduced
        basis state. Truncation (Step 6) keeps states with largest w_alpha,
        minimizing discarded weight eps = 1 - Sum_{kept} w_alpha.

        Assumes psi is real (from real symmetric eigenproblem).
        """
        # Expand sector wavefunction to full basis if needed
        if sector_indices is None:
            psi_full = psi
        else:
            psi_full = np.zeros(self.dim_total, dtype=psi.dtype)
            psi_full[sector_indices] = psi

        # Reshape to (chi_L, chi_R) and trace over right: rho = Psi @ Psi^T
        Psi = psi_full.reshape(self.dim_left, self.dim_right)
        return Psi @ Psi.T

    # -------------------------------------------------------------------------
    # DMRG Operations: Truncation (Table II, Steps 5-6)
    # -------------------------------------------------------------------------

    def truncate(self, rho, chi):
        """
        Diagonalize density matrix and select projection (Steps 5-6).

        Performs truncation WITHIN each Sz sector to preserve exact quantum
        numbers. This prevents Sz from becoming fractional garbage values.

        Step 5: Diagonalize rho -> eigenvalues w_alpha, eigenvectors |u^alpha>
        Step 6: Keep chi largest eigenvalues, form O = [u^1, ..., u^chi]^T

        Parameters
        ----------
        rho : ndarray, shape (dim_left, dim_left)
            Reduced density matrix from compute_density_matrix().
        chi : int
            Number of states to keep (new bond dimension).

        Returns
        -------
        O : ndarray, shape (chi, dim_left)
            Projection matrix with orthonormal rows.
            Used for truncation: A' = O @ A @ O.T (Eqs. 3, 6).
        w : ndarray, shape (dim_left,)
            All eigenvalues in DECREASING order (Figure 2 data).
        Sz_kept : ndarray, shape (chi,)
            Sz quantum number for each kept state (row of O).

        Notes
        -----
        Truncation Error (Section VII):
          - eps = 1 - Sum_{kept} w_alpha bounds error in local observables
          - For accurate DMRG, target eps < 10^-6
          - eps ~ exp(-const * chi) for gapped systems (fast decay)
          - eps ~ chi^{-p} for critical systems (power-law decay)

        Sector-wise truncation:
          - Density matrix is block-diagonal in Sz sectors
          - Diagonalize each block separately
          - Distribute chi among sectors proportional to eigenvalue weight
          - Each truncated state has well-defined Sz quantum number
        """
        Sz_left = self.left_block.Sz_total
        unique_Sz = np.unique(Sz_left)

        # Collect eigenvalues and eigenvectors from each sector
        sector_data = []  # list of (eigenvalues, eigenvectors, sector_indices, Sz_value)
        all_eigenvalues = []

        for Sz_val in unique_Sz:
            sector_idx = np.flatnonzero(np.abs(Sz_left - Sz_val) < 1e-10)
            if len(sector_idx) == 0:
                continue

            # Extract sector block of density matrix
            rho_sector = rho[np.ix_(sector_idx, sector_idx)]

            # Diagonalize sector
            w_sector, v_sector = np.linalg.eigh(rho_sector)

            # Sort by decreasing eigenvalue
            sort_idx = np.argsort(w_sector)[::-1]
            w_sector = np.maximum(w_sector[sort_idx], 0.0)
            v_sector = v_sector[:, sort_idx]

            sector_data.append((w_sector, v_sector, sector_idx, Sz_val))
            all_eigenvalues.extend(w_sector)

        # Sort all eigenvalues globally (for return value)
        all_eigenvalues = np.array(all_eigenvalues)
        all_eigenvalues = np.sort(all_eigenvalues)[::-1]

        # Distribute chi among sectors proportional to total weight
        # Collect all (eigenvalue, sector_index, within_sector_index) tuples
        eig_tuples = []
        for s_idx, (w_sector, v_sector, sector_idx, Sz_val) in enumerate(sector_data):
            for i, w_val in enumerate(w_sector):
                eig_tuples.append((w_val, s_idx, i))

        # Sort by eigenvalue (descending) and take top chi
        eig_tuples.sort(key=lambda x: -x[0])
        chi_actual = min(chi, len(eig_tuples))
        kept_tuples = eig_tuples[:chi_actual]

        # Build projection matrix O and Sz labels
        dim_left = self.dim_left
        O = np.zeros((chi_actual, dim_left))
        Sz_kept = np.zeros(chi_actual)

        for row, (w_val, s_idx, i) in enumerate(kept_tuples):
            w_sector, v_sector, sector_idx, Sz_val = sector_data[s_idx]
            # Embed sector eigenvector into full space
            O[row, sector_idx] = v_sector[:, i]
            Sz_kept[row] = Sz_val

        return O, all_eigenvalues, Sz_kept

    # -------------------------------------------------------------------------
    # Measurements
    # -------------------------------------------------------------------------

    def measure_all(self, psi, sector_indices=None, include_edges=(False, False)):
        """
        Batch measurement of all observables (optimized).

        Computes center bond strength, center Sz values, and optionally
        edge measurements in a single pass, avoiding redundant operations.

        Parameters
        ----------
        psi : ndarray
            Normalized wavefunction (sector or full basis).
        sector_indices : ndarray or None
            Maps sector -> full basis if psi is in sector basis.
        include_edges : tuple
            If (True, True), include both left and right edge measurements.
            If (True, False), include only left edge.
            If (False, True), include only right edge.
            If (False, False), no edge measurements.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'center_bond': <S_l · S_{l+1}> at center
            - 'center_left_Sz': <Sz> at left center site
            - 'center_right_Sz': <Sz> at right center site
            - 'left_edge_Sz': <Sz> at site 0 (if requested)
            - 'left_edge_bond': <S_0 · S_1> (if requested)
            - 'right_edge_Sz': <Sz> at site L-1 (if requested)
            - 'right_edge_bond': <S_{L-2} · S_{L-1}> (if requested)
        """
        # Expand psi to full basis ONCE
        if sector_indices is None:
            psi_full = psi
        else:
            psi_full = np.zeros(self.dim_total, dtype=psi.dtype)
            psi_full[sector_indices] = psi

        # Reshape ONCE
        Psi = psi_full.reshape(self.dim_left, self.dim_right)

        L, R = self.left_block, self.right_block
        result = {}

        if self.sparse:
            # Sparse-optimized path: use sparse matvec instead of dense einsum
            # Key insight: <psi|(A⊗I)|psi> = vdot(Psi, A @ Psi)
            #              <psi|(I⊗B)|psi> = vdot(Psi, Psi @ B.T)
            #              <psi|(A⊗B)|psi> = vdot(Psi, A @ Psi @ B.T)

            def exp_left(A):
                """<psi|(A⊗I)|psi> using sparse A."""
                return np.vdot(Psi, A @ Psi).real

            def exp_right(B):
                """<psi|(I⊗B)|psi> using sparse B."""
                return np.vdot(Psi, Psi @ B.T).real

            def exp_AB(A, B):
                """<psi|(A⊗B)|psi> using sparse A and B."""
                return np.vdot(Psi, A @ Psi @ B.T).real

            # Center bond: <S_l · S_{l+1}>
            SzSz = exp_AB(L.Sz_r, R.Sz_l)
            SpSm = exp_AB(L.Sp_r, R.Sm_l)
            SmSp = exp_AB(L.Sm_r, R.Sp_l)
            result["center_bond"] = float(SzSz + 0.5 * (SpSm + SmSp))

            # Center Sz values
            result["center_left_Sz"] = float(exp_left(L.Sz_r))
            result["center_right_Sz"] = float(exp_right(R.Sz_l))

            # Edge measurements if requested
            if include_edges:
                left_edge, right_edge = include_edges

                if left_edge:
                    result["left_edge_Sz"] = float(exp_left(L.Sz_l))
                    # Left edge bond: <S_0 · S_1> - both ops on left block
                    SzSz = exp_left(L.Sz_l @ L.Sz_r)
                    SpSm = exp_left(L.Sp_l @ L.Sm_r)
                    SmSp = exp_left(L.Sm_l @ L.Sp_r)
                    result["left_edge_bond"] = float(SzSz + 0.5 * (SpSm + SmSp))

                if right_edge:
                    result["right_edge_Sz"] = float(exp_right(R.Sz_r))
                    # Right edge bond: <S_{L-2} · S_{L-1}> - both ops on right block
                    SzSz = exp_right(R.Sz_l @ R.Sz_r)
                    SpSm = exp_right(R.Sp_l @ R.Sm_r)
                    SmSp = exp_right(R.Sm_l @ R.Sp_r)
                    result["right_edge_bond"] = float(SzSz + 0.5 * (SpSm + SmSp))
        else:
            # Dense path: use einsum (original implementation)
            Sz_l_center = L.Sz_r
            Sp_l_center = L.Sp_r
            Sm_l_center = L.Sm_r
            Sz_r_center = R.Sz_l
            Sp_r_center = R.Sp_l
            Sm_r_center = R.Sm_l

            # Center bond: <S_l · S_{l+1}>
            SzSz = np.einsum("ij,ik,jl,kl->", Psi, Sz_l_center, Sz_r_center, Psi)
            SpSm = np.einsum("ij,ik,jl,kl->", Psi, Sp_l_center, Sm_r_center, Psi)
            SmSp = np.einsum("ij,ik,jl,kl->", Psi, Sm_l_center, Sp_r_center, Psi)
            result["center_bond"] = float(SzSz + 0.5 * (SpSm + SmSp))

            # Center Sz values
            result["center_left_Sz"] = float(
                np.einsum("ij,ik,kj->", Psi, Sz_l_center, Psi)
            )
            result["center_right_Sz"] = float(
                np.einsum("ij,jk,ik->", Psi, Sz_r_center, Psi)
            )

            # Edge measurements if requested
            if include_edges:
                left_edge, right_edge = include_edges

                if left_edge:
                    Sz_0 = L.Sz_l
                    Sp_0 = L.Sp_l
                    Sm_0 = L.Sm_l
                    result["left_edge_Sz"] = float(
                        np.einsum("ij,ik,kj->", Psi, Sz_0, Psi)
                    )
                    SzSz = np.einsum("ij,ik,kl,lj->", Psi, Sz_0, Sz_l_center, Psi)
                    SpSm = np.einsum("ij,ik,kl,lj->", Psi, Sp_0, Sm_l_center, Psi)
                    SmSp = np.einsum("ij,ik,kl,lj->", Psi, Sm_0, Sp_l_center, Psi)
                    result["left_edge_bond"] = float(SzSz + 0.5 * (SpSm + SmSp))

                if right_edge:
                    Sz_L1 = R.Sz_r
                    Sp_L1 = R.Sp_r
                    Sm_L1 = R.Sm_r
                    result["right_edge_Sz"] = float(
                        np.einsum("ij,jk,ik->", Psi, Sz_L1, Psi)
                    )
                    SzSz = np.einsum("ij,jk,kl,il->", Psi, Sz_r_center, Sz_L1, Psi)
                    SpSm = np.einsum("ij,jk,kl,il->", Psi, Sp_r_center, Sm_L1, Psi)
                    SmSp = np.einsum("ij,jk,kl,il->", Psi, Sm_r_center, Sp_L1, Psi)
                    result["right_edge_bond"] = float(SzSz + 0.5 * (SpSm + SmSp))

        return result

    # -------------------------------------------------------------------------
    # Copy and Transform
    # -------------------------------------------------------------------------

    def copy(self):
        """
        Create independent deep copy.

        Returns
        -------
        Superblock
            New superblock with copied blocks and recomputed H.

        Notes
        -----
        - Creates new Block objects for left and right.
        - Hamiltonian cache (_H) is not copied; will be rebuilt on access.
        """
        return Superblock(
            self.left_block,
            self.right_block,
            J=self.J,
            boundary=self.boundary,
            sparse=self.sparse,
        )

    def to_sparse(self):
        """
        Convert to sparse format.

        Returns
        -------
        Superblock
            Superblock with sparse CSR matrices.

        Notes
        -----
        - Returns self if already sparse (no copy made).
        - Creates new Superblock with converted blocks otherwise.
        """
        if self.sparse:
            return self
        return Superblock(
            self.left_block,
            self.right_block,
            J=self.J,
            boundary=self.boundary,
            sparse=True,
        )

    def to_dense(self):
        """
        Convert to dense format.

        Returns
        -------
        Superblock
            Superblock with dense ndarray matrices.

        Notes
        -----
        - Returns self if already dense (no copy made).
        - Creates new Superblock with converted blocks otherwise.
        """
        if not self.sparse:
            return self
        return Superblock(
            self.left_block,
            self.right_block,
            J=self.J,
            boundary=self.boundary,
            sparse=False,
        )

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self):
        fmt = "sparse" if self.sparse else "dense"
        if self.sparse and self._H is not None:
            return (
                f"Superblock(L={self.total_length}, "
                f"dim={self.dim_left}x{self.dim_right}, nnz={self._H.nnz}, {fmt})"
            )
        return (
            f"Superblock(L={self.total_length}, "
            f"dim={self.dim_left}x{self.dim_right}, {fmt})"
        )

    def __str__(self):
        fmt = "sparse" if self.sparse else "dense"
        return (
            f"Superblock: {self.total_length} sites, "
            f"dim={self.dim_total}, J={self.J}, BC={self.boundary}, {fmt}"
        )
