"""
Block Data Structure for DMRG

This module provides the Block class for density-matrix renormalization group.
Reference: White, Phys. Rev. B 48, 10345 (1993), Sections II, IV, V

Module Organization:
  1. Block class        - DMRG block container with boundary operators
  2. Factory methods    - single_site (create initial block)
  3. DMRG operations    - enlarge_left, enlarge_right, truncate
  4. Transformations    - copy, reflect, to_sparse, to_dense

Block Structure (Table I):
  Each block B_l representing l sites contains:
    - H        : Block Hamiltonian (internal energy)
    - Sz_l     : Sz operator at LEFT boundary
    - Sp_l     : S+ operator at LEFT boundary
    - Sz_r     : Sz operator at RIGHT boundary
    - Sp_r     : S+ operator at RIGHT boundary
    - Sz_total : Total Sz quantum number for each basis state

Basis Convention:
  States ordered by magnetic quantum number m = S, S-1, ..., -S
    - S=1/2: |+1/2> at index 0, |-1/2> at index 1
    - S=1:   |+1> at index 0, |0> at index 1, |-1> at index 2

Key Formulas:
  - Eq. (2): H_BB' = H_B (x) I + I (x) H_B' + H_interaction
  - H_interaction = J * (Sz (x) Sz + (S+ (x) S- + S- (x) S+) / 2)
  - Eq. (3): H' = O * H * O^T  (truncation to reduced basis)
  - Eq. (6): A' = O * A * O^T  (operator transformation under truncation)
  - S- = (S+)^T  (lowering operator is transpose of raising operator)

Terminology:
  White 1993          Modern Tensor Network
  ---------------     ----------------------
  m (kept states)     chi (bond dimension)
  dim (site states)   d (local/physical dimension)
  Block B             MPS tensor with boundary operators

Memory Scaling:
  Dense:  ~5 * chi^2 * 8 bytes
  Sparse: ~5 * nnz * 12 bytes (CSR format)
"""

import numpy as np
from operators import (
    eye,
    get_spin_operators,
    is_hermitian,
    kron,
    to_dense,
    to_sparse,
    zeros,
)
from scipy import sparse as sp

# =============================================================================
# BLOCK CLASS
# =============================================================================


class Block:
    """
    DMRG block container (unified dense/sparse implementation).

    Stores operators as dense ndarray or sparse CSR matrices,
    controlled by the `sparse` attribute. Default is dense.

    Attributes
    ----------
    length : int
        Number of sites in the block.
    bond_dim : int
        Number of basis states (chi).
    H : ndarray or csr_matrix
        Block Hamiltonian, shape (chi, chi).
    Sz_l, Sp_l : ndarray or csr_matrix
        Spin operators at LEFT boundary.
    Sz_r, Sp_r : ndarray or csr_matrix
        Spin operators at RIGHT boundary.
    Sz_total : ndarray
        Total Sz for each basis state, shape (chi,).
    spin : float
        Spin quantum number S (0.5, 1.0, etc.).
    sparse : bool
        Storage format flag.
    """

    __slots__ = (
        "length",
        "bond_dim",
        "spin",
        "H",
        "Sz_l",
        "Sp_l",
        "Sz_r",
        "Sp_r",
        "Sz_total",
        "sparse",
    )

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self, length, bond_dim, spin, H, Sz_l, Sp_l, Sz_r, Sp_r, Sz_total, sparse=False
    ):
        """
        Initialize block.

        Parameters
        ----------
        length : int
            Number of sites in the block.
        bond_dim : int
            Number of basis states (chi).
        spin : float
            Spin quantum number S.
        H : ndarray or csr_matrix
            Block Hamiltonian, shape (chi, chi).
        Sz_l : ndarray or csr_matrix
            Left boundary Sz operator.
        Sp_l : ndarray or csr_matrix
            Left boundary S+ operator.
        Sz_r : ndarray or csr_matrix
            Right boundary Sz operator.
        Sp_r : ndarray or csr_matrix
            Right boundary S+ operator.
        Sz_total : ndarray
            Total Sz for each basis state, shape (chi,).
        sparse : bool, optional
            If True, store as sparse CSR (default: False).
        """
        # Block properties
        self.length = length
        self.bond_dim = bond_dim
        self.spin = spin
        self.sparse = sparse

        # Quantum numbers
        self.Sz_total = Sz_total.copy()

        # Operators (convert to appropriate format)
        if sparse:
            self.H = to_sparse(H)
            self.Sz_l = to_sparse(Sz_l)
            self.Sp_l = to_sparse(Sp_l)
            self.Sz_r = to_sparse(Sz_r)
            self.Sp_r = to_sparse(Sp_r)
        else:
            self.H = to_dense(H)
            self.Sz_l = to_dense(Sz_l)
            self.Sp_l = to_dense(Sp_l)
            self.Sz_r = to_dense(Sz_r)
            self.Sp_r = to_dense(Sp_r)

    # -------------------------------------------------------------------------
    # Factory Method
    # -------------------------------------------------------------------------

    @staticmethod
    def single_site(S, sparse=False):
        """
        Create single-site block with spin S.

        For a single site, H = 0 (no internal bonds) and left/right
        boundary operators are identical.

        Parameters
        ----------
        S : float
            Spin quantum number (0.5, 1.0, 1.5, ...).
        sparse : bool, optional
            If True, store as sparse CSR (default: False).

        Returns
        -------
        Block
            Single-site block with bond_dim = 2S+1.

        Notes
        -----
        Basis states ordered by m = S, S-1, ..., -S:
          - S=1/2: |+1/2> at index 0, |-1/2> at index 1
          - S=1:   |+1> at index 0, |0> at index 1, |-1> at index 2
        """
        Sz, Sp, _ = get_spin_operators(S, sparse=sparse)
        d = int(2 * S + 1)
        Sz_total = np.asarray(Sz.diagonal()) if sparse else np.diag(Sz)

        return Block(
            length=1,
            bond_dim=d,
            spin=S,
            H=zeros((d, d), sparse=sparse),
            Sz_l=Sz,
            Sp_l=Sp,
            Sz_r=Sz,
            Sp_r=Sp,
            Sz_total=Sz_total,
            sparse=sparse,
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def Sm_l(self):
        """
        S- operator at left boundary.

        Computed as Sm = Sp^T (transpose of raising operator).
        For real matrices: S-^dagger = S+, hence Sm = Sp.T.
        """
        if self.sparse:
            return self.Sp_l.T.tocsr()
        result = self.Sp_l.T
        result.flags.writeable = False
        return result

    @property
    def Sm_r(self):
        """
        S- operator at right boundary.

        Computed as Sm = Sp^T (transpose of raising operator).
        For real matrices: S-^dagger = S+, hence Sm = Sp.T.
        """
        if self.sparse:
            return self.Sp_r.T.tocsr()
        result = self.Sp_r.T
        result.flags.writeable = False
        return result

    # -------------------------------------------------------------------------
    # DMRG Operations: Enlargement (Table II, Step 2)
    # -------------------------------------------------------------------------

    def enlarge_right(self, J=1.0):
        """
        Add site to right: B_l -> B_l* (Eq. 2).

        Constructs enlarged Hamiltonian:
          H_{B*} = H_B (x) I + J * (Sz_r (x) Sz + (Sp_r (x) Sm + Sm_r (x) Sp) / 2)

        Uses Heisenberg interaction: H_int = J * S1 · S2
          = J * (Sz (x) Sz + (S+ (x) S- + S- (x) S+) / 2)

        Parameters
        ----------
        J : float, optional
            Heisenberg coupling constant (default: 1.0).

        Returns
        -------
        Block
            Enlarged block with length+1 sites, bond_dim*d states.
        """
        d = int(2 * self.spin + 1)
        Sz_site, Sp_site, Sm_site = get_spin_operators(self.spin, sparse=self.sparse)
        I_block = eye(self.bond_dim, sparse=self.sparse)
        I_site = eye(d, sparse=self.sparse)

        # Build Hamiltonian: H_B (x) I + J * (Sz*Sz + 0.5*(S+*S- + S-*S+))
        H_new = kron(self.H, I_site, sparse=self.sparse)
        H_new += J * kron(self.Sz_r, Sz_site, sparse=self.sparse)
        H_new += (0.5 * J) * (
            kron(self.Sp_r, Sm_site, sparse=self.sparse)
            + kron(self.Sm_r, Sp_site, sparse=self.sparse)
        )

        # Sz diagonal for tensor sum
        Sz_diag = np.asarray(Sz_site.diagonal()) if self.sparse else np.diag(Sz_site)

        return Block(
            length=self.length + 1,
            bond_dim=self.bond_dim * d,
            spin=self.spin,
            H=H_new,
            Sz_l=kron(self.Sz_l, I_site, sparse=self.sparse),
            Sp_l=kron(self.Sp_l, I_site, sparse=self.sparse),
            Sz_r=kron(I_block, Sz_site, sparse=self.sparse),
            Sp_r=kron(I_block, Sp_site, sparse=self.sparse),
            Sz_total=np.add.outer(self.Sz_total, Sz_diag).ravel(),
            sparse=self.sparse,
        )

    def enlarge_left(self, J=1.0):
        """
        Add site to left: B_l -> *B_l.

        Constructs enlarged Hamiltonian:
          H_{*B} = I (x) H_B + J * (Sz (x) Sz_l + (Sp (x) Sm_l + Sm (x) Sp_l) / 2)

        Uses Heisenberg interaction: H_int = J * S1 · S2
          = J * (Sz (x) Sz + (S+ (x) S- + S- (x) S+) / 2)

        Parameters
        ----------
        J : float, optional
            Heisenberg coupling constant (default: 1.0).

        Returns
        -------
        Block
            Enlarged block with length+1 sites, d*bond_dim states.
        """
        d = int(2 * self.spin + 1)
        Sz_site, Sp_site, Sm_site = get_spin_operators(self.spin, sparse=self.sparse)
        I_block = eye(self.bond_dim, sparse=self.sparse)
        I_site = eye(d, sparse=self.sparse)

        # Build Hamiltonian: I (x) H_B + J * (Sz*Sz + 0.5*(S+*S- + S-*S+))
        H_new = kron(I_site, self.H, sparse=self.sparse)
        H_new += J * kron(Sz_site, self.Sz_l, sparse=self.sparse)
        H_new += (0.5 * J) * (
            kron(Sp_site, self.Sm_l, sparse=self.sparse)
            + kron(Sm_site, self.Sp_l, sparse=self.sparse)
        )

        # Sz diagonal for tensor sum
        Sz_diag = np.asarray(Sz_site.diagonal()) if self.sparse else np.diag(Sz_site)

        return Block(
            length=self.length + 1,
            bond_dim=d * self.bond_dim,
            spin=self.spin,
            H=H_new,
            Sz_l=kron(Sz_site, I_block, sparse=self.sparse),
            Sp_l=kron(Sp_site, I_block, sparse=self.sparse),
            Sz_r=kron(I_site, self.Sz_r, sparse=self.sparse),
            Sp_r=kron(I_site, self.Sp_r, sparse=self.sparse),
            Sz_total=np.add.outer(Sz_diag, self.Sz_total).ravel(),
            sparse=self.sparse,
        )

    # -------------------------------------------------------------------------
    # DMRG Operations: Truncation (Table II, Steps 5-7)
    # -------------------------------------------------------------------------

    def truncate(self, O, Sz_new=None, sparse=None):
        """
        Truncate to reduced basis: A' = O*A*O^T (Eqs. 3, 6).

        Parameters
        ----------
        O : ndarray, shape (chi_new, chi_old)
            Projection matrix (rows = kept eigenstates).
        Sz_new : ndarray, shape (chi_new,), optional
            Sz quantum number for each kept state. If provided, used directly.
            If None, computes from dominant component (less accurate).
        sparse : bool or None, optional
            Output format. None preserves current format.

        Returns
        -------
        Block
            Truncated block with bond_dim = chi_new.

        Notes
        -----
        Assumes O is real (from real symmetric density matrix eigenproblem).
        When Sz_new is provided (from sector-wise truncation), each truncated
        state has an exact, well-defined Sz quantum number.
        """
        out_sparse = self.sparse if sparse is None else sparse
        chi_new = O.shape[0]
        OT = O.T

        # Transform: A' = O A O^T
        H_new = O @ self.H @ OT
        Sz_l_new = O @ self.Sz_l @ OT
        Sp_l_new = O @ self.Sp_l @ OT
        Sz_r_new = O @ self.Sz_r @ OT
        Sp_r_new = O @ self.Sp_r @ OT

        # Use provided Sz or compute from dominant component
        if Sz_new is not None:
            Sz_total_new = Sz_new.copy()
        else:
            # Fallback: assign Sz from dominant component of each truncated state
            dominant_idx = np.argmax(np.abs(O), axis=1)
            Sz_total_new = self.Sz_total[dominant_idx]

        return Block(
            length=self.length,
            bond_dim=chi_new,
            spin=self.spin,
            H=H_new,
            Sz_l=Sz_l_new,
            Sp_l=Sp_l_new,
            Sz_r=Sz_r_new,
            Sp_r=Sp_r_new,
            Sz_total=Sz_total_new,
            sparse=out_sparse,
        )

    # -------------------------------------------------------------------------
    # Copy and Transform
    # -------------------------------------------------------------------------

    def copy(self):
        """Create independent deep copy."""
        return Block(
            length=self.length,
            bond_dim=self.bond_dim,
            spin=self.spin,
            H=self.H,
            Sz_l=self.Sz_l,
            Sp_l=self.Sp_l,
            Sz_r=self.Sz_r,
            Sp_r=self.Sp_r,
            Sz_total=self.Sz_total,
            sparse=self.sparse,
        )

    def reflect(self):
        """
        Swap L <-> R boundaries.

        Used to form right blocks from left blocks in symmetric DMRG.

        Returns
        -------
        Block
            Reflected block with swapped boundary operators.
        """
        return Block(
            length=self.length,
            bond_dim=self.bond_dim,
            spin=self.spin,
            H=self.H,
            Sz_l=self.Sz_r,
            Sp_l=self.Sp_r,
            Sz_r=self.Sz_l,
            Sp_r=self.Sp_l,
            Sz_total=self.Sz_total,
            sparse=self.sparse,
        )

    def to_sparse(self):
        """
        Convert to sparse format.

        Returns
        -------
        Block
            Block with sparse CSR matrices.
        """
        if self.sparse:
            return self.copy()
        return Block(
            length=self.length,
            bond_dim=self.bond_dim,
            spin=self.spin,
            H=self.H,
            Sz_l=self.Sz_l,
            Sp_l=self.Sp_l,
            Sz_r=self.Sz_r,
            Sp_r=self.Sp_r,
            Sz_total=self.Sz_total,
            sparse=True,
        )

    def to_dense(self):
        """
        Convert to dense format.

        Returns
        -------
        Block
            Block with dense ndarray matrices.
        """
        if not self.sparse:
            return self.copy()
        return Block(
            length=self.length,
            bond_dim=self.bond_dim,
            spin=self.spin,
            H=self.H,
            Sz_l=self.Sz_l,
            Sp_l=self.Sp_l,
            Sz_r=self.Sz_r,
            Sp_r=self.Sp_r,
            Sz_total=self.Sz_total,
            sparse=False,
        )

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    def sector_indices(self, Sz_target):
        """
        Find basis indices in a given Sz sector.

        Parameters
        ----------
        Sz_target : float
            Target total Sz quantum number.

        Returns
        -------
        ndarray
            Indices where Sz_total matches target.
        """
        return np.flatnonzero(np.abs(self.Sz_total - Sz_target) < 1e-10)

    def validate(self, tol=1e-12):
        """
        Validate block consistency.

        Checks shapes, sparsity, and Hermiticity.

        Parameters
        ----------
        tol : float, optional
            Hermiticity tolerance (default: 1e-12).

        Returns
        -------
        bool
            True if all checks pass.

        Raises
        ------
        AssertionError
            If any check fails.
        """
        chi = self.bond_dim

        for name, mat in [
            ("H", self.H),
            ("Sz_l", self.Sz_l),
            ("Sp_l", self.Sp_l),
            ("Sz_r", self.Sz_r),
            ("Sp_r", self.Sp_r),
        ]:
            assert mat.shape == (chi, chi), f"{name} shape mismatch"
            if self.sparse:
                assert sp.issparse(mat), f"{name} should be sparse"

        assert self.Sz_total.shape == (chi,), "Sz_total shape mismatch"

        for name, mat in [("H", self.H), ("Sz_l", self.Sz_l), ("Sz_r", self.Sz_r)]:
            assert is_hermitian(mat, tol), f"{name} not Hermitian"

        return True

    # -------------------------------------------------------------------------
    # String Representations
    # -------------------------------------------------------------------------

    def __repr__(self):
        fmt = "sparse" if self.sparse else "dense"
        return f"Block(L={self.length}, chi={self.bond_dim}, S={self.spin}, {fmt})"

    def __str__(self):
        fmt = "sparse" if self.sparse else "dense"
        return (
            f"Block: {self.length} site(s), chi={self.bond_dim}, S={self.spin}, "
            f"Sz in [{self.Sz_total.min():.1f}, {self.Sz_total.max():.1f}], {fmt}"
        )
