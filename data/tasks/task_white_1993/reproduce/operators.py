"""
Operators and Matrix Utilities for DMRG

This module provides matrix operations and spin operators for DMRG calculations.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section II

Module Organization:
  1. Matrix Factories     - eye, zeros (create basic matrices)
  2. Type Conversions     - to_dense, to_sparse (format conversion)
  3. Matrix Properties    - is_hermitian (property checking)
  4. Tensor Operations    - kron (Kronecker product)
  5. Spin Operators       - get_spin_operators (physics-specific)

Basis Convention:
  States ordered by magnetic quantum number m = S, S-1, ..., -S
    - S=1/2: |+1/2> at index 0, |-1/2> at index 1
    - S=1:   |+1> at index 0, |0> at index 1, |-1> at index 2

Key Formulas (angular momentum algebra):
  Action on states:
    - Sz |S,m> = m |S,m>
    - S+ |S,m> = sqrt[S(S+1) - m(m+1)] |S,m+1>
    - S- |S,m> = sqrt[S(S+1) - m(m-1)] |S,m-1>

  Operator relations:
    - S- = (S+)^T  (lowering is transpose of raising for real matrices)
    - S^2 = Sz^2 + (S+ S- + S- S+) / 2 = S(S+1) I  (Casimir operator)

  SU(2) commutation relations:
    - [S+, S-] = 2 Sz
    - [Sz, S+] = S+
    - [Sz, S-] = -S-
"""

import numpy as np
from scipy import sparse as sp

# =============================================================================
# MATRIX FACTORIES
# =============================================================================


def eye(n, sparse=False):
    """
    Create identity matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.
    sparse : bool, optional
        If True, return sparse CSR matrix (default: False).

    Returns
    -------
    ndarray or csr_matrix
        Identity matrix of shape (n, n).

    Notes
    -----
    Used for tensor products: H_L (x) I_R extends left block to act on
    enlarged Hilbert space while leaving right subsystem unchanged.
    """
    if sparse:
        return sp.eye(n, format="csr")
    return np.eye(n)


def zeros(shape, sparse=False):
    """
    Create zero matrix.

    Parameters
    ----------
    shape : tuple of int
        Matrix shape (m, n).
    sparse : bool, optional
        If True, return sparse CSR matrix (default: False).

    Returns
    -------
    ndarray or csr_matrix
        Zero matrix of given shape.

    Notes
    -----
    Used for initializing block Hamiltonians. A single-site block has
    H = 0 since there are no internal bonds.
    """
    if sparse:
        return sp.csr_matrix(shape, dtype=np.float64)
    return np.zeros(shape, dtype=np.float64)


# =============================================================================
# TYPE CONVERSIONS
# =============================================================================


def to_dense(A, copy=True):
    """
    Convert matrix to dense ndarray.

    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix.
    copy : bool, optional
        If True, always return a copy (default: True).
        If False, may return a view for dense input.

    Returns
    -------
    ndarray
        Dense array.

    Notes
    -----
    - Creates new array from sparse input (toarray() always copies).
    - Copies dense input if copy=True.
    - Use copy=False for temporary read-only access to avoid allocation.
    """
    if sp.issparse(A):
        return A.toarray()  # toarray() always creates a new array
    result = np.asarray(A)
    return result.copy() if copy else result


def to_sparse(A, copy=True):
    """
    Convert matrix to sparse CSR format.

    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix.
    copy : bool, optional
        If True, always return a copy (default: True).
        If False, may share data for sparse input.

    Returns
    -------
    csr_matrix
        Sparse CSR matrix.

    Notes
    -----
    - Creates new matrix from dense input (csr_matrix() always copies).
    - Copies sparse input if copy=True.
    - Use copy=False for temporary read-only access to avoid allocation.
    """
    if sp.issparse(A):
        return A.tocsr().copy() if copy else A.tocsr()
    return sp.csr_matrix(A)  # csr_matrix() always creates a new matrix


# =============================================================================
# MATRIX PROPERTIES
# =============================================================================


def is_hermitian(A, tol=1e-12):
    """
    Check if matrix is Hermitian (A = A^dagger).

    Parameters
    ----------
    A : ndarray or sparse matrix
        Input matrix.
    tol : float, optional
        Tolerance for comparison (default: 1e-12).

    Returns
    -------
    bool
        True if A is Hermitian within tolerance.

    Notes
    -----
    - Works for both dense and sparse matrices.
    - For real matrices, Hermitian is equivalent to symmetric (A = A^T).
    - Physical observables (H, Sz) must be Hermitian; S+, S- are not.
    """
    if sp.issparse(A):
        diff = A - A.T
        if diff.nnz == 0:
            return True
        return np.abs(diff.data).max() <= tol
    return np.allclose(A, A.T, atol=tol)


# =============================================================================
# TENSOR OPERATIONS
# =============================================================================


def kron(A, B, sparse=False):
    """
    Compute Kronecker product A (x) B.

    Used for constructing tensor product operators in DMRG:
      - H_L (x) I_R  : left block acts, right identity
      - Sz_L (x) Sz_R : spin-spin interaction terms

    Parameters
    ----------
    A : ndarray or sparse matrix
        Left matrix, shape (m, n).
    B : ndarray or sparse matrix
        Right matrix, shape (p, q).
    sparse : bool, optional
        If True, return sparse CSR matrix (default: False).

    Returns
    -------
    ndarray or csr_matrix
        Kronecker product with shape (m*p, n*q).

    Notes
    -----
    Automatically converts inputs to match output format.
    Sparse output is preferred when result has low density.
    """
    if sparse:
        return sp.kron(to_sparse(A, copy=False), to_sparse(B, copy=False), format="csr")
    return np.kron(to_dense(A, copy=False), to_dense(B, copy=False))


# =============================================================================
# SPIN OPERATORS
# =============================================================================

# Pre-computed matrices for S=1/2 and S=1 (performance optimization).
# Sparse versions are lazily cached on first use.
# Memory: S=1/2 dense ~48 bytes each, S=1 dense ~72 bytes each.

_SQRT2 = np.sqrt(2.0)

# S=1/2: Pauli matrices scaled by 1/2
# Sz = (1/2) * sigma_z = diag(1/2, -1/2)
# S+ = sigma_+, S- = sigma_- (off-diagonal)
_SZ_HALF = np.array([[0.5, 0.0], [0.0, -0.5]], dtype=np.float64)
_SP_HALF = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
_SM_HALF = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

# S=1: Spin-1 matrices (3x3)
# Sz = diag(1, 0, -1)
# S+, S- have sqrt(2) coefficients on off-diagonals
_SZ_ONE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
)
_SP_ONE = np.array(
    [[0.0, _SQRT2, 0.0], [0.0, 0.0, _SQRT2], [0.0, 0.0, 0.0]], dtype=np.float64
)
_SM_ONE = np.array(
    [[0.0, 0.0, 0.0], [_SQRT2, 0.0, 0.0], [0.0, _SQRT2, 0.0]], dtype=np.float64
)

# Lazy cache for sparse versions: {(S, True): (Sz_sp, Sp_sp, Sm_sp)}
_SPARSE_CACHE = {}


def get_spin_operators(S, sparse=False):
    """
    Generate spin-S operators Sz, S+, S-.

    Parameters
    ----------
    S : float
        Spin quantum number (0.5 for spin-1/2, 1.0 for spin-1, etc.).
    sparse : bool, optional
        If True, return scipy.sparse CSR matrices (default: False).

    Returns
    -------
    Sz : ndarray or csr_matrix
        Diagonal operator with eigenvalues m = S, S-1, ..., -S.
    Sp : ndarray or csr_matrix
        Raising operator S+, maps |m> to |m+1> (super-diagonal).
    Sm : ndarray or csr_matrix
        Lowering operator S-, maps |m> to |m-1> (sub-diagonal).

    Notes
    -----
    - S=1/2 and S=1 use pre-computed matrices for O(1) access.
    - Sparse versions are cached after first creation.
    - Dense returns are copies; sparse returns are cached references.
    - Matrix dimension: (2S+1) x (2S+1).

    Examples
    --------
    >>> Sz, Sp, Sm = get_spin_operators(0.5)
    >>> Sz.shape
    (2, 2)
    >>> Sz, Sp, Sm = get_spin_operators(1.0, sparse=True)
    >>> sp.issparse(Sz)
    True
    """
    # Fast path: S = 1/2
    if S == 0.5:
        if sparse:
            if (0.5, True) not in _SPARSE_CACHE:
                _SPARSE_CACHE[(0.5, True)] = (
                    sp.csr_matrix(_SZ_HALF),
                    sp.csr_matrix(_SP_HALF),
                    sp.csr_matrix(_SM_HALF),
                )
            return _SPARSE_CACHE[(0.5, True)]
        return _SZ_HALF.copy(), _SP_HALF.copy(), _SM_HALF.copy()

    # Fast path: S = 1
    if S == 1.0:
        if sparse:
            if (1.0, True) not in _SPARSE_CACHE:
                _SPARSE_CACHE[(1.0, True)] = (
                    sp.csr_matrix(_SZ_ONE),
                    sp.csr_matrix(_SP_ONE),
                    sp.csr_matrix(_SM_ONE),
                )
            return _SPARSE_CACHE[(1.0, True)]
        return _SZ_ONE.copy(), _SP_ONE.copy(), _SM_ONE.copy()

    # General case: arbitrary spin S
    d = int(2 * S + 1)
    m_values = S - np.arange(d, dtype=np.float64)  # m = S, S-1, ..., -S

    # S+/S- coefficients: sqrt[S(S+1) - m(m+1)] for m -> m+1 (same for both by symmetry)
    Sp_coeffs = np.sqrt(S * (S + 1) - m_values[1:] * (m_values[1:] + 1))

    if sparse:
        Sz = sp.diags(m_values, offsets=0, format="csr")
        Sp = sp.diags(Sp_coeffs, offsets=1, shape=(d, d), format="csr")
        return Sz, Sp, Sp.T.tocsr()

    return np.diag(m_values), np.diag(Sp_coeffs, k=1), np.diag(Sp_coeffs, k=-1)
