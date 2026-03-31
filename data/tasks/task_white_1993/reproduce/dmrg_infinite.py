"""
Infinite System DMRG Algorithm

This module provides the infinite system density-matrix algorithm for 1D quantum systems.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section V.A, Table II

Module Organization:
  1. infinite_dmrg     - Main algorithm function (Table II implementation)

Superblock Configuration (Figure 1):
  B_l . . B_l^R  (two free sites in middle, blocks at ends)
    - B_l: left block of length l
    - B_l^R: reflected right block of length l
    - Total length: L = 2l + 2 (grows by 2 per iteration)

Algorithm (Table II):
  Step 1:  Initialize single-site blocks (4 sites: B_1 . . B_1^R)
  Step 2:  Form superblock H = H_L (x) I + I (x) H_R + J * (S_r^L . S_l^R)
  Step 3:  Diagonalize H -> ground state psi (Davidson/Lanczos)
  Step 4:  Form reduced density matrix rho = Tr_R(|psi><psi|) [Eq. 12]
  Step 5:  Diagonalize rho -> eigenvalues w_alpha, eigenvectors u^alpha
  Step 6:  Keep m largest eigenvalues/eigenvectors
  Step 7:  Transform operators: A' = O A O^T where O = [u^1, ..., u^m]^T
  Steps 8-9: Replace blocks with truncated versions
  Step 10: Go to step 2 (chain grows by 2 sites per iteration)

Convergence:
  - Each iteration pushes boundaries farther from center
  - After many steps, each block represents half of infinite chain
  - Block adapts to respond to infinite chain connected to it

Key Results (Section VII):
  S=1/2 Heisenberg chain (Bethe ansatz):
    - E/L -> 1/4 - ln(2) = -0.443147... as L -> infinity
    - Truncation error eps < 10^-7 achievable with m = 20 (OBC)

  S=1 Heisenberg chain:
    - Haldane gap Delta = 0.4105(1)
    - Truncation error eps < 10^-7 achievable with m = 48 (OBC)

Figure Support:
  - Figure 2: rho_eigenvalues (w_alpha vs alpha, steplike structure from spin degeneracies)
"""

import numpy as np
from block import Block
from superblock import Superblock


def infinite_dmrg(
    spin,
    chi,
    num_iterations,
    J=1.0,
    boundary="open",
    Sz_target=0,
    sparse=False,
    verbose=True,
):
    """
    Run infinite system DMRG algorithm (Table II).

    Grows chain by 2 sites per iteration, converging to infinite chain.
    At each step, diagonalizes B_l . . B_l^R and forms B_{l+1} from
    the most significant eigenstates of the reduced density matrix.

    Parameters
    ----------
    spin : float
        Spin quantum number (0.5 for spin-1/2, 1.0 for spin-1).
    chi : int
        Maximum number of states to keep (bond dimension m).
    num_iterations : int
        Number of DMRG iterations. Final length = 2 * (num_iterations + 1).
    J : float, optional
        Heisenberg exchange coupling constant (default: 1.0).
    boundary : {"open", "periodic"}, optional
        Boundary condition type (default: "open").
    Sz_target : float or None, optional
        Target total Sz quantum number sector (default: None = full space).
        Use None to search full Hilbert space without assuming ground state sector.
    sparse : bool, optional
        Use sparse matrices (default: False).
    verbose : bool, optional
        Print iteration progress (default: True).

    Returns
    -------
    dict
        Results dictionary with keys:
        - block_storage : dict mapping {length: Block} for finite DMRG
        - energy_per_site : float, final E_0(L)/L
        - rho_eigenvalues : ndarray, final w_alpha (Figure 2 data)
        - length : int, final chain length L

    Notes
    -----
    Algorithm convergence:
      - Converges in two senses simultaneously: block length -> infinity
        and block adapted to infinite chain environment
      - Open BCs much more accurate than periodic (m vs m^2 states needed)

    Recommended settings:
      - sparse=True when chi > 50 (memory efficiency)
      - Sz_target=0 for antiferromagnetic ground state
      - chi >= 20 for S=1/2, chi >= 50 for S=1 (eps < 10^-7)
    """
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary}")

    # Step 1: Initialize single-site block (Table II, Step 1)
    block = Block.single_site(spin, sparse=sparse)

    # Result storage
    energy_per_site = np.nan
    rho_eigenvalues = np.array([])
    length = 2
    block_storage = {}
    block_storage[1] = block.copy()

    if verbose:
        matrix_type = "sparse" if sparse else "dense"
        bc_label = "OBC" if boundary == "open" else "PBC"
        print(f"Infinite DMRG ({matrix_type}, {bc_label}): S={spin}, chi={chi}, J={J}")
        print("-" * 60)

    # Main iteration loop (Table II, Steps 2-10)
    for iteration in range(num_iterations):
        # Step 2a: Enlarge left block: B_l -> B_{l+1} = B_l .
        left_block = block.enlarge_right(J)

        # Step 2b: Form right block as reflection: B_l^R -> . B_l^R
        if boundary == "open":
            right_block = block.reflect().enlarge_left(J)
        else:
            # PBC: use B_l . B_l^R . configuration (Section V.C)
            right_block = block.reflect().enlarge_right(J)

        # Step 2c: Form superblock Hamiltonian (Eq. 2)
        superblock = Superblock(
            left_block=left_block,
            right_block=right_block,
            J=J,
            boundary=boundary,
            sparse=sparse,
        )

        length = superblock.total_length

        # Step 3: Diagonalize H -> ground state psi (Davidson/Lanczos)
        try:
            E, psi, sector_indices = superblock.diagonalize(Sz_target=Sz_target)
            E0 = E[0]
            psi_valid = True
        except ValueError:
            E0 = np.nan
            psi_valid = False

        energy_per_site = E0 / length

        if not psi_valid:
            break

        # Step 4: Form reduced density matrix rho (Eq. 12)
        rho = superblock.compute_density_matrix(psi[:, 0], sector_indices)

        # Steps 5-6: Diagonalize rho, keep chi largest w_alpha
        truncation_operator, rho_eigenvalues, Sz_kept = superblock.truncate(rho, chi)

        # Step 7: Transform to truncated basis: A' = O A O^T
        block = left_block.truncate(truncation_operator, Sz_new=Sz_kept)
        # Store for finite DMRG (Table III requires B_1 to B_{L-3})
        block_storage[block.length] = block.copy()

        if verbose:
            print(
                f"Iter {iteration + 1:3d}: L={length:4d}, chi={block.bond_dim:4d}, "
                f"E/L={energy_per_site:.10f}"
            )

    if verbose:
        print("-" * 60)
        print(f"Final: L={length}, E/L={energy_per_site:.10f}")

    result = {
        "block_storage": block_storage,
        "energy_per_site": energy_per_site,
        "rho_eigenvalues": rho_eigenvalues,
        "length": length,
    }

    return result


def infinite_dmrg_soft_bc(
    spin,
    chi,
    num_iterations,
    J=1.0,
    J_edge=None,
    boundary="open",
    Sz_target=None,
    sparse=False,
    verbose=True,
):
    """
    Run infinite system DMRG with soft boundary conditions (Table II).

    This variant supports different coupling strengths at the chain edges,
    which is needed for Figure 6(c) "soft" boundary conditions where
    J_edge = 0.236 is used to reduce boundary effects.

    Parameters
    ----------
    spin : float
        Spin quantum number (0.5 for spin-1/2, 1.0 for spin-1).
    chi : int
        Maximum number of states to keep (bond dimension m).
    num_iterations : int
        Number of DMRG iterations. Final length = 2 * (num_iterations + 1).
    J : float, optional
        Interior Heisenberg exchange coupling constant (default: 1.0).
    J_edge : float or None, optional
        Edge coupling for bonds (0,1) and (L-2,L-1) (default: None = use J).
        Figure 6(c) uses J_edge = 0.236.
    boundary : {"open", "periodic"}, optional
        Boundary condition type (default: "open").
    Sz_target : float or None, optional
        Target total Sz quantum number sector (default: None = full space).
        Use None to search full Hilbert space without assuming ground state sector.
    sparse : bool, optional
        Use sparse matrices (default: False).
    verbose : bool, optional
        Print iteration progress (default: True).

    Returns
    -------
    dict
        Results dictionary with keys:
        - block_storage : dict mapping {length: Block} for finite DMRG
        - energy_per_site : float, final E_0(L)/L
        - rho_eigenvalues : ndarray, final w_alpha (Figure 2 data)
        - length : int, final chain length L

    Notes
    -----
    Soft boundary conditions (Section VII, Figure 6c):
        By using a weaker coupling at the edges, the boundary perturbation
        is reduced, leading to more uniform bond strengths in the bulk.
        The optimal J_edge depends on the system but J_edge ~ 0.2-0.3 is
        typical for S=1/2 chains.

    Edge bond handling:
        - Bond 0: between site 0 and site 1 (uses J_edge)
        - Bond L-2: between site L-2 and site L-1 (uses J_edge)
        - All other bonds use J
    """
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary}")

    # Use J for edge bonds if J_edge not specified
    if J_edge is None:
        J_edge = J

    # Step 1: Initialize single-site block (Table II, Step 1)
    block = Block.single_site(spin, sparse=sparse)

    # Result storage
    energy_per_site = np.nan
    rho_eigenvalues = np.array([])
    length = 2
    block_storage = {}
    block_storage[1] = block.copy()

    if verbose:
        matrix_type = "sparse" if sparse else "dense"
        bc_label = "OBC" if boundary == "open" else "PBC"
        print(
            f"Infinite DMRG Soft BC ({matrix_type}, {bc_label}): "
            f"S={spin}, chi={chi}, J={J}, J_edge={J_edge}"
        )
        print("-" * 60)

    # Main iteration loop (Table II, Steps 2-10)
    for iteration in range(num_iterations):
        # Edge bond coupling: only applies on first enlargement (iteration 0)
        # for open boundary conditions. Both left and right blocks start from
        # the same block, so both edge bonds are created simultaneously.
        if boundary == "open" and iteration == 0:
            J_enlarge = J_edge  # Edge bonds (0,1) and (L-2,L-1)
        else:
            J_enlarge = J

        # Step 2a: Enlarge left block: B_l -> B_{l+1} = B_l .
        left_block = block.enlarge_right(J_enlarge)

        # Step 2b: Form right block as reflection: B_l^R -> . B_l^R
        if boundary == "open":
            right_block = block.reflect().enlarge_left(J_enlarge)
        else:
            right_block = block.reflect().enlarge_right(J)

        # Step 2c: Form superblock Hamiltonian (Eq. 2)
        # Center bond always uses J (interior bond)
        superblock = Superblock(
            left_block=left_block,
            right_block=right_block,
            J=J,
            boundary=boundary,
            sparse=sparse,
        )

        length = superblock.total_length

        # Step 3: Diagonalize H -> ground state psi (Davidson/Lanczos)
        try:
            E, psi, sector_indices = superblock.diagonalize(Sz_target=Sz_target)
            E0 = E[0]
            psi_valid = True
        except ValueError:
            E0 = np.nan
            psi_valid = False

        energy_per_site = E0 / length

        if not psi_valid:
            break

        # Step 4: Form reduced density matrix rho (Eq. 12)
        rho = superblock.compute_density_matrix(psi[:, 0], sector_indices)

        # Steps 5-6: Diagonalize rho, keep chi largest w_alpha
        truncation_operator, rho_eigenvalues, Sz_kept = superblock.truncate(rho, chi)

        # Step 7: Transform to truncated basis: A' = O A O^T
        block = left_block.truncate(truncation_operator, Sz_new=Sz_kept)
        # Store for finite DMRG (Table III requires B_1 to B_{L-3})
        block_storage[block.length] = block.copy()

        if verbose:
            print(
                f"Iter {iteration + 1:3d}: L={length:4d}, chi={block.bond_dim:4d}, "
                f"E/L={energy_per_site:.10f}"
            )

    if verbose:
        print("-" * 60)
        print(f"Final: L={length}, E/L={energy_per_site:.10f}")

    result = {
        "block_storage": block_storage,
        "energy_per_site": energy_per_site,
        "rho_eigenvalues": rho_eigenvalues,
        "length": length,
    }

    return result
