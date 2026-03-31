"""
Finite System DMRG Algorithm

This module provides the finite system density-matrix algorithm for 1D quantum systems.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section V.B, Table III

Module Organization:
  1. finite_dmrg             - Main algorithm function (Table III implementation)
  2. finite_dmrg_multitarget - Multi-target variant for Figure 8 (Section IV.B)
  3. finite_dmrg_measure     - Measurement variant for Figure 6 & 7 (local observables)

Superblock Configuration (Figure 1):
  B_l . . B_{L-l-2}^R  (fixed total length L)
    - B_l: left block of length l
    - B_{L-l-2}^R: reflected right block of length L-l-2
    - Two free sites (.) in middle
    - Total length: L = l + 2 + (L-l-2) = L (constant throughout)

Algorithm (Table III):
  Step 1:  (First half of I=1) Use infinite system algorithm for L/2-1 steps
           to build up lattice to L sites. Store block Hamiltonians and
           end operator matrices for B_1, B_2, ..., B_{L/2}.
  Step 2:  (Start of second half of I=1) Set l = L/2. Use B_l as block 1,
           and reflection of B_{L-l-2} as block 4.
  Step 3:  Steps 2-8 of Table II (form superblock, diagonalize, density matrix,
           truncate, transform operators).
  Step 4:  Store new block 1 as B_{l+1}, replacing old B_{l+1}.
  Step 5:  Replace block 4 with reflection of B_{L-l-2} from first half.
  Step 6:  If l < L-3, set l = l+1 and go to Step 3.
  Step 7:  (Start of iteration I, I>1) Make four initial blocks: first three
           are single sites, fourth is reflection of B_{L-3} from previous
           iteration. Set l = 1.
  Step 8:  Steps 2-8 of Table II.
  Step 9:  Store new block 1 as B_{l+1}, replacing old B_{l+1}.
  Step 10: Replace block 4 with reflection of B_{L-l-2}, from previous iteration
           (if l <= L/2-1) or first half of current iteration (if l > L/2-1).
  Step 11: If l < L-3, set l = l+1 and go to Step 8. If l = L-3, start new
           iteration by going to Step 7. (Stop after 2 or 3 iterations.)

Convergence (Section V.B):
  - After a few iterations, each B_l accurately represents left l sites
    of an L-site chain
  - Usually converges by middle of second iteration
  - 2-3 iterations typically sufficient
  - On last iteration, stop after diagonalizing B_{L/2-1} . . B_{L/2-1}^R
    and use this wave function to measure properties

Efficiency (Section V.B):
  - Can use m' < m states for right block (factor 2-3 speedup)
  - Right block only helps form density matrix; left block becomes new block
  - Keep m' states in both blocks except last iteration
  - Typically m/m' = 2 or 3; should not exceed m > n*m' where n = site states

Key Results (Section VII):
  S=1/2 Heisenberg chain (L=28, OBC):
    - Relative error < 10^-9 with m = 40 (Figure 3)
    - Much more accurate than infinite DMRG for finite systems

  S=1 Heisenberg chain (L=16, OBC):
    - Relative error < 10^-9 with m = 80 (Figure 4)

Figure Support:
  - Figure 2: Density-matrix eigenvalues w_alpha (32-site, OBC/PBC, S=1/2 and S=1)
  - Figure 3: Relative error Delta E/|E| vs m (28-site S=1/2, OBC vs PBC)
  - Figure 4: Relative error Delta E/|E| vs m (16-site S=1, OBC vs PBC)
  - Figure 5: Energy gap Delta_L vs 1/L (S=1/2 and S=1, PBC)
  - Figure 6: Local bond strength <S_i . S_{i+1}> (60-site S=1/2, OBC)
              -> finite_dmrg_measure
  - Figure 7: Local bond strength and magnetization (60-site S=1, OBC)
              -> finite_dmrg_measure with Sz_target=0/1
  - Figure 8: Density-matrix eigenvalues w_alpha (32-site S=1/2, OBC, 1-5 targets)
              -> finite_dmrg_multitarget
"""

import numpy as np
from dmrg_infinite import infinite_dmrg, infinite_dmrg_soft_bc
from superblock import Superblock


def finite_dmrg(
    spin,
    chi,
    L,
    num_sweeps=3,
    J=1.0,
    boundary="open",
    Sz_target=0,
    sparse=False,
    verbose=True,
):
    """
    Run finite system DMRG algorithm (Table III).

    Performs sweeping on a fixed-length L-site chain, converging to accurate
    representation of finite system ground state(s). At each step, uses
    superblock B_l . . B_{L-l-2}^R to form B_{l+1}.

    Parameters
    ----------
    spin : float
        Spin quantum number (0.5 for spin-1/2, 1.0 for spin-1).
    chi : int
        Maximum number of states to keep (bond dimension m).
    L : int
        Chain length (must be even and >= 4).
    num_sweeps : int, optional
        Number of full iterations (default: 3). Usually converges by middle
        of second iteration; 2-3 iterations typically sufficient.
    J : float, optional
        Heisenberg exchange coupling constant (default: 1.0).
    boundary : {"open", "periodic"}, optional
        Boundary condition type (default: "open").
        Open BCs much more accurate than periodic (m vs m^2 states needed).
    Sz_target : float, optional
        Target total Sz quantum number sector (default: 0).
    sparse : bool, optional
        Use sparse matrices (default: False).
    verbose : bool, optional
        Print sweep progress (default: True).

    Returns
    -------
    dict
        Results dictionary with keys:
        - energy_per_site : ndarray, shape (num_states,), E_k / L for each state
        - rho_eigenvalues : ndarray, final density matrix eigenvalues w_alpha

    Notes
    -----
    Algorithm structure (Table III):
      - Step 1: L/2-1 infinite DMRG steps to build up to L sites, storing
        block Hamiltonians B_1 to B_{L/2}
      - Steps 2-6: Second half of first iteration, l = L/2 to L-3
      - Steps 7-11: Subsequent iterations, l = 1 to L-3
      - Final: Stop at symmetric configuration B_{L/2-1} . . B_{L/2-1}^R
        and use wave function to measure properties

    Block storage strategy:
      - Store L-3 blocks: B_1 to B_{L-3}
      - Once new B_l formed, it replaces old B_l (only one set stored)
      - For l > L/2-1, use block from current iteration as right block

    Convergence criteria:
      - Energy change < 10^-10 between iterations indicates convergence
      - Truncation error eps = 1 - sum(kept w_alpha) should decrease
      - Symmetric configuration gives most accurate results (Section VI)

    Memory requirements:
      - Stores L/2 blocks: B_1 to B_{L/2}
      - Each block: ~5 * chi^2 * 8 bytes (dense)
    """
    if L < 4 or L % 2 != 0:
        raise ValueError(f"L must be even and >= 4, got {L}")
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary}")

    # -------------------------------------------------------------------------
    # Step 1: Initialize using infinite system algorithm (Table III, Step 1)
    # -------------------------------------------------------------------------
    # Use infinite DMRG for L/2-1 steps to build up lattice to L sites.
    # Store block Hamiltonians and end operator matrices for B_1 to B_{L/2}.
    num_infinite_steps = L // 2 - 1

    if verbose:
        matrix_type = "sparse" if sparse else "dense"
        bc_label = "OBC" if boundary == "open" else "PBC"
        print(
            f"Finite DMRG ({matrix_type}, {bc_label}): S={spin}, L={L}, chi={chi}, J={J}"
        )
        print("=" * 60)
        print(f"Phase 1: Infinite DMRG warmup ({num_infinite_steps} steps)")
        print("-" * 60)

    infinite_result = infinite_dmrg(
        spin=spin,
        chi=chi,
        num_iterations=num_infinite_steps,
        J=J,
        boundary=boundary,
        sparse=sparse,
        verbose=verbose,
    )

    # Result storage
    # Infinite DMRG provides initial approximate versions of B_1 to B_{L/2}
    energy_per_site = infinite_result["energy_per_site"]
    rho_eigenvalues = infinite_result.get("rho_eigenvalues", np.array([]))
    block_storage = infinite_result["block_storage"]
    # -------------------------------------------------------------------------
    # Sweeping phase (Table III, Steps 2-11)
    # -------------------------------------------------------------------------
    # First iteration (I=1):
    #   - First half: handled by infinite DMRG above
    #   - Second half: l = L/2 to L-3, using B_{L-l-2} from first half
    # Subsequent iterations (I>1):
    #   - l = 1 to L-3, using B_{L-l-2} from previous iteration
    #   - For l > L/2-1, use block from current iteration
    # Final iteration:
    #   - Stop at symmetric config B_{L/2-1} . . B_{L/2-1}^R
    # -------------------------------------------------------------------------
    if verbose:
        print("-" * 60)
        print(f"Phase 2: Finite DMRG sweeps ({num_sweeps} iterations)")
        print("-" * 60)

    for iteration in range(num_sweeps):
        # Determine sweep range for this iteration (Table III)
        if iteration == 0:
            left_start, left_end = L // 2, L - 3
        elif iteration == num_sweeps - 1:
            left_start, left_end = 1, L // 2  # Stop at symmetric config
        else:
            left_start, left_end = 1, L - 3  # Full sweep

        left_range = list(range(left_start, left_end))

        # Skip empty or invalid ranges (can happen for small L)
        if left_start >= left_end:
            left_range = []

        if verbose:
            print(f"\nIteration {iteration + 1} (l={left_start} to {left_end}):")

        if len(left_range) == 0:
            if verbose:
                print("  Skipped (empty range)")
            continue

        for left_len in left_range:
            # Step 2/7: Get left block B_l (Table III)
            if left_len not in block_storage:
                raise RuntimeError(f"Block B_{left_len} not found in storage")
            left_base = block_storage[left_len]
            # Step 5/10: Get right block as reflection of B_{L-l-2}
            right_len = L - left_len - 2
            if right_len not in block_storage:
                raise RuntimeError(f"Block B_{right_len} not found in storage")
            right_base = block_storage[right_len]

            # Use block from previous iteration (if l <= L/2-1) or
            # first half of current iteration (if l > L/2-1)
            # Enlarge blocks: B_l -> B_l . and B_r^R -> . B_r^R (Figure 1)
            left_block = left_base.enlarge_right(J)
            if boundary == "open":
                right_block = right_base.reflect().enlarge_left(J)
            else:
                right_block = right_base.reflect().enlarge_right(J)

            # Step 3/8a: Form superblock B_l . . B_{L-l-2}^R (total L sites)
            superblock = Superblock(
                left_block=left_block,
                right_block=right_block,
                J=J,
                boundary=boundary,
                sparse=sparse,
            )

            # Step 3/8b: Diagonalize superblock -> target state psi
            # (Table II, Steps 2-3: form H, Davidson/Lanczos diagonalization)
            try:
                E, psi, sector_indices = superblock.diagonalize(Sz_target=Sz_target)
                E0 = E[0]
                psi_valid = True
            except ValueError:
                E0 = np.nan
                psi_valid = False

            energy_per_site = E0 / L

            if not psi_valid:
                break

            # Step 3/8c: Form density matrix and truncate (Table II, Steps 4-7)
            # rho = Tr_R(|psi><psi|), keep m largest eigenvalues/eigenvectors
            rho = superblock.compute_density_matrix(psi[:, 0], sector_indices)
            truncation_operator, rho_eigenvalues, Sz_kept = superblock.truncate(
                rho, chi
            )

            # Step 4/9: Store new block B_{l+1}, replacing old B_{l+1}
            # Transform: H' = O H O^T, etc.
            block = left_block.truncate(truncation_operator, Sz_new=Sz_kept)
            block_storage[block.length] = block.copy()

            if verbose and left_len % max(1, (L // 2 - 1) // 4) == 0:
                print(
                    f"  l={left_len:3d}: E/L={energy_per_site:.10f}, chi={block.bond_dim:4d}"
                )

        if verbose:
            print(f"  Iteration {iteration + 1} done: E/L = {energy_per_site:.10f}")

    # B_{L/2-1} . . B_{L/2-1}^R gives most accurate results for measurements

    if verbose:
        print(f"Final: E/L = {energy_per_site:.10f}")
        print("=" * 60)

    result = {
        "energy_per_site": energy_per_site,
        "rho_eigenvalues": rho_eigenvalues,
    }

    return result


def finite_dmrg_multitarget(
    spin,
    chi,
    L,
    num_target_states=1,
    num_sweeps=3,
    J=1.0,
    boundary="open",
    Sz_target=0,
    sparse=False,
    verbose=True,
):
    """
    Run finite system DMRG with multiple target states (Table III, Section IV.B).

    This variant targets multiple low-lying eigenstates simultaneously.
    The density matrix is formed by summing over all target states:
        rho = Sum_{k=1}^{n_target} |psi_k><psi_k|

    This is needed for Figure 8 reproduction, which shows density-matrix
    eigenvalues for various numbers of target states (1, 2, 3, 4, 5).

    From Section VI: "The renormalization group method is most accurate when
    only one state is used as a target. Figure 8 shows the behavior of the
    density matrix eigenvalues w_alpha of a typical system as the number of
    target states is varied."

    Parameters
    ----------
    spin : float
        Spin quantum number (0.5 for spin-1/2, 1.0 for spin-1).
    chi : int
        Maximum number of states to keep (bond dimension m).
    L : int
        Chain length (must be even and >= 4).
    num_target_states : int, optional
        Number of lowest eigenstates to target (default: 1).
        The density matrix sums contributions from all target states.
    num_sweeps : int, optional
        Number of full iterations (default: 3).
    J : float, optional
        Heisenberg exchange coupling constant (default: 1.0).
    boundary : {"open", "periodic"}, optional
        Boundary condition type (default: "open").
    Sz_target : float, optional
        Target total Sz quantum number sector (default: 0).
    sparse : bool, optional
        Use sparse matrices (default: False).
    verbose : bool, optional
        Print sweep progress (default: True).

    Returns
    -------
    dict
        Results dictionary with keys:
        - energy_per_site : ndarray, shape (num_target_states,), E_k / L for each state
        - rho_eigenvalues : ndarray, final density matrix eigenvalues w_alpha

    Notes
    -----
    Multi-target density matrix (Section IV.B):
        When targeting n states psi_1, ..., psi_n, the reduced density matrix is:
            rho = (1/n) * Sum_{k=1}^{n} |psi_k><psi_k|
        or equivalently (unnormalized):
            rho = Sum_{k=1}^{n} Psi_k @ Psi_k.T

        The eigenvalues w_alpha from this combined rho determine which basis
        states are most important across ALL target states.

    Figure 8 context:
        "When a greater number of states are targeted for a fixed value of m,
        the size of the w_alpha which are discarded is increased, and there
        is a reduction in the accuracy."

        This is because more states compete for the limited chi slots,
        leading to larger truncation errors per state.
    """
    if L < 4 or L % 2 != 0:
        raise ValueError(f"L must be even and >= 4, got {L}")
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary}")
    if num_target_states < 1:
        raise ValueError(f"num_target_states must be >= 1, got {num_target_states}")

    # -------------------------------------------------------------------------
    # Step 1: Initialize using infinite system algorithm (Table III, Step 1)
    # -------------------------------------------------------------------------
    num_infinite_steps = L // 2 - 1

    if verbose:
        matrix_type = "sparse" if sparse else "dense"
        bc_label = "OBC" if boundary == "open" else "PBC"
        print(
            f"Finite DMRG Multi-Target ({matrix_type}, {bc_label}): "
            f"S={spin}, L={L}, chi={chi}, targets={num_target_states}"
        )
        print("=" * 60)
        print(f"Phase 1: Infinite DMRG warmup ({num_infinite_steps} steps)")
        print("-" * 60)

    infinite_result = infinite_dmrg(
        spin=spin,
        chi=chi,
        num_iterations=num_infinite_steps,
        J=J,
        boundary=boundary,
        sparse=sparse,
        verbose=verbose,
    )

    # Result storage
    energy_per_site = np.full(num_target_states, infinite_result["energy_per_site"])
    rho_eigenvalues = infinite_result.get("rho_eigenvalues", np.array([]))
    block_storage = infinite_result["block_storage"]

    # -------------------------------------------------------------------------
    # Sweeping phase with multi-target density matrix
    # -------------------------------------------------------------------------
    if verbose:
        print("-" * 60)
        print(f"Phase 2: Finite DMRG sweeps ({num_sweeps} iterations)")
        print(f"         Targeting {num_target_states} lowest states")
        print("-" * 60)

    for iteration in range(num_sweeps):
        # Determine sweep range for this iteration (Table III)
        if iteration == 0:
            left_start, left_end = L // 2, L - 3
        elif iteration == num_sweeps - 1:
            left_start, left_end = 1, L // 2  # Stop at symmetric config
        else:
            left_start, left_end = 1, L - 3  # Full sweep

        left_range = list(range(left_start, left_end))

        # Skip empty or invalid ranges
        if left_start >= left_end:
            left_range = []

        if verbose:
            print(f"\nIteration {iteration + 1} (l={left_start} to {left_end}):")

        if len(left_range) == 0:
            if verbose:
                print("  Skipped (empty range)")
            continue

        for left_len in left_range:
            # Get left block B_l
            if left_len not in block_storage:
                raise RuntimeError(f"Block B_{left_len} not found in storage")
            left_base = block_storage[left_len]

            # Get right block as reflection of B_{L-l-2}
            right_len = L - left_len - 2
            if right_len not in block_storage:
                raise RuntimeError(f"Block B_{right_len} not found in storage")
            right_base = block_storage[right_len]

            # Enlarge blocks
            left_block = left_base.enlarge_right(J)
            if boundary == "open":
                right_block = right_base.reflect().enlarge_left(J)
            else:
                right_block = right_base.reflect().enlarge_right(J)

            # Form superblock
            superblock = Superblock(
                left_block=left_block,
                right_block=right_block,
                J=J,
                boundary=boundary,
                sparse=sparse,
            )

            # Diagonalize for multiple target states
            try:
                E, psi, sector_indices = superblock.diagonalize(
                    num_states=num_target_states, Sz_target=Sz_target
                )
                psi_valid = True
            except ValueError:
                E = np.full(num_target_states, np.nan)
                psi_valid = False

            energy_per_site = E / L

            if not psi_valid:
                break

            # Compute multi-target density matrix: rho = Sum_k |psi_k><psi_k|
            # Section IV.B: sum contributions from all target states
            rho = np.zeros((superblock.dim_left, superblock.dim_left))
            num_actual_states = psi.shape[1]

            for k in range(num_actual_states):
                rho_k = superblock.compute_density_matrix(psi[:, k], sector_indices)
                rho += rho_k

            # Normalize: Tr(rho) = 1
            rho /= np.trace(rho)

            # Truncate using combined density matrix
            truncation_operator, rho_eigenvalues, Sz_kept = superblock.truncate(
                rho, chi
            )

            # Store new block
            block = left_block.truncate(truncation_operator, Sz_new=Sz_kept)
            block_storage[block.length] = block.copy()

            if verbose and left_len % max(1, (L // 2 - 1) // 4) == 0:
                print(
                    f"  l={left_len:3d}: E0/L={energy_per_site[0]:.10f}, chi={block.bond_dim:4d}"
                )

        if verbose:
            print(f"  Iteration {iteration + 1} done: E0/L = {energy_per_site[0]:.10f}")

    if verbose:
        print(f"Final: E0/L = {energy_per_site[0]:.10f}")
        print("=" * 60)

    result = {
        "energy_per_site": energy_per_site,
        "rho_eigenvalues": rho_eigenvalues,
    }

    return result


def finite_dmrg_measure(
    spin,
    chi,
    L,
    num_sweeps=3,
    J=1.0,
    J_edge=None,
    boundary="open",
    Sz_target=None,
    sparse=False,
    verbose=True,
):
    """
    Run finite system DMRG with local observable measurements.

    This variant performs a measurement sweep after convergence to compute:
    - Local bond strength: <S_i · S_{i+1}> for all bonds
    - Local magnetization: <S_i^z> for all sites

    Needed for Figure 6 and Figure 7 reproduction:
    - Figure 6: Local bond strength for S=1/2 chains (L=60, 61)
    - Figure 7: Local bond strength and magnetization for S=1 chain (L=60)

    Parameters
    ----------
    spin : float
        Spin quantum number (0.5 for spin-1/2, 1.0 for spin-1).
    chi : int
        Maximum number of states to keep (bond dimension m).
    L : int
        Chain length (must be even and >= 4 for standard DMRG).
    num_sweeps : int, optional
        Number of full iterations before measurement (default: 3).
    J : float, optional
        Heisenberg exchange coupling constant (default: 1.0).
    J_edge : float or None, optional
        Edge coupling for "soft" boundary conditions (default: None = use J).
        If specified, J_edge is used for bonds (0,1) and (L-2,L-1).
        Figure 6(c) uses J_edge = 0.236.
    boundary : {"open", "periodic"}, optional
        Boundary condition type (default: "open").
    Sz_target : float or None, optional
        Target total Sz quantum number sector (default: None = full space).
        Use None to search full Hilbert space without assuming ground state sector.
        For Figure 7(b), use Sz_target=1 to get triplet state.
    sparse : bool, optional
        Use sparse matrices (default: False).
    verbose : bool, optional
        Print sweep progress (default: True).

    Returns
    -------
    dict
        Results dictionary with keys:
        - energy_per_site : float, E_0 / L
        - bond_strength : ndarray, shape (L-1,), <S_i · S_{i+1}> for each bond
        - local_Sz : ndarray, shape (L,), <S_i^z> for each site
        - rho_eigenvalues : ndarray, final density matrix eigenvalues

    Notes
    -----
    Measurement approach:
        During the measurement sweep (left to right then right to left),
        at each superblock configuration B_l . . B_r^R, we measure:
        - Center bond: <S_l · S_{l+1}> between the two free sites
        - Center sites: <S_l^z> and <S_{l+1}^z>

        By sweeping through all positions, we obtain all local observables.

    Figure 6 context:
        "The effect of an open boundary is most easily seen for an S=1/2
        system by measuring the local bond strength <S_j · S_{j+1}>."

    Figure 7 context:
        "Figure 7(b) shows the local spin magnetization for one of the
        triplet states of a 60 site chain."
    """
    if L < 4:
        raise ValueError(f"L must be >= 4, got {L}")
    if boundary not in ("open", "periodic"):
        raise ValueError(f"boundary must be 'open' or 'periodic', got {boundary}")

    # Use J for edge bonds if J_edge not specified
    if J_edge is None:
        J_edge = J

    # -------------------------------------------------------------------------
    # Phase 1: Run standard finite DMRG to converge
    # -------------------------------------------------------------------------

    num_infinite_steps = (L - 1) // 2  # Works for both even and odd L

    if verbose:
        matrix_type = "sparse" if sparse else "dense"
        bc_label = "OBC" if boundary == "open" else "PBC"
        print(
            f"Finite DMRG Measure ({matrix_type}, {bc_label}): "
            f"S={spin}, L={L}, chi={chi}, J={J}"
        )
        if J_edge != J:
            print(f"  Soft BCs: J_edge={J_edge}")
        print("=" * 60)
        print(f"Phase 1: Infinite DMRG warmup ({num_infinite_steps} steps)")
        print("-" * 60)

    infinite_result = infinite_dmrg_soft_bc(
        spin=spin,
        chi=chi,
        num_iterations=num_infinite_steps,
        J=J,
        J_edge=J_edge,
        boundary=boundary,
        Sz_target=Sz_target,
        sparse=sparse,
        verbose=verbose,
    )

    energy_per_site = infinite_result["energy_per_site"]
    rho_eigenvalues = infinite_result.get("rho_eigenvalues", np.array([]))
    block_storage = infinite_result["block_storage"]

    # -------------------------------------------------------------------------
    # Phase 2: Convergence sweeps (same as finite_dmrg)
    # -------------------------------------------------------------------------
    if verbose:
        print("-" * 60)
        print(f"Phase 2: Convergence sweeps ({num_sweeps} iterations)")
        print("-" * 60)

    for iteration in range(num_sweeps):
        if iteration == 0:
            left_start, left_end = L // 2, L - 3
        else:
            left_start, left_end = 1, L - 3

        left_range = list(range(left_start, left_end))
        if left_start > left_end:
            left_range = []

        if verbose:
            print(f"\nIteration {iteration + 1} (l={left_start} to {left_end}):")

        if len(left_range) == 0:
            if verbose:
                print("  Skipped (empty range)")
            continue

        for left_len in left_range:
            # Step 2/7: Get left block B_l (Table III)
            if left_len not in block_storage:
                raise RuntimeError(f"Block B_{left_len} not found in storage")
            left_base = block_storage[left_len]
            # Step 5/10: Get right block as reflection of B_{L-l-2}
            right_len = L - left_len - 2
            if right_len not in block_storage:
                raise RuntimeError(f"Block B_{right_len} not found in storage")
            right_base = block_storage[right_len]

            # Get J for each bond (handle soft BCs for open boundary)
            # Edge bonds only exist in open boundary conditions
            if boundary == "open":
                J_left = J_edge if left_len == 1 else J  # bond 0 is edge
                J_right = J_edge if right_len == 1 else J  # bond L-2 is edge
            else:
                J_left = J
                J_right = J

            left_block = left_base.enlarge_right(J_left)
            if boundary == "open":
                right_block = right_base.reflect().enlarge_left(J_right)
            else:
                right_block = right_base.reflect().enlarge_right(J)

            superblock = Superblock(
                left_block=left_block,
                right_block=right_block,
                J=J,
                boundary=boundary,
                sparse=sparse,
            )

            try:
                E, psi, sector_indices = superblock.diagonalize(Sz_target=Sz_target)
                E0 = E[0]
                psi_valid = True
            except ValueError:
                E0 = np.nan
                psi_valid = False

            energy_per_site = E0 / L

            if not psi_valid:
                break

            rho = superblock.compute_density_matrix(psi[:, 0], sector_indices)
            truncation_operator, rho_eigenvalues, Sz_kept = superblock.truncate(
                rho, chi
            )

            block = left_block.truncate(truncation_operator, Sz_new=Sz_kept)
            block_storage[block.length] = block.copy()

            if verbose and left_len % max(1, (L // 2 - 1) // 4) == 0:
                print(
                    f"  l={left_len:3d}: E/L={energy_per_site:.10f}, chi={block.bond_dim:4d}"
                )

        if verbose:
            print(f"  Iteration {iteration + 1} done: E/L = {energy_per_site:.10f}")

    # -------------------------------------------------------------------------
    # Phase 3: Measurement sweep
    # -------------------------------------------------------------------------
    if verbose:
        print("-" * 60)
        print("Phase 3: Measurement sweep")
        print("-" * 60)

    # Initialize measurement arrays
    bond_strength = np.full(L - 1, np.nan)
    local_Sz = np.full(L, np.nan)

    # Sweep left to right to measure interior bonds and sites
    # At position left_len, superblock has:
    #   - left_block: sites 0 to left_len (enlarged from left_base)
    #   - right_block: sites left_len+1 to L-1 (enlarged from right_base)
    #   - Center bond: between site left_len and left_len+1 (bond index = left_len)
    #
    # Loop measures bonds 1 to L-3 and sites 1 to L-2
    # Edge bonds (0, L-2) and edge sites (0, L-1) measured at boundaries

    for left_len in range(1, L - 2):
        left_base = block_storage[left_len]
        right_len = L - left_len - 2
        right_base = block_storage[right_len]

        # Get J for each bond (handle soft BCs for open boundary)
        if boundary == "open":
            J_left = J_edge if left_len == 1 else J
            J_right = J_edge if right_len == 1 else J
        else:
            J_left = J
            J_right = J

        left_block = left_base.enlarge_right(J_left)
        if boundary == "open":
            right_block = right_base.reflect().enlarge_left(J_right)
        else:
            right_block = right_base.reflect().enlarge_right(J)

        superblock = Superblock(
            left_block=left_block,
            right_block=right_block,
            J=J,
            boundary=boundary,
            sparse=sparse,
        )

        try:
            E, psi, sector_indices = superblock.diagonalize(Sz_target=Sz_target)
            psi_valid = True
        except ValueError:
            psi_valid = False

        if not psi_valid:
            break

        # Batch measurement (optimized: single psi expansion and reshape)
        include_edges = (left_len == 1, right_len == 1)
        measurements = superblock.measure_all(
            psi[:, 0], sector_indices, include_edges=include_edges
        )

        # Center measurements
        bond_strength[left_len] = measurements["center_bond"]
        local_Sz[left_len] = measurements["center_left_Sz"]
        local_Sz[left_len + 1] = measurements["center_right_Sz"]

        # Edge measurements (only at boundaries)
        if left_len == 1:
            local_Sz[0] = measurements["left_edge_Sz"]
            bond_strength[0] = measurements["left_edge_bond"]

        if right_len == 1:
            local_Sz[L - 1] = measurements["right_edge_Sz"]
            bond_strength[L - 2] = measurements["right_edge_bond"]

        if verbose and left_len % max(1, L // 10) == 0:
            print(
                f"  Bond {left_len}: <S·S> = {bond_strength[left_len]:.6f}, "
                f"<Sz_{left_len}> = {local_Sz[left_len]:.6f}"
            )

    if verbose:
        print(f"\nMeasured {np.sum(~np.isnan(bond_strength))} bonds")
        print(f"Measured {np.sum(~np.isnan(local_Sz))} sites")
        print("=" * 60)

    result = {
        "energy_per_site": energy_per_site,
        "bond_strength": bond_strength,
        "local_Sz": local_Sz,
        "rho_eigenvalues": rho_eigenvalues,
    }

    return result
