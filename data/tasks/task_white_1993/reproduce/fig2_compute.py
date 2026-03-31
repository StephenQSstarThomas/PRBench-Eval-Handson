"""
Figure 2 Data Generation: Density-Matrix Eigenvalues

This module computes the density-matrix eigenvalues w_alpha for reproducing
Figure 2 of White's DMRG paper.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section VII, Figure 2

Module Organization:
  1. compute_fig2_data  - Run DMRG for all 4 configurations
  2. save_data          - Export eigenvalues to CSV format

Figure 2 Description (from paper):
  "Figure 2 shows the density-matrix eigenvalues w_alpha for a 32 site system
  for both open and periodic boundary conditions, and for both S=1/2 and S=1,
  targeting one state."

  The eigenvalues were obtained from the symmetric superblock configuration
  B_15 . . B_15^R (i.e., left block of 15 sites + 2 free sites + right block
  of 15 sites reflected).

Figure Specifications:
  - X-axis: alpha (eigenvalue index), linear scale, range 1 to ~50
  - Y-axis: w_alpha (density-matrix eigenvalues), log scale, 10^0 to 10^-7
  - System: L = 32 sites

Data Series (4 curves):
  1. Open BC, S=1/2    - fastest decay (most accurate with small m)
  2. Open BC, S=1      - slightly slower decay than S=1/2
  3. Periodic BC, S=1/2 - slower decay than open cases
  4. Periodic BC, S=1   - slowest decay

Key Results from Paper:
  - S=1/2 Open:   accuracy < 10^-7 with m=20 states
  - S=1 Open:     accuracy ~ 10^-7 with m=48 states
  - Periodic:     accuracy ~ 10^-5 to 10^-6 with m=50 states

Physical Interpretation:
  The density-matrix eigenvalues w_alpha represent the "importance" of each
  basis state for describing the ground state. The truncation error is:

      epsilon = 1 - sum_{alpha=1}^{m} w_alpha = sum_{alpha=m+1}^{infinity} w_alpha

  Faster decay of w_alpha means fewer states m are needed for a given accuracy.
  Open BCs are more accurate because the system is "less entangled" at the
  boundary compared to periodic BCs.

Output Format (CSV):
  alpha, Open S=1/2, Open S=1, Periodic S=1/2, Periodic S=1
  1,     0.498,      0.312,    0.201,          0.156
  2,     0.498,      0.312,    0.201,          0.156
  ...

  Where:
    - alpha: eigenvalue index (1, 2, 3, ...)
    - Open, S=1/2: w_alpha for Open BC, S=1/2
    - Open, S=1: w_alpha for Open BC, S=1
    - Periodic, S=1/2: w_alpha for Periodic BC, S=1/2
    - Periodic, S=1: w_alpha for Periodic BC, S=1

  Eigenvalues are sorted in DECREASING order (largest first).
  NaN values indicate the array was shorter than the maximum length
  (can happen because S=1 has larger local Hilbert space than S=1/2).
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from dmrg_finite import finite_dmrg

# =============================================================================
# DATA COMPUTATION
# =============================================================================


def compute_fig2_data(L=32, chi=100, num_sweeps=3, sparse=True, verbose=True):
    """
    Compute density-matrix eigenvalues for all 4 configurations in Figure 2.

    Runs finite DMRG for each (spin, boundary) combination and extracts the
    density-matrix eigenvalues from the final symmetric configuration.

    Parameters
    ----------
    L : int, optional
        Chain length (default: 32, as specified in Figure 2 caption).
    chi : int, optional
        Bond dimension m (default: 100). Must be large enough to capture
        the eigenvalue spectrum shown in the figure (~50 eigenvalues).
        The density matrix dimension is at most n*chi where n is the
        local Hilbert space dimension (n=2 for S=1/2, n=3 for S=1).
    num_sweeps : int, optional
        Number of finite DMRG sweeps (default: 3). Usually converges by
        the middle of the second sweep.
    sparse : bool, optional
        Use sparse matrix operations (default: True). Improves performance
        for large systems by using scipy.sparse eigensolvers.
    verbose : bool, optional
        Print progress information (default: True).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'Open, S=1/2': ndarray of w_alpha for Open BC, S=1/2
        - 'Open, S=1': ndarray of w_alpha for Open BC, S=1
        - 'Periodic, S=1/2': ndarray of w_alpha for Periodic BC, S=1/2
        - 'Periodic, S=1': ndarray of w_alpha for Periodic BC, S=1

        Each array contains ALL density-matrix eigenvalues (not truncated),
        sorted in decreasing order. Array lengths may differ because the
        density matrix dimension depends on the spin value.

    Notes
    -----
    The eigenvalues are obtained from the symmetric superblock configuration
    B_{L/2-1} . . B_{L/2-1}^R, which gives the most accurate representation
    of the ground state (see Section VI of the paper).

    For L=32, this is B_15 . . B_15^R (15-site left block, 2 free sites,
    15-site right block reflected).
    """
    results = {}

    # Define the 4 configurations: (spin, boundary_condition, result_key)
    # Order matches the legend in Figure 2
    configs = [
        (0.5, "open", "Open, S=1/2"),
        (1.0, "open", "Open, S=1"),
        (0.5, "periodic", "Periodic, S=1/2"),
        (1.0, "periodic", "Periodic, S=1"),
    ]

    for spin, boundary, label in configs:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Computing: S={spin}, {boundary} BC, L={L}")
            print(f"{'=' * 60}")

        # Run finite DMRG to converge the ground state
        result = finite_dmrg(
            spin=spin,
            chi=chi,
            L=L,
            num_sweeps=num_sweeps,
            J=1.0,
            boundary=boundary,
            Sz_target=0,  # Ground state is in Sz=0 sector
            sparse=sparse,
            verbose=verbose,
        )

        # Extract density-matrix eigenvalues
        # These come from superblock.truncate() which returns ALL eigenvalues
        # of the reduced density matrix, sorted in decreasing order
        w_alpha = result["rho_eigenvalues"]

        if verbose:
            print(f"\nObtained {len(w_alpha)} eigenvalues")
            print(f"Top 5: {w_alpha[:5]}")
            print(f"Truncation error (1 - sum): {1 - np.sum(w_alpha):.2e}")

        results[label] = w_alpha

    return results


# =============================================================================
# DATA EXPORT
# =============================================================================


def save_data(results, filepath="data/fig2.csv"):
    """
    Save density-matrix eigenvalues to CSV file.

    Creates a DataFrame with columns for eigenvalue index and each configuration.
    Arrays of different lengths are padded with NaN.

    Parameters
    ----------
    results : dict
        Output from compute_fig2_data(). Keys are configuration labels,
        values are eigenvalue arrays.
    filepath : str, optional
        Output CSV path (default: "data/fig2.csv").

    Returns
    -------
    pd.DataFrame
        The saved DataFrame with columns:
        - alpha: eigenvalue index (1, 2, 3, ...)
        - Open, S=1/2; Open, S=1; Periodic, S=1/2; Periodic, S=1: eigenvalues

    Notes
    -----
    The arrays may have different lengths because the density matrix
    dimension depends on the spin value:
      - S=1/2: local dimension n=2, so max density matrix dim = 2*chi
      - S=1:   local dimension n=3, so max density matrix dim = 3*chi

    Shorter arrays are padded with NaN to create a rectangular DataFrame.
    """
    # Define column order matching Figure 2 legend
    columns = ["Open, S=1/2", "Open, S=1", "Periodic, S=1/2", "Periodic, S=1"]

    # set max length of eigenvalue arrays
    max_len = 50

    # Build data dictionary with alpha index and padded eigenvalue arrays
    data: dict[str, Any] = {}
    data["alpha"] = np.arange(1, max_len + 1)

    for key in columns:
        arr = results[key]
        # Pad shorter arrays with NaN to match max_len
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), np.nan)])
        data[key] = arr[:max_len]

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"\nData saved to {filepath}")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("FIGURE 2 DATA GENERATION")
    print("Reference: White, Phys. Rev. B 48, 10345 (1993)")
    print("=" * 70)

    # Parameters from paper (Section VII, Figure 2 caption)
    L = 32  # 32-site system
    chi = 100  # Keep enough states to show ~50 eigenvalues
    num_sweeps = 3  # Converge the finite DMRG
    sparse = True  # Use sparse matrix operations

    # Compute density-matrix eigenvalues for all 4 configurations
    results = compute_fig2_data(
        L=L, chi=chi, num_sweeps=num_sweeps, sparse=sparse, verbose=True
    )

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "fig2.csv"
    )
    df = save_data(results, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nData shape: {df.shape}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # Validation against paper claims (Section VII)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("VALIDATION (Paper Section VII):")
    print("-" * 70)

    # Paper: "accuracy of better than 10^-7 keeping only m=20 states" (S=1/2 open)
    eigenvalues_open_s_half = results["Open, S=1/2"]
    truncation_error_m20 = 1 - np.sum(eigenvalues_open_s_half[:20])
    print(f"Open, S=1/2, m=20 truncation error: {truncation_error_m20:.2e}")
    print("  Paper claims: < 10^-7")

    # Paper: "accuracy of 10^-7 obtained with m=48" (S=1 open)
    eigenvalues_open_s1 = results["Open, S=1"]
    truncation_error_m48 = 1 - np.sum(eigenvalues_open_s1[:48])
    print(f"Open, S=1, m=48 truncation error: {truncation_error_m48:.2e}")
    print("  Paper claims: ~10^-7")

    # Paper: "m=50 yields accuracy roughly 10^-5 to 10^-6" (periodic cases)
    eigenvalues_periodic_s_half = results["Periodic, S=1/2"]
    eigenvalues_periodic_s1 = results["Periodic, S=1"]
    truncation_error_periodic_s_half_m50 = 1 - np.sum(eigenvalues_periodic_s_half[:50])
    truncation_error_periodic_s1_m50 = 1 - np.sum(eigenvalues_periodic_s1[:50])
    print(
        f"Periodic, S=1/2, m=50 truncation error: {truncation_error_periodic_s_half_m50:.2e}"
    )
    print(
        f"Periodic, S=1, m=50 truncation error: {truncation_error_periodic_s1_m50:.2e}"
    )
    print("  Paper claims: ~10^-5 to 10^-6")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
