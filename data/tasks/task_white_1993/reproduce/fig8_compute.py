"""
Figure 8 Data Generation: Density-Matrix Eigenvalues vs Target States

This module computes the density-matrix eigenvalues w_alpha for various numbers
of target states for Figure 8.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section VII, Figure 8

Figure 8 Description (from paper):
    "Density-matrix eigenvalues w_alpha vs alpha for a 32 site S=1/2 system
    with open boundary conditions, with various numbers of target states."

    Key physics:
    - "The renormalization group method is most accurate when only one state
      is used as a target."
    - "When a greater number of states are targeted for a fixed value of m,
      the size of the w_alpha which are discarded is increased, and there
      is a reduction in the accuracy."
    - Eigenvalues obtained from the B_15 . . B_15^R system (symmetric config)

Figure Specifications:
    - X-axis: alpha (eigenvalue index), linear scale, range 0 to 50
    - Y-axis: w_alpha (density-matrix eigenvalues), log scale, 10^0 to 10^-8
    - System: L=32, S=1/2, Open BCs
    - Sz_target: 0 (ground state sector)

Data Series (5 curves):
    1. 1 Target State:  open squares
    2. 2 Target States: open circles
    3. 3 Target States: open triangles
    4. 4 Target States: filled squares
    5. 5 Target States: filled circles

Output Format (CSV):
    alpha, 1 Target, 2 Targets, 3 Targets, 4 Targets, 5 Targets
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from dmrg_finite import finite_dmrg_multitarget


def compute_fig8_data(
    L=32,
    chi=100,
    target_states_list=None,
    num_sweeps=3,
    Sz_target=0,
    sparse=True,
    verbose=True,
):
    """
    Compute density-matrix eigenvalues for various numbers of target states.

    Parameters
    ----------
    L : int
        Chain length (default: 32, from paper).
    chi : int
        Bond dimension (default: 100).
    target_states_list : list[int] or None
        List of target state counts to compute.
        Default: [1, 2, 3, 4, 5] (from Figure 8).
    num_sweeps : int
        Number of finite DMRG sweeps (default: 3).
    Sz_target : float
        Target Sz sector (default: 0).
    sparse : bool
        Use sparse matrices (default: True).
    verbose : bool
        Print progress (default: True).

    Returns
    -------
    dict
        Dictionary with keys "1 Target", "2 Targets", etc.,
        each containing eigenvalue array.
    """
    if target_states_list is None:
        target_states_list = [1, 2, 3, 4, 5]

    results = {}

    for num_targets in target_states_list:
        label = (
            f"{num_targets} Target" if num_targets == 1 else f"{num_targets} Targets"
        )

        print("=" * 70)
        print(f"Computing eigenvalues for {label}")
        print("=" * 70)

        result = finite_dmrg_multitarget(
            spin=0.5,
            chi=chi,
            L=L,
            num_target_states=num_targets,
            num_sweeps=num_sweeps,
            J=1.0,
            boundary="open",
            Sz_target=Sz_target,
            sparse=sparse,
            verbose=verbose,
        )
        w_alpha = result["rho_eigenvalues"]

        results[label] = w_alpha
        print(f"\n{label}: {len(w_alpha)} eigenvalues")
        print(f"  Top 5: {w_alpha[:5]}")
        print(f"  Sum: {np.sum(w_alpha):.10f}")

    return results


def save_data(results, filepath="data/fig8.csv"):
    """
    Save eigenvalue data to CSV file.

    Creates a DataFrame with columns for eigenvalue index and each target count.
    Arrays of different lengths are padded with NaN.

    Parameters
    ----------
    results : dict
        Output from compute_fig8_data(). Keys are target labels,
        values are eigenvalue arrays.
    filepath : str, optional
        Output CSV path (default: "data/fig8.csv").

    Returns
    -------
    pd.DataFrame
        The saved DataFrame with columns:
        - alpha: eigenvalue index (1, 2, 3, ...)
        - 1 Target, 2 Targets, ...: eigenvalues for each target count
    """
    # Define column order matching Figure 8 legend
    columns = ["1 Target", "2 Targets", "3 Targets", "4 Targets", "5 Targets"]

    # set max length of eigenvalue arrays
    max_len = 50

    # Build data dictionary with alpha index and padded eigenvalue arrays
    data: dict[str, Any] = {}
    data["alpha"] = np.arange(1, max_len + 1)

    for key in columns:
        if key not in results:
            continue
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
    print("FIGURE 8 DATA GENERATION")
    print("Reference: White, Phys. Rev. B 48, 10345 (1993)")
    print("=" * 70)
    print("\nComputing density-matrix eigenvalues w_alpha")
    print("System: L=32, S=1/2, Open BCs, Sz=0")
    print("Target states: 1, 2, 3, 4, 5")

    # Parameters from paper
    L = 32  # 32-site chain
    chi = 100  # Bond dimension (states kept)
    target_states_list = [1, 2, 3, 4, 5]

    # Compute eigenvalues for each target count
    results = compute_fig8_data(
        L=L,
        chi=chi,
        target_states_list=target_states_list,
        num_sweeps=3,
        Sz_target=0,
        sparse=True,
        verbose=True,
    )

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "fig8.csv"
    )
    df = save_data(results, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nData shape: {df.shape}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
