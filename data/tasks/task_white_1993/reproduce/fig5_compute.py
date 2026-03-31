"""
Figure 5 Data Generation: Energy Gap vs Inverse Chain Length

This module computes the energy gap Δ_L between the ground state (S_z=0) and
first excited state (S_z=1) as a function of chain length for Figure 5.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section VII, Figure 5

Figure 5 Description (from paper):
    "Gaps Δ_L between the ground and first excited states for S=1/2 and S=1
    systems with periodic boundary conditions versus the inverse length of
    the chain 1/L."

    Key physics:
    - S=1: Gap tends to finite value (~0.411) as L → ∞ (Haldane gap)
    - S=1/2: Gap tends to zero as L → ∞ (gapless, Bethe ansatz)

    DMRG parameters from paper:
    - S=1/2: m=80
    - S=1: m=200 for largest systems

Figure Specifications:
    - X-axis: 1/L (inverse chain length), linear scale, range 0.00 to 0.10
    - Y-axis: Δ_L (energy gap), linear scale, range 0.0 to 0.5
    - Boundary conditions: Periodic

Data Series (2 curves):
    1. S=1: open circles - gap extrapolates to ~0.411
    2. S=1/2: filled circles - gap extrapolates to 0

Output Format (CSV):
    L, 1/L, Gap S=1/2, Gap S=1
"""

import os

import numpy as np
import pandas as pd
from dmrg_finite import finite_dmrg


def compute_gap(spin, L, chi, num_sweeps=3, sparse=True, verbose=True):
    """
    Compute energy gap Δ_L = E(S_z=1) - E(S_z=0) for periodic boundary conditions.

    Parameters
    ----------
    spin : float
        Spin value (0.5 or 1.0).
    L : int
        Chain length.
    chi : int
        Bond dimension (number of states kept).
    num_sweeps : int
        Number of finite DMRG sweeps.
    sparse : bool
        Use sparse matrix operations.
    verbose : bool
        Print progress.

    Returns
    -------
    float
        Energy gap Δ_L.
    """
    # Ground state: S_z^total = 0
    result_gs = finite_dmrg(
        spin=spin,
        chi=chi,
        L=L,
        num_sweeps=num_sweeps,
        J=1.0,
        boundary="periodic",
        Sz_target=0,
        sparse=sparse,
        verbose=verbose,
    )
    E_gs = result_gs["energy_per_site"] * L

    # First excited state: S_z^total = 1
    result_ex = finite_dmrg(
        spin=spin,
        chi=chi,
        L=L,
        num_sweeps=num_sweeps,
        J=1.0,
        boundary="periodic",
        Sz_target=1,
        sparse=sparse,
        verbose=verbose,
    )
    E_ex = result_ex["energy_per_site"] * L

    gap = E_ex - E_gs
    return gap


def compute_fig5_data(
    L_values_half=None,
    L_values_one=None,
    chi_half=80,
    chi_one=200,
    num_sweeps=3,
    sparse=True,
    verbose=True,
):
    """
    Compute energy gap for various chain lengths for S=1/2 and S=1.

    Parameters
    ----------
    L_values_half : list[int] or None
        Chain lengths for S=1/2. Default: [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 50, 64, 80, 100]
    L_values_one : list[int] or None
        Chain lengths for S=1. Default: [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 50, 64, 80, 100]
    chi_half : int
        Bond dimension for S=1/2 (default: 80, from paper).
    chi_one : int
        Bond dimension for S=1 (default: 200, from paper).
    num_sweeps : int
        Number of finite DMRG sweeps.
    sparse : bool
        Use sparse matrices.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Dictionary with computed gaps for both spin values.
    """
    # Default L values (even only) chosen for approximately even spacing in 1/L
    # L:   4    6    8   10   12   14   16   20   24   32   40   50   64   80  100
    # 1/L: 0.25 0.17 0.12 0.10 0.08 0.07 0.06 0.05 0.04 0.03 0.025 0.02 0.016 0.012 0.01
    if L_values_half is None:
        L_values_half = [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 50, 64, 80, 100]
    if L_values_one is None:
        L_values_one = [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 50, 64, 80, 100]

    results = {
        "S=1/2": {"L": [], "inv_L": [], "gap": []},
        "S=1": {"L": [], "inv_L": [], "gap": []},
    }

    # Compute S=1/2 gaps
    print("=" * 70)
    print("Computing S=1/2 gaps (periodic BCs)")
    print("=" * 70)
    for L in L_values_half:
        print(f"\n--- L={L}, chi={chi_half} ---")
        gap = compute_gap(
            spin=0.5,
            L=L,
            chi=chi_half,
            num_sweeps=num_sweeps,
            sparse=sparse,
            verbose=verbose,
        )
        results["S=1/2"]["L"].append(L)
        results["S=1/2"]["inv_L"].append(1.0 / L)
        results["S=1/2"]["gap"].append(gap)
        print(f"L={L}, 1/L={1.0 / L:.4f}, gap={gap:.6f}")

    # Compute S=1 gaps
    print("\n" + "=" * 70)
    print("Computing S=1 gaps (periodic BCs)")
    print("=" * 70)
    for L in L_values_one:
        print(f"\n--- L={L}, chi={chi_one} ---")
        gap = compute_gap(
            spin=1.0,
            L=L,
            chi=chi_one,
            num_sweeps=num_sweeps,
            sparse=sparse,
            verbose=verbose,
        )
        results["S=1"]["L"].append(L)
        results["S=1"]["inv_L"].append(1.0 / L)
        results["S=1"]["gap"].append(gap)
        print(f"L={L}, 1/L={1.0 / L:.4f}, gap={gap:.6f}")

    return results


def save_data(results, filepath="data/fig5.csv"):
    """
    Save gap data to CSV file.

    Creates a combined DataFrame with columns for both S=1/2 and S=1.
    """
    # Get all unique L values
    all_L = sorted(set(results["S=1/2"]["L"]) | set(results["S=1"]["L"]))

    # Create DataFrame
    data = {"L": [], "1/L": [], "Gap S=1/2": [], "Gap S=1": []}

    for L in all_L:
        data["L"].append(L)
        data["1/L"].append(1.0 / L)

        # Find S=1/2 gap for this L
        if L in results["S=1/2"]["L"]:
            idx = results["S=1/2"]["L"].index(L)
            data["Gap S=1/2"].append(results["S=1/2"]["gap"][idx])
        else:
            data["Gap S=1/2"].append(np.nan)

        # Find S=1 gap for this L
        if L in results["S=1"]["L"]:
            idx = results["S=1"]["L"].index(L)
            data["Gap S=1"].append(results["S=1"]["gap"][idx])
        else:
            data["Gap S=1"].append(np.nan)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"\nData saved to {filepath}")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FIGURE 5 DATA GENERATION")
    print("Reference: White, Phys. Rev. B 48, 10345 (1993)")
    print("=" * 70)
    print("\nComputing energy gaps Δ_L for periodic boundary conditions")
    print("Gap = E(S_z=1) - E(S_z=0)")

    # Parameters from paper
    # X-axis range 0 to 0.10 means L from 10 to infinity
    # Use L values that give 1/L in this range
    L_values = [4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 40, 50, 64, 80, 100]

    # Bond dimensions from paper
    chi_half = 80  # S=1/2: m=80
    chi_one = 200  # S=1: m=200 for largest systems

    # Compute gaps
    results = compute_fig5_data(
        L_values_half=L_values,
        L_values_one=L_values,
        chi_half=chi_half,
        chi_one=chi_one,
        num_sweeps=3,
        sparse=True,
        verbose=True,
    )

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "fig5.csv"
    )
    df = save_data(results, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nResults:")
    print(df.to_string(index=False))

    # Validation
    print("\n" + "-" * 70)
    print("VALIDATION (Paper Section VII):")
    print("-" * 70)
    print("S=1 gap should extrapolate to ~0.411 as L → ∞ (Haldane gap)")
    print("S=1/2 gap should extrapolate to 0 as L → ∞ (gapless)")

    # Find largest L gap values
    idx_max = np.argmax(results["S=1"]["L"])
    L_max = results["S=1"]["L"][idx_max]
    gap_s1_max = results["S=1"]["gap"][idx_max]
    print(f"\nS=1 at L={L_max}: gap = {gap_s1_max:.4f}")

    idx_max = np.argmax(results["S=1/2"]["L"])
    L_max = results["S=1/2"]["L"][idx_max]
    gap_s12_max = results["S=1/2"]["gap"][idx_max]
    print(f"S=1/2 at L={L_max}: gap = {gap_s12_max:.4f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
