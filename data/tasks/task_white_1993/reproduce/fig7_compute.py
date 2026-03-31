"""
Figure 7 Data Generation: Local Observables for S=1 Chain

This module computes local bond strength and magnetization for S=1 Heisenberg chain
with open boundary conditions for Figure 7 reproduction.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section VII, Figure 7

Figure 7 Description (from paper):
    "Local bond strength (a) and local value of S_z (b) for a S=1 system
    with 60 sites. The S=1/2 effective spins on the ends are clearly
    visible in (b), which shows one of the low-lying triplet states
    just above the ground state."

    Key physics (Haldane phase):
    - Nearly uniform bond strength in the bulk
    - Effective S=1/2 edge states at chain ends
    - Edge magnetization ~0.532 (slightly larger than 0.5)

Figure Specifications:
    Panel (a):
        - X-axis: i (site index), linear scale, range 0 to 60
        - Y-axis: <S_i · S_{i+1}>, linear scale, range -1.7 to -1.2

    Panel (b):
        - X-axis: i (site index), linear scale, range 0 to 60
        - Y-axis: <S_i^z>, linear scale, range -0.4 to 0.6

Panels:
    (a) L=60, S=1, Open BCs, Sz_target=0 - local bond strength (ground state)
    (b) L=60, S=1, Open BCs, Sz_target=1 - local magnetization (triplet state)

Output Format (CSV):
    i, Bond Strength, Local Sz
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from dmrg_finite import finite_dmrg_measure


def compute_fig7_data(
    L=60,
    chi=100,
    num_sweeps=3,
    sparse=True,
    verbose=True,
):
    """
    Compute local observables for Figure 7 panels.

    Parameters
    ----------
    L : int
        Chain length (default: 60).
    chi : int
        Bond dimension (default: 200, larger for S=1).
    num_sweeps : int
        Number of finite DMRG sweeps (default: 4).
    sparse : bool
        Use sparse matrices (default: True).
    verbose : bool
        Print progress (default: True).

    Returns
    -------
    dict
        Dictionary with keys:
        - "Bond Strength": bond strength array from ground state
        - "Local Sz": local magnetization from triplet state
    """
    results = {}

    # Panel (a): Bond strength from ground state (Sz_target=0)
    if verbose:
        print("=" * 70)
        print(f"Panel (a): L={L}, S=1, Open BCs, Sz=0 (ground state)")
        print("=" * 70)

    result_a = finite_dmrg_measure(
        spin=1.0,
        chi=chi,
        L=L,
        num_sweeps=num_sweeps,
        J=1.0,
        J_edge=None,
        boundary="open",
        Sz_target=0,  # Ground state (singlet for even L)
        sparse=sparse,
        verbose=verbose,
    )
    results["Bond Strength"] = result_a["bond_strength"]

    if verbose:
        print(f"\nPanel (a): E/L = {result_a['energy_per_site']:.10f}")
        valid_bonds = result_a["bond_strength"][~np.isnan(result_a["bond_strength"])]
        if len(valid_bonds) > 0:
            print(
                f"Bond strength range: [{np.min(valid_bonds):.4f}, "
                f"{np.max(valid_bonds):.4f}]"
            )
            print(f"Center bond strength: {valid_bonds[len(valid_bonds) // 2]:.6f}")

    # Panel (b): Local Sz from triplet state (Sz_target=1)
    if verbose:
        print("\n" + "=" * 70)
        print(f"Panel (b): L={L}, S=1, Open BCs, Sz=1 (triplet state)")
        print("=" * 70)

    result_b = finite_dmrg_measure(
        spin=1.0,
        chi=chi,
        L=L,
        num_sweeps=num_sweeps,
        J=1.0,
        J_edge=None,
        boundary="open",
        Sz_target=1,  # Triplet state (one of the low-lying excitations)
        sparse=sparse,
        verbose=verbose,
    )
    results["Local Sz"] = result_b["local_Sz"]

    if verbose:
        print(f"\nPanel (b): E/L = {result_b['energy_per_site']:.10f}")
        valid_Sz = result_b["local_Sz"][~np.isnan(result_b["local_Sz"])]
        if len(valid_Sz) > 0:
            print(f"Local Sz range: [{np.min(valid_Sz):.4f}, {np.max(valid_Sz):.4f}]")
            # Check edge magnetization (paper reports ~0.532)
            print(f"Edge magnetization (site 0): {valid_Sz[0]:.6f}")
            print(f"Edge magnetization (site {len(valid_Sz) - 1}): {valid_Sz[-1]:.6f}")
            print("(Paper reports edge magnetization ~0.532)")

    return results


def save_data(results, filepath="data/fig7.csv"):
    """
    Save local observable data to CSV file.

    Parameters
    ----------
    results : dict
        Output from compute_fig7_data().
    filepath : str
        Output CSV path.

    Returns
    -------
    pd.DataFrame
        The saved DataFrame.
    """
    # Define column order
    columns = ["Bond Strength", "Local Sz"]

    # Find maximum array length
    max_len = max(len(results[k]) for k in results)

    # Build data dictionary
    data: dict[str, Any] = {}
    data["i"] = np.arange(1, max_len + 1)

    for key in columns:
        arr = results[key]
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), np.nan)])
        data[key] = arr

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"\nData saved to {filepath}")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FIGURE 7 DATA GENERATION")
    print("Reference: White, Phys. Rev. B 48, 10345 (1993)")
    print("=" * 70)
    print("\nComputing local observables for S=1 Heisenberg chain")
    print("System: L=60, S=1, Open BCs")
    print("  Panel (a): Bond strength <S_i · S_{i+1}> (ground state, Sz=0)")
    print("  Panel (b): Local magnetization <S_i^z> (triplet state, Sz=1)")

    # Parameters
    L = 60
    chi = 100
    num_sweeps = 3

    # Compute local observables
    results = compute_fig7_data(
        L=L,
        chi=chi,
        num_sweeps=num_sweeps,
        sparse=True,
        verbose=True,
    )

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "fig7.csv"
    )
    df = save_data(results, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nData shape: {df.shape}")
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    # Validation
    print("\n" + "-" * 70)
    print("VALIDATION (Paper Section VII):")
    print("-" * 70)
    print("Panel (a): Bond strength should be nearly constant (~-1.45) in bulk")
    print("Panel (b): Edge magnetization should be ~0.532 (effective S=1/2)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
