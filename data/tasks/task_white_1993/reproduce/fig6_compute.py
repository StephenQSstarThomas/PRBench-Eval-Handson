"""
Figure 6 Data Generation: Local Bond Strength for S=1/2 Chains

This module computes local bond strength <S_i · S_{i+1}> for S=1/2 Heisenberg chains
with open boundary conditions for Figure 6 reproduction.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section VII, Figure 6

Figure 6 Description (from paper):
    "Local bond strength for (a) 60 site and (b) 61 site S=1/2 chains with
    open boundary conditions. The systems try to have strong single bonds
    at the outermost links of the chain; this induces a domain wall in the
    odd-length chain. In (c), the bond-strength alternation is diminished
    by weakening the local value of the exchange constant J on the outermost links."

Figure Specifications:
    - X-axis: i (site index), linear scale, range 0 to L
    - Y-axis: <S_i · S_{i+1}> (bond strength), linear scale, range -0.7 to -0.2

Panels:
    (a) L=60, S=1/2, Open BCs - shows bond-strength alternation
    (b) L=61, S=1/2, Open BCs - shows domain wall due to frustration
    (c) L=60, S=1/2, Open BCs, J_edge=0.236 - soft BCs reduce alternation

Output Format (CSV):
    i, Panel (a), Panel (b), Panel (c)
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from dmrg_finite import finite_dmrg_measure


def compute_fig6_data(
    chi=100,
    num_sweeps=3,
    sparse=True,
    verbose=True,
):
    """
    Compute local bond strength for all three panels of Figure 6.

    Parameters
    ----------
    chi : int
        Bond dimension (default: 100).
    num_sweeps : int
        Number of finite DMRG sweeps (default: 3).
    sparse : bool
        Use sparse matrices (default: True).
    verbose : bool
        Print progress (default: True).

    Returns
    -------
    dict
        Dictionary with keys "Panel (a)", "Panel (b)", "Panel (c)",
        each containing bond strength array.
    """
    results = {}

    # Panel (a): L=60, S=1/2, Open BCs
    if verbose:
        print("=" * 70)
        print("Panel (a): L=60, S=1/2, Open BCs")
        print("=" * 70)

    result_a = finite_dmrg_measure(
        spin=0.5,
        chi=chi,
        L=60,
        num_sweeps=num_sweeps,
        J=1.0,
        J_edge=None,  # Standard J at edges
        boundary="open",
        Sz_target=0,
        sparse=sparse,
        verbose=verbose,
    )
    results["Panel (a)"] = result_a["bond_strength"]

    if verbose:
        print(f"\nPanel (a): E/L = {result_a['energy_per_site']:.10f}")
        print(
            f"Bond strength range: [{np.nanmin(result_a['bond_strength']):.4f}, "
            f"{np.nanmax(result_a['bond_strength']):.4f}]"
        )

    # Panel (b): L=61, S=1/2, Open BCs (odd length - frustrated)
    if verbose:
        print("\n" + "=" * 70)
        print("Panel (b): L=61, S=1/2, Open BCs (odd length)")
        print("=" * 70)

    result_b = finite_dmrg_measure(
        spin=0.5,
        chi=chi,
        L=61,
        num_sweeps=num_sweeps,
        J=1.0,
        J_edge=None,
        boundary="open",
        Sz_target=None,  # Let algorithm find ground state sector
        sparse=sparse,
        verbose=verbose,
    )
    results["Panel (b)"] = result_b["bond_strength"]

    if verbose:
        print(f"\nPanel (b): E/L = {result_b['energy_per_site']:.10f}")
        print(
            f"Bond strength range: [{np.nanmin(result_b['bond_strength']):.4f}, "
            f"{np.nanmax(result_b['bond_strength']):.4f}]"
        )

    # Panel (c): L=60, S=1/2, Open BCs with soft boundary (J_edge=0.236)
    if verbose:
        print("\n" + "=" * 70)
        print("Panel (c): L=60, S=1/2, Open BCs, J_edge=0.236")
        print("=" * 70)

    result_c = finite_dmrg_measure(
        spin=0.5,
        chi=chi,
        L=60,
        num_sweeps=num_sweeps,
        J=1.0,
        J_edge=0.236,  # Soft boundary conditions
        boundary="open",
        Sz_target=0,
        sparse=sparse,
        verbose=verbose,
    )
    results["Panel (c)"] = result_c["bond_strength"]

    if verbose:
        print(f"\nPanel (c): E/L = {result_c['energy_per_site']:.10f}")
        print(
            f"Bond strength range: [{np.nanmin(result_c['bond_strength']):.4f}, "
            f"{np.nanmax(result_c['bond_strength']):.4f}]"
        )

    return results


def save_data(results, filepath="data/fig6.csv"):
    """
    Save bond strength data to CSV file.

    Parameters
    ----------
    results : dict
        Output from compute_fig6_data().
    filepath : str
        Output CSV path.

    Returns
    -------
    pd.DataFrame
        The saved DataFrame.
    """
    # Define column order
    columns = ["Panel (a)", "Panel (b)", "Panel (c)"]

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
    print("FIGURE 6 DATA GENERATION")
    print("Reference: White, Phys. Rev. B 48, 10345 (1993)")
    print("=" * 70)
    print("\nComputing local bond strength <S_i · S_{i+1}>")
    print("System: S=1/2, Open BCs")

    # Parameters
    chi = 100
    num_sweeps = 3

    # Compute bond strength for all panels
    results = compute_fig6_data(
        chi=chi,
        num_sweeps=num_sweeps,
        sparse=True,
        verbose=True,
    )

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "fig6.csv"
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
