"""
Figure 3 Data Generation: Relative Error vs Number of States Kept

This module computes the relative error in ground state energy as a function
of the number of states kept (bond dimension m) for reproducing Figure 3.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section VII, Figure 3

Module Organization:
  1. compute_fig3_data  - Run DMRG for various m values
  2. save_data          - Export results to CSV format

Figure 3 Description (from paper):
  "Relative error in the ground state energy ΔE/E for a 28 site S=1/2 system
  as a function of the number of states kept m, for periodic and open boundary
  conditions."

  Reference energies:
  - Periodic: E/L = -0.4442017 (exact diagonalization, Ref. 16)
  - Open: DMRG with m=60, accurate to ~10^-10

Figure Specifications:
  - X-axis: m (number of states kept), linear scale, range 0 to 125
  - Y-axis: ΔE/|E| (relative error), log scale, 10^-1 to 10^-9
  - System: L = 28 sites, S = 1/2

Data Series (2 curves):
  1. Open BCs (filled circles) - rapid convergence, error < 10^-9 at m=40
  2. Periodic BCs (open squares) - slower convergence, error ~10^-6 at m=100

Key Results from Paper:
  - Open BCs: extremely accurate even for small m
  - Periodic BCs: difficult to achieve accuracy better than 10^-7
  - Open case error decreases more rapidly than periodic case

Output Format (CSV):
  m, Open BCs, Periodic BCs
  4,  1.23e-02, 4.56e-01
  8,  3.45e-04, 1.23e-01
  ...

  Where:
    - m: number of states kept (bond dimension)
    - Open BCs: relative error |E_dmrg - E_exact| / |E_exact| for open BC
    - Periodic BCs: relative error for periodic BC
"""

import os
from typing import Any

import numpy as np
import pandas as pd
from dmrg_finite import finite_dmrg

# =============================================================================
# REFERENCE ENERGIES
# =============================================================================

# Exact ground state energy per site for 28-site S=1/2 Heisenberg chain
# Periodic: from exact diagonalization (Ref. 16 in paper)
E_EXACT_PERIODIC_PER_SITE = -0.4442017

# Open: computed from high-accuracy DMRG (m=60), accurate to ~10^-10
# We will compute this reference value with large m
E_REFERENCE_OPEN_CHI = 60


# =============================================================================
# DATA COMPUTATION
# =============================================================================


def compute_fig3_data(
    L: int = 28,
    m_values: list[int] | None = None,
    num_sweeps: int = 3,
    sparse: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute relative error in ground state energy for various m values.

    Runs finite DMRG for each (m, boundary) combination and computes the
    relative error compared to reference energies.

    Parameters
    ----------
    L : int, optional
        Chain length (default: 28, as specified in Figure 3).
    m_values : list[int] or None, optional
        Bond dimensions to compute. Default covers range 4 to 125.
    num_sweeps : int, optional
        Number of finite DMRG sweeps (default: 3).
    sparse : bool, optional
        Use sparse matrix operations (default: True). Improves performance
        for large systems by using scipy.sparse eigensolvers.
    verbose : bool, optional
        Print progress information (default: True).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'm': ndarray of bond dimension values
        - 'Open BCs': ndarray of relative errors for open BC
        - 'Periodic BCs': ndarray of relative errors for periodic BC

    Notes
    -----
    Reference energies:
    - Periodic: E/L = -0.4442017 from exact diagonalization
    - Open: computed with m=60 (paper claims accuracy ~10^-10)

    The relative error is computed as:
        ΔE/|E| = |E_dmrg - E_exact| / |E_exact|
    """
    if m_values is None:
        # Cover range shown in figure (0 to 125)
        # Use more points at small m where error changes rapidly
        m_values = [
            4,
            6,
            8,
            10,
            12,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            44,
            48,
            52,
            56,
            60,
            64,
            72,
            80,
            90,
            100,
            110,
            120,
        ]

    # First compute reference energy for open BCs using large m
    if verbose:
        print("=" * 70)
        print(f"Computing reference energy for Open BCs (m={E_REFERENCE_OPEN_CHI})")
        print("=" * 70)

    result_ref = finite_dmrg(
        spin=0.5,
        chi=E_REFERENCE_OPEN_CHI,
        L=L,
        num_sweeps=num_sweeps + 1,  # Extra sweep for reference
        J=1.0,
        boundary="open",
        Sz_target=0,
        sparse=sparse,
        verbose=verbose,
    )
    e_reference_open = result_ref["energy_per_site"]

    if verbose:
        print(
            f"\nReference energy (Open, m={E_REFERENCE_OPEN_CHI}): E/L = {e_reference_open:.12f}"
        )

    # Reference energy for periodic BCs (from exact diagonalization)
    e_exact_periodic = E_EXACT_PERIODIC_PER_SITE

    if verbose:
        print(f"Exact energy (Periodic): E/L = {E_EXACT_PERIODIC_PER_SITE:.12f}")

    # Compute errors for each m value
    errors_open = []
    errors_periodic = []

    for m in m_values:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Computing m = {m}")
            print("=" * 60)

        # Open BCs
        if verbose:
            print(f"\n--- Open BCs, m={m} ---")
        result_open = finite_dmrg(
            spin=0.5,
            chi=m,
            L=L,
            num_sweeps=num_sweeps,
            J=1.0,
            boundary="open",
            Sz_target=0,
            sparse=sparse,
            verbose=verbose,
        )
        e_open = result_open["energy_per_site"]
        rel_error_open = abs(e_open - e_reference_open) / abs(e_reference_open)
        if m >= E_REFERENCE_OPEN_CHI:
            rel_error_open = np.nan
        errors_open.append(rel_error_open)

        if verbose:
            print(f"E/L = {e_open:.12f}, relative error = {rel_error_open:.2e}")

        # Periodic BCs
        if verbose:
            print(f"\n--- Periodic BCs, m={m} ---")
        result_periodic = finite_dmrg(
            spin=0.5,
            chi=m,
            L=L,
            num_sweeps=num_sweeps,
            J=1.0,
            boundary="periodic",
            Sz_target=0,
            sparse=sparse,
            verbose=verbose,
        )
        e_periodic = result_periodic["energy_per_site"]
        rel_error_periodic = abs(e_periodic - e_exact_periodic) / abs(e_exact_periodic)
        errors_periodic.append(rel_error_periodic)

        if verbose:
            print(f"E/L = {e_periodic:.12f}, relative error = {rel_error_periodic:.2e}")

    return {
        "m": np.array(m_values),
        "Open BCs": np.array(errors_open),
        "Periodic BCs": np.array(errors_periodic),
    }


# =============================================================================
# DATA EXPORT
# =============================================================================


def save_data(results: dict[str, Any], filepath: str = "data/fig3.csv") -> pd.DataFrame:
    """
    Save relative error data to CSV file.

    Parameters
    ----------
    results : dict
        Output from compute_fig3_data().
    filepath : str, optional
        Output CSV path (default: "data/fig3.csv").

    Returns
    -------
    pd.DataFrame
        The saved DataFrame with columns: m, Open BCs, Periodic BCs
    """
    df = pd.DataFrame(
        {
            "m": results["m"],
            "Open BCs": results["Open BCs"],
            "Periodic BCs": results["Periodic BCs"],
        }
    )

    df.to_csv(filepath, index=False)
    print(f"\nData saved to {filepath}")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("FIGURE 3 DATA GENERATION")
    print("Reference: White, Phys. Rev. B 48, 10345 (1993)")
    print("=" * 70)

    # Parameters from paper (Section VII, Figure 3)
    L = 28  # 28-site system
    num_sweeps = 3  # Converge the finite DMRG
    sparse = True  # Use sparse matrix operations

    # Compute relative errors for various m values
    results = compute_fig3_data(L=L, num_sweeps=num_sweeps, sparse=sparse, verbose=True)

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "fig3.csv"
    )
    df = save_data(results, output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nData shape: {df.shape}")
    print("\nResults:")
    print(df.to_string(index=False))

    # Validation against paper claims
    print("\n" + "-" * 70)
    print("VALIDATION (Paper Section VII):")
    print("-" * 70)

    # Find m=40 result for Open BCs
    idx_40 = np.where(results["m"] == 40)[0]
    if len(idx_40) > 0:
        error_open_m40 = results["Open BCs"][idx_40[0]]
        print(f"Open BCs at m=40: error = {error_open_m40:.2e}")
        print("  Paper claims: < 10^-9")

    # Find m=100 result for Periodic BCs
    idx_100 = np.where(results["m"] == 100)[0]
    if len(idx_100) > 0:
        error_periodic_m100 = results["Periodic BCs"][idx_100[0]]
        print(f"Periodic BCs at m=100: error = {error_periodic_m100:.2e}")
        print("  Paper claims: ~10^-6")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
