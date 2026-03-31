"""
Figure 4 Data Generation: Relative Error vs Number of States Kept (S=1)

This module computes the relative error in ground state energy as a function
of the number of states kept (bond dimension m) for reproducing Figure 4.
Reference: White, Phys. Rev. B 48, 10345 (1993), Section VII, Figure 4

Module Organization:
  1. compute_fig4_data  - Run DMRG for various m values
  2. save_data          - Export results to CSV format

Figure 4 Description (from paper):
  "Relative error in the ground state energy E for a 16 site S=1 system,
  as a function of the number of states kept m, for periodic and open
  boundary conditions."

  Reference energies:
  - Periodic: E/L = -2.80585 (exact diagonalization, Ref. 17)
  - Open: DMRG with m=140 (paper claims high accuracy)

Figure Specifications:
  - X-axis: m (number of states kept), linear scale, range 0 to 125
  - Y-axis: ΔE/|E| (relative error), log scale, 10^-1 to 10^-9
  - System: L = 16 sites, S = 1

Data Series (2 curves):
  1. Open BCs (filled circles) - rapid convergence, error < 10^-9 at m=80
  2. Periodic BCs (open squares) - slower convergence, error ~10^-6 at m=125

Key Results from Paper:
  - S=1 shows same overall behavior as S=1/2 (Figure 3)
  - S=1 is slightly less accurate than S=1/2 for same m
  - Open BCs still much more accurate than periodic

Output Format (CSV):
  m, Open BCs, Periodic BCs
  4,  1.23e-01, 4.56e-01
  8,  3.45e-02, 1.23e-01
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

# Exact ground state energy per site for 16-site S=1 Heisenberg chain
# Periodic: from exact diagonalization (Ref. 17 in paper)
# Paper states E/L = -2.80585, but uses H = 2J*S·S convention
# Our code uses H = J*S·S, so we divide by 2
E_EXACT_PERIODIC_PER_SITE = -2.80585 / 2.0  # = -1.402925

# Open: computed from high-accuracy DMRG (m=140)
E_REFERENCE_OPEN_CHI = 140


# =============================================================================
# DATA COMPUTATION
# =============================================================================


def compute_fig4_data(
    L: int = 16,
    m_values: list[int] | None = None,
    num_sweeps: int = 3,
    sparse: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Compute relative error in ground state energy for various m values (S=1).

    Runs finite DMRG for each (m, boundary) combination and computes the
    relative error compared to reference energies.

    Parameters
    ----------
    L : int, optional
        Chain length (default: 16, as specified in Figure 4).
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
    - Periodic: E/L = -2.80585 from exact diagonalization (Ref. 17)
    - Open: computed with m=140 (paper claims high accuracy)

    The relative error is computed as:
        ΔE/|E| = |E_dmrg - E_exact| / |E_exact|

    S=1 requires larger m than S=1/2 for same accuracy because the local
    Hilbert space dimension is n=3 instead of n=2.
    """
    if m_values is None:
        # Cover range shown in figure (0 to 125)
        # S=1 needs larger m, so use denser sampling
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
        spin=1.0,
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
            spin=1.0,
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
            spin=1.0,
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


def save_data(results: dict[str, Any], filepath: str = "data/fig4.csv") -> pd.DataFrame:
    """
    Save relative error data to CSV file.

    Parameters
    ----------
    results : dict
        Output from compute_fig4_data().
    filepath : str, optional
        Output CSV path (default: "data/fig4.csv").

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
    print("FIGURE 4 DATA GENERATION")
    print("Reference: White, Phys. Rev. B 48, 10345 (1993)")
    print("=" * 70)

    # Parameters from paper (Section VII, Figure 4)
    L = 16  # 16-site system
    num_sweeps = 3  # Converge the finite DMRG
    sparse = True  # Use sparse matrix operations

    # Compute relative errors for various m values
    results = compute_fig4_data(L=L, num_sweeps=num_sweeps, sparse=sparse, verbose=True)

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "fig4.csv"
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

    # Find m=80 result for Open BCs
    idx_80 = np.where(results["m"] == 80)[0]
    if len(idx_80) > 0:
        error_open_m80 = results["Open BCs"][idx_80[0]]
        print(f"Open BCs at m=80: error = {error_open_m80:.2e}")
        print("  Paper claims: < 10^-9")

    # Find m=120 result for Periodic BCs (closest to 125)
    idx_120 = np.where(results["m"] == 120)[0]
    if len(idx_120) > 0:
        error_periodic_m120 = results["Periodic BCs"][idx_120[0]]
        print(f"Periodic BCs at m=120: error = {error_periodic_m120:.2e}")
        print("  Paper claims: ~10^-6 at m=125")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
