# optimiser.py
# =========================================================
# Grid-search optimiser: sweeps STC and TES sizes and finds
# the combination that minimises LCOH.
#
# Why grid search instead of scipy.minimize?
# For a student project, grid search is transparent — you can
# see the full LCOH surface and understand trade-offs.
# It also avoids getting stuck in local minima.
# Runtime: ~200 combinations × 8760 hours ≈ 30-60 seconds.
# =========================================================

import numpy as np
import params as P
import dispatcher
import economics


def run(climate: dict, Q_demand_W: np.ndarray) -> dict:
    """
    Sweep over (n_stc, E_tes_Wh) and compute LCOH for each combination.
    Returns the optimal configuration and the full results grid.
    """
    n_stc_values  = list(P.OPT_N_STC_RANGE)
    e_tes_values  = [e * 1e6 for e in P.OPT_E_TES_RANGE]  # MWh → Wh

    n_rows = len(n_stc_values)
    n_cols = len(e_tes_values)

    # Store LCOH and thermal share for every combination
    lcoh_grid    = np.full((n_rows, n_cols), np.nan)
    ts_grid      = np.full((n_rows, n_cols), np.nan)

    print(f"[Optimiser] Searching {n_rows} × {n_cols} = {n_rows*n_cols} combinations...")
    print(f"[Optimiser] STC: {n_stc_values[0]}–{n_stc_values[-1]} modules | "
          f"TES: {P.OPT_E_TES_RANGE.start}–{P.OPT_E_TES_RANGE.stop} MWh")

    best_lcoh   = float('inf')
    best_config = None

    for i, n_stc in enumerate(n_stc_values):
        for j, E_tes_Wh in enumerate(e_tes_values):

            # Run dispatch for this configuration
            disp = dispatcher.run(
                climate    = climate,
                Q_demand_W = Q_demand_W,
                n_stc      = n_stc,
                E_tes_Wh   = E_tes_Wh,
            )

            # Compute LCOH
            econ = economics.compute(disp, Q_demand_W)

            lcoh_grid[i, j] = econ['LCOH']
            ts_grid[i, j]   = econ['thermal_share']

            # Track best
            if econ['LCOH'] < best_lcoh:
                best_lcoh   = econ['LCOH']
                best_config = {
                    'n_stc':    n_stc,
                    'E_tes_Wh': E_tes_Wh,
                    'dispatch': disp,
                    'econ':     econ,
                }

        # Progress update every 5 STC values
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n_rows}] Best LCOH so far: {best_lcoh:.2f} €/MWh_th "
                  f"(N_stc={best_config['n_stc']}, "
                  f"E_tes={best_config['E_tes_Wh']/1e6:.0f} MWh)")

    print(f"\n[Optimiser] Done. Optimal LCOH = {best_lcoh:.2f} €/MWh_th")

    return {
        'best':           best_config,
        'lcoh_grid':      lcoh_grid,
        'ts_grid':        ts_grid,
        'n_stc_values':   n_stc_values,
        'e_tes_values_Wh': e_tes_values,
    }