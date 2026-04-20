# dispatcher.py
# =========================================================
# Rule-based energy dispatcher — merit order logic.
#
# Every hour the dispatcher asks: "who covers the demand,
# and in what order?" It always picks the cheapest source
# first (zero-cost solar/storage, then HP, then el. heater).
#
# MERIT ORDER (cheapest marginal cost first):
#   1. STC       — marginal cost = 0  (sun is free)
#   2. TES       — marginal cost = 0  (already stored)
#   3. Heat pump — marginal cost = electricity_price / COP
#   4. El. heater— marginal cost = electricity_price
#
# DOUBLE-COUNTING NOTE:
#   STC produces Q_solar each hour.  Two distinct paths exist:
#     a) Q_solar ≤ Q_demand  → STC covers demand directly.
#        The surplus is zero; TES does NOT charge.
#        Q_solar_direct = Q_solar (all of it goes to process).
#     b) Q_solar > Q_demand  → STC covers demand AND the surplus
#        charges the TES.  The fraction beyond demand never reaches
#        the process directly — it is stored and later counted as
#        Q_tes when discharged.
#        Q_solar_direct = Q_demand (only what the process absorbs).
#   The identity  Q_solar_direct + Q_tes ≤ Q_demand  holds every
#   hour by construction, so there is no double-counting.
#
# HOW TEAMMATES ADD THEIR COMPONENT:
#   1. Create your model file (e.g. hp_model.py) with a
#      simulate(Q_residual, climate, **kwargs) function.
#   2. Import it here and add it to the DISPATCH_ORDER list.
#   3. Your function receives the remaining unmet demand and
#      returns how much it covers + the new residual.
# =========================================================

import numpy as np
import params as P

import stc_model
import tes_model

# Teammates: uncomment their import when the file is ready
# import hp_model
# import el_heater_model
# import pv_model
# import bess_model


def run(climate: dict,
        Q_demand_W: np.ndarray,
        n_stc: int,
        E_tes_Wh: float,
        # Teammates: add their sizing parameters here, e.g.:
        # P_hp_kW: float = 0.0,
        # P_elheater_kW: float = 8000.0,
        # n_pv: int = 0,
        # E_bess_kWh: float = 0.0,
        ) -> dict:
    """
    Run the full annual dispatch for a given system configuration.

    Parameters
    ----------
    climate    : dict from climate_data.load_climate_data()
    Q_demand_W : [W] 8760-array — hourly process thermal demand
    n_stc      : number of STC modules
    E_tes_Wh   : TES capacity [Wh]

    Returns
    -------
    dict with hourly arrays for every component and key KPIs.
    """

    # --------------------------------------------------
    # STEP 1 — STC (priority 1: zero marginal cost)
    # --------------------------------------------------
    stc_res = stc_model.simulate(climate, n_stc)
    Q_solar = stc_res['Q_solar_W']   # total STC output, including surplus → TES

    # ANTI-DOUBLE-COUNT: separate what actually reaches the process
    # from what is surplus going into the TES.
    # Q_solar_direct:  the portion delivered straight to the process this hour.
    # Q_solar_surplus: the portion routed into TES (never touches the process directly).
    Q_solar_direct  = np.minimum(Q_solar, Q_demand_W)
    Q_solar_surplus = np.maximum(Q_solar - Q_demand_W, 0.0)
    # Downstream energy accounting uses Q_solar_direct to avoid double-counting
    # with Q_tes (which represents the same solar energy, stored then released).

    # --------------------------------------------------
    # STEP 2 — Waste heat array (available during production shift)
    # Built here so the dispatcher controls timing; passed to TES.
    # --------------------------------------------------
    in_shift  = ((climate['hour_of_day'] >= P.HOUR_START) &
                 (climate['hour_of_day'] <  P.HOUR_END))
    Q_waste_W = np.where(in_shift, P.Q_WASTE_MAX_W, 0.0)

    # --------------------------------------------------
    # STEP 3 — TES (priority 2: zero marginal cost)
    # Receives: full STC output (so it can charge from surplus),
    # full demand, and the waste heat array.
    # --------------------------------------------------
    tes_res    = tes_model.simulate(Q_solar, Q_demand_W, E_tes_Wh,
                                    Q_waste_W=Q_waste_W)
    Q_tes      = tes_res['Q_tes_W']
    Q_residual = tes_res['Q_residual']    # Still unmet after STC + TES
    SOC_TES    = tes_res['SOC']
    T_TES      = tes_res['T_tes']
    Q_waste_charged = tes_res['Q_waste_charged_W']

    # --------------------------------------------------
    # STEP 4 — Heat pump (priority 3: elec_price / COP)
    # Teammate: replace this block with hp_model.simulate()
    # --------------------------------------------------
    Q_hp = np.zeros_like(Q_residual)
    # When teammate is ready:
    # hp_res   = hp_model.simulate(Q_residual, climate,
    #                              P_hp_kW=P_hp_kW,
    #                              elec_available=Q_pv_elec + Q_bess_elec)
    # Q_hp       = hp_res['Q_hp_W']
    # Q_residual = hp_res['Q_residual']

    # --------------------------------------------------
    # STEP 5 — Electric heater (priority 4: most expensive)
    # Teammate: replace this block with el_heater_model.simulate()
    # --------------------------------------------------
    Q_elheater       = Q_residual.copy()
    Q_residual_final = np.zeros_like(Q_residual)
    # When teammate is ready:
    # el_res           = el_heater_model.simulate(Q_residual, P_elheater_kW)
    # Q_elheater       = el_res['Q_elheater_W']
    # Q_residual_final = el_res['Q_residual']

    # --------------------------------------------------
    # STEP 6 — Aggregate results
    # --------------------------------------------------
    # Q_solar_direct is used here (NOT the full Q_solar) to prevent
    # double-counting: the stored-then-discharged solar energy is
    # already captured in Q_tes.
    Q_covered = Q_solar_direct + Q_tes + Q_hp + Q_elheater
    # Final clamp ensures numerical precision never pushes above demand.
    Q_covered = np.minimum(Q_covered, Q_demand_W)

    return {
        # Hourly profiles
        'Q_solar_W':         Q_solar,           # total STC output (incl. surplus → TES)
        'Q_solar_direct_W':  Q_solar_direct,    # STC fraction delivered directly to process
        'Q_solar_surplus_W': Q_solar_surplus,   # STC fraction charged into TES
        'Q_tes_W':           Q_tes,
        'Q_hp_W':            Q_hp,
        'Q_elheater_W':      Q_elheater,
        'Q_covered_W':       Q_covered,
        'Q_unmet_W':         Q_residual_final,
        'Q_waste_charged_W': Q_waste_charged,
        'SOC_TES':           SOC_TES,
        'T_TES':             T_TES,
        # Metadata
        'n_stc':             n_stc,
        'E_tes_Wh':          E_tes_Wh,
        'A_stc_m2':          stc_res['A_total'],
        'V_tes_m3':          tes_res['V_tank_m3'],
        'eta_solar':         stc_res['eta'],
    }
