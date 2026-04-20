# economics.py
# =========================================================
# Techno-economic analysis: CAPEX, OPEX, LCOH.
#
# LCOH = Levelised Cost of Heat [€/MWh_th]
#      = Total annualised cost [€/yr]
#        / Annual useful heat delivered [MWh_th/yr]
#
# This is the single number the optimiser minimises.
# =========================================================

import numpy as np
import params as P


def capital_recovery_factor(r: float = P.DISCOUNT_RATE,
                             n: int   = P.LIFETIME_YR) -> float:
    """
    CRF converts a one-time CAPEX into an equivalent annual cost.
    Formula: CRF = r(1+r)^n / ((1+r)^n - 1)
    """
    return r * (1 + r)**n / ((1 + r)**n - 1)


CRF = capital_recovery_factor()


def compute(dispatch_result: dict,
            Q_demand_W: np.ndarray,
            # Teammates: add their sizing parameters here
            # P_hp_kW: float = 0.0,
            # P_elheater_kW: float = 8000.0,
            # n_pv: int = 0,
            # E_bess_kWh: float = 0.0,
            ) -> dict:
    """
    Compute full techno-economic metrics for a dispatch result.

    Returns a dict with CAPEX, OPEX, annual costs, LCOH, and
    the annual energy balance broken down by component.
    """
    n_stc    = dispatch_result['n_stc']
    E_tes_Wh = dispatch_result['E_tes_Wh']
    A_stc    = dispatch_result['A_stc_m2']

    # --------------------------------------------------
    # CAPEX [€] — one-time investment
    # --------------------------------------------------
    capex_stc = A_stc * P.CAPEX_STC_PER_M2
    capex_tes = (E_tes_Wh / 1000) * P.CAPEX_TES_PER_KWH   # convert Wh → kWh

    # Teammates: add their CAPEX lines here, e.g.:
    # capex_hp      = P_hp_kW * P.CAPEX_HP_PER_KW
    # capex_elheater= P_elheater_kW * P.CAPEX_EL_HEATER_KW
    # capex_pv      = (n_pv * P.PV_AREA_MODULE * P.PV_ETA * 1000) * P.CAPEX_PV_PER_KWP
    # capex_bess    = E_bess_kWh * P.CAPEX_BESS_PER_KWH
    capex_hp       = 0.0   # placeholder
    capex_elheater = 0.0   # placeholder
    capex_pv       = 0.0   # placeholder
    capex_bess     = 0.0   # placeholder

    capex_total = (capex_stc + capex_tes + capex_hp +
                   capex_elheater + capex_pv + capex_bess)

    # --------------------------------------------------
    # ANNUAL O&M [€/yr]
    # --------------------------------------------------
    om_stc       = capex_stc       * P.OM_STC_FRAC
    om_tes       = capex_tes       * P.OM_TES_FRAC
    om_hp        = capex_hp        * P.OM_HP_FRAC
    om_elheater  = capex_elheater  * P.OM_EL_FRAC
    om_pv        = capex_pv        * P.OM_PV_FRAC
    om_bess      = capex_bess      * P.OM_BESS_FRAC
    om_total     = om_stc + om_tes + om_hp + om_elheater + om_pv + om_bess

    # --------------------------------------------------
    # ANNUAL ELECTRICITY COST [€/yr]
    # (electricity consumed by HP and electric heater)
    # --------------------------------------------------
    E_hp_elec_Wh      = dispatch_result['Q_hp_W'].sum() / P.HP_COP          # Wh electrical
    E_elheater_elec_Wh= dispatch_result['Q_elheater_W'].sum() / P.EL_HEATER_EFF
    elec_cost_annual  = (E_hp_elec_Wh + E_elheater_elec_Wh) / 1000 * P.ELEC_PRICE

    # --------------------------------------------------
    # TOTAL ANNUAL COST [€/yr]
    # --------------------------------------------------
    annual_capex_cost = capex_total * CRF
    annual_total_cost = annual_capex_cost + om_total + elec_cost_annual

    # --------------------------------------------------
    # ENERGY BALANCE [MWh/yr]
    # --------------------------------------------------
    E_demand  = Q_demand_W.sum()                             / 1e6
    E_solar   = np.minimum(dispatch_result['Q_solar_W'],
                           Q_demand_W).sum()                 / 1e6
    E_tes_out = dispatch_result['Q_tes_W'].sum()             / 1e6
    E_hp      = dispatch_result['Q_hp_W'].sum()              / 1e6
    E_elh     = dispatch_result['Q_elheater_W'].sum()        / 1e6
    E_covered = dispatch_result['Q_covered_W'].sum()         / 1e6
    E_unmet   = dispatch_result['Q_unmet_W'].sum()           / 1e6

    # --------------------------------------------------
    # LCOH [€/MWh_th]
    # = total annual cost / annual heat delivered
    # --------------------------------------------------
    if E_covered > 0:
        lcoh = annual_total_cost / E_covered
    else:
        lcoh = float('inf')

    return {
        # Energy [MWh/yr]
        'E_demand_MWh':   E_demand,
        'E_solar_MWh':    E_solar,
        'E_tes_MWh':      E_tes_out,
        'E_hp_MWh':       E_hp,
        'E_elheater_MWh': E_elh,
        'E_covered_MWh':  E_covered,
        'E_unmet_MWh':    E_unmet,
        'thermal_share':  E_covered / E_demand * 100,
        # Costs [€]
        'capex_total':    capex_total,
        'capex_stc':      capex_stc,
        'capex_tes':      capex_tes,
        'om_annual':      om_total,
        'elec_cost_annual': elec_cost_annual,
        'annual_total_cost': annual_total_cost,
        # Key metric
        'LCOH':           lcoh,              # [€/MWh_th] — the number to minimise
        'CRF':            CRF,
    }


def print_summary(econ: dict, dispatch: dict) -> None:
    """Print a formatted results table to the console."""
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  TECHNO-ECONOMIC RESULTS — OPTIMAL CONFIGURATION")
    print(sep)
    print(f"  SYSTEM SIZING")
    print(f"  {'STC modules':<30} {dispatch['n_stc']:>8} modules")
    print(f"  {'STC field area':<30} {dispatch['A_stc_m2']:>8.0f} m²")
    print(f"  {'TES capacity':<30} {dispatch['E_tes_Wh']/1e6:>8.1f} MWh")
    print(f"  {'TES tank volume':<30} {dispatch['V_tes_m3']:>8.0f} m³")
    print(f"{'-'*54}")
    print(f"  ANNUAL ENERGY BALANCE")
    print(f"  {'Total demand':<30} {econ['E_demand_MWh']:>8.1f} MWh")
    print(f"  {'STC direct':<30} {econ['E_solar_MWh']:>8.1f} MWh  "
          f"({econ['E_solar_MWh']/econ['E_demand_MWh']*100:.1f}%)")
    print(f"  {'TES contribution':<30} {econ['E_tes_MWh']:>8.1f} MWh  "
          f"({econ['E_tes_MWh']/econ['E_demand_MWh']*100:.1f}%)")
    print(f"  {'HP contribution':<30} {econ['E_hp_MWh']:>8.1f} MWh  "
          f"({econ['E_hp_MWh']/econ['E_demand_MWh']*100:.1f}%)")
    print(f"  {'El. heater':<30} {econ['E_elheater_MWh']:>8.1f} MWh  "
          f"({econ['E_elheater_MWh']/econ['E_demand_MWh']*100:.1f}%)")
    print(f"  {'Thermal share (STC+TES)':<30} {econ['thermal_share']:>8.1f}%")
    print(f"{'-'*54}")
    print(f"  ECONOMICS")
    print(f"  {'Total CAPEX':<30} {econ['capex_total']/1e6:>8.2f} M€")
    print(f"    {'of which STC':<28} {econ['capex_stc']/1e6:>8.2f} M€")
    print(f"    {'of which TES':<28} {econ['capex_tes']/1e6:>8.2f} M€")
    print(f"  {'Annual O&M':<30} {econ['om_annual']/1e3:>8.1f} k€/yr")
    print(f"  {'Annual electricity cost':<30} {econ['elec_cost_annual']/1e3:>8.1f} k€/yr")
    print(f"  {'Total annual cost':<30} {econ['annual_total_cost']/1e3:>8.1f} k€/yr")
    print(f"{'-'*54}")
    print(f"  {'LCOH (Levelised Cost of Heat)':<30} {econ['LCOH']:>8.2f} €/MWh_th")
    print(f"  {'Capital Recovery Factor':<30} {econ['CRF']:>8.4f}")
    print(f"{sep}\n")