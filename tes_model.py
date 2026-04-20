# tes_model.py
# Thermal Energy Storage model — pressurised hot water tank
# YOUR file. Teammates do not edit this.
#
# Changes vs original:
#   1. HEAT QUALITY CHECK — the TES only discharges if its current
#      mean temperature T_tes ≥ P.T_RETURN.  Below that threshold
#      the stored energy is too cold for the process; it passes
#      the full deficit to the backup chain (HP / el. heater).
#   2. DISCHARGE RATE LIMITER — maximum discharge power per hour
#      is capped at  TES_MAX_DISCHARGE_FRAC × E_tes  [Wh/h = W].
#      This simulates a realistically sized heat-exchanger: a small
#      tank has a small HX and cannot dump its energy in one go.
#   3. WASTE HEAT INTEGRATION — if the tank temperature falls below
#      P.T_WASTE_HEAT (70 °C), available industrial waste heat
#      charges the tank up to T_WASTE_HEAT.  This keeps the TES
#      "warm" and reduces backup heating by teammates.

import numpy as np
from typing import Optional
import params as P


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

def capacity_joules(E_tes_Wh: float) -> float:
    """Convert TES capacity from Wh to Joules."""
    return E_tes_Wh * 3600.0


def tank_volume(E_tes_Wh: float) -> float:
    """Required water volume [m³] for a given TES capacity [Wh]."""
    E_J = capacity_joules(E_tes_Wh)
    return E_J / (P.RHO_WATER * P.CP_WATER * P.DELTA_T_TES)


def tank_temperature(E_stored_J: float, E_max_J: float) -> float:
    """
    Estimate the mean bulk temperature of the tank from stored energy.

    Assumes a perfectly mixed (single-node) tank:
        T_tes = T_TES_MIN  when E_stored = 0
        T_tes = T_TES_MAX  when E_stored = E_max

    This is the standard 1-node approximation used in EN 15316-4.
    """
    soc = np.clip(E_stored_J / E_max_J, 0.0, 1.0)
    return P.T_TES_MIN + soc * (P.T_TES_MAX - P.T_TES_MIN)


# ---------------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------------

def simulate(Q_solar_W: np.ndarray,
             Q_demand_W: np.ndarray,
             E_tes_Wh: float,
             Q_waste_W: Optional[np.ndarray] = None,
             ) -> dict:
    """
    Simulate TES charging and discharging over 8760 hours.

    The TES charges when STC produces more than the process demand
    (solar surplus), and discharges when STC is insufficient.

    Parameters
    ----------
    Q_solar_W  : [W] 8760-array — STC thermal output (from stc_model)
    Q_demand_W : [W] 8760-array — process thermal demand
    E_tes_Wh   : [Wh] TES storage capacity
    Q_waste_W  : [W] 8760-array — available waste heat power each hour.
                 Built by dispatcher.run() from P.Q_WASTE_MAX_W and
                 production shift hours.  Pass None (or zeros) to
                 disable waste heat.

    Returns
    -------
    dict with:
        Q_tes_W           [W]   TES heat delivered to process each hour
        SOC               [-]   State of charge (0 = empty, 1 = full)
        T_tes             [°C]  Mean tank temperature each hour
        Q_residual        [W]   Unmet demand after STC + TES (→ backup)
        Q_waste_charged_W [W]   Waste heat actually absorbed each hour
        E_tes_Wh          [Wh]  TES capacity (passed through)
        V_tank_m3         [m³]  Required tank volume
    """
    n     = len(Q_solar_W)
    E_max = capacity_joules(E_tes_Wh)

    if Q_waste_W is None:
        Q_waste_W = np.zeros(n)

    # Pre-allocate output arrays
    Q_tes           = np.zeros(n)
    SOC             = np.zeros(n)
    T_tes_arr       = np.zeros(n)
    Q_residual      = np.zeros(n)
    Q_waste_charged = np.zeros(n)

    E_stored = P.SOC_INITIAL * E_max

    # Max discharge energy per hour [J] — heat-exchanger size proxy
    E_discharge_max = P.TES_MAX_DISCHARGE_FRAC * E_max

    for i in range(n):

        # ---- Step A: current tank temperature -------------------------
        T_tes = tank_temperature(E_stored, E_max)

        # ---- Step B: passive thermal losses this hour [J] -------------
        E_loss = P.TES_LOSS_FRAC * E_stored * 3600.0

        # ---- Step C: waste heat charging ------------------------------
        # Industrial waste heat at T_WASTE_HEAT can raise the tank only
        # up to T_WASTE_HEAT (can't boost above its own temperature).
        # We charge only the energy gap between current state and
        # the target SOC corresponding to T_WASTE_HEAT.
        E_waste_in = 0.0
        if T_tes < P.T_WASTE_HEAT and Q_waste_W[i] > 0.0:
            soc_waste_target = (
                (P.T_WASTE_HEAT - P.T_TES_MIN)
                / (P.T_TES_MAX   - P.T_TES_MIN)
            )
            E_target   = soc_waste_target * E_max
            E_gap      = max(E_target - E_stored, 0.0)
            E_waste_in = min(Q_waste_W[i] * 3600.0, E_gap)
            Q_waste_charged[i] = E_waste_in / 3600.0   # store as [W] equivalent

        # ---- Step D: solar surplus / deficit --------------------------
        surplus_W = Q_solar_W[i] - Q_demand_W[i]

        if surplus_W >= 0.0:
            # ---- CHARGE from solar surplus ----------------------------
            # The surplus solar energy flows into the tank.
            # Note: this energy has already been excluded from
            # Q_solar_direct in the dispatcher, so there is NO
            # double-counting with Q_tes when it is later discharged.
            E_charge = surplus_W * 3600.0
            E_stored = min(
                E_stored + E_charge + E_waste_in - E_loss,
                E_max * P.SOC_MAX
            )
            E_stored      = max(E_stored, 0.0)
            Q_tes[i]      = 0.0
            Q_residual[i] = 0.0

        else:
            # ---- DISCHARGE to cover deficit ---------------------------

            # HEAT QUALITY CHECK:
            # If the bulk tank temperature is below T_RETURN, the stored
            # energy is too cold to be useful for the process (which
            # returns fluid at T_RETURN = 50 °C).  Attempting to
            # discharge would cause thermal mixing issues and deliver
            # sub-quality heat.  Pass the full deficit to backup.
            if T_tes < P.T_RETURN:
                Q_tes[i]      = 0.0
                Q_residual[i] = Q_demand_W[i] - Q_solar_W[i]   # entire deficit → backup
                E_stored      = max(E_stored + E_waste_in - E_loss, 0.0)

            else:
                E_deficit   = (-surplus_W) * 3600.0
                E_available = max(E_stored - P.SOC_MIN * E_max, 0.0)

                # DISCHARGE RATE LIMITER:
                # Physical HX cannot transfer energy faster than its
                # design throughput.  Cap to E_discharge_max [J/h].
                E_discharge = min(E_deficit, E_available, E_discharge_max)

                E_stored = max(
                    E_stored - E_discharge + E_waste_in - E_loss,
                    0.0
                )
                Q_tes[i]      = E_discharge / 3600.0        # [W]
                Q_residual[i] = max(
                    Q_demand_W[i] - Q_solar_W[i] - Q_tes[i], 0.0
                )

        # ---- Step E: record state -------------------------------------
        SOC[i]       = np.clip(E_stored / E_max, 0.0, 1.0)
        T_tes_arr[i] = tank_temperature(E_stored, E_max)

    return {
        'Q_tes_W':           Q_tes,
        'SOC':               SOC,
        'T_tes':             T_tes_arr,
        'Q_residual':        Q_residual,
        'Q_waste_charged_W': Q_waste_charged,
        'E_tes_Wh':          E_tes_Wh,
        'V_tank_m3':         tank_volume(E_tes_Wh),
    }
