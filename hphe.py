import numpy as np

def simulate_heat_pump_engine(
    Q_ST_hourly: np.ndarray,
    Q_demand_hourly: np.ndarray,
    T_cold_HP: float = 70.0,
    T_hot_HP: float = 157.0,
    T_cold_HE: float = 40.0,
    T_hot_HE: float = 157.0,
    COP_real: float = 2.32,
    epsilon_HE: float = 0.20,
    Q_waste_max: float = 5e6
) -> dict:
    """
    Simulate hourly operation of High-Temperature Heat Pump (HTHP)
    and Stirling Heat Engine in a hybrid solar industrial heat system.

    Heat Pump parameters based on:
    Enerin HoegTemp, IEA HPT Annex 58, Table 1
    (Tsource=80→50°C, Tsink=154→160°C, COP=2.2–2.45)

    Parameters
    ----------
    Q_ST_hourly : np.ndarray
        Hourly ST heat output array, shape (8760,), unit: Wh
    Q_demand_hourly : np.ndarray
        Hourly industrial heat demand array, shape (8760,), unit: Wh
    T_cold_HP : float
        Heat pump cold side (waste heat) inlet temperature, °C
        Default: 70°C (from project specification)
    T_hot_HP : float
        Heat pump hot side (process heat) mean temperature, °C
        Default: 157°C (mean of Tsink 154–160°C, Annex 58 Table 1)
    T_cold_HE : float
        Heat engine cold side (ambient) temperature, °C
        Default: 40°C (Seville ambient)
    T_hot_HE : float
        Heat engine hot side temperature, °C
        Default: 157°C (same as HP hot side)
    COP_real : float
        Actual heat pump COP [-]
        Default: 2.32 (mean of 2.2–2.45, Annex 58 Table 1)
    epsilon_HE : float
        Heat engine Carnot correction factor [-]
        Default: 0.20 (conservative estimate for Stirling engine)
    Q_waste_max : float
        Maximum available waste heat power, W
        Default: 5e6 W = 5 MW

    Returns
    -------
    dict with keys:
        'Q_HP_hourly'      : np.ndarray  Heat pump heat output [Wh]
        'W_elec_hourly'    : np.ndarray  Heat pump electricity input [Wh]
        'Q_waste_hourly'   : np.ndarray  Waste heat consumed [Wh]
        'W_engine_hourly'  : np.ndarray  Engine electricity output [Wh]
        'Q_excess_hourly'  : np.ndarray  Excess ST heat to engine [Wh]
        'Q_reject_hourly'  : np.ndarray  Engine heat rejection [Wh]
        'annual'           : dict        Annual summary statistics
        'monthly'          : dict        Monthly breakdown
        'performance'      : dict        COP and efficiency values
    """

    # ── Temperature conversion to Kelvin ─────────────────────────
    T_cold_HP_K = T_cold_HP + 273.15
    T_hot_HP_K  = T_hot_HP  + 273.15
    T_cold_HE_K = T_cold_HE + 273.15
    T_hot_HE_K  = T_hot_HE  + 273.15

    # ── Derived performance parameters ───────────────────────────
    COP_carnot  = T_hot_HP_K / (T_hot_HP_K - T_cold_HP_K)
    epsilon_HP  = COP_real / COP_carnot

    eta_carnot  = 1.0 - T_cold_HE_K / T_hot_HE_K
    eta_real    = epsilon_HE * eta_carnot

    # ── Hourly simulation arrays ──────────────────────────────────
    Q_HP_hourly     = np.zeros(8760)
    W_elec_hourly   = np.zeros(8760)
    Q_waste_hourly  = np.zeros(8760)
    W_engine_hourly = np.zeros(8760)
    Q_excess_hourly = np.zeros(8760)
    Q_reject_hourly = np.zeros(8760)

    # ── Main hourly loop ──────────────────────────────────────────
    for t in range(8760):
        Q_ST  = Q_ST_hourly[t]
        Q_dem = Q_demand_hourly[t]

        if Q_dem == 0:
            continue

        if Q_ST < Q_dem:
            # ── Heat Pump mode ────────────────────────────────────
            # ST output insufficient → HP covers the gap
            Q_HP              = Q_dem - Q_ST
            W_elec            = Q_HP / COP_real
            Q_waste_used      = min(Q_HP - W_elec, Q_waste_max)

            Q_HP_hourly[t]    = Q_HP
            W_elec_hourly[t]  = W_elec
            Q_waste_hourly[t] = Q_waste_used

        else:
            # ── Heat Engine mode ──────────────────────────────────
            # ST excess → engine generates electricity
            Q_excess              = Q_ST - Q_dem
            W_out                 = eta_real * Q_excess
            Q_reject              = Q_excess - W_out

            Q_excess_hourly[t]    = Q_excess
            W_engine_hourly[t]    = W_out
            Q_reject_hourly[t]    = Q_reject

    # ── Annual summary ────────────────────────────────────────────
    annual = {
        'HP_heat_MWh'    : float(np.sum(Q_HP_hourly)     / 1e6),
        'HP_elec_MWh'    : float(np.sum(W_elec_hourly)   / 1e6),
        'waste_heat_MWh' : float(np.sum(Q_waste_hourly)  / 1e6),
        'engine_elec_MWh': float(np.sum(W_engine_hourly) / 1e6),
        'excess_heat_MWh': float(np.sum(Q_excess_hourly) / 1e6),
        'heat_reject_MWh': float(np.sum(Q_reject_hourly) / 1e6),
        'HP_hours'       : int(np.sum(Q_HP_hourly > 0)),
        'engine_hours'   : int(np.sum(W_engine_hourly > 0)),
    }

    # Effective COP check (avoid divide by zero)
    if annual['HP_elec_MWh'] > 0:
        annual['COP_effective'] = annual['HP_heat_MWh'] / annual['HP_elec_MWh']
    else:
        annual['COP_effective'] = 0.0

    # ── Monthly breakdown ─────────────────────────────────────────
    days_per_month  = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_per_month = [d * 24 for d in days_per_month]

    HP_monthly     = np.zeros(12)
    engine_monthly = np.zeros(12)
    h = 0
    for m in range(12):
        idx = slice(h, h + hours_per_month[m])
        HP_monthly[m]     = np.sum(Q_HP_hourly[idx])     / 1e6
        engine_monthly[m] = np.sum(W_engine_hourly[idx]) / 1e6
        h += hours_per_month[m]

    monthly = {
        'HP_heat_MWh'    : HP_monthly,
        'engine_elec_MWh': engine_monthly,
    }

    # ── Performance parameters (for reporting) ───────────────────
    performance = {
        'COP_carnot'  : round(COP_carnot, 3),
        'COP_real'    : round(COP_real, 3),
        'epsilon_HP'  : round(epsilon_HP, 3),
        'eta_carnot'  : round(eta_carnot, 3),
        'eta_real'    : round(eta_real, 3),
        'epsilon_HE'  : round(epsilon_HE, 3),
    }

    return {
        'Q_HP_hourly'    : Q_HP_hourly,
        'W_elec_hourly'  : W_elec_hourly,
        'Q_waste_hourly' : Q_waste_hourly,
        'W_engine_hourly': W_engine_hourly,
        'Q_excess_hourly': Q_excess_hourly,
        'Q_reject_hourly': Q_reject_hourly,
        'annual'         : annual,
        'monthly'        : monthly,
        'performance'    : performance,
    }

