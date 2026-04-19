import numpy as np

def simulate_heat_pump_engine(
    Q_ST_hourly: np.ndarray,
    Q_demand_hourly: 10e6,
    T_cold_HP: float = 70.0,
    T_hot_HP: float = 157.0,
    T_cold_HE: float = 40.0,
    T_hot_HE: float = 157.0,
    COP_real: float = 2.32,
    epsilon_HE: float = 0.20,
    Q_waste_max: float = 5e6,
    # ── Economic parameters (source: Enerin IEA Annex 58) ──
    electricity_price: float = 70.0,   # €/MWh
    gas_price: float = 70.0,           # €/MWh
    maintenance_cost: float = 10.0,    # €/MWh recycled heat
    capex_per_kw: float = 700.0,       # €/kW (mid of 600–800)
    system_capacity_kw: float = 1000.0 # kW (1 MW reference unit)
) -> dict:
    """
    Simulate hourly operation of High-Temperature Heat Pump (HTHP)
    and Stirling Heat Engine, including economic cost calculation.

    Heat Pump parameters based on:
    Enerin HoegTemp, IEA HPT Annex 58, Table 1
    (Tsource=80→50°C, Tsink=154→160°C, COP=2.2–2.45)

    Economic parameters based on:
    Enerin IEA HPT Annex 58 project example
    (electricity=gas=70 €/MWh, maintenance=10 €/MWh recycled heat)

    Parameters
    ----------
    Q_ST_hourly : np.ndarray
        Hourly ST heat output array, shape (8760,), unit: Wh
    Q_demand_hourly : np.ndarray
        Hourly industrial heat demand array, shape (8760,), unit: Wh
    T_cold_HP : float
        Heat pump cold side temperature, °C. Default: 70°C
    T_hot_HP : float
        Heat pump hot side temperature, °C. Default: 157°C
    T_cold_HE : float
        Heat engine cold side temperature, °C. Default: 40°C
    T_hot_HE : float
        Heat engine hot side temperature, °C. Default: 157°C
    COP_real : float
        Actual heat pump COP. Default: 2.32 (Annex 58 Table 1)
    epsilon_HE : float
        Heat engine Carnot correction factor. Default: 0.20
    Q_waste_max : float
        Maximum available waste heat power, W. Default: 5e6 W
    electricity_price : float
        Electricity price, €/MWh. Default: 70 (Annex 58)
    gas_price : float
        Gas price for reference boiler, €/MWh. Default: 70 (Annex 58)
    maintenance_cost : float
        Maintenance cost, €/MWh recycled heat. Default: 10 (Annex 58)
    capex_per_kw : float
        Capital cost per kW thermal, €/kW. Default: 700 (Annex 58)
    system_capacity_kw : float
        Installed heat pump capacity, kW. Default: 1000 kW

    Returns
    -------
    dict with keys:
        'Q_HP_hourly'      : np.ndarray  Heat pump heat output [Wh]
        'W_elec_hourly'    : np.ndarray  Heat pump electricity input [Wh]
        'Q_waste_hourly'   : np.ndarray  Waste heat consumed [Wh]
        'W_engine_hourly'  : np.ndarray  Engine electricity output [Wh]
        'Q_excess_hourly'  : np.ndarray  Excess ST heat to engine [Wh]
        'Q_reject_hourly'  : np.ndarray  Engine heat rejection [Wh]
        'annual'           : dict        Annual energy + cost summary
        'monthly'          : dict        Monthly breakdown
        'performance'      : dict        COP and efficiency values
        'economics'        : dict        Full economic analysis
    """

    # ── Temperature conversion to Kelvin ─────────────────────────
    T_cold_HP_K = T_cold_HP + 273.15
    T_hot_HP_K  = T_hot_HP  + 273.15
    T_cold_HE_K = T_cold_HE + 273.15
    T_hot_HE_K  = T_hot_HE  + 273.15

    # ── Derived performance parameters ───────────────────────────
    COP_carnot = T_hot_HP_K / (T_hot_HP_K - T_cold_HP_K)
    epsilon_HP = COP_real / COP_carnot
    eta_carnot = 1.0 - T_cold_HE_K / T_hot_HE_K
    eta_real   = epsilon_HE * eta_carnot

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
            Q_HP              = Q_dem - Q_ST
            W_elec            = Q_HP / COP_real
            Q_waste_used      = min(Q_HP - W_elec, Q_waste_max)

            Q_HP_hourly[t]    = Q_HP
            W_elec_hourly[t]  = W_elec
            Q_waste_hourly[t] = Q_waste_used

        else:
            # ── Heat Engine mode ──────────────────────────────────
            Q_excess              = Q_ST - Q_dem
            W_out                 = eta_real * Q_excess
            Q_reject              = Q_excess - W_out

            Q_excess_hourly[t]    = Q_excess
            W_engine_hourly[t]    = W_out
            Q_reject_hourly[t]    = Q_reject

    # ── Annual energy summary ─────────────────────────────────────
    HP_heat_MWh     = float(np.sum(Q_HP_hourly)     / 1e6)
    HP_elec_MWh     = float(np.sum(W_elec_hourly)   / 1e6)
    waste_heat_MWh  = float(np.sum(Q_waste_hourly)  / 1e6)
    engine_elec_MWh = float(np.sum(W_engine_hourly) / 1e6)
    excess_heat_MWh = float(np.sum(Q_excess_hourly) / 1e6)
    heat_reject_MWh = float(np.sum(Q_reject_hourly) / 1e6)
    HP_hours        = int(np.sum(Q_HP_hourly > 0))
    engine_hours    = int(np.sum(W_engine_hourly > 0))
    COP_effective   = HP_heat_MWh / HP_elec_MWh if HP_elec_MWh > 0 else 0.0

    annual = {
        'HP_heat_MWh'    : HP_heat_MWh,
        'HP_elec_MWh'    : HP_elec_MWh,
        'waste_heat_MWh' : waste_heat_MWh,
        'engine_elec_MWh': engine_elec_MWh,
        'excess_heat_MWh': excess_heat_MWh,
        'heat_reject_MWh': heat_reject_MWh,
        'HP_hours'       : HP_hours,
        'engine_hours'   : engine_hours,
        'COP_effective'  : round(COP_effective, 3),
    }

    # ── Monthly breakdown ─────────────────────────────────────────
    days_per_month  = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_per_month = [d * 24 for d in days_per_month]
    HP_monthly      = np.zeros(12)
    engine_monthly  = np.zeros(12)
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

    # ── Performance parameters ────────────────────────────────────
    performance = {
        'COP_carnot' : round(COP_carnot, 3),
        'COP_real'   : round(COP_real,   3),
        'epsilon_HP' : round(epsilon_HP, 3),
        'eta_carnot' : round(eta_carnot, 3),
        'eta_real'   : round(eta_real,   3),
        'epsilon_HE' : round(epsilon_HE, 3),
    }

    # ── Economic analysis ─────────────────────────────────────────
    # Source: Enerin IEA HPT Annex 58

    # CAPEX
    capex_total = capex_per_kw * system_capacity_kw          # €

    # OPEX: electricity consumed by heat pump
    opex_electricity = HP_elec_MWh * electricity_price        # €/year

    # OPEX: maintenance (per MWh of recycled/waste heat)
    opex_maintenance = waste_heat_MWh * maintenance_cost      # €/year

    # Revenue: electricity generated by heat engine
    revenue_engine = engine_elec_MWh * electricity_price      # €/year

    # Total annual OPEX (net of engine revenue)
    opex_total = opex_electricity + opex_maintenance - revenue_engine

    # Reference: equivalent gas boiler cost for same heat output
    boiler_efficiency = 0.85                                   # Annex 58
    gas_consumption   = HP_heat_MWh / boiler_efficiency        # MWh gas
    cost_gas_boiler   = gas_consumption * gas_price            # €/year

    # Annual savings vs gas boiler
    annual_savings = cost_gas_boiler - opex_total              # €/year

    # Simple payback period
    if annual_savings > 0:
        payback_years = capex_total / annual_savings
    else:
        payback_years = float('inf')

    # Annual ROI
    roi = (annual_savings / capex_total) * 100 if capex_total > 0 else 0.0

    # LCOH: Levelized Cost of Heat
    # LCOH = (CAPEX/lifetime + OPEX) / annual_heat_output
    lifetime_years = 20                                        # Annex 58
    lcoh = ((capex_total / lifetime_years) + opex_total) / (HP_heat_MWh * 1000) if HP_heat_MWh > 0 else 0.0

    economics = {
        # Costs
        'capex_total_eur'      : round(capex_total,       2),
        'opex_electricity_eur' : round(opex_electricity,  2),
        'opex_maintenance_eur' : round(opex_maintenance,  2),
        'opex_total_eur'       : round(opex_total,        2),
        # Reference
        'cost_gas_boiler_eur'  : round(cost_gas_boiler,   2),
        'gas_consumption_MWh'  : round(gas_consumption,   2),
        # Revenue
        'revenue_engine_eur'   : round(revenue_engine,    2),
        # Key indicators
        'annual_savings_eur'   : round(annual_savings,    2),
        'payback_years'        : round(payback_years,     2),
        'roi_percent'          : round(roi,               2),
        'lcoh_eur_per_MWh'     : round(lcoh,              4),
        # Assumptions
        'electricity_price'    : electricity_price,
        'gas_price'            : gas_price,
        'maintenance_cost'     : maintenance_cost,
        'capex_per_kw'         : capex_per_kw,
        'lifetime_years'       : lifetime_years,
        'boiler_efficiency'    : boiler_efficiency,
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
        'economics'      : economics,
    }
