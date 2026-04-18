import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class HeaterGridConfig:
    """Configuration for electric-heater + grid backup section."""

    heater_p_max_mw: float = 10.0
    heater_efficiency: float = 0.99
    grid_import_max_mw: float = float("inf")

    weekday_start_hour: int = 8
    weekday_end_hour: int = 18   # exclusive
    weekend_start_hour: int = 0
    weekend_end_hour: int = 0    # exclusive; 0->0 means no weekend operation

    weekday_peak_start_hour: int = 8
    weekday_peak_end_hour: int = 20
    weekday_peak_price_eur_per_mwh: float = 140.0
    weekday_offpeak_price_eur_per_mwh: float = 70.0

    weekend_peak_start_hour: int = 10
    weekend_peak_end_hour: int = 18
    weekend_peak_price_eur_per_mwh: float = 110.0
    weekend_offpeak_price_eur_per_mwh: float = 60.0


class ElectricHeater:
    """Converts electricity (MW_el) into heat (MW_th)."""

    def __init__(self, p_max_mw: float, efficiency: float = 0.99):
        self.p_max_mw = p_max_mw
        self.efficiency = efficiency

    def dispatch(self, heat_demand_mw: float, available_power_mw: float) -> Tuple[float, float]:
        q_max_mw = self.p_max_mw * self.efficiency
        q_eh_mw = min(heat_demand_mw, q_max_mw)
        p_eh_mw = q_eh_mw / self.efficiency

        if p_eh_mw > available_power_mw:
            p_eh_mw = available_power_mw
            q_eh_mw = p_eh_mw * self.efficiency

        return p_eh_mw, q_eh_mw


class Grid:
    """Grid model for import-limited electricity purchase."""

    def __init__(self, price_import_eur_per_mwh: List[float], p_max_import_mw: float = float("inf")):
        self.price_import_eur_per_mwh = price_import_eur_per_mwh
        self.p_max_import_mw = p_max_import_mw

    def import_power(self, demand_mw: float) -> float:
        return min(demand_mw, self.p_max_import_mw)

    def compute_cost(self, p_grid_mw: float, t: int) -> float:
        return p_grid_mw * self.price_import_eur_per_mwh[t]


def build_is_weekend(time_steps: int, monday_as_day1: bool = True) -> List[bool]:
    """Return a bool vector where weekend=True for Sat/Sun."""
    if not monday_as_day1:
        raise NotImplementedError("Current implementation assumes day-1 is Monday.")

    is_weekend = [False] * time_steps
    for t in range(time_steps):
        day_index = t // 24
        weekday_idx = day_index % 7  # 0..6 => Mon..Sun
        is_weekend[t] = weekday_idx >= 5
    return is_weekend


def build_operating_schedule(time_steps: int, config: HeaterGridConfig) -> List[bool]:
    """Operating hours mask with weekday/weekend distinction."""
    is_weekend = build_is_weekend(time_steps)
    schedule = [False] * time_steps

    for t in range(time_steps):
        hour = t % 24
        if is_weekend[t]:
            schedule[t] = config.weekend_start_hour <= hour < config.weekend_end_hour
        else:
            schedule[t] = config.weekday_start_hour <= hour < config.weekday_end_hour

    return schedule


def generate_heat_demand(time_steps: int, demand_mw: float, config: HeaterGridConfig) -> List[float]:
    """Industrial heat demand with different weekday/weekend working windows."""
    schedule = build_operating_schedule(time_steps, config)
    heat_demand = [0.0] * time_steps
    for t in range(time_steps):
        if schedule[t]:
            heat_demand[t] = demand_mw
    return heat_demand


def generate_electricity_price(time_steps: int, config: HeaterGridConfig) -> List[float]:
    """TOU electricity price with weekday/weekend differentiation."""
    is_weekend = build_is_weekend(time_steps)
    price = [config.weekday_offpeak_price_eur_per_mwh] * time_steps

    for t in range(time_steps):
        hour = t % 24

        if is_weekend[t]:
            price[t] = config.weekend_offpeak_price_eur_per_mwh
            if config.weekend_peak_start_hour <= hour < config.weekend_peak_end_hour:
                price[t] = config.weekend_peak_price_eur_per_mwh
        else:
            if config.weekday_peak_start_hour <= hour < config.weekday_peak_end_hour:
                price[t] = config.weekday_peak_price_eur_per_mwh

    return price


def generate_pv_profile(time_steps: int, peak_power_mw: float = 4.0) -> List[float]:
    pv = [0.0] * time_steps
    for t in range(time_steps):
        hour = t % 24
        pv[t] = peak_power_mw * max(0.0, math.sin(math.pi * (hour - 6) / 12))
    return pv


def simulate_electric_heater_grid_section(
    heat_demand_mw: List[float],
    pv_power_dc_mw: List[float],
    electricity_price_eur_per_mwh: List[float],
    config: HeaterGridConfig,
    inverter_efficiency: float = 0.97,
    external_power_limit_mw: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Reusable electric-heater + grid function for main simulation cascade.

    external_power_limit_mw: optional vector for other modules to cap available electric power.
    """

    time_steps = len(heat_demand_mw)
    if len(pv_power_dc_mw) != time_steps or len(electricity_price_eur_per_mwh) != time_steps:
        raise ValueError("heat demand, PV profile and price profile must have same length")

    is_operating = build_operating_schedule(time_steps, config)
    is_weekend = build_is_weekend(time_steps)

    if external_power_limit_mw is None:
        external_power_limit_mw = [float("inf")] * time_steps
    elif len(external_power_limit_mw) != time_steps:
        raise ValueError("external_power_limit_mw length must match heat demand")

    heater = ElectricHeater(config.heater_p_max_mw, config.heater_efficiency)
    grid = Grid(electricity_price_eur_per_mwh, config.grid_import_max_mw)

    p_eh_total_mw = [0.0] * time_steps
    q_eh_mw = [0.0] * time_steps
    p_grid_mw = [0.0] * time_steps
    cost_eur = [0.0] * time_steps
    q_residual_mw = [0.0] * time_steps

    for t in range(time_steps):
        demand_t = heat_demand_mw[t] if is_operating[t] else 0.0
        pv_power_ac_t = pv_power_dc_mw[t] * inverter_efficiency
        available_pv_t = min(pv_power_ac_t, external_power_limit_mw[t])

        p_eh_from_pv_t, q_eh_t = heater.dispatch(demand_t, available_pv_t)

        remaining_heat = demand_t - q_eh_t
        if remaining_heat > 0:
            p_needed_grid_t = remaining_heat / heater.efficiency
            p_grid_t = grid.import_power(p_needed_grid_t)
            q_extra = p_grid_t * heater.efficiency
            q_eh_t += q_extra
            p_eh_t = p_eh_from_pv_t + p_grid_t
        else:
            p_grid_t = 0.0
            p_eh_t = p_eh_from_pv_t

        p_eh_total_mw[t] = p_eh_t
        q_eh_mw[t] = q_eh_t
        p_grid_mw[t] = p_grid_t
        cost_eur[t] = grid.compute_cost(p_grid_t, t)
        q_residual_mw[t] = max(heat_demand_mw[t] - q_eh_t, 0.0)

    return {
        "P_EH_MW": p_eh_total_mw,
        "Q_EH_MW": q_eh_mw,
        "P_grid_MW": p_grid_mw,
        "cost_EUR": cost_eur,
        "Q_residual_MW": q_residual_mw,
        "is_operating": is_operating,
        "is_weekend": is_weekend,
        "electricity_price_EUR_per_MWh": electricity_price_eur_per_mwh,
        "total_cost_EUR": sum(cost_eur),
    }


if __name__ == "__main__":
    cfg = HeaterGridConfig()
    n = 8760

    heat = generate_heat_demand(time_steps=n, demand_mw=10.0, config=cfg)
    price = generate_electricity_price(time_steps=n, config=cfg)
    pv = generate_pv_profile(time_steps=n, peak_power_mw=4.0)

    result = simulate_electric_heater_grid_section(
        heat_demand_mw=heat,
        pv_power_dc_mw=pv,
        electricity_price_eur_per_mwh=price,
        config=cfg,
    )

    avg_grid = sum(result["P_grid_MW"]) / len(result["P_grid_MW"]) if result["P_grid_MW"] else 0.0
    print(f"Total yearly grid cost (EUR): {result['total_cost_EUR']:.2f}")
    print(f"Average grid import (MW): {avg_grid:.3f}")
