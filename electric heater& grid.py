"""
electric_heater_grid.py

Reusable electric heater + grid backup module for hourly energy-system simulation.

Assumptions
-----------
1. Time step = 1 hour
   Therefore:
   - MW over one time step is treated as MWh
   - Cost = MW * (EUR/MWh) * 1h = EUR

2. Timestep 0 corresponds to Monday 00:00

3. Default demand generation:
   - Weekdays: operate from 08:00 to 18:00 (10 hours)
   - Weekends: no operation

4. Default electricity price:
   - Weekdays and weekends use different TOU tariffs

5. Default PV profile:
   - Simplified synthetic profile with daily solar shape
   - Includes a simple seasonal factor
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, Dict, Any


# ============================================================
# 1. CONFIGURATION
# ============================================================
@dataclass
class HeaterGridConfig:
    """Configuration for electric-heater + grid backup section."""

    # Electric heater
    heater_p_max_mw: float = 10.0
    heater_efficiency: float = 0.99

    # Grid
    grid_import_max_mw: float = float("inf")

    # Operating schedule
    weekday_start_hour: int = 8
    weekday_end_hour: int = 18   # exclusive

    weekend_start_hour: int = 0
    weekend_end_hour: int = 0    # exclusive; 0->0 means no weekend operation

    # Weekday TOU price
    weekday_peak_start_hour: int = 8
    weekday_peak_end_hour: int = 20
    weekday_peak_price_eur_per_mwh: float = 140.0
    weekday_offpeak_price_eur_per_mwh: float = 70.0

    # Weekend TOU price
    weekend_peak_start_hour: int = 10
    weekend_peak_end_hour: int = 18
    weekend_peak_price_eur_per_mwh: float = 110.0
    weekend_offpeak_price_eur_per_mwh: float = 60.0


# ============================================================
# 2. COMPONENT MODELS
# ============================================================
class ElectricHeater:
    """Converts electricity (MW_el) into heat (MW_th)."""

    def __init__(self, p_max_mw: float, efficiency: float = 0.99):
        if p_max_mw < 0:
            raise ValueError("Electric heater max power must be non-negative.")
        if not (0 < efficiency <= 1):
            raise ValueError("Electric heater efficiency must be in (0, 1].")

        self.p_max_mw = p_max_mw
        self.efficiency = efficiency

    def dispatch(self, heat_demand_mw: float, available_power_mw: float) -> Tuple[float, float]:
        """
        Dispatch heater using available electricity.

        Parameters
        ----------
        heat_demand_mw : float
            Required heat [MW_th]
        available_power_mw : float
            Available electricity [MW_el]

        Returns
        -------
        p_eh_mw : float
            Electrical power used [MW_el]
        q_eh_mw : float
            Heat produced [MW_th]
        """
        heat_demand_mw = max(0.0, heat_demand_mw)
        available_power_mw = max(0.0, available_power_mw)

        q_max_mw = self.p_max_mw * self.efficiency
        q_eh_mw = min(heat_demand_mw, q_max_mw)
        p_eh_mw = q_eh_mw / self.efficiency

        if p_eh_mw > available_power_mw:
            p_eh_mw = available_power_mw
            q_eh_mw = p_eh_mw * self.efficiency

        return p_eh_mw, q_eh_mw


class Grid:
    """Grid model for import-limited electricity purchase."""

    def __init__(self, price_import_eur_per_mwh: Sequence[float], p_max_import_mw: float = float("inf")):
        if p_max_import_mw < 0:
            raise ValueError("Grid import max power must be non-negative.")

        self.price_import_eur_per_mwh = list(price_import_eur_per_mwh)
        self.p_max_import_mw = p_max_import_mw

    def import_power(self, demand_mw: float) -> float:
        """Return grid import power [MW]."""
        return min(max(0.0, demand_mw), self.p_max_import_mw)

    def compute_cost(self, p_grid_mw: float, t: int) -> float:
        """
        Compute hourly electricity import cost [EUR].

        Assumes hourly timestep:
        MW * (EUR/MWh) * 1h = EUR
        """
        return max(0.0, p_grid_mw) * self.price_import_eur_per_mwh[t]


# ============================================================
# 3. TIME / SCHEDULE HELPERS
# ============================================================
def build_is_weekend(time_steps: int, monday_as_day1: bool = True) -> list[bool]:
    """
    Return a boolean vector where weekend=True for Saturday/Sunday.

    Assumes timestep 0 corresponds to Monday 00:00.
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be positive.")

    if not monday_as_day1:
        raise NotImplementedError("Current implementation assumes timestep 0 is Monday 00:00.")

    is_weekend = [False] * time_steps
    for t in range(time_steps):
        day_index = t // 24
        weekday_idx = day_index % 7  # 0..6 => Mon..Sun
        is_weekend[t] = weekday_idx >= 5
    return is_weekend


def build_operating_schedule(time_steps: int, config: HeaterGridConfig) -> list[bool]:
    """Operating schedule with weekday/weekend distinction."""
    is_weekend = build_is_weekend(time_steps)
    schedule = [False] * time_steps

    for t in range(time_steps):
        hour = t % 24
        if is_weekend[t]:
            schedule[t] = config.weekend_start_hour <= hour < config.weekend_end_hour
        else:
            schedule[t] = config.weekday_start_hour <= hour < config.weekday_end_hour

    return schedule


# ============================================================
# 4. PROFILE GENERATORS
# ============================================================
def generate_heat_demand(time_steps: int, demand_mw: float, config: HeaterGridConfig) -> list[float]:
    """
    Generate industrial heat demand [MW_th].

    During scheduled operating hours:
        heat_demand = demand_mw
    Otherwise:
        heat_demand = 0
    """
    if demand_mw < 0:
        raise ValueError("Heat demand must be non-negative.")

    schedule = build_operating_schedule(time_steps, config)
    heat_demand = [0.0] * time_steps

    for t in range(time_steps):
        if schedule[t]:
            heat_demand[t] = demand_mw

    return heat_demand


def generate_electricity_price(time_steps: int, config: HeaterGridConfig) -> list[float]:
    """
    Generate hourly TOU electricity price [EUR/MWh]
    with weekday/weekend differentiation.
    """
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


def generate_pv_profile(time_steps: int, peak_power_mw: float = 4.0) -> list[float]:
    """
    Generate simplified hourly PV production [MW] on DC side.

    Features:
    - Daily solar production curve
    - Simple seasonal factor
    - No weather variability
    """
    if peak_power_mw < 0:
        raise ValueError("PV peak power must be non-negative.")

    pv = [0.0] * time_steps
    for t in range(time_steps):
        hour = t % 24
        day_of_year = (t // 24) % 365

        # Daily daylight shape: sunrise ~6, sunset ~18
        daily_shape = max(0.0, math.sin(math.pi * (hour - 6) / 12))

        # Seasonal factor: slightly higher in summer, lower in winter
        season_factor = 0.75 + 0.25 * math.sin(2 * math.pi * (day_of_year - 80) / 365)

        pv[t] = peak_power_mw * season_factor * daily_shape

    return pv


# ============================================================
# 5. MAIN SIMULATION FUNCTION
# ============================================================
def simulate_electric_heater_grid_section(
    heat_demand_mw: Sequence[float],
    pv_power_dc_mw: Sequence[float],
    electricity_price_eur_per_mwh: Sequence[float],
    config: HeaterGridConfig,
    inverter_efficiency: float = 0.97,
    external_power_limit_mw: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Simulate electric heater + grid section over an hourly time series.

    Parameters
    ----------
    heat_demand_mw : Sequence[float]
        Hourly heat demand [MW_th]
    pv_power_dc_mw : Sequence[float]
        Hourly PV production on DC side [MW]
    electricity_price_eur_per_mwh : Sequence[float]
        Hourly electricity price [EUR/MWh]
    config : HeaterGridConfig
        System configuration
    inverter_efficiency : float
        PV inverter efficiency [-]
    external_power_limit_mw : Optional[Sequence[float]]
        Optional external electric power cap [MW],
        for integration with other modules

    Returns
    -------
    results : dict
        Detailed hourly and summary results
    """
    if not (0 < inverter_efficiency <= 1):
        raise ValueError("Inverter efficiency must be in (0, 1].")

    heat_demand_mw = list(heat_demand_mw)
    pv_power_dc_mw = list(pv_power_dc_mw)
    electricity_price_eur_per_mwh = list(electricity_price_eur_per_mwh)

    time_steps = len(heat_demand_mw)

    if len(pv_power_dc_mw) != time_steps or len(electricity_price_eur_per_mwh) != time_steps:
        raise ValueError("heat demand, PV profile, and price profile must have the same length")

    if external_power_limit_mw is None:
        external_power_limit_mw = [float("inf")] * time_steps
    else:
        external_power_limit_mw = list(external_power_limit_mw)
        if len(external_power_limit_mw) != time_steps:
            raise ValueError("external_power_limit_mw length must match heat demand")

    is_operating = build_operating_schedule(time_steps, config)
    is_weekend = build_is_weekend(time_steps)

    heater = ElectricHeater(config.heater_p_max_mw, config.heater_efficiency)
    grid = Grid(electricity_price_eur_per_mwh, config.grid_import_max_mw)

    p_eh_total_mw = [0.0] * time_steps
    q_eh_mw = [0.0] * time_steps
    p_grid_mw = [0.0] * time_steps
    cost_eur = [0.0] * time_steps
    q_residual_mw = [0.0] * time_steps
    pv_power_ac_mw = [0.0] * time_steps
    pv_used_mw = [0.0] * time_steps
    pv_unused_mw = [0.0] * time_steps

    for t in range(time_steps):
        # Demand is assumed already prepared by upstream profile generation
        demand_t = max(0.0, heat_demand_mw[t])

        pv_power_ac_t = max(0.0, pv_power_dc_mw[t]) * inverter_efficiency
        available_pv_t = min(pv_power_ac_t, max(0.0, external_power_limit_mw[t]))

        # Step 1: use PV first
        p_eh_from_pv_t, q_eh_t = heater.dispatch(demand_t, available_pv_t)

        # Step 2: use grid for remaining heat demand
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
        q_residual_mw[t] = max(demand_t - q_eh_t, 0.0)

        pv_power_ac_mw[t] = pv_power_ac_t
        pv_used_mw[t] = p_eh_from_pv_t
        pv_unused_mw[t] = max(0.0, available_pv_t - p_eh_from_pv_t)

    results = {
        # Hourly results
        "P_EH_MW": p_eh_total_mw,
        "Q_EH_MW": q_eh_mw,
        "P_grid_MW": p_grid_mw,
        "cost_EUR": cost_eur,
        "Q_residual_MW": q_residual_mw,
        "PV_power_AC_MW": pv_power_ac_mw,
        "PV_used_MW": pv_used_mw,
        "PV_unused_MW": pv_unused_mw,
        "is_operating": is_operating,
        "is_weekend": is_weekend,
        "electricity_price_EUR_per_MWh": electricity_price_eur_per_mwh,

        # Summary results
        "total_cost_EUR": sum(cost_eur),
        "total_grid_import_MWh": sum(p_grid_mw),
        "total_heat_from_EH_MWh": sum(q_eh_mw),
        "total_unserved_heat_MWh": sum(q_residual_mw),
        "total_pv_used_MWh": sum(pv_used_mw),
        "avg_grid_import_MW": sum(p_grid_mw) / time_steps,
    }

    return results


# ============================================================
# 6. PLOTTING FUNCTIONS (SEPARATE FROM SIMULATION)
# ============================================================
def generate_summary_plots(
    results: Dict[str, Any],
    output_dir: str | Path = "output_plots",
    file_prefix: str = "eh_grid",
) -> Dict[str, str]:
    """
    Generate summary plots.

    For non-8760 data, monthly cost will be skipped.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _generate_summary_svgs(results, out, file_prefix)

    p_grid = results["P_grid_MW"]
    q_eh = results["Q_EH_MW"]
    cost = results["cost_EUR"]
    is_operating = results["is_operating"]
    n = len(p_grid)

    files = {}

    # --------------------------------------------------------
    # Plot 1: Daily power chart (first 24h)
    # --------------------------------------------------------
    hours = list(range(min(24, n)))
    fig1 = plt.figure(figsize=(10, 4.5))
    plt.plot(hours, q_eh[:len(hours)], label="Q_EH_MW (Heat)", linewidth=2)
    plt.plot(hours, p_grid[:len(hours)], label="P_grid_MW (Grid)", linewidth=2)
    plt.title("Daily Power Profile")
    plt.xlabel("Hour of Day")
    plt.ylabel("Power [MW]")
    plt.grid(alpha=0.25)
    plt.legend()
    daily_path = out / f"{file_prefix}_daily_power.png"
    fig1.tight_layout()
    fig1.savefig(daily_path, dpi=140)
    plt.close(fig1)
    files["daily_power"] = str(daily_path)

    # --------------------------------------------------------
    # Plot 2: Weekly-view heatmap
    # --------------------------------------------------------
    hours_per_week = 168
    weeks = (n + hours_per_week - 1) // hours_per_week
    heatmap = [[0.0 for _ in range(hours_per_week)] for _ in range(weeks)]
    for t, op in enumerate(is_operating):
        w = t // hours_per_week
        hw = t % hours_per_week
        heatmap[w][hw] = 1.0 if op else 0.0

    fig2 = plt.figure(figsize=(12, 5))
    plt.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Weekly Operating Status View")
    plt.xlabel("Hour in Week (0-167)")
    plt.ylabel("Week Index")
    cbar = plt.colorbar()
    cbar.set_label("Operating (1) / Off (0)")
    weekly_path = out / f"{file_prefix}_weekly_view.png"
    fig2.tight_layout()
    fig2.savefig(weekly_path, dpi=140)
    plt.close(fig2)
    files["weekly_view"] = str(weekly_path)

    # --------------------------------------------------------
    # Plot 3: Monthly cost (only for 8760h non-leap year)
    # --------------------------------------------------------
    if n == 8760:
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_cost = [0.0] * 12

        idx = 0
        for m, d in enumerate(days_per_month):
            h_month = d * 24
            monthly_cost[m] = sum(cost[idx:idx + h_month])
            idx += h_month

        fig3 = plt.figure(figsize=(10, 4.8))
        plt.bar(month_names, monthly_cost)
        plt.title("Monthly Grid Electricity Cost")
        plt.xlabel("Month")
        plt.ylabel("Cost [EUR]")
        plt.grid(axis="y", alpha=0.25)
        monthly_path = out / f"{file_prefix}_monthly_cost.png"
        fig3.tight_layout()
        fig3.savefig(monthly_path, dpi=140)
        plt.close(fig3)
        files["monthly_cost"] = str(monthly_path)

    return files


def _generate_summary_svgs(results: Dict[str, Any], out: Path, file_prefix: str) -> Dict[str, str]:
    """Dependency-free SVG fallback output."""
    p_grid = results["P_grid_MW"]
    q_eh = results["Q_EH_MW"]
    cost = results["cost_EUR"]
    is_operating = results["is_operating"]
    n = len(p_grid)

    daily_path = out / f"{file_prefix}_daily_power.svg"
    weekly_path = out / f"{file_prefix}_weekly_view.svg"

    _write_daily_power_svg(daily_path, q_eh[:24], p_grid[:24])
    _write_weekly_view_svg(weekly_path, is_operating, n)

    files = {
        "daily_power": str(daily_path),
        "weekly_view": str(weekly_path),
    }

    if n == 8760:
        monthly_path = out / f"{file_prefix}_monthly_cost.svg"
        _write_monthly_cost_svg(monthly_path, cost)
        files["monthly_cost"] = str(monthly_path)

    return files


def _write_daily_power_svg(path: Path, q_day: Sequence[float], p_day: Sequence[float]) -> None:
    w, h = 960, 360
    margin = 50
    plot_w, plot_h = w - 2 * margin, h - 2 * margin
    ymax = max(max(q_day) if q_day else 0.0, max(p_day) if p_day else 0.0, 1.0)

    def points(vals: Sequence[float]) -> str:
        pts = []
        n = max(1, len(vals) - 1)
        for i, v in enumerate(vals):
            x = margin + (i / n) * plot_w
            y = margin + plot_h - (v / ymax) * plot_h
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{w/2}' y='28' text-anchor='middle' font-size='18'>Daily Power Profile</text>
<line x1='{margin}' y1='{margin+plot_h}' x2='{w-margin}' y2='{margin+plot_h}' stroke='black'/>
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{margin+plot_h}' stroke='black'/>
<polyline points='{points(q_day)}' fill='none' stroke='#D1495B' stroke-width='2.5'/>
<polyline points='{points(p_day)}' fill='none' stroke='#2E86AB' stroke-width='2.5'/>
<text x='{margin+10}' y='{margin+18}' fill='#D1495B' font-size='13'>Q_EH_MW</text>
<text x='{margin+100}' y='{margin+18}' fill='#2E86AB' font-size='13'>P_grid_MW</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_weekly_view_svg(path: Path, is_operating: Sequence[bool], n: int) -> None:
    hours_per_week = 168
    weeks = (n + hours_per_week - 1) // hours_per_week
    cell_w, cell_h = 4, 10
    margin_x, margin_y = 80, 40
    w = margin_x + hours_per_week * cell_w + 20
    h = margin_y + weeks * cell_h + 30

    rects = []
    for t, op in enumerate(is_operating):
        ww = t // hours_per_week
        hh = t % hours_per_week
        x = margin_x + hh * cell_w
        y = margin_y + ww * cell_h
        color = "#1B9AAA" if op else "#E6E6E6"
        rects.append(f"<rect x='{x}' y='{y}' width='{cell_w}' height='{cell_h}' fill='{color}'/>")

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{w/2}' y='24' text-anchor='middle' font-size='18'>Weekly Operating Status View</text>
<text x='10' y='{margin_y+15}' font-size='12'>Week</text>
<text x='{margin_x}' y='{h-8}' font-size='12'>Hour in Week (0-167)</text>
{''.join(rects)}
</svg>"""
    path.write_text(svg, encoding="utf-8")


def _write_monthly_cost_svg(path: Path, cost: Sequence[float]) -> None:
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_cost = [0.0] * 12
    idx = 0
    for m, d in enumerate(days_per_month):
        h_month = d * 24
        monthly_cost[m] = sum(cost[idx:idx + h_month])
        idx += h_month

    w, h = 980, 360
    margin = 50
    plot_w, plot_h = w - 2 * margin, h - 2 * margin
    bar_w = plot_w / 12 * 0.7
    ymax = max(monthly_cost) if monthly_cost else 1.0
    if ymax <= 0:
        ymax = 1.0

    bars = []
    labels = []
    for i, v in enumerate(monthly_cost):
        x = margin + i * (plot_w / 12) + ((plot_w / 12) - bar_w) / 2
        bh = (v / ymax) * plot_h
        y = margin + plot_h - bh
        bars.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{bh:.1f}' fill='#4C956C'/>")
        labels.append(f"<text x='{x + bar_w/2:.1f}' y='{margin+plot_h+16}' text-anchor='middle' font-size='11'>{month_names[i]}</text>")

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>
<rect width='100%' height='100%' fill='white'/>
<text x='{w/2}' y='24' text-anchor='middle' font-size='18'>Monthly Grid Electricity Cost</text>
<line x1='{margin}' y1='{margin+plot_h}' x2='{w-margin}' y2='{margin+plot_h}' stroke='black'/>
<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{margin+plot_h}' stroke='black'/>
{''.join(bars)}
{''.join(labels)}
</svg>"""
    path.write_text(svg, encoding="utf-8")


# ============================================================
# 7. TEST / DEMO
# ============================================================
if __name__ == "__main__":
    cfg = HeaterGridConfig()
    n = 8760

    # Generate input profiles
    heat = generate_heat_demand(time_steps=n, demand_mw=10.0, config=cfg)
    price = generate_electricity_price(time_steps=n, config=cfg)
    pv = generate_pv_profile(time_steps=n, peak_power_mw=4.0)

    # Run simulation
    result = simulate_electric_heater_grid_section(
        heat_demand_mw=heat,
        pv_power_dc_mw=pv,
        electricity_price_eur_per_mwh=price,
        config=cfg,
        inverter_efficiency=0.97,
    )

    # Generate plots separately
    plot_files = generate_summary_plots(
        result,
        output_dir="output_plots",
        file_prefix="eh_grid",
    )

    # Print summary
    print("=== Electric Heater + Grid Section Test ===")
    print(f"Total yearly grid cost (EUR): {result['total_cost_EUR']:.2f}")
    print(f"Average grid import (MW): {result['avg_grid_import_MW']:.3f}")
    print(f"Total grid import (MWh): {result['total_grid_import_MWh']:.2f}")
    print(f"Total heat from EH (MWh): {result['total_heat_from_EH_MWh']:.2f}")
    print(f"Total unserved heat (MWh): {result['total_unserved_heat_MWh']:.2f}")
    print(f"Total PV used by EH (MWh): {result['total_pv_used_MWh']:.2f}")

    print("\nGenerated plots:")
    for key, value in plot_files.items():
        print(f"  - {key}: {value}")
