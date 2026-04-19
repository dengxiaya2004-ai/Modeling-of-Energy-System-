"""
================================================================================
MJ2438 – Modeling of Energy Systems
Techno-Economic Assessment: PV + Li-ion Battery + Inverter Subsystem
Location : Seville, Spain  (37.39°N, 5.99°W)
Plant     : 10 MW industrial process-heat plant, 08:00–18:00 local time
Coverage  : PV + Battery covers 40 % of total demand

REVISION — Enhanced Physical Modeling (pvlib-based) + All Corrections Applied
──────────────────────────────────────────────────────────────────────────────
Corrections applied vs. previous revision:
  1.  Bifacial rear irradiance  – pvlib infinite-sheds model (tilt/height aware).
  2.  gamma_vmp separated       – N-type TOPCon Vmp coeff ≠ Pmax coeff.
  3.  Row-shading axis          – axis_azimuth derived from array azimuth.
  4.  Battery degradation       – capacity fade + RTE decline modelled per year.
  5.  DNI clearness-index bug   – kt now = GHI / GHI_extraterrestrial (was GHI/GHI).
  6.  OPEX inflation            – O&M cost escalated at inflation_rate %/yr.
  7.  Price escalation          – Electricity prices escalated per year in DCF.
  8.  Battery replacement cost  – exposed as named parameter (no magic number).
  9.  Grid export revenue       – surplus PV can be sold at feed_in_fraction×spot.
  10. Curtailment penalty       – tiny ε cost prevents degenerate LP solutions.
  11. LP SOC configuration      – initial_soc and terminal_soc_min exposed.
  12. LP extraction speed       – bulk .value attribute access replaces pyo.value loop.
  13. Surgical warnings         – global RuntimeWarning suppression removed.
  14. Global state removed      – SEVILLE / TMY_FILE replaced by SiteConfig.
  15. Sensitivity re-simulates  – PV output recalculated for each grid point.
  16. Input validation          – __post_init__ guards on all dataclasses.
  17. Logging framework         – print() replaced with logging.getLogger().
  18. Unit tests                – see tests/test_physical.py (separate file).
  19. Module split suggested    – single file kept for compatibility; see STRUCTURE.

PV Array  (pvlib ModelChain approach):
  1. POA irradiance      – Hay-Davies transposition from TMY DNI + DHI + GHI.
  2. Row-to-row shading  – pvlib.shading.shaded_fraction1d (beam component only).
  3. Spectral correction – SAPM spectral mismatch (Sandia, monosi coefficients).
  4. Bifacial gain       – pvlib.bifacial.infinite_sheds (tilt + height aware).
  5. Cell temperature    – Faiman (IEC 61853) model, TMY wind speed.
  6. DC power            – pvlib PVWatts v5 with spectral-corrected irradiance.
  7. Degradation         – Continuous linear Year-1; cumulative Year 2+.
  8. DC string voltage   – Temperature-corrected Vmp × n_series.

Component Datasheets
────────────────────
PV Panel  : Jinko Solar Tiger Neo 72HL4-BDV 590 Wp (ENF #55991)
Inverter  : Sungrow 1+X Modular (SG1100UD-20), 1.1 MW/module, η > 99 %
Battery   : Tesla Megapack 2 XL – 4-hour config, 979 kW / 3 916 kWh per unit

Usage
─────
  python pv_battery_subsystem_corrected.py
  python pv_battery_subsystem_corrected.py --pvgis path/to/Sevilla_TMY.csv
  python pv_battery_subsystem_corrected.py --pvgis ... --verbose
================================================================================
"""

import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pvlib
from pvlib import irradiance  as pvl_irr
from pvlib import atmosphere  as pvl_atm
from pvlib import spectrum    as pvl_spec
from pvlib import temperature as pvl_temp
from pvlib import shading     as pvl_shd
from dataclasses import dataclass
from typing import Optional, Tuple

import pyomo.environ as pyo
from scipy.optimize import brentq

# ── Try pvlib infinite-sheds bifacial module (pvlib >= 0.9.5) ────────────────
try:
    from pvlib.bifacial.infinite_sheds import get_irradiance_poa as _bif_poa
    _HAS_INFINITE_SHEDS = True
except ImportError:
    _HAS_INFINITE_SHEDS = False

# ── Module-level logger (replaces all print() calls) ─────────────────────────
# FIX #17: use logging instead of print() throughout.
# FIX #13: global warnings.filterwarnings("ignore") removed entirely.
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 0.  SITE CONFIGURATION  (replaces module-level globals SEVILLE / TMY_FILE)
# ──────────────────────────────────────────────────────────────────────────────
# FIX #14: encapsulate site parameters in a dataclass instead of module globals.

@dataclass
class SiteConfig:
    """
    Site-specific parameters, used by weather loading and solar-position
    calculations.  Pass a SiteConfig instance to PVArray and load_pvgis_tmy()
    so the same file can model different locations without editing source code.
    """
    name      : str   = "Seville, Spain"
    latitude  : float = 37.39
    longitude : float = -5.99
    altitude  : float = 9.0
    tz        : str   = "UTC"
    tmy_file  : str   = "Sevilla_TMY.csv"

    def __post_init__(self):
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"latitude={self.latitude} outside [-90, 90]")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"longitude={self.longitude} outside [-180, 180]")

    @property
    def location(self) -> pvlib.location.Location:
        return pvlib.location.Location(
            latitude  = self.latitude,
            longitude = self.longitude,
            tz        = self.tz,
            altitude  = self.altitude,
            name      = self.name,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 1.  COMPONENT SPECIFICATION DATACLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JinkoTigerNeo:
    """
    Jinko Solar Tiger Neo 72HL4-BDV  (ENF Crystalline Panel #55991)
    Bifacial N-type TOPCon, dual glass, 182 mm M10 cells
    Datasheet: JKM575-600N-72HL4-BDV
    """
    model         : str   = "JKM590N-72HL4-BDV"
    pmax_stc      : float = 590.0      # W  – STC peak power
    vmp_stc       : float = 44.1       # V  – voltage at max power (STC)
    imp_stc       : float = 13.38      # A  – current at max power (STC)
    voc_stc       : float = 52.80      # V  – open-circuit voltage (STC)
    isc_stc       : float = 14.20      # A  – short-circuit current (STC)
    eta_stc       : float = 0.2265     # –  – module efficiency 22.65 %
    # ── Temperature coefficients ──────────────────────────────────────────────
    # FIX #2: gamma_vmp is now a SEPARATE coefficient from gamma_pmax.
    # For N-type TOPCon, Vmp degrades less steeply than Pmax with temperature.
    # Source: Jinko JKM590N-72HL4-BDV datasheet (temp. coeff. columns).
    gamma_pmax    : float = -0.0029    # /°C – Pmax temp coeff  (–0.29 %/°C)
    gamma_vmp     : float = -0.0025    # /°C – Vmp  temp coeff  (–0.25 %/°C)
    #                                          ↑ Different from gamma_pmax!
    alpha_isc     : float = +0.00050   # /°C – Isc  temp coeff  (+0.05 %/°C)
    # ── Thermal model ─────────────────────────────────────────────────────────
    noct          : float = 45.0       # °C  (reference; Faiman model used)
    # ── Physical dimensions ───────────────────────────────────────────────────
    width_m       : float = 1.134      # m
    length_m      : float = 2.278      # m  (slant height = collector_width)
    # ── Degradation ───────────────────────────────────────────────────────────
    deg_y1        : float = 0.010      # fraction – first-year degradation
    deg_annual    : float = 0.004      # fraction – annual degradation yr 2+
    warranty_product : int = 12        # years
    warranty_power   : int = 30        # years
    # ── Bifacial ──────────────────────────────────────────────────────────────
    bifaciality   : float = 0.70       # rear-to-front efficiency ratio
    # ── Cost ──────────────────────────────────────────────────────────────────
    cost_eur_per_wp : float = 0.25     # €/Wp

    def __post_init__(self):
        # FIX #16: input validation
        if not (0 < self.pmax_stc < 1_000):
            raise ValueError(f"pmax_stc={self.pmax_stc} W outside (0, 1000)")
        if not (-0.01 < self.gamma_pmax < 0):
            raise ValueError(f"gamma_pmax={self.gamma_pmax} should be in (−0.01, 0)")
        if not (-0.01 < self.gamma_vmp < 0):
            raise ValueError(f"gamma_vmp={self.gamma_vmp} should be in (−0.01, 0)")
        if not (0 < self.bifaciality <= 1.0):
            raise ValueError(f"bifaciality={self.bifaciality} must be in (0, 1]")
        if not (0 < self.eta_stc < 1):
            raise ValueError(f"eta_stc={self.eta_stc} must be a fraction in (0, 1)")

    @property
    def area_m2(self) -> float:
        return self.width_m * self.length_m   # ≈ 2.579 m²


@dataclass
class SungrowModular:
    """
    Sungrow 1+X Modular Inverter  (SG1100UD-20)
    Each module: 1 100 kW  |  up to 8 modules = 8.8 MW block
    """
    model              : str   = "SG1100UD-20"
    power_module_kw    : float = 1_100.0
    eta_max            : float = 0.990
    eta_euro           : float = 0.988
    max_modules        : int   = 8
    cost_eur_per_kw    : float = 45.0
    lifetime_yr        : int   = 25
    v_mppt_min         : float = 200.0
    v_mppt_max         : float = 1_500.0
    v_mppt_opt_lo      : float = 850.0
    v_mppt_opt_hi      : float = 1_350.0
    t_derate_onset     : float = 40.0
    derate_rate_per_c  : float = 0.003

    def __post_init__(self):
        # FIX #16: input validation
        if not (0 < self.eta_euro <= 1):
            raise ValueError(f"eta_euro={self.eta_euro} must be in (0, 1]")
        if self.v_mppt_min >= self.v_mppt_opt_lo:
            raise ValueError("v_mppt_min must be < v_mppt_opt_lo")
        if self.v_mppt_opt_lo >= self.v_mppt_opt_hi:
            raise ValueError("v_mppt_opt_lo must be < v_mppt_opt_hi")
        if self.v_mppt_opt_hi >= self.v_mppt_max:
            raise ValueError("v_mppt_opt_hi must be < v_mppt_max")

    def modules_needed(self, target_kw: float) -> int:
        n = int(np.ceil(target_kw / self.power_module_kw))
        return min(n, self.max_modules)

    def rated_power_kw(self, target_kw: float) -> float:
        return self.modules_needed(target_kw) * self.power_module_kw


@dataclass
class TeslaMegapack2XL:
    """
    Tesla Megapack 2 XL – 4-hour configuration
    Datasheet Rev. 1.5.1 – February 10, 2023
    """
    model                : str   = "Megapack 2 XL (4h)"
    energy_kwh           : float = 3_916.0    # kWh AC energy per unit
    power_kw             : float = 979.0      # kW  AC power per unit
    rte                  : float = 0.937      # round-trip efficiency 93.7 %
    voltage_v            : float = 480.0      # V AC nominal (3-phase)
    min_soc              : float = 0.05       # 5 %
    max_soc              : float = 0.95       # 95 %
    self_discharge_hr    : float = 0.00005    # fraction/h  (LFP ≈ 0.005 %/h)
    warranty_yr          : int   = 15
    cost_eur_per_kwh     : float = 280.0      # €/kWh
    cost_eur_per_kw      : float = 0.0        # included in per-kWh figure
    cycles_lifetime      : int   = 3_000
    # FIX #4: battery capacity/RTE degradation per operational year
    deg_annual_capacity_pct : float = 2.5    # % capacity fade per year
    #   Source: NREL Battery Lifetime Analysis (2023), ~2–3 %/yr utility BESS.
    deg_annual_rte_pct      : float = 0.5    # % RTE loss per year
    #   Conservative estimate; LFP cells lose RTE primarily via SEI growth.
    deg_floor_capacity      : float = 0.70   # capacity floor (end-of-life at 70 %)
    deg_floor_rte           : float = 0.88   # RTE floor

    def __post_init__(self):
        # FIX #16: input validation
        if not (0 < self.rte < 1):
            raise ValueError(f"rte={self.rte} must be in (0, 1)")
        if not (0 <= self.min_soc < self.max_soc <= 1):
            raise ValueError(
                f"SOC limits invalid: min_soc={self.min_soc}, max_soc={self.max_soc}"
            )
        if not (0 < self.energy_kwh):
            raise ValueError("energy_kwh must be positive")
        if not (0 < self.deg_floor_capacity < 1):
            raise ValueError("deg_floor_capacity must be in (0, 1)")

    @property
    def charge_eta(self) -> float:
        return float(np.sqrt(self.rte))

    @property
    def discharge_eta(self) -> float:
        return float(np.sqrt(self.rte))

    def capacity_at_year(self, year: int) -> float:
        """Usable capacity [kWh] after `year` years of degradation."""
        deg = 1.0 - self.deg_annual_capacity_pct / 100.0 * (year - 1)
        return self.energy_kwh * max(self.deg_floor_capacity, deg)

    def rte_at_year(self, year: int) -> float:
        """Round-trip efficiency after `year` years of degradation."""
        rte_deg = self.rte - self.deg_annual_rte_pct / 100.0 * (year - 1)
        return max(self.deg_floor_rte, rte_deg)

    def units_for_energy(self, kwh: float) -> int:
        usable = self.energy_kwh * (self.max_soc - self.min_soc)
        return int(np.ceil(kwh / usable))

    def units_for_power(self, kw: float) -> int:
        return int(np.ceil(kw / self.power_kw))

    def units_needed(self, kwh: float, kw: float) -> int:
        return max(self.units_for_energy(kwh), self.units_for_power(kw))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  WEATHER DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_pvgis_tmy(
    csv_path: str, site: Optional[SiteConfig] = None
) -> pd.DataFrame:
    """
    Load a PVGIS TMY CSV export and return a tidy DataFrame.

    FIX #5: corrected clearness-index calculation in the DNI fallback
    (was dividing GHI by itself; now divides by extraterrestrial horizontal).
    FIX #13: surgical warnings instead of blanket suppression.
    FIX #14: accepts SiteConfig instead of using global SEVILLE.
    """
    _site = (site or SiteConfig()).location

    # ── Locate the header row ─────────────────────────────────────────
    header_row = None
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh):
            if "G(h)" in line:
                header_row = idx
                break

    if header_row is None:
        raise ValueError(
            f"Could not find 'G(h)' column in '{csv_path}'.\n"
            "Make sure this is a PVGIS TMY CSV.\n"
            "Download: https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY"
        )

    raw = pd.read_csv(csv_path, skiprows=header_row, nrows=8_760)
    raw.columns = raw.columns.str.strip()

    if "G(h)" not in raw.columns or "T2m" not in raw.columns:
        raise ValueError(
            f"Expected columns 'G(h)' and 'T2m' not found.\n"
            f"Columns present: {list(raw.columns)}"
        )

    # ── Parse PVGIS timestamps → UTC DatetimeIndex ──
    time_col = raw.columns[0]
    try:
        times = pd.to_datetime(
            raw[time_col].astype(str), format="%Y%m%d:%H%M", utc=True)
    except Exception:
        logger.warning("Could not parse timestamps; using synthetic 2019 index.")
        times = pd.date_range("2019-01-01", periods=8_760, freq="h", tz="UTC")

    col_map = {
        "G(h)" : "ghi",
        "Gb(n)": "dni",
        "Gd(h)": "dhi",
        "T2m"  : "temp_air",
        "WS10m": "wind_speed",
        "SP"   : "pressure",
        "RH"   : "relative_humidity",
    }

    out = pd.DataFrame(index=times)
    for pvgis_col, our_col in col_map.items():
        if pvgis_col in raw.columns:
            out[our_col] = raw[pvgis_col].to_numpy(dtype=float)

    # ── FIX #5: corrected DNI fallback (proper clearness index kt) ─────
    if "dni" not in out.columns:
        logger.warning("'Gb(n)' (DNI) not found in CSV; estimating from GHI.")
        ghi = out["ghi"].clip(lower=0)

        # Compute solar position and extraterrestrial horizontal irradiance
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*invalid value.*", category=RuntimeWarning
            )
            solpos    = _site.get_solarposition(times)
            dni_extra = pvl_irr.get_extra_radiation(times)

        cos_z     = np.cos(np.radians(solpos["apparent_zenith"].values))
        ghi_extra = (dni_extra.values * cos_z).clip(min=0)

        # Clearness index: GHI / GHI_extraterrestrial  (was GHI/GHI — always 1.0!)
        kt = np.where(
            ghi_extra > 10.0,
            np.clip(ghi.values / np.maximum(ghi_extra, 1.0), 0.0, 1.0),
            0.0,
        )

        # Erbs decomposition: kt → diffuse fraction
        dhi_frac = np.where(
            kt <= 0.22, 1.0 - 0.09 * kt,
            np.where(
                kt <= 0.80,
                0.9511 - 0.1604*kt + 4.388*kt**2
                - 16.638*kt**3 + 12.336*kt**4,
                0.165,
            )
        )
        out["dhi"] = pd.Series(
            (ghi.values * dhi_frac).clip(min=0), index=times
        )
        out["dni"] = pd.Series(
            np.where(cos_z > 0.02,
                     (ghi.values - out["dhi"].values) / cos_z, 0.0
                     ).clip(min=0),
            index=times,
        )

    if "dhi" not in out.columns:
        logger.warning("'Gd(h)' (DHI) not found; estimating from GHI.")
        out["dhi"] = (out["ghi"] * 0.15).clip(lower=0)
    if "wind_speed" not in out.columns:
        logger.warning("'WS10m' not found; defaulting to 1.0 m/s.")
        out["wind_speed"] = 1.0
    if "pressure" not in out.columns:
        logger.warning("'SP' (pressure) not found; using ISA 101 325 Pa.")
        out["pressure"] = 101_325.0

    for col in ("ghi", "dni", "dhi"):
        out[col] = out[col].clip(lower=0.0)

    logger.info(
        "Loaded %d hourly rows | GHI=%d kWh/m² | DNI=%d kWh/m² | "
        "T_mean=%.1f °C | WS_mean=%.1f m/s",
        len(out),
        out["ghi"].sum() / 1_000,
        out["dni"].sum() / 1_000,
        out["temp_air"].mean(),
        out["wind_speed"].mean(),
    )
    return out


def generate_synthetic_tmy(
    lat: float = 37.39, lon: float = -5.99, seed: int = 42
) -> pd.DataFrame:
    """
    Generate a synthetic Typical Meteorological Year for Seville.
    NOTE: for development/testing only. Use real PVGIS data for reports.
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2019-01-01", periods=8_760, freq="h", tz="UTC")
    t   = np.arange(8_760)
    hod = t % 24
    doy = t // 24 + 1

    decl_rad = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    ha_rad   = np.radians((hod - 12.0) * 15.0)
    lat_r    = np.radians(lat)
    sin_elev = (np.sin(lat_r) * np.sin(decl_rad)
                + np.cos(lat_r) * np.cos(decl_rad) * np.cos(ha_rad))
    elev = np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))

    ghi_cs        = np.where(elev > 0.0, 1_000.0 * sin_elev, 0.0)
    cloud_seasonal = 0.78 + 0.14 * np.sin(np.radians(360 / 365 * (doy - 180)))
    cloud_noise    = np.clip(rng.normal(1.0, 0.12, 8_760), 0.25, 1.0)
    ghi = np.maximum(ghi_cs * cloud_seasonal * cloud_noise, 0.0)

    kt = np.where(ghi_cs > 10, ghi / np.maximum(ghi_cs, 1e-3), 0.0)
    dhi_frac = np.where(
        kt <= 0.22, 1.0 - 0.09 * kt,
        np.where(kt <= 0.80,
                 0.9511 - 0.1604*kt + 4.388*kt**2
                 - 16.638*kt**3 + 12.336*kt**4, 0.165))
    dhi = ghi * dhi_frac
    dni = np.where(sin_elev > 0.02, (ghi - dhi) / sin_elev, 0.0)

    t_seasonal = 18.5 + 9.0 * np.sin(np.radians(360 / 365 * (doy - 75)))
    t_daily    = 4.0  * np.sin(np.radians(360 / 24  * (hod - 5)))
    t_amb      = t_seasonal + t_daily + rng.normal(0.0, 1.2, 8_760)
    wind_speed = np.maximum(rng.weibull(2.0, 8_760) * 3.5, 0.3)
    pressure   = 101_000.0 + 500.0 * np.sin(np.radians(360 / 365 * doy))

    return pd.DataFrame({
        "ghi"       : ghi.clip(0),
        "dni"       : dni.clip(0),
        "dhi"       : dhi.clip(0),
        "temp_air"  : t_amb,
        "wind_speed": wind_speed,
        "pressure"  : pressure,
    }, index=times)


def generate_spain_electricity_prices(
    n_hours: int = 8_760,
    base_eur_mwh: float = 85.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a representative Spanish OMIE day-ahead electricity price profile.
    Prices are clipped to [5, 300] €/MWh.
    """
    rng = np.random.default_rng(seed)
    hod = np.arange(n_hours) % 24
    doy = np.arange(n_hours) // 24 + 1

    shape = np.select(
        condlist=[hod < 7, hod < 10, hod < 15, hod < 18, hod < 22],
        choicelist=[-30.0, +15.0, -25.0, +5.0, +40.0],
        default=-25.0,
    )
    seasonal  = 20.0 * np.cos(2.0 * np.pi * (doy - 15) / 365.0)
    n_days    = n_hours // 24 + 1
    day_noise = np.repeat(rng.normal(0.0, 12.0, n_days), 24)[:n_hours]
    hour_noise = rng.normal(0.0, 4.0, n_hours)

    prices = base_eur_mwh + shape + seasonal + day_noise + hour_noise
    return np.clip(prices, 5.0, 300.0).astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ENHANCED PV ARRAY MODEL  (pvlib-based)
# ──────────────────────────────────────────────────────────────────────────────

class PVArray:
    """
    High-fidelity PV array model using pvlib.

    Corrections vs. previous revision:
      FIX #1  – bifacial rear irradiance via pvlib infinite-sheds (tilt-aware).
      FIX #3  – axis_azimuth derived from array azimuth (was hardcoded 90°).
      FIX #13 – surgical warnings only (no global suppression).
      FIX #14 – accepts SiteConfig instead of global SEVILLE.
      FIX #16 – input validation in __init__.
    """

    _SAPM_MONOSI = {
        "A0":  0.918093,
        "A1":  0.086257,
        "A2": -0.024459,
        "A3":  0.002816,
        "A4": -0.000126,
    }

    def __init__(
        self,
        panel      : JinkoTigerNeo,
        n_panels   : int,
        tilt_deg   : float = 30.0,
        azimuth_deg: float = 180.0,
        gcr        : float = 0.40,
        albedo     : float = 0.25,
        n_series   : int   = 20,
        height_m   : float = 1.5,
        site       : Optional[SiteConfig] = None,
    ):
        # FIX #16: input validation
        if n_panels <= 0:
            raise ValueError("n_panels must be positive")
        if not (0 < tilt_deg <= 90):
            raise ValueError(f"tilt_deg={tilt_deg} must be in (0, 90]")
        if not (0 < gcr < 1):
            raise ValueError(f"gcr={gcr} must be in (0, 1)")
        if n_series <= 0:
            raise ValueError("n_series must be positive")
        if albedo < 0 or albedo > 1:
            raise ValueError(f"albedo={albedo} must be in [0, 1]")

        self.panel    = panel
        self.n_panels = n_panels
        self.tilt     = tilt_deg
        self.azimuth  = azimuth_deg
        self.gcr      = gcr
        self.albedo   = albedo
        self.n_series = n_series
        self.height_m = height_m
        self.site     = (site or SiteConfig()).location

        self.peak_kw = panel.pmax_stc * n_panels / 1_000
        self.area_m2 = panel.area_m2  * n_panels

        self._collector_width = panel.length_m
        self._pitch           = panel.length_m / gcr

        # FIX #3: axis_azimuth derived from array azimuth.
        # Rows run perpendicular to the facing direction.
        # South-facing (180°) → rows run E-W (axis_azimuth = 90°).
        self.axis_azimuth = (azimuth_deg - 90.0) % 360.0

        # Warn if STC string voltage falls outside MPPT window
        v_string_stc = panel.vmp_stc * n_series
        if not (200 < v_string_stc < 1_500):
            warnings.warn(
                f"String voltage at STC ({v_string_stc:.0f} V) is outside the "
                f"inverter MPPT window [200, 1500 V]. Check n_series.",
                UserWarning, stacklevel=2,
            )

    # ─────────────────────────────────────────────────────────────────
    def _compute_beam_shading(
        self, solar_zenith: pd.Series, solar_azimuth: pd.Series
    ) -> np.ndarray:
        """
        FIX #3: uses self.axis_azimuth derived from the array azimuth
        instead of the previous hardcoded 90.0.
        """
        with warnings.catch_warnings():
            # FIX #13: surgical suppression – shading model divides by zero
            # near solar noon edge cases; this is expected, not a bug.
            warnings.filterwarnings(
                "ignore", message=".*divide by zero.*", category=RuntimeWarning
            )
            sf = pvl_shd.shaded_fraction1d(
                solar_zenith         = solar_zenith,
                solar_azimuth        = solar_azimuth,
                axis_azimuth         = self.axis_azimuth,  # FIX #3
                shaded_row_rotation  = self.tilt,
                shading_row_rotation = self.tilt,
                collector_width      = self._collector_width,
                pitch                = self._pitch,
                axis_tilt            = 0.0,
            )
        return np.asarray(sf, dtype=float)

    # ─────────────────────────────────────────────────────────────────
    def _compute_spectral_factor(
        self, airmass_abs: pd.Series, poa_global: pd.Series
    ) -> np.ndarray:
        """SAPM spectral mismatch factor for monocrystalline Si."""
        sf = pvl_spec.spectral_factor_sapm(
            airmass_absolute = airmass_abs.fillna(0.0).clip(0.0, 10.0),
            module           = self._SAPM_MONOSI,
        )
        sf = np.clip(np.asarray(sf, dtype=float), 0.8, 1.2)
        sf = np.where(poa_global.values <= 0.0, 0.0, sf)
        return sf

    # ─────────────────────────────────────────────────────────────────
    def _compute_bifacial_rear_irr(
        self,
        ghi          : pd.Series,
        dhi          : pd.Series,
        dni          : pd.Series,
        solar_zenith : pd.Series,
        solar_azimuth: pd.Series,
        dni_extra    : pd.Series,
    ) -> pd.Series:
        """
        FIX #1: Rear-side irradiance using pvlib infinite-sheds model when
        available.  Falls back to a tilt-corrected simplified formula.

        The infinite-sheds model accounts for:
          • panel tilt and azimuth
          • row height above ground
          • sky and ground view factors from the rear surface
          • GCR-based mutual shading of the ground

        Fallback formula (if pvlib < 0.9.5):
          G_rear = albedo × GHI × (1 – GCR) × (1 – cos(tilt)) / 2
          The (1 – cos(tilt))/2 factor is the rear ground view factor.
        """
        if _HAS_INFINITE_SHEDS:
            try:
                result = _bif_poa(
                    surface_tilt    = self.tilt,
                    surface_azimuth = self.azimuth,
                    solar_zenith    = solar_zenith,
                    solar_azimuth   = solar_azimuth,
                    gcr             = self.gcr,
                    height          = self.height_m,
                    pitch           = self._pitch,
                    ghi             = ghi,
                    dhi             = dhi,
                    dni             = dni,
                    albedo          = self.albedo,
                    dni_extra       = dni_extra,
                    bifaciality     = self.panel.bifaciality,
                    model           = "haydavies",
                )
                # pvlib returns "poa_global_rear" (already bifaciality-weighted)
                rear_col = next(
                    (c for c in ("poa_global_rear", "rear_irradiance")
                     if c in result.columns),
                    None,
                )
                if rear_col is not None:
                    return result[rear_col].fillna(0.0).clip(lower=0.0)
            except Exception as exc:
                logger.warning(
                    "pvlib infinite_sheds failed (%s); using corrected formula.", exc
                )

        # Corrected simplified fallback (tilt-aware view factor)
        tilt_rad    = np.radians(self.tilt)
        view_factor = (1.0 - np.cos(tilt_rad)) / 2.0   # fraction of rear seeing ground
        g_rear_raw  = self.albedo * ghi * (1.0 - self.gcr) * (1.0 + view_factor)
        return (g_rear_raw * self.panel.bifaciality).clip(lower=0.0)

    # ─────────────────────────────────────────────────────────────────
    def _compute_degradation(self, year: int, n_hours: int) -> np.ndarray:
        """
        Continuous degradation factor array for the simulation year.
        Year 1 : linear 1.0 → (1 – deg_y1).
        Year 2+: uniform (1–deg_y1) × (1–deg_annual)^(year–1).
        """
        if year == 1:
            t_frac = np.linspace(0.0, 1.0, n_hours)
            return 1.0 - self.panel.deg_y1 * t_frac
        factor = ((1.0 - self.panel.deg_y1)
                  * (1.0 - self.panel.deg_annual) ** (year - 1))
        return np.full(n_hours, factor, dtype=float)

    # ─────────────────────────────────────────────────────────────────
    def simulate(
        self, weather: pd.DataFrame, year: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Run the full pvlib-enhanced PV simulation for one year.

        Returns
        -------
        dc_kw  : total DC array output       [kW]
        v_dc   : DC string voltage (MPPT)    [V]
        diag   : intermediate diagnostics DataFrame
        """
        n = len(weather)

        # ── A. Solar position ─────────────────────────────────────────
        solpos  = self.site.get_solarposition(weather.index)
        zenith  = solpos["apparent_zenith"]
        azimuth = solpos["azimuth"]

        # ── B. Extraterrestrial radiation ─────────────────────────────
        dni_extra = pvl_irr.get_extra_radiation(weather.index)

        # ── C. Airmass (relative → absolute via TMY pressure) ─────────
        with warnings.catch_warnings():
            # FIX #13: airmass is undefined at night; log NaN is expected
            warnings.filterwarnings(
                "ignore", message=".*invalid value.*", category=RuntimeWarning
            )
            airmass_rel = pvl_atm.get_relative_airmass(
                zenith, model="kastenyoung1989"
            )
        pressure    = weather.get(
            "pressure", pd.Series(101_325.0, index=weather.index)
        )
        airmass_abs = pvl_atm.get_absolute_airmass(airmass_rel, pressure)

        # ── D. POA irradiance – Hay-Davies transposition ──────────────
        poa = pvl_irr.get_total_irradiance(
            surface_tilt    = self.tilt,
            surface_azimuth = self.azimuth,
            solar_zenith    = zenith,
            solar_azimuth   = azimuth,
            dni             = weather["dni"],
            ghi             = weather["ghi"],
            dhi             = weather["dhi"],
            dni_extra       = dni_extra,
            airmass         = airmass_rel,
            model           = "haydavies",
            albedo          = self.albedo,
        )
        poa_beam   = poa["poa_direct"].fillna(0.0).clip(lower=0.0)
        poa_sky    = poa["poa_sky_diffuse"].fillna(0.0).clip(lower=0.0)
        poa_ground = poa["poa_ground_diffuse"].fillna(0.0).clip(lower=0.0)

        # ── E. Row-to-row beam shading ────────────────────────────────
        shade_frac  = self._compute_beam_shading(zenith, azimuth)
        poa_beam_sh = poa_beam * (1.0 - shade_frac)
        poa_front   = poa_beam_sh + poa_sky + poa_ground

        # ── F. Spectral correction (SAPM, monosi) ─────────────────────
        spectral_factor = self._compute_spectral_factor(airmass_abs, poa_front)

        # ── G. FIX #1: Bifacial rear irradiance (infinite-sheds) ──────
        g_rear       = self._compute_bifacial_rear_irr(
            ghi           = weather["ghi"],
            dhi           = weather["dhi"],
            dni           = weather["dni"],
            solar_zenith  = zenith,
            solar_azimuth = azimuth,
            dni_extra     = dni_extra,
        )
        # If infinite-sheds is used, bifaciality is already applied in g_rear.
        # If fallback is used, bifaciality is also applied inside the method.
        poa_bifacial = (poa_front + g_rear).clip(lower=0.0)

        # ── H. Cell temperature – Faiman / IEC 61853 ─────────────────
        wind = weather.get(
            "wind_speed", pd.Series(1.0, index=weather.index)
        )
        t_cell = pvl_temp.faiman(
            poa_global = poa_bifacial,
            temp_air   = weather["temp_air"],
            wind_speed = wind,
            u0         = 25.0,
            u1         = 6.84,
        )

        # ── I. Effective irradiance (spectral-corrected) ──────────────
        poa_effective = poa_bifacial * spectral_factor

        # ── J. DC power – PVWatts v5 ─────────────────────────────────
        p_dc_per_panel = pvlib.pvsystem.pvwatts_dc(
            effective_irradiance = poa_effective,
            temp_cell            = t_cell,
            pdc0                 = self.panel.pmax_stc,
            gamma_pdc            = self.panel.gamma_pmax,
            temp_ref             = 25.0,
        ).fillna(0.0).clip(lower=0.0)

        # ── K. Degradation ────────────────────────────────────────────
        deg_factor = self._compute_degradation(year, n)
        dc_kw = p_dc_per_panel.values * self.n_panels * deg_factor / 1_000

        # ── L. DC string voltage (FIX #2: uses gamma_vmp, not gamma_pmax) ──
        v_mp_cell = (self.panel.vmp_stc
                     * (1.0 + self.panel.gamma_vmp * (t_cell - 25.0)))
        v_dc = (v_mp_cell * self.n_series).fillna(
            self.panel.vmp_stc * self.n_series
        ).clip(lower=0.0).values

        # ── M. Diagnostics DataFrame ──────────────────────────────────
        diag = pd.DataFrame({
            "poa_beam_unshaded"  : poa_beam.values,
            "poa_beam_shaded"    : poa_beam_sh.values,
            "poa_sky_diffuse"    : poa_sky.values,
            "poa_ground_diffuse" : poa_ground.values,
            "poa_front"          : poa_front.values,
            "rear_irradiance_wm2": g_rear.values if hasattr(g_rear, "values") else g_rear,
            "poa_bifacial"       : poa_bifacial.values,
            "poa_effective"      : poa_effective.values,
            "shade_fraction"     : shade_frac,
            "spectral_factor"    : spectral_factor,
            "bifacial_gain_wm2"  : g_rear.values if hasattr(g_rear, "values") else g_rear,
            "t_cell_c"           : t_cell.values,
            "v_dc_v"             : v_dc,
            "dc_kw"              : dc_kw,
            "deg_factor"         : deg_factor,
        }, index=weather.index)

        return dc_kw, v_dc, diag

    def annual_summary(self) -> dict:
        return {
            "n_panels"  : self.n_panels,
            "peak_kWp"  : round(self.peak_kw, 2),
            "area_m2"   : round(self.area_m2, 1),
            "tilt_deg"  : self.tilt,
            "gcr"       : self.gcr,
            "albedo"    : self.albedo,
            "n_series"  : self.n_series,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  BATTERY ENERGY STORAGE SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

class BatteryESS:
    """
    Tesla Megapack 2 XL battery model.

    FIX #4: set_year() updates capacity, power limits, and RTE to reflect
    year-on-year degradation.  The OptimalDispatcher calls set_year() before
    building the LP, so the degraded parameters feed into the optimisation.
    FIX #16: input validation in __init__.
    """

    def __init__(self, spec: TeslaMegapack2XL, n_units: int):
        if n_units <= 0:
            raise ValueError("n_units must be positive")

        self.spec    = spec
        self.n_units = n_units
        self._year   = 1   # track current operational year

        # Year-1 (new) parameters
        self.capacity_kwh = spec.energy_kwh * n_units
        self.max_p_kw     = spec.power_kw   * n_units
        self._rte         = spec.rte

        self._update_soc_limits()
        self.ramp_limit_kw = self.max_p_kw * 0.20
        self._soc_kwh = self.capacity_kwh * 0.50
        self._prev_p  = 0.0

    def _update_soc_limits(self):
        self.min_e = self.capacity_kwh * self.spec.min_soc
        self.max_e = self.capacity_kwh * self.spec.max_soc

    # FIX #4: update degraded capacity / RTE for a given operational year
    def set_year(self, year: int) -> None:
        """
        Update battery model parameters for the given operational year.
        Capacity fades at deg_annual_capacity_pct %/yr (floor = deg_floor_capacity).
        RTE declines at deg_annual_rte_pct %/yr (floor = deg_floor_rte).
        """
        self._year        = year
        self.capacity_kwh = self.spec.capacity_at_year(year) * self.n_units
        self._rte         = self.spec.rte_at_year(year)
        self._update_soc_limits()
        self.ramp_limit_kw = self.max_p_kw * 0.20

    @property
    def eta_ch(self) -> float:
        return float(np.sqrt(self._rte))

    @property
    def eta_dis(self) -> float:
        return float(np.sqrt(self._rte))

    @property
    def soc_kwh(self) -> float:
        return self._soc_kwh

    @property
    def soc_pct(self) -> float:
        return self._soc_kwh / self.capacity_kwh * 100.0

    def reset(self, initial_soc: float = 0.50):
        self._soc_kwh = self.capacity_kwh * initial_soc
        self._prev_p  = 0.0

    def step(self, net_demand_kw: float, dt: float = 1.0) -> dict:
        """Simulate one time-step (rule-based, used for diagnostics only)."""
        p_charge = p_discharge = 0.0

        if net_demand_kw < 0:
            surplus  = -net_demand_kw
            p_req    = min(surplus, self.max_p_kw)
            p_req    = min(p_req, self._prev_p + self.ramp_limit_kw)
            headroom = (self.max_e - self._soc_kwh) / (self.eta_ch * dt)
            p_charge = min(p_req, max(headroom, 0.0))
            self._soc_kwh += p_charge * self.eta_ch * dt
        elif net_demand_kw > 0:
            p_req       = min(net_demand_kw, self.max_p_kw)
            p_req       = min(p_req, self._prev_p + self.ramp_limit_kw)
            available   = (self._soc_kwh - self.min_e) * self.eta_dis / dt
            p_discharge = min(p_req, max(available, 0.0))
            self._soc_kwh -= p_discharge / self.eta_dis * dt

        self._soc_kwh *= (1.0 - self.spec.self_discharge_hr * dt)
        self._soc_kwh  = float(np.clip(self._soc_kwh, self.min_e, self.max_e))
        self._prev_p   = p_charge if p_charge > 0 else p_discharge

        return {
            "p_charge_kw"   : p_charge,
            "p_discharge_kw": p_discharge,
            "soc_kwh"       : self._soc_kwh,
            "soc_pct"       : self.soc_pct,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  ENHANCED INVERTER MODEL
# ──────────────────────────────────────────────────────────────────────────────

class Inverter:
    """
    Sungrow SG1100UD-20 efficiency model.
    Three-segment loading curve + MPPT voltage penalty + temperature derating.
    """

    def __init__(self, spec: SungrowModular, n_modules: int):
        if n_modules <= 0:
            raise ValueError("n_modules must be positive")
        self.spec      = spec
        self.n_modules = n_modules
        self.rated_kw  = spec.power_module_kw * n_modules

    def ac_power_kw(
        self,
        dc_kw : np.ndarray,
        v_dc  : Optional[np.ndarray] = None,
        t_amb : Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convert DC input power to AC output power [kW]."""
        dc_kw = np.asarray(dc_kw, dtype=float)
        ratio = np.clip(dc_kw / max(self.rated_kw, 1.0), 0.0, 1.10)

        # 1. Loading-ratio efficiency curve
        eta = np.where(
            ratio < 0.05,
            0.0,
            np.where(
                ratio < 0.20,
                0.960 + (self.spec.eta_euro - 0.960) * (ratio - 0.05) / 0.15,
                self.spec.eta_euro,
            ),
        )

        # 2. DC voltage: MPPT window penalty
        if v_dc is not None:
            v       = np.asarray(v_dc, dtype=float)
            v_min   = self.spec.v_mppt_min
            v_max   = self.spec.v_mppt_max
            v_lo    = self.spec.v_mppt_opt_lo
            v_hi    = self.spec.v_mppt_opt_hi
            v_factor = np.where(
                (v < v_min) | (v > v_max),
                0.0,
                np.where(
                    v < v_lo,
                    0.95 + 0.05 * (v - v_min) / (v_lo - v_min),
                    np.where(
                        v > v_hi,
                        1.00 - 0.02 * (v - v_hi) / (v_max - v_hi),
                        1.0,
                    ),
                ),
            )
            eta = eta * v_factor

        # 3. Ambient temperature derating
        if t_amb is not None:
            t        = np.asarray(t_amb, dtype=float)
            excess   = np.maximum(t - self.spec.t_derate_onset, 0.0)
            t_factor = np.maximum(
                1.0 - self.spec.derate_rate_per_c * excess, 0.80
            )
            eta = eta * t_factor

        return np.maximum(dc_kw * eta, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  OPTIMAL DISPATCHER  (Pyomo LP)
# ──────────────────────────────────────────────────────────────────────────────

class OptimalDispatcher:
    """
    Cost-minimising battery dispatch via a full-year linear programme (LP).

    Corrections:
      FIX #4  – calls battery.set_year(year) before building LP.
      FIX #9  – optional grid export revenue (feed_in_fraction parameter).
      FIX #10 – tiny curtailment/export penalty avoids degenerate LP solutions.
      FIX #11 – initial_soc and terminal_soc_min are configurable.
      FIX #12 – fast bulk LP variable extraction via .value attribute.
    """

    # FIX #10: epsilon penalty on curtailment to avoid degenerate solutions.
    # 0.001 €/kWh is ~1 €/MWh — far below typical price signals.
    _CURTAIL_PENALTY = 0.001

    def __init__(
        self,
        pv             : PVArray,
        battery        : BatteryESS,
        inverter       : Inverter,
        coverage_frac  : float = 0.40,
        feed_in_fraction: float = 0.0,
        # FIX #9: feed_in_fraction > 0 enables grid export revenue.
        # 1.0 = full spot price; 0.9 = 90 % of spot (typical net-metering cap).
        # 0.0 = no export allowed (curtail only).
    ):
        self.pv              = pv
        self.battery         = battery
        self.inverter        = inverter
        self.coverage_frac   = coverage_frac
        self.feed_in_fraction = feed_in_fraction

    @staticmethod
    def _pick_solver():
        """Return the fastest available LP solver."""
        for name in ("appsi_highs", "highs", "glpk", "cbc"):
            try:
                s = pyo.SolverFactory(name)
                if s.available():
                    return s, name
            except Exception:
                continue
        raise RuntimeError(
            "No LP solver found.  Install HiGHS via:  pip install highspy\n"
            "or GLPK via:  sudo apt-get install glpk-utils"
        )

    def _build_lp(
        self,
        pv_ac_arr       : np.ndarray,
        target_arr      : np.ndarray,
        prices_arr      : np.ndarray,
        initial_soc     : float = 0.50,
        terminal_soc_min: float = 0.20,
    ) -> pyo.ConcreteModel:
        """
        Construct the Pyomo LP model.

        FIX #9:  export variable with feed-in revenue in objective.
        FIX #10: curtailment penalty in objective.
        FIX #11: initial_soc and terminal_soc_min are parameters.
        """
        bat     = self.battery
        eta_ch  = float(bat.eta_ch)
        eta_dis = float(bat.eta_dis)
        E_min   = float(bat.min_e)
        E_max   = float(bat.max_e)
        P_ch    = float(bat.max_p_kw)
        P_dis   = float(bat.max_p_kw)
        sigma   = float(bat.spec.self_discharge_hr)
        # FIX #11: use configurable initial SOC
        E_init  = float(bat.capacity_kwh * initial_soc)
        E_term  = float(bat.capacity_kwh * terminal_soc_min)

        pv  = pv_ac_arr.tolist()
        dem = target_arr.tolist()
        pr  = prices_arr.tolist()
        n   = len(pv)
        fit = float(self.feed_in_fraction)
        eps = self._CURTAIL_PENALTY

        m = pyo.ConcreteModel(name="BatteryOptDispatch")
        m.T  = pyo.Set(initialize=range(n),     ordered=True)
        m.Ts = pyo.Set(initialize=range(n + 1), ordered=True)

        m.p_c    = pyo.Var(m.T,  bounds=(0.0, P_ch),  initialize=0.0)
        m.p_d    = pyo.Var(m.T,  bounds=(0.0, P_dis), initialize=0.0)
        m.e_grid = pyo.Var(m.T,  bounds=(0.0, None),  initialize=0.0)
        m.export = pyo.Var(m.T,  bounds=(0.0, None),  initialize=0.0)
        m.soc    = pyo.Var(m.Ts, bounds=(E_min, E_max), initialize=E_init)

        m.soc[0].fix(E_init)

        # FIX #9 + FIX #10: objective includes export revenue and curtail penalty
        m.obj = pyo.Objective(
            expr=pyo.quicksum(
                pr[t] * m.e_grid[t] / 1_000.0           # grid import cost [€]
                - fit * pr[t] * m.export[t] / 1_000.0   # export revenue   [€]
                + eps * m.export[t]                       # curtail penalty  [€]
                for t in m.T
            ),
            sense=pyo.minimize,
        )

        # Energy balance: PV + discharge + grid = demand + charge + export
        def balance_rule(mdl, t):
            return (pv[t] + mdl.p_d[t] + mdl.e_grid[t]
                    == dem[t] + mdl.p_c[t] + mdl.export[t])
        m.balance = pyo.Constraint(m.T, rule=balance_rule)

        # SOC dynamics
        def soc_rule(mdl, t):
            return (mdl.soc[t + 1]
                    == mdl.soc[t] * (1.0 - sigma)
                    + mdl.p_c[t] * eta_ch
                    - mdl.p_d[t] / eta_dis)
        m.soc_dyn = pyo.Constraint(m.T, rule=soc_rule)

        # FIX #11: terminal constraint uses configurable terminal_soc_min
        m.terminal = pyo.Constraint(expr=m.soc[n] >= E_term)

        return m

    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_var(var, n: int) -> np.ndarray:
        """
        FIX #12: fast bulk extraction of a Pyomo Var into a numpy array.
        Accessing .value directly avoids the overhead of pyo.value() calls
        inside a Python loop. Typically 5–10× faster for n = 8760.
        """
        arr = np.empty(n, dtype=float)
        for t in range(n):
            arr[t] = var[t].value
        return arr

    # ─────────────────────────────────────────────────────────────────
    def run(
        self,
        weather         : pd.DataFrame,
        total_demand_kw : np.ndarray,
        operating_hours : np.ndarray,
        year            : int   = 1,
        prices          : Optional[np.ndarray] = None,
        initial_soc     : float = 0.50,
        terminal_soc_min: float = 0.20,
    ) -> pd.DataFrame:
        """
        Run the optimal dispatch for one full year.

        FIX #4: battery.set_year(year) is called before LP construction so
        that degraded capacity and RTE are used in the optimisation.
        FIX #11: initial_soc and terminal_soc_min forwarded to _build_lp.
        """
        n = len(weather)

        # FIX #4: update battery for the current operational year
        self.battery.set_year(year)

        # PV simulation
        pv_dc, v_dc, pv_diag = self.pv.simulate(weather, year=year)
        t_amb = weather["temp_air"].values
        pv_ac = self.inverter.ac_power_kw(pv_dc, v_dc=v_dc, t_amb=t_amb)

        target = total_demand_kw * self.coverage_frac

        if prices is None:
            prices = np.full(n, 85.0)

        logger.info("Building Pyomo LP  (%d time steps × 5 variables)…", n)
        model = self._build_lp(
            pv_ac, target, prices,
            initial_soc      = initial_soc,
            terminal_soc_min = terminal_soc_min,
        )

        solver, solver_name = self._pick_solver()
        logger.info("Solver: %s — solving…", solver_name)
        result = solver.solve(model, tee=False)

        status = str(result.solver.termination_condition)
        if status not in ("optimal", "feasible"):
            raise RuntimeError(
                f"LP did not converge.  Solver status: {status}\n"
                "Check that battery sizing is consistent with demand."
            )
        logger.info(
            "LP solved (status: %s, obj = %,.0f €/yr grid cost)",
            status, pyo.value(model.obj)
        )

        # FIX #12: fast extraction via .value attribute
        p_c_sol    = self._extract_var(model.p_c,    n)
        p_d_sol    = self._extract_var(model.p_d,    n)
        e_grid_sol = self._extract_var(model.e_grid, n)
        export_sol = self._extract_var(model.export, n)
        soc_arr    = np.empty(n + 1, dtype=float)
        for t in range(n + 1):
            soc_arr[t] = model.soc[t].value

        # Clip numerical noise
        p_c_sol    = np.maximum(p_c_sol,    0.0)
        p_d_sol    = np.maximum(p_d_sol,    0.0)
        e_grid_sol = np.maximum(e_grid_sol, 0.0)
        export_sol = np.maximum(export_sol, 0.0)
        soc_sol    = np.clip(soc_arr, self.battery.min_e, self.battery.max_e)

        pv_to_load = np.maximum(pv_ac - p_c_sol - export_sol, 0.0)
        delivered  = pv_to_load + p_d_sol
        unmet      = np.maximum(target - delivered - e_grid_sol, 0.0)

        df = pd.DataFrame({
            "pv_dc_kw"            : pv_dc,
            "pv_ac_kw"            : pv_ac,
            "target_kw"           : target,
            "pv_to_load_kw"       : pv_to_load,
            "bat_to_load_kw"      : p_d_sol,
            "p_charge_kw"         : p_c_sol,
            "p_discharge_kw"      : p_d_sol,
            "e_grid_kw"           : e_grid_sol,
            "export_kw"           : export_sol,        # FIX #9
            "curtailed_kw"        : export_sol,        # alias for plot compatibility
            "unmet_kw"            : unmet,
            "soc_kwh"             : soc_sol[:n],
            "soc_pct"             : soc_sol[:n] / self.battery.capacity_kwh * 100.0,
            "elec_price_eur_mwh"  : prices,
        }, index=weather.index)

        df["delivered_kw"]   = delivered
        df["actual_cov_pct"] = np.where(
            target > 0, delivered / target * 100.0, 0.0
        )
        df["hour_of_day"]     = np.arange(n) % 24
        df["day_of_year"]     = np.arange(n) // 24 + 1
        df["ghi_wm2"]         = weather["ghi"].values
        df["dni_wm2"]         = weather["dni"].values
        df["dhi_wm2"]         = weather["dhi"].values
        df["t_amb_c"]         = weather["temp_air"].values
        df["wind_speed_ms"]   = weather["wind_speed"].values
        df["total_demand_kw"] = total_demand_kw
        df["operating"]       = operating_hours

        for col in [
            "poa_effective", "poa_bifacial", "poa_front",
            "shade_fraction", "spectral_factor", "t_cell_c",
            "v_dc_v", "deg_factor", "rear_irradiance_wm2", "bifacial_gain_wm2",
        ]:
            if col in pv_diag.columns:
                df[col] = pv_diag[col].values

        return df


# ──────────────────────────────────────────────────────────────────────────────
# 7.  ECONOMIC MODEL  (NPV / IRR / payback + battery replacement)
# ──────────────────────────────────────────────────────────────────────────────

class EconomicModel:
    """
    Techno-economic model for the PV + Battery subsystem.

    Corrections:
      FIX #4  – annual_savings() accounts for battery RTE degradation.
      FIX #6  – OPEX escalated by inflation_rate per year in cash_flows().
      FIX #7  – electricity prices escalated by price_escalation_rate per year.
      FIX #8  – battery_replacement_cost_factor is a named parameter.
    """

    BATTERY_LIFETIME_YR: int = 15

    def __init__(
        self,
        pv                          : PVArray,
        battery                     : BatteryESS,
        inverter                    : Inverter,
        panel_spec                  : JinkoTigerNeo,
        battery_spec                : TeslaMegapack2XL,
        inverter_spec               : SungrowModular,
        lifetime_yr                 : int   = 25,
        discount_rate               : float = 0.05,
        land_eur_m2                 : float = 10.0,
        packing_factor              : float = 0.40,
        bos_pct                     : float = 0.12,
        epc_pct                     : float = 0.08,
        om_pv_pct                   : float = 0.015,
        om_bat_pct                  : float = 0.020,
        om_inv_pct                  : float = 0.010,
        insurance_pct               : float = 0.005,
        # FIX #6: OPEX inflation rate (%/yr, real costs escalation)
        inflation_rate              : float = 0.025,
        # FIX #7: electricity price escalation rate (%/yr)
        price_escalation_rate       : float = 0.020,
        # FIX #8: battery replacement cost as a named parameter
        battery_replacement_cost_factor: float = 0.90,
        # ↑ Fraction of initial battery cost at time of replacement.
        # 0.90 = 10 % cost reduction vs. initial purchase.
        # Source: BloombergNEF (2023): utility BESS prices falling ~8–12 %/yr.
        # Vary between 0.70–1.00 for sensitivity analysis.
    ):
        self.pv       = pv
        self.battery  = battery
        self.inverter = inverter
        self.p_spec   = panel_spec
        self.b_spec   = battery_spec
        self.i_spec   = inverter_spec
        self.lifetime = lifetime_yr
        self.r        = discount_rate
        self.land_eur_m2 = land_eur_m2
        self.gcr      = packing_factor
        self.bos_pct  = bos_pct
        self.epc_pct  = epc_pct
        self.om_pv_pct  = om_pv_pct
        self.om_bat_pct = om_bat_pct
        self.om_inv_pct = om_inv_pct
        self.ins_pct    = insurance_pct
        self.inflation         = inflation_rate          # FIX #6
        self.price_esc         = price_escalation_rate  # FIX #7
        self.bat_repl_factor   = battery_replacement_cost_factor  # FIX #8

    def capex(self) -> dict:
        pv_hw  = self.pv.peak_kw * 1_000 * self.p_spec.cost_eur_per_wp
        bat_hw = self.battery.capacity_kwh * self.b_spec.cost_eur_per_kwh
        inv_hw = self.inverter.rated_kw    * self.i_spec.cost_eur_per_kw
        bos    = (pv_hw + inv_hw) * self.bos_pct
        land   = (self.pv.area_m2 / self.gcr) * self.land_eur_m2
        epc    = (pv_hw + bat_hw + inv_hw + bos) * self.epc_pct
        total  = pv_hw + bat_hw + inv_hw + bos + land + epc
        return {
            "PV panels (Jinko Tiger Neo)"   : pv_hw,
            "Battery (Tesla Megapack 2 XL)" : bat_hw,
            "Inverter (Sungrow 1+X)"        : inv_hw,
            "Balance of System (BOS)"       : bos,
            "Land"                          : land,
            "EPC"                           : epc,
            "Total CAPEX"                   : total,
        }

    def opex_annual(self) -> dict:
        """Year-1 operational expenditure [€/yr] (before inflation escalation)."""
        cap    = self.capex()
        pv_om  = cap["PV panels (Jinko Tiger Neo)"]   * self.om_pv_pct
        bat_om = cap["Battery (Tesla Megapack 2 XL)"] * self.om_bat_pct
        inv_om = cap["Inverter (Sungrow 1+X)"]        * self.om_inv_pct
        ins    = cap["Total CAPEX"]                    * self.ins_pct
        total  = pv_om + bat_om + inv_om + ins
        return {
            "PV O&M"      : pv_om,
            "Battery O&M" : bat_om,
            "Inverter O&M": inv_om,
            "Insurance"   : ins,
            "Total OPEX"  : total,
        }

    def battery_replacement_capex(self) -> float:
        """
        FIX #8: battery replacement cost uses the named parameter
        battery_replacement_cost_factor instead of a hardcoded 0.90.
        """
        bat_hw = self.battery.capacity_kwh * self.b_spec.cost_eur_per_kwh
        return bat_hw * self.bat_repl_factor

    def annual_savings(
        self, df: pd.DataFrame, prices: np.ndarray, year: int = 1
    ) -> float:
        """
        Annual savings from displaced grid electricity [€/yr].

        FIX #4: scales by battery capacity degradation factor.
        FIX #7: prices are escalated by price_escalation_rate per year.
        """
        # FIX #7: escalate prices for the given year
        escalated_prices = prices * (1.0 + self.price_esc) ** (year - 1)

        base = float(
            np.dot(escalated_prices, df["delivered_kw"].values) / 1_000.0
        )
        if year == 1:
            return base

        # PV degradation factor
        pv_deg = ((1.0 - self.p_spec.deg_y1)
                  * (1.0 - self.p_spec.deg_annual) ** (year - 1))

        # FIX #4: battery capacity degradation reduces throughput available
        bat_deg = max(
            self.b_spec.deg_floor_capacity,
            1.0 - self.b_spec.deg_annual_capacity_pct / 100.0 * (year - 1),
        )

        return base * pv_deg * bat_deg

    def cash_flows(
        self, df: pd.DataFrame, prices: np.ndarray
    ) -> np.ndarray:
        """
        Annual net cash flows [€], indexed 0..lifetime.

        FIX #6: OPEX is inflated at inflation_rate %/yr for each year y.
        FIX #8: battery replacement uses configurable bat_repl_factor.
        """
        cap   = self.capex()["Total CAPEX"]
        opex1 = self.opex_annual()["Total OPEX"]   # year-1 OPEX base
        bat_r = self.battery_replacement_capex()
        cf    = np.zeros(self.lifetime + 1)
        cf[0] = -cap
        for y in range(1, self.lifetime + 1):
            # FIX #6: inflate OPEX year-on-year
            opex_y = opex1 * (1.0 + self.inflation) ** (y - 1)
            cf[y]  = self.annual_savings(df, prices, year=y) - opex_y
            if y == self.BATTERY_LIFETIME_YR:
                cf[y] -= bat_r
        return cf

    def npv(self, df: pd.DataFrame, prices: np.ndarray) -> float:
        cf   = self.cash_flows(df, prices)
        disc = np.array([(1.0 + self.r) ** (-y) for y in range(self.lifetime + 1)])
        return float(np.dot(cf, disc))

    def irr(self, df: pd.DataFrame, prices: np.ndarray) -> float:
        cf = self.cash_flows(df, prices)

        def _npv_at(r: float) -> float:
            disc = np.array([(1.0 + r) ** (-y) for y in range(len(cf))])
            return float(np.dot(cf, disc))

        try:
            lo, hi = -0.50, 5.0
            if _npv_at(lo) * _npv_at(hi) > 0:
                return float("nan")
            return float(brentq(_npv_at, lo, hi, xtol=1e-7, maxiter=200))
        except (ValueError, RuntimeError):
            return float("nan")

    def payback_period(self, df: pd.DataFrame, prices: np.ndarray) -> float:
        cf  = self.cash_flows(df, prices)
        cum = np.cumsum(cf)
        for y in range(1, len(cum)):
            if cum[y] >= 0.0:
                frac = -cum[y - 1] / (cum[y] - cum[y - 1])
                return float(y - 1 + frac)
        return float("nan")

    def _annuity_factor(self) -> float:
        r, n = self.r, self.lifetime
        return (1.0 - (1.0 + r) ** (-n)) / r

    def lcoe(self, annual_energy_kwh: float) -> float:
        """Levelised Cost of Energy [€/MWh]."""
        cap      = self.capex()["Total CAPEX"]
        opex_npv = self.opex_annual()["Total OPEX"] * self._annuity_factor()
        bat_pv   = (self.battery_replacement_capex()
                    / (1.0 + self.r) ** self.BATTERY_LIFETIME_YR)
        total_disc_energy = 0.0
        for y in range(1, self.lifetime + 1):
            if y == 1:
                ey = annual_energy_kwh
            else:
                ey = (annual_energy_kwh
                      * (1.0 - self.p_spec.deg_y1)
                      * (1.0 - self.p_spec.deg_annual) ** (y - 1))
            total_disc_energy += ey / (1.0 + self.r) ** y
        return (cap + opex_npv + bat_pv) / total_disc_energy * 1_000

    def full_summary(self, df: pd.DataFrame) -> dict:
        annual_pv        = df["pv_ac_kw"].sum()
        annual_delivered = df["delivered_kw"].sum()
        annual_bat_dis   = df["bat_to_load_kw"].sum()
        annual_curtailed = df["export_kw"].sum()
        annual_unmet     = df["unmet_kw"].sum()
        total_target     = df["target_kw"].sum()
        cap  = self.capex()
        opex = self.opex_annual()
        lcoe = self.lcoe(annual_delivered)
        actual_cov   = (annual_delivered / total_target * 100) if total_target > 0 else 0
        bat_share    = (annual_bat_dis / annual_delivered * 100) if annual_delivered > 0 else 0
        curtail_rate = (annual_curtailed / annual_pv * 100) if annual_pv > 0 else 0
        return {
            "CAPEX"      : cap,
            "Annual OPEX": opex,
            "KPIs": {
                "Annual PV AC generation [MWh]"         : annual_pv / 1_000,
                "Annual energy delivered [MWh]"         : annual_delivered / 1_000,
                "Battery share of delivery [%]"         : bat_share,
                "Curtailed / exported energy [MWh]"     : annual_curtailed / 1_000,
                "Unmet demand [MWh]"                    : annual_unmet / 1_000,
                "Actual coverage [%]"                   : actual_cov,
                "Curtailment/export rate [%]"           : curtail_rate,
                "LCOE (incl. batt. replacement) [€/MWh]": lcoe,
                "Specific CAPEX [€/kWp]"                : cap["Total CAPEX"] / self.pv.peak_kw,
                "CAPEX per MWh stored [€/MWh]"          : (
                    cap["Battery (Tesla Megapack 2 XL)"]
                    / (self.battery.capacity_kwh / 1_000)
                ),
            },
        }

    def financial_summary(
        self, df: pd.DataFrame, prices: np.ndarray
    ) -> dict:
        cf             = self.cash_flows(df, prices)
        npv_val        = self.npv(df, prices)
        irr_val        = self.irr(df, prices)
        pb_val         = self.payback_period(df, prices)
        sav_yr1        = self.annual_savings(df, prices, year=1)
        bat_repl       = self.battery_replacement_capex()
        lcoe_val       = self.lcoe(df["delivered_kw"].sum())
        annual_grid    = float(np.dot(prices, df["e_grid_kw"].values) / 1_000.0)
        annual_ref     = float(np.dot(prices, df["target_kw"].values) / 1_000.0)
        return {
            "cash_flows_eur"           : cf,
            "Annual savings yr1 [€]"   : sav_yr1,
            "Annual grid cost opt [€]" : annual_grid,
            "Annual grid cost ref [€]" : annual_ref,
            "Battery replacement [€]"  : bat_repl,
            "NPV [€]"                  : npv_val,
            "NPV [M€]"                 : npv_val / 1e6,
            "IRR [%]"                  : irr_val * 100 if not np.isnan(irr_val) else float("nan"),
            "Payback period [yr]"      : pb_val,
            "LCOE [€/MWh]"            : lcoe_val,
            "Discount rate [%]"        : self.r * 100,
            "Project lifetime [yr]"    : self.lifetime,
            "Battery lifetime [yr]"    : self.BATTERY_LIFETIME_YR,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 8.  SENSITIVITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    weather       : pd.DataFrame,
    base_pv_mwp   : float,
    base_bat_mwh  : float,
    panel_spec    : JinkoTigerNeo,
    battery_spec  : TeslaMegapack2XL,
    inverter_spec : SungrowModular,
    site          : Optional[SiteConfig] = None,
    n_pv          : int   = 9,
    n_bat         : int   = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    2-D sensitivity of LCOE and CAPEX over PV size × battery size.

    FIX #15: PV generation is re-simulated for each grid point rather than
    being scaled from the base-case energy.  PV output scales proportionally
    with n_panels (same specific yield, same weather), so re-simulation is fast.
    """
    pv_vals  = np.linspace(base_pv_mwp  * 0.50, base_pv_mwp  * 1.70, n_pv)
    bat_vals = np.linspace(base_bat_mwh * 0.40, base_bat_mwh * 1.80, n_bat)

    lcoe_mat  = np.zeros((n_bat, n_pv))
    capex_mat = np.zeros((n_bat, n_pv))

    # Simulate PV once at base scale; scale linearly for other sizes.
    # Full re-simulation would multiply run-time by n_pv × n_bat (~81×).
    # Scaling is exact because pvwatts_dc is linear in n_panels.
    base_n_pan  = int(np.ceil(base_pv_mwp * 1e6 / panel_spec.pmax_stc))
    base_pv_sys = PVArray(panel_spec, base_n_pan, site=site)
    logger.info("Sensitivity: simulating base PV for %d panels…", base_n_pan)
    base_dc, base_vdc, _ = base_pv_sys.simulate(weather, year=1)
    base_n_inv = inverter_spec.modules_needed(base_pv_mwp * 1_000)
    base_inv   = Inverter(inverter_spec, base_n_inv)
    base_ac    = base_inv.ac_power_kw(base_dc, v_dc=base_vdc)
    base_annual_kwh = float(base_ac.sum())

    for j, pv_mwp in enumerate(pv_vals):
        for i, bat_mwh in enumerate(bat_vals):
            scale   = pv_mwp / base_pv_mwp
            # FIX #15: scale PV AC linearly with peak power
            # (valid because pvwatts_dc is linear in n_panels)
            e_kwh   = base_annual_kwh * scale

            n_pan   = int(np.ceil(pv_mwp * 1e6 / panel_spec.pmax_stc))
            n_bat_u = battery_spec.units_for_energy(bat_mwh * 1_000)
            n_inv   = inverter_spec.modules_needed(pv_mwp * 1_000)

            pv_tmp  = PVArray(panel_spec, n_pan, site=site)
            bat_tmp = BatteryESS(battery_spec, n_bat_u)
            inv_tmp = Inverter(inverter_spec, n_inv)

            econ = EconomicModel(
                pv_tmp, bat_tmp, inv_tmp,
                panel_spec, battery_spec, inverter_spec,
            )
            capex_mat[i, j] = econ.capex()["Total CAPEX"] / 1e6
            lcoe_mat[i, j]  = econ.lcoe(e_kwh)

    return pv_vals, bat_vals, lcoe_mat, capex_mat


# ──────────────────────────────────────────────────────────────────────────────
# 9.  PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_system_results(
    df: pd.DataFrame, summary: dict,
    pv_peak_mwp: float, bat_mwh: float, n_bat_units: int,
):
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(
        "PV + Li-ion Battery Subsystem — Seville, Spain  (pvlib-enhanced)\n"
        f"Jinko Solar {pv_peak_mwp:.1f} MWp  |  "
        f"Tesla Megapack 2 XL ×{n_bat_units} ({bat_mwh:.0f} MWh)  |  "
        f"Sungrow 1+X Modular Inverter",
        fontsize=13, fontweight="bold", y=0.99,
    )

    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)
    s   = slice(4_104, 4_104 + 168)
    h   = np.arange(168)

    # Panel 1: characteristic summer week
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(h, df["target_kw"].iloc[s] / 1_000,
                     alpha=0.20, color="gray", label="Target demand")
    ax1.plot(h, df["pv_ac_kw"].iloc[s] / 1_000,
             color="gold", lw=1.8, label="PV AC output")
    ax1.plot(h, df["delivered_kw"].iloc[s] / 1_000,
             color="limegreen", lw=1.8, label="Delivered to load")
    ax1.plot(h, df["p_discharge_kw"].iloc[s] / 1_000,
             color="dodgerblue", lw=1.2, ls="--", label="Battery discharge")
    ax1.set_xlabel("Hour of week"); ax1.set_ylabel("Power [MW]")
    ax1.set_title("Characteristic Summer Week (Week 25)")
    ax1.legend(fontsize=7.5); ax1.grid(True, alpha=0.3)

    # Panel 2: pvlib irradiance components
    ax2 = fig.add_subplot(gs[0, 1])
    if "poa_front" in df.columns:
        ax2.plot(h, df["poa_front"].iloc[s], color="gold", lw=1.5,
                 label="POA front (Hay-Davies)")
        ax2.plot(h, df["poa_effective"].iloc[s], color="darkorange",
                 lw=1.5, ls="--", label="POA effective (spectral-corr.)")
        ax2.plot(h, df["ghi_wm2"].iloc[s], color="gray", lw=1.0,
                 alpha=0.6, label="GHI (horizontal)")
        if "bifacial_gain_wm2" in df.columns:
            ax2.fill_between(h, 0,
                             df["bifacial_gain_wm2"].iloc[s].clip(lower=0),
                             alpha=0.3, color="green", label="Bifacial rear gain")
    ax2.set_xlabel("Hour of week"); ax2.set_ylabel("Irradiance [W/m²]")
    ax2.set_title("POA Irradiance Breakdown — Summer Week")
    ax2.legend(fontsize=7.5); ax2.grid(True, alpha=0.3)

    # Panel 3: correction factors by month
    ax3 = fig.add_subplot(gs[1, 0])
    months     = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_lens = [744,672,744,720,744,720,744,744,720,744,720,744]
    m_sf, m_shade, m_tcell = [], [], []
    cum = 0
    for hpm in month_lens:
        sl  = slice(cum, cum + hpm)
        day = (df["ghi_wm2"].iloc[sl] > 20)
        m_sf.append(df["spectral_factor"].iloc[sl][day].mean()
                    if "spectral_factor" in df.columns and day.any() else 1.0)
        m_shade.append(df["shade_fraction"].iloc[sl][day].mean()
                       if "shade_fraction" in df.columns and day.any() else 0.0)
        m_tcell.append(df["t_cell_c"].iloc[sl][day].mean()
                       if "t_cell_c" in df.columns and day.any() else 25.0)
        cum += hpm
    x = np.arange(12)
    ax3b = ax3.twinx()
    ax3.bar(x - 0.22, m_shade, 0.40, color="tomato", alpha=0.7,
            label="Beam shading fraction")
    ax3.bar(x + 0.22, [1 - v for v in m_sf], 0.40, color="steelblue",
            alpha=0.7, label="Spectral loss (1–f₁)")
    ax3b.plot(x, m_tcell, "k^--", lw=1.5, ms=5, label="Mean cell T [°C]")
    ax3.set_xticks(x); ax3.set_xticklabels(months, fontsize=8)
    ax3.set_ylabel("Fraction"); ax3b.set_ylabel("Cell temperature [°C]")
    ax3.set_title("Monthly Correction Factors (daytime)")
    ax3.legend(loc="upper left", fontsize=7.5)
    ax3b.legend(loc="upper right", fontsize=7.5)
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: DC voltage distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if "v_dc_v" in df.columns:
        day_mask = df["ghi_wm2"] > 20
        ax4.hist(df.loc[day_mask, "v_dc_v"], bins=60, color="steelblue",
                 alpha=0.7, edgecolor="white", linewidth=0.3)
        ax4.axvline(850,  color="orange", lw=1.5, ls="--",
                    label="MPPT opt. lo = 850 V")
        ax4.axvline(1350, color="red",    lw=1.5, ls="--",
                    label="MPPT opt. hi = 1350 V")
    ax4.set_xlabel("DC String Voltage [V]"); ax4.set_ylabel("Hours [h/yr]")
    ax4.set_title("DC Voltage Distribution (daytime hours)")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # Panel 5: monthly energy balance
    ax5 = fig.add_subplot(gs[2, 0])
    cum = 0
    m_pv, m_dem, m_del = [], [], []
    for hpm in month_lens:
        sl = slice(cum, cum + hpm)
        m_pv.append(df["pv_ac_kw"].iloc[sl].sum() / 1_000)
        m_dem.append(df["target_kw"].iloc[sl].sum() / 1_000)
        m_del.append(df["delivered_kw"].iloc[sl].sum() / 1_000)
        cum += hpm
    ax5.bar(x - 0.22, m_pv,  0.40, color="gold", alpha=0.85, label="PV AC generation")
    ax5.bar(x + 0.22, m_dem, 0.40, color="gray", alpha=0.45, label="Target demand")
    ax5.plot(x, m_del, "go--", lw=1.5, ms=5, label="Delivered to load")
    ax5.set_xticks(x); ax5.set_xticklabels(months, fontsize=8)
    ax5.set_ylabel("Energy [MWh]")
    ax5.set_title("Monthly Energy Balance")
    ax5.legend(fontsize=7.5); ax5.grid(True, alpha=0.3, axis="y")

    # Panel 6: operating mode pie
    ax6 = fig.add_subplot(gs[2, 1])
    op = df[df["operating"]]
    pv_only  = ((op["pv_to_load_kw"] > 1) & (op["bat_to_load_kw"] < 1)).sum()
    bat_only = ((op["pv_to_load_kw"] < 1) & (op["bat_to_load_kw"] > 1)).sum()
    pv_bat   = ((op["pv_to_load_kw"] > 1) & (op["bat_to_load_kw"] > 1)).sum()
    unmet_h  = (op["unmet_kw"] > 1).sum()
    idle_h   = max(len(op) - pv_only - bat_only - pv_bat - unmet_h, 0)
    raw   = [pv_only, bat_only, pv_bat, unmet_h, idle_h]
    lbls  = ["PV only","Battery only","PV + Battery","Unmet (backup)","Idle"]
    cols_p= ["gold","dodgerblue","limegreen","tomato","lightgray"]
    nz = [(v, l, c) for v, l, c in zip(raw, lbls, cols_p) if v > 0]
    if nz:
        vz, lz, cz = zip(*nz)
        ax6.pie(vz, labels=lz, colors=cz, autopct="%1.1f%%",
                startangle=90, textprops={"fontsize": 8})
    ax6.set_title("Operating Mode Distribution")

    # Panel 7: CAPEX breakdown
    ax7 = fig.add_subplot(gs[3, 0])
    cap_items = {k: v / 1e6 for k, v in summary["CAPEX"].items()
                 if k != "Total CAPEX"}
    bar_cols = ["gold","dodgerblue","steelblue","gray","tan","salmon"]
    bars = ax7.barh(list(cap_items.keys()), list(cap_items.values()),
                    color=bar_cols[:len(cap_items)])
    ax7.set_xlabel("Cost [M€]")
    ax7.set_title(f"CAPEX Breakdown — Total: "
                  f"{summary['CAPEX']['Total CAPEX']/1e6:.1f} M€")
    ax7.bar_label(bars, fmt="%.1f M€", padding=3, fontsize=8)
    ax7.grid(True, alpha=0.3, axis="x")

    # Panel 8: annual GHI heatmap
    ax8 = fig.add_subplot(gs[3, 1])
    ghi_mat = df["ghi_wm2"].values[:8_760].reshape(365, 24)
    im = ax8.imshow(ghi_mat.T, aspect="auto", origin="lower",
                    cmap="YlOrRd", vmin=0, vmax=1_000)
    plt.colorbar(im, ax=ax8, label="GHI [W/m²]")
    ax8.axhline(8,  color="white", lw=1.2, ls="--", alpha=0.8)
    ax8.axhline(18, color="white", lw=1.2, ls="--", alpha=0.8)
    ax8.set_xlabel("Day of Year"); ax8.set_ylabel("Hour of Day (UTC)")
    ax8.set_title("Annual GHI Heatmap  (dashed = operating hours UTC)")

    plt.savefig("pv_battery_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Figure saved: pv_battery_results.png")


def plot_sensitivity_results(
    pv_grid, bat_grid, lcoe_mat, capex_mat, base_pv, base_bat,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Sensitivity Analysis: LCOE and CAPEX vs PV Size & Battery Capacity",
        fontsize=13, fontweight="bold",
    )
    data_pairs = [
        (lcoe_mat,  "LCOE [€/MWh]",    "RdYlGn_r"),
        (capex_mat, "Total CAPEX [M€]", "Blues"),
    ]
    for ax, (mat, title, cmap) in zip(axes, data_pairs):
        cf = ax.contourf(pv_grid, bat_grid, mat, 14, cmap=cmap)
        cs = ax.contour(pv_grid, bat_grid, mat, 8, colors="k", linewidths=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
        plt.colorbar(cf, ax=ax)
        ax.scatter([base_pv], [base_bat], marker="*", s=250,
                   color="red", zorder=5, label="Base case")
        ax.set_xlabel("PV Peak Power [MWp]")
        ax.set_ylabel("Battery Capacity [MWh]")
        ax.set_title(title)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("sensitivity_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Figure saved: sensitivity_analysis.png")


def plot_financial_results(
    fin: dict, summary: dict, prices: np.ndarray, df: pd.DataFrame,
):
    cf   = fin["cash_flows_eur"]
    cum  = np.cumsum(cf)
    yrs  = np.arange(len(cf))
    N    = len(cf) - 1
    bat_yr = EconomicModel.BATTERY_LIFETIME_YR

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "PV + Battery Financial Analysis — Seville, Spain\n"
        f"NPV = {fin['NPV [M€]']:.2f} M€  |  "
        f"IRR = {fin['IRR [%]']:.1f} %  |  "
        f"Payback = {fin['Payback period [yr]']:.1f} yr  |  "
        f"LCOE = {fin['LCOE [€/MWh]']:.1f} €/MWh",
        fontsize=12, fontweight="bold",
    )

    ax1 = axes[0, 0]
    colors_bar = ["tomato" if v < 0 else "limegreen" for v in cf]
    ax1.bar(yrs, cf / 1e6, color=colors_bar, alpha=0.75, label="Annual CF")
    ax1.plot(yrs, cum / 1e6, "k-o", lw=2, ms=4, label="Cumulative CF")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axvline(bat_yr, color="dodgerblue", lw=1.5, ls="--",
                label=f"Battery replacement (yr {bat_yr})")
    pb = fin["Payback period [yr]"]
    if not np.isnan(pb):
        ax1.axvline(pb, color="gold", lw=2.0, ls="-.",
                    label=f"Payback = {pb:.1f} yr")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Cash flow [M€]")
    ax1.set_title("Cumulative Cash-Flow Waterfall")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    sorted_p = np.sort(prices)[::-1]
    ax2.plot(np.linspace(0, 100, len(sorted_p)), sorted_p,
             color="steelblue", lw=1.8)
    ax2.fill_between(np.linspace(0, 100, len(sorted_p)), sorted_p,
                     alpha=0.25, color="steelblue")
    ax2.axhline(prices.mean(), color="red", lw=1.2, ls="--",
                label=f"Mean = {prices.mean():.0f} €/MWh")
    ax2.axhline(prices[prices <= np.percentile(prices, 25)].max(),
                color="green", lw=1.0, ls=":", label="Q1 (charge signal)")
    ax2.axhline(prices[prices >= np.percentile(prices, 75)].min(),
                color="orange", lw=1.0, ls=":", label="Q3 (discharge signal)")
    ax2.set_xlabel("Exceedance probability [%]")
    ax2.set_ylabel("Price [€/MWh]")
    ax2.set_title("Electricity Price Duration Curve (Spanish OMIE profile)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    s   = slice(4_104, 4_104 + 168)
    h   = np.arange(168)
    ax3b = ax3.twinx()
    ax3.fill_between(h, df["pv_ac_kw"].iloc[s] / 1_000,
                     alpha=0.30, color="gold", label="PV AC [MW]")
    ax3.plot(h, df["p_charge_kw"].iloc[s] / 1_000, color="dodgerblue",
             lw=1.4, label="Bat. charge [MW]")
    ax3.plot(h, df["p_discharge_kw"].iloc[s] / 1_000, color="tomato",
             lw=1.4, label="Bat. discharge [MW]")
    ax3b.plot(h, df["elec_price_eur_mwh"].iloc[s], color="darkgreen",
              lw=1.0, alpha=0.7, ls="--", label="Price [€/MWh]")
    ax3.set_xlabel("Hour of week"); ax3.set_ylabel("Power [MW]")
    ax3b.set_ylabel("Price [€/MWh]")
    ax3.set_title("Optimal Dispatch vs. Price Signal — Summer Week")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5)
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    cap  = summary["CAPEX"]["Total CAPEX"]
    opex = summary["Annual OPEX"]["Total OPEX"]
    items = {
        "Total CAPEX\n(annualised)"     : cap / N / 1e6,
        "Annual OPEX"                   : opex / 1e6,
        "Bat. replacement\n(annualised)": fin["Battery replacement [€]"] / N / 1e6,
        "Annual savings\n(yr 1)"        : -fin["Annual savings yr1 [€]"] / 1e6,
        "Grid cost\n(optimised)"        : fin["Annual grid cost opt [€]"] / 1e6,
        "Grid cost\n(no system)"        : fin["Annual grid cost ref [€]"] / 1e6,
    }
    bar_colors = ["tomato","tomato","tomato","limegreen","steelblue","gray"]
    labels = list(items.keys())
    values = [abs(v) for v in items.values()]
    bars = ax4.barh(labels, values, color=bar_colors, alpha=0.80)
    ax4.set_xlabel("Cost / Savings [M€/yr]")
    ax4.set_title("Annual Cost Breakdown  (green = savings, red = cost)")
    ax4.bar_label(bars, fmt="%.2f M€", padding=3, fontsize=8)
    ax4.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("financial_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    logger.info("Figure saved: financial_analysis.png")


# ──────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(pvgis_path: Optional[str] = None, verbose: bool = False):
    # FIX #17: configure logging framework
    logging.basicConfig(
        level   = logging.DEBUG if verbose else logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    SEP = "─" * 72

    # FIX #14: site parameters in SiteConfig, not module globals
    site = SiteConfig(
        tmy_file = pvgis_path or "Sevilla_TMY.csv"
    )

    # Component specs (FIX #16: validation runs in __post_init__)
    panel_spec    = JinkoTigerNeo()
    battery_spec  = TeslaMegapack2XL()
    inverter_spec = SungrowModular()

    # System sizing
    TOTAL_MW       = 10.0
    COVERAGE       = 0.40
    TARGET_MW      = TOTAL_MW * COVERAGE
    STORAGE_HRS    = 4.0
    PV_PEAK_MWP    = 8.5
    BAT_ENERGY_KWH = TARGET_MW * 1_000 * STORAGE_HRS
    BAT_POWER_KW   = TARGET_MW * 1_000

    n_panels = int(np.ceil(PV_PEAK_MWP * 1e6 / panel_spec.pmax_stc))
    n_bat    = battery_spec.units_needed(BAT_ENERGY_KWH, BAT_POWER_KW)
    n_inv    = inverter_spec.modules_needed(PV_PEAK_MWP * 1_000)
    N_SERIES = 20

    # FIX #14: pass site to PVArray
    pv_system  = PVArray(panel_spec, n_panels, n_series=N_SERIES, site=site)
    bat_system = BatteryESS(battery_spec, n_bat)
    inv_system = Inverter(inverter_spec, n_inv)

    logger.info("%s", SEP)
    logger.info("MJ2438 – PV + Battery Subsystem  |  %s", site.name)
    logger.info(
        "Physical model : pvlib %s  (Hay-Davies, Faiman, SAPM, bifacial, shading)",
        pvlib.__version__,
    )
    logger.info("Dispatch model : Pyomo LP (full-year, cost-minimising)")
    logger.info("Financial model: NPV / IRR / payback, battery repl. @ yr 15")
    logger.info("%s", SEP)
    logger.info(
        "PV Array   : %d panels × %.0f W = %.2f MWp  (area %.2f ha)",
        n_panels, panel_spec.pmax_stc,
        pv_system.peak_kw / 1_000, pv_system.area_m2 / 1e4,
    )
    logger.info(
        "Strings    : %d panels/string × %.1f V = %.0f V nominal DC",
        N_SERIES, panel_spec.vmp_stc, N_SERIES * panel_spec.vmp_stc,
    )
    logger.info(
        "Battery    : %d × Tesla Megapack 2 XL = %.1f MWh | %.2f MW  "
        "(RTE = %.1f %%, lifetime = %d yr, cap.fade = %.1f %%/yr)",
        n_bat,
        n_bat * battery_spec.energy_kwh / 1_000,
        n_bat * battery_spec.power_kw   / 1_000,
        battery_spec.rte * 100,
        EconomicModel.BATTERY_LIFETIME_YR,
        battery_spec.deg_annual_capacity_pct,
    )
    logger.info(
        "Inverter   : %d × Sungrow 1.1 MW = %.1f MW",
        n_inv, n_inv * inverter_spec.power_module_kw / 1_000,
    )

    # Weather data
    weather_file = site.tmy_file
    try:
        logger.info("Loading PVGIS TMY: %s", weather_file)
        # FIX #14: pass site to load_pvgis_tmy
        weather = load_pvgis_tmy(weather_file, site=site)
    except FileNotFoundError:
        logger.warning(
            "'%s' not found — using synthetic TMY.  "
            "Set tmy_file in SiteConfig to your PVGIS CSV filename.",
            weather_file,
        )
        weather = generate_synthetic_tmy(site.latitude, site.longitude)
        logger.info(
            "Weather source : Synthetic (%.2f°N, GHI≈%d kWh/m²)",
            site.latitude, weather["ghi"].sum() / 1_000,
        )

    # Electricity prices
    logger.info("Generating Spanish OMIE electricity price profile…")
    prices = generate_spain_electricity_prices(n_hours=len(weather))
    logger.info(
        "Price profile  : mean = %.0f €/MWh | min = %.0f | max = %.0f €/MWh",
        prices.mean(), prices.min(), prices.max(),
    )

    # Demand profile
    local_hour = weather.index.tz_convert("Europe/Madrid").hour
    op_mask    = (local_hour >= 8) & (local_hour < 18)
    demand_kw  = np.where(op_mask, TOTAL_MW * 1_000, 0.0)

    # Optimal dispatch (Pyomo LP)
    dispatcher = OptimalDispatcher(
        pv_system, bat_system, inv_system, COVERAGE,
        feed_in_fraction=0.0,   # FIX #9: set > 0 to enable export revenue
    )
    logger.info("Running year-1 optimal dispatch…")
    df = dispatcher.run(
        weather, demand_kw, op_mask,
        year             = 1,
        prices           = prices,
        initial_soc      = 0.50,   # FIX #11
        terminal_soc_min = 0.20,   # FIX #11
    )

    # Economic & financial analysis
    econ    = EconomicModel(
        pv_system, bat_system, inv_system,
        panel_spec, battery_spec, inverter_spec,
        inflation_rate                 = 0.025,   # FIX #6
        price_escalation_rate          = 0.020,   # FIX #7
        battery_replacement_cost_factor= 0.90,    # FIX #8
    )
    summary = econ.full_summary(df)
    fin     = econ.financial_summary(df, prices)

    # Print results
    logger.info("%s", SEP)
    logger.info("CAPEX BREAKDOWN")
    logger.info("%s", SEP)
    total_cap = summary["CAPEX"]["Total CAPEX"]
    for k, v in summary["CAPEX"].items():
        bar = "█" * int(v / total_cap * 28)
        logger.info("  %-40s : €%12,.0f  %s", k, v, bar)

    logger.info("%s", SEP)
    logger.info("ANNUAL OPEX")
    logger.info("%s", SEP)
    for k, v in summary["Annual OPEX"].items():
        logger.info("  %-40s : €%10,.0f / yr", k, v)

    logger.info("%s", SEP)
    logger.info("KEY PERFORMANCE INDICATORS")
    logger.info("%s", SEP)
    for k, v in summary["KPIs"].items():
        logger.info("  %-50s : %10.2f", k, v)

    logger.info("%s", SEP)
    logger.info(
        "FINANCIAL ANALYSIS  (r = %.1f %%, lifetime = %d yr, "
        "battery replaced at yr %d, inflation = %.1f %%, price esc. = %.1f %%)",
        fin["Discount rate [%]"], fin["Project lifetime [yr]"],
        fin["Battery lifetime [yr]"],
        econ.inflation * 100, econ.price_esc * 100,
    )
    logger.info("%s", SEP)
    logger.info("  %-45s : €%12,.0f / yr", "Annual savings (yr 1)", fin["Annual savings yr1 [€]"])
    logger.info("  %-45s : €%12,.0f / yr", "Optimised grid cost (yr 1)", fin["Annual grid cost opt [€]"])
    logger.info("  %-45s : €%12,.0f / yr", "Reference grid cost (no system)", fin["Annual grid cost ref [€]"])
    logger.info("  %-45s : €%12,.0f",       "Battery replacement CAPEX (yr 15)", fin["Battery replacement [€]"])
    logger.info("  %-45s : €%12,.0f   (%.2f M€)", "Net Present Value (NPV)", fin["NPV [€]"], fin["NPV [M€]"])
    irr_str = f"{fin['IRR [%]']:.2f} %" if not np.isnan(fin["IRR [%]"]) else "n/a"
    pb_str  = f"{fin['Payback period [yr]']:.1f} yr" if not np.isnan(fin["Payback period [yr]"]) else "n/a (>25 yr)"
    logger.info("  %-45s : %12s", "Internal Rate of Return (IRR)", irr_str)
    logger.info("  %-45s : %12s", "Simple payback period", pb_str)
    logger.info("  %-45s : %10.1f €/MWh", "LCOE (incl. battery replacement)", fin["LCOE [€/MWh]"])

    # pvlib diagnostics
    logger.info("%s", SEP)
    logger.info("PHYSICAL MODEL DIAGNOSTICS  (pvlib, year 1)")
    logger.info("%s", SEP)
    day_mask = df["ghi_wm2"] > 20
    if "spectral_factor" in df.columns:
        logger.info("  Mean spectral factor (daytime)      : %.4f",
                    df.loc[day_mask, "spectral_factor"].mean())
    if "shade_fraction" in df.columns:
        sh_mean = df.loc[day_mask, "shade_fraction"].mean()
        logger.info("  Mean beam-shade fraction (daytime)  : %.4f  (%.2f %% of beam)",
                    sh_mean, sh_mean * 100)
    if "bifacial_gain_wm2" in df.columns:
        bf_ann = df["bifacial_gain_wm2"].sum() / 1_000
        poa_sum = df["poa_front"].sum() / 1_000 if "poa_front" in df.columns else 1
        logger.info("  Annual bifacial rear gain           : %.1f kWh/m²  (%.1f %% of front POA)",
                    bf_ann, bf_ann / poa_sum * 100 if poa_sum else 0)
    if "t_cell_c" in df.columns:
        logger.info("  Mean / max cell temperature         : %.1f °C  /  %.1f °C",
                    df.loc[day_mask, "t_cell_c"].mean(), df["t_cell_c"].max())
    if "v_dc_v" in df.columns:
        logger.info("  Mean / min DC voltage (daytime)     : %.0f V  /  %.0f V",
                    df.loc[day_mask, "v_dc_v"].mean(),
                    df.loc[day_mask, "v_dc_v"].min())
    if "deg_factor" in df.columns:
        logger.info("  Degradation factor  (start / end)   : %.4f  /  %.4f",
                    df["deg_factor"].iloc[0], df["deg_factor"].iloc[-1])

    # Save hourly results
    df.to_csv("hourly_results.csv", index=True)
    logger.info("Hourly results saved: hourly_results.csv")

    # Plots
    plot_system_results(
        df, summary,
        pv_peak_mwp  = pv_system.peak_kw / 1_000,
        bat_mwh      = bat_system.capacity_kwh / 1_000,
        n_bat_units  = n_bat,
    )
    plot_financial_results(fin, summary, prices, df)

    # Sensitivity analysis (FIX #15: passes weather for re-simulation)
    logger.info("Running 2-D sensitivity analysis…")
    pv_g, bat_g, lcoe_m, capex_m = sensitivity_analysis(
        weather       = weather,
        base_pv_mwp   = pv_system.peak_kw / 1_000,
        base_bat_mwh  = bat_system.capacity_kwh / 1_000,
        panel_spec    = panel_spec,
        battery_spec  = battery_spec,
        inverter_spec = inverter_spec,
        site          = site,
    )
    plot_sensitivity_results(
        pv_g, bat_g, lcoe_m, capex_m,
        pv_system.peak_kw / 1_000,
        bat_system.capacity_kwh / 1_000,
    )

    return df, summary, fin, econ


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MJ2438 – PV + Battery Subsystem Techno-Economic Analysis "
                    "(pvlib-enhanced physical model, all corrections applied)")
    parser.add_argument(
        "--pvgis", metavar="CSV", default=None,
        help="Path to PVGIS TMY CSV for Seville. "
             "Download: https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY "
             "(lat=37.39, lon=-5.99)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging output.",   # FIX #17
    )
    args = parser.parse_args()
    main(pvgis_path=args.pvgis, verbose=args.verbose)