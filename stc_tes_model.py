# =============================================================
# stc_tes_model.py
# YOUR component — Solar Thermal Collectors + Thermal Storage
# Food factory, Seville, Spain
#
# WHAT THIS FILE DOES:
#   1. Loads real climate data from PVGIS (seville_tmy.csv)
#   2. Simulates the Absolicon T160 solar collector field
#   3. Simulates the hot water thermal energy storage tank
#   4. Finds the optimal STC + TES sizing that minimises LCOH
#   5. Exports results for the team (system_results.csv)
#
# HOW TO RUN:
#   python stc_tes_model.py
#
# OUTPUT FOR TEAMMATES:
#   Load system_results.csv — it contains hourly Q_solar_W,
#   Q_tes_W, Q_residual_W (what still needs to be covered by
#   HP, el. heater, PV, BESS) and SOC_TES for every hour.
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================================================
# SECTION 1 — PARAMETERS
# =============================================================

# --- Industry ---
LOAD_W      = 10e6   # [W]   Thermal load = 10 MW
T_SUPPLY    = 100    # [°C]  Required supply temperature
T_RETURN    = 50     # [°C]  Return temperature
HOUR_START  = 8      # [h]   Shift start
HOUR_END    = 18     # [h]   Shift end

# --- Absolicon T160 collectors ---
# Source: official Absolicon T160 datasheet
ETA0        = 0.762   # [-]         Peak optical efficiency
A1          = 0.665   # [W/(m²·K)]  Linear heat loss coefficient
A2          = 0.00378 # [W/(m²·K²)] Quadratic heat loss coefficient
APERTURE_W  = 1.7     # [m]   Aperture width per module
MODULE_L    = 12.0    # [m]   Module length
A_MODULE    = APERTURE_W * MODULE_L   # [m²] = 20.4 m² per module
DNI_MIN     = 50      # [W/m²] Minimum DNI to operate collectors

# --- TES: pressurised hot water tank ---
CP_WATER    = 4186   # [J/(kg·K)]
RHO_WATER   = 975    # [kg/m³] at ~75°C
T_TES_MAX   = 95     # [°C]  Maximum tank temperature
T_TES_MIN   = 55     # [°C]  Minimum useful tank temperature
DELTA_T_TES = T_TES_MAX - T_TES_MIN   # = 40°C useful swing
TES_LOSS    = 0.003  # [1/h]  Passive thermal loss per hour
SOC_MAX     = 1.00   # [-]   Maximum state of charge
SOC_MIN     = 0.05   # [-]   Minimum state of charge (pump safety)
SOC_INIT    = 0.50   # [-]   Initial state of charge

# --- Economics ---
# Source: IRENA 2023, IEA SHC Task 64, Eurostat Spain 2024
DISCOUNT_RATE    = 0.06  # [-]  Annual discount rate
LIFETIME_YR      = 25    # [yr] Project lifetime
CAPEX_STC_M2     = 380   # [€/m²]  Installed STC cost (Absolicon, industrial)
CAPEX_TES_KWH    = 40    # [€/kWh] Installed TES cost (water tank)
OM_STC_FRAC      = 0.015 # [-/yr]  O&M as fraction of CAPEX (1.5%/yr)
OM_TES_FRAC      = 0.010 # [-/yr]  O&M as fraction of CAPEX (1.0%/yr)

# --- Optimiser search space ---
OPT_N_STC_MIN  = 100    # Minimum number of STC modules to try
OPT_N_STC_MAX  = 1500   # Maximum number of STC modules to try
OPT_N_STC_STEP = 50     # Step size
OPT_E_TES_MIN  = 5      # Minimum TES capacity [MWh] to try
OPT_E_TES_MAX  = 200    # Maximum TES capacity [MWh] to try
OPT_E_TES_STEP = 10     # Step size [MWh]


# =============================================================
# SECTION 2 — CLIMATE DATA
# =============================================================

def load_climate_data(filepath):
    """
    Load real hourly climate data from a PVGIS TMY CSV file.

    HOW TO GET THE FILE:
      1. Go to https://re.jrc.ec.europa.eu/pvg_tools
      2. Click TMY in the left menu
      3. Search for Seville, Spain on the map
      4. Click Download -> CSV format
      5. Save the file as seville_tmy.csv in your project folder

    Returns a dict with 8760-element numpy arrays (one per hour of the year):
      DNI         [W/m²]  Direct Normal Irradiance (used by STC collectors)
      GHI         [W/m²]  Global Horizontal Irradiance
      T_amb       [°C]    Ambient air temperature at 2m height
      month       [1..12] Month corresponding to each hour
      hour_of_day [0..23] Hour of the day corresponding to each hour
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"\nFile not found: {filepath}\n"
            "Please download the TMY CSV from PVGIS:\n"
            "  1. Go to https://re.jrc.ec.europa.eu/pvg_tools\n"
            "  2. Click TMY, search Seville, download CSV\n"
            "  3. Save as 'seville_tmy.csv' in the same folder as this script"
        )

    # Find the header row (the row that contains "time(UTC)")
    header_row = None
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if 'time(UTC)' in line:
                header_row = i
                break

    if header_row is None:
        raise ValueError(
            "Could not find 'time(UTC)' header in the PVGIS file.\n"
            "Make sure you downloaded the correct CSV format from PVGIS TMY."
        )

    # Read exactly 8760 rows of hourly data
    df = pd.read_csv(filepath, skiprows=header_row, nrows=8760)

    # Extract the columns we need (standard PVGIS column names)
    DNI  = np.maximum(df['Gb(n)'].values.astype(float), 0.0)
    GHI  = np.maximum(df['Gd(h)'].values.astype(float), 0.0)
    Tamb = df['T2m'].values.astype(float)

    # Build month and hour-of-day vectors
    n = 8760
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month       = np.zeros(n, dtype=int)
    hour_of_day = np.zeros(n, dtype=int)
    idx = 0
    for m, days in enumerate(days_per_month, start=1):
        for _ in range(days):
            for h in range(24):
                if idx >= n:
                    break
                month[idx]       = m
                hour_of_day[idx] = h
                idx += 1

    print(f"[Climate] Annual average DNI:          {DNI.mean():.1f} W/m²")
    print(f"[Climate] Peak DNI:                    {DNI.max():.1f} W/m²")
    print(f"[Climate] Annual average temperature:  {Tamb.mean():.1f} °C")
    print(f"[Climate] Sun hours (DNI > 50 W/m²):   {(DNI > 50).sum()} h/year")

    return {
        'DNI':         DNI,
        'GHI':         GHI,
        'T_amb':       Tamb,
        'month':       month,
        'hour_of_day': hour_of_day,
        'n_hours':     n,
    }


# =============================================================
# SECTION 3 — DEMAND PROFILE
# =============================================================

def build_demand(climate):
    """
    Build the hourly thermal demand profile for the factory.
    The factory requires LOAD_W watts every day from HOUR_START to HOUR_END.

    Returns Q_demand : [W] 8760-array
    """
    Q = np.zeros(climate['n_hours'])
    for i in range(climate['n_hours']):
        h = climate['hour_of_day'][i]
        if HOUR_START <= h < HOUR_END:
            Q[i] = LOAD_W
    return Q


# =============================================================
# SECTION 4 — STC MODEL (Absolicon T160)
# =============================================================

def simulate_stc(climate, n_modules):
    """
    Simulate the Absolicon T160 solar collector field over 8760 hours.

    Physics: EN ISO 9806 efficiency formula
      eta = eta0 - a1*(Tm - Ta)/G - a2*(Tm - Ta)^2 / G
      Q   = eta * A_total * G

    where:
      Tm = mean fluid temperature = (T_supply + T_return) / 2 = 75°C
      Ta = ambient temperature from PVGIS [°C]
      G  = DNI (Direct Normal Irradiance) [W/m²]

    Parameters
    ----------
    climate   : dict from load_climate_data()
    n_modules : int — number of Absolicon T160 modules to simulate

    Returns
    -------
    Q_solar_W : [W]  8760-array — thermal power produced each hour
    eta       : [-]  8760-array — instantaneous collector efficiency
    A_total   : [m²] total aperture area of the field
    """
    DNI   = climate['DNI']
    T_amb = climate['T_amb']
    n     = climate['n_hours']

    A_total = n_modules * A_MODULE
    T_m     = (T_SUPPLY + T_RETURN) / 2    # Mean fluid temperature = 75°C

    Q_solar = np.zeros(n)
    eta     = np.zeros(n)

    # Only compute where there is enough sunlight (vectorised for speed)
    sun = DNI > DNI_MIN
    G   = DNI[sun]
    Ta  = T_amb[sun]

    # Reduced temperature difference (EN ISO 9806 standard parameter)
    X       = (T_m - Ta) / G              # [m²·K/W]
    eta_sun = ETA0 - A1 * X - A2 * X**2 * G
    eta_sun = np.maximum(eta_sun, 0.0)    # Efficiency cannot be negative

    Q_solar[sun] = eta_sun * A_total * G  # [W]
    eta[sun]     = eta_sun

    return Q_solar, eta, A_total


# =============================================================
# SECTION 5 — TES MODEL (hot water storage tank)
# =============================================================

def simulate_tes(Q_solar_W, Q_demand_W, E_tes_Wh):
    """
    Simulate the pressurised hot water TES over 8760 hours.

    Control logic (rule-based, merit order step 2):
      If STC produces MORE than demand  → charge TES with surplus
      If STC produces LESS than demand  → discharge TES to compensate
      If TES is full and surplus exists → energy is curtailed (wasted)
      If TES is empty and deficit exists → backup system must cover

    Parameters
    ----------
    Q_solar_W  : [W]  8760-array — STC thermal output (from simulate_stc)
    Q_demand_W : [W]  8760-array — process thermal demand
    E_tes_Wh   : [Wh] TES storage capacity

    Returns
    -------
    Q_tes_W    : [W]  8760-array — power delivered by TES to process
    SOC        : [-]  8760-array — state of charge (0=empty, 1=full)
    Q_residual : [W]  8760-array — unmet demand after STC + TES
    V_tank_m3  : [m³] required water volume for this TES capacity
    """
    n     = len(Q_solar_W)
    E_max = E_tes_Wh * 3600.0             # Convert Wh → J

    Q_tes      = np.zeros(n)
    SOC        = np.zeros(n)
    Q_residual = np.zeros(n)

    E_stored = SOC_INIT * E_max           # Start at 50% charge

    for i in range(n):
        # Passive thermal losses this hour [J]
        E_loss = TES_LOSS * E_stored * 3600.0

        # Solar surplus relative to process demand:
        # Positive = STC exceeds demand → we have energy to store
        # Negative = STC insufficient  → we need to draw from storage
        surplus_W = Q_solar_W[i] - Q_demand_W[i]

        if surplus_W >= 0:
            # ---- CHARGE: store the solar surplus ----
            E_charge = surplus_W * 3600.0
            E_stored = min(E_stored + E_charge - E_loss, E_max * SOC_MAX)
            E_stored = max(E_stored, 0.0)
            Q_tes[i]      = 0.0           # TES gives nothing to process
            Q_residual[i] = 0.0           # STC already covered demand fully

        else:
            # ---- DISCHARGE: cover the solar deficit ----
            E_deficit   = (-surplus_W) * 3600.0
            E_available = max(E_stored - SOC_MIN * E_max, 0.0)
            E_discharge = min(E_deficit, E_available)

            E_stored = max(E_stored - E_discharge - E_loss, 0.0)

            Q_tes[i] = E_discharge / 3600.0    # [W]

            # Residual = what TES could not cover (goes to backup)
            Q_residual[i] = max(
                Q_demand_W[i] - Q_solar_W[i] - Q_tes[i], 0.0
            )

        SOC[i] = E_stored / E_max

    # Required tank volume from thermodynamics: E = m·cp·deltaT
    V_tank = (E_tes_Wh * 3600.0) / (RHO_WATER * CP_WATER * DELTA_T_TES)

    return Q_tes, SOC, Q_residual, V_tank


# =============================================================
# SECTION 6 — ECONOMICS (STC + TES only)
# =============================================================

def compute_economics(n_modules, E_tes_Wh, A_stc_m2,
                      Q_solar_W, Q_tes_W, Q_covered_W, Q_demand_W):
    """
    Compute CAPEX, OPEX and LCOH for the STC + TES subsystem.

    LCOH (Levelised Cost of Heat) [€/MWh_th]:
      The total cost per MWh of useful heat delivered over the project
      lifetime. This is the number the optimiser minimises.

      LCOH = (CAPEX × CRF + O&M) / E_delivered_annual

    Note: electricity cost is zero for STC+TES (no electricity consumed).

    Returns a dict with all economic metrics.
    """
    # Capital Recovery Factor: converts one-time CAPEX to annual cost
    r   = DISCOUNT_RATE
    n   = LIFETIME_YR
    CRF = r * (1 + r)**n / ((1 + r)**n - 1)

    # CAPEX [€]
    capex_stc = A_stc_m2 * CAPEX_STC_M2
    capex_tes = (E_tes_Wh / 1000.0) * CAPEX_TES_KWH   # Wh → kWh
    capex_tot = capex_stc + capex_tes

    # Annual O&M [€/yr]
    om_annual = capex_stc * OM_STC_FRAC + capex_tes * OM_TES_FRAC

    # Total annual cost [€/yr]
    annual_cost = capex_tot * CRF + om_annual

    # Energy delivered to the process [MWh/yr]
    E_covered_MWh = Q_covered_W.sum() / 1e6

    # LCOH [€/MWh_th]
    LCOH = annual_cost / E_covered_MWh if E_covered_MWh > 0 else float('inf')

    # Energy balance
    E_dem = Q_demand_W.sum() / 1e6
    E_sol = np.minimum(Q_solar_W, Q_demand_W).sum() / 1e6
    E_tes = Q_tes_W.sum() / 1e6

    return {
        'LCOH':           LCOH,
        'CRF':            CRF,
        'capex_stc':      capex_stc,
        'capex_tes':      capex_tes,
        'capex_tot':      capex_tot,
        'om_annual':      om_annual,
        'annual_cost':    annual_cost,
        'E_demand_MWh':   E_dem,
        'E_solar_MWh':    E_sol,
        'E_tes_MWh':      E_tes,
        'E_covered_MWh':  E_covered_MWh,
        'thermal_share':  E_covered_MWh / E_dem * 100,
    }


# =============================================================
# SECTION 7 — OPTIMISER
# Finds the (n_stc, E_tes) combination that minimises LCOH.
# =============================================================

def run_optimiser(climate, Q_demand_W):
    """
    Grid search over (n_modules, E_tes_Wh) to minimise LCOH.

    For each combination it runs the full annual simulation
    and computes the LCOH. The combination with the lowest LCOH
    is the optimal design.

    Returns the optimal configuration and the full LCOH grid
    (useful for plotting the optimisation surface).
    """
    n_stc_vals = list(range(OPT_N_STC_MIN, OPT_N_STC_MAX, OPT_N_STC_STEP))
    e_tes_vals = list(range(OPT_E_TES_MIN, OPT_E_TES_MAX, OPT_E_TES_STEP))
    e_tes_Wh   = [e * 1e6 for e in e_tes_vals]   # MWh → Wh

    n_rows = len(n_stc_vals)
    n_cols = len(e_tes_vals)

    lcoh_grid = np.full((n_rows, n_cols), np.nan)
    ts_grid   = np.full((n_rows, n_cols), np.nan)

    print(f"\n[Optimiser] Searching {n_rows} x {n_cols} = "
          f"{n_rows * n_cols} combinations...")

    best_lcoh   = float('inf')
    best_result = None

    for i, n_stc in enumerate(n_stc_vals):
        for j, E_Wh in enumerate(e_tes_Wh):

            # Simulate STC
            Q_solar, eta, A_stc = simulate_stc(climate, n_stc)

            # Simulate TES
            Q_tes, SOC, Q_res, V_tank = simulate_tes(
                Q_solar, Q_demand_W, E_Wh
            )

            # Total heat covered (STC direct + TES discharge)
            Q_covered = np.minimum(Q_solar + Q_tes, Q_demand_W)

            # Compute economics
            econ = compute_economics(
                n_stc, E_Wh, A_stc,
                Q_solar, Q_tes, Q_covered, Q_demand_W
            )

            lcoh_grid[i, j] = econ['LCOH']
            ts_grid[i, j]   = econ['thermal_share']

            if econ['LCOH'] < best_lcoh:
                best_lcoh = econ['LCOH']
                best_result = {
                    'n_stc':     n_stc,
                    'E_tes_Wh':  E_Wh,
                    'A_stc_m2':  A_stc,
                    'V_tes_m3':  V_tank,
                    'Q_solar_W': Q_solar,
                    'Q_tes_W':   Q_tes,
                    'Q_res_W':   Q_res,
                    'Q_covered': Q_covered,
                    'SOC_tes':   SOC,
                    'eta':       eta,
                    'econ':      econ,
                }

        # Progress update every 5 rows
        if (i + 1) % 5 == 0 or i == n_rows - 1:
            print(f"  [{i+1:>3}/{n_rows}] Best LCOH so far: "
                  f"{best_lcoh:.2f} €/MWh_th  "
                  f"(N_stc={best_result['n_stc']}, "
                  f"E_tes={best_result['E_tes_Wh']/1e6:.0f} MWh)")

    print(f"\n[Optimiser] Optimal LCOH = {best_lcoh:.2f} €/MWh_th")

    return {
        'best':         best_result,
        'lcoh_grid':    lcoh_grid,
        'ts_grid':      ts_grid,
        'n_stc_vals':   n_stc_vals,
        'e_tes_vals':   e_tes_vals,
    }


# =============================================================
# SECTION 8 — RESULTS TABLE
# =============================================================

def print_results(res, Q_demand_W):
    """Print a formatted summary table to the console."""
    econ = res['econ']
    sep  = "=" * 58

    print(f"\n{sep}")
    print(f"  STC + TES OPTIMAL CONFIGURATION — RESULTS SUMMARY")
    print(sep)
    print(f"  SYSTEM SIZING")
    print(f"  {'STC modules':<34} {res['n_stc']:>8} modules")
    print(f"  {'STC field area':<34} {res['A_stc_m2']:>8.0f} m²")
    print(f"  {'STC peak power (est.)':<34} "
          f"{ETA0 * res['A_stc_m2'] * 900 / 1e6:>8.1f} MW")
    print(f"  {'TES capacity':<34} {res['E_tes_Wh']/1e6:>8.1f} MWh")
    print(f"  {'TES tank volume':<34} {res['V_tes_m3']:>8.0f} m³")
    print(f"  {'TES temperature range':<34}   {int(T_TES_MIN)}–{int(T_TES_MAX)} °C")
    print(f"{'-'*58}")
    print(f"  ANNUAL ENERGY BALANCE")
    E = econ['E_demand_MWh']
    print(f"  {'Total annual demand':<34} {E:>8.1f} MWh  (100.0%)")
    print(f"  {'STC direct to process':<34} "
          f"{econ['E_solar_MWh']:>8.1f} MWh  "
          f"({econ['E_solar_MWh']/E*100:.1f}%)")
    print(f"  {'TES contribution':<34} "
          f"{econ['E_tes_MWh']:>8.1f} MWh  "
          f"({econ['E_tes_MWh']/E*100:.1f}%)")
    print(f"  {'Total covered (STC+TES)':<34} "
          f"{econ['E_covered_MWh']:>8.1f} MWh  "
          f"({econ['thermal_share']:.1f}%)")
    print(f"  {'Residual (for HP + boiler)':<34} "
          f"{E - econ['E_covered_MWh']:>8.1f} MWh  "
          f"({100 - econ['thermal_share']:.1f}%)")
    print(f"{'-'*58}")
    print(f"  ECONOMICS (STC + TES only)")
    print(f"  {'CAPEX — STC':<34} "
          f"{econ['capex_stc']/1e6:>8.2f} M€")
    print(f"  {'CAPEX — TES':<34} "
          f"{econ['capex_tes']/1e6:>8.2f} M€")
    print(f"  {'Total CAPEX':<34} "
          f"{econ['capex_tot']/1e6:>8.2f} M€")
    print(f"  {'Annual O&M':<34} "
          f"{econ['om_annual']/1e3:>8.1f} k€/yr")
    print(f"  {'Total annual cost':<34} "
          f"{econ['annual_cost']/1e3:>8.1f} k€/yr")
    print(f"  {'Capital Recovery Factor':<34} "
          f"{econ['CRF']:>8.4f}")
    print(f"{'-'*58}")
    print(f"  {'LCOH (STC + TES)':<34} "
          f"{econ['LCOH']:>8.2f} €/MWh_th")
    print(f"{sep}\n")


# =============================================================
# SECTION 9 — PLOTS
# =============================================================

def plot_all(opt, Q_demand_W, climate):
    """Generate all output plots for the optimal configuration."""
    res    = opt['best']
    econ   = res['econ']
    months = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']

    # ---- Plot 1: LCOH optimisation surface ----
    fig, ax = plt.subplots(figsize=(9, 5))
    cf = ax.contourf(
        opt['e_tes_vals'],
        opt['n_stc_vals'],
        opt['lcoh_grid'],
        levels=25, cmap='RdYlGn_r'
    )
    plt.colorbar(cf, ax=ax, label='LCOH [€/MWh_th]')
    ax.plot(
        res['E_tes_Wh'] / 1e6,
        res['n_stc'],
        'w*', markersize=16,
        label=f"Optimum: {econ['LCOH']:.1f} €/MWh_th\n"
              f"N_stc={res['n_stc']}, "
              f"E_tes={res['E_tes_Wh']/1e6:.0f} MWh"
    )
    ax.set_xlabel('TES capacity [MWh]')
    ax.set_ylabel('Number of STC modules')
    ax.set_title('LCOH optimisation surface — STC field size vs TES capacity')
    ax.legend(fontsize=9)
    plt.tight_layout()

    # ---- Plot 2: Monthly energy breakdown ----
    E_monthly = np.zeros((12, 3))
    for m in range(1, 13):
        idx = climate['month'] == m
        E_monthly[m-1, 0] = np.minimum(
            res['Q_solar_W'][idx], Q_demand_W[idx]
        ).sum() / 1e6
        E_monthly[m-1, 1] = res['Q_tes_W'][idx].sum() / 1e6
        E_monthly[m-1, 2] = res['Q_res_W'][idx].sum()  / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    x   = np.arange(12)
    w   = 0.6
    ax.bar(x, E_monthly[:, 0], w,
           label='STC (direct)', color='#F5A623')
    ax.bar(x, E_monthly[:, 1], w,
           bottom=E_monthly[:, 0],
           label='TES', color='#4A90D9')
    ax.bar(x, E_monthly[:, 2], w,
           bottom=E_monthly[:, 0] + E_monthly[:, 1],
           label='Residual (for teammates)', color='#D0021B',
           alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel('Thermal energy [MWh]')
    ax.set_title('Monthly thermal energy breakdown — STC + TES (optimal configuration)')
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()

    # ---- Plot 3: Typical summer week (June) ----
    idx_jun = np.where(climate['month'] == 6)[0][:168]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.fill_between(range(168), Q_demand_W[idx_jun] / 1e6,
                     alpha=0.12, color='gray', label='Demand area')
    ax1.fill_between(range(168),
                     np.minimum(res['Q_solar_W'][idx_jun],
                                Q_demand_W[idx_jun]) / 1e6,
                     alpha=0.85, color='#F5A623', label='STC')
    ax1.fill_between(range(168),
                     res['Q_tes_W'][idx_jun] / 1e6,
                     alpha=0.85, color='#4A90D9', label='TES')
    ax1.plot(range(168), Q_demand_W[idx_jun] / 1e6,
             'k--', lw=1.3, label='Demand')
    ax1.set_ylabel('Thermal power [MW]')
    ax1.set_title('Typical summer week (June) — power balance and TES SOC')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.4)

    ax2.plot(range(168), res['SOC_tes'][idx_jun] * 100,
             'b-', lw=1.5, label='SOC')
    ax2.axhline(SOC_MIN * 100, color='r', ls='--', lw=1, label='Min SOC')
    ax2.axhline(100, color='g', ls='--', lw=1, label='Max SOC')
    ax2.set_ylabel('TES SOC [%]')
    ax2.set_xlabel('Hour of the week')
    ax2.set_xticks(range(0, 169, 24))
    ax2.set_xticklabels(['Mon','Tue','Wed','Thu',
                          'Fri','Sat','Sun',''])
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(alpha=0.4)
    plt.tight_layout()

    # ---- Plot 4: Typical winter week (January) ----
    idx_jan = np.where(climate['month'] == 1)[0][:168]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.fill_between(range(168), Q_demand_W[idx_jan] / 1e6,
                     alpha=0.12, color='gray')
    ax1.fill_between(range(168),
                     np.minimum(res['Q_solar_W'][idx_jan],
                                Q_demand_W[idx_jan]) / 1e6,
                     alpha=0.85, color='#F5A623', label='STC')
    ax1.fill_between(range(168),
                     res['Q_tes_W'][idx_jan] / 1e6,
                     alpha=0.85, color='#4A90D9', label='TES')
    ax1.plot(range(168), Q_demand_W[idx_jan] / 1e6,
             'k--', lw=1.3, label='Demand')
    ax1.set_ylabel('Thermal power [MW]')
    ax1.set_title('Typical winter week (January) — power balance and TES SOC')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.4)

    ax2.plot(range(168), res['SOC_tes'][idx_jan] * 100,
             'b-', lw=1.5, label='SOC')
    ax2.axhline(SOC_MIN * 100, color='r', ls='--', lw=1, label='Min SOC')
    ax2.axhline(100, color='g', ls='--', lw=1, label='Max SOC')
    ax2.set_ylabel('TES SOC [%]')
    ax2.set_xlabel('Hour of the week')
    ax2.set_xticks(range(0, 169, 24))
    ax2.set_xticklabels(['Mon','Tue','Wed','Thu',
                          'Fri','Sat','Sun',''])
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(alpha=0.4)
    plt.tight_layout()

    plt.show()


# =============================================================
# SECTION 10 — EXPORT FOR TEAMMATES
# =============================================================

def export_for_team(res, Q_demand_W, climate):
    """
    Export hourly results to CSV so teammates can use them
    as input to their components.

    Teammates load this file with:
        import pandas as pd
        df = pd.read_csv('system_results.csv')

    Key columns:
        Q_residual_W  [W]  — unmet demand after STC + TES
                             (this is what HP + el. heater must cover)
        Q_solar_W     [W]  — STC thermal output each hour
        Q_tes_W       [W]  — TES thermal output each hour
        SOC_tes       [-]  — TES state of charge each hour
        DNI           [W/m²] — solar resource for PV teammate
        T_amb         [°C]   — temperature for HP teammate
    """
    n = climate['n_hours']
    df = pd.DataFrame({
        'hour_of_year':  np.arange(1, n + 1),
        'month':         climate['month'],
        'hour_of_day':   climate['hour_of_day'],
        'DNI_W_m2':      climate['DNI'],
        'GHI_W_m2':      climate['GHI'],
        'T_amb_C':       climate['T_amb'],
        'Q_demand_W':    Q_demand_W,
        'Q_solar_W':     res['Q_solar_W'],
        'Q_tes_W':       res['Q_tes_W'],
        'Q_residual_W':  res['Q_res_W'],
        'SOC_tes':       res['SOC_tes'],
    })
    df.to_csv('system_results.csv', index=False)
    print("Hourly results exported to system_results.csv")
    print(f"  -> Teammates can load it with: pd.read_csv('system_results.csv')")
    print(f"  -> Key column for them: Q_residual_W "
          f"(peak = {res['Q_res_W'].max()/1e6:.1f} MW, "
          f"annual = {res['Q_res_W'].sum()/1e6:.0f} MWh)")


# =============================================================
# SECTION 11 — MAIN (entry point)
# =============================================================

if __name__ == '__main__':
    print("=" * 58)
    print("  STC + TES model — Seville, Spain")
    print("  Absolicon T160 + hot water storage")
    print("=" * 58)

    # Step 1 — Load climate data
    print("\n[Step 1] Loading PVGIS climate data...")
    climate = load_climate_data('seville_tmy.csv')

    # Step 2 — Build demand profile
    print("\n[Step 2] Building thermal demand profile...")
    Q_demand = build_demand(climate)
    print(f"[Demand] Annual thermal demand:  "
          f"{Q_demand.sum()/1e6:.1f} MWh/year")
    print(f"[Demand] Annual operating hours: "
          f"{(Q_demand > 0).sum()} h/year")

    # Step 3 — Run optimiser
    print("\n[Step 3] Running techno-economic optimiser...")
    opt = run_optimiser(climate, Q_demand)

    # Step 4 — Print results table
    print_results(opt['best'], Q_demand)

    # Step 5 — Export CSV for teammates
    print("[Step 5] Exporting results for teammates...")
    export_for_team(opt['best'], Q_demand, climate)

    # Step 6 — Generate plots
    print("\n[Step 6] Generating plots...")
    plot_all(opt, Q_demand, climate)