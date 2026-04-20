# params.py
# =========================================================
# All technical and economic parameters for the hybrid system.
# Every teammate reads from this file — edit only here.
# =========================================================

# ----------------------------------------------------------
# INDUSTRY (food factory, Seville, Spain)
# ----------------------------------------------------------
LOAD_W         = 10e6      # [W]   Thermal load = 10 MW
T_SUPPLY       = 100       # [°C]  Required supply temperature
T_RETURN       = 50        # [°C]  Return temperature from process
HOUR_START     = 8         # [h]   Shift start
HOUR_END       = 18        # [h]   Shift end

# ----------------------------------------------------------
# LOCATION (Seville, Spain)
# Used for solar position calculations in STC model.
# ----------------------------------------------------------
LATITUDE       = 37.4      # [°N]  Seville latitude

# ----------------------------------------------------------
# SOLAR COLLECTORS: ABSOLICON T160
# Source: Absolicon official datasheet
# ----------------------------------------------------------
ETA0           = 0.762     # [-]         Peak optical efficiency
A1             = 0.665     # [W/(m²·K)]  Linear heat loss coeff.
A2             = 0.00378   # [W/(m²·K²)] Quadratic heat loss coeff.
APERTURE_W     = 1.7       # [m]   Collector aperture width
MODULE_L       = 12.0      # [m]   Module length
A_MODULE       = APERTURE_W * MODULE_L   # [m²] = 20.4 m² per module
DNI_MIN        = 50        # [W/m²] Minimum DNI to operate

# Incidence Angle Modifier (IAM) — single-axis N-S horizontal tracker
# Model: IAM(θ) = max(0, 1 − b0·(1/cos(θ) − 1))   [ISO 9806 / ASHRAE]
# b0 ≈ 0.10 is typical for parabolic-trough concentrating collectors.
# Source: IEA SHC Task 33, Fischer et al. (2004)
IAM_B0         = 0.10      # [-]   Incidence angle modifier coefficient

# ----------------------------------------------------------
# TES: PRESSURISED HOT WATER STORAGE
# NOTE: T_TES_MAX raised to 105 °C (pressurised tank, ~2 bar)
#       so the tank can store heat above T_SUPPLY = 100 °C.
#       Previous value of 95 °C was below process supply temperature
#       and prevented useful TES discharge at full quality.
# ----------------------------------------------------------
CP_WATER       = 4186      # [J/(kg·K)]
RHO_WATER      = 975       # [kg/m³] at ~75 °C
T_TES_MAX      = 105       # [°C]  Max TES temperature (pressurised, ~2 bar)
T_TES_MIN      = 55        # [°C]  Min TES temperature (= T_RETURN + 5 °C margin)
DELTA_T_TES    = T_TES_MAX - T_TES_MIN   # = 50 °C usable swing
TES_LOSS_FRAC  = 0.003     # [1/h] Passive heat loss fraction per hour
SOC_MAX        = 1.00
SOC_MIN        = 0.05
SOC_INITIAL    = 0.50

# TES discharge rate limiter — simulates heat-exchanger physical sizing.
# Max discharge power [W] = TES_MAX_DISCHARGE_FRAC × E_tes_Wh  (i.e., Wh → W in 1 h)
# Examples with this value:
#   5 MWh TES  → max 2.5 MW  (small HX, partial coverage of 10 MW demand)
#  50 MWh TES  → max 25 MW   (large HX, never the binding constraint)
TES_MAX_DISCHARGE_FRAC = 0.50   # [-]  fraction of rated capacity dischargeable per hour

# ----------------------------------------------------------
# WASTE HEAT SOURCE
# From the project brief: industrial waste heat at ~70 °C,
# available during production shift (same hours as demand).
# Q_WASTE_MAX_W is a PLACEHOLDER — size it based on your
# specific industrial process data.
# ----------------------------------------------------------
T_WASTE_HEAT   = 70.0      # [°C]  Waste heat source temperature
Q_WASTE_MAX_W  = 500e3     # [W]   Max waste heat power available (placeholder: 500 kW)

# ----------------------------------------------------------
# HEAT PUMP (placeholder — teammate fills details)
# ----------------------------------------------------------
HP_COP         = 3.5       # [-]   Coefficient of Performance
HP_T_MAX       = 95        # [°C]  Max supply temperature

# ----------------------------------------------------------
# ELECTRIC HEATER (placeholder — teammate fills details)
# ----------------------------------------------------------
EL_HEATER_EFF  = 0.99      # [-]   Electrical to thermal efficiency

# ----------------------------------------------------------
# PV PANELS (placeholder — teammate fills details)
# ----------------------------------------------------------
PV_ETA         = 0.20      # [-]   Panel efficiency
PV_AREA_MODULE = 2.0       # [m²]  Area per PV module
PV_TEMP_COEFF  = -0.004    # [1/°C] Power temperature coefficient

# ----------------------------------------------------------
# BESS (placeholder — teammate fills details)
# ----------------------------------------------------------
BESS_ETA_CHARGE    = 0.95  # [-]  Charging efficiency
BESS_ETA_DISCHARGE = 0.95  # [-]  Discharging efficiency
BESS_LOSS_FRAC     = 0.001 # [1/h] Self-discharge per hour
BESS_SOC_MAX       = 0.95
BESS_SOC_MIN       = 0.10

# ----------------------------------------------------------
# ELECTRICITY
# Source: Eurostat Spain industrial tariff 2024
# ----------------------------------------------------------
ELEC_PRICE     = 0.12      # [€/kWh] Industrial electricity price Spain
ELEC_PRICE_W   = ELEC_PRICE / 1000 / 3600  # [€/J] for calculations

# ----------------------------------------------------------
# ECONOMICS
# Source: IRENA 2023, IEA SHC Task 64, literature review
# ----------------------------------------------------------
DISCOUNT_RATE  = 0.06      # [-]    6% annual discount rate
LIFETIME_YR    = 25        # [yr]   Project lifetime

# Capital costs [€ per unit]
CAPEX_STC_PER_M2    = 380  # [€/m²]   Installed STC cost (Absolicon T160, industrial)
CAPEX_TES_PER_KWH   = 40   # [€/kWh]  Installed TES cost (water tank, industrial scale)
CAPEX_HP_PER_KW     = 500  # [€/kW_th] Installed heat pump cost
CAPEX_EL_HEATER_KW  = 80   # [€/kW_th] Installed electric heater cost
CAPEX_PV_PER_KWP    = 700  # [€/kWp]  Installed PV cost (ground-mounted, Spain 2024)
CAPEX_BESS_PER_KWH  = 350  # [€/kWh]  Installed BESS cost (Li-ion, 2024)

# Annual O&M costs as % of CAPEX
OM_STC_FRAC    = 0.015     # 1.5% per year
OM_TES_FRAC    = 0.010     # 1.0% per year
OM_HP_FRAC     = 0.025     # 2.5% per year
OM_EL_FRAC     = 0.015     # 1.5% per year
OM_PV_FRAC     = 0.010     # 1.0% per year
OM_BESS_FRAC   = 0.020     # 2.0% per year

# ----------------------------------------------------------
# OPTIMISER SEARCH SPACE
# These ranges define what the optimiser will try.
# Teammates can add their own ranges below.
# ----------------------------------------------------------
OPT_N_STC_RANGE   = range(100, 1400, 50)   # Number of STC modules to try
OPT_E_TES_RANGE   = range(5,  200,  10)    # TES capacity [MWh] to try

# Teammates add their ranges here, for example:
# OPT_P_HP_RANGE  = range(1000, 8000, 500)  # HP size [kW_th]
# OPT_N_PV_RANGE  = range(100,  3000, 100)  # Number of PV modules
