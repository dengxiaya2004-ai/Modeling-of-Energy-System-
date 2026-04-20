# stc_model.py
# Solar Thermal Collector model — Absolicon T160
# YOUR file. Teammates do not edit this.
#
# Changes vs original:
#   1. IAM (Incidence Angle Modifier) applied to ETA0.
#      Parabolic-trough collectors lose optical efficiency when the
#      sun is not perpendicular to the aperture.  For a single-axis
#      N-S horizontal tracker this is fully characterised by the
#      transversal incidence angle θ.
#   2. T_RETURN cut-off: the collector is switched OFF in any hour
#      where the irradiance is too weak to heat the fluid even to
#      T_RETURN.  Without this guard, the model can deliver a tiny
#      positive yield even when the stagnation temperature is below
#      the process return line — physically impossible.

import numpy as np
import params as P


# ---------------------------------------------------------------
# SOLAR POSITION
# ---------------------------------------------------------------

def compute_solar_angles(climate: dict):
    """
    Compute solar altitude and azimuth for all 8760 hours.

    Uses a simplified astronomical model (Cooper declination +
    hour-angle approach).  Accuracy is ±1–2 °, sufficient for
    hourly energy simulations.

    Location: Seville, Spain  (lat = P.LATITUDE = 37.4 °N)

    Returns
    -------
    alpha_deg  : np.ndarray [°]  Solar elevation  (negative = below horizon)
    az_deg     : np.ndarray [°]  Azimuth from south, positive toward west
    """
    lat_rad = np.radians(P.LATITUDE)
    n       = climate['n_hours']
    hour    = climate['hour_of_day']   # integer 0–23

    # Day-of-year: index 0 = hour 0 of Jan 1 = day 1
    doy = np.floor(np.arange(0, n) / 24.0).astype(int) + 1  # 1..365

    # Solar declination [rad]  — Cooper (1969)
    delta = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))

    # Hour angle [rad]: 0 at solar noon, negative AM, positive PM
    # Approximation: solar time ≈ standard time (Seville lon ≈ −6°, UTC+1)
    omega = np.radians(15.0 * (hour - 12))

    # Solar altitude [rad]
    sin_alpha = (np.sin(lat_rad) * np.sin(delta)
                 + np.cos(lat_rad) * np.cos(delta) * np.cos(omega))
    alpha = np.arcsin(np.clip(sin_alpha, -1.0, 1.0))

    # Solar azimuth from south [rad], positive westward
    cos_alpha = np.cos(alpha)
    safe_cos  = np.where(cos_alpha > 1e-6, cos_alpha, 1e-6)
    cos_az    = np.clip(
        (np.sin(lat_rad) * sin_alpha - np.sin(delta)) / (np.cos(lat_rad) * safe_cos),
        -1.0, 1.0
    )
    az = np.arccos(cos_az)
    az = np.where(omega > 0, az, -az)   # afternoon → west of south → positive

    return np.degrees(alpha), np.degrees(az)


# ---------------------------------------------------------------
# INCIDENCE ANGLE MODIFIER
# ---------------------------------------------------------------

def compute_iam(alpha_deg: np.ndarray, az_deg: np.ndarray) -> np.ndarray:
    """
    Compute the Incidence Angle Modifier (IAM) for a single-axis
    horizontal N-S tracker (E-W tracking), as on the Absolicon T160.

    Incidence angle formula for horizontal N-S axis  (Duffie & Beckman):
        cos(θ) = sqrt( sin²(α)  +  cos²(α) · cos²(γ) )
    where α = solar elevation, γ = azimuth from south.

    Verification:
        • Solar noon  (γ = 0°): cos(θ) = 1  → θ = 0°  (perfectly tracked) ✓
        • Sunrise/set (α = 0°): cos(θ) = |cos(γ)| → maximum incidence    ✓

    IAM model  (ISO 9806 ASHRAE):
        IAM(θ) = max(0,  1 − b0 · (1/cos(θ) − 1))
    with b0 = P.IAM_B0 ≈ 0.10  (typical for parabolic troughs,
    within the range reported for Absolicon T160 in IEA SHC Task 33).

    Returns
    -------
    iam : np.ndarray [-]  shape (n_hours,), values in [0, 1]
    """
    alpha = np.radians(alpha_deg)
    az    = np.radians(az_deg)

    cos_theta = np.sqrt(
        np.sin(alpha) ** 2 + np.cos(alpha) ** 2 * np.cos(az) ** 2
    )
    # Clamp away from zero to avoid 1/cos(θ) → ∞ near sunrise/sunset
    cos_theta = np.clip(cos_theta, 1e-4, 1.0)

    iam = 1.0 - P.IAM_B0 * (1.0 / cos_theta - 1.0)
    return np.maximum(iam, 0.0)


# ---------------------------------------------------------------
# MAIN SIMULATION
# ---------------------------------------------------------------

def simulate(climate: dict, n_modules: int) -> dict:
    """
    Simulate the STC field over 8760 hours.

    Parameters
    ----------
    climate   : dict from climate_data.load_climate_data()
    n_modules : int — number of Absolicon T160 modules

    Returns
    -------
    dict with:
        Q_solar_W  [W]   8760-array — thermal power produced each hour
        eta        [-]   8760-array — instantaneous efficiency (incl. IAM)
        iam        [-]   8760-array — incidence angle modifier
        A_total    [m²]  total aperture area
        n_modules  int   number of modules (passed through)
    """
    DNI   = climate['DNI']
    T_amb = climate['T_amb']
    n     = climate['n_hours']

    A_total = n_modules * P.A_MODULE

    # -- 1. Solar position and IAM (computed for all 8760 hours) --
    alpha_deg, az_deg = compute_solar_angles(climate)
    iam               = compute_iam(alpha_deg, az_deg)

    Q_solar = np.zeros(n)
    eta_arr = np.zeros(n)

    # Basic operating mask: DNI above threshold AND sun above horizon
    sun_mask = (DNI > P.DNI_MIN) & (alpha_deg > 0.0)

    G   = DNI[sun_mask]
    Ta  = T_amb[sun_mask]
    IAM = iam[sun_mask]

    # -- 2. T_RETURN cut-off check --
    # The collector must be able to deliver useful heat at T_RETURN
    # (the coldest acceptable fluid temperature).  We evaluate η at
    # T_RETURN — the "easiest" operating point.  If η ≤ 0 there,
    # the stagnation temperature is below T_RETURN and the collector
    # cannot even pre-heat the return fluid; skip this hour entirely.
    #
    # Reduced temperature at T_RETURN:
    #   X_ret = (T_return − T_amb) / G
    X_ret      = (P.T_RETURN - Ta) / G
    eta_at_ret = P.ETA0 * IAM - P.A1 * X_ret - P.A2 * X_ret ** 2 * G
    can_operate = eta_at_ret > 0.0   # boolean mask within sun_mask

    # -- 3. Efficiency at mean fluid temperature T_m (with IAM) --
    T_m   = (P.T_SUPPLY + P.T_RETURN) / 2.0   # mean fluid temperature [°C]
    X_m   = (T_m - Ta) / G
    eta_m = P.ETA0 * IAM - P.A1 * X_m - P.A2 * X_m ** 2 * G
    eta_m = np.maximum(eta_m, 0.0)

    # Zero out hours where collector cannot reach T_RETURN
    eta_m[~can_operate] = 0.0

    # -- 4. Thermal output --
    idx = np.where(sun_mask)[0]
    Q_solar[idx] = eta_m * A_total * G
    eta_arr[idx] = eta_m

    return {
        'Q_solar_W': Q_solar,
        'eta':       eta_arr,
        'iam':       iam,
        'A_total':   A_total,
        'n_modules': n_modules,
    }
