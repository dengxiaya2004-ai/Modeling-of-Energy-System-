# climate_data.py
# Loads real PVGIS TMY data from the manually downloaded CSV.

import numpy as np
import pandas as pd

def load_climate_data(filepath: str) -> dict:
    """
    Load hourly climate data from a PVGIS TMY CSV file.
    Download from: https://re.jrc.ec.europa.eu/pvg_tools -> TMY -> Seville

    Returns a dict with numpy arrays of length 8760:
        DNI      [W/m²]  Direct Normal Irradiance
        GHI      [W/m²]  Global Horizontal Irradiance
        T_amb    [°C]    Ambient air temperature at 2m
        month    [1..12] Month index for each hour
        hour_of_day [0..23] Hour of day for each hour
    """
    import os
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"\nFile not found: {filepath}\n"
            "Please download the TMY CSV from PVGIS:\n"
            "  1. Go to https://re.jrc.ec.europa.eu/pvg_tools\n"
            "  2. Click 'TMY', search Seville, download CSV\n"
            "  3. Save as 'seville_tmy.csv' in the project folder"
        )

    # Find the header row (contains "time(UTC)")
    header_row = None
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if 'time(UTC)' in line:
                header_row = i
                break

    if header_row is None:
        raise ValueError("Could not find 'time(UTC)' header in PVGIS file.")

    df = pd.read_csv(filepath, skiprows=header_row, nrows=8760)

    # PVGIS column names
    DNI  = df['Gb(n)'].values.astype(float)
    GHI  = df['Gd(h)'].values.astype(float)
    Tamb = df['T2m'].values.astype(float)

    # Remove physically impossible negatives
    DNI  = np.maximum(DNI,  0.0)
    GHI  = np.maximum(GHI,  0.0)

    # Build month and hour-of-day vectors
    n = 8760
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month      = np.zeros(n, dtype=int)
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

    climate = {
        'DNI':         DNI,
        'GHI':         GHI,
        'T_amb':       Tamb,
        'month':       month,
        'hour_of_day': hour_of_day,
        'n_hours':     n,
    }

    print(f"[Climate] Annual average DNI:            {DNI.mean():.1f} W/m²")
    print(f"[Climate] Peak DNI:                      {DNI.max():.1f} W/m²")
    print(f"[Climate] Annual average temperature:    {Tamb.mean():.1f} °C")
    print(f"[Climate] Sun hours (DNI > 50 W/m²):     {(DNI > 50).sum()} h/year")
    return climate

