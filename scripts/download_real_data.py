#!/usr/bin/env python3
"""
Download and prepare real-world data for full-year RL model validation.

Downloads:
  - PVGIS TMY data (solar radiation, wind speed, temperature) for Frankfurt, Germany
  - Generates Autobahn traffic profiles based on BASt statistics

Produces:
  - data/real_world/frankfurt_corridor_weather.csv  (8760 rows)
  - data/real_world/frankfurt_corridor_traffic.csv   (8760 rows)
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ev_charge_manager.data.pvgis_client import load_or_download_tmy, FRANKFURT_LAT, FRANKFURT_LON
from ev_charge_manager.data.traffic_profiles import load_or_generate_traffic


def main():
    cache_dir = str(project_root / "data" / "cache")
    output_dir = str(project_root / "data" / "real_world")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # --- Weather data ---
    print(f"Downloading PVGIS TMY data for Frankfurt ({FRANKFURT_LAT}N, {FRANKFURT_LON}E)...")
    try:
        weather_df = load_or_download_tmy(
            latitude=FRANKFURT_LAT,
            longitude=FRANKFURT_LON,
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"PVGIS API download failed: {e}")
        print("Generating fallback synthetic TMY data based on Central European statistics...")
        weather_df = _generate_fallback_weather()

    weather_path = os.path.join(output_dir, "frankfurt_corridor_weather.csv")
    weather_df.to_csv(weather_path, index=False)
    print(f"Weather data saved: {weather_path} ({len(weather_df)} rows)")
    print(f"  GHI range: {weather_df['ghi_wm2'].min():.0f} - {weather_df['ghi_wm2'].max():.0f} W/m2")
    print(f"  Wind speed range: {weather_df['wind_speed_ms'].min():.1f} - {weather_df['wind_speed_ms'].max():.1f} m/s")
    print(f"  Temperature range: {weather_df['temperature_c'].min():.1f} - {weather_df['temperature_c'].max():.1f} C")
    print(f"  Annual avg GHI: {weather_df['ghi_wm2'].mean():.0f} W/m2")

    # --- Traffic data ---
    print("\nGenerating Autobahn traffic profile (AADT=50000, 15% EV penetration)...")
    traffic_df = load_or_generate_traffic(
        base_aadt=50000.0,
        ev_penetration=0.15,
        cache_dir=cache_dir,
        seed=42,
    )

    traffic_path = os.path.join(output_dir, "frankfurt_corridor_traffic.csv")
    traffic_df.to_csv(traffic_path, index=False)
    print(f"Traffic data saved: {traffic_path} ({len(traffic_df)} rows)")
    print(f"  EV charging traffic range: {traffic_df['ev_vehicles_per_hour'].min():.1f} - {traffic_df['ev_vehicles_per_hour'].max():.1f} veh/hr")
    print(f"  Annual avg EV charging rate: {traffic_df['ev_vehicles_per_hour'].mean():.1f} veh/hr")
    print(f"  Total annual EV charging vehicles: {traffic_df['ev_vehicles_per_hour'].sum():.0f}")

    print("\nData preparation complete.")


def _generate_fallback_weather():
    """
    Generate realistic Central European weather data as fallback
    if PVGIS API is unavailable.
    """
    import numpy as np

    rows = []
    np.random.seed(42)

    # Frankfurt: latitude ~50N
    # Annual GHI ~1050 kWh/m2/year, avg wind ~3.5 m/s at 10m
    from datetime import date, timedelta

    start = date(2024, 1, 1)
    for day_offset in range(365):
        current = start + timedelta(days=day_offset)
        month = current.month
        day = current.day
        day_of_year = day_offset + 1

        # Solar declination angle (approximate)
        declination = 23.45 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))
        # Day length in hours (approximate for 50N latitude)
        cos_ha = -np.tan(np.radians(50)) * np.tan(np.radians(declination))
        cos_ha = np.clip(cos_ha, -1, 1)
        day_length = (2 / 15) * np.degrees(np.arccos(cos_ha))
        sunrise = 12 - day_length / 2
        sunset = 12 + day_length / 2

        # Monthly peak GHI (clear-sky noon) - Central Europe
        monthly_peak_ghi = {
            1: 250, 2: 350, 3: 500, 4: 650, 5: 780,
            6: 850, 7: 830, 8: 750, 9: 600, 10: 400,
            11: 280, 12: 200,
        }
        peak_ghi = monthly_peak_ghi[month]

        # Monthly avg temperature
        monthly_temp = {
            1: 1.0, 2: 2.0, 3: 6.0, 4: 10.5, 5: 15.0,
            6: 18.0, 7: 20.0, 8: 19.5, 9: 15.5, 10: 10.5,
            11: 5.5, 12: 2.5,
        }
        base_temp = monthly_temp[month]

        # Monthly avg wind speed at 10m
        monthly_wind = {
            1: 4.0, 2: 3.8, 3: 3.9, 4: 3.5, 5: 3.2,
            6: 3.0, 7: 2.9, 8: 2.8, 9: 3.0, 10: 3.3,
            11: 3.7, 12: 4.1,
        }
        base_wind = monthly_wind[month]

        # Cloud cover factor for the day (random)
        cloud_factor = np.random.beta(3, 2)  # skewed toward clearer

        for hour in range(24):
            # Solar GHI
            if sunrise <= hour <= sunset:
                solar_angle = np.pi * (hour - sunrise) / (sunset - sunrise)
                clear_sky_ghi = peak_ghi * np.sin(solar_angle)
                ghi = clear_sky_ghi * cloud_factor * np.random.uniform(0.85, 1.0)
            else:
                ghi = 0.0

            # Temperature: diurnal cycle
            diurnal_range = 8.0 + 4.0 * np.sin(np.radians((month - 1) * 30))
            temp_offset = diurnal_range * 0.5 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else -diurnal_range * 0.3
            temp = base_temp + temp_offset + np.random.normal(0, 1.5)

            # Wind speed: slight diurnal variation + noise
            wind = base_wind * (1 + 0.2 * np.sin(np.pi * hour / 12)) + np.random.exponential(0.5)
            wind = max(0.0, wind)

            rows.append({
                "month": month,
                "day": day,
                "hour": hour,
                "ghi_wm2": max(0.0, round(ghi, 1)),
                "wind_speed_ms": round(wind, 1),
                "temperature_c": round(temp, 1),
            })

    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
