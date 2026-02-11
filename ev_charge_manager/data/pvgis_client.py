"""
PVGIS TMY (Typical Meteorological Year) API client.

Downloads hourly solar radiation, wind speed, and temperature data
from the EU Joint Research Centre's PVGIS service.
https://re.jrc.ec.europa.eu/pvg_tools/en/
"""

import os
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Default location: Frankfurt am Main, Germany (major A5/A3 highway corridor)
FRANKFURT_LAT = 50.11
FRANKFURT_LON = 8.68

PVGIS_TMY_URL = "https://re.jrc.ec.europa.eu/api/v5_2/tmy"


def download_pvgis_tmy(
    latitude: float = FRANKFURT_LAT,
    longitude: float = FRANKFURT_LON,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """
    Download TMY data from PVGIS API.

    Returns a DataFrame with 8760 rows (one per hour of the typical year):
        month, day, hour, ghi_wm2, wind_speed_ms, temperature_c
    """
    params = {
        "lat": latitude,
        "lon": longitude,
        "outputformat": "json",
    }

    response = requests.get(PVGIS_TMY_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    hourly_records = data["outputs"]["tmy_hourly"]

    rows = []
    for rec in hourly_records:
        # PVGIS time format: "20050101:0010" (YYYYMMdd:HHmm)
        time_str = rec["time(UTC)"]
        date_part, time_part = time_str.split(":")
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        hour = int(time_part[:2])

        rows.append({
            "month": month,
            "day": day,
            "hour": hour,
            "ghi_wm2": float(rec.get("G(h)", 0.0)),
            "wind_speed_ms": float(rec.get("WS10m", 0.0)),
            "temperature_c": float(rec.get("T2m", 0.0)),
        })

    df = pd.DataFrame(rows)
    return df


def load_or_download_tmy(
    latitude: float = FRANKFURT_LAT,
    longitude: float = FRANKFURT_LON,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Load TMY data from cache, downloading if not present."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    filename = f"pvgis_tmy_{latitude}_{longitude}.csv"
    filepath = cache_path / filename

    if filepath.exists():
        df = pd.read_csv(filepath)
        if len(df) >= 8760:
            return df

    df = download_pvgis_tmy(latitude, longitude, cache_dir)
    df.to_csv(filepath, index=False)
    return df
