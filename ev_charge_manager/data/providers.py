"""
Real-world data providers for weather and traffic.

These providers load hourly data from CSV files and supply it to the
simulation via the same interface as the synthetic generators.
"""

import random
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd


class RealWeatherProvider:
    """
    Provides real solar irradiance, wind speed, and temperature
    from a PVGIS TMY dataset.
    """

    def __init__(self, csv_path: str):
        """
        Load weather data from CSV.

        Expected columns: month, day, hour, ghi_wm2, wind_speed_ms, temperature_c
        """
        self.df = pd.read_csv(csv_path)
        self._build_lookup()

    def _build_lookup(self):
        """Build a fast lookup dict keyed by (month, day, hour)."""
        self._lookup: Dict[tuple, Dict[str, float]] = {}
        for _, row in self.df.iterrows():
            key = (int(row["month"]), int(row["day"]), int(row["hour"]))
            self._lookup[key] = {
                "ghi_wm2": float(row["ghi_wm2"]),
                "wind_speed_ms": float(row["wind_speed_ms"]),
                "temperature_c": float(row["temperature_c"]),
            }

    def get_conditions(self, timestamp: datetime) -> Dict[str, float]:
        """Get raw weather conditions for the given timestamp."""
        key = (timestamp.month, timestamp.day, timestamp.hour)
        if key in self._lookup:
            return self._lookup[key]
        # Fallback: find closest available hour for the month/day
        for delta in [1, -1, 2, -2]:
            fallback_key = (timestamp.month, timestamp.day, max(0, min(23, timestamp.hour + delta)))
            if fallback_key in self._lookup:
                return self._lookup[fallback_key]
        # Ultimate fallback
        return {"ghi_wm2": 0.0, "wind_speed_ms": 3.0, "temperature_c": 10.0}

    def get_solar_output_kw(self, timestamp: datetime, peak_power_kw: float) -> float:
        """
        Convert GHI irradiance to solar panel output.

        Uses the simplified model:
            output = peak_power_kw * (GHI / 1000) * temperature_derating

        where temperature_derating accounts for reduced efficiency at high temps
        (panels are rated at STC = 25 deg C, ~0.4%/deg C loss above that).
        """
        conditions = self.get_conditions(timestamp)
        ghi = conditions["ghi_wm2"]
        temp = conditions["temperature_c"]

        # Temperature derating: panels lose ~0.4% per degree above 25 C
        temp_derating = max(0.75, 1.0 - 0.004 * max(0.0, temp - 25.0))

        output = peak_power_kw * (ghi / 1000.0) * temp_derating
        return max(0.0, output)

    def get_wind_output_kw(
        self,
        timestamp: datetime,
        base_power_kw: float,
        rated_speed: float = 12.0,
        cut_in: float = 3.0,
        cut_out: float = 25.0,
    ) -> float:
        """
        Convert wind speed to turbine output using a simplified power curve.

        - Below cut_in: zero output
        - Between cut_in and rated_speed: cubic interpolation
        - Between rated_speed and cut_out: full rated power
        - Above cut_out: zero (turbine shuts down)
        """
        conditions = self.get_conditions(timestamp)
        ws = conditions["wind_speed_ms"]

        if ws < cut_in or ws > cut_out:
            return 0.0
        elif ws >= rated_speed:
            return base_power_kw
        else:
            # Cubic interpolation between cut-in and rated speed
            fraction = ((ws - cut_in) / (rated_speed - cut_in)) ** 3
            return base_power_kw * fraction


class RealTrafficProvider:
    """
    Provides real hourly EV traffic rates from a generated
    Autobahn traffic profile.
    """

    def __init__(self, csv_path: str, noise_factor: float = 0.05):
        """
        Load traffic data from CSV.

        Expected columns: month, day_of_year, day_of_week, hour,
                          total_vehicles_per_hour, ev_vehicles_per_hour
        """
        self.df = pd.read_csv(csv_path)
        self.noise_factor = noise_factor
        self._build_lookup()

    def _build_lookup(self):
        """Build a fast lookup dict keyed by (day_of_year, hour)."""
        self._lookup: Dict[tuple, float] = {}
        for _, row in self.df.iterrows():
            key = (int(row["day_of_year"]), int(row["hour"]))
            self._lookup[key] = float(row["ev_vehicles_per_hour"])

    def get_vehicles_per_hour(self, timestamp: datetime) -> float:
        """Get EV charging traffic rate for the given timestamp."""
        day_of_year = timestamp.timetuple().tm_yday
        # Clamp to 365 for TMY alignment
        if day_of_year > 365:
            day_of_year = 365
        key = (day_of_year, timestamp.hour)

        if key in self._lookup:
            base_rate = self._lookup[key]
        else:
            # Fallback: average rate
            base_rate = self.df["ev_vehicles_per_hour"].mean()

        # Add light noise
        noise = random.uniform(1.0 - self.noise_factor, 1.0 + self.noise_factor)
        return max(0.0, base_rate * noise)

    def get_total_vehicles_per_hour(self, timestamp: datetime) -> float:
        """Get total traffic rate (all vehicles) for the given timestamp."""
        day_of_year = timestamp.timetuple().tm_yday
        if day_of_year > 365:
            day_of_year = 365

        row_mask = (self.df["day_of_year"] == day_of_year) & (self.df["hour"] == timestamp.hour)
        matched = self.df.loc[row_mask, "total_vehicles_per_hour"]
        if len(matched) > 0:
            return float(matched.iloc[0])
        return self.df["total_vehicles_per_hour"].mean()
