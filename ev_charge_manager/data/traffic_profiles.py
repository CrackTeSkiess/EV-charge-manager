"""
German Autobahn traffic profile generator.

Generates realistic hourly traffic profiles based on published
BASt (Bundesanstalt fuer Strassenwesen) counting station patterns.
"""

import numpy as np
import pandas as pd
from typing import Optional


# Seasonal monthly multipliers (relative to annual average).
# Based on published DTV (Durchschnittlicher Taeglicher Verkehr) data.
MONTHLY_MULTIPLIERS = {
    1: 0.85,   # January - winter low
    2: 0.87,
    3: 0.95,
    4: 1.00,
    5: 1.05,
    6: 1.08,
    7: 1.12,   # July - summer peak (holiday travel)
    8: 1.10,
    9: 1.02,
    10: 0.98,
    11: 0.90,
    12: 0.88,  # December - winter low
}

# Hourly distribution of daily traffic (fraction of daily total per hour).
# Based on BASt permanent counting station data for German Autobahn.
# Index = hour (0-23), value = fraction of daily total.
WEEKDAY_HOURLY_FRACTIONS = [
    0.012,  # 00:00
    0.008,  # 01:00
    0.007,  # 02:00
    0.007,  # 03:00
    0.010,  # 04:00
    0.020,  # 05:00
    0.045,  # 06:00
    0.065,  # 07:00 - morning rush start
    0.070,  # 08:00 - morning rush peak
    0.060,  # 09:00
    0.055,  # 10:00
    0.055,  # 11:00
    0.055,  # 12:00
    0.055,  # 13:00
    0.058,  # 14:00
    0.062,  # 15:00
    0.068,  # 16:00 - evening rush start
    0.070,  # 17:00 - evening rush peak
    0.062,  # 18:00
    0.050,  # 19:00
    0.040,  # 20:00
    0.032,  # 21:00
    0.022,  # 22:00
    0.015,  # 23:00
]

SATURDAY_HOURLY_FRACTIONS = [
    0.015,  # 00:00
    0.010,  # 01:00
    0.008,  # 02:00
    0.007,  # 03:00
    0.008,  # 04:00
    0.012,  # 05:00
    0.025,  # 06:00
    0.040,  # 07:00
    0.055,  # 08:00
    0.065,  # 09:00
    0.068,  # 10:00 - Saturday peak (shopping/leisure)
    0.068,  # 11:00
    0.065,  # 12:00
    0.062,  # 13:00
    0.060,  # 14:00
    0.058,  # 15:00
    0.055,  # 16:00
    0.050,  # 17:00
    0.045,  # 18:00
    0.040,  # 19:00
    0.035,  # 20:00
    0.028,  # 21:00
    0.020,  # 22:00
    0.015,  # 23:00
]

SUNDAY_HOURLY_FRACTIONS = [
    0.015,  # 00:00
    0.012,  # 01:00
    0.010,  # 02:00
    0.008,  # 03:00
    0.007,  # 04:00
    0.008,  # 05:00
    0.015,  # 06:00
    0.025,  # 07:00
    0.038,  # 08:00
    0.050,  # 09:00
    0.058,  # 10:00
    0.062,  # 11:00
    0.060,  # 12:00
    0.058,  # 13:00
    0.060,  # 14:00
    0.065,  # 15:00 - Sunday return peak
    0.068,  # 16:00
    0.070,  # 17:00 - Sunday return peak
    0.060,  # 18:00
    0.050,  # 19:00
    0.040,  # 20:00
    0.032,  # 21:00
    0.022,  # 22:00
    0.017,  # 23:00
]

# Day-of-week total traffic multipliers (relative to weekday average)
DAY_OF_WEEK_MULTIPLIERS = {
    0: 1.00,  # Monday
    1: 1.02,  # Tuesday
    2: 1.03,  # Wednesday
    3: 1.02,  # Thursday
    4: 1.05,  # Friday (higher due to weekend departures)
    5: 0.85,  # Saturday
    6: 0.70,  # Sunday
}

# ---------------------------------------------------------------------------
# Shared constants used by training (environment.py), SUMO (network_generator)
# and real-world validation (validate_real_world.py).
# ---------------------------------------------------------------------------

# Fraction of EVs on the highway that stop to charge at the corridor's
# stations.  Must be the same everywhere so demand magnitudes match.
EV_CHARGING_STOP_FRACTION: float = 0.20

# Peak hourly fraction across the weekday profile — used to normalise the
# time-of-day curve to [0, 1] so that the peak hour gives occupancy ≈ 1.
PEAK_HOURLY_FRACTION: float = max(WEEKDAY_HOURLY_FRACTIONS)


def generate_autobahn_traffic_profile(
    base_aadt: float = 50000.0,
    ev_penetration: float = 0.15,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a full-year hourly traffic profile for a German Autobahn segment.

    Args:
        base_aadt: Annual Average Daily Traffic (total vehicles, both directions).
        ev_penetration: Fraction of traffic that is EVs needing charging.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with 8760 rows indexed by (month, day_of_week, hour)
        and column 'vehicles_per_hour' representing EV charging demand.
    """
    if seed is not None:
        np.random.seed(seed)

    rows = []

    # Build a full year day-by-day
    # Use 2024 as reference (leap year: 366 days, we use 365 for TMY alignment)
    from datetime import date, timedelta
    start = date(2024, 1, 1)

    for day_offset in range(365):
        current_date = start + timedelta(days=day_offset)
        month = current_date.month
        dow = current_date.weekday()  # 0=Monday, 6=Sunday

        # Select hourly fractions based on day type
        if dow == 6:  # Sunday
            hourly_fracs = SUNDAY_HOURLY_FRACTIONS
        elif dow == 5:  # Saturday
            hourly_fracs = SATURDAY_HOURLY_FRACTIONS
        else:  # Weekday
            hourly_fracs = WEEKDAY_HOURLY_FRACTIONS

        # Daily traffic = AADT * seasonal * day-of-week multiplier
        daily_total = (
            base_aadt
            * MONTHLY_MULTIPLIERS[month]
            * DAY_OF_WEEK_MULTIPLIERS[dow]
        )

        for hour in range(24):
            hourly_total = daily_total * hourly_fracs[hour]
            # EV fraction of total traffic
            ev_vehicles = hourly_total * ev_penetration
            ev_charging = ev_vehicles * EV_CHARGING_STOP_FRACTION

            rows.append({
                "month": month,
                "day_of_year": day_offset + 1,
                "day_of_week": dow,
                "hour": hour,
                "total_vehicles_per_hour": hourly_total,
                "ev_vehicles_per_hour": ev_charging,
            })

    df = pd.DataFrame(rows)
    return df


def load_or_generate_traffic(
    base_aadt: float = 50000.0,
    ev_penetration: float = 0.15,
    cache_dir: str = "data/cache",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Load traffic profile from cache, generating if not present."""
    from pathlib import Path

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    filename = f"autobahn_traffic_aadt{int(base_aadt)}_ev{ev_penetration:.2f}.csv"
    filepath = cache_path / filename

    if filepath.exists():
        df = pd.read_csv(filepath)
        if len(df) >= 8760:
            return df

    df = generate_autobahn_traffic_profile(base_aadt, ev_penetration, seed)
    df.to_csv(filepath, index=False)
    return df
