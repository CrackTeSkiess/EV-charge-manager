"""
Rule-Based Energy Management Baselines
=======================================
Provides three deterministic baseline energy managers for comparison against
the RL-trained micro-agent.  All implement a common ``step()`` interface so
they can be swapped directly into the comparison scripts.

Baselines
---------
NaiveBaseline
    Always charge battery from renewables; never explicitly discharge.
    Discharge only when grid demand exceeds renewable supply.

TOUBaseline
    Time-of-Use heuristic: charge battery during off-peak hours,
    discharge during peak hours.

MPCBaseline
    Simplified Model Predictive Control: look ahead N hours and decide
    whether to charge or discharge based on whether the current price is
    below the average of the lookahead window.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional


class _BaselineAgent:
    """Minimal shared interface for all rule-based agents."""

    def step(
        self,
        hour: int,
        demand_kw: float,
        renewable_kw: float,
        battery_soc: float,
        battery_kwh: float,
        battery_kw: float,
    ) -> Dict[str, float]:
        """
        Decide dispatch for one time-step.

        Parameters
        ----------
        hour : int
            Hour of the day (0-23).
        demand_kw : float
            Power demand at this station (kW).
        renewable_kw : float
            Available renewable power (solar + wind, kW).
        battery_soc : float
            Current battery state-of-charge (0–1).
        battery_kwh : float
            Battery capacity (kWh).
        battery_kw : float
            Max charge/discharge rate (kW).

        Returns
        -------
        dict with:
            grid_power       – power drawn from grid (kW)
            battery_change_kwh – signed kWh change to battery
                                 (positive = charge, negative = discharge)
            renewable_used   – renewable power actually used (kW)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# NaiveBaseline
# ---------------------------------------------------------------------------

class NaiveBaseline(_BaselineAgent):
    """
    Naive always-on baseline.

    Dispatch priority:
      1. Use all available renewables first.
      2. Discharge battery if there is unmet demand and SOC > min_soc.
      3. Draw remaining demand from grid.
      4. Charge battery with excess renewables if SOC < max_soc.

    No price awareness — this mirrors the project's original "baseline"
    strategy and serves as the weakest comparison point.
    """

    def __init__(
        self,
        pricing_schedule=None,
        min_soc: float = 0.10,
        max_soc: float = 0.95,
        efficiency: float = 0.90,
    ):
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.efficiency = efficiency

    def step(
        self,
        hour: int,
        demand_kw: float,
        renewable_kw: float,
        battery_soc: float,
        battery_kwh: float,
        battery_kw: float,
    ) -> Dict[str, float]:
        # 1. Renewable first
        renewable_used = min(renewable_kw, demand_kw)
        unmet = demand_kw - renewable_used
        excess_renewable = max(0.0, renewable_kw - demand_kw)

        battery_change_kwh = 0.0
        grid_power = 0.0

        # 2. Discharge battery to cover unmet demand
        if unmet > 0 and battery_soc > self.min_soc:
            available_discharge_kw = min(
                battery_kw,
                (battery_soc - self.min_soc) * battery_kwh,
            )
            discharged_kw = min(unmet, available_discharge_kw)
            battery_change_kwh = -discharged_kw  # kWh per 1-hour step
            unmet -= discharged_kw

        # 3. Grid covers the rest
        grid_power = max(0.0, unmet)

        # 4. Charge battery with excess renewables
        if excess_renewable > 0 and battery_soc < self.max_soc:
            available_charge_kw = min(
                battery_kw,
                (self.max_soc - battery_soc) * battery_kwh,
            )
            charge_kw = min(excess_renewable, available_charge_kw)
            battery_change_kwh += charge_kw * self.efficiency

        return {
            "grid_power": grid_power,
            "battery_change_kwh": battery_change_kwh,
            "renewable_used": renewable_used,
        }


# ---------------------------------------------------------------------------
# TOUBaseline
# ---------------------------------------------------------------------------

class TOUBaseline(_BaselineAgent):
    """
    Time-of-Use heuristic baseline.

    Charging schedule (hardcoded to match GridPricingSchedule defaults):
      - Off-peak  (23:00–07:00, $0.08/kWh): charge battery to max_soc
      - Peak      (17:00–21:00, $0.35/kWh): discharge battery down to min_soc
      - Shoulder  (07:00–17:00, $0.15/kWh): only charge with excess renewables

    This is the obvious hand-coded rule that a knowledgeable operator would
    apply without any ML.
    """

    def __init__(
        self,
        pricing_schedule=None,
        off_peak_hours: tuple = (23, 7),   # 23:00–07:00
        peak_hours: tuple = (17, 21),       # 17:00–21:00
        min_soc: float = 0.10,
        max_soc: float = 0.95,
        efficiency: float = 0.90,
    ):
        self.off_peak_hours = off_peak_hours
        self.peak_hours = peak_hours
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.efficiency = efficiency

    def _in_range(self, hour: int, range_tuple: tuple) -> bool:
        start, end = range_tuple
        if start > end:  # overnight wrap
            return hour >= start or hour < end
        return start <= hour < end

    def step(
        self,
        hour: int,
        demand_kw: float,
        renewable_kw: float,
        battery_soc: float,
        battery_kwh: float,
        battery_kw: float,
    ) -> Dict[str, float]:
        # Always use renewables first
        renewable_used = min(renewable_kw, demand_kw)
        unmet = demand_kw - renewable_used
        excess_renewable = max(0.0, renewable_kw - demand_kw)

        battery_change_kwh = 0.0
        grid_power = 0.0

        if self._in_range(hour, self.peak_hours):
            # Peak: discharge battery to grid to earn arbitrage / reduce grid draw
            if battery_soc > self.min_soc:
                available_discharge_kw = min(
                    battery_kw,
                    (battery_soc - self.min_soc) * battery_kwh,
                )
                discharged_kw = min(unmet + battery_kw * 0.5, available_discharge_kw)
                battery_change_kwh = -discharged_kw
                unmet = max(0.0, unmet - discharged_kw)

        elif self._in_range(hour, self.off_peak_hours):
            # Off-peak: charge battery aggressively with grid power
            if battery_soc < self.max_soc:
                available_charge_kw = min(
                    battery_kw,
                    (self.max_soc - battery_soc) * battery_kwh,
                )
                # Charge with grid during off-peak (cheap)
                charge_from_grid = min(battery_kw, available_charge_kw)
                battery_change_kwh = charge_from_grid * self.efficiency
                grid_power += charge_from_grid  # draw extra from grid to charge

        # Grid covers remaining unmet demand
        grid_power += max(0.0, unmet)

        # Charge with excess renewables (any time, if capacity allows)
        if excess_renewable > 0 and battery_soc < self.max_soc:
            available_charge_kw = min(
                battery_kw,
                (self.max_soc - battery_soc) * battery_kwh,
            )
            charge_kw = min(excess_renewable, available_charge_kw)
            battery_change_kwh += charge_kw * self.efficiency

        return {
            "grid_power": grid_power,
            "battery_change_kwh": battery_change_kwh,
            "renewable_used": renewable_used,
        }


# ---------------------------------------------------------------------------
# MPCBaseline
# ---------------------------------------------------------------------------

class MPCBaseline(_BaselineAgent):
    """
    Simplified Model Predictive Control (MPC) baseline.

    At each hour h, the agent looks ahead ``lookahead_hours`` time steps
    using the known TOU pricing schedule.  If the average future price is
    higher than the current price, the battery is charged now (buy cheap,
    sell later).  If the average future price is lower, the battery is
    discharged now (sell expensive before the price drops).

    This baseline is more sophisticated than TOU because it dynamically
    adjusts the charge/discharge decision based on the price gradient rather
    than fixed clock-based rules.
    """

    def __init__(
        self,
        pricing_schedule=None,
        lookahead_hours: int = 3,
        min_soc: float = 0.10,
        max_soc: float = 0.95,
        efficiency: float = 0.90,
    ):
        self.lookahead_hours = lookahead_hours
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.efficiency = efficiency

        # Build a 24-hour price vector from the pricing schedule
        if pricing_schedule is not None:
            self._prices = [pricing_schedule.get_price(h) for h in range(24)]
        else:
            # Default TOU prices matching GridPricingSchedule defaults
            default_prices = []
            for h in range(24):
                if h >= 23 or h < 7:
                    default_prices.append(0.08)   # off-peak
                elif 17 <= h < 21:
                    default_prices.append(0.35)   # peak
                else:
                    default_prices.append(0.15)   # shoulder
            self._prices = default_prices

    def step(
        self,
        hour: int,
        demand_kw: float,
        renewable_kw: float,
        battery_soc: float,
        battery_kwh: float,
        battery_kw: float,
    ) -> Dict[str, float]:
        # Always use renewables first
        renewable_used = min(renewable_kw, demand_kw)
        unmet = demand_kw - renewable_used
        excess_renewable = max(0.0, renewable_kw - demand_kw)

        battery_change_kwh = 0.0
        grid_power = 0.0

        # MPC decision: compare current price with average of lookahead window
        current_price = self._prices[hour % 24]
        future_prices = [
            self._prices[(hour + k) % 24] for k in range(1, self.lookahead_hours + 1)
        ]
        avg_future_price = float(np.mean(future_prices))

        if current_price < avg_future_price:
            # Prices will rise: charge battery now (buy cheap)
            if battery_soc < self.max_soc:
                available_charge_kw = min(
                    battery_kw,
                    (self.max_soc - battery_soc) * battery_kwh,
                )
                charge_kw = min(battery_kw, available_charge_kw)
                battery_change_kwh = charge_kw * self.efficiency
                grid_power += charge_kw  # draw from grid to charge
        elif current_price > avg_future_price:
            # Prices will fall: discharge battery now (sell expensive)
            if battery_soc > self.min_soc:
                available_discharge_kw = min(
                    battery_kw,
                    (battery_soc - self.min_soc) * battery_kwh,
                )
                discharged_kw = min(unmet + available_discharge_kw * 0.5,
                                    available_discharge_kw)
                battery_change_kwh = -discharged_kw
                unmet = max(0.0, unmet - discharged_kw)

        # Grid covers remaining unmet demand
        grid_power += max(0.0, unmet)

        # Always charge with excess renewables when SOC allows
        if excess_renewable > 0 and battery_soc < self.max_soc:
            available_charge_kw = min(
                battery_kw,
                (self.max_soc - battery_soc) * battery_kwh,
            )
            charge_kw = min(excess_renewable, available_charge_kw)
            battery_change_kwh += charge_kw * self.efficiency

        return {
            "grid_power": grid_power,
            "battery_change_kwh": battery_change_kwh,
            "renewable_used": renewable_used,
        }
