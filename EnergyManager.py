"""
Energy Manager Module
Manages energy supply for charging areas with configurable sources (grid, solar, wind, battery).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class EnergySourceConfig:
    """Base configuration for an energy source."""
    source_type: str = "grid"


@dataclass
class GridSourceConfig(EnergySourceConfig):
    """Grid connection with configurable max power."""
    source_type: str = "grid"
    max_power_kw: float = float('inf')


@dataclass
class SolarSourceConfig(EnergySourceConfig):
    """Solar panels with time-of-day output curve."""
    source_type: str = "solar"
    peak_power_kw: float = 100.0
    sunrise_hour: float = 6.0
    sunset_hour: float = 20.0
    peak_hour: float = 13.0


@dataclass
class WindSourceConfig(EnergySourceConfig):
    """Wind turbines with stochastic output."""
    source_type: str = "wind"
    base_power_kw: float = 50.0
    min_power_kw: float = 10.0
    max_power_kw: float = 100.0
    variability: float = 0.3


@dataclass
class BatteryStorageConfig(EnergySourceConfig):
    """Battery storage with charge/discharge capabilities."""
    source_type: str = "battery"
    capacity_kwh: float = 500.0
    max_charge_rate_kw: float = 100.0
    max_discharge_rate_kw: float = 100.0
    initial_soc: float = 0.5
    min_soc: float = 0.1
    max_soc: float = 0.95
    round_trip_efficiency: float = 0.90


@dataclass
class EnergyManagerConfig:
    """Configuration for the energy manager of a single charging area."""
    source_configs: List[EnergySourceConfig] = field(default_factory=list)


# =============================================================================
# ENERGY SOURCE CLASSES
# =============================================================================

class EnergySource:
    """Base energy source providing power to a charging area."""

    def __init__(self, config: EnergySourceConfig):
        self.config = config
        self.current_output_kw: float = 0.0

    def step(self, timestamp: datetime, time_step_minutes: float) -> None:
        """Update source output for current timestep."""
        pass

    def get_available_power_kw(self) -> float:
        return self.current_output_kw


class GridSource(EnergySource):
    """Grid connection with configurable max power."""

    def __init__(self, config: GridSourceConfig):
        super().__init__(config)
        self.current_output_kw = config.max_power_kw

    def step(self, timestamp: datetime, time_step_minutes: float) -> None:
        self.current_output_kw = self.config.max_power_kw  # type: ignore[union-attr]


class SolarSource(EnergySource):
    """Solar panel output following time-of-day Gaussian bell curve."""

    def __init__(self, config: SolarSourceConfig):
        super().__init__(config)

    def step(self, timestamp: datetime, time_step_minutes: float) -> None:
        cfg: SolarSourceConfig = self.config  # type: ignore[assignment]
        hour = timestamp.hour + timestamp.minute / 60.0

        if hour < cfg.sunrise_hour or hour > cfg.sunset_hour:
            self.current_output_kw = 0.0
        else:
            width = (cfg.sunset_hour - cfg.sunrise_hour) / 4.0
            self.current_output_kw = cfg.peak_power_kw * math.exp(
                -0.5 * ((hour - cfg.peak_hour) / width) ** 2
            )


class WindSource(EnergySource):
    """Wind power with stochastic variation (random walk with mean reversion)."""

    def __init__(self, config: WindSourceConfig):
        super().__init__(config)
        cfg: WindSourceConfig = config
        self._previous_output = cfg.base_power_kw
        self.current_output_kw = cfg.base_power_kw

    def step(self, timestamp: datetime, time_step_minutes: float) -> None:
        cfg: WindSourceConfig = self.config  # type: ignore[assignment]
        noise = random.gauss(0, cfg.variability * cfg.base_power_kw)
        mean_reversion = 0.1 * (cfg.base_power_kw - self._previous_output)
        new_output = self._previous_output + noise + mean_reversion
        self.current_output_kw = max(cfg.min_power_kw, min(cfg.max_power_kw, new_output))
        self._previous_output = self.current_output_kw


class BatteryStorage(EnergySource):
    """
    Battery storage that charges from surplus and discharges during deficit.
    Managed by EnergyManager — does not produce power on its own.
    """

    def __init__(self, config: BatteryStorageConfig):
        super().__init__(config)
        cfg: BatteryStorageConfig = config
        self.current_soc: float = cfg.initial_soc
        self.current_kwh: float = cfg.capacity_kwh * cfg.initial_soc
        self.current_output_kw = 0.0

    def step(self, timestamp: datetime, time_step_minutes: float) -> None:
        # Battery output is managed by EnergyManager.request_energy()
        pass

    def can_discharge(self) -> float:
        """Max discharge power available right now (kW)."""
        cfg: BatteryStorageConfig = self.config  # type: ignore[assignment]
        if self.current_soc <= cfg.min_soc:
            return 0.0
        return cfg.max_discharge_rate_kw

    def can_charge(self) -> float:
        """Max charge power acceptable right now (kW)."""
        cfg: BatteryStorageConfig = self.config  # type: ignore[assignment]
        if self.current_soc >= cfg.max_soc:
            return 0.0
        return cfg.max_charge_rate_kw

    def discharge(self, energy_kwh: float) -> float:
        """Discharge energy. Returns actual energy delivered (after efficiency loss)."""
        cfg: BatteryStorageConfig = self.config  # type: ignore[assignment]
        available = self.current_kwh - (cfg.capacity_kwh * cfg.min_soc)
        actual = min(energy_kwh, max(0.0, available))
        self.current_kwh -= actual
        self.current_soc = self.current_kwh / cfg.capacity_kwh
        return actual * math.sqrt(cfg.round_trip_efficiency)

    def charge(self, energy_kwh: float) -> float:
        """Charge from surplus energy. Returns actual energy accepted."""
        cfg: BatteryStorageConfig = self.config  # type: ignore[assignment]
        space = (cfg.capacity_kwh * cfg.max_soc) - self.current_kwh
        actual = min(energy_kwh * math.sqrt(cfg.round_trip_efficiency), max(0.0, space))
        self.current_kwh += actual
        self.current_soc = self.current_kwh / cfg.capacity_kwh
        return actual


# =============================================================================
# ENERGY MANAGER
# =============================================================================

class EnergyManager:
    """
    Manages energy supply for a single ChargingArea.

    Each simulation step:
    1. Updates all source outputs based on timestamp
    2. When requested, calculates total available power
    3. Manages battery storage (charge surplus, discharge deficit)
    """

    def __init__(self, config: EnergyManagerConfig):
        self.config = config
        self.sources: List[EnergySource] = []
        self.battery: Optional[BatteryStorage] = None
        self._initialize_sources()

        # Current state
        self.total_available_kw: float = 0.0
        self.total_demand_kw: float = 0.0
        self.curtailment_kw: float = 0.0
        self.battery_soc: float = self.battery.current_soc if self.battery else 0.0

        # Cumulative statistics
        self.stats = {
            'total_curtailment_events': 0,
            'total_energy_curtailed_kwh': 0.0,
            'total_battery_discharged_kwh': 0.0,
            'total_battery_charged_kwh': 0.0,
            'total_surplus_kwh': 0.0,
        }

    def _initialize_sources(self) -> None:
        """Create source instances from config."""
        for cfg in self.config.source_configs:
            if isinstance(cfg, BatteryStorageConfig):
                storage = BatteryStorage(cfg)
                self.battery = storage
                self.sources.append(storage)
            elif isinstance(cfg, GridSourceConfig):
                self.sources.append(GridSource(cfg))
            elif isinstance(cfg, SolarSourceConfig):
                self.sources.append(SolarSource(cfg))
            elif isinstance(cfg, WindSourceConfig):
                self.sources.append(WindSource(cfg))

    @property
    def is_unlimited(self) -> bool:
        """True if no energy management configured (backwards compatibility)."""
        return len(self.config.source_configs) == 0

    def step(self, timestamp: datetime, time_step_minutes: float) -> None:
        """Update all source outputs for current timestep."""
        for source in self.sources:
            source.step(timestamp, time_step_minutes)

        # Compute total non-battery supply
        self.total_available_kw = sum(
            s.get_available_power_kw() for s in self.sources
            if not isinstance(s, BatteryStorage)
        )
        if self.battery:
            self.battery_soc = self.battery.current_soc

    def request_energy(self, demand_kw: float, time_step_minutes: float) -> float:
        """
        Request energy to power chargers.

        Returns actual available power in kW.

        Logic:
        1. Sum all non-battery source outputs
        2. If surplus: charge battery with excess
        3. If deficit: discharge battery to help meet demand
        4. Return final available power (may be less than demand)
        """
        if self.is_unlimited:
            return demand_kw

        self.total_demand_kw = demand_kw
        supply_kw = self.total_available_kw
        time_step_hours = time_step_minutes / 60.0

        if supply_kw >= demand_kw:
            # Surplus: charge battery with excess
            surplus_kw = supply_kw - demand_kw
            if self.battery and surplus_kw > 0:
                charge_kw = min(surplus_kw, self.battery.can_charge())
                energy_to_store = charge_kw * time_step_hours
                actual_stored = self.battery.charge(energy_to_store)
                self.stats['total_battery_charged_kwh'] += actual_stored
                remaining_surplus = (surplus_kw - charge_kw) * time_step_hours
                self.stats['total_surplus_kwh'] += remaining_surplus

            self.curtailment_kw = 0.0
            return demand_kw
        else:
            # Deficit: try battery discharge
            deficit_kw = demand_kw - supply_kw
            battery_available_kw = 0.0

            if self.battery:
                battery_available_kw = min(deficit_kw, self.battery.can_discharge())
                if battery_available_kw > 0:
                    energy_needed = battery_available_kw * time_step_hours
                    actual_delivered = self.battery.discharge(energy_needed)
                    self.stats['total_battery_discharged_kwh'] += actual_delivered

            actual_available = supply_kw + battery_available_kw
            self.curtailment_kw = max(0.0, demand_kw - actual_available)

            if self.curtailment_kw > 0:
                self.stats['total_curtailment_events'] += 1
                self.stats['total_energy_curtailed_kwh'] += self.curtailment_kw * time_step_hours

            return actual_available

    def get_state(self) -> Dict:
        """Get current energy manager state for KPI tracking and broadcast."""
        return {
            'total_available_kw': self.total_available_kw,
            'total_demand_kw': self.total_demand_kw,
            'curtailment_kw': self.curtailment_kw,
            'battery_soc': self.battery_soc if self.battery else None,
            'source_outputs': {
                type(s).__name__: s.current_output_kw for s in self.sources
            },
            'stats': self.stats.copy()
        }

    def __repr__(self) -> str:
        sources_str = ", ".join(type(s).__name__ for s in self.sources)
        return (f"EnergyManager(sources=[{sources_str}], "
                f"available={self.total_available_kw:.0f}kW, "
                f"curtailment={self.curtailment_kw:.0f}kW)")
