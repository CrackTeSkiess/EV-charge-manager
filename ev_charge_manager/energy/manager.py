"""
Energy Manager Module with Priority: Solar/Wind -> Battery -> Grid
Grid is only used when battery is empty.
Excess renewable energy charges the battery.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Sequence
from enum import Enum, auto
from datetime import datetime, timedelta
import random
import math
import numpy as np

# HierarchicalEnergyManager is imported lazily inside _create_hierarchical_manager
# to avoid a circular import (HierarchicalEnergyManager imports from this module).

# ---------------------------------------------------------------------------
# Shared charger infrastructure constants
# ---------------------------------------------------------------------------
# These must be used consistently across training (environment.py), SUMO
# validation (traci_bridge.py, network_generator.py) and internal comparison
# (validate_sumo.py, validate_real_world.py) so that demand magnitudes
# seen during training match those encountered at validation time.
CHARGER_RATED_POWER_KW: float = 150.0
"""DC fast charger rated power (kW).  Typical Level-3 highway charger."""

CHARGER_AVG_DRAW_FACTOR: float = 0.85
"""Average fraction of rated power actually drawn by a vehicle.

Accounts for tapering at high SOC, connector losses, and vehicles that
are nearly full.  effective_power = CHARGER_RATED_POWER_KW * CHARGER_AVG_DRAW_FACTOR
"""


class EnergySourceType(Enum):
    """Types of energy sources available."""
    GRID = auto()
    SOLAR = auto()
    WIND = auto()
    BATTERY = auto()


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

class EnergySource:
    """
    Represents a single energy source at runtime.
    Tracks current state and availability.
    """
    
    def __init__(self, config: EnergySourceConfig, weather_provider=None):
        self.config = config
        self.weather_provider = weather_provider
        self.source_type = self._get_source_type()
        
        # Current state
        self.current_output_kw: float = 0.0
        self.available_power_kw: float = 0.0
        
        # Battery-specific state (only used if battery)
        self.current_soc: float = 0.0
        self.stored_energy_kwh: float = 0.0
        
        # Initialize battery state if applicable
        if isinstance(config, BatteryStorageConfig):
            self.current_soc = config.initial_soc
            self.stored_energy_kwh = config.capacity_kwh * config.initial_soc
    
    def _get_source_type(self) -> EnergySourceType:
        """Determine source type from config."""
        if isinstance(self.config, SolarSourceConfig):
            return EnergySourceType.SOLAR
        elif isinstance(self.config, WindSourceConfig):
            return EnergySourceType.WIND
        elif isinstance(self.config, BatteryStorageConfig):
            return EnergySourceType.BATTERY
        else:
            return EnergySourceType.GRID
    
    def update_availability(self, timestamp: datetime, weather_factor: float = 1.0):
        """Update available power based on time and weather."""
        hour = timestamp.hour + timestamp.minute / 60.0
        
        if isinstance(self.config, SolarSourceConfig):
            if self.weather_provider is not None:
                # Real data mode: use actual GHI from weather provider
                self.available_power_kw = self.weather_provider.get_solar_output_kw(
                    timestamp, self.config.peak_power_kw
                )
            else:
                # Synthetic mode: parabolic curve during daylight
                if self.config.sunrise_hour <= hour <= self.config.sunset_hour:
                    # Parabolic curve peaking at peak_hour
                    time_from_peak = abs(hour - self.config.peak_hour)
                    day_length = self.config.sunset_hour - self.config.sunrise_hour
                    peak_offset = abs(self.config.peak_hour - self.config.sunrise_hour)

                    # Normalized parabolic curve
                    if time_from_peak <= peak_offset:
                        availability = 1 - (time_from_peak / peak_offset) ** 2
                    else:
                        availability = 0
                else:
                    availability = 0

                self.available_power_kw = self.config.peak_power_kw * availability * weather_factor
            
        elif isinstance(self.config, WindSourceConfig):
            if self.weather_provider is not None:
                # Real data mode: use actual wind speed from weather provider
                self.available_power_kw = self.weather_provider.get_wind_output_kw(
                    timestamp, self.config.base_power_kw
                )
            else:
                # Synthetic mode: variable but can be day or night
                base = (self.config.min_power_kw + self.config.max_power_kw) / 2
                variation = (self.config.max_power_kw - self.config.min_power_kw) / 2

                # Daily pattern + random noise
                daily_factor = 0.5 + 0.5 * math.sin(2 * math.pi * hour / 24)
                noise = random.uniform(-self.config.variability, self.config.variability)

                power = base + variation * (daily_factor + noise)
                self.available_power_kw = float(np.clip(power, self.config.min_power_kw, self.config.max_power_kw))
            
        elif isinstance(self.config, GridSourceConfig):
            # Grid: always available at max power
            self.available_power_kw = self.config.max_power_kw
            
        elif isinstance(self.config, BatteryStorageConfig):
            # Battery: available power depends on SOC and charge/discharge limits
            max_discharge = min(
                self.config.max_discharge_rate_kw,
                (self.current_soc - self.config.min_soc) * self.config.capacity_kwh
            )
            self.available_power_kw = max(0, max_discharge)
    
    def request_energy(self, power_kw: float, duration_hours: float = 1.0) -> float:
        """
        Request energy from this source.
        Returns actual power delivered.
        """
        if self.source_type == EnergySourceType.BATTERY:
            # Check if this is actually a battery config
            if not isinstance(self.config, BatteryStorageConfig):
                return 0.0
            
            # Discharge battery
            available = self.available_power_kw
            actual = min(power_kw, available)
            
            if actual > 0:
                # Reduce stored energy with efficiency loss
                energy_out = actual * duration_hours
                efficiency = self.config.round_trip_efficiency
                self.stored_energy_kwh -= energy_out / efficiency
                # Clamp to valid range
                self.stored_energy_kwh = max(0, self.stored_energy_kwh)
                # Update SOC
                self.current_soc = self.stored_energy_kwh / self.config.capacity_kwh
                # Clamp SOC
                self.current_soc = float(np.clip(self.current_soc, self.config.min_soc, self.config.max_soc))
            
            self.current_output_kw = actual
            return actual
        
        else:
            # Grid, Solar, Wind: just limit by available power
            actual = min(power_kw, self.available_power_kw)
            self.current_output_kw = actual
            return actual
    
    def charge_battery(self, power_kw: float, duration_hours: float = 1.0) -> float:
        """
        Charge the battery (only for battery sources).
        Returns actual power accepted.
        """
        # Must be a battery source
        if self.source_type != EnergySourceType.BATTERY:
            return 0.0
        
        # Must have battery config
        if not isinstance(self.config, BatteryStorageConfig):
            return 0.0
        
        # Calculate available charging capacity
        max_charge_energy = (self.config.max_soc - self.current_soc) * self.config.capacity_kwh
        max_charge_power = min(
            self.config.max_charge_rate_kw,
            max_charge_energy / duration_hours if duration_hours > 0 else float('inf')
        )
        
        actual_power = min(power_kw, max_charge_power)
        
        if actual_power > 0:
            # Add energy with efficiency
            efficiency = self.config.round_trip_efficiency
            energy_in = actual_power * duration_hours * efficiency
            self.stored_energy_kwh += energy_in
            # Clamp to valid range
            max_storage = self.config.capacity_kwh * self.config.max_soc
            self.stored_energy_kwh = min(self.stored_energy_kwh, max_storage)
            # Update SOC
            self.current_soc = self.stored_energy_kwh / self.config.capacity_kwh
            # Clamp SOC
            self.current_soc = float(np.clip(self.current_soc, self.config.min_soc, self.config.max_soc))
        
        return actual_power
    
    def is_battery_empty(self) -> bool:
        """Check if battery is at minimum SOC."""
        if not isinstance(self.config, BatteryStorageConfig):
            return False
        return self.current_soc <= self.config.min_soc + 0.01
    
    def is_battery_full(self) -> bool:
        """Check if battery is at maximum SOC."""
        if not isinstance(self.config, BatteryStorageConfig):
            return False
        return self.current_soc >= self.config.max_soc - 0.01
    
    def get_battery_soc(self) -> float:
        """Get battery state of charge."""
        if not isinstance(self.config, BatteryStorageConfig):
            return 0.0
        return self.current_soc


class EnergyManager:
    """
    Manages energy supply with strict priority:
    1. Solar + Wind (renewables) - used first
    2. Battery - used when renewables insufficient
    3. Grid - ONLY used when battery is empty
    
    Excess renewable energy charges the battery.
    """
    
    def __init__(self, manager_id: Optional[str] = None, 
                 config: Optional[EnergyManagerConfig] = None):
        self.id = manager_id or str(uuid.uuid4())[:8]
        self.config = config or EnergyManagerConfig()
        
        # Initialize energy sources from config
        self.sources: Dict[EnergySourceType, List[EnergySource]] = {
            EnergySourceType.GRID: [],
            EnergySourceType.SOLAR: [],
            EnergySourceType.WIND: [],
            EnergySourceType.BATTERY: [],
        }
        
        self._initialize_sources()
        
        # Current state
        self.current_timestamp: Optional[datetime] = None
        self.total_renewable_available_kw: float = 0.0
        self.total_battery_available_kw: float = 0.0
        self.total_grid_available_kw: float = 0.0
        
        # Statistics
        self.stats = {
            'total_energy_requested_kwh': 0.0,
            'total_energy_delivered_kwh': 0.0,
            'renewable_used_kwh': 0.0,
            'battery_discharged_kwh': 0.0,
            'battery_charged_kwh': 0.0,
            'grid_used_kwh': 0.0,
            'shortage_events': 0,
            'total_shortage_kwh': 0.0,
            'renewable_wasted_kwh': 0.0,
        }
        
        # Tracking
        self.power_shortage_active: bool = False
        self.grid_was_used: bool = False
    
    def _initialize_sources(self):
        """Create energy sources from configuration."""
        for source_config in self.config.source_configs:
            source = EnergySource(source_config)
            self.sources[source.source_type].append(source)
            
    @property
    def is_unlimited(self) -> bool:
        """True if no energy management configured (backwards compatibility)."""
        return len(self.config.source_configs) == 0
    
    def step(self, timestamp: datetime, time_step_minutes: float = 1.0) -> Dict:
        """
        Update energy sources and manage battery charging from excess renewables.
        """
        self.current_timestamp = timestamp
        dt_hours = time_step_minutes / 60.0
        
        # Update all source availabilities
        for source_list in self.sources.values():
            for source in source_list:
                source.update_availability(timestamp)
        
        # Calculate total available power by type
        self.total_renewable_available_kw = sum(
            s.available_power_kw for s in self.sources[EnergySourceType.SOLAR]
        ) + sum(
            s.available_power_kw for s in self.sources[EnergySourceType.WIND]
        )
        
        self.total_battery_available_kw = sum(
            s.available_power_kw for s in self.sources[EnergySourceType.BATTERY]
        )
        
        self.total_grid_available_kw = sum(
            s.available_power_kw for s in self.sources[EnergySourceType.GRID]
        )
        
        # Charge battery from excess renewables BEFORE any demand
        self._charge_battery_from_excess_renewables(dt_hours)
        
        # Reset tracking
        self.power_shortage_active = False
        self.grid_was_used = False
        
        return self.get_energy_state()
    
    def _create_hierarchical_manager(
        self,
        config: EnergyManagerConfig,
        station_id: str,
    ):
        """Create hierarchical manager with RL agent."""
        # Lazy import to avoid circular dependency with HierarchicalEnergyManager
        from ev_charge_manager.energy.hierarchical_manager import HierarchicalEnergyManager
        from ev_charge_manager.energy.agent import GridPricingSchedule

        # Create time-of-use pricing
        pricing = GridPricingSchedule(
            off_peak_price=0.08,
            shoulder_price=0.15,
            peak_price=0.35,
        )
        
        return HierarchicalEnergyManager(
            manager_id=station_id,
            config=config,
            pricing_schedule=pricing,
            enable_rl_agent=True,
            device="cpu",  # Or cuda if available
        )
    
    def _charge_battery_from_excess_renewables(self, duration_hours: float):
        """
        Charge battery using excess renewable energy.
        This happens before any demand is served.

        Power drawn from renewables is debited from available_power_kw on each
        renewable source so the same electrons cannot be double-spent when
        chargers subsequently call request_energy().
        """
        if self.total_renewable_available_kw <= 0:
            return

        # Use up to 30% of total renewable capacity for battery pre-charging
        charge_budget_kw = self.total_renewable_available_kw * 0.3

        for battery in self.sources[EnergySourceType.BATTERY]:
            if battery.is_battery_full() or charge_budget_kw <= 0:
                continue

            actual_charge_kw = battery.charge_battery(charge_budget_kw, duration_hours)

            if actual_charge_kw > 0:
                self.stats['battery_charged_kwh'] += actual_charge_kw * duration_hours

                # Debit consumed power from renewable sources (solar first, then wind)
                # so chargers cannot double-spend it via request_energy().
                remaining_to_debit = actual_charge_kw
                for solar in self.sources[EnergySourceType.SOLAR]:
                    if remaining_to_debit <= 0:
                        break
                    debit = min(remaining_to_debit, solar.available_power_kw)
                    solar.available_power_kw -= debit
                    remaining_to_debit -= debit
                for wind in self.sources[EnergySourceType.WIND]:
                    if remaining_to_debit <= 0:
                        break
                    debit = min(remaining_to_debit, wind.available_power_kw)
                    wind.available_power_kw -= debit
                    remaining_to_debit -= debit

                # Keep the cached totals consistent
                self.total_renewable_available_kw = max(
                    0.0, self.total_renewable_available_kw - actual_charge_kw
                )
                charge_budget_kw -= actual_charge_kw
    
    def request_energy(self, power_kw: float, charger_id: str,
                      duration_hours: float = 1.0) -> float:
        """
        Request energy with strict priority:
        1. Solar + Wind (renewables)
        2. Battery (if renewables insufficient)
        3. Grid (ONLY if battery is empty)
        """
        remaining_need = power_kw
        total_allocated = 0.0
        
        # Track what we use
        renewable_used = 0.0
        battery_used = 0.0
        grid_used = 0.0
        
        # STEP 1: Use renewables first (solar + wind)
        if remaining_need > 0:
            # Try solar first
            for solar in self.sources[EnergySourceType.SOLAR]:
                if remaining_need <= 0:
                    break
                available = solar.available_power_kw - solar.current_output_kw
                if available > 0:
                    request = min(remaining_need, available)
                    actual = solar.request_energy(request, duration_hours)
                    renewable_used += actual
                    remaining_need -= actual
                    total_allocated += actual
            
            # Then wind
            for wind in self.sources[EnergySourceType.WIND]:
                if remaining_need <= 0:
                    break
                available = wind.available_power_kw - wind.current_output_kw
                if available > 0:
                    request = min(remaining_need, available)
                    actual = wind.request_energy(request, duration_hours)
                    renewable_used += actual
                    remaining_need -= actual
                    total_allocated += actual
        
        # STEP 2: Use battery if renewables insufficient
        if remaining_need > 0:
            for battery in self.sources[EnergySourceType.BATTERY]:
                if remaining_need <= 0:
                    break
                
                # Check if battery has energy
                if battery.is_battery_empty():
                    continue
                
                available = battery.available_power_kw
                if available > 0:
                    request = min(remaining_need, available)
                    actual = battery.request_energy(request, duration_hours)
                    battery_used += actual
                    remaining_need -= actual
                    total_allocated += actual
        
        # STEP 3: Use grid ONLY if battery is empty and we still need power
        if remaining_need > 0:
            # Check if all batteries are empty
            batteries = self.sources[EnergySourceType.BATTERY]
            batteries_empty = all(b.is_battery_empty() for b in batteries) if batteries else True
            
            # Only use grid if batteries are empty or no batteries exist
            if batteries_empty or not batteries:
                for grid in self.sources[EnergySourceType.GRID]:
                    if remaining_need <= 0:
                        break
                    
                    available = grid.available_power_kw - grid.current_output_kw
                    if available > 0:
                        request = min(remaining_need, available)
                        actual = grid.request_energy(request, duration_hours)
                        grid_used += actual
                        remaining_need -= actual
                        total_allocated += actual
                        self.grid_was_used = True
        
        # Update statistics
        energy_delivered = total_allocated * duration_hours
        self.stats['total_energy_delivered_kwh'] += energy_delivered
        self.stats['renewable_used_kwh'] += renewable_used * duration_hours
        self.stats['battery_discharged_kwh'] += battery_used * duration_hours
        self.stats['grid_used_kwh'] += grid_used * duration_hours
        
        # Track shortage
        if remaining_need > 0.01 * power_kw:
            self.stats['shortage_events'] += 1
            self.stats['total_shortage_kwh'] += remaining_need * duration_hours
            self.power_shortage_active = True
        
        return total_allocated
    
    def get_available_power(self) -> float:
        """Get total currently available power from all sources."""
        total = 0.0
        for source_list in self.sources.values():
            for source in source_list:
                total += source.available_power_kw
        return total
    
    def get_renewable_power(self) -> float:
        """Get currently available renewable power."""
        total = 0.0
        for source in self.sources[EnergySourceType.SOLAR]:
            total += source.available_power_kw
        for source in self.sources[EnergySourceType.WIND]:
            total += source.available_power_kw
        return total
    
    def get_battery_soc(self) -> float:
        """Get average battery state of charge."""
        batteries = self.sources[EnergySourceType.BATTERY]
        if not batteries:
            return 0.0
        return np.mean([b.get_battery_soc() for b in batteries]).item()
    
    def is_battery_empty(self) -> bool:
        """Check if all batteries are empty."""
        batteries = self.sources[EnergySourceType.BATTERY]
        if not batteries:
            return True
        return all(b.is_battery_empty() for b in batteries)
    
    def get_energy_state(self) -> Dict:
        """Get current energy system state for monitoring."""
        battery_soc = self.get_battery_soc()
        
        total_renewable = self.get_renewable_power()
        
        # Calculate total capacity
        total_capacity = 0.0
        for source_list in self.sources.values():
            for s in source_list:
                if isinstance(s.config, SolarSourceConfig):
                    total_capacity += s.config.peak_power_kw
                elif isinstance(s.config, WindSourceConfig):
                    total_capacity += s.config.max_power_kw
                elif isinstance(s.config, BatteryStorageConfig):
                    total_capacity += s.config.max_discharge_rate_kw
                elif isinstance(s.config, GridSourceConfig):
                    total_capacity += s.config.max_power_kw
        
        return {
            'manager_id': self.id,
            'timestamp': self.current_timestamp,
            'total_available_kw': self.get_available_power(),
            'renewable_available_kw': total_renewable,
            'battery_available_kw': self.total_battery_available_kw,
            'grid_available_kw': self.total_grid_available_kw,
            'battery_soc_percent': battery_soc * 100,
            'renewable_fraction': total_renewable / max(1, total_capacity),
            'shortage_active': self.power_shortage_active,
            'grid_was_used': self.grid_was_used,
            'battery_empty': self.is_battery_empty(),
        }
    
    def get_statistics(self) -> Dict:
        """Get cumulative statistics."""
        total_used = (
            self.stats['renewable_used_kwh'] + 
            self.stats['battery_discharged_kwh'] + 
            self.stats['grid_used_kwh']
        )
        
        stats = self.stats.copy()
        if total_used > 0:
            stats['renewable_fraction'] = self.stats['renewable_used_kwh'] / total_used
            stats['grid_fraction'] = self.stats['grid_used_kwh'] / total_used
            stats['battery_fraction'] = self.stats['battery_discharged_kwh'] / total_used
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics."""
        for key in self.stats:
            self.stats[key] = 0.0
    
    def __repr__(self) -> str:
        renewable = self.get_renewable_power()
        battery_soc = self.get_battery_soc()
        return (f"EnergyManager({self.id}, "
                f"renewable={renewable:.1f}kW, "
                f"battery_soc={battery_soc*100:.1f}%, "
                f"grid_used={self.grid_was_used}, "
                f"shortage={self.power_shortage_active})")