"""
Rule-based energy management baseline for comparison with RL agent.

Implements a deterministic strategy:
1. Renewables first for demand
2. Charge battery from excess renewables
3. Off-peak: charge battery from grid (if SOC < 80%)
4. Peak: discharge battery to offset grid (if SOC > 20%)
5. Grid for remaining shortfall
"""

from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from ev_charge_manager.energy.manager import (
    EnergyManagerConfig,
    SolarSourceConfig,
    WindSourceConfig,
    GridSourceConfig,
    BatteryStorageConfig,
)
from ev_charge_manager.energy.agent import GridPricingSchedule


class RuleBasedEnergyManager:
    """
    Rule-based energy manager with the same interface as HierarchicalEnergyManager.

    Uses a fixed policy:
    - Always use renewables first
    - Charge battery from excess renewables
    - Off-peak (23:00-07:00): charge battery from grid at 50% max rate if SOC < 0.80
    - Peak (17:00-21:00): discharge battery if SOC > 0.20
    - Grid fills any remaining demand
    """

    def __init__(
        self,
        manager_id: str,
        config: EnergyManagerConfig,
        pricing_schedule: Optional[GridPricingSchedule] = None,
        weather_provider=None,
    ):
        self.id = manager_id
        self.config = config
        self.pricing = pricing_schedule or GridPricingSchedule()
        self.weather_provider = weather_provider

        # Parse sources
        self.solar_sources: List[SolarSourceConfig] = []
        self.wind_sources: List[WindSourceConfig] = []
        self.grid_sources: List[GridSourceConfig] = []
        self.battery_config: Optional[BatteryStorageConfig] = None
        self.has_battery = False

        for source_config in config.source_configs:
            if isinstance(source_config, SolarSourceConfig):
                self.solar_sources.append(source_config)
            elif isinstance(source_config, WindSourceConfig):
                self.wind_sources.append(source_config)
            elif isinstance(source_config, GridSourceConfig):
                self.grid_sources.append(source_config)
            elif isinstance(source_config, BatteryStorageConfig):
                self.battery_config = source_config
                self.has_battery = True

        if not self.has_battery:
            self.battery_config = BatteryStorageConfig()

        # Battery state
        self.current_soc: float = self.battery_config.initial_soc

        # Statistics
        self.daily_stats = {
            'grid_cost_total': 0.0,
            'grid_energy_kwh': 0.0,
            'renewable_energy_kwh': 0.0,
            'battery_cycles': 0.0,
            'arbitrage_profit': 0.0,
            'battery_charged_kwh': 0.0,
            'battery_charge_cost': 0.0,
        }
        self.hourly_history: List[Dict] = []

    def _calculate_solar_output(self, timestamp: datetime) -> float:
        """Calculate solar output using real data or synthetic fallback."""
        if self.weather_provider is not None:
            total = 0.0
            for solar in self.solar_sources:
                total += self.weather_provider.get_solar_output_kw(
                    timestamp, solar.peak_power_kw
                )
            return total

        # Synthetic fallback
        hour = timestamp.hour + timestamp.minute / 60.0
        total = 0.0
        for solar in self.solar_sources:
            if solar.sunrise_hour <= hour <= solar.sunset_hour:
                time_from_peak = abs(hour - solar.peak_hour)
                peak_offset = abs(solar.peak_hour - solar.sunrise_hour)
                if time_from_peak <= peak_offset:
                    availability = 1 - (time_from_peak / peak_offset) ** 2
                else:
                    availability = 0
                total += solar.peak_power_kw * availability
        return total

    def _calculate_wind_output(self, timestamp: datetime) -> float:
        """Calculate wind output using real data or synthetic fallback."""
        if self.weather_provider is not None:
            total = 0.0
            for wind in self.wind_sources:
                total += self.weather_provider.get_wind_output_kw(
                    timestamp, wind.base_power_kw
                )
            return total

        # Synthetic fallback
        import math
        import random
        hour = timestamp.hour
        total = 0.0
        for wind in self.wind_sources:
            daily_factor = 0.5 + 0.5 * math.sin(2 * math.pi * hour / 24)
            noise = random.uniform(-wind.variability, wind.variability)
            power = wind.base_power_kw + (wind.max_power_kw - wind.min_power_kw) * 0.5 * (daily_factor + noise)
            total += float(np.clip(power, wind.min_power_kw, wind.max_power_kw))
        return total

    def step(
        self,
        timestamp: datetime,
        demand_kw: float,
        time_step_minutes: float = 60.0,
        training_mode: bool = False,
    ) -> Dict[str, float]:
        """Execute one timestep with rule-based energy management."""
        hour = timestamp.hour
        dt_hours = time_step_minutes / 60.0

        # Get renewable output
        solar_output = self._calculate_solar_output(timestamp)
        wind_output = self._calculate_wind_output(timestamp)
        grid_price = self.pricing.get_price(hour)
        renewable_total = solar_output + wind_output

        # 1. Use renewables for demand
        renewable_to_demand = min(renewable_total, demand_kw)
        remaining_demand = demand_kw - renewable_to_demand
        excess_renewable = renewable_total - renewable_to_demand

        # 2. Battery logic
        battery_power = 0.0  # positive = charge, negative = discharge
        max_charge = self.battery_config.max_charge_rate_kw if self.has_battery else 0
        max_discharge = self.battery_config.max_discharge_rate_kw if self.has_battery else 0

        if self.has_battery:
            # Charge from excess renewables
            if excess_renewable > 0 and self.current_soc < 0.95:
                charge_headroom = (0.95 - self.current_soc) * self.battery_config.capacity_kwh / dt_hours
                charge_rate = min(excess_renewable, max_charge, charge_headroom)
                battery_power += charge_rate
                excess_renewable -= charge_rate

            # Off-peak: charge from grid
            if (hour >= 23 or hour < 7) and self.current_soc < 0.80:
                charge_headroom = (0.80 - self.current_soc) * self.battery_config.capacity_kwh / dt_hours
                grid_charge_rate = min(max_charge * 0.5, charge_headroom)
                grid_charge_rate = max(0, grid_charge_rate - battery_power)  # Don't exceed max
                battery_power += grid_charge_rate

            # Peak: discharge to offset grid
            if 17 <= hour <= 21 and self.current_soc > 0.20 and remaining_demand > 0:
                discharge_headroom = (self.current_soc - 0.20) * self.battery_config.capacity_kwh / dt_hours
                discharge_rate = min(remaining_demand, max_discharge, discharge_headroom)
                battery_power -= discharge_rate
                remaining_demand -= discharge_rate

            # Shoulder: discharge only if renewables insufficient
            if not (17 <= hour <= 21) and not (hour >= 23 or hour < 7):
                if remaining_demand > 0 and self.current_soc > 0.30:
                    discharge_headroom = (self.current_soc - 0.30) * self.battery_config.capacity_kwh / dt_hours
                    discharge_rate = min(remaining_demand * 0.5, max_discharge * 0.3, discharge_headroom)
                    battery_power -= discharge_rate
                    remaining_demand -= discharge_rate

            # Update SOC
            soc_change = battery_power * dt_hours / self.battery_config.capacity_kwh
            self.current_soc = float(np.clip(self.current_soc + soc_change, 0.0, 1.0))

        # 3. Grid for remaining demand
        grid_power = max(0.0, remaining_demand)

        # Build result
        energy_flows = {
            'renewable_to_demand': renewable_to_demand,
            'grid_power': grid_power,
            'battery_power': battery_power,
            'curtailed_renewable': max(0, excess_renewable),
            'total_demand_met': renewable_to_demand + grid_power + max(0, -battery_power),
            'demand_shortfall': max(0, remaining_demand - grid_power),
        }

        # Update statistics
        self._update_stats(energy_flows, grid_price, dt_hours)

        # Record history
        self.hourly_history.append({
            'hour': hour,
            'solar': solar_output,
            'wind': wind_output,
            'demand': demand_kw,
            'grid_price': grid_price,
            'grid_used': grid_power,
            'battery_soc': self.current_soc,
            'reward': 0.0,
            'demand_shortfall': energy_flows.get('demand_shortfall', 0.0),
        })

        return {
            **energy_flows,
            'solar_output': solar_output,
            'wind_output': wind_output,
            'grid_price': grid_price,
            'battery_soc': self.current_soc,
            'reward': 0.0,
        }

    def _update_stats(self, flows: Dict, grid_price: float, dt_hours: float):
        """Update daily statistics."""
        grid_energy = flows['grid_power'] * dt_hours
        self.daily_stats['grid_energy_kwh'] += grid_energy
        self.daily_stats['grid_cost_total'] += grid_energy * grid_price
        self.daily_stats['renewable_energy_kwh'] += flows['renewable_to_demand'] * dt_hours

        if flows['battery_power'] > 0:  # Charging
            charge_energy = flows['battery_power'] * dt_hours
            self.daily_stats['battery_charged_kwh'] += charge_energy
            self.daily_stats['battery_charge_cost'] += charge_energy * grid_price
        elif flows['battery_power'] < 0:  # Discharging
            discharge_energy = abs(flows['battery_power']) * dt_hours
            charged_kwh = self.daily_stats['battery_charged_kwh']
            avg_charge_price = (
                self.daily_stats['battery_charge_cost'] / max(0.001, charged_kwh)
            )
            net_profit = discharge_energy * (grid_price - avg_charge_price)
            self.daily_stats['arbitrage_profit'] += max(0.0, net_profit)

    def get_daily_summary(self) -> Dict:
        """Get summary of daily operation."""
        total_demand = sum(h['demand'] for h in self.hourly_history)
        total_grid = sum(h['grid_used'] for h in self.hourly_history)

        return {
            'manager_id': self.id,
            'total_demand_kwh': total_demand,
            'grid_dependency': total_grid / total_demand if total_demand > 0 else 0,
            'renewable_fraction': (total_demand - total_grid) / total_demand if total_demand > 0 else 0,
            'total_grid_cost': self.daily_stats['grid_cost_total'],
            'avg_grid_price': self.daily_stats['grid_cost_total'] / self.daily_stats['grid_energy_kwh']
                if self.daily_stats['grid_energy_kwh'] > 0 else 0,
            'arbitrage_profit': self.daily_stats['arbitrage_profit'],
            'rl_agent_enabled': False,
        }

    def reset_daily_stats(self):
        """Reset for new day (preserve battery SOC across days)."""
        for key in self.daily_stats:
            self.daily_stats[key] = 0.0
        self.hourly_history.clear()
