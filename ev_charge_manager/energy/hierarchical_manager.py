"""
Hierarchical Energy Manager with Dual RL Agents
- Macro agent (PPO): Optimizes infrastructure
- Micro agent (EnergyManagerAgent): Optimizes real-time energy flows
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from ev_charge_manager.energy.manager import (
    EnergyManagerConfig,
    EnergySourceConfig,
    GridSourceConfig,
    SolarSourceConfig,
    WindSourceConfig,
    BatteryStorageConfig,
    EnergySourceType,
)
from ev_charge_manager.energy.agent import EnergyManagerAgent, GridPricingSchedule


class HierarchicalEnergyManager:
    """
    EnergyManager with embedded RL agent for real-time optimization.
    
    This manager:
    1. Uses RL agent to decide energy flows each timestep
    2. Learns to arbitrage grid prices (buy low, sell high via battery)
    3. Maximizes renewable self-consumption
    4. Reports performance to macro-level optimizer
    """
    
    def __init__(
        self,
        manager_id: str,
        config: EnergyManagerConfig,
        pricing_schedule: Optional[GridPricingSchedule] = None,
        enable_rl_agent: bool = True,
        device: str = "cpu",
        weather_provider=None,
    ):
        self.id = manager_id
        self.config = config
        self.pricing = pricing_schedule or GridPricingSchedule()
        self.enable_rl_agent = enable_rl_agent
        self.weather_provider = weather_provider
        
        # Extract battery parameters from config
        battery_configs = [
            c for c in config.source_configs 
            if isinstance(c, BatteryStorageConfig)
        ]
        
        if battery_configs:
            self.battery_config = battery_configs[0]
            self.has_battery = True
        else:
            self.battery_config = BatteryStorageConfig()  # Default
            self.has_battery = False
        
        # Create RL agent if enabled
        self.rl_agent: Optional[EnergyManagerAgent] = None
        if enable_rl_agent and self.has_battery:
            self.rl_agent = EnergyManagerAgent(
                agent_id=f"{manager_id}_micro",
                pricing_schedule=self.pricing,
                battery_capacity_kwh=self.battery_config.capacity_kwh,
                battery_max_power_kw=self.battery_config.max_discharge_rate_kw,
                device=device,
            )
        
        # Runtime state
        self.current_timestamp: Optional[datetime] = None
        self.solar_sources: List[SolarSourceConfig] = []
        self.wind_sources: List[WindSourceConfig] = []
        self.grid_sources: List[GridSourceConfig] = []
        
        self._parse_config()
        
        # Statistics
        self.daily_stats = {
            'grid_cost_total': 0.0,
            'grid_energy_kwh': 0.0,
            'renewable_energy_kwh': 0.0,
            'battery_cycles': 0.0,
            'arbitrage_profit': 0.0,
            'battery_charged_kwh': 0.0,   # total energy stored today
            'battery_charge_cost': 0.0,   # total cost paid to charge today
        }
        
        self.hourly_history: List[Dict] = []
    
    def _parse_config(self):
        """Parse configuration into source lists."""
        for source_config in self.config.source_configs:
            if isinstance(source_config, SolarSourceConfig):
                self.solar_sources.append(source_config)
            elif isinstance(source_config, WindSourceConfig):
                self.wind_sources.append(source_config)
            elif isinstance(source_config, GridSourceConfig):
                self.grid_sources.append(source_config)
    
    def _calculate_solar_output(self, timestamp: datetime) -> float:
        """Calculate current solar output based on time of day."""
        if self.weather_provider is not None:
            # Real data mode: use actual GHI from weather provider per source
            total_output = 0.0
            for solar in self.solar_sources:
                total_output += self.weather_provider.get_solar_output_kw(
                    timestamp, solar.peak_power_kw
                )
            return total_output

        # Synthetic mode: parabolic curve
        hour = timestamp.hour + timestamp.minute / 60.0
        total_output = 0.0

        for solar in self.solar_sources:
            if solar.sunrise_hour <= hour <= solar.sunset_hour:
                # Parabolic curve
                time_from_peak = abs(hour - solar.peak_hour)
                peak_offset = abs(solar.peak_hour - solar.sunrise_hour)

                if time_from_peak <= peak_offset:
                    availability = 1 - (time_from_peak / peak_offset) ** 2
                else:
                    availability = 0

                # Add some randomness
                availability *= np.random.uniform(0.9, 1.0)
                total_output += solar.peak_power_kw * availability

        return total_output
    
    def _calculate_wind_output(self, timestamp: datetime) -> float:
        """Calculate current wind output with variability."""
        if self.weather_provider is not None:
            # Real data mode: use actual wind speed from weather provider per source
            total_output = 0.0
            for wind in self.wind_sources:
                total_output += self.weather_provider.get_wind_output_kw(
                    timestamp, wind.base_power_kw
                )
            return total_output

        # Synthetic mode: daily pattern + noise
        import math
        import random

        hour = timestamp.hour
        total_output = 0.0

        for wind in self.wind_sources:
            daily_factor = 0.5 + 0.5 * math.sin(2 * math.pi * hour / 24)
            noise = random.uniform(-wind.variability, wind.variability)

            power = wind.base_power_kw + (wind.max_power_kw - wind.min_power_kw) * 0.5 * (daily_factor + noise)
            total_output += float(np.clip(power, wind.min_power_kw, wind.max_power_kw))

        return total_output
    
    def step(
        self,
        timestamp: datetime,
        demand_kw: float,
        time_step_minutes: float = 60.0,
        training_mode: bool = True,
    ) -> Dict[str, float]:
        """
        Execute one timestep with RL agent control.
        
        Returns energy flows and updates agent.
        """
        self.current_timestamp = timestamp
        hour = timestamp.hour
        
        # Get current conditions
        solar_output = self._calculate_solar_output(timestamp)
        wind_output = self._calculate_wind_output(timestamp)
        grid_price = self.pricing.get_price(hour)
        
        # RL Agent decision
        if self.rl_agent and self.has_battery:
            # Select action
            action, log_prob, value = self.rl_agent.select_action(
                timestamp=timestamp,
                solar_output=solar_output,
                wind_output=wind_output,
                demand_kw=demand_kw,
                grid_price=grid_price,
                deterministic=not training_mode,
            )
            
            # Interpret action into energy flows
            energy_flows = self.rl_agent.interpret_action(
                action=action,
                solar_output=solar_output,
                wind_output=wind_output,
                demand_kw=demand_kw,
            )
            
            # Update battery SOC
            battery_energy_change = energy_flows['battery_power'] * (time_step_minutes / 60.0)
            new_soc = self.rl_agent.current_soc + (
                battery_energy_change / self.battery_config.capacity_kwh
            )
            self.rl_agent.current_soc = float(np.clip(new_soc, 0.0, 1.0))
            
            # Compute reward
            reward = self.rl_agent.compute_reward(energy_flows, grid_price, timestamp)
            
            # Store transition
            if training_mode:
                obs = self.rl_agent._get_observation(
                    timestamp, solar_output, wind_output, demand_kw, grid_price
                )
                self.rl_agent.store_transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    log_prob=log_prob,
                    value=value,
                    done=(hour == 23),  # End of day
                )
            
            # Periodic update (end of hour)
            if training_mode and timestamp.minute == 0 and len(self.rl_agent.trajectory_buffer) >= 24:
                self.rl_agent.update(n_epochs=3)
            
        else:
            # Fallback: simple priority without RL
            energy_flows = self._simple_priority_dispatch(
                solar_output, wind_output, demand_kw, grid_price, time_step_minutes
            )
            reward = 0.0
        
        # Update statistics
        self._update_stats(energy_flows, grid_price, time_step_minutes)
        
        # Record history
        self.hourly_history.append({
            'hour': hour,
            'solar': solar_output,
            'wind': wind_output,
            'demand': demand_kw,
            'grid_price': grid_price,
            'grid_used': energy_flows['grid_power'],
            'battery_soc': self.rl_agent.current_soc if self.rl_agent else 0.5,
            'reward': reward,
            'demand_shortfall': energy_flows.get('demand_shortfall', 0.0),
        })
        
        return {
            **energy_flows,
            'solar_output': solar_output,
            'wind_output': wind_output,
            'grid_price': grid_price,
            'battery_soc': self.rl_agent.current_soc if self.rl_agent else 0.5,
            'reward': reward,
        }
    
    def _simple_priority_dispatch(
        self,
        solar: float,
        wind: float,
        demand: float,
        grid_price: float,
        dt_hours: float,
    ) -> Dict[str, float]:
        """Simple dispatch without RL (fallback)."""
        renewable = solar + wind
        renewable_to_demand = min(renewable, demand)
        remaining = demand - renewable_to_demand
        
        # Use grid for remainder
        grid_power = remaining
        
        return {
            'renewable_to_demand': renewable_to_demand,
            'grid_power': grid_power,
            'battery_power': 0.0,
            'curtailed_renewable': max(0, renewable - renewable_to_demand),
            'total_demand_met': renewable_to_demand + grid_power,
            'demand_shortfall': max(0, remaining - grid_power),
        }
    
    def _update_stats(self, flows: Dict, grid_price: float, dt_hours: float):
        """Update daily statistics."""
        grid_energy = flows['grid_power'] * dt_hours
        self.daily_stats['grid_energy_kwh'] += grid_energy
        self.daily_stats['grid_cost_total'] += grid_energy * grid_price
        self.daily_stats['renewable_energy_kwh'] += flows['renewable_to_demand'] * dt_hours

        # Arbitrage profit: track charge cost, then compute net profit at discharge
        if flows['battery_power'] > 0:  # Charging — record what we paid
            charge_energy = flows['battery_power'] * dt_hours
            self.daily_stats['battery_charged_kwh'] += charge_energy
            self.daily_stats['battery_charge_cost'] += charge_energy * grid_price
        elif flows['battery_power'] < 0:  # Discharging — net profit = sell - avg buy
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
        total_renewable = sum(h['solar'] + h['wind'] for h in self.hourly_history)
        
        return {
            'manager_id': self.id,
            'total_demand_kwh': total_demand,
            'grid_dependency': total_grid / total_demand if total_demand > 0 else 0,
            'renewable_fraction': (total_demand - total_grid) / total_demand if total_demand > 0 else 0,
            'total_grid_cost': self.daily_stats['grid_cost_total'],
            'avg_grid_price': self.daily_stats['grid_cost_total'] / self.daily_stats['grid_energy_kwh'] 
                if self.daily_stats['grid_energy_kwh'] > 0 else 0,
            'arbitrage_profit': self.daily_stats['arbitrage_profit'],
            'rl_agent_enabled': self.rl_agent is not None,
        }
    
    def reset_daily_stats(self):
        """Reset for new day."""
        for key in self.daily_stats:
            self.daily_stats[key] = 0.0
        self.hourly_history.clear()
        if self.rl_agent:
            self.rl_agent.reset(initial_soc=self.battery_config.initial_soc)