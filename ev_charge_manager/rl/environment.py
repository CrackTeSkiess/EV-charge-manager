"""
Multi-Agent Reinforcement Learning Environment for Charging Area Optimization
With Hierarchical Energy Management (Macro + Micro RL Agents)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Callable, Sequence
from dataclasses import dataclass, field
import random
from collections import deque
from datetime import datetime, timedelta

from ev_charge_manager.simulation import Simulation, SimulationParameters
from ev_charge_manager.energy import (
    CHARGER_RATED_POWER_KW,
    CHARGER_AVG_DRAW_FACTOR,
    EnergyManagerConfig,
    EnergySourceConfig,
    GridSourceConfig,
    SolarSourceConfig,
    WindSourceConfig,
    BatteryStorageConfig,
)
from ev_charge_manager.energy import HierarchicalEnergyManager, GridPricingSchedule
from ev_charge_manager.highway import Highway
from ev_charge_manager.charging import ChargingArea
from ev_charge_manager.data.traffic_profiles import (
    WEEKDAY_HOURLY_FRACTIONS,
    PEAK_HOURLY_FRACTION,
)


@dataclass
class WeatherProfile:
    """Weather profile for the highway location."""
    min_solar: float = 0.0
    max_solar: float = 1.0
    solar_variance: float = 0.1
    
    min_wind: float = 0.0
    max_wind: float = 1.0
    wind_variance: float = 0.2
    
    def sample_solar(self, hour: int) -> float:
        """Sample solar availability for given hour."""
        if 6 <= hour <= 18:
            base = self.max_solar * max(0, 1 - abs(hour - 12) / 6)
        else:
            base = self.min_solar
        
        noise = random.uniform(-self.solar_variance, self.solar_variance)
        return float(np.clip(base + noise, 0, 1))
    
    def sample_wind(self) -> float:
        """Sample wind availability."""
        base = (self.min_wind + self.max_wind) / 2
        noise = random.uniform(-self.wind_variance, self.wind_variance)
        return float(np.clip(base + noise, 0, 1))


@dataclass
class CostParameters:
    """Cost parameters for optimization."""
    # Building costs (per unit)
    charger_cost_per_kw: float = 500.0
    waiting_spot_cost: float = 10000.0
    base_station_cost: float = 500000.0
    
    # Solar/wind/battery costs
    solar_cost_per_kw: float = 1000.0
    wind_cost_per_kw: float = 1200.0
    battery_cost_per_kwh: float = 300.0
    
    # Operational costs
    stranded_vehicle_penalty: float = 10000.0
    blackout_penalty: float = 50000.0
    energy_cost_weight: float = 1.0
    
    # Micro-RL agent costs (defaults must match CLI defaults in hierarchical.py)
    grid_peak_price: float = 0.25
    grid_shoulder_price: float = 0.15
    grid_offpeak_price: float = 0.08


class ChargingAreaAgent:
    """Agent representing a single charging area."""
    
    def __init__(self, agent_id: int, highway_position: float):
        self.agent_id = agent_id
        self.position = highway_position
        self.action_dim = 8
        self.local_traffic_estimate = 0.0
        self.neighbor_distances = []
        self.last_performance = {}
        
    def get_observation_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
    
    def get_action_space(self) -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)


class MultiAgentChargingEnv(gym.Env):
    """
    Multi-agent environment with hierarchical RL:
    - Macro agents (PPO): Optimize infrastructure placement and sizing
    - Micro agents (per station): Optimize real-time energy arbitrage
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        n_agents: int = 3,
        highway_length_km: float = 300.0,
        traffic_range: Tuple[float, float] = (20, 100),
        weather_profile: Optional[WeatherProfile] = None,
        cost_params: Optional[CostParameters] = None,
        simulation_duration_hours: float = 24.0,  # Full day for energy arbitrage
        eval_mode: bool = False,
        collaboration_weight: float = 0.5,
        enable_micro_rl: bool = True,
        micro_rl_device: str = "cpu",
    ):
        super().__init__()
        
        self.n_agents = n_agents
        self.highway_length_km = highway_length_km
        self.traffic_range = traffic_range
        self.weather = weather_profile or WeatherProfile()
        self._base_weather = self.weather  # preserve user-supplied bounds for reset()
        self.cost_params = cost_params or CostParameters()
        self.simulation_duration = simulation_duration_hours
        self.eval_mode = eval_mode
        self.collaboration_weight = collaboration_weight
        self.enable_micro_rl = enable_micro_rl
        self.micro_rl_device = micro_rl_device
        self.price_variance: float = 0.0  # curriculum-controlled grid price noise
        
        # Initialize macro agents
        self.agents: List[ChargingAreaAgent] = []
        self._init_agents()
        
        # Spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_agents, 8), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_agents, 28),  # Extended for micro-RL stats
            dtype=np.float32
        )
        
        # State
        self.current_episode = 0
        self.current_positions: np.ndarray = np.linspace(50, highway_length_km - 50, n_agents)
        self.current_configs: List[EnergyManagerConfig] = []
        self.current_hierarchical_managers: List[HierarchicalEnergyManager] = []
        self.current_traffic: float = 60.0
        
        # Tracking
        self.episode_history: deque = deque(maxlen=100)
        self.best_config: Optional[Dict] = None
        self.best_cost: float = float('inf')

        # P4.1: pre-trained micro-agent weights to inject on each _decode_actions() call
        self._pretrained_micro_agents: list = []
        # P4.2: persistent managers so micro-RL weights survive across episodes
        self._persistent_managers: list = [None] * n_agents
        
        # Micro-RL training tracking
        self.micro_rl_rewards: List[List[float]] = [[] for _ in range(n_agents)]
        
    def _init_agents(self):
        """Initialize macro agent representations."""
        initial_positions = np.linspace(50, self.highway_length_km - 50, self.n_agents)
        for i, pos in enumerate(initial_positions):
            self.agents.append(ChargingAreaAgent(i, pos))

    def set_micro_agents(self, trained_agents: list) -> None:
        """
        Store pre-trained micro-agent objects so their network weights are copied
        into every HierarchicalEnergyManager created by _decode_actions().

        Call this once after micro-RL training completes (or after loading saved
        micro-agents) and before macro-RL training begins.  The weights are copied
        — not shared — so macro training cannot corrupt them.
        """
        self._pretrained_micro_agents = list(trained_agents)

    def _decode_actions(self, actions: np.ndarray) -> Tuple[
        List[float], 
        List[EnergyManagerConfig], 
        List[int], 
        List[int],
        List[HierarchicalEnergyManager]
    ]:
        """
        Decode raw actions into positions, configs, and hierarchical managers.
        
        Actions per agent (8 dimensions):
        - [0]: Position adjustment (-20km to +20km)
        - [1]: Number of chargers (2-12)
        - [2]: Number of waiting spots (4-20)
        - [3]: Grid max power (100-1000 kW)
        - [4]: Solar peak power (0-500 kW)
        - [5]: Wind base power (0-400 kW)
        - [6]: Battery power rate (0-200 kW)
        - [7]: Battery capacity (0-800 kWh)
        """
        positions = []
        configs = []
        managers = []
        n_chargers_list = []
        n_waiting_list = []
        
        base_spacing = self.highway_length_km / (self.n_agents + 1)
        
        for i, action in enumerate(actions):
            # Position
            base_pos = base_spacing * (i + 1)
            adjustment = float(action[0]) * 20.0
            pos = float(np.clip(base_pos + adjustment, 10, self.highway_length_km - 10))
            
            if i > 0:
                pos = max(pos, positions[-1] + 30)
            # Always enforce the upper bound, for every agent including the last
            pos = float(np.clip(pos, 10, self.highway_length_km - 10))

            positions.append(pos)
            
            # Infrastructure
            n_chargers = int(np.clip((action[1] + 1) * 5 + 2, 2, 12))
            n_waiting = int(np.clip((action[2] + 1) * 8 + 4, 4, 20))
            n_chargers_list.append(n_chargers)
            n_waiting_list.append(n_waiting)
            
            # Energy sources
            grid_kw = float(np.clip((action[3] + 1) * 450 + 100, 100, 1000))
            solar_kw = float(np.clip((action[4] + 1) * 250, 0, 500))
            wind_kw = float(np.clip((action[5] + 1) * 200, 0, 400))
            battery_power = float(np.clip((action[6] + 1) * 100, 0, 200))
            battery_storage = float(np.clip((action[7] + 1) * 400, 0, 800))
            
            # Build source configs
            source_configs: List[EnergySourceConfig] = []
            source_configs.append(GridSourceConfig(max_power_kw=grid_kw))
            
            if solar_kw > 10:
                source_configs.append(SolarSourceConfig(peak_power_kw=solar_kw))
            if wind_kw > 10:
                source_configs.append(WindSourceConfig(base_power_kw=wind_kw))
            if battery_storage > 50:
                source_configs.append(BatteryStorageConfig(
                    capacity_kwh=battery_storage,
                    max_charge_rate_kw=battery_power,
                    max_discharge_rate_kw=battery_power,
                    initial_soc=0.5,
                    min_soc=0.1,
                    max_soc=0.95,
                    round_trip_efficiency=0.90,
                ))
            
            config = EnergyManagerConfig(source_configs=source_configs)
            configs.append(config)
            
            # Create or update hierarchical manager with micro-RL agent
            pricing = GridPricingSchedule(
                off_peak_price=self.cost_params.grid_offpeak_price,
                shoulder_price=self.cost_params.grid_shoulder_price,
                peak_price=self.cost_params.grid_peak_price,
            )

            # P4.2: reuse the persistent manager so micro-RL weights survive across
            # episodes (essential for simultaneous-mode learning).
            if self._persistent_managers[i] is not None:
                manager = self._persistent_managers[i]
                # Update infrastructure config while preserving rl_agent learned weights.
                manager.config = config
                # _parse_config() appends, so clear source lists first.
                manager.solar_sources.clear()
                manager.wind_sources.clear()
                manager.grid_sources.clear()
                manager._parse_config()
                # Sync battery parameters so the agent's dispatch stays physically valid.
                new_battery = [
                    c for c in config.source_configs
                    if isinstance(c, BatteryStorageConfig)
                ]
                if new_battery:
                    manager.battery_config = new_battery[0]
                    manager.has_battery = True
                    if manager.rl_agent is not None:
                        manager.rl_agent.battery_capacity_kwh = new_battery[0].capacity_kwh
                        manager.rl_agent.battery_max_power_kw = new_battery[0].max_discharge_rate_kw
                else:
                    manager.has_battery = False
            else:
                manager = HierarchicalEnergyManager(
                    manager_id=f"EM-{i}",
                    config=config,
                    pricing_schedule=pricing,
                    enable_rl_agent=self.enable_micro_rl,
                    device=self.micro_rl_device,
                )
                self._persistent_managers[i] = manager

            # P4.1: inject pre-trained micro-agent weights if set_micro_agents() was called.
            # Weights are copied so macro training cannot corrupt them.
            if (self._pretrained_micro_agents
                    and i < len(self._pretrained_micro_agents)
                    and manager.rl_agent is not None):
                manager.rl_agent.network.load_state_dict(
                    self._pretrained_micro_agents[i].network.state_dict()
                )

            managers.append(manager)
        
        return positions, configs, n_chargers_list, n_waiting_list, managers
    
    def _calculate_building_cost(
        self, 
        positions: List[float], 
        configs: Sequence[EnergyManagerConfig],
        n_chargers_list: List[int],
        n_waiting_list: List[int]
    ) -> float:
        """Calculate total building cost."""
        total_cost = 0.0
        
        for i, (pos, config, n_chargers, n_waiting) in enumerate(
            zip(positions, configs, n_chargers_list, n_waiting_list)
        ):
            station_cost = self.cost_params.base_station_cost
            charger_cost = n_chargers * 150 * self.cost_params.charger_cost_per_kw
            waiting_cost = n_waiting * self.cost_params.waiting_spot_cost
            
            # Energy infrastructure costs
            solar_cost = 0.0
            wind_cost = 0.0
            battery_cost = 0.0
            
            for source_config in config.source_configs:
                if isinstance(source_config, SolarSourceConfig):
                    solar_cost += source_config.peak_power_kw * self.cost_params.solar_cost_per_kw
                elif isinstance(source_config, WindSourceConfig):
                    wind_cost += source_config.base_power_kw * self.cost_params.wind_cost_per_kw
                elif isinstance(source_config, BatteryStorageConfig):
                    battery_cost += source_config.capacity_kwh * self.cost_params.battery_cost_per_kwh
            
            station_total = (
                station_cost + charger_cost + waiting_cost + 
                solar_cost + wind_cost + battery_cost
            )
            total_cost += station_total
        
        # Amortize over 10 years
        daily_cost = total_cost / 3650.0
        return daily_cost
    
    def _extract_energy_stats(self, config: EnergyManagerConfig) -> Dict[str, float]:
        """Extract energy configuration statistics."""
        stats = {
            'grid_max_kw': 0.0,
            'solar_peak_kw': 0.0,
            'wind_base_kw': 0.0,
            'battery_rate_kw': 0.0,
            'battery_cap_kwh': 0.0,
        }
        
        for source in config.source_configs:
            if isinstance(source, GridSourceConfig):
                stats['grid_max_kw'] = source.max_power_kw
            elif isinstance(source, SolarSourceConfig):
                stats['solar_peak_kw'] = source.peak_power_kw
            elif isinstance(source, WindSourceConfig):
                stats['wind_base_kw'] = source.base_power_kw
            elif isinstance(source, BatteryStorageConfig):
                stats['battery_rate_kw'] = source.max_discharge_rate_kw
                stats['battery_cap_kwh'] = source.capacity_kwh
        
        return stats
    
    def _run_simulation_with_micro_rl(
        self, 
        positions: List[float], 
        managers: List[HierarchicalEnergyManager],
        n_chargers_list: List[int],
        n_waiting_list: List[int]
    ) -> Dict:
        """
        Run full-day simulation with micro-RL agents controlling energy each hour.
        """
        # Reset managers for new day
        for manager in managers:
            manager.reset_daily_stats()
        
        # Simulate each hour
        hours = int(self.simulation_duration)
        base_date = datetime(2024, 6, 15, 0, 0)
        
        total_micro_rewards = [0.0 for _ in range(self.n_agents)]
        hourly_demands = [[] for _ in range(self.n_agents)]
        
        for hour in range(hours):
            timestamp = base_date + timedelta(hours=hour)

            # Apply curriculum-controlled price variance to grid pricing each hour
            if self.price_variance > 0:
                noise = random.gauss(0, self.price_variance)
                for manager in managers:
                    ps = manager.pricing
                    ps.peak_price = max(0.05, self.cost_params.grid_peak_price * (1 + noise))
                    ps.shoulder_price = max(0.03, self.cost_params.grid_shoulder_price * (1 + noise))
                    ps.off_peak_price = max(0.01, self.cost_params.grid_offpeak_price * (1 + noise))

            # Calculate traffic-based demand for each station.
            # Uses the BASt weekday hourly profile so the agent trains on the
            # same demand shape that SUMO produces during validation.
            hourly_shape = WEEKDAY_HOURLY_FRACTIONS[hour % 24] / PEAK_HOURLY_FRACTION

            for i, manager in enumerate(managers):
                # Demand = n_chargers × effective_power × occupancy × noise
                # hourly_shape ∈ [0.1, 1.0]: time-of-day curve from BASt data
                # traffic ratio: episode-level traffic volume / max traffic
                effective_power = CHARGER_RATED_POWER_KW * CHARGER_AVG_DRAW_FACTOR
                occupancy = min(1.5, hourly_shape * (self.current_traffic / self.traffic_range[1]))
                demand_noise = random.uniform(0.8, 1.2)
                demand_kw = n_chargers_list[i] * effective_power * occupancy * demand_noise
                
                # Run micro-RL step
                result = manager.step(
                    timestamp=timestamp,
                    demand_kw=demand_kw,
                    time_step_minutes=60.0,
                    training_mode=not self.eval_mode,
                )
                
                total_micro_rewards[i] += result.get('reward', 0.0)
                hourly_demands[i].append({
                    'hour': hour,
                    'demand': demand_kw,
                    'grid_used': result['grid_power'],
                    'grid_price': result['grid_price'],
                    'battery_soc': result['battery_soc'],
                })
        
        # Compile results
        total_grid_cost = sum(m.daily_stats['grid_cost_total'] for m in managers)
        total_grid_energy = sum(m.daily_stats['grid_energy_kwh'] for m in managers)
        total_renewable = sum(m.daily_stats['renewable_energy_kwh'] for m in managers)
        total_arbitrage = sum(m.daily_stats['arbitrage_profit'] for m in managers)
        total_demand_kwh = sum(
            h['demand'] for m in managers for h in m.hourly_history
        )
        
        # Count shortages: hours where demand was not fully met
        shortage_events = 0
        for manager in managers:
            for h in manager.hourly_history:
                if h.get('demand_shortfall', 0.0) > 0.01:  # 0.01 kW threshold avoids float noise
                    shortage_events += 1
        
        # Compile per-station statistics
        station_statistics = {}
        for i, manager in enumerate(managers):
            station_statistics[f"AREA-{i+1:02d}"] = {
                'grid_cost': manager.daily_stats['grid_cost_total'],
                'grid_energy_kwh': manager.daily_stats['grid_energy_kwh'],
                'renewable_energy_kwh': manager.daily_stats['renewable_energy_kwh'],
                'arbitrage_profit': manager.daily_stats['arbitrage_profit'],
                'micro_rl_reward': total_micro_rewards[i],
                'avg_battery_soc': np.mean([h['battery_soc'] for h in manager.hourly_history]) if manager.hourly_history else 0.5,
            }
        
        return {
            'total_grid_cost': total_grid_cost,
            'total_grid_energy_kwh': total_grid_energy,
            'total_renewable_kwh': total_renewable,
            'total_arbitrage_profit': total_arbitrage,
            'total_demand_kwh': total_demand_kwh,
            'shortage_events': shortage_events,
            'station_statistics': station_statistics,
            'micro_rl_rewards': total_micro_rewards,
            'hourly_data': hourly_demands,
        }
    
    def _compute_rewards(
        self, 
        sim_results: Dict, 
        building_cost: float
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Compute rewards combining macro and micro RL performance.
        """
        # Extract metrics
        grid_cost = sim_results['total_grid_cost']
        shortage_events = sim_results['shortage_events']
        arbitrage_profit = sim_results['total_arbitrage_profit']
        micro_rewards = sim_results['micro_rl_rewards']
        
        # Stranded vehicles estimate (based on shortages)
        total_stranded = shortage_events * 2.0  # Estimate 2 vehicles per shortage
        
        # Total cost (building + operation - arbitrage profit)
        total_cost = (
            building_cost +
            grid_cost +
            total_stranded * self.cost_params.stranded_vehicle_penalty +
            shortage_events * self.cost_params.blackout_penalty -
            arbitrage_profit * 0.5  # Partial credit for arbitrage
        )
        
        # Base reward (negative cost)
        base_reward = -total_cost / 100000.0
        
        # Agent-specific rewards
        agent_rewards = np.zeros(self.n_agents)
        
        for i, agent in enumerate(self.agents):
            station_stats = sim_results['station_statistics'].get(f"AREA-{i+1:02d}", {})
            
            # Local metrics
            local_grid_cost = station_stats.get('grid_cost', 0)
            local_micro_reward = micro_rewards[i] if i < len(micro_rewards) else 0
            
            # Local reward combines infrastructure efficiency and micro-RL performance
            local_reward = -(
                local_grid_cost / 10000.0 +
                abs(station_stats.get('avg_battery_soc', 0.5) - 0.5) * 0.1  # Prefer mid-range SOC
            ) + local_micro_reward * 0.01  # Small bonus for micro-RL performance
            
            # Combine global and local
            agent_rewards[i] = (
                (1 - self.collaboration_weight) * local_reward +
                self.collaboration_weight * base_reward
            )
        
        global_reward = base_reward
        
        info = {
            'total_cost': total_cost,
            'building_cost': building_cost,
            'grid_cost': grid_cost,
            'arbitrage_profit': arbitrage_profit,
            'stranded_vehicles': total_stranded,
            'shortage_events': shortage_events,
            'micro_rl_rewards': micro_rewards,
            'total_demand_kwh': sim_results.get('total_demand_kwh', 0),
            'renewable_fraction': sim_results['total_renewable_kwh'] /
                max(1.0, sim_results.get('total_demand_kwh', 1.0)),
        }
        
        return agent_rewards, global_reward, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.current_episode += 1
        self.current_traffic = float(random.uniform(*self.traffic_range))
        
        # Randomise weather within the bounds set by the user-supplied WeatherProfile.
        # _base_weather is set once in __init__ and never overwritten, so CLI args
        # like --solar-max are always respected as hard ceilings.
        w = self._base_weather
        self.weather = WeatherProfile(
            min_solar=float(random.uniform(0.0, w.max_solar * 0.2)),
            max_solar=float(random.uniform(w.max_solar * 0.6, w.max_solar)),
            solar_variance=w.solar_variance,
            min_wind=float(random.uniform(0.0, w.max_wind * 0.3)),
            max_wind=float(random.uniform(w.max_wind * 0.4, w.max_wind)),
            wind_variance=w.wind_variance,
        )
        
        base_positions = np.linspace(50, self.highway_length_km - 50, self.n_agents)
        for i, agent in enumerate(self.agents):
            agent.position = float(base_positions[i])
        
        self.micro_rl_rewards = [[] for _ in range(self.n_agents)]
        
        observations = self._get_observations()
        
        return observations, {
            'traffic': self.current_traffic,
            'weather': self.weather,
            'episode': self.current_episode,
        }
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        observations = []
        
        sorted_indices = sorted(range(self.n_agents), key=lambda i: self.agents[i].position)

        # P3.1 fix: enumerate so sort_pos is the position in the sorted list,
        # not reusing the agent_id (idx) as an index into sorted_indices.
        # P3.2 fix: one-hot encodes sort_pos (the output slot), not agent_id,
        # because observations are returned in position-sorted order.
        for sort_pos, idx in enumerate(sorted_indices):
            agent = self.agents[idx]

            obs = np.zeros(28, dtype=np.float32)

            # Sort-position one-hot (first 3 dims) — encodes output slot, not agent_id
            if sort_pos < 3:
                obs[sort_pos] = 1.0

            # Position features
            obs[3] = agent.position / self.highway_length_km

            # Distance to neighbours — use sort_pos for boundary checks and indexing
            if sort_pos > 0:
                left_neighbour = self.agents[sorted_indices[sort_pos - 1]]
                obs[4] = (agent.position - left_neighbour.position) / self.highway_length_km
            else:
                obs[4] = agent.position / self.highway_length_km

            if sort_pos < self.n_agents - 1:
                right_neighbour = self.agents[sorted_indices[sort_pos + 1]]
                obs[5] = (right_neighbour.position - agent.position) / self.highway_length_km
            else:
                obs[5] = (self.highway_length_km - agent.position) / self.highway_length_km
            
            # Global traffic
            obs[6] = self.current_traffic / 100.0
            
            # Weather
            obs[7] = self.weather.min_solar
            obs[8] = self.weather.max_solar
            obs[9] = self.weather.solar_variance
            obs[10] = self.weather.min_wind
            obs[11] = self.weather.max_wind
            obs[12] = self.weather.wind_variance
            
            # Historical performance
            if agent.last_performance:
                obs[13] = agent.last_performance.get('utilization', 0)
                obs[14] = agent.last_performance.get('shortages', 0) / 10
                obs[15] = agent.last_performance.get('cost', 0) / 1000000
            
            # Global context
            obs[16] = self.n_agents / 10
            obs[17] = self.highway_length_km / 500
            obs[18] = self.simulation_duration / 24
            
            # Time of day encoding (noon peak)
            hour = 12
            obs[19] = np.sin(2 * np.pi * hour / 24)
            obs[20] = np.cos(2 * np.pi * hour / 24)
            
            # Current configuration
            if self.current_configs and idx < len(self.current_configs):
                energy_stats = self._extract_energy_stats(self.current_configs[idx])
                obs[21] = energy_stats['solar_peak_kw'] / 500
                obs[22] = energy_stats['wind_base_kw'] / 400
                obs[23] = energy_stats['battery_rate_kw'] / 200
                obs[24] = energy_stats['battery_cap_kwh'] / 800
            
            # Micro-RL performance indicators
            if idx < len(self.micro_rl_rewards) and self.micro_rl_rewards[idx]:
                obs[25] = np.mean(self.micro_rl_rewards[idx]) / 100.0
            else:
                obs[25] = 0.0
            
            # Grid pricing context
            obs[26] = self.cost_params.grid_peak_price / 0.5
            obs[27] = self.cost_params.grid_offpeak_price / 0.5
            
            observations.append(obs)
        
        return np.array(observations)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """Execute one step with given actions."""
        # Decode actions
        positions, configs, n_chargers_list, n_waiting_list, managers = self._decode_actions(actions)
        self.current_positions = np.array(positions)
        self.current_configs = list(configs)
        self.current_n_chargers = list(n_chargers_list)
        self.current_n_waiting = list(n_waiting_list)
        self.current_hierarchical_managers = managers
        
        for i, pos in enumerate(positions):
            self.agents[i].position = pos
        
        # Calculate building cost
        building_cost = self._calculate_building_cost(
            positions, configs, n_chargers_list, n_waiting_list
        )
        
        # Run full-day simulation with micro-RL control
        try:
            sim_results = self._run_simulation_with_micro_rl(
                positions, managers, n_chargers_list, n_waiting_list
            )
        except Exception as e:
            print(f"Simulation failed: {e}")
            sim_results = {
                'total_grid_cost': 1e6,
                'total_grid_energy_kwh': 0,
                'total_renewable_kwh': 0,
                'total_arbitrage_profit': 0,
                'total_demand_kwh': 0,
                'shortage_events': 100,
                'station_statistics': {},
                'micro_rl_rewards': [0] * self.n_agents,
            }
        
        # Store micro-RL rewards for observation
        self.micro_rl_rewards = [[r] for r in sim_results['micro_rl_rewards']]
        
        # Compute rewards
        agent_rewards, global_reward, info = self._compute_rewards(sim_results, building_cost)
        
        # Update agent performance
        for i, agent in enumerate(self.agents):
            station_stats = sim_results['station_statistics'].get(f"AREA-{i+1:02d}", {})
            agent.last_performance = {
                'utilization': station_stats.get('grid_energy_kwh', 0) / max(1, station_stats.get('grid_energy_kwh', 0) + station_stats.get('renewable_energy_kwh', 0)),
                'shortages': sim_results['shortage_events'] / self.n_agents,
                'cost': info['total_cost'],
            }
        
        # Check for best configuration
        if info['total_cost'] < self.best_cost:
            self.best_cost = info['total_cost']
            self.best_config = {
                'positions': np.array(positions),
                'configs': list(configs),
                'managers': managers,
                'n_chargers': n_chargers_list,
                'n_waiting': n_waiting_list,
                'cost': info['total_cost'],
                'grid_cost': info['grid_cost'],
                'arbitrage_profit': info['arbitrage_profit'],
            }
        
        terminated = True
        truncated = False
        
        observations = self._get_observations()
        
        info['global_reward'] = global_reward
        info['best_cost'] = self.best_cost
        
        return observations, agent_rewards, terminated, truncated, info
    
    def render(self):
        """Render current state."""
        print(f"\nEpisode {self.current_episode}")
        print(f"Traffic: {self.current_traffic:.1f} veh/hr")
        print(f"Positions: {self.current_positions}")
        print(f"Best cost so far: {self.best_cost:.2f}")
        
        if self.best_config:
            print(f"\nBest configuration:")
            for i in range(self.n_agents):
                energy_stats = self._extract_energy_stats(self.best_config['configs'][i])
                print(f"  Station {i+1} at {self.best_config['positions'][i]:.1f}km: "
                      f"{self.best_config['n_chargers'][i]} chargers, "
                      f"{energy_stats['solar_peak_kw']:.0f}kW solar, "
                      f"{energy_stats['battery_cap_kwh']:.0f}kWh battery")
            print(f"  Grid cost: ${self.best_config['grid_cost']:,.2f}")
            print(f"  Arbitrage profit: ${self.best_config['arbitrage_profit']:,.2f}")
    
    def get_best_config(self) -> Optional[Dict]:
        """Get best configuration found so far."""
        return self.best_config