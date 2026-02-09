"""
RL Environment Wrapper
Gymnasium-compatible environment for PPO training
"""

from datetime import timedelta
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from Environment import Environment, SimulationConfig
from ChargingArea import ChargingArea
from EnergyManager import BatteryStorageConfig, EnergyManager, GridSourceConfig, SolarSourceConfig, WindSourceConfig
from stranded_vehicle_tracker import StrandedVehicleTracker
from blackout_tracker import BlackoutTracker
from optimization_config import PPOConfig, CostParameters, DEFAULT_PPO_CONFIG, DEFAULT_COST_PARAMS


@dataclass
class RLState:
    """Structured state representation"""
    time_features: np.ndarray  # [hour_sin, hour_cos, day_of_week]
    station_states: np.ndarray  # Per-station: [queue_len, battery_soc, available_chargers, grid_capacity]
    vehicle_states: np.ndarray  # Variable length: [soc, dist_to_station, urgency]
    energy_features: np.ndarray  # [solar_factor, wind_factor, elec_price, grid_carbon]
    system_features: np.ndarray  # [total_demand, total_available, n_active_vehicles]


class EVChargingRLEnv(gym.Env):
    """
    Gymnasium environment for EV charging operational control
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 infrastructure_config: Dict,
                 simulation_config: SimulationConfig,
                 ppo_config: PPOConfig = DEFAULT_PPO_CONFIG,
                 cost_params: CostParameters = DEFAULT_COST_PARAMS,
                 render_mode: Optional[str] = None):
        super().__init__()
        
        self.infrastructure = infrastructure_config
        self.sim_config = simulation_config
        self.ppo_config = ppo_config
        self.cost_params = cost_params
        self.render_mode = render_mode
        
        # Initialize base simulation
        self.env = self._create_environment()
        
        # Trackers for objectives
        self.stranded_tracker = StrandedVehicleTracker()
        self.blackout_tracker = BlackoutTracker()
        
        # State dimensions
        self.max_stations = self.infrastructure.get('n_stations', 5)
        self.max_vehicles = 100
        
        # Action space: Continuous actions per station        
        self.highway_length = self.sim_config.highway_length_km
        self.action_space = spaces.Dict({
            # Number of stations the agent wants to "activate"
            "num_to_activate": spaces.Discrete(self.max_stations, start=1),
            
            # 1D Positions for all possible stations [0, highway_length]
            "position": spaces.Box(low=0, high=self.highway_length, shape=(self.max_stations,), dtype=np.float32),
            
            # Number of: chargers, solar kW, wind kW, grid kW, battery kWh for each station
            "config": spaces.Dict({
                "n_chargers": spaces.Discrete(20, start=1),
                "solar_kw": spaces.Box(low=0, high=1000, shape=(), dtype=np.float32),
                "wind_kw": spaces.Box(low=0, high=1000, shape=(), dtype=np.float32),
                "grid_kw": spaces.Box(low=0, high=1000, shape=(), dtype=np.float32),
                "waiting_ratio_spots": spaces.Box(low=1.0, high=10.0, shape=(), dtype=np.float32),
                "battery_capacity_kwh": spaces.Box(low=0, high=500, shape=(), dtype=np.float32),
                "battery_power_kw": spaces.Box(low=0, high=500, shape=(), dtype=np.float32)
            })
        })
        
        # Observation space
        self.observation_space = spaces.Dict({
            'time': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            'stations': spaces.Box(
                low=0, high=1, 
                shape=(self.max_stations, 8),  # 8 features per station
                dtype=np.float32
            ),
            'vehicles': spaces.Box(
                low=0, high=1,
                shape=(self.max_vehicles, 4),  # 4 features per vehicle
                dtype=np.float32
            ),
            'energy': spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32),
            'system': spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32),
            'mask': spaces.Box(low=0, high=1, shape=(self.max_vehicles,), dtype=np.float32)
        })
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_length = 0
        self.current_step = 0
        
        # Cached state
        self._current_state: Optional[RLState] = None
    
    def _create_environment(self) -> Environment:
        """Create base simulation environment from infrastructure config"""
        # Build charging areas from config
        charging_areas = []
        for i, station_config in enumerate(self.infrastructure.get('stations', [])):
            from ChargingArea import ChargingArea
            from EnergyManager import EnergyManagerConfig            
            source_configs = []
            
            source_configs.append(GridSourceConfig(max_power_kw=station_config.get('grid_kw', 500)))
            
            source_configs.append(SolarSourceConfig(peak_power_kw=station_config.get('solar_kw', 200)))
            
            source_configs.append(WindSourceConfig(base_power_kw=station_config.get('wind_kw', 150),
                                                   min_power_kw=station_config.get('min_wind_kw', 15),
                                                   max_power_kw=station_config.get('max_wind_kw', 300)))

            source_configs.append(BatteryStorageConfig(capacity_kwh=station_config.get('battery_power_kw', 100),
                                                       max_charge_rate_kw=station_config.get('battery_charge_kw', 50),
                                                       max_discharge_rate_kw=station_config.get('battery_discharge_kw', 50))) 
            
            single_config = EnergyManagerConfig(source_configs=source_configs)
            
            area = ChargingArea(
                area_id=f"AREA-{i:02d}",
                name=f"Station {i+1}",
                location_km=station_config.get('location_km', i * 60),
                num_chargers=station_config.get('n_chargers', 4),
                waiting_spots=station_config.get('waiting_spots', 6),
                energy_manager=EnergyManager(single_config)
            )
            charging_areas.append(area)
        
        # Override simulation config with charging areas
        from Highway import Highway
        highway = Highway(
            length_km=self.sim_config.highway_length_km,
            charging_areas=charging_areas
        )
        
        # Create environment with this highway
        env = Environment(self.sim_config)
        env.highway = highway
        
        return env
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Reset trackers
        self.stranded_tracker.reset()
        self.blackout_tracker.reset()
        
        # Reset simulation
        self.env.reset()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Reset episode stats
        self.episode_reward = 0.0
        self.episode_length = 0
        self.current_step = 0
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one timestep with RL action
        
        Action format: Flattened array [station_0_actions, station_1_actions, ...]
        Per station: [grid_fraction (-1,1), battery_action (-1,1), charger_0_power, ...]
        """
        self.current_step += 1
        
        # Decode actions and apply to simulation
        self._apply_actions(action)
        
        # Step simulation
        events = self.env.step()
        
        # Update trackers with new state
        self._update_trackers(events)
        
        # Calculate reward
        reward = self._calculate_reward(events)
        self.episode_reward += reward
        
        # Check termination
        terminated = self.env.current_time >= self.env.config.simulation_start_time + timedelta(hours=self.env.config.simulation_duration_hours)
        truncated = self.current_step >= self.ppo_config.max_episode_steps
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        info['events'] = events
        
        return obs, reward, terminated, truncated, info
    
    def _apply_actions(self, action: np.ndarray) -> None:
        """
        Decode and apply RL actions to simulation
        """
        # Reshape to (n_stations, action_dim)
        n_stations = len(self.env.highway.charging_areas)
        actions = action[:n_stations * self.action_dim].reshape(n_stations, -1)
        
        for i, (area_id, area) in enumerate(self.env.highway.charging_areas.items()):
            if i >= n_stations:
                break
            
            station_action = actions[i]
            
            # Decode: grid_fraction from [-1,1] to [0,1]
            grid_frac = (station_action[0] + 1) / 2
            
            # Decode: battery action from [-1,1] to [charge, discharge, idle]
            battery_action = station_action[1]  # -1=charge, 0=idle, 1=discharge
            
            # Decode: charger powers from [-1,1] to [0, max_power]
            charger_powers = (station_action[2:] + 1) / 2
            
            # Apply to energy manager (store for step execution)
            # This modifies the energy manager's target setpoints
            # Actual application happens during area.step()
            self._set_station_targets(area, grid_frac, battery_action, charger_powers)
    
    def _set_station_targets(self, area: ChargingArea, 
                            grid_frac: float,
                            battery_action: float,
                            charger_powers: np.ndarray) -> None:
        """
        Set target operating points for station
        These are used during the station's step() method
        """
        # Store targets in area for use during step
        area.rl_targets = {
            'grid_fraction': grid_frac,
            'battery_action': battery_action,  # Negative=charge, Positive=discharge
            'charger_powers': charger_powers[:len(area.chargers)]
        }
    
    def _update_trackers(self, events: Dict) -> None:
        """Update stranded and blackout trackers from simulation events"""
        # Process highway events
        for area_id, area_events in events.get('highway', {}).get('area_events', {}).items():
            # Check for stranded vehicles from power-down events
            for vehicle_down in area_events.get('vehicles_powered_down', []):
                # These vehicles lost power and were requeued - not stranded yet
                pass
            
            # Check for energy shortage (blackout)
            if area_events.get('energy_shortage', False):
                energy_state = area_events.get('energy_state', {})
                # Get area object
                area = self.env.highway.charging_areas.get(area_id)
                if area:
                    self.blackout_tracker.check_energy_balance(
                        station_id=area_id,
                        timestamp=self.env.current_time,
                        total_demand_kw=energy_state.get('current_demand_kw', 0),
                        grid_available_kw=energy_state.get('grid_available_kw', 0),
                        battery_available_kw=energy_state.get('battery_available_kw', 0),
                        renewable_available_kw=(
                            energy_state.get('solar_available_kw', 0) + 
                            energy_state.get('wind_available_kw', 0)
                        ),
                        vehicles_charging=sum(
                            1 for c in area.chargers 
                            if c.status.name == 'OCCUPIED'
                        )
                    )
        
        # Update blackout tracker step
        self.blackout_tracker.step(self.env.current_time)
        
        # TODO: Integrate with vehicle tracker for stranded detection
        # This requires vehicle-level simulation integration
    
    def _calculate_reward(self, events: Dict) -> float:
        """
        Calculate multi-component reward
        """
        reward = 0.0
        
        # Revenue from completed charging
        for area_events in events.get('highway', {}).get('area_events', {}).values():
            for session in area_events.get('completed_sessions', []):
                energy_kwh = session.get('energy_delivered', 0)
                reward += energy_kwh * 0.5  # Simplified revenue per kWh
        
        # Costs
        # Energy cost
        for area_id, area in self.env.highway.charging_areas.items():
            energy_stats = area.energy_manager.get_state() # pyright: ignore[reportOptionalMemberAccess]
            energy_cost = (
                energy_stats.get('grid_import_kwh', 0) * self.cost_params.electricity_base_price +
                energy_stats.get('battery_discharged_kwh', 0) * self.cost_params.battery_degradation_cost_per_kwh
            )
            reward -= energy_cost
        
        # Penalties
        stranded = self.stranded_tracker.total_stranded_count
        reward -= stranded * self.cost_params.stranded_vehicle_penalty
        
        blackouts = self.blackout_tracker.total_blackouts
        reward -= blackouts * self.cost_params.blackout_penalty
        
        # Waiting time penalty
        for area in self.env.highway.charging_areas.values():
            queue_len = len(area.queue)
            reward -= queue_len * self.cost_params.waiting_time_penalty_per_min
        
        # Renewable utilization bonus
        for area in self.env.highway.charging_areas.values():
            energy_stats = area.energy_manager.get_state() # pyright: ignore[reportOptionalMemberAccess]
            renewable_kwh = (
                energy_stats.get('solar_used_kwh', 0) + 
                energy_stats.get('wind_used_kwh', 0)
            )
            reward += renewable_kwh * self.cost_params.renewable_bonus_per_kwh
        
        return reward
    
    def _get_observation(self) -> Dict:
        """Construct observation dict from simulation state"""
        # Time features
        hour = self.env.current_time.hour
        minute = self.env.current_time.minute
        time_features = np.array([
            np.sin(2 * np.pi * hour / 24),  # Hour sine
            np.cos(2 * np.pi * hour / 24),  # Hour cosine
            self.env.current_time.weekday() / 7,  # Day of week
            (hour * 60 + minute) / (24 * 60)  # Time of day normalized
        ], dtype=np.float32)
        
        # Station states (pad to max_stations)
        station_states = np.zeros((self.max_stations, 6), dtype=np.float32)
        for i, (area_id, area) in enumerate(self.env.highway.charging_areas.items()):
            if i >= self.max_stations:
                break
            
            energy_state = area.energy_manager.get_state()
            station_states[i] = [
                len(area.queue) / max(area.waiting_spots, 1),  # Queue occupancy
                energy_state.get('battery_soc', 0) / 100,  # Battery SOC
                len(area.get_available_chargers()) / max(len(area.chargers), 1),  # Available ratio
                energy_state.get('grid_available_kw', 0) / max(energy_state.get('total_available_kw', 1), 1),  # Grid fraction
                1.0 if energy_state.get('shortage_active', False) else 0.0,  # Shortage flag
                area.get_utilization_rate()  # Utilization
            ]
        
        # Vehicle states (variable length, pad to max_vehicles)
        # TODO: Integrate with actual vehicle tracking
        vehicle_states = np.zeros((self.max_vehicles, 4), dtype=np.float32)
        mask = np.zeros(self.max_vehicles, dtype=np.float32)
        
        # Placeholder: would extract from vehicle tracker
        # For now, use queue information as proxy
        vehicle_idx = 0
        for area in self.env.highway.charging_areas.values():
            for entry in area.queue:
                if vehicle_idx >= self.max_vehicles:
                    break
                # Placeholder features
                vehicle_states[vehicle_idx] = [
                    0.5,  # Estimated SOC
                    0.5,  # Distance to station (normalized)
                    1.0,  # Urgency (high if in queue)
                    0.0   # Charging status
                ]
                mask[vehicle_idx] = 1.0
                vehicle_idx += 1
        
        # Energy features
        # Aggregate across all stations
        total_solar = sum(
            area.energy_manager.get_state().get('solar_available_kw', 0) # pyright: ignore[reportOptionalMemberAccess]
            for area in self.env.highway.charging_areas.values()
        )
        total_wind = sum(
            area.energy_manager.get_state().get('wind_available_kw', 0) # pyright: ignore[reportOptionalMemberAccess]
            for area in self.env.highway.charging_areas.values()
        )
        total_available = sum(
            area.energy_manager.get_state().get('total_available_kw', 0) # pyright: ignore[reportOptionalMemberAccess]
            for area in self.env.highway.charging_areas.values()
        )
        
        energy_features = np.array([
            total_solar / max(total_available, 1),
            total_wind / max(total_available, 1),
            0.5,  # Electricity price (would be from price model)
            0.5   # Grid carbon intensity
        ], dtype=np.float32)
        
        # System features
        system_features = np.array([
            sum(area.energy_manager.total_demand_kw for area in self.env.highway.charging_areas.values()), # pyright: ignore[reportOptionalMemberAccess]
            total_available,
            vehicle_idx,  # Active vehicles
            self.current_step / self.ppo_config.max_episode_steps  # Progress
        ], dtype=np.float32)
        
        return {
            'time': time_features,
            'stations': station_states,
            'vehicles': vehicle_states,
            'energy': energy_features,
            'system': system_features,
            'mask': mask
        }
    
    def _get_info(self) -> Dict:
        """Additional info for logging"""
        return {
            'stranded_metrics': self.stranded_tracker.get_metrics(),
            'blackout_metrics': self.blackout_tracker.get_metrics(),
            'episode_reward': self.episode_reward,
            'step': self.current_step
        }
        
    def _flatten_state(self, state: Dict) -> np.ndarray:
        """Flatten structured state dict into a single vector for PPO input"""
        time_flat = state['time']
        stations_flat = state['stations'].flatten()
        vehicles_flat = state['vehicles'].flatten()
        energy_flat = state['energy']
        system_flat = state['system']
        mask_flat = state['mask']
        
        return np.concatenate([
            time_flat,
            stations_flat,
            vehicles_flat,
            energy_flat,
            system_flat,
            mask_flat
        ]).astype(np.float32)
    
    def get_objectives(self) -> Tuple[float, float, float]:
        """
        Get the three optimization objectives for this episode
        Called at end of episode for Optuna evaluation
        """
        # f1: Infrastructure cost (amortized per episode)
        # Would calculate from infrastructure config
        
        # f2: Stranded vehicles (per day rate)
        sim_hours = self.sim_config.simulation_duration_hours
        stranded_rate = self.stranded_tracker.get_stranding_rate(sim_hours)
        
        # f3: Blackout events (per year rate)
        blackout_rate = self.blackout_tracker.get_blackout_rate(sim_hours)
        
        # Placeholder for cost - would calculate properly
        infrastructure_cost = 0.0  # TODO: Calculate from config
        
        return infrastructure_cost, stranded_rate, blackout_rate
    
    def render(self):
        """Render environment state"""
        if self.render_mode == "human":
            print(f"Step {self.current_step}, Time: {self.env.current_time}")
            print(f"Stranded: {self.stranded_tracker.total_stranded_count}, "
                  f"Blackouts: {self.blackout_tracker.total_blackouts}")
    
    def close(self):
        """Cleanup"""
        pass
