"""
Hierarchical Training System for Dual-RL Charging Optimization

Supports three training modes:
1. SEQUENTIAL: Train micro-RL first, then macro-RL (recommended for stability)
2. SIMULTANEOUS: Train both together (faster but unstable)
3. CURRICULUM: Progressive training with increasing complexity
"""

from __future__ import annotations
from datetime import datetime, timedelta

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import json
import os
from collections import defaultdict
from collections import deque
import copy

from ev_charge_manager.rl.environment import MultiAgentChargingEnv, CostParameters, WeatherProfile
from ev_charge_manager.rl.ppo_trainer import MultiAgentPPO
from ev_charge_manager.energy import EnergyManagerAgent, GridPricingSchedule


# ---------------------------------------------------------------------------
# Frankfurt real-world data loader (used by MicroRLTrainer)
# ---------------------------------------------------------------------------

def _load_frankfurt_data() -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Load hourly averages from the Frankfurt corridor CSV files.

    Returns three dicts keyed by hour-of-day (0-23):
      demand_profile   – mean EV charging demand (kW) per hour
      solar_profile    – mean solar output fraction (0–1) per hour
      wind_profile     – mean wind output fraction (0–1) per hour

    Falls back to hardcoded synthetic profiles if the data files are not found.
    """
    base = os.path.join(
        os.path.dirname(__file__),  # ev_charge_manager/rl/
        "..", "..",                  # project root
        "data", "real_world",
    )
    traffic_path = os.path.normpath(os.path.join(base, "frankfurt_corridor_traffic.csv"))
    weather_path = os.path.normpath(os.path.join(base, "frankfurt_corridor_weather.csv"))

    demand_profile: Dict[int, float] = {}
    solar_profile: Dict[int, float] = {}
    wind_profile: Dict[int, float] = {}

    # --- Traffic → demand --------------------------------------------------
    if os.path.exists(traffic_path):
        import csv
        hourly_ev: Dict[int, list] = defaultdict(list)
        with open(traffic_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                hour = int(row["hour"])
                hourly_ev[hour].append(float(row["ev_vehicles_per_hour"]))
        for h in range(24):
            ev_arr = float(np.mean(hourly_ev.get(h, [30.0])))
            # Convert EV arrivals/hr → charging demand kW
            # Each EV: 150 kW rated, 85% draw, 45-min average dwell
            demand_profile[h] = min(ev_arr * 150.0 * 0.85 * (45 / 60), 6 * 150.0)
    else:
        # Synthetic fallback (original hardcoded behaviour)
        for h in range(24):
            base_demand = 200.0
            if 7 <= h <= 9 or 17 <= h <= 19:
                multiplier = 1.5
            elif h <= 5:
                multiplier = 0.3
            else:
                multiplier = 1.0
            demand_profile[h] = base_demand * multiplier

    # --- Weather → solar & wind fraction -----------------------------------
    if os.path.exists(weather_path):
        import csv
        hourly_ghi: Dict[int, list] = defaultdict(list)
        hourly_wind: Dict[int, list] = defaultdict(list)
        with open(weather_path) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                hour = int(row["hour"])
                hourly_ghi[hour].append(float(row["ghi_wm2"]))
                hourly_wind[hour].append(float(row["wind_speed_ms"]))
        for h in range(24):
            # Solar fraction: GHI / 1000 W/m² (clear-sky peak), capped at 1
            mean_ghi = float(np.mean(hourly_ghi.get(h, [0.0])))
            solar_profile[h] = float(np.clip(mean_ghi / 1000.0, 0.0, 1.0))
            # Wind fraction: simple power curve (cut-in 3 m/s, rated 12 m/s)
            mean_wind = float(np.mean(hourly_wind.get(h, [5.0])))
            if mean_wind < 3.0:
                wind_frac = 0.0
            else:
                wind_frac = min(1.0, (mean_wind - 3.0) / (12.0 - 3.0))
            wind_profile[h] = wind_frac
    else:
        # Synthetic fallback
        for h in range(24):
            if 6 <= h <= 18:
                solar_profile[h] = max(0.0, 1 - ((h - 13) / 7) ** 2)
            else:
                solar_profile[h] = 0.0
            wind_profile[h] = 0.5 + 0.5 * np.sin(2 * np.pi * h / 24)

    return demand_profile, solar_profile, wind_profile


# Cache at module level so the CSV is read only once per process
_FRANKFURT_DEMAND: Optional[Dict[int, float]] = None
_FRANKFURT_SOLAR: Optional[Dict[int, float]] = None
_FRANKFURT_WIND: Optional[Dict[int, float]] = None


def _get_frankfurt_profiles() -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    global _FRANKFURT_DEMAND, _FRANKFURT_SOLAR, _FRANKFURT_WIND
    if _FRANKFURT_DEMAND is None:
        _FRANKFURT_DEMAND, _FRANKFURT_SOLAR, _FRANKFURT_WIND = _load_frankfurt_data()
    return _FRANKFURT_DEMAND, _FRANKFURT_SOLAR, _FRANKFURT_WIND


class TrainingMode(Enum):
    """Training strategy selection."""
    SEQUENTIAL = auto()      # Micro first, then macro (RECOMMENDED)
    SIMULTANEOUS = auto()    # Both together (fast but unstable)
    CURRICULUM = auto()      # Progressive complexity
    FROZEN_MICRO = auto()    # Train macro with pre-trained frozen micro


@dataclass
class TrainingConfig:
    """Configuration for hierarchical training."""
    # Mode selection
    mode: TrainingMode = TrainingMode.SEQUENTIAL
    
    # Micro-RL training (Phase 1 in SEQUENTIAL mode)
    micro_episodes: int = 1000
    micro_steps_per_episode: int = 24  # One day
    micro_update_frequency: int = 24   # Update every day
    micro_lr: float = 3e-4
    
    # Macro-RL training (Phase 2 in SEQUENTIAL mode)
    macro_episodes: int = 500
    macro_episodes_per_update: int = 10
    macro_lr: float = 3e-4
    
    # Simultaneous training parameters
    simultaneous_micro_updates: int = 1  # Micro updates per macro step
    simultaneous_freeze_micro_after: int = 1000  # Freeze micro after N episodes
    
    # Curriculum parameters
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {'days': 1, 'price_variance': 0.0},   # Simple: fixed prices
        {'days': 3, 'price_variance': 0.2},   # Medium: some variance
        {'days': 7, 'price_variance': 0.5},   # Complex: full variance
    ])
    
    # Evaluation
    eval_frequency: int = 50
    eval_episodes: int = 10
    
    # Checkpointing
    checkpoint_frequency: int = 100
    output_dir: str = "./hierarchical_models"

    # Optional subdirectory overrides (default to output_dir)
    data_dir: Optional[str] = None
    models_dir: Optional[str] = None


class MicroRLTrainer:
    """
    Trainer for micro-level RL agents (energy arbitrage).
    Trains each station's energy manager independently.
    """
    
    def __init__(
        self,
        base_config: Dict,  # Fixed infrastructure config
        n_stations: int,
        pricing: GridPricingSchedule,
        device: str = "cpu",
    ):
        self.base_config = base_config
        self.n_stations = n_stations
        self.pricing = pricing
        self.device = device
        
        # Create one trainer per station
        self.agents: List[EnergyManagerAgent] = []
        self._create_agents()
        
        # Training stats — one inner list per station, one entry per episode
        self.episode_rewards: List[List[float]] = [[] for _ in range(n_stations)]
        self.episode_grid_costs: List[List[float]] = [[] for _ in range(n_stations)]

        # Hourly breakdown from the most-recently completed episode (per station)
        self.last_episode_hourly: List[List[Dict]] = [[] for _ in range(n_stations)]

        self.best_avg_reward: float = float('-inf')
    
    def _create_agents(self):
        """Create micro-RL agents for each station."""
        for i in range(self.n_stations):
            agent = EnergyManagerAgent(
                agent_id=f"micro_{i}",
                pricing_schedule=self.pricing,
                battery_capacity_kwh=self.base_config.get('battery_kwh', 500),
                battery_max_power_kw=self.base_config.get('battery_kw', 100),
                lr=3e-4,
                device=self.device,
            )
            self.agents.append(agent)
    
    def train_episode(self, episode_idx: int) -> Dict:
        """
        Train all micro-agents for one episode (24 hours).
        Returns training metrics.
        """
        daily_rewards = [[] for _ in range(self.n_stations)]
        daily_grid_costs = [0.0 for _ in range(self.n_stations)]
        hourly_data_per_station: List[List[Dict]] = [[] for _ in range(self.n_stations)]

        # Simulate one day per station
        for station_idx, agent in enumerate(self.agents):
            agent.reset(initial_soc=0.5)

            base_date = datetime(2024, 6, 15, 0, 0)

            for hour in range(24):
                timestamp = base_date + timedelta(hours=hour)

                # Generate synthetic demand and generation
                demand_kw = self._generate_demand(hour, station_idx)
                solar = self._generate_solar(hour, station_idx)
                wind = self._generate_wind(hour, station_idx)
                price = self.pricing.get_price(hour)

                # Agent selects action
                action, log_prob, value = agent.select_action(
                    timestamp, solar, wind, demand_kw, price,
                    deterministic=False,
                )

                # Interpret action and get energy flows
                flows = agent.interpret_action(action, solar, wind, demand_kw)

                # Update battery SOC: battery_power is in kW, each step is 1 hour → kWh
                battery_change = flows['battery_power'] * 1.0  # kW × 1 h = kWh
                new_soc = agent.current_soc + battery_change / agent.battery_capacity_kwh
                agent.current_soc = float(np.clip(new_soc, 0.0, 1.0))

                # Compute reward
                reward = agent.compute_reward(flows, price, timestamp)
                daily_rewards[station_idx].append(reward)

                # Store transition
                obs = agent._get_observation(timestamp, solar, wind, demand_kw, price)
                agent.store_transition(obs, action, reward, log_prob, value, done=(hour == 23))

                daily_grid_costs[station_idx] += flows['grid_power'] * price * (1 / 24)
                hourly_data_per_station[station_idx].append({
                    'hour': hour,
                    'reward': float(reward),
                    'grid_cost': float(flows['grid_power'] * price),
                    'battery_soc': float(agent.current_soc),
                    'action': float(action) if np.isscalar(action) else action.tolist(),
                })

            # Update agent policy at end of day
            if len(agent.trajectory_buffer) >= 24:
                agent.update(n_epochs=5, batch_size=24)

            episode_total = float(np.sum(daily_rewards[station_idx]))
            self.episode_rewards[station_idx].append(episode_total)
            self.episode_grid_costs[station_idx].append(daily_grid_costs[station_idx])

        # Persist hourly breakdown for the most recently completed episode
        self.last_episode_hourly = hourly_data_per_station

        # Return metrics
        avg_rewards = [
            float(np.mean(r[-100:])) if len(r) > 0 else 0.0
            for r in self.episode_rewards
        ]

        return {
            'avg_reward': float(np.mean(avg_rewards)),
            'per_station_rewards': [float(np.sum(r)) for r in daily_rewards],
            'grid_costs': [float(c) for c in daily_grid_costs],
            'hourly_data': hourly_data_per_station,
        }
    
    def _generate_demand(self, hour: int, station_idx: int) -> float:
        """
        Generate hourly EV charging demand (kW) from Frankfurt corridor data.

        Uses real-world EV arrival rates converted to kW demand.  Falls back
        to the original synthetic profile when the data file is unavailable.
        """
        demand_profile, _, _ = _get_frankfurt_profiles()
        base = demand_profile.get(hour % 24, 200.0)
        noise = np.random.uniform(0.9, 1.1)
        return base * noise

    def _generate_solar(self, hour: int, station_idx: int) -> float:
        """
        Generate hourly solar output (kW) from Frankfurt weather data.

        Converts mean GHI fraction to kW using the station's installed solar
        capacity (default 300 kW peak).  Falls back to synthetic parabolic
        profile when the data file is unavailable.
        """
        _, solar_profile, _ = _get_frankfurt_profiles()
        base_capacity = self.base_config.get("solar_kw", 300.0)
        fraction = solar_profile.get(hour % 24, 0.0)
        noise = np.random.uniform(0.90, 1.05)
        return base_capacity * fraction * noise

    def _generate_wind(self, hour: int, station_idx: int) -> float:
        """
        Generate hourly wind output (kW) from Frankfurt weather data.

        Converts mean wind-speed fraction to kW using the station's installed
        wind capacity (default 150 kW).  Falls back to synthetic sinusoidal
        profile when the data file is unavailable.
        """
        _, _, wind_profile = _get_frankfurt_profiles()
        base_capacity = self.base_config.get("wind_kw", 150.0)
        fraction = wind_profile.get(hour % 24, 0.5)
        noise = np.random.uniform(0.7, 1.3)
        return base_capacity * fraction * noise
    
    def evaluate(self, n_episodes: int = 5) -> Dict:
        """Evaluate trained micro-agents."""
        eval_rewards = []
        eval_grid_costs = []
        
        for episode in range(n_episodes):
            episode_rewards = []
            episode_grid_costs = []
            
            for station_idx, agent in enumerate(self.agents):
                agent.reset(initial_soc=0.5)
                base_date = datetime(2024, 6, 15, 0, 0)
                
                daily_reward = 0
                daily_grid_cost = 0
                
                for hour in range(24):
                    timestamp = base_date + timedelta(hours=hour)
                    demand = self._generate_demand(hour, station_idx)
                    solar = self._generate_solar(hour, station_idx)
                    wind = self._generate_wind(hour, station_idx)
                    price = self.pricing.get_price(hour)
                    
                    # Deterministic action
                    action, _, _ = agent.select_action(
                        timestamp, solar, wind, demand, price,
                        deterministic=True,
                    )
                    
                    flows = agent.interpret_action(action, solar, wind, demand)

                    # Update SOC: battery_power is in kW, each step is 1 hour → kWh
                    battery_change = flows['battery_power'] * 1.0  # kW × 1 h = kWh
                    agent.current_soc = float(np.clip(
                        agent.current_soc + battery_change / agent.battery_capacity_kwh,
                        0.0, 1.0
                    ))
                    
                    reward = agent.compute_reward(flows, price, timestamp)
                    daily_reward += reward
                    daily_grid_cost += flows['grid_power'] * price * (1/24)
                
                episode_rewards.append(daily_reward)
                episode_grid_costs.append(daily_grid_cost)
            
            eval_rewards.append(np.mean(episode_rewards))
            eval_grid_costs.append(np.mean(episode_grid_costs))
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_grid_cost': np.mean(eval_grid_costs),
            'rewards_per_station': episode_rewards,
        }
    
    def save_history(self, output_dir: str):
        """
        Persist micro-RL training history to *output_dir*.

        Writes two files:
        - ``micro_history.json``   — per-station episode rewards and grid costs
        - ``micro_final_day.json`` — hourly breakdown from the last episode
        """
        os.makedirs(output_dir, exist_ok=True)

        history = {
            "n_stations": self.n_stations,
            "n_episodes": max(len(r) for r in self.episode_rewards) if self.episode_rewards else 0,
            "per_station_rewards": [list(r) for r in self.episode_rewards],
            "per_station_grid_costs": [list(c) for c in self.episode_grid_costs],
        }
        with open(os.path.join(output_dir, "micro_history.json"), "w") as fh:
            json.dump(history, fh, indent=2)

        final_day = {
            "n_stations": self.n_stations,
            "stations": self.last_episode_hourly,
        }
        with open(os.path.join(output_dir, "micro_final_day.json"), "w") as fh:
            json.dump(final_day, fh, indent=2)

    def save(self, path: str):
        """Save all micro-agents."""
        os.makedirs(path, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(path, f"micro_agent_{i}.pt"))

        # Save config
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(self.base_config, f)
    
    def load(self, path: str):
        """Load all micro-agents."""
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(path, f"micro_agent_{i}.pt"))


class HierarchicalTrainer:
    """
    Main trainer coordinating macro and micro RL training.
    """
    
    def __init__(
        self,
        env: MultiAgentChargingEnv,
        config: TrainingConfig,
        device: str = "auto",
    ):
        self.env = env
        self.config = config
        self.device = device

        # Resolve subdirectory paths (fall back to output_dir)
        self.data_dir = config.data_dir or config.output_dir
        self.models_dir = config.models_dir or config.output_dir

        # Create output directories
        for d in (config.output_dir, self.data_dir, self.models_dir):
            os.makedirs(d, exist_ok=True)
        
        # Training state
        self.current_phase: str = "init"
        self.episode_count: int = 0
        self.best_macro_reward: float = float('-inf')
        
        # Trainers
        self.micro_trainer: Optional[MicroRLTrainer] = None
        self.macro_trainer: Optional[MultiAgentPPO] = None
        
        # Logging
        self.training_history: List[Dict] = []
    
    def train_sequential(self) -> Dict:
        """
        RECOMMENDED: Train micro-RL first, then macro-RL.
        
        Phase 1: Train micro-agents on fixed "average" infrastructure
        Phase 2: Train macro-agents using pre-trained micro-agents
        """
        print("=" * 70)
        print("SEQUENTIAL TRAINING MODE")
        print("Phase 1: Micro-RL training (energy arbitrage)")
        print("Phase 2: Macro-RL training (infrastructure optimization)")
        print("=" * 70)
        
        # === PHASE 1: MICRO-RL TRAINING ===
        self.current_phase = "micro"
        
        # Create "average" infrastructure for micro training
        average_infrastructure = {
            'battery_kwh': 500,
            'battery_kw': 100,
            'solar_kw': 300,
            'wind_kw': 150,
            'grid_kw': 500,
        }
        
        pricing = GridPricingSchedule(
            off_peak_price=self.env.cost_params.grid_offpeak_price,
            shoulder_price=self.env.cost_params.grid_shoulder_price,
            peak_price=self.env.cost_params.grid_peak_price,
        )
        
        self.micro_trainer = MicroRLTrainer(
            base_config=average_infrastructure,
            n_stations=self.env.n_agents,
            pricing=pricing,
            device=self.device,
        )
        
        print(f"\n--- Phase 1: Training {self.config.micro_episodes} micro-RL episodes ---")
        
        for episode in range(self.config.micro_episodes):
            metrics = self.micro_trainer.train_episode(episode)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean([np.mean(r[-100:]) for r in self.micro_trainer.episode_rewards])
                print(f"Micro Episode {episode+1}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Grid Cost = ${np.mean(metrics['grid_costs']):.2f}")
            
            # Early stopping if converged
            if episode > 200:
                recent_rewards = [np.mean(r[-50:]) for r in self.micro_trainer.episode_rewards]
                if np.std(recent_rewards) < 0.1 * abs(np.mean(recent_rewards)):
                    print(f"Micro-RL converged at episode {episode}")
                    break
        
        # Evaluate micro-agents
        micro_eval = self.micro_trainer.evaluate(n_episodes=10)
        print(f"\nMicro-RL Final Evaluation:")
        print(f"  Avg Reward: {micro_eval['avg_reward']:.2f}")
        print(f"  Avg Grid Cost: ${micro_eval['avg_grid_cost']:.2f}")
        
        # Save micro-agents and training history
        micro_path = os.path.join(self.models_dir, "micro_pretrained")
        self.micro_trainer.save(micro_path)
        self.micro_trainer.save_history(self.data_dir)
        print(f"Micro-agents saved to {micro_path}")
        print(f"Micro history saved to {self.data_dir}/micro_history.json")
        
        # === PHASE 2: MACRO-RL TRAINING ===
        self.current_phase = "macro"

        # Inject trained micro-agent weights into the environment so every
        # HierarchicalEnergyManager created during macro training starts from the
        # learned policy rather than a random initialisation.
        self.env.enable_micro_rl = True
        self.env.set_micro_agents(self.micro_trainer.agents)

        self.macro_trainer = MultiAgentPPO(
            env=self.env,
            lr=self.config.macro_lr,
            device=self.device,
        )
        
        print(f"\n--- Phase 2: Training {self.config.macro_episodes} macro-RL episodes ---")
        print("Using PRE-TRAINED micro-agents (frozen)")
        
        # Train macro with frozen micro
        self.macro_trainer.train(
            total_episodes=self.config.macro_episodes,
            episodes_per_update=self.config.macro_episodes_per_update,
            eval_interval=self.config.eval_frequency,
            save_interval=self.config.checkpoint_frequency,
            save_dir=self.models_dir,
            history_dir=self.data_dir,
        )
        
        # Final evaluation
        final_eval = self.macro_trainer.evaluate(n_episodes=20)
        
        return {
            'mode': 'sequential',
            'micro_episodes': self.config.micro_episodes,
            'macro_episodes': self.config.macro_episodes,
            'micro_final_reward': micro_eval['avg_reward'],
            'macro_final_reward': final_eval['avg_reward'],
            'best_config': self.env.get_best_config(),
        }
    
    def train_simultaneous(self) -> Dict:
        """
        Train both macro and micro at the same time.
        Faster but potentially unstable.
        """
        print("=" * 70)
        print("SIMULTANEOUS TRAINING MODE")
        print("Training both macro and micro RL together")
        print("WARNING: This can be unstable!")
        print("=" * 70)
        
        self.current_phase = "simultaneous"
        
        # Create macro trainer (micro will be created per episode)
        self.macro_trainer = MultiAgentPPO(
            env=self.env,
            lr=self.config.macro_lr,
            device=self.device,
        )
        
        # Training loop
        for episode in range(self.config.macro_episodes):
            self.episode_count = episode

            # Collect trajectories with simultaneous micro learning
            trajectories = self._collect_simultaneous_trajectories()

            # Update macro
            macro_metrics = self.macro_trainer.update(trajectories)

            # Log
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([np.sum(r) for r in trajectories['rewards']])
                print(f"Episode {episode+1}: Macro Reward = {avg_reward:.2f}, "
                      f"Policy Loss = {macro_metrics['policy_loss']:.4f}")

            # Periodic evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                self.macro_trainer.evaluate(n_episodes=5)
        
        return {
            'mode': 'simultaneous',
            'episodes': self.config.macro_episodes,
            'best_config': self.env.get_best_config(),
        }
    
    def _collect_simultaneous_trajectories(self) -> Dict:
        """
        Collect trajectories with both macro and micro learning.
        """
        # This is complex - need to coordinate both levels
        # For now, use standard macro collection (micro learns inside env.step)
        
        if self.macro_trainer is None:
            self.macro_trainer = MultiAgentPPO(
                env=self.env,
                lr=self.config.macro_lr,
                device=self.device,
            )    
        return self.macro_trainer.collect_trajectories(
            self.config.macro_episodes_per_update
        )
    
    def train_curriculum(self) -> Dict:
        """
        Progressive training with increasing complexity.
        """
        print("=" * 70)
        print("CURRICULUM TRAINING MODE")
        print("Progressive complexity increase")
        print("=" * 70)
        
        results = []
        
        for stage_idx, stage in enumerate(self.config.curriculum_stages):
            print(f"\n--- Curriculum Stage {stage_idx + 1}/{len(self.config.curriculum_stages)} ---")
            print(f"Days: {stage['days']}, Price Variance: {stage['price_variance']}")
            
            # Adjust environment complexity
            self.env.simulation_duration = stage['days'] * 24
            
            # Train at this level
            stage_config = copy.deepcopy(self.config)
            stage_config.macro_episodes = stage['days'] * 100
            
            # Apply curriculum price variance to the environment
            self.env.price_variance = stage.get('price_variance', 0.0)

            if stage_idx == 0:
                # First stage: train micro using stage-specific episode count
                original_macro_eps = self.config.macro_episodes
                self.config.macro_episodes = stage_config.macro_episodes
                result = self.train_sequential()
                self.config.macro_episodes = original_macro_eps
            else:
                # Subsequent stages: fine-tune with increasing difficulty
                result = self._fine_tune_stage(stage_config, stage)
            
            results.append({
                'stage': stage_idx,
                'config': stage,
                'result': result,
            })
        
        return {
            'mode': 'curriculum',
            'stages': results,
        }
    
    def _fine_tune_stage(self, config: TrainingConfig, stage: Dict) -> Dict:
        """Fine-tune for a curriculum stage, returning full evaluation metrics."""
        # Apply curriculum price variance to the environment
        self.env.price_variance = stage.get('price_variance', 0.0)

        if self.macro_trainer is None:
            self.macro_trainer = MultiAgentPPO(
                env=self.env,
                lr=config.macro_lr,
                device=self.device,
            )
        self.macro_trainer.train(
            total_episodes=config.macro_episodes,
            episodes_per_update=config.macro_episodes_per_update,
            save_dir=self.models_dir,
            history_dir=self.data_dir,
        )

        # Evaluate so stages 2+ report real metrics
        eval_result = self.macro_trainer.evaluate(n_episodes=10)
        return {
            'status': 'fine_tuned',
            'macro_episodes': config.macro_episodes,
            'macro_final_reward': eval_result['avg_reward'],
            'avg_reward': eval_result['avg_reward'],
            'avg_cost': eval_result['avg_cost'],
            'best_config': eval_result.get('best_config'),
        }
    
    def train(self) -> Dict:
        """Main entry point - selects training mode."""
        start_time = time.time()
        
        if self.config.mode == TrainingMode.SEQUENTIAL:
            result = self.train_sequential()
        elif self.config.mode == TrainingMode.SIMULTANEOUS:
            result = self.train_simultaneous()
        elif self.config.mode == TrainingMode.CURRICULUM:
            result = self.train_curriculum()
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")
        
        elapsed = time.time() - start_time
        result['training_time_seconds'] = elapsed
        
        # Save final results
        with open(os.path.join(self.data_dir, "training_result.json"), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\nTraining completed in {elapsed/3600:.2f} hours")
        return result


# Convenience functions for common training scenarios

def train_efficient(
    n_stations: int = 3,
    highway_length: float = 300.0,
    output_dir: str = "./models_efficient",
) -> Dict:
    """
    MOST EFFICIENT: Sequential training with good defaults.
    """
    config = TrainingConfig(
        mode=TrainingMode.SEQUENTIAL,
        micro_episodes=500,      # Quick micro training
        macro_episodes=300,      # Macro optimization
        output_dir=output_dir,
    )
    
    env = MultiAgentChargingEnv(
        n_agents=n_stations,
        highway_length_km=highway_length,
        simulation_duration_hours=24.0,
        enable_micro_rl=True,
    )
    
    trainer = HierarchicalTrainer(env, config)
    return trainer.train()


def train_fast(
    n_stations: int = 3,
    output_dir: str = "./models_fast",
) -> Dict:
    """
    FASTEST: Simultaneous training (may be less stable).
    """
    config = TrainingConfig(
        mode=TrainingMode.SIMULTANEOUS,
        macro_episodes=500,
        output_dir=output_dir,
    )
    
    env = MultiAgentChargingEnv(
        n_agents=n_stations,
        enable_micro_rl=True,
    )
    
    trainer = HierarchicalTrainer(env, config)
    return trainer.train()


def train_robust(
    n_stations: int = 3,
    output_dir: str = "./models_robust",
) -> Dict:
    """
    MOST ROBUST: Curriculum learning with extensive micro training.
    """
    config = TrainingConfig(
        mode=TrainingMode.CURRICULUM,
        micro_episodes=2000,     # Extensive micro training
        macro_episodes=1000,     # Extensive macro training
        curriculum_stages=[
            {'days': 1, 'price_variance': 0.0},
            {'days': 3, 'price_variance': 0.3},
            {'days': 7, 'price_variance': 0.6},
        ],
        output_dir=output_dir,
    )
    
    env = MultiAgentChargingEnv(
        n_agents=n_stations,
        enable_micro_rl=True,
    )
    
    trainer = HierarchicalTrainer(env, config)
    return trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['efficient', 'fast', 'robust'], default='efficient')
    parser.add_argument('--stations', type=int, default=3)
    parser.add_argument('--output', type=str, default='./models')
    
    args = parser.parse_args()
    
    if args.mode == 'efficient':
        result = train_efficient(args.stations, output_dir=args.output)
    elif args.mode == 'fast':
        result = train_fast(args.stations, output_dir=args.output)
    else:
        result = train_robust(args.stations, output_dir=args.output)
    
    print("\nFinal Result:")
    print(json.dumps(result, indent=2, default=str))
