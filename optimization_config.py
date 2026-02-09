"""
Optimization Configuration
Centralized configuration for Optuna and PPO hyperparameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch


@dataclass
class InfrastructureSearchSpace:
    """Search space for infrastructure optimization"""
    # Station count
    n_stations_min: int = 1
    n_stations_max: int = 5
    
    # Location along highway (km)
    location_min: float = 0.0
    location_max: float = 300.0
    
    # Chargers per station
    chargers_min: int = 2
    chargers_max: int = 20
    
    # Waiting spots to charger ratio per station
    waiting_ratio_spots_min: float = 1.0
    waiting_ratio_spots_max: float = 10.0
    
    # Battery capacity (kWh) - log scale
    battery_min_kwh: float = 100.0
    battery_max_kwh: float = 2000.0
    
    # Solar capacity (kW)
    solar_min_kw: float = 0.0
    solar_max_kw: float = 500.0
    
    # Wind capacity (kW)
    wind_min_kw: float = 0.0
    wind_max_kw: float = 300.0
    
    # Grid connection (kW) - log scale
    grid_min_kw: float = 100.0
    grid_max_kw: float = 2000.0


@dataclass
class OptunaConfig:
    """Configuration for Optuna multi-objective optimization"""
    # Algorithm
    sampler: str = "TPESampler"  # or "CmaEsSampler", "NSGAIISampler"
    pruner: str = "HyperbandPruner"  # or "MedianPruner", "NoPruner"
    
    # Search parameters
    n_trials: int = 500
    n_startup_trials: int = 50  # Random exploration before TPE
    n_warmup_steps: int = 10  # For pruner
    
    # Multi-objective
    n_objectives: int = 3
    directions: List[str] = field(default_factory=lambda: ["minimize", "minimize", "minimize"])
    reference_point: Optional[List[float]] = None  # For hypervolume
    
    # Parallelization
    n_jobs: int = -1  # Use all cores
    study_name: str = "ev_charging_infrastructure"
    storage: Optional[str] = None  # "sqlite:///optuna_study.db" for persistence
    
    # Early stopping
    hypervolume_threshold: float = 0.01  # Stop if improvement < 1%


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Network architecture
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
    activation: str = "relu"  # or "tanh"
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Training
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 1e-3
    lr_decay: bool = True
    batch_size: int = 2048
    minibatch_size: int = 64
    epochs_per_update: int = 10
    max_grad_norm: float = 0.5
    
    # Episode settings
    max_episode_steps: int = 1000
    n_training_episodes: int = 200
    n_eval_episodes: int = 20
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CostParameters:
    """Economic parameters for objective calculation"""
    # CAPEX ($)
    fixed_site_cost: float = 50000.0  # Per station
    charger_cost_per_unit: float = 25000.0  # 50kW charger
    battery_cost_per_kwh: float = 300.0
    battery_cost_per_kw: float = 150.0
    solar_cost_per_kw: float = 1000.0
    wind_cost_per_kw: float = 1200.0
    grid_connection_cost_per_kw: float = 200.0
    
    # OPEX ($/kWh)
    electricity_base_price: float = 0.15
    electricity_peak_price: float = 0.30
    battery_degradation_cost_per_kwh: float = 0.05
    
    # Penalties
    stranded_vehicle_penalty: float = 1000.0  # Per vehicle
    blackout_penalty: float = 500.0  # Per event
    waiting_time_penalty_per_min: float = 0.1
    renewable_bonus_per_kwh: float = 0.02


@dataclass
class EarlyWarningConfig:
    """Configuration for early warning system extraction"""
    # Gradient analysis
    value_gradient_threshold: float = 10.0
    gradient_window_size: int = 10
    
    # Action sensitivity
    action_change_threshold: float = 0.3
    
    # Attention analysis
    top_k_vehicles: int = 3
    
    # REWS scoring weights
    rews_gradient_weight: float = 0.4
    rews_action_weight: float = 0.3
    rews_attention_weight: float = 0.3
    
    # Alert levels
    rews_yellow_threshold: float = 0.6
    rews_red_threshold: float = 0.8


# Default configurations
DEFAULT_INFRA_SEARCH = InfrastructureSearchSpace()
DEFAULT_OPTUNA_CONFIG = OptunaConfig()
DEFAULT_PPO_CONFIG = PPOConfig()
DEFAULT_COST_PARAMS = CostParameters()
DEFAULT_EW_CONFIG = EarlyWarningConfig()