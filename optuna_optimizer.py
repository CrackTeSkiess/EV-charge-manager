"""
Optuna Multi-Objective Optimizer
Infrastructure optimization using TPE with hypervolume pruning
"""

import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.visualization import plot_pareto_front, plot_hypervolume_history
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
import json
import pickle
from datetime import datetime

from optimization_config import OptunaConfig, InfrastructureSearchSpace, DEFAULT_OPTUNA_CONFIG, DEFAULT_INFRA_SEARCH
from rl_environment import EVChargingRLEnv
from ppo_agent import PPOAgent
from Environment import SimulationConfig


class InfrastructureObjective:
    """
    Callable objective function for Optuna optimization
    """
    
    def __init__(self,
                 base_sim_config: SimulationConfig,
                 optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG,
                 ppo_training_episodes: int = 200):
        self.base_sim_config = base_sim_config
        self.optuna_config = optuna_config
        self.ppo_training_episodes = ppo_training_episodes
        
        # Cache for trained policies (transfer learning)
        self.policy_cache: Dict[str, PPOAgent] = {}
        
        # Evaluation counter
        self.n_evaluations = 0
    
    def __call__(self, trial: optuna.Trial) -> Tuple[float, float, float]:
        """
        Evaluate a candidate infrastructure configuration
        
        Returns: (cost, stranded_rate, blackout_rate)
        """
        self.n_evaluations += 1
        
        # Sample infrastructure configuration
        infrastructure = self._sample_infrastructure(trial)
        
        # Check for pruned similar configurations
        #if trial.should_prune():
        #    raise optuna.TrialPruned()
        
        # Create environment with this infrastructure
        env = EVChargingRLEnv(
            infrastructure_config=infrastructure,
            simulation_config=self.base_sim_config
        )
        
        # Try to load similar policy for warm start
        agent = self._get_or_create_agent(env, infrastructure)
        
        # Train PPO agent
        training_stats = self._train_agent(env, agent)
        
        # Evaluate trained policy
        objectives = self._evaluate_policy(env, agent)
        
        # Report intermediate values for pruning
        trial.report(objectives[0], step=1)  # Cost
        trial.report(objectives[1], step=2)  # Stranded
        trial.report(objectives[2], step=3)  # Blackout
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Cache policy for potential transfer
        config_key = self._config_hash(infrastructure)
        self.policy_cache[config_key] = agent
        
        # Store additional info
        trial.set_user_attr('infrastructure', infrastructure)
        trial.set_user_attr('training_stats', training_stats)
        
        return objectives
    
    '''
    def _sample_infrastructure(self, trial: optuna.Trial) -> Dict:
        """Sample infrastructure configuration from search space"""
        search = DEFAULT_INFRA_SEARCH
        
        # Number of stations
        n_stations = trial.suggest_int(
            'n_stations', 
            search.n_stations_min, 
            search.n_stations_max
        )
        
        stations = []
        for i in range(n_stations):
            # Location (conditional on previous to ensure ordering)
            if i == 0:
                loc_min = search.location_min
            else:
                loc_min = stations[i-1]['location_km'] + 10  # Min 10km spacing
                
            loc_min = min(search.location_max - 0.1, loc_min)  # Ensure we don't exceed max
            
            location = trial.suggest_float(
                f'station_{i}_location',
                loc_min,
                search.location_max
            )
            
            # Chargers
            n_chargers = trial.suggest_int(
                f'station_{i}_chargers',
                search.chargers_min,
                search.chargers_max,
                log=True
            )
            
            # Battery - log scale
            battery_kwh = trial.suggest_float(
                f'station_{i}_battery_kwh',
                search.battery_min_kwh,
                search.battery_max_kwh,
                log=True
            )
            battery_power_kw = battery_kwh / 4  # 4-hour battery assumption
            
            # Solar
            solar_kw = trial.suggest_float(
                f'station_{i}_solar_kw',
                search.solar_min_kw,
                search.solar_max_kw
            )
            
            # Wind
            wind_kw = trial.suggest_float(
                f'station_{i}_wind_kw',
                search.wind_min_kw,
                search.wind_max_kw
            )
            
            # Grid connection - log scale
            grid_kw = trial.suggest_float(
                f'station_{i}_grid_kw',
                search.grid_min_kw,
                search.grid_max_kw,
                log=True
            )
            
            # Waiting spots ratio
            n_waiting_spots_ratio = trial.suggest_float(
                f'station_{i}_waiting_ratio_spots',
                search.waiting_ratio_spots_min,
                search.waiting_ratio_spots_max,
                log=True
            )
            
            stations.append({
                'location_km': location,
                'n_chargers': n_chargers,
                'battery_capacity_kwh': battery_kwh,
                'battery_power_kw': battery_power_kw,
                'solar_kw': solar_kw,
                'wind_kw': wind_kw,
                'grid_kw': grid_kw,
                'waiting_spots': int(n_chargers * n_waiting_spots_ratio)
            })
        
        return {
            'n_stations': n_stations,
            'stations': stations
        }
    '''
    def _sample_infrastructure(self, trial: optuna.Trial) -> Dict:
        """Sample infrastructure configuration from search space"""
        search = DEFAULT_INFRA_SEARCH
        
        # Number of stations
        n_stations = trial.suggest_int(
            'n_stations', 
            search.n_stations_min, 
            search.n_stations_max
        )

            
        # Chargers
        n_chargers = trial.suggest_int(
            'n_chargers',
            search.chargers_min,
            search.chargers_max,
            log=True
        )
            
        # Battery - log scale
        battery_kwh = trial.suggest_float(
            'battery_capacity_kwh',
            search.battery_min_kwh,
            search.battery_max_kwh,
            log=True
        )
        battery_power_kw = battery_kwh / 4  # 4-hour battery assumption
            
        # Solar
        solar_kw = trial.suggest_float(
            'solar_kw',
                search.solar_min_kw,
                search.solar_max_kw
            )
            
        # Wind
        wind_kw = trial.suggest_float(
            'wind_kw',
            search.wind_min_kw,
            search.wind_max_kw
        )
            
        # Grid connection - log scale
        grid_kw = trial.suggest_float(
            'grid_kw',
            search.grid_min_kw,
            search.grid_max_kw,
            log=True
        )
            
        # Waiting spots ratio
        n_waiting_spots_ratio = trial.suggest_float(
            'waiting_ratio_spots',
            search.waiting_ratio_spots_min,
            search.waiting_ratio_spots_max,
            log=True
        )
            
        
        return {
            'n_stations': n_stations,
            'n_chargers': n_chargers,
            'battery_capacity_kwh': battery_kwh,
            'battery_power_kw': battery_power_kw,
            'solar_kw': solar_kw,
            'wind_kw': wind_kw,
            'grid_kw': grid_kw,
            'waiting_spot_ratio': n_waiting_spots_ratio
        }
    
    def _get_or_create_agent(self, env: EVChargingRLEnv, 
                            infrastructure: Dict) -> PPOAgent:
        """Get cached agent or create new one with potential warm start"""
        # Calculate state and action dimensions from environment
        obs_sample = env.observation_space.sample()
        state_dim = len(env._flatten_state(obs_sample))
        action_dim = env.action_space.shape[0] # pyright: ignore[reportOptionalSubscript]
        
        agent = PPOAgent(state_dim, action_dim)
        
        # Try to find similar config for warm start
        # (simplified - would use more sophisticated similarity)
        
        return agent
    
    def _train_agent(self, env: EVChargingRLEnv, agent: PPOAgent) -> Dict:
        """Train PPO agent for N episodes"""
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'actor_losses': [],
            'critic_losses': []
        }
        
        for episode in range(self.ppo_training_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action
                action, value, log_prob = agent.select_action(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Store transition
                agent.store_transition(obs, action, reward, value, log_prob, 
                                      terminated or truncated)
                
                episode_reward += reward
                episode_length += 1
                
                # Update if batch full
                if len(agent.states) >= agent.config.batch_size:
                    update_stats = agent.update(next_obs if not (terminated or truncated) else None)
                    stats['actor_losses'].append(update_stats.get('actor_loss', 0))
                    stats['critic_losses'].append(update_stats.get('critic_loss', 0))
                
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(episode_length)
        
        return stats
    
    def _evaluate_policy(self, env: EVChargingRLEnv, agent: PPOAgent) -> Tuple[float, float, float]:
        """Evaluate trained policy and return objectives"""
        # Run evaluation episodes
        n_eval = 20
        all_objectives = []
        
        for _ in range(n_eval):
            obs, info = env.reset()
            
            while True:
                action, _, _ = agent.select_action(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Get final objectives
            objectives = env.get_objectives()
            all_objectives.append(objectives)
        
        # Average across evaluation episodes
        mean_objectives = np.mean(all_objectives, axis=0)
        return tuple(mean_objectives)
    
    def _config_hash(self, infrastructure: Dict) -> str:
        """Create simple hash for config caching"""
        # Simplified - would use proper hashing
        return str(infrastructure['n_stations'])


class OptunaOptimizer:
    """
    Main optimizer class managing Optuna study
    """
    
    def __init__(self,
                 base_sim_config: SimulationConfig,
                 optuna_config: OptunaConfig = DEFAULT_OPTUNA_CONFIG):
        self.config = optuna_config
        self.base_sim_config = base_sim_config
        
        # Create objective function
        self.objective = InfrastructureObjective(base_sim_config, optuna_config)
        
        # Create or load study
        self.study = self._create_study()
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with appropriate sampler and pruner"""
        # Select sampler
        if self.config.sampler == "TPESampler":
            sampler = TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                multivariate=True,
                seed=42
            )
        elif self.config.sampler == "NSGAIISampler":
            sampler = NSGAIISampler(seed=42)
        else:
            sampler = TPESampler()
        
        # Select pruner
        if self.config.pruner == "HyperbandPruner":
            pruner = HyperbandPruner(
                min_resource=1,
                reduction_factor=3
            )
        elif self.config.pruner == "MedianPruner":
            pruner = MedianPruner(
                n_startup_trials=self.config.n_warmup_steps,
                n_warmup_steps=self.config.n_warmup_steps
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        study = optuna.create_study(
            directions=self.config.directions,
            sampler=sampler,
            pruner=pruner,
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=True
        )
        
        return study
    
    def optimize(self, n_trials: Optional[int] = None) -> List[optuna.Trial]:
        """
        Run optimization
        
        Returns: List of Pareto-optimal trials
        """
        n_trials = n_trials or self.config.n_trials
        
        print(f"Starting optimization with {n_trials} trials...")
        print(f"Objectives: Cost, Stranded Rate, Blackout Rate")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        # Get Pareto front
        pareto_trials = self.study.best_trials
        
        print(f"\nOptimization complete!")
        print(f"Total trials: {len(self.study.trials)}")
        print(f"Pareto-optimal solutions: {len(pareto_trials)}")
        
        return pareto_trials # pyright: ignore[reportReturnType]
    
    def get_pareto_front_data(self) -> List[Dict]:
        """Get data for all Pareto-optimal solutions"""
        data = []
        for trial in self.study.best_trials:
            data.append({
                'trial_number': trial.number,
                'objectives': trial.values,  # (cost, stranded, blackout)
                'infrastructure': trial.user_attrs.get('infrastructure', {}),
                'training_stats': trial.user_attrs.get('training_stats', {})
            })
        return data
    
    def save_results(self, filepath: str):
        """Save optimization results"""
        results = {
            'study_name': self.config.study_name,
            'n_trials': len(self.study.trials),
            'pareto_front': self.get_pareto_front_data(),
            'best_trials': [t.number for t in self.study.best_trials]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save study for later analysis
        if self.config.storage:
            print(f"Study saved to: {self.config.storage}")
    
    def plot_pareto_front(self, save_path: Optional[str] = None):
        """Generate Pareto front visualization"""
        try:
            fig = plot_pareto_front(self.study)
            if save_path:
                fig.write_image(save_path)
            return fig
        except Exception as e:
            print(f"Could not generate plot: {e}")
            return None
    
    def plot_hypervolume(self, save_path: Optional[str] = None):
        """Plot hypervolume over optimization history"""
        try:
            fig = plot_hypervolume_history(self.study, reference_point=[1e6, 1.0, 1.0])  # Example reference point
            if save_path:
                fig.write_image(save_path)
            return fig
        except Exception as e:
            print(f"Could not generate plot: {e}")
            return None