"""
Main Optimization Script
Entry point for running the complete two-stage optimization
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from optimization_config import (
    OptunaConfig, PPOConfig,
    DEFAULT_OPTUNA_CONFIG, DEFAULT_PPO_CONFIG
)
from optuna_optimizer import OptunaOptimizer
from Environment import SimulationConfig as EnvSimConfig


def parse_args():
    parser = argparse.ArgumentParser(description='EV Charging Infrastructure Optimization')
    
    # Infrastructure search
    parser.add_argument('--highway-length', type=float, default=300.0)
    parser.add_argument('--max-stations', type=int, default=5)
    parser.add_argument('--max-chargers', type=int, default=20)
    
    # Optimization
    parser.add_argument('--n-trials', type=int, default=500)
    parser.add_argument('--sampler', type=str, default='TPESampler', 
                       choices=['TPESampler', 'NSGAIISampler'])
    parser.add_argument('--pruner', type=str, default='HyperbandPruner',
                       choices=['HyperbandPruner', 'MedianPruner', 'None'])
    
    # RL training
    parser.add_argument('--ppo-episodes', type=int, default=200)
    parser.add_argument('--device', type=str, default='auto')
    
    # Execution
    parser.add_argument('--output-dir', type=str, default='./optimization_results')
    parser.add_argument('--study-name', type=str, default=None)
    parser.add_argument('--parallel', type=int, default=-1)
    
    return parser.parse_args()


def create_simulation_config(args) -> EnvSimConfig:
    """Create base simulation configuration"""
    return EnvSimConfig(
        highway_length_km=args.highway_length,
        num_charging_areas=3,  # Will be overridden by optimization
        chargers_per_area=4,
        waiting_spots_per_area=8,
        arrival_rate_per_minute=60.0,
        simulation_duration_hours=4.0,
        time_step_minutes=1.0,
        random_seed=42
    )


def create_optuna_config(args) -> OptunaConfig:
    """Create Optuna configuration from args"""
    config = DEFAULT_OPTUNA_CONFIG
    config.n_trials = args.n_trials
    config.sampler = args.sampler
    config.pruner = args.pruner if args.pruner != 'None' else None # pyright: ignore[reportAttributeAccessIssue]
    config.n_jobs = args.parallel
    
    if args.study_name:
        config.study_name = args.study_name
    else:
        config.study_name = f"ev_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Enable storage for persistence
    config.storage = f"sqlite:///{args.output_dir}/{config.study_name}.db"
    
    return config


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EV CHARGING INFRASTRUCTURE OPTIMIZATION")
    print("Multi-Objective: Cost | Stranded Vehicles | Blackout Events")
    print("=" * 60)
    
    # Create configurations
    sim_config = create_simulation_config(args)
    optuna_config = create_optuna_config(args)
    
    print(f"\nConfiguration:")
    print(f"  Highway length: {args.highway_length} km")
    print(f"  Max stations: {args.max_stations}")
    print(f"  Optimization trials: {args.n_trials}")
    print(f"  PPO episodes per trial: {args.ppo_episodes}")
    print(f"  Sampler: {args.sampler}")
    print(f"  Parallel jobs: {args.parallel}")
    
    # Create and run optimizer
    optimizer = OptunaOptimizer(sim_config, optuna_config)
    
    try:
        pareto_trials = optimizer.optimize(n_trials=args.n_trials)
        
        # Save results
        results_file = output_dir / f"{optuna_config.study_name}_results.json"
        optimizer.save_results(str(results_file))
        print(f"\nResults saved to: {results_file}")
        
        # Generate visualizations
        try:
            optimizer.plot_pareto_front(
                str(output_dir / f"{optuna_config.study_name}_pareto.png")
            )
            optimizer.plot_hypervolume(
                str(output_dir / f"{optuna_config.study_name}_hypervolume.png")
            )
        except Exception as e:
            print(f"Could not generate plots: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("PARETO-OPTIMAL SOLUTIONS")
        print("=" * 60)
        
        for i, trial in enumerate(pareto_trials[:5]):  # Top 5
            cost, stranded, blackout = trial.values # pyright: ignore[reportAttributeAccessIssue]
            infra = trial.user_attrs.get('infrastructure', {})
            
            print(f"\nSolution {i+1} (Trial {trial.number}):")
            print(f"  Cost: ${cost:,.2f}")
            print(f"  Stranded vehicles/day: {stranded:.2f}")
            print(f"  Blackout events/year: {blackout:.2f}")
            print(f"  Stations: {infra.get('n_stations', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("Optimization complete!")
        
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print("Saving partial results...")
        optimizer.save_results(str(output_dir / f"{optuna_config.study_name}_partial.json"))
    
    except Exception as e:
        print(f"\nError during optimization: {e}")
        raise


if __name__ == "__main__":
    main()