"""
Training script for Multi-Agent PPO Charging Area Optimization
"""

import argparse
import json
from datetime import datetime

import numpy as np

from ev_charge_manager.rl.environment import MultiAgentChargingEnv, WeatherProfile, CostParameters
from ev_charge_manager.rl.ppo_trainer import MultiAgentPPO


def parse_args():
    parser = argparse.ArgumentParser(description='Train multi-agent PPO for charging optimization')
    
    # Environment parameters
    parser.add_argument('--n-agents', type=int, default=3, help='Number of charging areas')
    parser.add_argument('--highway-length', type=float, default=300.0, help='Highway length in km')
    parser.add_argument('--traffic-min', type=float, default=30.0, help='Min traffic (veh/hr)')
    parser.add_argument('--traffic-max', type=float, default=80.0, help='Max traffic (veh/hr)')
    parser.add_argument('--sim-duration', type=float, default=4.0, help='Simulation duration (hours)')
    
    # Weather parameters
    parser.add_argument('--solar-min', type=float, default=0.0, help='Min solar availability')
    parser.add_argument('--solar-max', type=float, default=1.0, help='Max solar availability')
    parser.add_argument('--wind-min', type=float, default=0.0, help='Min wind availability')
    parser.add_argument('--wind-max', type=float, default=1.0, help='Max wind availability')
    
    # Cost parameters
    parser.add_argument('--stranded-penalty', type=float, default=10000.0, help='Penalty per stranded vehicle')
    parser.add_argument('--blackout-penalty', type=float, default=50000.0, help='Penalty per blackout')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000, help='Total training episodes')
    parser.add_argument('--episodes-per-update', type=int, default=10, help='Episodes per PPO update')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per update')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-eps', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--collab-weight', type=float, default=0.5, help='Collaboration weight (0-1)')
    
    # System
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='./models', help='Model save directory')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Multi-Agent PPO Charging Area Optimization")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Agents: {args.n_agents}")
    print(f"  Highway: {args.highway_length}km")
    print(f"  Traffic: {args.traffic_min}-{args.traffic_max} veh/hr")
    print(f"  Episodes: {args.episodes}")
    print(f"  Collaboration weight: {args.collab_weight}")
    
    # Create environment
    weather = WeatherProfile(
        min_solar=args.solar_min,
        max_solar=args.solar_max,
        min_wind=args.wind_min,
        max_wind=args.wind_max,
    )
    
    cost_params = CostParameters(
        stranded_vehicle_penalty=args.stranded_penalty,
        blackout_penalty=args.blackout_penalty,
    )
    
    env = MultiAgentChargingEnv(
        n_agents=args.n_agents,
        highway_length_km=args.highway_length,
        traffic_range=(args.traffic_min, args.traffic_max),
        weather_profile=weather,
        cost_params=cost_params,
        simulation_duration_hours=args.sim_duration,
        collaboration_weight=args.collab_weight,
    )
    
    # Create trainer
    trainer = MultiAgentPPO(
        env=env,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_eps,
        device=args.device,
    )
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(f"{args.save_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    try:
        trainer.train(
            total_episodes=args.episodes,
            episodes_per_update=args.episodes_per_update,
            n_epochs=args.epochs,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save("interrupted_model.pt")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    results = trainer.evaluate(n_episodes=10)
    
    # Save results
    with open(f"{args.save_dir}/results.json", 'w') as f:
        json.dump({
            'avg_reward': float(results['avg_reward']),
            'avg_cost': float(results['avg_cost']),
            'best_config': {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in results['best_config'].items()
            } if results['best_config'] else None,
        }, f, indent=2)
    
    print(f"\nResults saved to {args.save_dir}/")


if __name__ == "__main__":
    main()