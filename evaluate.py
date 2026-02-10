"""
Evaluation script for trained Multi-Agent PPO model
"""

import argparse
import torch
import numpy as np
import json

from ev_charge_manager.rl.environment import MultiAgentChargingEnv, WeatherProfile, CostParameters
from ev_charge_manager.rl.ppo_trainer import MultiAgentPPO


def evaluate_pretrained(
    model_path: str,
    n_episodes: int = 20,
    render: bool = True,
    deterministic: bool = True,
):
    """Evaluate a trained model."""
    
    # Load configuration if available
    try:
        with open(model_path.replace('.pt', '_config.json'), 'r') as f:
            config = json.load(f)
    except:
        config = {
            'n_agents': 3,
            'highway_length': 300.0,
            'traffic_min': 30.0,
            'traffic_max': 80.0,
        }
    
    # Create environment
    env = MultiAgentChargingEnv(
        n_agents=config.get('n_agents', 3),
        highway_length_km=config.get('highway_length', 300.0),
        traffic_range=(config.get('traffic_min', 30.0), config.get('traffic_max', 80.0)),
        eval_mode=True,
    )
    
    # Create trainer and load model
    trainer = MultiAgentPPO(env, device='cpu')
    trainer.load(model_path)
    
    # Run evaluation episodes
    all_rewards = []
    all_costs = []
    all_stranded = []
    all_blackouts = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        episode_reward = 0
        
        while not done:
            actions = []
            for i, actor in enumerate(trainer.actors):
                obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                action, _ = actor.get_action(obs_tensor, deterministic=deterministic)
                actions.append(action[0])
            
            actions = np.array(actions)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            episode_reward += np.sum(rewards)
        
        all_rewards.append(episode_reward)
        all_costs.append(info.get('total_cost', 0))
        all_stranded.append(info.get('stranded_vehicles', 0))
        all_blackouts.append(info.get('blackout_events', 0))
        
        if render:
            print(f"\nEpisode {ep+1}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Cost: ${info.get('total_cost', 0):,.2f}")
            print(f"  Stranded: {info.get('stranded_vehicles', 0):.1f}")
            print(f"  Blackouts: {info.get('blackout_events', 0)}")
            print(f"  Renewable: {info.get('renewable_fraction', 0):.1%}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"\nRewards:")
    print(f"  Mean: {np.mean(all_rewards):.2f} (+/- {np.std(all_rewards):.2f})")
    print(f"  Min: {np.min(all_rewards):.2f}, Max: {np.max(all_rewards):.2f}")
    print(f"\nCosts:")
    print(f"  Mean: ${np.mean(all_costs):,.2f} (+/- ${np.std(all_costs):,.2f})")
    print(f"  Min: ${np.min(all_costs):,.2f}, Max: ${np.max(all_costs):,.2f}")
    print(f"\nStranded Vehicles:")
    print(f"  Mean: {np.mean(all_stranded):.2f} (+/- {np.std(all_stranded):.2f})")
    print(f"\nBlackout Events:")
    print(f"  Mean: {np.mean(all_blackouts):.2f} (+/- {np.std(all_blackouts):.2f})")
    
    # Show best configuration
    best = env.get_best_config()
    if best:
        print("\n" + "=" * 60)
        print("OPTIMAL CONFIGURATION")
        print("=" * 60)
        print(f"Total Cost: ${best['cost']:,.2f}")
        print(f"\nStation Layout:")
        for i in range(env.n_agents):
            config = best['configs'][i]
            print(f"\n  Station {i+1} @ {best['positions'][i]:.1f} km:")
            print(f"    Chargers: {best['n_chargers'][i]}")
            print(f"    Waiting spots: {best['n_waiting'][i]}")
            print(f"    Grid: {config.grid_capacity_kw:.0f} kW")
            print(f"    Solar: {config.solar_capacity_kw:.0f} kW")
            print(f"    Wind: {config.wind_capacity_kw:.0f} kW")
            print(f"    Battery: {config.battery_capacity_kw:.0f} kW / {config.battery_storage_kwh:.0f} kWh")
    
    return {
        'rewards': all_rewards,
        'costs': all_costs,
        'stranded': all_stranded,
        'blackouts': all_blackouts,
        'best_config': best,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true', help='Render episodes')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy')
    
    args = parser.parse_args()
    
    evaluate_pretrained(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
        deterministic=not args.stochastic,
    )