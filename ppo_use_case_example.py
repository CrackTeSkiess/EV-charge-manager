"""
Complete Use Case Example for Multi-Agent PPO Charging Area Optimization

This example demonstrates:
1. Setting up the environment with realistic parameters
2. Training the multi-agent PPO system
3. Evaluating the trained model
4. Visualizing and analyzing results
"""

import numpy as np
import json
from datetime import datetime

# Import our modules
from ChargingAreaEnv import (
    MultiAgentChargingEnv, 
    WeatherProfile, 
    CostParameters
)
from PPOTrainer import MultiAgentPPO
from EnergyManager import (
    EnergyManagerConfig,
    GridSourceConfig,
    SolarSourceConfig,
    WindSourceConfig,
    BatteryStorageConfig,
)


def scenario_1_basic_training():
    """
    Scenario 1: Basic training with default parameters.
    Good for getting started and understanding the system.
    """
    print("=" * 70)
    print("SCENARIO 1: Basic Training")
    print("=" * 70)
    
    # Create environment with default settings
    env = MultiAgentChargingEnv(
        n_agents=4,  # 3 charging stations
        highway_length_km=300.0,
        traffic_range=(30.0, 80.0),  # 30-80 vehicles per hour
        simulation_duration_hours=20,
        collaboration_weight=0.5,  # Balanced local/global optimization
    )
    
    # Create trainer
    trainer = MultiAgentPPO(
        env=env,
        lr=3e-4,
        gamma=0.99,
        device="auto",
    )
    
    # Train for a small number of episodes (quick test)
    print("\nStarting quick training run...")
    trainer.train(
        total_episodes=100,  # Small number for demo
        episodes_per_update=5,
        n_epochs=5,
    )
    
    # Evaluate
    results = trainer.evaluate(n_episodes=5)
    
    return trainer, results


def scenario_2_high_renewable_scenario():
    """
    Scenario 2: High renewable energy scenario.
    Optimizes for solar-heavy configuration in sunny climate.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: High Renewable Energy (Solar Farm)")
    print("=" * 70)
    
    # Sunny weather profile
    sunny_weather = WeatherProfile(
        min_solar=0.1,
        max_solar=1.0,  # Peak sun
        solar_variance=0.05,  # Stable sun
        min_wind=0.2,
        max_wind=0.6,
        wind_variance=0.15,
    )
    
    # Higher penalties to encourage reliability
    high_reliability_costs = CostParameters(
        stranded_vehicle_penalty=20000.0,  # Very high penalty for stranded vehicles
        blackout_penalty=100000.0,  # Critical penalty for blackouts
        solar_cost_per_kw=800.0,  # Cheaper solar
        battery_cost_per_kwh=250.0,  # Cheaper batteries
    )
    
    env = MultiAgentChargingEnv(
        n_agents=4,
        highway_length_km=400.0,
        traffic_range=(50.0, 120.0),  # Higher traffic
        weather_profile=sunny_weather,
        cost_params=high_reliability_costs,
        simulation_duration_hours=6.0,  # Longer simulation
        collaboration_weight=0.7,  # More global coordination
    )
    
    trainer = MultiAgentPPO(
        env=env,
        lr=2e-4,  # Lower learning rate for stability
        gamma=0.995,  # Higher discount for long-term planning
    )
    
    # Train longer
    trainer.train(
        total_episodes=500,
        episodes_per_update=10,
        n_epochs=10,
        eval_interval=100,
        save_interval=200,
    )
    
    # Final evaluation
    results = trainer.evaluate(n_episodes=10)
    
    # Save best configuration
    best_config = env.get_best_config()
    if best_config:
        save_configuration(best_config, "solar_farm_config.json")
    
    return trainer, results


def scenario_3_windy_coastal_highway():
    """
    Scenario 3: Windy coastal highway with wind turbines.
    Tests the system's ability to optimize for wind power.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Windy Coastal Highway")
    print("=" * 70)
    
    # Windy weather profile
    windy_weather = WeatherProfile(
        min_solar=0.2,
        max_solar=0.7,  # Less sun (coastal fog)
        solar_variance=0.15,
        min_wind=0.4,  # Consistent wind
        max_wind=1.0,  # Strong wind
        wind_variance=0.2,
    )
    
    coastal_costs = CostParameters(
        wind_cost_per_kw=1000.0,  # Cheaper offshore wind
        battery_cost_per_kwh=300.0,
        stranded_vehicle_penalty=15000.0,
    )
    
    env = MultiAgentChargingEnv(
        n_agents=3,
        highway_length_km=250.0,  # Shorter coastal highway
        traffic_range=(40.0, 90.0),
        weather_profile=windy_weather,
        cost_params=coastal_costs,
        simulation_duration_hours=4.0,
        collaboration_weight=0.6,
    )
    
    trainer = MultiAgentPPO(env=env, lr=3e-4)
    
    trainer.train(
        total_episodes=300,
        episodes_per_update=10,
        n_epochs=10,
    )
    
    results = trainer.evaluate(n_episodes=5)
    
    return trainer, results


def scenario_4_urban_constrained():
    """
    Scenario 4: Urban environment with space and power constraints.
    Tests optimization under tight constraints.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Urban Constrained Environment")
    print("=" * 70)
    
    # Moderate weather (urban heat island)
    urban_weather = WeatherProfile(
        min_solar=0.0,
        max_solar=0.8,  # Buildings block some sun
        solar_variance=0.1,
        min_wind=0.1,  # Buildings block wind
        max_wind=0.5,
        wind_variance=0.1,
    )
    
    # Expensive land, expensive grid upgrades
    urban_costs = CostParameters(
        base_station_cost=1000000.0,  # Expensive urban land
        charger_cost_per_kw=600.0,
        waiting_spot_cost=15000.0,
        stranded_vehicle_penalty=25000.0,  # Very bad to strand vehicles in city
    )
    
    env = MultiAgentChargingEnv(
        n_agents=5,  # More stations for dense urban area
        highway_length_km=200.0,  # Short urban highway
        traffic_range=(80.0, 150.0),  # Heavy urban traffic
        weather_profile=urban_weather,
        cost_params=urban_costs,
        simulation_duration_hours=4.0,
        collaboration_weight=0.8,  # High coordination needed
    )
    
    trainer = MultiAgentPPO(env=env, lr=1e-4)  # Conservative learning rate
    
    trainer.train(
        total_episodes=800,
        episodes_per_update=10,
        n_epochs=10,
        eval_interval=100,
    )
    
    results = trainer.evaluate(n_episodes=10)
    
    return trainer, results


def scenario_5_pretrained_evaluation():
    """
    Scenario 5: Load a pretrained model and evaluate on new scenarios.
    Demonstrates model reuse and generalization.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: Pretrained Model Evaluation")
    print("=" * 70)
    
    # Create environment (same as training)
    env = MultiAgentChargingEnv(
        n_agents=3,
        highway_length_km=300.0,
        traffic_range=(30.0, 80.0),
        eval_mode=True,  # Deterministic evaluation
    )
    
    # Create trainer and load model
    trainer = MultiAgentPPO(env=env)
    
    try:
        trainer.load("best_model.pt")
        print("Loaded pretrained model")
    except FileNotFoundError:
        print("No pretrained model found, using random initialization")
    
    # Evaluate on different traffic scenarios
    test_scenarios = [
        ("Low Traffic", 20.0),
        ("Medium Traffic", 60.0),
        ("High Traffic", 100.0),
        ("Rush Hour", 140.0),
    ]
    
    results = {}
    for name, traffic in test_scenarios:
        print(f"\nTesting {name} ({traffic} veh/hr)...")
        
        # Override traffic for this test
        env.current_traffic = traffic
        
        # Run evaluation
        episode_results = []
        for _ in range(3):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                actions = []
                for i, actor in enumerate(trainer.actors):
                    import torch
                    obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                    action, _ = actor.get_action(obs_tensor, deterministic=True)
                    actions.append(action[0])
                
                obs, rewards, terminated, truncated, info = env.step(np.array(actions))
                done = terminated or truncated
                total_reward += np.sum(rewards)
            
            episode_results.append({
                'reward': total_reward,
                'cost': info.get('total_cost', 0),
                'stranded': info.get('stranded_vehicles', 0),
                'blackouts': info.get('blackout_events', 0),
            })
        
        # Average results
        results[name] = {
            'avg_reward': np.mean([r['reward'] for r in episode_results]),
            'avg_cost': np.mean([r['cost'] for r in episode_results]),
            'avg_stranded': np.mean([r['stranded'] for r in episode_results]),
            'avg_blackouts': np.mean([r['blackouts'] for r in episode_results]),
        }
        
        print(f"  Avg Cost: ${results[name]['avg_cost']:,.0f}")
        print(f"  Avg Stranded: {results[name]['avg_stranded']:.1f}")
        print(f"  Avg Blackouts: {results[name]['avg_blackouts']:.1f}")
    
    return trainer, results


def save_configuration(config: dict, filename: str):
    """Save configuration to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            serializable_config[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], '__dict__'):
            # Handle list of EnergyManagerConfig objects
            serializable_config[key] = []
            for item in value:
                if hasattr(item, 'source_configs'):
                    # It's an EnergyManagerConfig
                    sources = []
                    for source in item.source_configs:
                        source_dict = {'type': type(source).__name__}
                        source_dict.update(source.__dict__)
                        sources.append(source_dict)
                    serializable_config[key].append({'source_configs': sources})
                else:
                    serializable_config[key].append(str(item))
        else:
            serializable_config[key] = value
    
    with open(filename, 'w') as f:
        json.dump(serializable_config, f, indent=2)
    
    print(f"Configuration saved to {filename}")


def print_best_configuration(env: MultiAgentChargingEnv):
    """Print the best configuration found in a readable format."""
    best = env.get_best_config()
    if not best:
        print("No best configuration found yet")
        return
    
    print("\n" + "=" * 70)
    print("OPTIMAL CHARGING AREA CONFIGURATION")
    print("=" * 70)
    print(f"Total Cost: ${best['cost']:,.2f}")
    print(f"Highway Length: {env.highway_length_km} km")
    print(f"Number of Stations: {env.n_agents}")
    
    print("\nStation Details:")
    print("-" * 70)
    
    for i in range(env.n_agents):
        pos = best['positions'][i]
        n_chargers = best['n_chargers'][i]
        n_waiting = best['n_waiting'][i]
        config = best['configs'][i]
        
        print(f"\nStation {i+1} at {pos:.1f} km:")
        print(f"  Infrastructure: {n_chargers} chargers, {n_waiting} waiting spots")
        
        # Extract energy sources
        for source in config.source_configs:
            if isinstance(source, GridSourceConfig):
                print(f"  Grid: {source.max_power_kw:.0f} kW max")
            elif isinstance(source, SolarSourceConfig):
                print(f"  Solar: {source.peak_power_kw:.0f} kW peak "
                      f"(sunrise {source.sunrise_hour:.0f}h, sunset {source.sunset_hour:.0f}h)")
            elif isinstance(source, WindSourceConfig):
                print(f"  Wind: {source.base_power_kw:.0f} kW base "
                      f"({source.min_power_kw:.0f}-{source.max_power_kw:.0f} kW range)")
            elif isinstance(source, BatteryStorageConfig):
                print(f"  Battery: {source.capacity_kwh:.0f} kWh "
                      f"({source.max_discharge_rate_kw:.0f} kW max rate, "
                      f"{source.initial_soc*100:.0f}% initial SOC)")
    
    # Calculate spacing
    print("\nSpacing Analysis:")
    print("-" * 70)
    positions = sorted(best['positions'])
    for i in range(len(positions)):
        if i == 0:
            dist_from_start = positions[i]
            print(f"  Entrance to Station 1: {dist_from_start:.1f} km")
        else:
            spacing = positions[i] - positions[i-1]
            print(f"  Station {i} to Station {i+1}: {spacing:.1f} km")
    
    dist_to_end = env.highway_length_km - positions[-1]
    print(f"  Station {len(positions)} to Exit: {dist_to_end:.1f} km")


def compare_strategies():
    """
    Compare different collaboration strategies.
    """
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON: Local vs Global vs Balanced")
    print("=" * 70)
    
    strategies = {
        "Pure Local (0.0)": 0.0,
        "Mostly Local (0.3)": 0.3,
        "Balanced (0.5)": 0.5,
        "Mostly Global (0.7)": 0.7,
        "Pure Global (1.0)": 1.0,
    }
    
    results = {}
    
    for name, collab_weight in strategies.items():
        print(f"\nTesting {name}...")
        
        env = MultiAgentChargingEnv(
            n_agents=3,
            highway_length_km=300.0,
            traffic_range=(40.0, 80.0),
            collaboration_weight=collab_weight,
        )
        
        trainer = MultiAgentPPO(env=env)
        
        # Quick training
        trainer.train(
            total_episodes=200,
            episodes_per_update=10,
            n_epochs=5,
        )
        
        # Evaluate
        eval_results = trainer.evaluate(n_episodes=5)
        results[name] = {
            'avg_reward': eval_results['avg_reward'],
            'avg_cost': eval_results['avg_cost'],
            'best_cost': env.best_cost,
        }
        
        print(f"  Best Cost: ${results[name]['best_cost']:,.0f}")
        print(f"  Avg Reward: {results[name]['avg_reward']:.2f}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Best Cost':>15} {'Avg Reward':>15}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<20} ${res['best_cost']:>13,.0f} {res['avg_reward']:>14.2f}")
    
    return results


def main():
    """
    Main execution - choose which scenario to run.
    """
    print("=" * 70)
    print("MULTI-AGENT PPO CHARGING AREA OPTIMIZATION")
    print("Use Case Examples")
    print("=" * 70)
    
    # Choose scenario
    import sys
    
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
    else:
        print("\nAvailable scenarios:")
        print("  1: basic      - Quick basic training")
        print("  2: solar      - High renewable (solar farm)")
        print("  3: wind       - Windy coastal highway")
        print("  4: urban      - Urban constrained environment")
        print("  5: eval       - Evaluate pretrained model")
        print("  6: compare    - Compare collaboration strategies")
        print("\nUsage: python use_case_example.py [scenario]")
        scenario = input("\nEnter scenario (1-6) or press Enter for basic: ").strip() or "1"
    
    # Run selected scenario
    if scenario in ["1", "basic"]:
        trainer, results = scenario_1_basic_training()
        print_best_configuration(trainer.env)
        
    elif scenario in ["2", "solar"]:
        trainer, results = scenario_2_high_renewable_scenario()
        print_best_configuration(trainer.env)
        
    elif scenario in ["3", "wind"]:
        trainer, results = scenario_3_windy_coastal_highway()
        print_best_configuration(trainer.env)
        
    elif scenario in ["4", "urban"]:
        trainer, results = scenario_4_urban_constrained()
        print_best_configuration(trainer.env)
        
    elif scenario in ["5", "eval"]:
        trainer, results = scenario_5_pretrained_evaluation()
        
    elif scenario in ["6", "compare"]:
        results = compare_strategies()
        
    else:
        print(f"Unknown scenario: {scenario}")
        return
    
    print("\n" + "=" * 70)
    print("SCENARIO COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()