"""
Complete Use Case Example for Hierarchical Multi-Agent RL Charging Optimization

Demonstrates three training modes:
1. SEQUENTIAL (Recommended): Micro-RL first, then Macro-RL
2. SIMULTANEOUS: Both together (faster but unstable)
3. CURRICULUM: Progressive complexity
"""

import argparse
import json
from datetime import datetime
import numpy as np
from typing import Dict

# Import our modules
from ChargingAreaEnv import (
    MultiAgentChargingEnv, 
    WeatherProfile, 
    CostParameters
)
from HierarchicalTrainer import (
    HierarchicalTrainer,
    TrainingConfig,
    TrainingMode,
    train_efficient,
    train_fast,
    train_robust,
)
from EnergyManager import (
    EnergyManagerConfig,
    GridSourceConfig,
    SolarSourceConfig,
    WindSourceConfig,
    BatteryStorageConfig,
)
from HierarchicalEnergyManager import GridPricingSchedule


def scenario_1_sequential_training():
    """
    RECOMMENDED: Sequential training (Micro-RL first, then Macro-RL).
    Most stable and efficient approach.
    """
    print("=" * 70)
    print("SCENARIO 1: Sequential Training (RECOMMENDED)")
    print("Phase 1: Train micro-RL agents (energy arbitrage)")
    print("Phase 2: Train macro-RL agents (infrastructure optimization)")
    print("=" * 70)
    
    # Create environment
    env = MultiAgentChargingEnv(
        n_agents=3,
        highway_length_km=300.0,
        traffic_range=(30.0, 80.0),
        simulation_duration_hours=24.0,  # Full day for arbitrage
        enable_micro_rl=True,
        micro_rl_device="cpu",
    )
    
    # Configure training
    config = TrainingConfig(
        mode=TrainingMode.SEQUENTIAL,
        micro_episodes=500,      # Train micro-agents for 500 days
        macro_episodes=300,      # Then train macro for 300 episodes
        micro_update_frequency=24,
        eval_frequency=50,
        checkpoint_frequency=100,
        output_dir="./models_sequential",
    )
    
    # Create trainer and train
    trainer = HierarchicalTrainer(env, config, device="cpu")
    result = trainer.train()
    
    # Display results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    print(f"Training time: {result['training_time_seconds']/3600:.2f} hours")
    print(f"Micro-RL final reward: {result.get('micro_final_reward', 'N/A'):.2f}")
    print(f"Macro-RL final reward: {result.get('macro_final_reward', 'N/A'):.2f}")
    
    # Show best configuration
    best = result.get('best_config')
    if best:
        print_best_configuration(env, best)
    
    return trainer, result


def scenario_2_simultaneous_training():
    """
    FAST BUT UNSTABLE: Train both macro and micro at same time.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Simultaneous Training (FAST BUT UNSTABLE)")
    print("Training both levels together")
    print("=" * 70)
    
    env = MultiAgentChargingEnv(
        n_agents=3,
        highway_length_km=300.0,
        enable_micro_rl=True,
    )
    
    config = TrainingConfig(
        mode=TrainingMode.SIMULTANEOUS,
        macro_episodes=500,
        simultaneous_micro_updates=1,
        eval_frequency=50,
        output_dir="./models_simultaneous",
    )
    
    trainer = HierarchicalTrainer(env, config)
    result = trainer.train()
    
    print(f"\nTraining completed in {result['training_time_seconds']/3600:.2f} hours")
    
    return trainer, result


def scenario_3_curriculum_training():
    """
    MOST ROBUST: Progressive curriculum learning.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Curriculum Training (MOST ROBUST)")
    print("Progressive complexity increase")
    print("=" * 70)
    
    env = MultiAgentChargingEnv(
        n_agents=3,
        highway_length_km=300.0,
        enable_micro_rl=True,
    )
    
    config = TrainingConfig(
        mode=TrainingMode.CURRICULUM,
        micro_episodes=1000,
        macro_episodes=500,
        curriculum_stages=[
            {'days': 1, 'price_variance': 0.0},   # Simple: fixed prices
            {'days': 3, 'price_variance': 0.3},   # Medium: some variance
            {'days': 7, 'price_variance': 0.6},   # Complex: full variance
        ],
        output_dir="./models_curriculum",
    )
    
    trainer = HierarchicalTrainer(env, config)
    result = trainer.train()
    
    print(f"\nCurriculum training completed")
    print(f"Total time: {result['training_time_seconds']/3600:.2f} hours")
    
    return trainer, result


def scenario_4_pretrained_micro_evaluation():
    """
    Load pre-trained micro-RL and evaluate macro performance.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Pre-trained Micro-RL Evaluation")
    print("Using saved micro-agents, training only macro")
    print("=" * 70)
    
    # First, train or load micro-agents
    micro_path = "./models_sequential/micro_pretrained"
    
    if not os.path.exists(micro_path):
        print("No pre-trained micro-agents found. Training first...")
        # Quick micro training
        from HierarchicalTrainer import MicroRLTrainer
        
        pricing = GridPricingSchedule(
            off_peak_price=0.08,
            shoulder_price=0.15,
            peak_price=0.35,
        )
        
        micro_trainer = MicroRLTrainer(
            base_config={'battery_kwh': 500, 'battery_kw': 100},
            n_stations=3,
            pricing=pricing,
        )
        
        for episode in range(200):
            micro_trainer.train_episode(episode)
        
        micro_trainer.save(micro_path)
        print(f"Micro-agents saved to {micro_path}")
    
    # Now train macro with frozen micro
    env = MultiAgentChargingEnv(
        n_agents=3,
        highway_length_km=300.0,
        enable_micro_rl=True,  # Will use pre-trained micro
    )
    
    config = TrainingConfig(
        mode=TrainingMode.FROZEN_MICRO,
        macro_episodes=400,
        output_dir="./models_frozen_micro",
    )
    
    trainer = HierarchicalTrainer(env, config)
    # Load pre-trained micro into trainer
    if trainer.micro_trainer is None:
        trainer.micro_trainer = MicroRLTrainer(
            base_config={'battery_kwh': 500, 'battery_kw': 100},
            n_stations=3,
            pricing=GridPricingSchedule(),
        )
    trainer.micro_trainer.load(micro_path)
    
    result = trainer.train()
    
    return trainer, result


def scenario_5_arbitrage_analysis():
    """
    Analyze micro-RL arbitrage performance in detail.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: Arbitrage Performance Analysis")
    print("Detailed analysis of micro-RL energy arbitrage")
    print("=" * 70)
    
    # Create simple environment for analysis
    env = MultiAgentChargingEnv(
        n_agents=1,  # Single station for clarity
        highway_length_km=100.0,
        enable_micro_rl=True,
    )
    
    # Fixed configuration for analysis
    fixed_config = {
        'positions': [50.0],
        'configs': [EnergyManagerConfig(source_configs=[
            GridSourceConfig(max_power_kw=500),
            SolarSourceConfig(peak_power_kw=300),
            WindSourceConfig(base_power_kw=100),
            BatteryStorageConfig(
                capacity_kwh=1000,
                max_charge_rate_kw=200,
                max_discharge_rate_kw=200,
            ),
        ])],
        'n_chargers': [6],
        'n_waiting': [10],
    }
    
    # Train micro-RL only
    from HierarchicalTrainer import MicroRLTrainer
    
    pricing = GridPricingSchedule(
        off_peak_price=0.08,
        shoulder_price=0.15,
        peak_price=0.35,
    )
    
    micro_trainer = MicroRLTrainer(
        base_config={'battery_kwh': 1000, 'battery_kw': 200},
        n_stations=1,
        pricing=pricing,
    )
    
    print("Training micro-RL agent...")
    for episode in range(300):
        metrics = micro_trainer.train_episode(episode)
        
        if (episode + 1) % 50 == 0:
            avg_rewards = [float(np.mean(r[-100:])) if len(r) > 0 else 0.0 for r in micro_trainer.episode_rewards]
            avg_reward = np.mean(avg_rewards)
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.2f}")
    
    # Evaluate and analyze
    print("\nEvaluating arbitrage performance...")
    
    # Run one detailed day
    agent = micro_trainer.agents[0]
    agent.reset(initial_soc=0.5)
    
    base_date = datetime(2024, 6, 15, 0, 0)
    hourly_data = []
    
    for hour in range(24):
        timestamp = base_date + timedelta(hours=hour)
        
        # Generate conditions
        demand = 200 + (100 if 7 <= hour <= 9 or 17 <= hour <= 19 else 0)
        solar = 300 * max(0, 1 - abs(hour - 13) / 7) if 6 <= hour <= 18 else 0
        wind = 100 + 50 * np.sin(2 * np.pi * hour / 24)
        price = pricing.get_price(hour)
        
        # Get action
        action, _, _ = agent.select_action(timestamp, solar, wind, demand, price, deterministic=True)
        flows = agent.interpret_action(action, solar, wind, demand)
        
        # Update SOC
        battery_change = flows['battery_power'] * (1/24)
        agent.current_soc = float(np.clip(
            agent.current_soc + battery_change / agent.battery_capacity_kwh,
            0.0, 1.0
        ))
        
        hourly_data.append({
            'hour': hour,
            'price': price,
            'demand': demand,
            'solar': solar,
            'wind': wind,
            'battery_action': 'CHARGE' if flows['battery_power'] > 0 else 'DISCHARGE' if flows['battery_power'] < 0 else 'IDLE',
            'battery_power': abs(flows['battery_power']),
            'grid_used': flows['grid_power'],
            'battery_soc': agent.current_soc,
        })
    
    # Print analysis
    print("\nHourly Arbitrage Strategy:")
    print("-" * 90)
    print(f"{'Hour':>4} | {'Price':>6} | {'Demand':>6} | {'Solar':>5} | {'Wind':>5} | {'Action':>10} | {'Grid':>5} | {'SOC':>5}")
    print("-" * 90)
    
    for h in hourly_data:
        print(f"{h['hour']:4d} | ${h['price']:4.2f} | {h['demand']:6.1f} | "
              f"{h['solar']:5.1f} | {h['wind']:5.1f} | {h['battery_action']:>10} | "
              f"{h['grid_used']:5.1f} | {h['battery_soc']*100:4.0f}%")
    
    # Calculate arbitrage profit
    charging_cost = sum(h['price'] * h['battery_power'] / 24 
                       for h in hourly_data if h['battery_action'] == 'CHARGE')
    discharging_value = sum(h['price'] * h['battery_power'] / 24 
                           for h in hourly_data if h['battery_action'] == 'DISCHARGE')
    arbitrage_profit = discharging_value - charging_cost
    
    print("-" * 90)
    print(f"\nArbitrage Analysis:")
    print(f"  Charging cost: ${charging_cost:.2f}")
    print(f"  Discharging value: ${discharging_value:.2f}")
    print(f"  Net arbitrage profit: ${arbitrage_profit:.2f}")
    
    return micro_trainer, hourly_data


def scenario_6_compare_training_modes():
    """
    Compare all three training modes on same problem.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 6: Training Mode Comparison")
    print("Comparing Sequential, Simultaneous, and Curriculum")
    print("=" * 70)
    
    results = {}
    
    # Sequential
    print("\n--- Testing SEQUENTIAL ---")
    _, result_seq = scenario_1_sequential_training()
    results['sequential'] = {
        'time': result_seq['training_time_seconds'],
        'reward': result_seq.get('macro_final_reward', 0),
        'cost': result_seq.get('best_config', {}).get('cost', float('inf')),
    }
    
    # Fast (simultaneous) - reduced episodes for comparison
    print("\n--- Testing SIMULTANEOUS ---")
    env = MultiAgentChargingEnv(n_agents=3, enable_micro_rl=True)
    config = TrainingConfig(
        mode=TrainingMode.SIMULTANEOUS,
        macro_episodes=200,  # Reduced for comparison
        output_dir="./models_compare_sim",
    )
    trainer = HierarchicalTrainer(env, config)
    result_sim = trainer.train()
    results['simultaneous'] = {
        'time': result_sim['training_time_seconds'],
        'reward': result_sim.get('macro_final_reward', 0),
        'cost': result_sim.get('best_config', {}).get('cost', float('inf')),
    }
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Mode':<15} {'Time (hrs)':>12} {'Final Reward':>14} {'Best Cost':>14}")
    print("-" * 70)
    for mode, res in results.items():
        print(f"{mode:<15} {res['time']/3600:>11.2f} {res['reward']:>14.2f} ${res['cost']:>13,.0f}")
    
    return results


def print_best_configuration(env: MultiAgentChargingEnv, best: Dict):
    """Print best configuration in readable format."""
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)
    print(f"Total Cost: ${best['cost']:,.2f}")
    print(f"Grid Cost: ${best.get('grid_cost', 0):,.2f}")
    print(f"Arbitrage Profit: ${best.get('arbitrage_profit', 0):,.2f}")
    print(f"Highway Length: {env.highway_length_km} km")
    
    print("\nStation Details:")
    print("-" * 70)
    
    for i in range(env.n_agents):
        pos = best['positions'][i]
        n_chargers = best['n_chargers'][i]
        n_waiting = best['n_waiting'][i]
        
        # Extract energy config
        config = best['configs'][i]
        energy_stats = {
            'grid': 0, 'solar': 0, 'wind': 0, 
            'battery_rate': 0, 'battery_cap': 0
        }
        
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
                
        
        print(f"\n  Station {i+1} at {pos:.1f} km:")
        print(f"    Infrastructure: {n_chargers} chargers, {n_waiting} waiting spots")
        print(f"    Grid: {energy_stats['grid']:.0f} kW")
        print(f"    Solar: {energy_stats['solar']:.0f} kW")
        print(f"    Wind: {energy_stats['wind']:.0f} kW")
        print(f"    Battery: {energy_stats['battery_cap']:.0f} kWh "
              f"({energy_stats['battery_rate']:.0f} kW rate)")
    
    # Spacing
    print("\n  Spacing:")
    positions = sorted(best['positions'])
    for i in range(len(positions)):
        if i == 0:
            print(f"    Entrance to Station 1: {positions[i]:.1f} km")
        else:
            print(f"    Station {i} to {i+1}: {positions[i] - positions[i-1]:.1f} km")
    print(f"    Station {len(positions)} to Exit: {env.highway_length_km - positions[-1]:.1f} km")


def main():
    """Main execution."""
    print("=" * 70)
    print("HIERARCHICAL MULTI-AGENT RL FOR CHARGING OPTIMIZATION")
    print("Use Case Examples")
    print("=" * 70)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=[
        '1', 'sequential',
        '2', 'simultaneous', 
        '3', 'curriculum',
        '4', 'pretrained',
        '5', 'arbitrage',
        '6', 'compare',
        'efficient', 'fast', 'robust'
    ], default='1', help='Scenario to run')
    parser.add_argument('--stations', type=int, default=3)
    parser.add_argument('--output', type=str, default='./models')
    
    args = parser.parse_args()
    
    scenario = args.scenario.lower()
    
    if scenario in ['1', 'sequential']:
        scenario_1_sequential_training()
    elif scenario in ['2', 'simultaneous']:
        scenario_2_simultaneous_training()
    elif scenario in ['3', 'curriculum']:
        scenario_3_curriculum_training()
    elif scenario in ['4', 'pretrained']:
        scenario_4_pretrained_micro_evaluation()
    elif scenario in ['5', 'arbitrage']:
        scenario_5_arbitrage_analysis()
    elif scenario in ['6', 'compare']:
        scenario_6_compare_training_modes()
    elif scenario == 'efficient':
        train_efficient(args.stations, output_dir=args.output)
    elif scenario == 'fast':
        train_fast(args.stations, output_dir=args.output)
    elif scenario == 'robust':
        train_robust(args.stations, output_dir=args.output)
    else:
        print(f"Unknown scenario: {scenario}")
        print("Use --help for available scenarios")


if __name__ == "__main__":
    import os
    from datetime import timedelta
    main()