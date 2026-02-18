"""
Hierarchical Architecture Ablation Study
=========================================
Compares five conditions to validate the contribution of the full
hierarchical dual-RL architecture:

  1. Full Hierarchical  — Macro-PPO (placement) + Micro-AC (energy)
  2. Macro Only         — PPO infrastructure, TOU rule-based energy
  3. Micro Only         — Uniform placement, RL energy arbitrage
  4. No RL              — Uniform placement, TOU rule-based energy
  5. Random Placement   — Random positions, TOU rule-based energy (sanity check)

Usage
-----
  python scripts/run_ablation_hierarchy.py
  python scripts/run_ablation_hierarchy.py --seeds 0 1 2 --micro-episodes 500
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical RL ablation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Random seeds for each condition")
    parser.add_argument("--micro-episodes", type=int, default=500,
                        help="Micro-RL training episodes")
    parser.add_argument("--macro-episodes", type=int, default=300,
                        help="Macro-RL training episodes")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/ablation_hierarchy",
                        help="Output directory")
    parser.add_argument("--no-visualize", action="store_true")
    return parser.parse_args()


def run_full_hierarchical(seed: int, args: argparse.Namespace, out_dir: str) -> dict:
    """Condition 1: Full hierarchical — standard hierarchical.py sequential run."""
    cmd = [
        sys.executable, "hierarchical.py",
        "--seed", str(seed),
        "--mode", "sequential",
        "--micro-episodes", str(args.micro_episodes),
        "--macro-episodes", str(args.macro_episodes),
        "--n-agents", str(args.n_agents),
        "--save-dir", out_dir,
        "--run-name", f"full_seed{seed}",
        "--no-visualize",
    ]
    subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    result_path = os.path.join(out_dir, f"full_seed{seed}", "data", "training_result.json")
    if not os.path.exists(result_path):
        result_path = os.path.join(out_dir, "training_result.json")
    if os.path.exists(result_path):
        with open(result_path) as fh:
            return json.load(fh)
    return {}


def run_macro_only(seed: int, args: argparse.Namespace, out_dir: str) -> dict:
    """
    Condition 2: Macro-Only — PPO for infrastructure, TOU rule-based energy.
    Uses enable_macro_rl=True, enable_micro_rl=False.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    from ev_charge_manager.rl.environment import (
        MultiAgentChargingEnv, WeatherProfile, CostParameters
    )
    from ev_charge_manager.rl.hierarchical_trainer import (
        HierarchicalTrainer, TrainingConfig, TrainingMode
    )

    env = MultiAgentChargingEnv(
        n_agents=args.n_agents,
        highway_length_km=300.0,
        traffic_range=(30.0, 80.0),
        simulation_duration_hours=24.0,
        enable_micro_rl=False,   # No micro-RL: energy handled by TOU inside manager
        enable_macro_rl=True,
    )

    config = TrainingConfig(
        mode=TrainingMode.SEQUENTIAL,
        micro_episodes=0,    # Skip micro training
        macro_episodes=args.macro_episodes,
        output_dir=os.path.join(out_dir, f"macro_only_seed{seed}"),
    )

    # Skip micro phase by manually running only macro
    from ev_charge_manager.rl.ppo_trainer import MultiAgentPPO
    import time
    start = time.time()
    macro_trainer = MultiAgentPPO(env=env, lr=3e-4, device="cpu")
    macro_trainer.train(
        total_episodes=args.macro_episodes,
        episodes_per_update=10,
        save_dir=config.output_dir,
        history_dir=config.output_dir,
    )
    elapsed = time.time() - start
    final_eval = macro_trainer.evaluate(n_episodes=10)

    result = {
        "condition": "macro_only",
        "seed": seed,
        "macro_final_reward": final_eval["avg_reward"],
        "training_time_seconds": elapsed,
        "best_config": env.get_best_config(),
    }
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "ablation_result.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


def run_micro_only(seed: int, args: argparse.Namespace, out_dir: str) -> dict:
    """
    Condition 3: Micro-Only — fixed uniform placement, RL energy arbitrage.
    Uses enable_macro_rl=False (uniform positions), enable_micro_rl=True.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    from ev_charge_manager.rl.hierarchical_trainer import (
        HierarchicalTrainer, MicroRLTrainer, TrainingConfig, TrainingMode
    )
    from ev_charge_manager.energy.agent import GridPricingSchedule
    import time

    pricing = GridPricingSchedule()
    base_config = {
        "battery_kwh": 500.0,
        "battery_kw": 100.0,
        "solar_kw": 300.0,
        "wind_kw": 150.0,
    }
    save_path = os.path.join(out_dir, f"micro_only_seed{seed}")
    os.makedirs(save_path, exist_ok=True)

    start = time.time()
    micro = MicroRLTrainer(
        base_config=base_config,
        n_stations=args.n_agents,
        pricing=pricing,
        device="cpu",
    )
    for ep in range(args.micro_episodes):
        micro.train_episode(ep)

    micro_eval = micro.evaluate(n_episodes=10)
    elapsed = time.time() - start

    result = {
        "condition": "micro_only",
        "seed": seed,
        "micro_final_reward": micro_eval["avg_reward"],
        "avg_grid_cost": micro_eval["avg_grid_cost"],
        "training_time_seconds": elapsed,
        "placement": "uniform",
    }
    with open(os.path.join(save_path, "ablation_result.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


def run_no_rl(seed: int, args: argparse.Namespace, out_dir: str) -> dict:
    """
    Condition 4: No RL — uniform placement + TOU rule-based energy.
    Evaluates TOU baseline over N episodes and reports average metrics.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    from ev_charge_manager.energy.baselines import TOUBaseline
    from ev_charge_manager.energy.agent import GridPricingSchedule
    from datetime import datetime, timedelta

    pricing = GridPricingSchedule()
    battery_kwh = 500.0
    battery_kw = 100.0
    solar_capacity_kw = 300.0
    wind_capacity_kw = 150.0

    episode_costs = []
    episode_arbitrage = []

    for ep in range(20):
        total_grid_cost = 0.0
        total_arbitrage = 0.0

        for station_idx in range(args.n_agents):
            agent = TOUBaseline(pricing_schedule=pricing)
            soc = 0.5

            for hour in range(24):
                price = pricing.get_price(hour)
                demand_kw = 200.0 * (1.5 if 7 <= hour <= 9 or 17 <= hour <= 19 else
                                     0.3 if hour <= 5 else 1.0)
                demand_kw *= np.random.uniform(0.9, 1.1)
                solar_kw = solar_capacity_kw * max(0, 1 - ((hour - 13) / 7) ** 2) * (
                    np.random.uniform(0.95, 1.0) if 6 <= hour <= 18 else 0.0
                )
                wind_kw = wind_capacity_kw * (0.5 + 0.5 * np.sin(2 * np.pi * hour / 24))
                wind_kw *= np.random.uniform(0.7, 1.3)

                result = agent.step(
                    hour=hour,
                    demand_kw=demand_kw,
                    renewable_kw=solar_kw + wind_kw,
                    battery_soc=soc,
                    battery_kwh=battery_kwh,
                    battery_kw=battery_kw,
                )
                soc = float(np.clip(
                    soc + result["battery_change_kwh"] / battery_kwh, 0, 1
                ))
                total_grid_cost += result["grid_power"] * price
                total_arbitrage += max(0, -result["battery_change_kwh"]) * price * 0.9

        episode_costs.append(total_grid_cost)
        episode_arbitrage.append(total_arbitrage)

    save_path = os.path.join(out_dir, f"no_rl_seed{seed}")
    os.makedirs(save_path, exist_ok=True)
    result = {
        "condition": "no_rl",
        "seed": seed,
        "avg_grid_cost": float(np.mean(episode_costs)),
        "avg_arbitrage_profit": float(np.mean(episode_arbitrage)),
        "std_grid_cost": float(np.std(episode_costs, ddof=1)),
        "std_arbitrage_profit": float(np.std(episode_arbitrage, ddof=1)),
        "placement": "uniform",
        "energy_policy": "TOU",
    }
    with open(os.path.join(save_path, "ablation_result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def aggregate_condition(results: list, condition: str) -> dict:
    """Aggregate results across seeds for a condition."""
    def stats(values):
        arr = [v for v in values if v is not None]
        if not arr:
            return {"mean": None, "std": None}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }

    metrics = {
        "condition": condition,
        "n_seeds": len(results),
    }
    for key in ["micro_final_reward", "macro_final_reward", "avg_grid_cost",
                "avg_arbitrage_profit", "training_time_seconds"]:
        values = [r.get(key) for r in results]
        if any(v is not None for v in values):
            metrics[key] = stats(values)

    # Extract best_config cost if present
    costs = []
    arbitrage = []
    for r in results:
        bc = r.get("best_config")
        if isinstance(bc, dict):
            if bc.get("cost") is not None:
                costs.append(bc["cost"])
            if bc.get("arbitrage_profit") is not None:
                arbitrage.append(bc["arbitrage_profit"])
    if costs:
        metrics["best_cost"] = stats(costs)
    if arbitrage:
        metrics["best_arbitrage_profit"] = stats(arbitrage)

    return metrics


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    conditions = {
        "full_hierarchical": [],
        "macro_only": [],
        "micro_only": [],
        "no_rl": [],
    }

    print("=" * 70)
    print("  Hierarchical Architecture Ablation Study")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")

        print("  [1/4] Full Hierarchical...")
        try:
            r = run_full_hierarchical(seed, args, args.output)
            conditions["full_hierarchical"].append(r)
            best = r.get("best_config") or {}
            print(f"        cost=${best.get('cost', 'N/A')}  "
                  f"arbitrage=${best.get('arbitrage_profit', 'N/A')}")
        except Exception as e:
            print(f"        FAILED: {e}")

        print("  [2/4] Macro Only...")
        try:
            r = run_macro_only(seed, args, args.output)
            conditions["macro_only"].append(r)
            best = r.get("best_config") or {}
            print(f"        reward={r.get('macro_final_reward', 'N/A'):.3f}")
        except Exception as e:
            print(f"        FAILED: {e}")

        print("  [3/4] Micro Only...")
        try:
            r = run_micro_only(seed, args, args.output)
            conditions["micro_only"].append(r)
            print(f"        reward={r.get('micro_final_reward', 'N/A'):.3f}  "
                  f"grid_cost=${r.get('avg_grid_cost', 'N/A'):.2f}")
        except Exception as e:
            print(f"        FAILED: {e}")

        print("  [4/4] No RL (TOU baseline)...")
        try:
            r = run_no_rl(seed, args, args.output)
            conditions["no_rl"].append(r)
            print(f"        grid_cost=${r.get('avg_grid_cost', 0):.2f}  "
                  f"arbitrage=${r.get('avg_arbitrage_profit', 0):.2f}")
        except Exception as e:
            print(f"        FAILED: {e}")

    # Aggregate
    aggregated = {}
    for cond, results in conditions.items():
        if results:
            aggregated[cond] = aggregate_condition(results, cond)

    # Print summary table
    print("\n" + "=" * 70)
    print("  Ablation Results (mean ± std)")
    print("=" * 70)
    for cond, stats in aggregated.items():
        print(f"\n  {cond}")
        for k, v in stats.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"    {k:<35}: {v['mean']:.3f} ± {v['std']:.3f}")

    # Save
    out_path = os.path.join(args.output, "ablation_results.json")
    with open(out_path, "w") as fh:
        json.dump({
            "config": vars(args),
            "per_condition": {k: v for k, v in conditions.items()},
            "aggregated": aggregated,
        }, fh, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
