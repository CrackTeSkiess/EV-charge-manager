"""
Multi-Seed RL vs. Baseline Comparison with Statistical Tests
=============================================================
Runs both the RL policy and each rule-based baseline (Naive, TOU, MPC) for
N independent seeds, then applies the Wilcoxon signed-rank test to determine
whether the RL improvement in arbitrage profit is statistically significant.

Usage
-----
  python scripts/run_comparison_seeds.py
  python scripts/run_comparison_seeds.py --seeds 0 1 2 3 4 5 6 7 8 9
  python scripts/run_comparison_seeds.py --episodes 50 --output results/comparison
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed RL vs. baseline comparison with significance tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)),
                        help="List of random seeds")
    parser.add_argument("--episodes", type=int, default=24,
                        help="Number of simulation hours per evaluation episode")
    parser.add_argument("--output", type=str, default="results/comparison",
                        help="Output directory for comparison results")
    parser.add_argument("--n-stations", type=int, default=3)
    return parser.parse_args()


def run_policy_episode(policy_type: str, seed: int, episodes: int, n_stations: int) -> dict:
    """
    Run a single evaluation episode for a given policy type and seed.

    policy_type: 'rl' | 'naive' | 'tou' | 'mpc'
    Returns: dict with arbitrage_profit, grid_cost, renewable_fraction, avg_battery_soc
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    from ev_charge_manager.energy.agent import GridPricingSchedule
    from ev_charge_manager.energy.baselines import NaiveBaseline, TOUBaseline, MPCBaseline
    from datetime import datetime, timedelta

    pricing = GridPricingSchedule(
        off_peak_price=0.08,
        shoulder_price=0.15,
        peak_price=0.35,
    )

    # Battery and solar/wind parameters (same as default training config)
    battery_kwh = 500.0
    battery_kw = 100.0
    solar_capacity_kw = 300.0
    wind_capacity_kw = 150.0

    total_arbitrage_profit = 0.0
    total_grid_cost = 0.0
    total_renewable_kwh = 0.0
    total_demand_kwh = 0.0
    soc_readings = []

    for station_idx in range(n_stations):
        if policy_type == "rl":
            try:
                from ev_charge_manager.energy.agent import EnergyManagerAgent
                agent = EnergyManagerAgent(
                    agent_id=f"rl_{station_idx}",
                    pricing_schedule=pricing,
                    battery_capacity_kwh=battery_kwh,
                    battery_max_power_kw=battery_kw,
                    device="cpu",
                )
                # Try to load pre-trained weights if available
                micro_path = os.path.join(
                    str(Path(__file__).parent.parent),
                    "models", "hierarchical", "micro_pretrained",
                    f"micro_agent_{station_idx}.pt",
                )
                if os.path.exists(micro_path):
                    agent.load(micro_path)
            except Exception:
                # Fall back to TOU if RL not available
                policy_type = "tou"
        elif policy_type == "naive":
            agent = NaiveBaseline(pricing_schedule=pricing)
        elif policy_type == "tou":
            agent = TOUBaseline(pricing_schedule=pricing)
        elif policy_type == "mpc":
            agent = MPCBaseline(pricing_schedule=pricing, lookahead_hours=3)
        else:
            raise ValueError(f"Unknown policy: {policy_type}")

        current_soc = 0.5
        base_date = datetime(2024, 6, 15, 0, 0)

        station_grid_cost = 0.0
        station_renewable_kwh = 0.0
        station_demand_kwh = 0.0
        station_arbitrage = 0.0

        for hour in range(episodes):
            timestamp = base_date + timedelta(hours=hour)
            price = pricing.get_price(hour % 24)

            # Synthetic demand (same as MicroRLTrainer._generate_demand)
            base_demand = 200.0
            if 7 <= (hour % 24) <= 9 or 17 <= (hour % 24) <= 19:
                multiplier = 1.5
            elif (hour % 24) <= 5:
                multiplier = 0.3
            else:
                multiplier = 1.0
            demand_kw = base_demand * multiplier * np.random.uniform(0.9, 1.1)

            # Synthetic solar
            h = hour % 24
            if 6 <= h <= 18:
                solar_avail = max(0.0, 1 - ((h - 13) / 7) ** 2)
            else:
                solar_avail = 0.0
            solar_kw = solar_capacity_kw * solar_avail * np.random.uniform(0.95, 1.0)

            # Synthetic wind
            wind_pattern = 0.5 + 0.5 * np.sin(2 * np.pi * h / 24)
            wind_kw = wind_capacity_kw * wind_pattern * np.random.uniform(0.7, 1.3)

            renewable_avail = solar_kw + wind_kw

            # Policy decision
            if hasattr(agent, "select_action"):
                # RL agent
                action, _, _ = agent.select_action(
                    timestamp, solar_kw / solar_capacity_kw,
                    wind_kw / wind_capacity_kw, demand_kw, price,
                    deterministic=True,
                )
                flows = agent.interpret_action(action, solar_kw, wind_kw, demand_kw)
                battery_change = flows["battery_power"] * 1.0
                agent.current_soc = float(np.clip(
                    agent.current_soc + battery_change / agent.battery_capacity_kwh, 0, 1
                ))
                current_soc = agent.current_soc
                grid_power = flows["grid_power"]
                renewable_used = min(renewable_avail, demand_kw - grid_power)
            else:
                # Rule-based agent
                result = agent.step(
                    hour=hour % 24,
                    demand_kw=demand_kw,
                    renewable_kw=renewable_avail,
                    battery_soc=current_soc,
                    battery_kwh=battery_kwh,
                    battery_kw=battery_kw,
                )
                grid_power = result["grid_power"]
                battery_change = result["battery_change_kwh"]
                current_soc = float(np.clip(
                    current_soc + battery_change / battery_kwh, 0, 1
                ))
                renewable_used = result.get("renewable_used", min(renewable_avail, demand_kw))

            # Economics
            grid_cost = grid_power * price
            # Arbitrage profit: sell back at current price when battery discharges to grid
            battery_discharge_kw = max(0.0, -battery_change)
            arbitrage_revenue = battery_discharge_kw * price * 0.9  # 90% efficiency credit

            station_grid_cost += grid_cost
            station_renewable_kwh += renewable_used / 60.0  # kWh (hourly steps)
            station_demand_kwh += demand_kw / 60.0
            station_arbitrage += arbitrage_revenue
            soc_readings.append(current_soc)

        total_grid_cost += station_grid_cost
        total_renewable_kwh += station_renewable_kwh
        total_demand_kwh += station_demand_kwh
        total_arbitrage_profit += station_arbitrage

    return {
        "policy": policy_type,
        "seed": seed,
        "arbitrage_profit": total_arbitrage_profit,
        "grid_cost": total_grid_cost,
        "renewable_fraction": total_renewable_kwh / max(1.0, total_demand_kwh),
        "avg_battery_soc": float(np.mean(soc_readings)) if soc_readings else 0.5,
    }


def wilcoxon_test(a: list, b: list) -> dict:
    """Paired Wilcoxon signed-rank test between arrays a and b (a - b)."""
    try:
        from scipy.stats import wilcoxon, ttest_rel
        import numpy as np
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        differences = a_arr - b_arr

        stat, p_val = wilcoxon(a_arr, b_arr)

        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(a_arr, ddof=1) ** 2 + np.std(b_arr, ddof=1) ** 2) / 2)
        cohens_d = (np.mean(a_arr) - np.mean(b_arr)) / pooled_std if pooled_std > 0 else 0.0

        # 95% CI for mean difference (using t-distribution)
        from scipy.stats import t
        n = len(differences)
        se = np.std(differences, ddof=1) / np.sqrt(n)
        t_crit = t.ppf(0.975, df=n - 1)
        ci_lower = float(np.mean(differences) - t_crit * se)
        ci_upper = float(np.mean(differences) + t_crit * se)

        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "mean_difference": float(np.mean(differences)),
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "significant_at_0.05": bool(p_val < 0.05),
        }
    except ImportError:
        # scipy not available — use simple t-test via numpy
        import numpy as np
        a_arr = np.array(a, dtype=float)
        b_arr = np.array(b, dtype=float)
        differences = a_arr - b_arr
        n = len(differences)
        mean_diff = float(np.mean(differences))
        se = float(np.std(differences, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        # Approximate p-value using normal distribution
        z = mean_diff / se if se > 0 else 0.0
        import math
        p_approx = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        return {
            "statistic": z,
            "p_value": p_approx,
            "cohens_d": None,
            "mean_difference": mean_diff,
            "ci_95_lower": mean_diff - 1.96 * se,
            "ci_95_upper": mean_diff + 1.96 * se,
            "significant_at_0.05": bool(p_approx < 0.05),
            "note": "scipy not available; used normal approximation",
        }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    policies = ["rl", "naive", "tou", "mpc"]
    all_results: dict = {p: [] for p in policies}

    print("=" * 70)
    print("  Multi-Seed RL vs. Baseline Comparison")
    print(f"  Seeds: {args.seeds}")
    print(f"  Policies: {policies}")
    print("=" * 70)

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        for policy in policies:
            try:
                result = run_policy_episode(policy, seed, args.episodes, args.n_stations)
                all_results[policy].append(result)
                print(f"  {policy:8s}: arbitrage=${result['arbitrage_profit']:8.2f}  "
                      f"grid_cost=${result['grid_cost']:8.2f}  "
                      f"renewable={result['renewable_fraction']:.2%}")
            except Exception as exc:
                print(f"  {policy:8s}: FAILED — {exc}")

    # Statistical tests (RL vs. each baseline)
    print("\n" + "=" * 70)
    print("  Statistical Tests (Wilcoxon signed-rank, RL vs. baseline)")
    print("=" * 70)

    tests = {}
    for baseline in ["naive", "tou", "mpc"]:
        rl_profits = [r["arbitrage_profit"] for r in all_results["rl"]]
        baseline_profits = [r["arbitrage_profit"] for r in all_results[baseline]]
        if len(rl_profits) >= 3 and len(baseline_profits) >= 3:
            n = min(len(rl_profits), len(baseline_profits))
            test_result = wilcoxon_test(rl_profits[:n], baseline_profits[:n])
            tests[f"rl_vs_{baseline}"] = test_result
            sig = "**SIGNIFICANT**" if test_result["significant_at_0.05"] else "not significant"
            print(f"  RL vs {baseline:6s}: p={test_result['p_value']:.4f}  "
                  f"mean_diff={test_result['mean_difference']:+.2f}  "
                  f"d={test_result.get('cohens_d') or 'N/A'}  {sig}")

    # Aggregate summaries per policy
    summary = {}
    for policy in policies:
        results = all_results[policy]
        if not results:
            continue
        arbitrage = [r["arbitrage_profit"] for r in results]
        grid_cost = [r["grid_cost"] for r in results]
        renewable = [r["renewable_fraction"] for r in results]
        soc = [r["avg_battery_soc"] for r in results]
        summary[policy] = {
            "n_seeds": len(results),
            "arbitrage_profit": {
                "mean": float(np.mean(arbitrage)), "std": float(np.std(arbitrage, ddof=1) if len(arbitrage) > 1 else 0),
            },
            "grid_cost": {
                "mean": float(np.mean(grid_cost)), "std": float(np.std(grid_cost, ddof=1) if len(grid_cost) > 1 else 0),
            },
            "renewable_fraction": {
                "mean": float(np.mean(renewable)), "std": float(np.std(renewable, ddof=1) if len(renewable) > 1 else 0),
            },
            "avg_battery_soc": {
                "mean": float(np.mean(soc)), "std": float(np.std(soc, ddof=1) if len(soc) > 1 else 0),
            },
        }

    print("\n" + "=" * 70)
    print("  Summary (mean ± std across seeds)")
    print("=" * 70)
    print(f"  {'Policy':<8}  {'Arbitrage ($)':>16}  {'Grid Cost ($)':>16}  {'Renewable':>12}")
    print("-" * 70)
    for policy in policies:
        if policy not in summary:
            continue
        s = summary[policy]
        arb = s["arbitrage_profit"]
        grd = s["grid_cost"]
        ren = s["renewable_fraction"]
        print(f"  {policy:<8}  {arb['mean']:>10.2f}±{arb['std']:>5.2f}  "
              f"{grd['mean']:>10.2f}±{grd['std']:>5.2f}  "
              f"{ren['mean']:>8.2%}±{ren['std']:>5.2%}")

    # Save everything
    output = {
        "config": {
            "seeds": args.seeds,
            "episodes_per_seed": args.episodes,
            "n_stations": args.n_stations,
        },
        "per_seed_results": all_results,
        "summary": summary,
        "statistical_tests": tests,
    }
    out_path = os.path.join(args.output, "comparison_results.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
