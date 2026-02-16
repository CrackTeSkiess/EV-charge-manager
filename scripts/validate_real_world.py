#!/usr/bin/env python3
"""
Full-year real-world validation of the best RL model.

Runs the pre-trained hierarchical RL model and a rule-based baseline
over 365 days of real weather (PVGIS TMY) and traffic (BASt-based)
data, then compares performance.

Usage:
    python scripts/validate_real_world.py --model-dir models/hierarchical --days 365
    python scripts/validate_real_world.py --days 7   # Quick test
"""

import argparse
import json
import os
import sys
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from ev_charge_manager.energy.manager import (
    CHARGER_RATED_POWER_KW,
    CHARGER_AVG_DRAW_FACTOR,
    EnergyManagerConfig,
    GridSourceConfig,
    SolarSourceConfig,
    WindSourceConfig,
    BatteryStorageConfig,
)
from ev_charge_manager.energy.agent import EnergyManagerAgent, GridPricingSchedule
from ev_charge_manager.energy.hierarchical_manager import HierarchicalEnergyManager
from ev_charge_manager.data.providers import RealWeatherProvider, RealTrafficProvider
from ev_charge_manager.data.baseline_strategy import RuleBasedEnergyManager


def _find_best_config(training_result: Dict, model_dir: str) -> Dict:
    """
    Locate the best_config dict from available training artifacts.

    Tries, in order:
      1. training_result["best_config"] (standard location)
      2. results.json in the same model_dir
      3. Last entry with best_config in training_history.jsonl
    Returns the raw best_config dict, or raises SystemExit with guidance.
    """
    # --- 1. Direct key in training_result ---
    best = training_result.get("best_config")
    if best is not None and isinstance(best, dict):
        return best

    # --- 2. Try results.json (written by hierarchical.py entry point) ---
    results_path = os.path.join(model_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        best = results.get("best_config")
        if best is not None and isinstance(best, dict):
            print("  (loaded best_config from results.json)")
            return best

    # --- 3. Scan training_history.jsonl for last best_config snapshot ---
    history_path = os.path.join(model_dir, "training_history.jsonl")
    if os.path.isfile(history_path):
        last_best = None
        with open(history_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "best_config" in record and record["best_config"]:
                    last_best = record["best_config"]
        if last_best is not None:
            print("  (loaded best_config from training_history.jsonl)")
            return last_best

    # --- Nothing found ---
    print("\nERROR: Could not find 'best_config' in any training artifact.")
    print(f"  Searched: training_result.json, results.json, training_history.jsonl")
    print(f"  In directory: {model_dir}")
    print(f"\n  Keys found in training_result.json: {list(training_result.keys())}")
    print("\n  The model may not have completed macro-agent training.")
    print("  Re-run training with enough macro episodes for the agent to")
    print("  find a valid infrastructure configuration, e.g.:")
    print("    python hierarchical.py --macro-episodes 500 --mode sequential")
    sys.exit(1)


def _parse_positions(raw) -> List[float]:
    """Parse positions from either a numpy array string or a list."""
    if isinstance(raw, list):
        return [float(x) for x in raw]
    # String like "[ 85.96679211 130.         240.82274199]"
    pos_str = str(raw).strip("[] ")
    return [float(x) for x in pos_str.split() if x]


def _parse_config_str(config_str: str) -> EnergyManagerConfig:
    """Parse an EnergyManagerConfig from its string representation."""
    import re
    source_configs = []

    grid_match = re.search(r"GridSourceConfig\(.*?max_power_kw=([\d.]+)", config_str)
    if grid_match:
        source_configs.append(GridSourceConfig(max_power_kw=float(grid_match.group(1))))

    solar_match = re.search(r"SolarSourceConfig\(.*?peak_power_kw=([\d.]+)", config_str)
    if solar_match:
        source_configs.append(SolarSourceConfig(peak_power_kw=float(solar_match.group(1))))

    wind_match = re.search(r"WindSourceConfig\(.*?base_power_kw=([\d.]+)", config_str)
    if wind_match:
        source_configs.append(WindSourceConfig(base_power_kw=float(wind_match.group(1))))

    battery_cap_match = re.search(r"BatteryStorageConfig\(.*?capacity_kwh=([\d.]+)", config_str)
    battery_rate_match = re.search(r"max_charge_rate_kw=([\d.]+)", config_str)
    if battery_cap_match and battery_rate_match:
        source_configs.append(BatteryStorageConfig(
            capacity_kwh=float(battery_cap_match.group(1)),
            max_charge_rate_kw=float(battery_rate_match.group(1)),
            max_discharge_rate_kw=float(battery_rate_match.group(1)),
            initial_soc=0.5,
            min_soc=0.1,
            max_soc=0.95,
            round_trip_efficiency=0.9,
        ))

    return EnergyManagerConfig(source_configs=source_configs)


def _build_default_configs(n_stations: int, run_config: Dict) -> List[EnergyManagerConfig]:
    """Build default EnergyManagerConfig list from run_config when configs are missing."""
    base_cfg = EnergyManagerConfig(source_configs=[
        GridSourceConfig(max_power_kw=run_config.get("grid_kw", 500)),
        SolarSourceConfig(peak_power_kw=run_config.get("solar_kw", 300)),
        BatteryStorageConfig(
            capacity_kwh=run_config.get("battery_kwh", 500),
            max_charge_rate_kw=run_config.get("battery_kw", 100),
            max_discharge_rate_kw=run_config.get("battery_kw", 100),
            initial_soc=0.5, min_soc=0.1, max_soc=0.95,
            round_trip_efficiency=0.9,
        ),
    ])
    return [base_cfg] * n_stations


def parse_best_config(training_result: Dict, run_config: Dict,
                      model_dir: str = "") -> Dict:
    """
    Parse the best infrastructure configuration from training results.

    Searches multiple artifact files for best_config, handles both string
    and list representations, and falls back to run_config defaults when
    energy source configs are missing.

    Returns dict with positions, configs (EnergyManagerConfig), n_chargers, n_waiting.
    """
    best = _find_best_config(training_result, model_dir)

    positions = _parse_positions(best["positions"])
    n_stations = len(positions)
    n_chargers = best.get("n_chargers", [6] * n_stations)
    n_waiting = best.get("n_waiting", [10] * n_stations)

    # Parse energy configs — may be string representations or absent
    raw_configs = best.get("configs")
    if raw_configs and isinstance(raw_configs, list) and len(raw_configs) == n_stations:
        configs = []
        for cfg in raw_configs:
            if isinstance(cfg, str) and "Config(" in cfg:
                configs.append(_parse_config_str(cfg))
            else:
                configs = _build_default_configs(n_stations, run_config)
                print("  (energy configs not parseable, using defaults from run_config)")
                break
    else:
        configs = _build_default_configs(n_stations, run_config)
        print("  (no energy configs in best_config, using defaults from run_config)")

    return {
        "positions": positions,
        "configs": configs,
        "n_chargers": n_chargers,
        "n_waiting": n_waiting,
    }


def create_rl_managers(
    infra: Dict,
    run_config: Dict,
    micro_path: str,
    weather_provider: RealWeatherProvider,
    device: str = "cpu",
) -> List[HierarchicalEnergyManager]:
    """Create HierarchicalEnergyManager instances with loaded micro-RL agents."""
    pricing = GridPricingSchedule(
        off_peak_price=run_config.get("grid_offpeak_price", 0.08),
        shoulder_price=run_config.get("grid_shoulder_price", 0.15),
        peak_price=run_config.get("grid_peak_price", 0.25),
    )

    managers = []
    for i, config in enumerate(infra["configs"]):
        manager = HierarchicalEnergyManager(
            manager_id=f"RL-Station-{i}",
            config=config,
            pricing_schedule=pricing,
            enable_rl_agent=True,
            device=device,
            weather_provider=weather_provider,
        )

        # Load pre-trained micro agent weights
        agent_path = os.path.join(micro_path, f"micro_agent_{i}.pt")
        if manager.rl_agent and os.path.exists(agent_path):
            manager.rl_agent.load(agent_path)
            print(f"  Loaded micro-agent {i} from {agent_path}")
        else:
            print(f"  Warning: No micro-agent found for station {i} (path: {agent_path})")

        managers.append(manager)

    return managers


def create_baseline_managers(
    infra: Dict,
    run_config: Dict,
    weather_provider: RealWeatherProvider,
) -> List[RuleBasedEnergyManager]:
    """Create RuleBasedEnergyManager instances with same infrastructure."""
    pricing = GridPricingSchedule(
        off_peak_price=run_config.get("grid_offpeak_price", 0.08),
        shoulder_price=run_config.get("grid_shoulder_price", 0.15),
        peak_price=run_config.get("grid_peak_price", 0.25),
    )

    managers = []
    for i, config in enumerate(infra["configs"]):
        manager = RuleBasedEnergyManager(
            manager_id=f"Baseline-Station-{i}",
            config=config,
            pricing_schedule=pricing,
            weather_provider=weather_provider,
        )
        managers.append(manager)

    return managers


def run_year(
    managers,
    weather_provider: RealWeatherProvider,
    traffic_provider: RealTrafficProvider,
    n_chargers: List[int],
    run_config: Dict,
    n_days: int = 365,
    label: str = "agent",
) -> Dict:
    """
    Run year-long validation with given managers.

    Returns dict with 'hourly' DataFrame and 'daily' DataFrame of metrics.
    """
    all_hourly = []
    all_daily = []
    n_stations = len(managers)

    for day in range(n_days):
        date = datetime(2024, 1, 1) + timedelta(days=day)

        # Reset daily stats (preserving battery SOC across days)
        for m in managers:
            m.reset_daily_stats()

        for hour in range(24):
            timestamp = date.replace(hour=hour, minute=0, second=0)

            for i, manager in enumerate(managers):
                # Get traffic-based demand using shared charger power constant
                traffic_rate = traffic_provider.get_vehicles_per_hour(timestamp)
                effective_power = CHARGER_RATED_POWER_KW * CHARGER_AVG_DRAW_FACTOR
                traffic_max = run_config.get("traffic_max", 80.0)
                occupancy = min(1.0, traffic_rate / traffic_max)
                demand_kw = n_chargers[i] * effective_power * occupancy
                demand_kw *= random.uniform(0.9, 1.1)  # Light noise
                demand_kw = max(10.0, demand_kw)  # Minimum load

                result = manager.step(
                    timestamp=timestamp,
                    demand_kw=demand_kw,
                    time_step_minutes=60.0,
                    training_mode=False,
                )

                all_hourly.append({
                    "day": day,
                    "date": date.strftime("%Y-%m-%d"),
                    "hour": hour,
                    "station": i,
                    "timestamp": timestamp.isoformat(),
                    "demand_kw": demand_kw,
                    "solar_output_kw": result.get("solar_output", 0.0),
                    "wind_output_kw": result.get("wind_output", 0.0),
                    "grid_power_kw": result.get("grid_power", 0.0),
                    "battery_power_kw": result.get("battery_power", 0.0),
                    "battery_soc": result.get("battery_soc", 0.5),
                    "grid_price": result.get("grid_price", 0.15),
                    "renewable_to_demand_kw": result.get("renewable_to_demand", 0.0),
                    "demand_shortfall_kw": result.get("demand_shortfall", 0.0),
                    "curtailed_kw": result.get("curtailed_renewable", 0.0),
                    "traffic_vph": traffic_rate,
                })

        # Daily summary per station
        for i, m in enumerate(managers):
            summary = m.get_daily_summary()
            summary["day"] = day
            summary["date"] = date.strftime("%Y-%m-%d")
            summary["month"] = date.month
            summary["quarter"] = (date.month - 1) // 3 + 1
            summary["day_of_week"] = date.weekday()
            summary["station"] = i
            all_daily.append(summary)

        # Progress
        if (day + 1) % 30 == 0 or day == 0:
            total_grid_cost = sum(
                m.daily_stats["grid_cost_total"] for m in managers
            )
            print(f"  [{label}] Day {day+1}/{n_days} ({date.strftime('%b %d')}): "
                  f"grid_cost=${total_grid_cost:.2f}")

    return {
        "hourly": pd.DataFrame(all_hourly),
        "daily": pd.DataFrame(all_daily),
    }


def compute_comparison(rl_results: Dict, baseline_results: Dict) -> Dict:
    """Compute annual and quarterly comparison metrics."""
    rl_h = rl_results["hourly"]
    bl_h = baseline_results["hourly"]
    rl_d = rl_results["daily"]
    bl_d = baseline_results["daily"]

    def annual_metrics(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict:
        return {
            "total_grid_cost": float(hourly_df["grid_power_kw"].values @ hourly_df["grid_price"].values),
            "total_grid_energy_mwh": float(hourly_df["grid_power_kw"].sum() / 1000.0),
            "total_renewable_energy_mwh": float(
                (hourly_df["solar_output_kw"] + hourly_df["wind_output_kw"]).sum() / 1000.0
            ),
            "renewable_used_mwh": float(hourly_df["renewable_to_demand_kw"].sum() / 1000.0),
            "renewable_fraction": float(
                hourly_df["renewable_to_demand_kw"].sum()
                / max(1, hourly_df["demand_kw"].sum())
            ),
            "total_demand_mwh": float(hourly_df["demand_kw"].sum() / 1000.0),
            "demand_shortfall_hours": int((hourly_df["demand_shortfall_kw"] > 1.0).sum()),
            "avg_battery_soc": float(hourly_df["battery_soc"].mean()),
            "peak_grid_draw_kw": float(hourly_df["grid_power_kw"].max()),
            "total_curtailed_mwh": float(hourly_df["curtailed_kw"].sum() / 1000.0),
            "arbitrage_profit": float(daily_df["arbitrage_profit"].sum()),
        }

    def quarterly_metrics(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict:
        quarters = {}
        for q in range(1, 5):
            q_months = list(range((q - 1) * 3 + 1, q * 3 + 1))
            # Filter daily by quarter
            q_daily = daily_df[daily_df["quarter"] == q]
            # Filter hourly by matching days
            q_days = q_daily["day"].unique()
            q_hourly = hourly_df[hourly_df["day"].isin(q_days)]

            if len(q_hourly) == 0:
                continue

            quarters[f"Q{q}"] = {
                "grid_cost": float(q_hourly["grid_power_kw"].values @ q_hourly["grid_price"].values),
                "renewable_fraction": float(
                    q_hourly["renewable_to_demand_kw"].sum()
                    / max(1, q_hourly["demand_kw"].sum())
                ),
                "avg_battery_soc": float(q_hourly["battery_soc"].mean()),
                "arbitrage_profit": float(q_daily["arbitrage_profit"].sum()),
            }
        return quarters

    comparison = {
        "rl_annual": annual_metrics(rl_h, rl_d),
        "baseline_annual": annual_metrics(bl_h, bl_d),
        "rl_quarterly": quarterly_metrics(rl_h, rl_d),
        "baseline_quarterly": quarterly_metrics(bl_h, bl_d),
    }

    # Compute deltas
    rl_a = comparison["rl_annual"]
    bl_a = comparison["baseline_annual"]
    comparison["delta"] = {
        key: rl_a[key] - bl_a[key]
        for key in rl_a
        if isinstance(rl_a[key], (int, float))
    }
    comparison["delta_pct"] = {
        key: ((rl_a[key] - bl_a[key]) / max(abs(bl_a[key]), 1e-6)) * 100
        for key in rl_a
        if isinstance(rl_a[key], (int, float)) and abs(bl_a[key]) > 1e-6
    }

    return comparison


def print_comparison(comparison: Dict):
    """Print formatted comparison table."""
    rl = comparison["rl_annual"]
    bl = comparison["baseline_annual"]
    delta = comparison["delta"]

    print("\n" + "=" * 80)
    print("FULL-YEAR VALIDATION RESULTS: RL Agent vs Rule-Based Baseline")
    print("=" * 80)

    fmt = "{:<35s} {:>15s} {:>15s} {:>12s}"
    print(fmt.format("Metric", "RL Agent", "Baseline", "Delta"))
    print("-" * 80)

    rows = [
        ("Total Grid Cost ($)", f"${rl['total_grid_cost']:,.2f}", f"${bl['total_grid_cost']:,.2f}",
         f"${delta['total_grid_cost']:+,.2f}"),
        ("Total Grid Energy (MWh)", f"{rl['total_grid_energy_mwh']:,.1f}", f"{bl['total_grid_energy_mwh']:,.1f}",
         f"{delta['total_grid_energy_mwh']:+,.1f}"),
        ("Renewable Used (MWh)", f"{rl['renewable_used_mwh']:,.1f}", f"{bl['renewable_used_mwh']:,.1f}",
         f"{delta['renewable_used_mwh']:+,.1f}"),
        ("Renewable Fraction (%)", f"{rl['renewable_fraction']:.1%}", f"{bl['renewable_fraction']:.1%}",
         f"{delta['renewable_fraction']:+.1%}"),
        ("Arbitrage Profit ($)", f"${rl['arbitrage_profit']:,.2f}", f"${bl['arbitrage_profit']:,.2f}",
         f"${delta['arbitrage_profit']:+,.2f}"),
        ("Net Operating Cost ($)",
         f"${rl['total_grid_cost'] - rl['arbitrage_profit']:,.2f}",
         f"${bl['total_grid_cost'] - bl['arbitrage_profit']:,.2f}",
         f"${(rl['total_grid_cost'] - rl['arbitrage_profit']) - (bl['total_grid_cost'] - bl['arbitrage_profit']):+,.2f}"),
        ("Demand Shortfall Hours", f"{rl['demand_shortfall_hours']}", f"{bl['demand_shortfall_hours']}",
         f"{delta['demand_shortfall_hours']:+d}"),
        ("Avg Battery SOC", f"{rl['avg_battery_soc']:.3f}", f"{bl['avg_battery_soc']:.3f}",
         f"{delta['avg_battery_soc']:+.3f}"),
        ("Peak Grid Draw (kW)", f"{rl['peak_grid_draw_kw']:,.1f}", f"{bl['peak_grid_draw_kw']:,.1f}",
         f"{delta['peak_grid_draw_kw']:+,.1f}"),
        ("Total Curtailed (MWh)", f"{rl['total_curtailed_mwh']:,.1f}", f"{bl['total_curtailed_mwh']:,.1f}",
         f"{delta['total_curtailed_mwh']:+,.1f}"),
    ]

    for label, rl_val, bl_val, d_val in rows:
        print(fmt.format(label, rl_val, bl_val, d_val))

    # Quarterly breakdown
    print("\n" + "-" * 80)
    print("QUARTERLY BREAKDOWN (Grid Cost / Renewable Fraction)")
    print("-" * 80)
    q_fmt = "{:<10s} {:>20s} {:>20s}"
    print(q_fmt.format("Quarter", "RL Agent", "Baseline"))
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        rl_q = comparison["rl_quarterly"].get(q, {})
        bl_q = comparison["baseline_quarterly"].get(q, {})
        if rl_q and bl_q:
            print(q_fmt.format(
                q,
                f"${rl_q['grid_cost']:,.0f} / {rl_q['renewable_fraction']:.1%}",
                f"${bl_q['grid_cost']:,.0f} / {bl_q['renewable_fraction']:.1%}",
            ))

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Full-year real-world validation of trained RL model"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/hierarchical",
        help="Directory containing trained model and configs"
    )
    parser.add_argument(
        "--weather-csv", type=str, default=None,
        help="Path to weather CSV (default: data/real_world/frankfurt_corridor_weather.csv)"
    )
    parser.add_argument(
        "--traffic-csv", type=str, default=None,
        help="Path to traffic CSV (default: data/real_world/frankfurt_corridor_traffic.csv)"
    )
    parser.add_argument(
        "--days", type=int, default=365,
        help="Number of days to simulate (default: 365)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="validation_output",
        help="Directory for output files"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots"
    )

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = str(project_root / args.model_dir)
    output_dir = str(project_root / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model configuration ---
    print("Loading model configuration...")
    with open(os.path.join(model_dir, "run_config.json"), "r") as f:
        run_config = json.load(f)

    # Try training_result.json first, fall back to results.json
    tr_path = os.path.join(model_dir, "training_result.json")
    results_path = os.path.join(model_dir, "results.json")
    if os.path.isfile(tr_path):
        with open(tr_path, "r") as f:
            training_result = json.load(f)
    elif os.path.isfile(results_path):
        print("  (using results.json — training_result.json not found)")
        with open(results_path, "r") as f:
            training_result = json.load(f)
    else:
        print(f"\nERROR: Neither training_result.json nor results.json found in {model_dir}")
        sys.exit(1)

    infra = parse_best_config(training_result, run_config, model_dir=model_dir)
    print(f"  Stations: {len(infra['positions'])}")
    for i, pos in enumerate(infra["positions"]):
        print(f"    Station {i}: {pos:.1f} km, {infra['n_chargers'][i]} chargers")

    # --- Load real-world data ---
    weather_csv = args.weather_csv or str(project_root / "data/real_world/frankfurt_corridor_weather.csv")
    traffic_csv = args.traffic_csv or str(project_root / "data/real_world/frankfurt_corridor_traffic.csv")

    if not os.path.exists(weather_csv) or not os.path.exists(traffic_csv):
        print("Real-world data not found. Running download script...")
        from scripts.download_real_data import main as download_main
        download_main()

    print(f"Loading weather data from {weather_csv}")
    weather_provider = RealWeatherProvider(weather_csv)
    print(f"Loading traffic data from {traffic_csv}")
    traffic_provider = RealTrafficProvider(traffic_csv)

    # --- Create managers ---
    micro_path = os.path.join(model_dir, "micro_pretrained")

    print("\nCreating RL managers...")
    rl_managers = create_rl_managers(
        infra, run_config, micro_path, weather_provider, device="cpu"
    )

    print("Creating baseline managers...")
    baseline_managers = create_baseline_managers(
        infra, run_config, weather_provider
    )

    # --- Run year-long validation ---
    n_days = min(args.days, 365)
    print(f"\n{'='*60}")
    print(f"Running {n_days}-day validation...")
    print(f"{'='*60}")

    print(f"\n--- RL Agent ({n_days} days) ---")
    # Reset seeds for fair comparison
    random.seed(args.seed)
    np.random.seed(args.seed)
    rl_results = run_year(
        rl_managers, weather_provider, traffic_provider,
        infra["n_chargers"], run_config=run_config, n_days=n_days, label="RL"
    )

    print(f"\n--- Rule-Based Baseline ({n_days} days) ---")
    random.seed(args.seed)
    np.random.seed(args.seed)
    baseline_results = run_year(
        baseline_managers, weather_provider, traffic_provider,
        infra["n_chargers"], run_config=run_config, n_days=n_days, label="Baseline"
    )

    # --- Compare ---
    comparison = compute_comparison(rl_results, baseline_results)
    print_comparison(comparison)

    # --- Save results ---
    print(f"\nSaving results to {output_dir}/")

    rl_results["hourly"].to_csv(os.path.join(output_dir, "rl_hourly.csv"), index=False)
    rl_results["daily"].to_csv(os.path.join(output_dir, "rl_daily.csv"), index=False)
    baseline_results["hourly"].to_csv(os.path.join(output_dir, "baseline_hourly.csv"), index=False)
    baseline_results["daily"].to_csv(os.path.join(output_dir, "baseline_daily.csv"), index=False)

    # Save comparison as JSON (convert numpy types)
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj

    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump(convert_types(comparison), f, indent=2)

    print(f"  rl_hourly.csv ({len(rl_results['hourly'])} rows)")
    print(f"  rl_daily.csv ({len(rl_results['daily'])} rows)")
    print(f"  baseline_hourly.csv ({len(baseline_results['hourly'])} rows)")
    print(f"  baseline_daily.csv ({len(baseline_results['daily'])} rows)")
    print(f"  comparison.json")

    # --- Generate plots ---
    if not args.no_plots:
        try:
            from ev_charge_manager.visualization.validation_plots import generate_all_plots
            print("\nGenerating validation plots...")
            generate_all_plots(rl_results, baseline_results, comparison, output_dir)
            print("  Plots saved to validation_output/plots/")
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
