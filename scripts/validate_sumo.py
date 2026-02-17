#!/usr/bin/env python3
"""
SUMO/TraCI validation of the trained hierarchical RL model.

Uses SUMO as a high-fidelity traffic simulation backend to validate the
macro agent's infrastructure design while the micro agent manages energy
step-by-step each simulated hour.

Flow:
    1. Load trained macro config (station positions, chargers, energy sources)
    2. Generate SUMO highway network from that config
    3. Run SUMO traffic via TraCI — vehicles drive, stop to charge
    4. Each hour: query real demand from SUMO → feed to micro RL agent
    5. Collect metrics and optionally compare to internal simulation

Usage:
    python scripts/validate_sumo.py --model-dir models/hierarchical --days 1
    python scripts/validate_sumo.py --model-dir models/hierarchical --sumo-binary sumo-gui
    python scripts/validate_sumo.py --model-dir models/hierarchical --days 7 --compare-internal
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from ev_charge_manager.sumo import require_sumo, SUMO_AVAILABLE
from ev_charge_manager.sumo.network_generator import SUMONetworkGenerator
from ev_charge_manager.sumo.traci_bridge import TraCIBridge
from ev_charge_manager.energy.manager import (
    CHARGER_RATED_POWER_KW,
    CHARGER_AVG_DRAW_FACTOR,
    EnergyManagerConfig,
    GridSourceConfig,
    SolarSourceConfig,
    WindSourceConfig,
    BatteryStorageConfig,
)
from ev_charge_manager.data.traffic_profiles import (
    WEEKDAY_HOURLY_FRACTIONS,
    PEAK_HOURLY_FRACTION,
)
from ev_charge_manager.energy.agent import GridPricingSchedule
from ev_charge_manager.energy.hierarchical_manager import HierarchicalEnergyManager


# ---------------------------------------------------------------------------
# Model loading (mirrors scripts/validate_real_world.py)
# ---------------------------------------------------------------------------

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
                # Not a parseable string — use defaults
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


def create_managers(
    infra: Dict,
    run_config: Dict,
    micro_path: str,
    device: str = "cpu",
) -> List[HierarchicalEnergyManager]:
    """
    Create HierarchicalEnergyManager instances with loaded micro-RL agents.

    No weather_provider is used — demand comes from SUMO instead of weather-
    based synthetic generation.
    """
    pricing = GridPricingSchedule(
        off_peak_price=run_config.get("grid_offpeak_price", 0.08),
        shoulder_price=run_config.get("grid_shoulder_price", 0.15),
        peak_price=run_config.get("grid_peak_price", 0.25),
    )

    managers = []
    for i, config in enumerate(infra["configs"]):
        manager = HierarchicalEnergyManager(
            manager_id=f"SUMO-Station-{i}",
            config=config,
            pricing_schedule=pricing,
            enable_rl_agent=True,
            device=device,
        )

        # Load pre-trained micro agent weights
        agent_path = os.path.join(micro_path, f"micro_agent_{i}.pt")
        if manager.rl_agent and os.path.exists(agent_path):
            manager.rl_agent.load(agent_path)
            print(f"  Loaded micro-agent {i} from {agent_path}")
        else:
            print(f"  Warning: No micro-agent found for station {i} "
                  f"(path: {agent_path})")

        managers.append(manager)

    return managers


# ---------------------------------------------------------------------------
# Internal simulation comparison (optional)
# ---------------------------------------------------------------------------

def run_internal_simulation(
    infra: Dict,
    run_config: Dict,
    micro_path: str,
    n_days: int,
    seed: int,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run the project's internal simulation for comparison.

    Uses the same HierarchicalEnergyManager with synthetic demand,
    mirroring _run_simulation_with_micro_rl() from environment.py.
    """
    random.seed(seed)
    np.random.seed(seed)

    managers = create_managers(infra, run_config, micro_path, device)
    n_stations = len(managers)
    n_chargers = infra["n_chargers"]

    all_hourly = []
    base_date = datetime(2024, 6, 15, 0, 0)
    base_traffic = 60.0

    for day in range(n_days):
        for m in managers:
            m.reset_daily_stats()

        for hour in range(24):
            timestamp = base_date + timedelta(days=day, hours=hour)

            # BASt hourly shape (matching environment.py training logic)
            hourly_shape = WEEKDAY_HOURLY_FRACTIONS[hour] / PEAK_HOURLY_FRACTION

            for i, manager in enumerate(managers):
                # Must match the training formula in environment.py
                effective_power = CHARGER_RATED_POWER_KW * CHARGER_AVG_DRAW_FACTOR
                traffic_max = run_config.get("traffic_max", 80.0)
                occupancy = min(1.5, hourly_shape * (base_traffic / traffic_max))
                demand_noise = random.uniform(0.8, 1.2)
                demand_kw = n_chargers[i] * effective_power * occupancy * demand_noise

                result = manager.step(
                    timestamp=timestamp,
                    demand_kw=demand_kw,
                    time_step_minutes=60.0,
                    training_mode=False,
                )

                all_hourly.append({
                    "day": day,
                    "hour": hour,
                    "station": i,
                    "source": "internal",
                    "demand_kw": demand_kw,
                    "grid_power_kw": result.get("grid_power", 0.0),
                    "solar_output_kw": result.get("solar_output", 0.0),
                    "wind_output_kw": result.get("wind_output", 0.0),
                    "battery_soc": result.get("battery_soc", 0.5),
                    "grid_price": result.get("grid_price", 0.15),
                    "renewable_to_demand_kw": result.get("renewable_to_demand", 0.0),
                    "demand_shortfall_kw": result.get("demand_shortfall", 0.0),
                    "battery_power_kw": result.get("battery_power", 0.0),
                    "reward": result.get("reward", 0.0),
                })

    return pd.DataFrame(all_hourly)


# ---------------------------------------------------------------------------
# SUMO simulation
# ---------------------------------------------------------------------------

def run_sumo_simulation(
    sumo_config_path: str,
    charging_edge_ids: List[str],
    managers: List[HierarchicalEnergyManager],
    n_chargers: List[int],
    n_days: int = 1,
    sumo_binary: str = "sumo",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the SUMO simulation, feeding real vehicle demand to micro agents.

    Each simulated hour:
      1. SUMO advances 3600 seconds of traffic
      2. TraCI queries charging demand at each station
      3. Micro agent decides energy flows via manager.step()
      4. Hourly metrics are recorded

    Returns DataFrame with hourly metrics.
    """
    bridge = TraCIBridge(
        sumo_config_path=sumo_config_path,
        charging_edge_ids=charging_edge_ids,
        n_chargers=n_chargers,
        sumo_binary=sumo_binary,
    )

    all_hourly = []
    n_stations = len(managers)
    base_date = datetime(2024, 6, 15, 0, 0)

    try:
        print("  Starting SUMO...")
        bridge.start()

        total_hours = n_days * 24
        for step_hour in range(total_hours):
            day = step_hour // 24
            hour = step_hour % 24
            timestamp = base_date + timedelta(days=day, hours=hour)

            # Reset daily stats at start of each day
            if hour == 0:
                for m in managers:
                    m.reset_daily_stats()

            # Advance SUMO by one hour
            target_time_s = float((step_hour + 1) * 3600)
            bridge.advance_to(target_time_s)

            # Query real demand from SUMO
            demands, snapshots = bridge.get_all_station_demands()
            traffic_stats = bridge.get_traffic_stats()

            # Feed demand to each station's micro agent
            for i, manager in enumerate(managers):
                result = manager.step(
                    timestamp=timestamp,
                    demand_kw=demands[i],
                    time_step_minutes=60.0,
                    training_mode=False,
                )

                snap = snapshots[i]

                all_hourly.append({
                    "day": day,
                    "hour": hour,
                    "station": i,
                    "source": "sumo",
                    "demand_kw": demands[i],
                    "grid_power_kw": result.get("grid_power", 0.0),
                    "solar_output_kw": result.get("solar_output", 0.0),
                    "wind_output_kw": result.get("wind_output", 0.0),
                    "battery_soc": result.get("battery_soc", 0.5),
                    "grid_price": result.get("grid_price", 0.15),
                    "renewable_to_demand_kw": result.get("renewable_to_demand", 0.0),
                    "demand_shortfall_kw": result.get("demand_shortfall", 0.0),
                    "battery_power_kw": result.get("battery_power", 0.0),
                    "reward": result.get("reward", 0.0),
                    # SUMO-specific columns
                    "sumo_vehicles_charging": snap.n_vehicles_charging,
                    "sumo_vehicles_waiting": snap.n_vehicles_waiting,
                    "sumo_active_vehicles": traffic_stats.active_vehicles,
                    "sumo_teleported": traffic_stats.teleported_vehicles,
                })

            # Progress reporting
            if hour == 0 or hour == 12:
                total_demand = sum(demands)
                print(f"  Day {day+1} Hour {hour:02d}: "
                      f"demand={total_demand:.0f} kW, "
                      f"vehicles={traffic_stats.active_vehicles}, "
                      f"sim_time={bridge.get_simulation_time():.0f}s")

    finally:
        bridge.close()
        print("  SUMO closed.")

    return pd.DataFrame(all_hourly)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(hourly_df: pd.DataFrame) -> Dict:
    """Compute summary metrics from hourly data."""
    if hourly_df.empty:
        return {}

    total_grid_energy_kwh = hourly_df["grid_power_kw"].sum()  # 1-hour steps
    total_demand_kwh = hourly_df["demand_kw"].sum()
    total_renewable_kwh = hourly_df["renewable_to_demand_kw"].sum()
    grid_cost = float(
        (hourly_df["grid_power_kw"] * hourly_df["grid_price"]).sum()
    )
    shortfall_hours = int((hourly_df["demand_shortfall_kw"] > 1.0).sum())

    return {
        "total_grid_cost": grid_cost,
        "total_grid_energy_kwh": total_grid_energy_kwh,
        "total_demand_kwh": total_demand_kwh,
        "total_renewable_kwh": total_renewable_kwh,
        "renewable_fraction": float(
            total_renewable_kwh / max(1.0, total_demand_kwh)
        ),
        "demand_shortfall_hours": shortfall_hours,
        "avg_battery_soc": float(hourly_df["battery_soc"].mean()),
        "peak_grid_draw_kw": float(hourly_df["grid_power_kw"].max()),
        "total_reward": float(hourly_df["reward"].sum()),
    }


def print_results(metrics: Dict, label: str = "SUMO Validation") -> None:
    """Print a formatted results table."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    fmt = "  {:<35s} {:>20s}"
    rows = [
        ("Total Grid Cost ($)", f"${metrics.get('total_grid_cost', 0):,.2f}"),
        ("Total Grid Energy (kWh)", f"{metrics.get('total_grid_energy_kwh', 0):,.1f}"),
        ("Total Demand (kWh)", f"{metrics.get('total_demand_kwh', 0):,.1f}"),
        ("Renewable Used (kWh)", f"{metrics.get('total_renewable_kwh', 0):,.1f}"),
        ("Renewable Fraction", f"{metrics.get('renewable_fraction', 0):.1%}"),
        ("Demand Shortfall Hours", f"{metrics.get('demand_shortfall_hours', 0)}"),
        ("Avg Battery SOC", f"{metrics.get('avg_battery_soc', 0):.3f}"),
        ("Peak Grid Draw (kW)", f"{metrics.get('peak_grid_draw_kw', 0):,.1f}"),
        ("Total Micro-RL Reward", f"{metrics.get('total_reward', 0):,.2f}"),
    ]
    for label_str, value in rows:
        print(fmt.format(label_str, value))
    print(f"{'='*70}")


def print_comparison(sumo_metrics: Dict, internal_metrics: Dict) -> None:
    """Print side-by-side SUMO vs internal simulation comparison."""
    print(f"\n{'='*80}")
    print("  SUMO vs Internal Simulation Comparison")
    print(f"{'='*80}")

    fmt = "  {:<30s} {:>15s} {:>15s} {:>12s}"
    print(fmt.format("Metric", "SUMO", "Internal", "Delta"))
    print(f"  {'-'*72}")

    keys_labels = [
        ("total_grid_cost", "Grid Cost ($)", "${:,.2f}"),
        ("total_grid_energy_kwh", "Grid Energy (kWh)", "{:,.1f}"),
        ("renewable_fraction", "Renewable Fraction", "{:.1%}"),
        ("demand_shortfall_hours", "Shortfall Hours", "{}"),
        ("avg_battery_soc", "Avg Battery SOC", "{:.3f}"),
        ("peak_grid_draw_kw", "Peak Grid (kW)", "{:,.1f}"),
        ("total_reward", "Micro-RL Reward", "{:,.2f}"),
    ]

    for key, label, val_fmt in keys_labels:
        s_val = sumo_metrics.get(key, 0)
        i_val = internal_metrics.get(key, 0)
        delta = s_val - i_val
        print(fmt.format(
            label,
            val_fmt.format(s_val),
            val_fmt.format(i_val),
            f"{delta:+,.2f}" if isinstance(delta, float) else f"{delta:+d}",
        ))

    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SUMO/TraCI validation of trained hierarchical RL model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/hierarchical",
        help="Directory containing trained model and configs",
    )
    parser.add_argument(
        "--output-dir", type=str, default="validation_output/sumo",
        help="Directory for output files and SUMO network",
    )
    parser.add_argument(
        "--days", type=int, default=1,
        help="Number of days to simulate",
    )
    parser.add_argument(
        "--sumo-binary", type=str, default="sumo",
        choices=["sumo", "sumo-gui"],
        help="SUMO binary (sumo-gui for visual debugging)",
    )
    parser.add_argument(
        "--base-aadt", type=float, default=50000.0,
        help="Annual Average Daily Traffic (total vehicles)",
    )
    parser.add_argument(
        "--ev-penetration", type=float, default=0.15,
        help="Fraction of traffic that is EVs",
    )
    parser.add_argument(
        "--highway-length", type=float, default=None,
        help="Override highway length in km (default: from model config)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--compare-internal", action="store_true", default=True,
        help="Run the internal simulation for side-by-side comparison (default: True)",
    )
    parser.add_argument(
        "--no-compare-internal", action="store_true",
        help="Skip internal simulation comparison",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Check SUMO availability
    require_sumo()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = str(project_root / args.model_dir)
    output_dir = str(project_root / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load trained model configuration
    # ------------------------------------------------------------------
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
    highway_length = args.highway_length or run_config.get("highway_length", 300.0)

    print(f"  Highway: {highway_length:.0f} km")
    print(f"  Stations: {len(infra['positions'])}")
    for i, pos in enumerate(infra["positions"]):
        print(f"    Station {i}: {pos:.1f} km, "
              f"{infra['n_chargers'][i]} chargers, "
              f"{infra['n_waiting'][i]} waiting spots")

    # ------------------------------------------------------------------
    # 2. Generate SUMO network from macro agent's best config
    # ------------------------------------------------------------------
    print("\nGenerating SUMO network...")
    sumo_dir = os.path.join(output_dir, "sumo_network")
    generator = SUMONetworkGenerator(
        highway_length_km=highway_length,
        station_positions=infra["positions"],
        n_chargers=infra["n_chargers"],
        n_waiting=infra["n_waiting"],
        output_dir=sumo_dir,
        base_aadt=args.base_aadt,
        ev_penetration=args.ev_penetration,
        simulation_seconds=args.days * 86400,
        seed=args.seed,
    )
    sumo_files = generator.generate()
    print(f"  Network files written to {sumo_dir}/")
    for key, path in sumo_files.items():
        print(f"    {key}: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # 3. Create managers with loaded micro-agents
    # ------------------------------------------------------------------
    micro_path = os.path.join(model_dir, "micro_pretrained")
    print("\nCreating energy managers with micro-RL agents...")
    managers = create_managers(infra, run_config, micro_path)

    # ------------------------------------------------------------------
    # 4. Run SUMO simulation with micro-agent energy management
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Running {args.days}-day SUMO validation...")
    print(f"{'='*60}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    sumo_hourly = run_sumo_simulation(
        sumo_config_path=sumo_files["config"],
        charging_edge_ids=generator.charging_edge_ids,
        managers=managers,
        n_chargers=infra["n_chargers"],
        n_days=args.days,
        sumo_binary=args.sumo_binary,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 5. Compute and display metrics
    # ------------------------------------------------------------------
    sumo_metrics = compute_metrics(sumo_hourly)
    print_results(sumo_metrics, "SUMO/TraCI Validation Results")

    # ------------------------------------------------------------------
    # 6. Optional: compare against internal simulation
    # ------------------------------------------------------------------
    internal_hourly = None
    internal_metrics = None

    if args.compare_internal and not args.no_compare_internal:
        print(f"\n--- Running internal simulation for comparison ---")
        random.seed(args.seed)
        np.random.seed(args.seed)

        internal_hourly = run_internal_simulation(
            infra=infra,
            run_config=run_config,
            micro_path=micro_path,
            n_days=args.days,
            seed=args.seed,
        )
        internal_metrics = compute_metrics(internal_hourly)
        print_results(internal_metrics, "Internal Simulation Results")
        print_comparison(sumo_metrics, internal_metrics)

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    print(f"\nSaving results to {output_dir}/")

    sumo_hourly.to_csv(os.path.join(output_dir, "sumo_hourly.csv"), index=False)
    print(f"  sumo_hourly.csv ({len(sumo_hourly)} rows)")

    if internal_hourly is not None:
        internal_hourly.to_csv(
            os.path.join(output_dir, "internal_hourly.csv"), index=False
        )
        print(f"  internal_hourly.csv ({len(internal_hourly)} rows)")

    # Save metrics summary
    summary = {
        "sumo_metrics": sumo_metrics,
        "config": {
            "highway_length_km": highway_length,
            "n_stations": len(infra["positions"]),
            "station_positions": infra["positions"],
            "n_chargers": infra["n_chargers"],
            "n_waiting": infra["n_waiting"],
            "days": args.days,
            "base_aadt": args.base_aadt,
            "ev_penetration": args.ev_penetration,
            "seed": args.seed,
        },
    }
    if internal_metrics is not None:
        summary["internal_metrics"] = internal_metrics
        summary["delta"] = {
            key: sumo_metrics.get(key, 0) - internal_metrics.get(key, 0)
            for key in sumo_metrics
            if isinstance(sumo_metrics.get(key, 0), (int, float))
        }

    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    with open(os.path.join(output_dir, "sumo_validation_summary.json"), "w") as f:
        json.dump(convert_types(summary), f, indent=2)
    print(f"  sumo_validation_summary.json")

    # ------------------------------------------------------------------
    # 8. Generate plots (optional)
    # ------------------------------------------------------------------
    if not args.no_plots and not sumo_hourly.empty:
        try:
            _generate_plots(sumo_hourly, internal_hourly, output_dir)
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

    print("\nSUMO validation complete.")


def _generate_plots(
    sumo_hourly: pd.DataFrame,
    internal_hourly: Optional[pd.DataFrame],
    output_dir: str,
) -> None:
    """Generate comprehensive SUMO vs Internal comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    has_internal = internal_hourly is not None and not internal_hourly.empty

    # Colour palette
    C_SUMO = "#1f77b4"
    C_INT = "#d62728"
    C_SOLAR = "#f4a261"
    C_WIND = "#2a9d8f"
    C_GRID = "#6c757d"

    # Pre-aggregate by hour (mean across stations and days)
    sumo_by_hour = sumo_hourly.groupby("hour").mean(numeric_only=True)
    int_by_hour = (
        internal_hourly.groupby("hour").mean(numeric_only=True)
        if has_internal else None
    )

    # =====================================================================
    # Figure 1: Comparison Dashboard (main deliverable)
    # =====================================================================
    if has_internal:
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(
            "SUMO vs Internal Simulation — Validation Dashboard",
            fontsize=16, fontweight="bold", y=0.98,
        )
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.30)

        # --- Panel 1: Demand overlay (top-left, spans 2 cols) ---
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(sumo_by_hour.index, sumo_by_hour["demand_kw"],
                 color=C_SUMO, marker="o", ms=5, lw=2, label="SUMO")
        ax1.plot(int_by_hour.index, int_by_hour["demand_kw"],
                 color=C_INT, marker="s", ms=5, lw=2, ls="--", label="Internal")
        ax1.fill_between(
            sumo_by_hour.index,
            sumo_by_hour["demand_kw"], int_by_hour["demand_kw"],
            alpha=0.12, color="grey", label="Gap",
        )
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Mean Demand (kW)")
        ax1.set_title("Hourly Charging Demand")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_xticks(range(24))

        # --- Panel 2: Summary metrics bar chart (top-right) ---
        ax2 = fig.add_subplot(gs[0, 2])
        sumo_m = compute_metrics(sumo_hourly)
        int_m = compute_metrics(internal_hourly)
        metric_keys = [
            ("total_grid_cost", "Grid Cost ($)"),
            ("total_demand_kwh", "Demand (kWh)"),
            ("total_renewable_kwh", "Renewable (kWh)"),
            ("peak_grid_draw_kw", "Peak Grid (kW)"),
        ]
        labels = [lbl for _, lbl in metric_keys]
        sumo_vals = [sumo_m.get(k, 0) for k, _ in metric_keys]
        int_vals = [int_m.get(k, 0) for k, _ in metric_keys]
        x_pos = np.arange(len(labels))
        bar_w = 0.35
        ax2.barh(x_pos - bar_w / 2, sumo_vals, bar_w,
                 color=C_SUMO, label="SUMO", alpha=0.85)
        ax2.barh(x_pos + bar_w / 2, int_vals, bar_w,
                 color=C_INT, label="Internal", alpha=0.85)
        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_title("Key Metrics Comparison")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, axis="x", alpha=0.3)

        # --- Panel 3: Grid power overlay (mid-left) ---
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(sumo_by_hour.index, sumo_by_hour["grid_power_kw"],
                 color=C_SUMO, lw=2, label="SUMO")
        ax3.plot(int_by_hour.index, int_by_hour["grid_power_kw"],
                 color=C_INT, lw=2, ls="--", label="Internal")
        ax3.set_xlabel("Hour of Day")
        ax3.set_ylabel("Grid Power (kW)")
        ax3.set_title("Grid Draw")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.5, 23.5)

        # --- Panel 4: Energy mix — SUMO (mid-center) ---
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.stackplot(
            sumo_by_hour.index,
            sumo_by_hour["solar_output_kw"],
            sumo_by_hour["wind_output_kw"],
            sumo_by_hour["grid_power_kw"],
            labels=["Solar", "Wind", "Grid"],
            colors=[C_SOLAR, C_WIND, C_GRID], alpha=0.75,
        )
        ax4.set_xlabel("Hour of Day")
        ax4.set_ylabel("Power (kW)")
        ax4.set_title("Energy Mix — SUMO")
        ax4.legend(loc="upper left", fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-0.5, 23.5)

        # --- Panel 5: Energy mix — Internal (mid-right) ---
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.stackplot(
            int_by_hour.index,
            int_by_hour["solar_output_kw"],
            int_by_hour["wind_output_kw"],
            int_by_hour["grid_power_kw"],
            labels=["Solar", "Wind", "Grid"],
            colors=[C_SOLAR, C_WIND, C_GRID], alpha=0.75,
        )
        ax5.set_xlabel("Hour of Day")
        ax5.set_ylabel("Power (kW)")
        ax5.set_title("Energy Mix — Internal")
        ax5.legend(loc="upper left", fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(-0.5, 23.5)

        # --- Panel 6: Battery SOC overlay (bottom-left) ---
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.plot(sumo_by_hour.index, sumo_by_hour["battery_soc"],
                 color=C_SUMO, lw=2, label="SUMO")
        ax6.plot(int_by_hour.index, int_by_hour["battery_soc"],
                 color=C_INT, lw=2, ls="--", label="Internal")
        ax6.set_xlabel("Hour of Day")
        ax6.set_ylabel("Battery SOC")
        ax6.set_title("Mean Battery SOC")
        ax6.set_ylim(0, 1)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(-0.5, 23.5)

        # --- Panel 7: Per-station demand comparison (bottom-center) ---
        ax7 = fig.add_subplot(gs[2, 1])
        stations = sorted(sumo_hourly["station"].unique())
        sumo_st = sumo_hourly.groupby("station")["demand_kw"].mean()
        int_st = internal_hourly.groupby("station")["demand_kw"].mean()
        x_st = np.arange(len(stations))
        ax7.bar(x_st - 0.2, [sumo_st.get(s, 0) for s in stations], 0.4,
                color=C_SUMO, label="SUMO", alpha=0.85)
        ax7.bar(x_st + 0.2, [int_st.get(s, 0) for s in stations], 0.4,
                color=C_INT, label="Internal", alpha=0.85)
        ax7.set_xlabel("Station")
        ax7.set_ylabel("Mean Demand (kW)")
        ax7.set_title("Per-Station Avg Demand")
        ax7.set_xticks(x_st)
        ax7.set_xticklabels([f"S{s}" for s in stations])
        ax7.legend(fontsize=8)
        ax7.grid(True, axis="y", alpha=0.3)

        # --- Panel 8: Reward overlay (bottom-right) ---
        ax8 = fig.add_subplot(gs[2, 2])
        sumo_reward = sumo_hourly.groupby("hour")["reward"].mean()
        int_reward = internal_hourly.groupby("hour")["reward"].mean()
        ax8.plot(sumo_reward.index, sumo_reward.values,
                 color=C_SUMO, lw=2, label="SUMO")
        ax8.plot(int_reward.index, int_reward.values,
                 color=C_INT, lw=2, ls="--", label="Internal")
        ax8.set_xlabel("Hour of Day")
        ax8.set_ylabel("Mean Reward")
        ax8.set_title("Micro-RL Reward")
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim(-0.5, 23.5)

        fig.savefig(os.path.join(plots_dir, "comparison_dashboard.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  comparison_dashboard.png")

    # =====================================================================
    # Figure 2: Demand profile (standalone, always generated)
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    sumo_demand = sumo_by_hour["demand_kw"]
    ax.plot(sumo_demand.index, sumo_demand.values,
            color=C_SUMO, marker="o", ms=5, lw=2, label="SUMO")
    if has_internal:
        int_demand = int_by_hour["demand_kw"]
        ax.plot(int_demand.index, int_demand.values,
                color=C_INT, marker="s", ms=5, lw=2, ls="--", label="Internal")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Avg Demand (kW)")
    ax.set_title("Hourly Charging Demand: SUMO vs Internal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(24))
    fig.savefig(os.path.join(plots_dir, "demand_profile.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # Figure 3: Energy sources side-by-side
    # =====================================================================
    n_cols = 2 if has_internal else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), squeeze=False)
    for ax_idx, (df_h, label) in enumerate([
        (sumo_by_hour, "SUMO"),
        (int_by_hour, "Internal"),
    ]):
        if df_h is None:
            continue
        ax = axes[0, ax_idx]
        ax.stackplot(
            df_h.index,
            df_h["solar_output_kw"],
            df_h["wind_output_kw"],
            df_h["grid_power_kw"],
            labels=["Solar", "Wind", "Grid"],
            colors=[C_SOLAR, C_WIND, C_GRID], alpha=0.75,
        )
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"{label} — Energy Sources")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 23.5)
    fig.savefig(os.path.join(plots_dir, "energy_sources.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # Figure 4: Battery SOC trajectories per station
    # =====================================================================
    n_cols = 2 if has_internal else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), squeeze=False)
    for ax_idx, (df, label) in enumerate([
        (sumo_hourly, "SUMO"),
        (internal_hourly if has_internal else None, "Internal"),
    ]):
        if df is None:
            continue
        ax = axes[0, ax_idx]
        for station in sorted(df["station"].unique()):
            sdf = df[df["station"] == station]
            ax.plot(range(len(sdf)), sdf["battery_soc"].values,
                    label=f"Station {station}")
        ax.set_xlabel("Simulation Hour")
        ax.set_ylabel("Battery SOC")
        ax.set_title(f"Battery SOC — {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    fig.savefig(os.path.join(plots_dir, "battery_soc.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # =====================================================================
    # Figure 5: SUMO vehicles at stations
    # =====================================================================
    if "sumo_vehicles_charging" in sumo_hourly.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        for station in sorted(sumo_hourly["station"].unique()):
            sdf = sumo_hourly[sumo_hourly["station"] == station]
            ax.plot(range(len(sdf)), sdf["sumo_vehicles_charging"].values,
                    label=f"Station {station} (charging)")
            ax.plot(range(len(sdf)), sdf["sumo_vehicles_waiting"].values,
                    linestyle="--", alpha=0.5,
                    label=f"Station {station} (waiting)")
        ax.set_xlabel("Simulation Hour")
        ax.set_ylabel("Vehicle Count")
        ax.set_title("SUMO — Vehicles at Charging Stations")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(plots_dir, "sumo_vehicles.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # =====================================================================
    # Figure 6: Correlation scatter (SUMO vs Internal)
    # =====================================================================
    if has_internal:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Merge on (day, hour, station) for point-by-point comparison
        merged = sumo_hourly.merge(
            internal_hourly,
            on=["day", "hour", "station"],
            suffixes=("_sumo", "_int"),
        )

        for ax, col, label in zip(
            axes,
            ["demand_kw", "grid_power_kw", "battery_soc"],
            ["Demand (kW)", "Grid Power (kW)", "Battery SOC"],
        ):
            s_col = f"{col}_sumo"
            i_col = f"{col}_int"
            if s_col in merged.columns and i_col in merged.columns:
                ax.scatter(merged[i_col], merged[s_col],
                           alpha=0.4, s=20, color=C_SUMO)
                lims = [
                    min(merged[i_col].min(), merged[s_col].min()),
                    max(merged[i_col].max(), merged[s_col].max()),
                ]
                if lims[0] < lims[1]:
                    ax.plot(lims, lims, "k--", alpha=0.5, lw=1, label="y = x")
                ax.set_xlabel(f"Internal {label}")
                ax.set_ylabel(f"SUMO {label}")
                ax.set_title(f"{label}: Correlation")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                # Annotate R-squared
                if len(merged) > 1:
                    corr = merged[[s_col, i_col]].corr().iloc[0, 1]
                    r_sq = corr ** 2
                    ax.text(
                        0.05, 0.92, f"R² = {r_sq:.3f}",
                        transform=ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                    )

        fig.suptitle("SUMO vs Internal — Point-by-Point Correlation",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(plots_dir, "correlation_scatter.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  correlation_scatter.png")

    print(f"  Plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
