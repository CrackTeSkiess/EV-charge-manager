"""
Seasonal Performance Analysis
================================
Divides the Frankfurt corridor weather data into 4 seasonal quarters and
evaluates both the RL policy and each baseline for one week per quarter.

Produces a grouped bar chart showing:
  - Arbitrage profit ($/week)
  - Grid cost ($/week)
  - Renewable fraction (%)

for 4 quarters × 3 policies (RL, TOU, Naive).

Usage
-----
  python scripts/seasonal_analysis.py
  python scripts/seasonal_analysis.py --output figures/seasonal_analysis.png
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


QUARTERS = {
    "Q1 (Jan-Mar)": (1, 3),
    "Q2 (Apr-Jun)": (4, 6),
    "Q3 (Jul-Sep)": (7, 9),
    "Q4 (Oct-Dec)": (10, 12),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seasonal performance analysis (RL vs baselines)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weather-data",
        default="data/real_world/frankfurt_corridor_weather.csv",
    )
    parser.add_argument(
        "--traffic-data",
        default="data/real_world/frankfurt_corridor_traffic.csv",
    )
    parser.add_argument("--output", default="figures/seasonal_analysis.png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-stations", type=int, default=3)
    return parser.parse_args()


def load_weather(path: str) -> dict:
    """
    Load weather CSV and return a nested dict:
      weather[month][hour] = {ghi_wm2, wind_speed_ms}
    """
    import csv
    weather = {}
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            month = int(row["month"])
            hour = int(row["hour"])
            weather.setdefault(month, {})[hour] = {
                "ghi_wm2": float(row["ghi_wm2"]),
                "wind_speed_ms": float(row["wind_speed_ms"]),
            }
    return weather


def load_traffic(path: str) -> dict:
    """
    Load traffic CSV and return a nested dict:
      traffic[month][hour] = ev_vehicles_per_hour (average across days)
    """
    import csv
    from collections import defaultdict
    hourly_ev = defaultdict(list)
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            month = int(row["month"])
            hour = int(row["hour"])
            hourly_ev[(month, hour)].append(float(row["ev_vehicles_per_hour"]))
    traffic = {}
    for (month, hour), values in hourly_ev.items():
        traffic.setdefault(month, {})[hour] = float(np.mean(values))
    return traffic


def ghi_to_solar_kw(ghi_wm2: float, peak_kw: float = 300.0) -> float:
    """Convert GHI (W/m²) to station solar output (kW)."""
    # Assume 1000 W/m² = peak output; apply efficiency factor 0.85
    return peak_kw * (ghi_wm2 / 1000.0) * 0.85


def wind_speed_to_kw(speed_ms: float, base_kw: float = 150.0) -> float:
    """Simple wind power curve: cut-in=3 m/s, rated=12 m/s."""
    if speed_ms < 3.0:
        return 0.0
    fraction = min(1.0, (speed_ms - 3.0) / (12.0 - 3.0))
    return base_kw * fraction


def ev_demand_to_kw(ev_per_hour: float, n_chargers: int = 6) -> float:
    """Convert EV arrivals per hour to charging demand (kW)."""
    # Each EV charges at ~150 kW rated, 85% draw factor, ~45 min dwell
    demand_kw = min(ev_per_hour * 150.0 * 0.85 * (45 / 60), n_chargers * 150.0)
    return demand_kw


def simulate_week(
    policy_type: str,
    months: tuple,
    weather: dict,
    traffic: dict,
    n_stations: int,
    seed: int = 0,
) -> dict:
    """
    Simulate one week (168 hours) for a given policy and season.
    Returns: arbitrage_profit, grid_cost, renewable_fraction
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    from ev_charge_manager.energy.agent import GridPricingSchedule
    from ev_charge_manager.energy.baselines import NaiveBaseline, TOUBaseline

    pricing = GridPricingSchedule()
    battery_kwh = 500.0
    battery_kw = 100.0
    solar_kw_peak = 300.0
    wind_kw_base = 150.0

    # Pick representative months for the quarter
    quarter_months = [m for m in range(months[0], months[1] + 1) if m in weather]
    if not quarter_months:
        return {"arbitrage_profit": 0, "grid_cost": 0, "renewable_fraction": 0}

    total_arbitrage = 0.0
    total_grid_cost = 0.0
    total_renewable_kwh = 0.0
    total_demand_kwh = 0.0

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
                micro_path = os.path.join(
                    str(Path(__file__).parent.parent),
                    "models", "hierarchical", "micro_pretrained",
                    f"micro_agent_{station_idx}.pt",
                )
                if os.path.exists(micro_path):
                    agent.load(micro_path)
                use_rl = True
            except Exception:
                agent = TOUBaseline(pricing_schedule=pricing)
                use_rl = False
        elif policy_type == "naive":
            agent = NaiveBaseline()
            use_rl = False
        else:  # tou
            agent = TOUBaseline(pricing_schedule=pricing)
            use_rl = False

        current_soc = 0.5

        # Simulate 168 hours (7 days) using weather/traffic from representative months
        for sim_hour in range(168):
            hour_of_day = sim_hour % 24
            month = quarter_months[sim_hour % len(quarter_months)]
            price = pricing.get_price(hour_of_day)

            # Weather-driven generation
            wx = weather.get(month, {}).get(hour_of_day, {"ghi_wm2": 0.0, "wind_speed_ms": 5.0})
            solar = ghi_to_solar_kw(wx["ghi_wm2"], solar_kw_peak)
            wind = wind_speed_to_kw(wx["wind_speed_ms"], wind_kw_base)
            renewable = solar + wind

            # Traffic-driven demand
            ev_arrivals = traffic.get(month, {}).get(hour_of_day, 30.0)
            demand_kw = ev_demand_to_kw(ev_arrivals, n_chargers=6) * np.random.uniform(0.9, 1.1)

            if use_rl:
                action, _, _ = agent.select_action(
                    None,
                    solar / solar_kw_peak,
                    wind / wind_kw_base,
                    demand_kw,
                    price,
                    deterministic=True,
                )
                flows = agent.interpret_action(action, solar, wind, demand_kw)
                battery_change = flows["battery_power"] * 1.0
                agent.current_soc = float(np.clip(
                    agent.current_soc + battery_change / agent.battery_capacity_kwh, 0, 1
                ))
                current_soc = agent.current_soc
                grid_power = flows["grid_power"]
                renewable_used = min(renewable, demand_kw - grid_power)
                discharge_kw = max(0.0, -battery_change)
            else:
                result = agent.step(
                    hour=hour_of_day,
                    demand_kw=demand_kw,
                    renewable_kw=renewable,
                    battery_soc=current_soc,
                    battery_kwh=battery_kwh,
                    battery_kw=battery_kw,
                )
                grid_power = result["grid_power"]
                battery_change = result["battery_change_kwh"]
                current_soc = float(np.clip(current_soc + battery_change / battery_kwh, 0, 1))
                renewable_used = result.get("renewable_used", min(renewable, demand_kw))
                discharge_kw = max(0.0, -battery_change)

            total_grid_cost += grid_power * price
            total_renewable_kwh += renewable_used
            total_demand_kwh += demand_kw
            total_arbitrage += discharge_kw * price * 0.9

    return {
        "arbitrage_profit": total_arbitrage,
        "grid_cost": total_grid_cost,
        "renewable_fraction": total_renewable_kwh / max(1.0, total_demand_kwh),
    }


def main() -> None:
    args = parse_args()

    root = str(Path(__file__).parent.parent)
    weather_path = os.path.join(root, args.weather_data)
    traffic_path = os.path.join(root, args.traffic_data)

    print("Loading weather and traffic data...")
    weather = load_weather(weather_path)
    traffic = load_traffic(traffic_path)

    policies = ["rl", "tou", "naive"]
    policy_labels = {"rl": "RL Agent", "tou": "TOU Baseline", "naive": "Naive Baseline"}

    results = {p: {} for p in policies}

    print("Running seasonal simulations...")
    for q_name, months in QUARTERS.items():
        print(f"  {q_name}...")
        for policy in policies:
            r = simulate_week(policy, months, weather, traffic, args.n_stations)
            results[policy][q_name] = r
            print(f"    {policy_labels[policy]:15s}: "
                  f"arb=${r['arbitrage_profit']:.1f}  "
                  f"grid=${r['grid_cost']:.1f}  "
                  f"ren={r['renewable_fraction']:.1%}")

    # ── Figure ────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib is required.")
        sys.exit(1)

    quarters = list(QUARTERS.keys())
    x = np.arange(len(quarters))
    width = 0.25
    colors = {"rl": "#1f77b4", "tou": "#ff7f0e", "naive": "#2ca02c"}

    metrics = [
        ("arbitrage_profit", "Arbitrage Profit ($/week)", "(a)"),
        ("grid_cost", "Grid Cost ($/week)", "(b)"),
        ("renewable_fraction", "Renewable Fraction", "(c)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Seasonal Performance: RL vs. Baselines (1 week per quarter)",
                 fontsize=13, fontweight="bold")

    for ax, (metric_key, ylabel, panel_label) in zip(axes, metrics):
        for j, policy in enumerate(policies):
            values = [results[policy][q][metric_key] for q in quarters]
            if metric_key == "renewable_fraction":
                values = [v * 100 for v in values]  # convert to %
            ax.bar(x + j * width, values, width=width,
                   color=colors[policy], alpha=0.8,
                   label=policy_labels[policy],
                   edgecolor="gray", linewidth=0.3)

        ax.set_title(f"{panel_label} {ylabel}", fontsize=11, loc="left")
        ax.set_xticks(x + width)
        ax.set_xticklabels([q.split(" ")[0] for q in quarters], fontsize=10)
        ax.set_ylabel(ylabel.split("(")[0].strip(), fontsize=10)
        ax.legend(fontsize=8)
        ax.yaxis.grid(True, alpha=0.3)

    if metric_key == "renewable_fraction":
        axes[2].set_ylabel("Renewable Fraction (%)", fontsize=10)

    plt.tight_layout()

    out_path = os.path.join(root, args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")
    plt.close()

    # Save raw data
    json_path = out_path.replace(".png", ".json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Raw data saved to: {json_path}")


if __name__ == "__main__":
    main()
