"""
Validation-specific visualizations for RL vs Baseline comparison.

Generates plots comparing the RL agent against a rule-based baseline
over a full year of real-world weather and traffic data.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def generate_all_plots(
    rl_results: Dict,
    baseline_results: Dict,
    comparison: Dict,
    output_dir: str,
):
    """Generate all validation plots."""
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    rl_h = rl_results["hourly"]
    bl_h = baseline_results["hourly"]
    rl_d = rl_results["daily"]
    bl_d = baseline_results["daily"]

    plot_annual_cost_comparison(comparison, plot_dir)
    plot_monthly_renewable_fraction(rl_h, bl_h, rl_d, bl_d, plot_dir)
    plot_sample_day_profiles(rl_h, bl_h, plot_dir)
    plot_battery_soc_trajectory(rl_h, bl_h, plot_dir)
    plot_cost_heatmap(rl_h, bl_h, plot_dir)
    plot_arbitrage_monthly(rl_d, bl_d, plot_dir)
    plot_summary_dashboard(comparison, plot_dir)

    print(f"  Generated 7 validation plots in {plot_dir}/")


def plot_annual_cost_comparison(comparison: Dict, plot_dir: str):
    """Bar chart comparing annual costs: RL vs Baseline."""
    rl = comparison["rl_annual"]
    bl = comparison["baseline_annual"]

    categories = ["Grid Cost", "Arbitrage Profit", "Net Cost"]
    rl_values = [
        rl["total_grid_cost"],
        rl["arbitrage_profit"],
        rl["total_grid_cost"] - rl["arbitrage_profit"],
    ]
    bl_values = [
        bl["total_grid_cost"],
        bl["arbitrage_profit"],
        bl["total_grid_cost"] - bl["arbitrage_profit"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, rl_values, width, label="RL Agent", color="#2196F3")
    bars2 = ax.bar(x + width / 2, bl_values, width, label="Baseline", color="#FF9800")

    ax.set_ylabel("Cost ($)")
    ax.set_title("Annual Cost Comparison: RL Agent vs Rule-Based Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"${height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "annual_cost_comparison.png"), dpi=150)
    plt.close()


def plot_monthly_renewable_fraction(rl_h, bl_h, rl_d, bl_d, plot_dir: str):
    """Line chart of monthly renewable fraction, both strategies overlaid."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, daily_df, color in [
        ("RL Agent", rl_d, "#2196F3"),
        ("Baseline", bl_d, "#FF9800"),
    ]:
        monthly = daily_df.groupby("month").agg(
            renewable_frac=("renewable_fraction", "mean"),
        ).reset_index()
        ax.plot(monthly["month"], monthly["renewable_frac"], "o-",
                label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Month")
    ax.set_ylabel("Renewable Fraction")
    ax.set_title("Monthly Renewable Fraction Over Full Year")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "monthly_renewable_fraction.png"), dpi=150)
    plt.close()


def plot_sample_day_profiles(rl_h, bl_h, plot_dir: str):
    """Stacked area charts for sample days (winter, spring, summer, autumn)."""
    sample_days = {
        "Winter (Jan 15)": 14,
        "Spring (Apr 15)": 105,
        "Summer (Jul 15)": 196,
        "Autumn (Oct 15)": 288,
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (season_label, day_idx) in enumerate(sample_days.items()):
        ax = axes[idx]

        # Get RL data for this day, station 0
        day_data = rl_h[(rl_h["day"] == day_idx) & (rl_h["station"] == 0)]
        if len(day_data) == 0:
            ax.set_title(f"{season_label} (no data)")
            continue

        hours = day_data["hour"].values
        solar = day_data["solar_output_kw"].values
        wind = day_data["wind_output_kw"].values
        grid = day_data["grid_power_kw"].values
        battery = np.clip(-day_data["battery_power_kw"].values, 0, None)  # Discharge only
        demand = day_data["demand_kw"].values

        ax.stackplot(
            hours,
            solar, wind, battery, grid,
            labels=["Solar", "Wind", "Battery", "Grid"],
            colors=["#FFC107", "#4CAF50", "#9C27B0", "#F44336"],
            alpha=0.8,
        )
        ax.plot(hours, demand, "k--", linewidth=2, label="Demand")

        ax.set_title(f"{season_label} (RL Agent, Station 0)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Power (kW)")
        ax.set_xlim(0, 23)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Energy Source Mix on Sample Days", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sample_day_profiles.png"), dpi=150)
    plt.close()


def plot_battery_soc_trajectory(rl_h, bl_h, plot_dir: str):
    """Battery SOC over sample weeks in summer and winter."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    weeks = {
        "Winter Week (Jan 7-13)": (6, 12),  # days 6-12
        "Summer Week (Jul 1-7)": (182, 188),  # days 182-188
    }

    for idx, (week_label, (start_day, end_day)) in enumerate(weeks.items()):
        ax = axes[idx]

        for label, hourly_df, color, ls in [
            ("RL Agent", rl_h, "#2196F3", "-"),
            ("Baseline", bl_h, "#FF9800", "--"),
        ]:
            week_data = hourly_df[
                (hourly_df["day"] >= start_day) & (hourly_df["day"] <= end_day)
                & (hourly_df["station"] == 0)
            ].sort_values(["day", "hour"])

            if len(week_data) == 0:
                continue

            x = np.arange(len(week_data))
            ax.plot(x, week_data["battery_soc"].values, ls,
                    label=label, color=color, linewidth=1.5)

        ax.set_title(week_label)
        ax.set_ylabel("Battery SOC")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)

        # X-axis labels
        n_hours = (end_day - start_day + 1) * 24
        tick_positions = np.arange(0, n_hours, 24)
        tick_labels = [f"Day {i+1}" for i in range(len(tick_positions))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)

    plt.suptitle("Battery SOC Trajectory (Station 0)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "battery_soc_trajectory.png"), dpi=150)
    plt.close()


def plot_cost_heatmap(rl_h, bl_h, plot_dir: str):
    """12x24 heatmap of average hourly grid cost, RL vs Baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for idx, (label, hourly_df) in enumerate([("RL Agent", rl_h), ("Baseline", bl_h)]):
        ax = axes[idx]

        # Add month column from date string
        hourly_df = hourly_df.copy()
        if "date" in hourly_df.columns:
            hourly_df["month"] = pd.to_datetime(hourly_df["date"]).dt.month
        else:
            hourly_df["month"] = ((hourly_df["day"]) // 30 + 1).clip(1, 12)

        # Compute hourly grid cost
        hourly_df["hourly_grid_cost"] = hourly_df["grid_power_kw"] * hourly_df["grid_price"]

        pivot = hourly_df.groupby(["month", "hour"])["hourly_grid_cost"].mean().unstack(fill_value=0)

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_title(f"{label}: Avg Hourly Grid Cost ($)")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Month")
        ax.set_yticks(range(12))
        ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8)
        ax.set_xticks(range(0, 24, 3))
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Grid Cost Heatmap (Month x Hour)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "cost_heatmap.png"), dpi=150)
    plt.close()


def plot_arbitrage_monthly(rl_d, bl_d, plot_dir: str):
    """Monthly arbitrage profit comparison."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, daily_df, color in [
        ("RL Agent", rl_d, "#2196F3"),
        ("Baseline", bl_d, "#FF9800"),
    ]:
        monthly = daily_df.groupby("month")["arbitrage_profit"].sum().reset_index()
        ax.bar(
            monthly["month"] + (-0.2 if label == "RL Agent" else 0.2),
            monthly["arbitrage_profit"],
            width=0.4,
            label=label,
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("Month")
    ax.set_ylabel("Arbitrage Profit ($)")
    ax.set_title("Monthly Arbitrage Profit: RL Agent vs Baseline")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "arbitrage_monthly.png"), dpi=150)
    plt.close()


def plot_summary_dashboard(comparison: Dict, plot_dir: str):
    """Single-page summary dashboard with key metrics."""
    rl = comparison["rl_annual"]
    bl = comparison["baseline_annual"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Net cost comparison
    ax = axes[0, 0]
    rl_net = rl["total_grid_cost"] - rl["arbitrage_profit"]
    bl_net = bl["total_grid_cost"] - bl["arbitrage_profit"]
    bars = ax.bar(["RL Agent", "Baseline"], [rl_net, bl_net],
                  color=["#2196F3", "#FF9800"])
    ax.set_title("Annual Net Cost")
    ax.set_ylabel("$ / year")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9)

    # 2. Renewable fraction
    ax = axes[0, 1]
    bars = ax.bar(["RL Agent", "Baseline"],
                  [rl["renewable_fraction"] * 100, bl["renewable_fraction"] * 100],
                  color=["#2196F3", "#FF9800"])
    ax.set_title("Renewable Fraction")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    # 3. Grid energy
    ax = axes[0, 2]
    bars = ax.bar(["RL Agent", "Baseline"],
                  [rl["total_grid_energy_mwh"], bl["total_grid_energy_mwh"]],
                  color=["#2196F3", "#FF9800"])
    ax.set_title("Total Grid Energy")
    ax.set_ylabel("MWh / year")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9)

    # 4. Arbitrage profit
    ax = axes[1, 0]
    bars = ax.bar(["RL Agent", "Baseline"],
                  [rl["arbitrage_profit"], bl["arbitrage_profit"]],
                  color=["#2196F3", "#FF9800"])
    ax.set_title("Arbitrage Profit")
    ax.set_ylabel("$ / year")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9)

    # 5. Demand shortfall
    ax = axes[1, 1]
    bars = ax.bar(["RL Agent", "Baseline"],
                  [rl["demand_shortfall_hours"], bl["demand_shortfall_hours"]],
                  color=["#2196F3", "#FF9800"])
    ax.set_title("Demand Shortfall Hours")
    ax.set_ylabel("Hours / year")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height()}", ha="center", va="bottom", fontsize=9)

    # 6. Peak grid draw
    ax = axes[1, 2]
    bars = ax.bar(["RL Agent", "Baseline"],
                  [rl["peak_grid_draw_kw"], bl["peak_grid_draw_kw"]],
                  color=["#2196F3", "#FF9800"])
    ax.set_title("Peak Grid Draw")
    ax.set_ylabel("kW")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        "Full-Year Validation Summary: RL Agent vs Rule-Based Baseline",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "summary_dashboard.png"), dpi=150)
    plt.close()
