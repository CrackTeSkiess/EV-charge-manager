"""
Training Visualizer
Plots learning curves and battery strategy from hierarchical RL training history.

Reads the three files written by Phase 1:
- training_history.jsonl  — one JSON record per macro-RL update
- micro_history.json      — per-station episode rewards and grid costs
- micro_final_day.json    — hourly breakdown from the last micro episode
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from .tool import ChartConfig

plt.style.use('seaborn-v0_8-whitegrid')


def _rolling_mean(values: List[float], window: int) -> np.ndarray:
    """Return a rolling mean of *values* with the given *window*."""
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()


def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


class TrainingVisualizer:
    """
    Visualization suite for hierarchical RL training metrics.

    All plot methods accept an optional *output_dir*.  When supplied the figure
    is saved as a PNG and *show_plot* defaults to ``False``; otherwise the
    figure is shown interactively.

    Typical usage::

        tv = TrainingVisualizer()
        tv.generate_full_report("./models/hierarchical")
    """

    # Station colours consistent across all plots
    _STATION_COLORS = ["#2E86AB", "#A23B72", "#28A745", "#FFC107", "#DC3545",
                       "#17A2B8", "#6C757D", "#FF7F50"]

    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig()

    # -------------------------------------------------------------------------
    # PUBLIC PLOT METHODS
    # -------------------------------------------------------------------------

    def plot_macro_learning_curves(
        self,
        history_path: str,
        output_dir: Optional[str] = None,
        show_plot: Optional[bool] = None,
        smoothing_window: int = 10,
    ) -> plt.Figure:
        """
        Plot macro-RL (PPO) learning curves from ``training_history.jsonl``.

        **Row 1 — Reward:** raw episode reward (faint) + rolling mean (bold),
        plus a dashed line at the best reward ever seen.

        **Row 2 — Losses:** policy loss and value loss on the left axis;
        entropy on a secondary right axis (different scale).

        Parameters
        ----------
        history_path:
            Path to ``training_history.jsonl``.
        output_dir:
            If given, saves ``macro_learning_curves.png`` there and skips
            interactive display unless *show_plot* is explicitly ``True``.
        smoothing_window:
            Rolling-mean window size (number of updates).
        """
        records = self._load_jsonl(history_path)
        if not records:
            return self._empty_figure(f"No data in {history_path}")

        episodes = [r["episode"] for r in records]
        rewards = [r["reward"] for r in records]
        costs = [r.get("cost", 0.0) for r in records]
        policy_losses = [r["policy_loss"] for r in records]
        value_losses = [r["value_loss"] for r in records]
        entropies = [r["entropy"] for r in records]

        reward_smooth = _rolling_mean(rewards, smoothing_window)
        best_reward = float(np.max(reward_smooth))

        fig = plt.figure(figsize=(12, 8), layout="constrained")
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.35)

        # --- Row 1: Reward ---
        ax_rew = fig.add_subplot(gs[0])
        ax_rew.plot(episodes, rewards, color="#AED6F1", linewidth=0.8,
                    alpha=0.6, label="Raw reward")
        ax_rew.plot(episodes, reward_smooth, color="#2E86AB", linewidth=2.0,
                    label=f"Rolling mean (w={smoothing_window})")
        ax_rew.axhline(best_reward, color="#28A745", linewidth=1.2,
                       linestyle="--", label=f"Best: {best_reward:.1f}")
        ax_rew.set_ylabel("Episode Reward", fontsize=self.config.label_fontsize)
        ax_rew.set_title("Macro-RL Learning Curves", fontsize=self.config.title_fontsize,
                         fontweight="bold")
        ax_rew.legend(fontsize=self.config.tick_fontsize, loc="lower right")
        ax_rew.tick_params(labelsize=self.config.tick_fontsize)

        # --- Row 2: Losses + Entropy ---
        ax_loss = fig.add_subplot(gs[1])
        ax_ent = ax_loss.twinx()

        ax_loss.plot(episodes, _rolling_mean(policy_losses, smoothing_window),
                     color="#A23B72", linewidth=1.8, label="Policy loss")
        ax_loss.plot(episodes, _rolling_mean(value_losses, smoothing_window),
                     color="#DC3545", linewidth=1.8, linestyle="--", label="Value loss")
        ax_ent.plot(episodes, _rolling_mean(entropies, smoothing_window),
                    color="#FFC107", linewidth=1.5, linestyle=":", label="Entropy")

        ax_loss.set_xlabel("Episode", fontsize=self.config.label_fontsize)
        ax_loss.set_ylabel("Loss", fontsize=self.config.label_fontsize)
        ax_ent.set_ylabel("Entropy", fontsize=self.config.label_fontsize, color="#FFC107")
        ax_ent.tick_params(axis="y", colors="#FFC107",
                           labelsize=self.config.tick_fontsize)
        ax_loss.tick_params(labelsize=self.config.tick_fontsize)

        # Combined legend
        lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
        lines_ent, labels_ent = ax_ent.get_legend_handles_labels()
        ax_loss.legend(lines_loss + lines_ent, labels_loss + labels_ent,
                       fontsize=self.config.tick_fontsize, loc="upper right")

        self._finalize(fig, output_dir, "macro_learning_curves.png", show_plot,
                       use_tight_layout=False)
        return fig

    def plot_micro_convergence(
        self,
        micro_history_path: str,
        output_dir: Optional[str] = None,
        show_plot: Optional[bool] = None,
        smoothing_window: int = 20,
    ) -> plt.Figure:
        """
        Plot micro-RL (Actor-Critic) convergence from ``micro_history.json``.

        **Row 1 — Reward per station:** rolling-mean daily reward for each
        station on shared axes — reveals which station's energy arbitrage is
        hardest.

        **Row 2 — Grid cost per station:** rolling-mean daily grid cost,
        showing cost reduction over training.

        Parameters
        ----------
        micro_history_path:
            Path to ``micro_history.json``.
        smoothing_window:
            Rolling-mean window applied to each station's series.
        """
        data = self._load_json(micro_history_path)
        if data is None:
            return self._empty_figure(f"Cannot read {micro_history_path}")

        n_stations: int = data["n_stations"]
        per_station_rewards: List[List[float]] = data["per_station_rewards"]
        per_station_costs: List[List[float]] = data["per_station_grid_costs"]

        fig = plt.figure(figsize=(12, 7), layout="constrained")
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.38)

        ax_rew = fig.add_subplot(gs[0])
        ax_cost = fig.add_subplot(gs[1])

        for i in range(n_stations):
            color = self._STATION_COLORS[i % len(self._STATION_COLORS)]
            label = f"Station {i + 1}"

            rewards = per_station_rewards[i]
            costs = per_station_costs[i]
            episodes = list(range(1, len(rewards) + 1))

            ax_rew.plot(episodes, rewards, color=color, linewidth=0.6,
                        alpha=0.3)
            ax_rew.plot(episodes, _rolling_mean(rewards, smoothing_window),
                        color=color, linewidth=2.0, label=label)

            cost_eps = list(range(1, len(costs) + 1))
            ax_cost.plot(cost_eps, costs, color=color, linewidth=0.6,
                         alpha=0.3)
            ax_cost.plot(cost_eps, _rolling_mean(costs, smoothing_window),
                         color=color, linewidth=2.0, label=label)

        ax_rew.set_ylabel("Daily Reward", fontsize=self.config.label_fontsize)
        ax_rew.set_title("Micro-RL Convergence per Station",
                         fontsize=self.config.title_fontsize, fontweight="bold")
        ax_rew.legend(fontsize=self.config.tick_fontsize, loc="lower right")
        ax_rew.tick_params(labelsize=self.config.tick_fontsize)

        ax_cost.set_xlabel("Episode", fontsize=self.config.label_fontsize)
        ax_cost.set_ylabel("Daily Grid Cost ($)",
                            fontsize=self.config.label_fontsize)
        ax_cost.legend(fontsize=self.config.tick_fontsize, loc="upper right")
        ax_cost.tick_params(labelsize=self.config.tick_fontsize)

        self._finalize(fig, output_dir, "micro_convergence.png", show_plot,
                       use_tight_layout=False)
        return fig

    def plot_battery_strategy(
        self,
        micro_final_day_path: str,
        output_dir: Optional[str] = None,
        show_plot: Optional[bool] = None,
    ) -> plt.Figure:
        """
        Plot the battery strategy learned by micro-agents from
        ``micro_final_day.json``.

        **Row 1 — Battery SOC:** state-of-charge across 24 hours, one line
        per station.  A cheap-charge/expensive-discharge pattern indicates
        successful time-of-use arbitrage.

        **Row 2 — Charge/Discharge Action:** positive = charging the battery,
        negative = discharging (providing power to chargers).  Background
        bands mark off-peak (green), shoulder (yellow), and peak (red) hours.

        Parameters
        ----------
        micro_final_day_path:
            Path to ``micro_final_day.json``.
        """
        data = self._load_json(micro_final_day_path)
        if data is None:
            return self._empty_figure(f"Cannot read {micro_final_day_path}")

        n_stations: int = data["n_stations"]
        stations: List[List[Dict]] = data["stations"]

        # Default time-of-use bands (matches GridPricingSchedule defaults)
        tou_bands = [
            (0, 7, "#D5F5E3", "Off-peak"),    # midnight–7 am
            (7, 9, "#FEF9E7", "Shoulder"),     # 7–9 am
            (9, 17, "#FDEDEC", "Peak"),        # 9 am–5 pm
            (17, 21, "#FEF9E7", "Shoulder"),   # 5–9 pm
            (21, 24, "#D5F5E3", ""),           # 9 pm–midnight (off-peak, no duplicate label)
        ]
        tou_colors = {"Off-peak": "#28A745", "Shoulder": "#FFC107", "Peak": "#DC3545"}

        fig = plt.figure(figsize=(12, 7), layout="constrained")
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.38)
        ax_soc = fig.add_subplot(gs[0])
        ax_act = fig.add_subplot(gs[1])

        for ax in (ax_soc, ax_act):
            labeled = set()
            for start, end, bg_color, band_label in tou_bands:
                ax.axvspan(start, end, alpha=0.12, color=bg_color, zorder=0)
                if band_label and band_label not in labeled:
                    ax.axvspan(start, end, alpha=0, color=bg_color,
                               label=band_label)  # invisible proxy for legend
                    labeled.add(band_label)

        for i in range(n_stations):
            if i >= len(stations) or not stations[i]:
                continue
            color = self._STATION_COLORS[i % len(self._STATION_COLORS)]
            label = f"Station {i + 1}"
            hours = [h["hour"] for h in stations[i]]
            soc = [h["battery_soc"] for h in stations[i]]
            actions = [h["action"] if isinstance(h["action"], (int, float))
                       else h["action"][0] for h in stations[i]]

            ax_soc.plot(hours, soc, color=color, linewidth=2.0,
                        marker="o", markersize=3, label=label)
            ax_act.bar([h + i * 0.8 / n_stations for h in hours],
                       actions, width=0.8 / n_stations, color=color,
                       alpha=0.75, label=label)

        ax_soc.set_ylim(0, 1.05)
        ax_soc.set_ylabel("Battery SOC", fontsize=self.config.label_fontsize)
        ax_soc.set_title("Learned Battery Strategy (Final Episode)",
                          fontsize=self.config.title_fontsize, fontweight="bold")
        ax_soc.set_xticks(range(0, 24, 2))
        ax_soc.tick_params(labelsize=self.config.tick_fontsize)

        # Add TOU legend using coloured patches
        import matplotlib.patches as mpatches
        tou_patches = [mpatches.Patch(color=c, alpha=0.4, label=l)
                       for l, c in tou_colors.items()]
        station_lines, station_labels = ax_soc.get_legend_handles_labels()
        ax_soc.legend(station_lines + tou_patches,
                      station_labels + [p.get_label() for p in tou_patches],
                      fontsize=self.config.tick_fontsize, loc="lower right",
                      ncol=2)

        ax_act.axhline(0, color="#6C757D", linewidth=0.8, linestyle="--")
        ax_act.set_xlabel("Hour of Day", fontsize=self.config.label_fontsize)
        ax_act.set_ylabel("Action (charge ↑ / discharge ↓)",
                           fontsize=self.config.label_fontsize)
        ax_act.set_xticks(range(0, 24, 2))
        ax_act.tick_params(labelsize=self.config.tick_fontsize)
        ax_act.legend(fontsize=self.config.tick_fontsize, loc="upper right")

        self._finalize(fig, output_dir, "battery_strategy.png", show_plot,
                       use_tight_layout=False)
        return fig

    def plot_config_evolution(
        self,
        history_path: str,
        highway_length: float = 300.0,
        output_dir: Optional[str] = None,
        show_plot: Optional[bool] = None,
    ) -> plt.Figure:
        """
        Plot how infrastructure configuration evolved over training.

        Plots the **current policy output** (solid lines) to show what the
        agent is actually exploring, and the **all-time best** (dashed lines)
        for reference.  Falls back to best_config only if no current_config
        records are present.

        Produces a 3-row figure:

        - **Row 1 — Station positions** (km along highway) vs episode.
          Background bands divide the highway into thirds.
        - **Row 2 — Charger counts** per station vs episode.
        - **Row 3 — Waiting spots** per station vs episode.

        Parameters
        ----------
        history_path:
            Path to ``training_history.jsonl``.
        highway_length:
            Total highway length in km (used for background bands).
        """
        records = self._load_jsonl(history_path)

        # Current-policy snapshots (preferred) and all-time-best snapshots
        cur_snapshots = [r for r in records if "current_config" in r]
        best_snapshots = [r for r in records if "best_config" in r]

        if not cur_snapshots and not best_snapshots:
            return self._empty_figure(
                "No config snapshots in training history.\n"
                "(They are written at eval intervals — check eval_interval setting.)"
            )

        # Use current_config as primary series when available
        primary_key = "current_config" if cur_snapshots else "best_config"
        primary = cur_snapshots if cur_snapshots else best_snapshots

        episodes = [s["episode"] for s in primary]
        n_stations = len(primary[0][primary_key]["positions"])

        # Build per-station series for the primary (current policy) view
        positions  = [[s[primary_key]["positions"][i]  for s in primary]
                      for i in range(n_stations)]
        n_chargers = [[s[primary_key]["n_chargers"][i] for s in primary]
                      for i in range(n_stations)]
        n_waiting  = [[s[primary_key]["n_waiting"][i]  for s in primary]
                      for i in range(n_stations)]

        # Build all-time-best series for dashed reference lines
        has_best = bool(best_snapshots)
        if has_best:
            best_eps = [s["episode"] for s in best_snapshots]
            best_positions  = [[s["best_config"]["positions"][i]  for s in best_snapshots]
                               for i in range(n_stations)]
            best_n_chargers = [[s["best_config"]["n_chargers"][i] for s in best_snapshots]
                               for i in range(n_stations)]
            best_n_waiting  = [[s["best_config"]["n_waiting"][i]  for s in best_snapshots]
                               for i in range(n_stations)]

        fig = plt.figure(figsize=(12, 9), layout="constrained")
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.40)
        ax_pos  = fig.add_subplot(gs[0])
        ax_chr  = fig.add_subplot(gs[1])
        ax_wait = fig.add_subplot(gs[2])

        # Highway-thirds background on position plot
        third = highway_length / 3
        band_colors = ["#EBF5FB", "#E8F8F5", "#FEF9E7"]
        band_labels = ["First third", "Middle third", "Final third"]
        for k, (color, label) in enumerate(zip(band_colors, band_labels)):
            ax_pos.axhspan(k * third, (k + 1) * third, alpha=0.35,
                           color=color, label=label, zorder=0)

        for i in range(n_stations):
            color = self._STATION_COLORS[i % len(self._STATION_COLORS)]
            label = f"Station {i + 1}"
            kw = dict(color=color, linewidth=2.0, marker="o", markersize=3,
                      label=label)
            ax_pos.plot(episodes,  positions[i],  **kw)
            ax_chr.plot(episodes,  n_chargers[i], **kw)
            ax_wait.plot(episodes, n_waiting[i],  **kw)

            # Overlay all-time best as thin dashed lines when we have both
            if has_best and primary_key == "current_config":
                best_kw = dict(color=color, linewidth=1.0, linestyle="--",
                               alpha=0.5)
                ax_pos.plot(best_eps,  best_positions[i],  **best_kw)
                ax_chr.plot(best_eps,  best_n_chargers[i], **best_kw)
                ax_wait.plot(best_eps, best_n_waiting[i],  **best_kw)

        ax_pos.set_ylabel("Position (km)", fontsize=self.config.label_fontsize)
        ax_pos.set_ylim(0, highway_length)
        title_suffix = " (solid=policy, dashed=best)" if has_best and primary_key == "current_config" else ""
        ax_pos.set_title(f"Infrastructure Configuration Evolution{title_suffix}",
                          fontsize=self.config.title_fontsize, fontweight="bold")
        # Station legend on top, highway bands legend inside
        station_handles, station_labels = ax_pos.get_legend_handles_labels()
        # Separate station lines from band patches
        line_h = station_handles[:n_stations]
        line_l = station_labels[:n_stations]
        band_h = station_handles[n_stations:]
        band_l = station_labels[n_stations:]
        ax_pos.legend(line_h + band_h, line_l + band_l,
                      fontsize=self.config.tick_fontsize, loc="upper left",
                      ncol=2)
        ax_pos.tick_params(labelsize=self.config.tick_fontsize)

        ax_chr.set_ylabel("Chargers per station",
                           fontsize=self.config.label_fontsize)
        ax_chr.yaxis.get_major_locator().set_params(integer=True)
        ax_chr.legend(fontsize=self.config.tick_fontsize, loc="upper left")
        ax_chr.tick_params(labelsize=self.config.tick_fontsize)

        ax_wait.set_xlabel("Episode", fontsize=self.config.label_fontsize)
        ax_wait.set_ylabel("Waiting spots per station",
                            fontsize=self.config.label_fontsize)
        ax_wait.yaxis.get_major_locator().set_params(integer=True)
        ax_wait.legend(fontsize=self.config.tick_fontsize, loc="upper left")
        ax_wait.tick_params(labelsize=self.config.tick_fontsize)

        self._finalize(fig, output_dir, "config_evolution.png", show_plot,
                       use_tight_layout=False)
        return fig

    def plot_cost_breakdown(
        self,
        best_config: Optional[Dict] = None,
        training_result_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        show_plot: Optional[bool] = None,
    ) -> plt.Figure:
        """
        Plot a cost breakdown for the best infrastructure configuration.

        Accepts either a ``best_config`` dict directly or a path to
        ``training_result.json`` (from which ``best_config`` is extracted).

        The figure shows two panels:

        - **Left — Stacked bar per station:** grid electricity cost (blue) and
          other operational costs (orange) stacked, with the arbitrage profit
          shown as a negative bar (green) below zero.
        - **Right — Total cost waterfall:** aggregate grid cost, other costs,
          minus arbitrage profit, equals net cost.

        Parameters
        ----------
        best_config:
            Dict with keys ``positions``, ``n_chargers``, ``n_waiting``,
            ``cost``, ``grid_cost``, ``arbitrage_profit``.  Provide either
            this or *training_result_path*.
        training_result_path:
            Path to ``training_result.json``; ``best_config`` is read from
            the ``best_config`` key inside.
        """
        if best_config is None and training_result_path is not None:
            data = self._load_json(training_result_path)
            if data is None:
                return self._empty_figure(f"Cannot read {training_result_path}")
            best_config = data.get("best_config")

        if not best_config:
            return self._empty_figure(
                "No best_config provided.\n"
                "Pass best_config= or training_result_path= to this method."
            )

        n_stations    = len(best_config["positions"])
        positions     = best_config["positions"]
        n_chargers    = best_config["n_chargers"]
        total_cost    = float(best_config.get("cost", 0))
        grid_cost     = float(best_config.get("grid_cost", 0))
        arbitrage     = float(best_config.get("arbitrage_profit", 0))
        other_cost    = max(0.0, total_cost - grid_cost + arbitrage)

        # Approximate per-station split (proportional to charger count)
        total_chargers = max(1, sum(n_chargers))
        weights = [c / total_chargers for c in n_chargers]
        station_grid    = [grid_cost  * w for w in weights]
        station_other   = [other_cost * w for w in weights]
        station_arb     = [arbitrage  * w for w in weights]

        station_labels = [f"S{i+1}\n{positions[i]:.0f}km\n{n_chargers[i]}chr"
                          for i in range(n_stations)]
        x = np.arange(n_stations)

        fig = plt.figure(figsize=(13, 6), layout="constrained")
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.30)
        ax_bar  = fig.add_subplot(gs[0])
        ax_fall = fig.add_subplot(gs[1])

        # --- Left: stacked bars per station ---
        bar_kw = dict(width=0.55, zorder=2)
        b1 = ax_bar.bar(x, station_grid,  bottom=0,           color="#2E86AB",
                        label="Grid cost",       **bar_kw)
        b2 = ax_bar.bar(x, station_other, bottom=station_grid, color="#FFC107",
                        label="Other costs",     **bar_kw)
        b3 = ax_bar.bar(x, [-a for a in station_arb], bottom=0, color="#28A745",
                        alpha=0.75, label="Arbitrage profit", **bar_kw)

        # Value annotations
        for i in range(n_stations):
            net = station_grid[i] + station_other[i] - station_arb[i]
            ax_bar.text(x[i], max(station_grid[i] + station_other[i], 0) + 20,
                        f"${net:,.0f}", ha="center", va="bottom",
                        fontsize=self.config.annotation_fontsize, fontweight="bold")

        ax_bar.axhline(0, color="#6C757D", linewidth=0.8, linestyle="--")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(station_labels, fontsize=self.config.tick_fontsize)
        ax_bar.set_ylabel("Cost ($)", fontsize=self.config.label_fontsize)
        ax_bar.set_title("Cost Breakdown by Station",
                          fontsize=self.config.title_fontsize, fontweight="bold")
        ax_bar.legend(fontsize=self.config.tick_fontsize)
        ax_bar.tick_params(axis="y", labelsize=self.config.tick_fontsize)

        # --- Right: aggregate waterfall ---
        waterfall_labels = ["Grid\ncost", "Other\ncosts", "Arbitrage\nprofit", "Net\ncost"]
        waterfall_values = [grid_cost, other_cost, -arbitrage, total_cost]
        waterfall_colors = ["#2E86AB", "#FFC107", "#28A745", "#DC3545"]

        bottoms = [0, grid_cost, grid_cost + other_cost, 0]
        for k, (val, bot, col) in enumerate(
                zip(waterfall_values, bottoms, waterfall_colors)):
            height = abs(val)
            ax_fall.bar(k, height, bottom=bot if val >= 0 else bot + val,
                        color=col, alpha=0.85, width=0.6, zorder=2)
            ax_fall.text(k, bot + val + (50 if val >= 0 else -50),
                         f"${abs(val):,.0f}", ha="center",
                         va="bottom" if val >= 0 else "top",
                         fontsize=self.config.annotation_fontsize,
                         fontweight="bold")

        # Connector lines between bars
        run = grid_cost
        ax_fall.plot([0.3, 0.7], [run, run], color="#6C757D", lw=0.8, ls="--")
        run += other_cost
        ax_fall.plot([1.3, 1.7], [run, run], color="#6C757D", lw=0.8, ls="--")

        ax_fall.set_xticks(range(4))
        ax_fall.set_xticklabels(waterfall_labels,
                                fontsize=self.config.tick_fontsize)
        ax_fall.set_ylabel("Cost ($)", fontsize=self.config.label_fontsize)
        ax_fall.set_title("Aggregate Waterfall",
                           fontsize=self.config.title_fontsize, fontweight="bold")
        ax_fall.tick_params(axis="y", labelsize=self.config.tick_fontsize)
        ax_fall.axhline(0, color="#6C757D", linewidth=0.8, linestyle="--")

        self._finalize(fig, output_dir, "cost_breakdown.png", show_plot,
                       use_tight_layout=False)
        return fig

    def plot_curriculum_stages(
        self,
        training_result_path: str,
        output_dir: Optional[str] = None,
        show_plot: Optional[bool] = None,
    ) -> plt.Figure:
        """
        Plot per-stage results from a curriculum training run.

        Reads the ``stages`` array from a curriculum-mode
        ``training_result.json``.  Produces a side-by-side bar chart
        (one group per stage) showing:

        - Final macro reward
        - Final total cost
        - Arbitrage profit (from best_config, if available)

        Parameters
        ----------
        training_result_path:
            Path to ``training_result.json`` from a curriculum run.
        """
        data = self._load_json(training_result_path)
        if data is None:
            return self._empty_figure(f"Cannot read {training_result_path}")

        if data.get("mode") != "curriculum" or "stages" not in data:
            return self._empty_figure(
                "training_result.json does not contain curriculum stages.\n"
                "(This plot requires mode='curriculum'.)"
            )

        stages = data["stages"]
        if not stages:
            return self._empty_figure("No curriculum stages found in results.")

        # Extract per-stage metrics — be tolerant of missing keys
        stage_labels = []
        rewards, costs, profits = [], [], []
        for s in stages:
            cfg = s.get("config", {})
            days = cfg.get("days", s.get("stage", "?"))
            variance = cfg.get("price_variance", "")
            stage_labels.append(f"Stage {s['stage'] + 1}\n{days}d / σ={variance}")

            result = s.get("result", {})
            rewards.append(float(result.get("macro_final_reward",
                                            result.get("avg_reward", 0))))
            bc = result.get("best_config") or {}
            costs.append(float(bc.get("cost", result.get("avg_cost", 0))))
            profits.append(float(bc.get("arbitrage_profit", 0)))

        n = len(stages)
        x = np.arange(n)
        w = 0.55

        fig = plt.figure(figsize=(max(10, 3 * n + 4), 6), layout="constrained")
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
        ax_rew  = fig.add_subplot(gs[0])
        ax_cost = fig.add_subplot(gs[1])
        ax_arb  = fig.add_subplot(gs[2])

        # Left: final reward per stage
        ax_rew.bar(x, rewards, width=w, color="#2E86AB", zorder=2)
        ax_rew.set_xticks(x)
        ax_rew.set_xticklabels(stage_labels, fontsize=self.config.tick_fontsize)
        ax_rew.set_ylabel("Final Reward", fontsize=self.config.label_fontsize)
        ax_rew.tick_params(labelsize=self.config.tick_fontsize)
        ax_rew.set_title("Curriculum Stage Progression",
                          fontsize=self.config.title_fontsize, fontweight="bold")

        # Middle: total cost per stage
        bar_colors = [self._STATION_COLORS[i % len(self._STATION_COLORS)]
                      for i in range(n)]
        ax_cost.bar(x, costs, width=w, color=bar_colors, alpha=0.85, zorder=2)
        for i, c in enumerate(costs):
            if c:
                ax_cost.text(i, c * 1.01, f"${c:,.0f}",
                             ha="center", va="bottom",
                             fontsize=self.config.annotation_fontsize,
                             fontweight="bold")
        ax_cost.set_xticks(x)
        ax_cost.set_xticklabels(stage_labels, fontsize=self.config.tick_fontsize)
        ax_cost.set_ylabel("Total Cost ($)", fontsize=self.config.label_fontsize)
        ax_cost.set_title("Total Cost per Stage",
                           fontsize=self.config.title_fontsize, fontweight="bold")
        ax_cost.tick_params(labelsize=self.config.tick_fontsize)

        # Right: arbitrage profit per stage
        ax_arb.bar(x, profits, color=bar_colors, alpha=0.85, width=w, zorder=2)
        for i, p in enumerate(profits):
            if p:
                ax_arb.text(i, p * 1.01, f"${p:,.0f}",
                             ha="center", va="bottom",
                             fontsize=self.config.annotation_fontsize,
                             fontweight="bold")
        ax_arb.set_xticks(x)
        ax_arb.set_xticklabels(stage_labels, fontsize=self.config.tick_fontsize)
        ax_arb.set_ylabel("Arbitrage Profit ($)",
                            fontsize=self.config.label_fontsize)
        ax_arb.set_title("Arbitrage Profit",
                           fontsize=self.config.title_fontsize, fontweight="bold")
        ax_arb.tick_params(labelsize=self.config.tick_fontsize)

        self._finalize(fig, output_dir, "curriculum_stages.png", show_plot,
                       use_tight_layout=False)
        return fig

    def plot_mode_comparison(
        self,
        result_dirs: List[str],
        output_dir: Optional[str] = None,
        show_plot: Optional[bool] = None,
        smoothing_window: int = 10,
    ) -> plt.Figure:
        """
        Compare multiple training runs (different modes) on a single figure.

        Reads ``training_result.json`` and ``training_history.jsonl`` from
        each directory in *result_dirs*.  Produces a 3-panel figure:

        - **Top-left — Smoothed learning curves:** one line per run, labelled
          by the training mode read from ``training_result.json``.
        - **Top-right — Final reward & cost bar chart:** grouped bars per run.
        - **Bottom — Best config cost breakdown:** grid cost vs arbitrage
          profit per run as a horizontal bar chart.

        Parameters
        ----------
        result_dirs:
            List of ``save_dir`` paths, one per training run to compare.
        smoothing_window:
            Rolling-mean window for learning curves.
        """
        if not result_dirs:
            return self._empty_figure("No result directories provided.")

        # Collect data from each run
        runs: List[Dict] = []
        for d in result_dirs:
            result = self._load_json(os.path.join(d, "training_result.json"))
            history = self._load_jsonl(os.path.join(d, "training_history.jsonl"))
            if result is None and not history:
                print(f"Warning: no usable data in {d} — skipping")
                continue
            mode = (result or {}).get("mode", os.path.basename(d.rstrip("/\\")))
            bc   = (result or {}).get("best_config") or {}
            runs.append({
                "label":      mode.capitalize(),
                "history":    history,
                "final_reward": float((result or {}).get(
                    "macro_final_reward", 0)),
                "cost":       float(bc.get("cost", 0)),
                "grid_cost":  float(bc.get("grid_cost", 0)),
                "arbitrage":  float(bc.get("arbitrage_profit", 0)),
            })

        if not runs:
            return self._empty_figure("Could not load data from any of the provided directories.")

        n_runs = len(runs)
        x = np.arange(n_runs)
        run_labels = [r["label"] for r in runs]
        run_colors = [self._STATION_COLORS[i % len(self._STATION_COLORS)]
                      for i in range(n_runs)]

        fig = plt.figure(figsize=(14, 8), layout="constrained")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)
        ax_curve = fig.add_subplot(gs[0, 0])
        ax_bar   = fig.add_subplot(gs[0, 1])
        ax_horiz = fig.add_subplot(gs[1, :])

        # --- Top-left: overlaid learning curves ---
        for run, color in zip(runs, run_colors):
            hist = run["history"]
            if hist:
                eps = [r["episode"] for r in hist]
                rews = [r["reward"] for r in hist]
                ax_curve.plot(eps, rews, color=color, linewidth=0.6, alpha=0.3)
                ax_curve.plot(eps, _rolling_mean(rews, smoothing_window),
                              color=color, linewidth=2.0, label=run["label"])
        ax_curve.set_xlabel("Episode", fontsize=self.config.label_fontsize)
        ax_curve.set_ylabel("Reward", fontsize=self.config.label_fontsize)
        ax_curve.set_title("Learning Curves",
                            fontsize=self.config.title_fontsize, fontweight="bold")
        ax_curve.legend(fontsize=self.config.tick_fontsize)
        ax_curve.tick_params(labelsize=self.config.tick_fontsize)

        # --- Top-right: final reward bars (one per run) ---
        w = 0.55
        final_rewards = [r["final_reward"] for r in runs]
        ax_bar.bar(x, final_rewards, width=w, color=run_colors,
                   alpha=0.85, zorder=2)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(run_labels, fontsize=self.config.tick_fontsize)
        ax_bar.set_ylabel("Final Reward", fontsize=self.config.label_fontsize)
        ax_bar.set_title("Final Reward by Mode",
                          fontsize=self.config.title_fontsize, fontweight="bold")
        ax_bar.tick_params(labelsize=self.config.tick_fontsize)

        # --- Bottom: horizontal cost breakdown per run ---
        y = np.arange(n_runs)
        grid_costs = [r["grid_cost"] for r in runs]
        arbitrages = [r["arbitrage"] for r in runs]
        other_costs = [max(0.0, runs[i]["cost"] - grid_costs[i] + arbitrages[i])
                       for i in range(n_runs)]

        ax_horiz.barh(y, grid_costs,  color="#2E86AB", label="Grid cost",
                      height=0.4, zorder=2)
        ax_horiz.barh(y, other_costs, left=grid_costs, color="#FFC107",
                      label="Other costs", height=0.4, zorder=2)
        ax_horiz.barh(y, [-a for a in arbitrages], color="#28A745",
                      alpha=0.75, label="Arbitrage profit", height=0.4, zorder=2)
        ax_horiz.axvline(0, color="#6C757D", linewidth=0.8, linestyle="--")
        ax_horiz.set_yticks(y)
        ax_horiz.set_yticklabels(run_labels, fontsize=self.config.tick_fontsize)
        ax_horiz.set_xlabel("Cost ($)", fontsize=self.config.label_fontsize)
        ax_horiz.set_title("Cost Breakdown Comparison",
                            fontsize=self.config.title_fontsize, fontweight="bold")
        ax_horiz.legend(fontsize=self.config.tick_fontsize, loc="lower right")
        ax_horiz.tick_params(labelsize=self.config.tick_fontsize)

        self._finalize(fig, output_dir, "mode_comparison.png", show_plot,
                       use_tight_layout=False)
        return fig

    def plot_best_model_operations(
        self,
        history_path: str,
        output_dir: Optional[str] = None,
        show_plot: Optional[bool] = None,
        smoothing_window: int = 10,
    ) -> plt.Figure:
        """
        Plot charging, queuing, and stranding over training time for the
        best model episodes.

        Reads ``training_history.jsonl`` and renders a 3-row figure:

        - **Row 1 — Charging demand:** total charging demand (kWh) served
          per episode, with a rolling mean overlay.  Higher values indicate
          the infrastructure is handling more traffic.
        - **Row 2 — Queuing (shortage events):** average shortage events per
          episode.  Each shortage represents an hour where a station could
          not fully meet demand — a proxy for queue pressure.
        - **Row 3 — Stranding:** estimated stranded vehicles per episode.
          The best-model marker highlights the episode with the lowest cost.

        Parameters
        ----------
        history_path:
            Path to ``training_history.jsonl``.
        output_dir:
            If given, saves ``best_model_operations.png`` there.
        smoothing_window:
            Rolling-mean window size (number of updates).
        """
        records = self._load_jsonl(history_path)
        if not records:
            return self._empty_figure(f"No data in {history_path}")

        # Check that operational metrics are present
        if "stranded_vehicles" not in records[0]:
            return self._empty_figure(
                "No operational metrics in training history.\n"
                "Re-run training to record charging/queuing/stranding data."
            )

        episodes = [r["episode"] for r in records]
        charging = [r.get("charging_demand_kwh", 0.0) for r in records]
        shortage = [r.get("shortage_events", 0.0) for r in records]
        stranded = [r.get("stranded_vehicles", 0.0) for r in records]

        charging_smooth = _rolling_mean(charging, smoothing_window)
        shortage_smooth = _rolling_mean(shortage, smoothing_window)
        stranded_smooth = _rolling_mean(stranded, smoothing_window)

        # Identify the best-model episode (lowest cost)
        costs = [r.get("cost", float("inf")) for r in records]
        best_idx = int(np.argmin(costs))
        best_ep = episodes[best_idx]

        fig = plt.figure(figsize=(12, 10), layout="constrained")
        gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.38)

        # --- Row 1: Charging demand ---
        ax_charge = fig.add_subplot(gs[0])
        ax_charge.fill_between(
            episodes, charging, alpha=0.15, color="#2E86AB",
        )
        ax_charge.plot(
            episodes, charging, color="#AED6F1", linewidth=0.8, alpha=0.5,
            label="Raw",
        )
        ax_charge.plot(
            episodes, charging_smooth, color="#2E86AB", linewidth=2.0,
            label=f"Rolling mean (w={smoothing_window})",
        )
        ax_charge.axvline(
            best_ep, color="#28A745", linewidth=1.2, linestyle="--",
            label=f"Best model (ep {best_ep})",
        )
        ax_charge.set_ylabel(
            "Charging Demand (kWh)", fontsize=self.config.label_fontsize,
        )
        ax_charge.set_title(
            "Best-Model Operations Over Training",
            fontsize=self.config.title_fontsize, fontweight="bold",
        )
        ax_charge.legend(fontsize=self.config.tick_fontsize, loc="lower right")
        ax_charge.tick_params(labelsize=self.config.tick_fontsize)

        # --- Row 2: Queuing (shortage events) ---
        ax_queue = fig.add_subplot(gs[1])
        ax_queue.fill_between(
            episodes, shortage, alpha=0.15, color="#FFC107",
        )
        ax_queue.plot(
            episodes, shortage, color="#FFE082", linewidth=0.8, alpha=0.5,
            label="Raw",
        )
        ax_queue.plot(
            episodes, shortage_smooth, color="#FFA000", linewidth=2.0,
            label=f"Rolling mean (w={smoothing_window})",
        )
        ax_queue.axvline(
            best_ep, color="#28A745", linewidth=1.2, linestyle="--",
            label=f"Best model (ep {best_ep})",
        )
        ax_queue.set_ylabel(
            "Shortage Events (queue pressure)",
            fontsize=self.config.label_fontsize,
        )
        ax_queue.legend(fontsize=self.config.tick_fontsize, loc="upper right")
        ax_queue.tick_params(labelsize=self.config.tick_fontsize)

        # --- Row 3: Stranding ---
        ax_strand = fig.add_subplot(gs[2])
        ax_strand.fill_between(
            episodes, stranded, alpha=0.15, color="#DC3545",
        )
        ax_strand.plot(
            episodes, stranded, color="#F5B7B1", linewidth=0.8, alpha=0.5,
            label="Raw",
        )
        ax_strand.plot(
            episodes, stranded_smooth, color="#DC3545", linewidth=2.0,
            label=f"Rolling mean (w={smoothing_window})",
        )
        ax_strand.axvline(
            best_ep, color="#28A745", linewidth=1.2, linestyle="--",
            label=f"Best model (ep {best_ep})",
        )
        ax_strand.set_xlabel("Episode", fontsize=self.config.label_fontsize)
        ax_strand.set_ylabel(
            "Stranded Vehicles", fontsize=self.config.label_fontsize,
        )
        ax_strand.legend(fontsize=self.config.tick_fontsize, loc="upper right")
        ax_strand.tick_params(labelsize=self.config.tick_fontsize)

        self._finalize(
            fig, output_dir, "best_model_operations.png", show_plot,
            use_tight_layout=False,
        )
        return fig

    def generate_full_report(
        self,
        save_dir: str,
        show_plots: bool = False,
        highway_length: float = 300.0,
        plots_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Generate all training plots from a completed training run.

        Data files are read from *data_dir* (falls back to *save_dir*).
        Plots are saved to *plots_dir* (falls back to
        ``{save_dir}/training_plots/``).

        Parameters
        ----------
        highway_length:
            Used for the background bands in ``plot_config_evolution``.
            If a ``run_config.json`` is found, its value is used
            automatically.
        plots_dir:
            Explicit directory for generated plots.
        data_dir:
            Directory containing training data files.

        Returns
        -------
        List of file paths that were successfully written.
        """
        plots_dir = plots_dir or os.path.join(save_dir, "training_plots")
        data_dir = data_dir or save_dir
        os.makedirs(plots_dir, exist_ok=True)
        generated: List[str] = []

        macro_path   = os.path.join(data_dir, "training_history.jsonl")
        micro_hist   = os.path.join(data_dir, "micro_history.json")
        micro_day    = os.path.join(data_dir, "micro_final_day.json")
        result_path  = os.path.join(data_dir, "training_result.json")
        run_cfg_path = os.path.join(data_dir, "run_config.json")

        # Try to read highway_length from run_config.json
        run_cfg = self._load_json(run_cfg_path)
        if run_cfg and "highway_length" in run_cfg:
            highway_length = float(run_cfg["highway_length"])

        def _try(method, *args, filename: str, **kwargs):
            try:
                method(*args, output_dir=plots_dir, show_plot=show_plots, **kwargs)
                p = os.path.join(plots_dir, filename)
                generated.append(p)
            except Exception as exc:
                print(f"Warning: {filename} failed: {exc}")

        if os.path.exists(macro_path):
            _try(self.plot_macro_learning_curves, macro_path,
                 filename="macro_learning_curves.png")
            _try(self.plot_config_evolution, macro_path,
                 filename="config_evolution.png",
                 highway_length=highway_length)
        else:
            print(f"Note: {macro_path} not found — skipping macro plots")

        if os.path.exists(micro_hist):
            _try(self.plot_micro_convergence, micro_hist,
                 filename="micro_convergence.png")
        else:
            print(f"Note: {micro_hist} not found — skipping micro convergence")

        if os.path.exists(micro_day):
            _try(self.plot_battery_strategy, micro_day,
                 filename="battery_strategy.png")
        else:
            print(f"Note: {micro_day} not found — skipping battery strategy")

        if os.path.exists(result_path):
            _try(self.plot_cost_breakdown,
                 filename="cost_breakdown.png",
                 training_result_path=result_path)
            # Curriculum-specific plot (silently skipped for non-curriculum runs)
            result_data = self._load_json(result_path)
            if result_data and result_data.get("mode") == "curriculum":
                _try(self.plot_curriculum_stages, result_path,
                     filename="curriculum_stages.png")
        else:
            print(f"Note: {result_path} not found — skipping cost breakdown")

        # Best-model operational metrics (charging / queuing / stranding)
        if os.path.exists(macro_path):
            _try(self.plot_best_model_operations, macro_path,
                 filename="best_model_operations.png")

        print(f"Training report: {len(generated)} plots saved to {plots_dir}")
        return generated

    # -------------------------------------------------------------------------
    # PRIVATE HELPERS
    # -------------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        """Load a newline-delimited JSON file."""
        records: List[Dict] = []
        try:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not read {path}: {exc}")
        return records

    @staticmethod
    def _load_json(path: str) -> Optional[Dict]:
        """Load a regular JSON file, returning None on failure."""
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not read {path}: {exc}")
            return None

    def _finalize(
        self,
        fig: plt.Figure,
        output_dir: Optional[str],
        filename: str,
        show_plot: Optional[bool],
        use_tight_layout: bool = True,
    ) -> None:
        """Save and/or show *fig* according to *output_dir* and *show_plot*."""
        if use_tight_layout:
            fig.tight_layout()
        if output_dir:
            _save_fig(fig, os.path.join(output_dir, filename), dpi=self.config.dpi)
            if show_plot is not True:
                plt.close(fig)
                return
        if show_plot is not False:
            plt.show()

    @staticmethod
    def _empty_figure(message: str) -> plt.Figure:
        """Return a blank figure carrying an error *message*."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, message, ha="center", va="center",
                fontsize=13, transform=ax.transAxes, wrap=True)
        ax.axis("off")
        return fig
