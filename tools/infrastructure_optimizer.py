"""
Infrastructure Optimization Module
Finds the Pareto front of (chargers_per_area, waiting_spots_per_area) combinations
that achieve a target stranding rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from ev_charge_manager.simulation import Simulation, SimulationParameters, EarlyStopCondition
from ev_charge_manager.vehicle.generator import TemporalDistribution


@dataclass
class InfraResult:
    """Result of a single (chargers, waiting_spots) simulation."""
    chargers_per_area: int
    waiting_spots_per_area: int
    avg_strandings_per_hour: float
    total_strandings: int
    total_vehicles: int
    avg_wait_time: float
    avg_utilization: float


class InfrastructureOptimizer:
    """
    Grid-searches over (chargers_per_area, waiting_spots_per_area) and
    identifies the Pareto front satisfying a stranding rate target.
    """

    def __init__(
        self,
        highway_length_km: float = 200.0,
        num_stations: int = 5,
        vehicles_per_hour: float = 60.0,
        simulation_duration_hours: float = 10.0,
        random_seed: int = 42
    ):
        self.highway_length_km = highway_length_km
        self.num_stations = num_stations
        self.vehicles_per_hour = vehicles_per_hour
        self.simulation_duration_hours = simulation_duration_hours
        self.random_seed = random_seed

        self.results: List[InfraResult] = []

    def _run_single(
        self,
        chargers: int,
        waiting_spots: int,
        verbose: bool = False
    ) -> InfraResult:
        """Run one simulation with given infrastructure parameters."""
        params = SimulationParameters(
            highway_length_km=self.highway_length_km,
            num_charging_areas=self.num_stations,
            chargers_per_area=chargers,
            waiting_spots_per_area=waiting_spots,
            vehicles_per_hour=self.vehicles_per_hour,
            temporal_distribution=TemporalDistribution.RUSH_HOUR,
            simulation_duration_hours=self.simulation_duration_hours,
            random_seed=self.random_seed,
            early_stop=EarlyStopCondition.disabled(),
            allow_queue_overflow=False
        )

        sim = Simulation(params)
        result = sim.run(progress_interval=0, verbose=False)

        stats = result.summary_statistics
        duration_h = self.simulation_duration_hours
        total_strandings = stats.get('total_cars_zero_battery', 0)
        avg_per_hour = total_strandings / max(1.0, duration_h)

        res = InfraResult(
            chargers_per_area=chargers,
            waiting_spots_per_area=waiting_spots,
            avg_strandings_per_hour=avg_per_hour,
            total_strandings=total_strandings,
            total_vehicles=stats.get('total_cars_quit_waiting', 0) + total_strandings,
            avg_wait_time=stats.get('avg_wait_time', 0),
            avg_utilization=stats.get('avg_utilization', 0)
        )

        if verbose:
            print(f"  chargers={chargers:2d}  waiting={waiting_spots:2d}  "
                  f"strandings/h={avg_per_hour:6.2f}  "
                  f"total={total_strandings:4d}  "
                  f"util={res.avg_utilization:.1%}")

        return res

    def run_grid_search(
        self,
        charger_range: range,
        waiting_spots_range: range,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run simulations for all combinations of chargers and waiting spots.
        """
        total = len(charger_range) * len(waiting_spots_range)
        count = 0

        if verbose:
            print(f"Running grid search: {len(charger_range)} charger values "
                  f"x {len(waiting_spots_range)} waiting spot values = {total} simulations")

        self.results = []
        for c in charger_range:
            for w in waiting_spots_range:
                count += 1
                if verbose:
                    print(f"[{count}/{total}]", end=" ")
                res = self._run_single(c, w, verbose=verbose)
                self.results.append(res)

        return self.get_results_dataframe()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        return pd.DataFrame([
            {
                'chargers_per_area': r.chargers_per_area,
                'waiting_spots_per_area': r.waiting_spots_per_area,
                'avg_strandings_per_hour': r.avg_strandings_per_hour,
                'total_strandings': r.total_strandings,
                'avg_wait_time': r.avg_wait_time,
                'avg_utilization': r.avg_utilization,
            }
            for r in self.results
        ])

    def compute_pareto_front(
        self,
        max_strandings_per_hour: float = 10.0
    ) -> pd.DataFrame:
        """
        Extract the Pareto front from grid search results.

        A point (c, w) is Pareto-optimal if no other feasible point has
        both c' <= c AND w' <= w (i.e. strictly better on both objectives).
        Feasibility: avg_strandings_per_hour <= max_strandings_per_hour.
        """
        df = self.get_results_dataframe()

        # Filter feasible solutions
        feasible = df[df['avg_strandings_per_hour'] <= max_strandings_per_hour].copy()

        if feasible.empty:
            print(f"No feasible solutions found with <= {max_strandings_per_hour} strandings/hour")
            return feasible

        # Find Pareto-optimal points (minimize both chargers and waiting spots)
        pareto_mask = []
        for _, row in feasible.iterrows():
            dominated = False
            for _, other in feasible.iterrows():
                if (other['chargers_per_area'] <= row['chargers_per_area'] and
                    other['waiting_spots_per_area'] <= row['waiting_spots_per_area'] and
                    (other['chargers_per_area'] < row['chargers_per_area'] or
                     other['waiting_spots_per_area'] < row['waiting_spots_per_area'])):
                    dominated = True
                    break
            pareto_mask.append(not dominated)

        pareto = feasible[pareto_mask].sort_values('chargers_per_area').reset_index(drop=True)
        return pareto

    def plot_pareto(
        self,
        max_strandings_per_hour: float = 10.0,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Plot the Pareto front and all grid search results.
        """
        df = self.get_results_dataframe()
        pareto = self.compute_pareto_front(max_strandings_per_hour)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            f'Infrastructure Optimization — Pareto Front\n'
            f'{self.highway_length_km}km highway, {self.num_stations} stations, '
            f'{self.vehicles_per_hour} veh/h, overflow=OFF',
            fontsize=14, fontweight='bold'
        )

        # --- Left: heatmap of strandings/h across the grid ---
        ax1 = axes[0]
        chargers_vals = sorted(df['chargers_per_area'].unique())
        waiting_vals = sorted(df['waiting_spots_per_area'].unique())

        matrix = np.full((len(waiting_vals), len(chargers_vals)), np.nan)
        for _, row in df.iterrows():
            ci = chargers_vals.index(row['chargers_per_area'])
            wi = waiting_vals.index(row['waiting_spots_per_area'])
            matrix[wi, ci] = row['avg_strandings_per_hour']

        im = ax1.imshow(matrix, origin='lower', aspect='auto', cmap='RdYlGn_r',
                        interpolation='nearest')
        ax1.set_xticks(range(len(chargers_vals)))
        ax1.set_xticklabels(chargers_vals)
        ax1.set_yticks(range(len(waiting_vals)))
        ax1.set_yticklabels(waiting_vals)
        ax1.set_xlabel('Chargers per Area')
        ax1.set_ylabel('Waiting Spots per Area')
        ax1.set_title('Avg Strandings / Hour')
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Strandings / hour')

        # Overlay threshold contour
        contour = ax1.contour(
            range(len(chargers_vals)), range(len(waiting_vals)),
            matrix, levels=[max_strandings_per_hour],
            colors='black', linewidths=2, linestyles='--'
        )
        ax1.clabel(contour, fmt=f'{max_strandings_per_hour:.0f}/h', fontsize=10)

        # Mark Pareto points on heatmap
        for _, row in pareto.iterrows():
            ci = chargers_vals.index(row['chargers_per_area'])
            wi = waiting_vals.index(row['waiting_spots_per_area'])
            ax1.plot(ci, wi, 'k*', markersize=14)

        # --- Right: Pareto front scatter ---
        ax2 = axes[1]

        # All feasible points
        feasible = df[df['avg_strandings_per_hour'] <= max_strandings_per_hour]
        infeasible = df[df['avg_strandings_per_hour'] > max_strandings_per_hour]

        ax2.scatter(infeasible['chargers_per_area'], infeasible['waiting_spots_per_area'],
                   c='lightcoral', alpha=0.5, s=60, label=f'> {max_strandings_per_hour}/h', marker='x')
        ax2.scatter(feasible['chargers_per_area'], feasible['waiting_spots_per_area'],
                   c='lightgreen', alpha=0.6, s=60, label=f'<= {max_strandings_per_hour}/h')

        # Pareto front line
        if not pareto.empty:
            # Step-line for Pareto front
            pc = pareto['chargers_per_area'].values
            pw = pareto['waiting_spots_per_area'].values

            # Build step coordinates
            step_c = []
            step_w = []
            for i in range(len(pc)):
                step_c.append(pc[i])
                step_w.append(pw[i])
                if i < len(pc) - 1:
                    step_c.append(pc[i + 1])
                    step_w.append(pw[i])

            ax2.plot(step_c, step_w, 'k-', linewidth=2, label='Pareto front')
            ax2.scatter(pc, pw, c='black', s=120, zorder=5, marker='*',
                       label='Pareto-optimal')

            # Annotate Pareto points
            for _, row in pareto.iterrows():
                ax2.annotate(
                    f"{row['avg_strandings_per_hour']:.1f}/h",
                    (row['chargers_per_area'], row['waiting_spots_per_area']),
                    textcoords='offset points', xytext=(8, 8), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )

        ax2.set_xlabel('Chargers per Area')
        ax2.set_ylabel('Waiting Spots per Area')
        ax2.set_title(f'Pareto Front (target <= {max_strandings_per_hour} strandings/h)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        if show_plot:
            plt.show()

        return fig

    def print_pareto_report(self, max_strandings_per_hour: float = 10.0):
        """Print a summary table of Pareto-optimal configurations."""
        pareto = self.compute_pareto_front(max_strandings_per_hour)

        print(f"\n{'='*70}")
        print(f"PARETO-OPTIMAL CONFIGURATIONS (<= {max_strandings_per_hour} strandings/hour)")
        print(f"{'='*70}")

        if pareto.empty:
            print("No feasible solutions found.")
            return

        print(f"{'Chargers':>10} {'Waiting':>10} {'Strand/h':>10} "
              f"{'Total':>8} {'Wait(min)':>10} {'Util':>8}")
        print("-" * 60)
        for _, row in pareto.iterrows():
            print(f"{int(row['chargers_per_area']):>10} "
                  f"{int(row['waiting_spots_per_area']):>10} "
                  f"{row['avg_strandings_per_hour']:>10.2f} "
                  f"{int(row['total_strandings']):>8} "
                  f"{row['avg_wait_time']:>10.1f} "
                  f"{row['avg_utilization']:>7.1%}")

        print(f"\n{len(pareto)} Pareto-optimal solutions found.")


def demo_optimization():
    """Demonstrate the infrastructure optimization."""
    print("=" * 70)
    print("INFRASTRUCTURE OPTIMIZATION")
    print("=" * 70)

    optimizer = InfrastructureOptimizer(
        highway_length_km=200.0,
        num_stations=4,
        vehicles_per_hour=50.0,
        simulation_duration_hours=10.0,
        random_seed=42
    )

    # Grid search over chargers and waiting spots
    optimizer.run_grid_search(
        charger_range=range(2, 16, 1),
        waiting_spots_range=range(2, 80, 4),
        verbose=True
    )

    # Print Pareto report
    optimizer.print_pareto_report(max_strandings_per_hour=10.0)

    # Plot
    optimizer.plot_pareto(
        max_strandings_per_hour=10.0,
        save_path='infra_pareto.png',
        show_plot=True
    )

    print("\nOptimization complete!")
    return optimizer


if __name__ == "__main__":
    demo_optimization()
