"""
Traffic Stranding Analysis Module
Analyzes how stranding rates change with increasing traffic levels over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from Simulation import Simulation, SimulationParameters, EarlyStopCondition
from VehicleGenerator import TemporalDistribution


@dataclass
class TrafficLevelConfig:
    """Configuration for a single traffic level test."""
    vehicles_per_hour: float
    label: str
    color: str


class TrafficStrandingAnalyzer:
    """
    Analyzes stranding rates across different traffic levels and time steps.
    
    Runs multiple simulations with increasing traffic and generates
    time-series plots of stranding rates.
    """
    
    def __init__(
        self,
        highway_length_km: float = 200.0,
        num_stations: int = 5,
        chargers_per_station: int = 10,
        waiting_spots: int = 30,
        simulation_duration_hours: float = 5.0,
        random_seed: int = 42
    ):
        self.highway_length_km = highway_length_km
        self.num_stations = num_stations
        self.chargers_per_station = chargers_per_station
        self.waiting_spots = waiting_spots
        self.simulation_duration_hours = simulation_duration_hours
        self.random_seed = random_seed
        
        self.results: Dict[str, pd.DataFrame] = {}
        
    def run_traffic_level(
        self,
        vehicles_per_hour: float,
        label: str,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Run simulation for a specific traffic level.
        
        Returns DataFrame with time-series data including:
        - step
        - timestamp
        - cars_zero_battery (strandings this step)
        - cumulative_strandings
        - avg_stranding_rate (rolling window)
        """
        if verbose:
            print(f"Running simulation for {label} ({vehicles_per_hour} veh/h)...")
        
        params = SimulationParameters(
            highway_length_km=self.highway_length_km,
            num_charging_areas=self.num_stations,
            chargers_per_area=self.chargers_per_station,
            waiting_spots_per_area=self.waiting_spots,
            vehicles_per_hour=vehicles_per_hour,
            temporal_distribution=TemporalDistribution.RANDOM_POISSON,
            simulation_duration_hours=self.simulation_duration_hours,
            random_seed=self.random_seed,
            early_stop=EarlyStopCondition.disabled()  # Run full duration
        )
        
        sim = Simulation(params)
        result = sim.run(progress_interval=60, verbose=verbose)
        
        # Extract KPI data
        df = result.kpi_dataframe.copy()
        
        # Add derived columns
        df['cumulative_strandings'] = df['cars_zero_battery'].cumsum()
        df['cumulative_vehicles'] = df['vehicles_entered'].cumsum()
        
        # Calculate rolling average stranding rate (10-step window)
        df['strandings_rolling_avg'] = df['cars_zero_battery'].rolling(
            window=10, min_periods=1
        ).mean()
        
        # Calculate stranding rate per 100 vehicles entered
        df['stranding_rate_per_100'] = (
            df['cumulative_strandings'] / df['cumulative_vehicles'].replace(0, np.nan) * 100
        ).fillna(0)
        
        # Convert timestamp to hours from start
        start_time = df['timestamp'].iloc[0]
        df['hours_from_start'] = (df['timestamp'] - start_time).dt.total_seconds() / 3600
        
        self.results[label] = df
        
        if verbose:
            total_stranded = df['cars_zero_battery'].sum()
            total_entered = df['vehicles_entered'].sum()
            print(f"  Complete: {total_stranded} strandings out of {total_entered} vehicles")
            print(f"  Overall rate: {total_stranded/max(1,total_entered)*100:.2f}%")
        
        return df
    
    def run_traffic_sweep(
        self,
        traffic_levels: List[TrafficLevelConfig],
        verbose: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Run simulations for multiple traffic levels.
        """
        for config in traffic_levels:
            self.run_traffic_level(
                config.vehicles_per_hour,
                config.label,
                verbose=verbose
            )
        
        return self.results
    
    def plot_stranding_by_timestep(
        self,
        metric: str = 'cars_zero_battery',
        rolling_window: int = 10,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Plot stranding count by time step for all traffic levels.
        
        Args:
            metric: Column to plot ('cars_zero_battery', 'strandings_rolling_avg', etc.)
            rolling_window: If > 1, plot rolling average
            save_path: Path to save figure
            show_plot: Whether to display plot
        """
        if not self.results:
            raise ValueError("No results to plot. Run simulations first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Stranding Analysis: {self.highway_length_km}km Highway, '
            f'{self.num_stations} Stations, {self.chargers_per_station} Chargers/Station',
            fontsize=14, fontweight='bold'
        )
        
        # Plot 1: Raw stranding count by time step
        ax1 = axes[0, 0]
        for label, df in self.results.items():
            if rolling_window > 1:
                data = df[metric].rolling(window=rolling_window, min_periods=1).mean()
                ylabel = f'{metric} ({rolling_window}-step avg)'
            else:
                data = df[metric]
                ylabel = metric
            
            ax1.plot(df['step'], data, label=label, alpha=0.8, linewidth=1.5)
        
        ax1.set_xlabel('Simulation Step (minutes)')
        ax1.set_ylabel(ylabel)
        ax1.set_title('Strandings per Time Step')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative strandings over time
        ax2 = axes[0, 1]
        for label, df in self.results.items():
            ax2.plot(df['step'], df['cumulative_strandings'], 
                    label=label, alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Simulation Step (minutes)')
        ax2.set_ylabel('Cumulative Strandings')
        ax2.set_title('Total Strandings Over Time')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Stranding rate per 100 vehicles (normalized)
        ax3 = axes[1, 0]
        for label, df in self.results.items():
            ax3.plot(df['step'], df['stranding_rate_per_100'], 
                    label=label, alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Simulation Step (minutes)')
        ax3.set_ylabel('Strandings per 100 Vehicles Entered')
        ax3.set_title('Normalized Stranding Rate')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics bar chart
        ax4 = axes[1, 1]
        labels = []
        total_strandings = []
        total_entered = []
        rates = []
        
        for label, df in self.results.items():
            labels.append(label)
            stranded = df['cars_zero_battery'].sum()
            entered = df['vehicles_entered'].sum()
            total_strandings.append(stranded)
            total_entered.append(entered)
            rates.append(stranded / max(1, entered) * 100)
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, total_strandings, width, 
                       label='Total Strandings', color='coral', alpha=0.8)
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, rates, width, 
                            label='Stranding Rate %', color='steelblue', alpha=0.8)
        
        ax4.set_xlabel('Traffic Level')
        ax4.set_ylabel('Total Strandings', color='coral')
        ax4_twin.set_ylabel('Stranding Rate (%)', color='steelblue')
        ax4.set_title('Summary by Traffic Level')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def plot_stranding_heatmap(
        self,
        time_bin_minutes: int = 10,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Create heatmap of stranding rates by time and traffic level.
        """
        if not self.results:
            raise ValueError("No results to plot.")
        
        # Prepare data for heatmap
        max_steps = max(len(df) for df in self.results.values())
        traffic_labels = list(self.results.keys())
        
        # Create matrix: rows = traffic levels, cols = time bins
        time_bins = np.arange(0, max_steps, time_bin_minutes)
        matrix = np.zeros((len(traffic_labels), len(time_bins)))
        
        for i, (label, df) in enumerate(self.results.items()):
            for j, start_step in enumerate(time_bins):
                end_step = start_step + time_bin_minutes
                mask = (df['step'] >= start_step) & (df['step'] < end_step)
                if mask.any():
                    # Average strandings in this time bin
                    matrix[i, j] = df.loc[mask, 'cars_zero_battery'].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        ax.set_yticks(range(len(traffic_labels)))
        ax.set_yticklabels(traffic_labels)
        ax.set_xticks(range(0, len(time_bins), max(1, len(time_bins)//10)))
        ax.set_xticklabels([f"{int(time_bins[i])}" for i in range(0, len(time_bins), max(1, len(time_bins)//10))])
        
        ax.set_xlabel(f'Time Step (bins of {time_bin_minutes} minutes)')
        ax.set_ylabel('Traffic Level')
        ax.set_title('Stranding Rate Heatmap by Traffic Level and Time')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Strandings per Time Bin')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate summary report of all traffic levels.
        """
        rows = []
        
        for label, df in self.results.items():
            total_stranded = df['cars_zero_battery'].sum()
            total_entered = df['vehicles_entered'].sum()
            total_completed = df['vehicles_exited'].sum()
            
            row = {
                'traffic_level': label,
                'vehicles_per_hour': float(label.split()[0]),
                'total_vehicles_entered': total_entered,
                'total_stranded': total_stranded,
                'total_completed': total_completed,
                'stranding_rate_pct': total_stranded / max(1, total_entered) * 100,
                'completion_rate_pct': total_completed / max(1, total_entered) * 100,
                'peak_strandings_in_one_step': df['cars_zero_battery'].max(),
                'avg_strandings_per_step': df['cars_zero_battery'].mean(),
                'first_stranding_step': df[df['cars_zero_battery'] > 0]['step'].min() if (df['cars_zero_battery'] > 0).any() else None,
                'simulation_steps': len(df)
            }
            rows.append(row)
        
        report_df = pd.DataFrame(rows)
        return report_df


def demo_analysis():
    """
    Demonstrate the traffic stranding analysis.
    """
    print("=" * 70)
    print("TRAFFIC STRANDING ANALYSIS DEMO")
    print("=" * 70)
    
    # Define traffic levels to test
    traffic_levels = [
        TrafficLevelConfig(30, "30 veh/h", "green"),
        TrafficLevelConfig(60, "60 veh/h", "blue"),
        TrafficLevelConfig(90, "90 veh/h", "orange"),
        TrafficLevelConfig(120, "120 veh/h", "red"),
        TrafficLevelConfig(150, "150 veh/h", "darkred"),
        TrafficLevelConfig(200, "200 veh/h", "grey"),
        TrafficLevelConfig(300, "300 veh/h", "pink"),
    ]
    
    # Create analyzer
    analyzer = TrafficStrandingAnalyzer(
        highway_length_km=200.0,
        num_stations=4,
        chargers_per_station=10,
        waiting_spots=30,
        simulation_duration_hours=20.0,
        random_seed=42
    )
    
    # Run all simulations
    print("\nRunning traffic sweep...")
    analyzer.run_traffic_sweep(traffic_levels, verbose=True)
    
    # Generate report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    report = analyzer.generate_report()
    print(report.to_string(index=False))
    
    # Create visualizations
    print("\nGenerating plots...")
    
    # Main time-series plot
    fig1 = analyzer.plot_stranding_by_timestep(
        metric='cars_zero_battery',
        rolling_window=10,
        save_path='stranding_by_timestep.png',
        show_plot=True
    )
    
    # Heatmap
    fig2 = analyzer.plot_stranding_heatmap(
        time_bin_minutes=15,
        save_path='stranding_heatmap.png',
        show_plot=True
    )
    
    print("\nAnalysis complete!")
    return analyzer


if __name__ == "__main__":
    demo_analysis()