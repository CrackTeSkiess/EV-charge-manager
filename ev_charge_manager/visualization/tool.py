"""
Visualization Tool Module
Rich visual analytics for EV charging simulation KPIs.
"""

from __future__ import annotations

import uuid
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter, HourLocator
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class ChartConfig:
    """Configuration for chart appearance."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    save_path: Optional[str] = None
    show_plot: bool = True
    title_fontsize: int = 14
    label_fontsize: int = 11
    tick_fontsize: int = 9
    annotation_fontsize: int = 9
    color_scheme: str = "viridis"
    dark_mode: bool = False


class VisualizationTool:
    """
    Comprehensive visualization suite for EV charging simulation KPIs.
    
    Provides:
    - Time series analysis
    - Distribution and histogram plots
    - Correlation and heatmap visualizations
    - Real-time dashboard components
    - Station-specific analysis
    - Animated progression (optional)
    """
    
    def __init__(self, kpi_dataframe: Optional[pd.DataFrame] = None, 
                 config: Optional[ChartConfig] = None):
        self.df = kpi_dataframe
        self.config = config or ChartConfig()
        self.id = str(uuid.uuid4())[:8]
        self._station_data: Optional[Dict[str, pd.DataFrame]] = None
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#28A745',
            'warning': '#FFC107',
            'danger': '#DC3545',
            'info': '#17A2B8',
            'neutral': '#6C757D',
            'background': '#F8F9FA' if not self.config.dark_mode else '#212529',
            'text': '#212529' if not self.config.dark_mode else '#F8F9FA'
        }
        
        # Custom colormaps
        self.cmap_diverging = LinearSegmentedColormap.from_list(
            'custom_diverging', ['#DC3545', '#F8F9FA', '#28A745']
        )
        self.cmap_sequential = LinearSegmentedColormap.from_list(
            'custom_sequential', ['#F8F9FA', '#2E86AB', '#1A5276']
        )
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Update the KPI DataFrame."""
        self.df = df.copy()
        
    def set_station_data(self, station_data: Dict[str, pd.DataFrame]) -> None:
        """
        Set station data for visualization.
        
        This allows storing station tracking data collected during simulation
        for later visualization.
        
        Args:
            station_data: Dict mapping station_id to DataFrame with columns:
                         ['timestamp', 'occupancy', 'queue_length', 'total_chargers', 'waiting_spots']
        """
        self._station_data = station_data
    
    # ========================================================================
    # CORE TIME SERIES PLOTS
    # ========================================================================

    def plot_charging_area_occupancy(self, station_data: Dict[str, pd.DataFrame] = None,  # pyright: ignore[reportArgumentType]
                                     **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Plot charger occupancy and queue length over time for each charging area.
        
        Creates a multi-panel figure showing for each station:
        - Charger occupancy (number of chargers in use over time)
        - Queue length (number of vehicles waiting over time)
        - Combined utilization view
        
        Args:
            station_data: Dict mapping station_id to DataFrame with columns:
                         ['timestamp', 'occupancy', 'queue_length', 'total_chargers', 'waiting_spots']
                         If None, uses data set via set_station_data()
            **kwargs: Additional configuration options
        
        Returns:
            matplotlib Figure object
        """
        config = self._merge_config(kwargs)
        
        # Use stored data if none provided
        if station_data is None:
            station_data = self._station_data
        
        # If no station data available, we can't create the visualization
        if station_data is None or not station_data:
            return self._empty_figure("No charging area data available.\n"
                                    "Use set_station_data() or pass station_data parameter.")
        
        num_stations = len(station_data)
        if num_stations == 0:
            return self._empty_figure("No charging stations to display")
        
        # Calculate grid layout - aim for 2 columns, or 1 if only 1-2 stations
        n_cols = 2 if num_stations >= 2 else 1
        n_rows = (num_stations + n_cols - 1) // n_cols
        
        # Create figure with subplots for each station
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(max(14, 7 * n_cols), 5 * n_rows), 
                                dpi=config.dpi,
                                squeeze=False)
        fig.patch.set_facecolor(self.colors['background'])
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for idx, (station_id, df) in enumerate(station_data.items()):
            if idx >= len(axes_flat):
                break
                
            ax = axes_flat[idx]
            ax.set_facecolor(self.colors['background'])
            
            # Ensure we have the required columns
            if 'timestamp' not in df.columns:
                # Try to use index as timestamp
                df = df.copy()
                df['timestamp'] = df.index
            
            time_index = pd.to_datetime(df['timestamp']) if not pd.api.types.is_datetime64_any_dtype(df['timestamp']) else df['timestamp']
            
            # Get data columns with fallbacks
            occupancy = df.get('occupancy', df.get('total_charging', pd.Series([0] * len(df))))
            queue = df.get('queue_length', df.get('total_queued', pd.Series([0] * len(df))))
            total_chargers = df.get('total_chargers', pd.Series([4] * len(df)))  # Default 4
            waiting_spots = df.get('waiting_spots', pd.Series([6] * len(df)))    # Default 6
            
            # Calculate occupancy rate
            occupancy_rate = (occupancy / total_chargers * 100).fillna(0)
            queue_rate = (queue / waiting_spots * 100).fillna(0)
            
            # Primary axis: Occupancy
            ax.fill_between(time_index, occupancy, alpha=0.3, color=self.colors['success'], 
                          label='Chargers Occupied')
            line1, = ax.plot(time_index, occupancy, color=self.colors['success'], 
                           linewidth=2, label='Chargers Occupied')
            
            # Mark full occupancy periods
            full_occupancy = occupancy >= total_chargers
            if full_occupancy.any():
                max_chargers_val = total_chargers.max() if hasattr(total_chargers, 'max') else total_chargers.iloc[0]
                ax.fill_between(time_index, 0, max_chargers_val, 
                              where=full_occupancy, alpha=0.2, color=self.colors['danger'],
                              label='Full Capacity')
            
            ax.set_ylabel('Chargers Occupied', fontsize=config.label_fontsize, 
                         color=self.colors['success'], fontweight='bold')
            ax.tick_params(axis='y', labelcolor=self.colors['success'])
            max_val = max(total_chargers.max() if hasattr(total_chargers, 'max') else total_chargers.iloc[0], 
                         occupancy.max()) * 1.1
            ax.set_ylim(0, max_val)
            
            # Secondary axis: Queue
            ax2 = ax.twinx()
            ax2.fill_between(time_index, queue, alpha=0.3, color=self.colors['warning'], 
                           label='Queue Length')
            line2, = ax2.plot(time_index, queue, color=self.colors['warning'], 
                            linewidth=2, linestyle='--', label='Queue Length')
            
            # Mark queue overflow
            queue_full = queue >= waiting_spots
            if queue_full.any():
                max_waiting_val = waiting_spots.max() if hasattr(waiting_spots, 'max') else waiting_spots.iloc[0]
                ax2.fill_between(time_index, 0, max_waiting_val, 
                               where=queue_full, alpha=0.2, color=self.colors['danger'],
                               label='Queue Full')
            
            ax2.set_ylabel('Queue Length', fontsize=config.label_fontsize, 
                          color=self.colors['warning'], fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=self.colors['warning'])
            max_queue_val = max(waiting_spots.max() if hasattr(waiting_spots, 'max') else waiting_spots.iloc[0], 
                               queue.max()) * 1.2
            ax2.set_ylim(0, max_queue_val)
            
            # Title with summary statistics
            avg_occupancy = occupancy_rate.mean()
            max_queue = queue.max()
            peak_time = time_index[queue.idxmax()] if queue.max() > 0 else None
            
            title = f'Station {station_id}'
            if hasattr(df, 'attrs') and 'name' in df.attrs:
                title = df.attrs['name']
            
            subtitle = f'Avg Util: {avg_occupancy:.1f}% | Max Queue: {int(max_queue)}'
            if peak_time is not None:
                subtitle += f' at {peak_time.strftime("%H:%M")}'
            
            ax.set_title(f'{title}\n{subtitle}', fontsize=config.title_fontsize, 
                        color=self.colors['text'], fontweight='bold')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(HourLocator(interval=max(1, len(time_index) // 8)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.tick_params(axis='x', colors=self.colors['text'])
            
            # Combined legend
            lines = [line1, line2]
            labels = ['Charging', 'Queue']
            ax.legend(lines, labels, loc='upper left', fontsize=config.tick_fontsize)
            
            # Add capacity indicators as horizontal lines
            chargers_val = total_chargers.iloc[0] if len(total_chargers) > 0 else 4
            ax.axhline(y=chargers_val, color=self.colors['success'], linestyle=':', alpha=0.5)
            waiting_val = waiting_spots.iloc[0] if len(waiting_spots) > 0 else 6
            ax2.axhline(y=waiting_val, color=self.colors['warning'], linestyle=':', alpha=0.5)
        
        # Hide unused subplots
        for idx in range(len(station_data), len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        # Overall title
        fig.suptitle('Charging Area Occupancy and Queue Over Time', 
                    fontsize=config.title_fontsize + 2, fontweight='bold',
                    color=self.colors['text'], y=0.995)
        
        plt.tight_layout()
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_station_occupancy.png", 
                       dpi=config.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig

    def plot_station_utilization_summary(self, station_data: Dict[str, pd.DataFrame] = None, # pyright: ignore[reportArgumentType]
                                        **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Create a summary heatmap showing utilization across all stations over time.
        
        Args:
            station_data: Dict mapping station_id to DataFrame with occupancy data.
                         If None, uses data set via set_station_data()
            **kwargs: Configuration options
            
        Returns:
            matplotlib Figure object
        """
        config = self._merge_config(kwargs)
        
        # Use stored data if none provided
        if station_data is None:
            station_data = self._station_data
            
        if station_data is None or not station_data:
            return self._empty_figure("No station data available for summary.\n"
                                    "Use set_station_data() or pass station_data parameter.")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=config.dpi,
                                gridspec_kw={'height_ratios': [2, 1]})
        fig.patch.set_facecolor(self.colors['background'])
        
        # Prepare data for heatmap
        stations = list(station_data.keys())
        
        # Get common time index
        all_times = None
        for df in station_data.values():
            times = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else df.index
            if all_times is None:
                all_times = times
            else:
                all_times = all_times.union(times) # pyright: ignore[reportCallIssue, reportArgumentType]
        
        all_times = all_times.sort_values() # pyright: ignore[reportOptionalMemberAccess]
        
        # Create matrices for heatmap
        occupancy_matrix = []
        queue_matrix = []
        
        for station_id in stations:
            df = station_data[station_id]
            times = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else df.index
            
            # Reindex to common time grid
            df_indexed = df.set_index(times).reindex(all_times, method='ffill').fillna(0)
            
            occupancy = df_indexed.get('occupancy', df_indexed.get('total_charging', 0))
            total_chargers = df_indexed.get('total_chargers', 4)
            occupancy_rate = (occupancy / total_chargers * 100).fillna(0) # pyright: ignore[reportAttributeAccessIssue]
            
            queue = df_indexed.get('queue_length', df_indexed.get('total_queued', 0))
            waiting_spots = df_indexed.get('waiting_spots', 6)
            queue_rate = (queue / waiting_spots * 100).fillna(0) # pyright: ignore[reportAttributeAccessIssue]
            
            occupancy_matrix.append(occupancy_rate.values)
            queue_matrix.append(queue_rate.values)
        
        occupancy_matrix = np.array(occupancy_matrix)
        queue_matrix = np.array(queue_matrix)
        
        # Plot 1: Charger Occupancy Heatmap
        ax1 = axes[0]
        ax1.set_facecolor(self.colors['background'])
        
        # Sample time points for x-axis labels (avoid overcrowding)
        n_time_points = len(all_times)
        step = max(1, n_time_points // 20)
        
        im1 = ax1.imshow(occupancy_matrix, aspect='auto', cmap='YlOrRd', 
                        interpolation='nearest', vmin=0, vmax=100)
        
        ax1.set_yticks(range(len(stations)))
        ax1.set_yticklabels([f'Station {s}' for s in stations])
        ax1.set_xticks(range(0, n_time_points, step))
        ax1.set_xticklabels([all_times[i].strftime('%H:%M') for i in range(0, n_time_points, step)], 
                           rotation=45, ha='right')
        ax1.set_title('Charger Occupancy Rate (%)', fontsize=config.title_fontsize, 
                     fontweight='bold', color=self.colors['text'])
        ax1.set_xlabel('Time', fontsize=config.label_fontsize, color=self.colors['text'])
        
        cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
        cbar1.set_label('Occupancy %', color=self.colors['text'])
        
        # Add contour lines for high utilization
        ax1.contour(occupancy_matrix, levels=[80, 95], colors=['orange', 'red'], 
                   linewidths=2, linestyles='--')
        
        # Plot 2: Queue Heatmap
        ax2 = axes[1]
        ax2.set_facecolor(self.colors['background'])
        
        vmax_queue = min(100, queue_matrix.max()) if queue_matrix.max() > 0 else 100
        im2 = ax2.imshow(queue_matrix, aspect='auto', cmap='Reds', 
                        interpolation='nearest', vmin=0, vmax=vmax_queue)
        
        ax2.set_yticks(range(len(stations)))
        ax2.set_yticklabels([f'Station {s}' for s in stations])
        ax2.set_xticks(range(0, n_time_points, step))
        ax2.set_xticklabels([all_times[i].strftime('%H:%M') for i in range(0, n_time_points, step)], 
                           rotation=45, ha='right')
        ax2.set_title('Queue Occupancy Rate (%)', fontsize=config.title_fontsize, 
                     fontweight='bold', color=self.colors['text'])
        ax2.set_xlabel('Time', fontsize=config.label_fontsize, color=self.colors['text'])
        
        cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
        cbar2.set_label('Queue %', color=self.colors['text'])
        
        # Highlight queue overflow (>100%)
        if queue_matrix.max() > 100:
            overflow_mask = queue_matrix > 100
            # Overlay hatch pattern for overflow
            for i in range(len(stations)):
                for j in range(n_time_points):
                    if overflow_mask[i, j]:
                        ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,  # pyright: ignore[reportPrivateImportUsage]
                                                   fill=False, edgecolor='red', 
                                                   hatch='///', alpha=0.5))
        
        plt.tight_layout()
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_station_utilization_heatmap.png", 
                       dpi=config.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig
    
    def plot_live_stranding_map(self, highway_length_km: float, 
                            station_positions: List[float],
                            recent_strandings: List[Tuple[float, datetime]],
                            time_window_minutes: float = 30,
                            **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Real-time style map showing recent strandings with time decay.
        Useful for identifying emerging problem areas during simulation.
        """
        config = self._merge_config(kwargs)
        fig, ax = plt.subplots(figsize=(16, 6), dpi=config.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        current_time = kwargs.get('current_time', datetime.now())
        
        # Draw highway
        ax.plot([0, highway_length_km], [0, 0], color=self.colors['neutral'], 
            linewidth=8, alpha=0.6, solid_capstyle='round')
        
        # Draw stations with capacity indicators
        for i, pos in enumerate(station_positions):
            # Station marker
            ax.scatter([pos], [0], s=400, c=self.colors['success'], 
                    marker='s', zorder=5, edgecolors='white', linewidth=2)
            ax.annotate(f'S{i+1}\n{pos:.0f}km', xy=(pos, 0), xytext=(0, 25),
                    textcoords='offset points', ha='center', fontsize=9,
                    color=self.colors['text'], fontweight='bold')
        
        # Plot strandings with time-based fading
        for pos, time in recent_strandings:
            age_minutes = (current_time - time).total_seconds() / 60
            if age_minutes > time_window_minutes:
                continue
                
            # Fade older strandings
            alpha = 1.0 - (age_minutes / time_window_minutes)
            size = 200 + (time_window_minutes - age_minutes) * 10
            
            # Vertical position based on recency (newer = higher)
            y_offset = 0.1 + (1 - alpha) * 0.3
            
            ax.scatter([pos], [y_offset], s=size, c=self.colors['danger'], 
                    marker='x', alpha=alpha, linewidth=3, zorder=10)
            
            # Time label
            ax.annotate(f'{age_minutes:.0f}m ago', xy=(pos, y_offset), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=7, alpha=alpha,
                    color=self.colors['danger'])
        
        # Add danger zones (areas with multiple recent strandings)
        if len(recent_strandings) >= 2:
            positions = [p for p, t in recent_strandings]
            # Highlight clusters
            from scipy.cluster.hierarchy import fclusterdata
            if len(positions) >= 3:
                clusters = fclusterdata(np.array(positions).reshape(-1, 1), 
                                    t=15, criterion='distance')
                unique_clusters = set(clusters)
                for cluster_id in unique_clusters:
                    cluster_positions = [p for p, c in zip(positions, clusters) if c == cluster_id]
                    if len(cluster_positions) >= 2:
                        center = np.mean(cluster_positions)
                        ax.axvspan(center-5, center+5, alpha=0.1,  # pyright: ignore[reportArgumentType]
                                color=self.colors['danger'], zorder=0)
                        ax.annotate('DANGER ZONE', xy=(center, 0.5),  # pyright: ignore[reportArgumentType]
                                ha='center', fontsize=10, color=self.colors['danger'],
                                fontweight='bold', rotation=90, alpha=0.5)
        
        ax.set_xlim(-10, highway_length_km + 10)
        ax.set_ylim(-0.5, 1)
        ax.set_xlabel('Highway Position (km)', fontsize=config.label_fontsize, 
                    color=self.colors['text'])
        ax.set_title(f'Live Stranding Map (Last {time_window_minutes:.0f} minutes)', 
                    fontsize=config.title_fontsize, fontweight='bold',
                    color=self.colors['text'])
        ax.set_yticks([])
        ax.tick_params(colors=self.colors['text'])
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['success'], 
                markersize=12, label='Charging Station'),
            Line2D([0], [0], marker='x', color='w', markerfacecolor=self.colors['danger'], 
                markersize=10, label='Stranding (recent)'),
            Line2D([0], [0], marker='s', color=self.colors['danger'], 
                markersize=12, alpha=0.3, label='Danger Zone')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=config.tick_fontsize)
        
        plt.tight_layout()
        return fig

    def plot_stranding_analysis(self, 
                            stranding_data: Dict[str, Any],
                            highway_length_km: float,
                            station_positions: List[float],
                            station_names: Optional[List[str]] = None,
                            **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Comprehensive stranding analysis visualization.
        
        Args:
            stranding_data: Output from VehicleTracker.get_detailed_stranding_data()
            highway_length_km: Total highway length
            station_positions: List of station positions in km
            station_names: Optional list of station names
        """
        config = self._merge_config(kwargs)
        
        # Create figure with 4 subplots in a grid
        fig = plt.figure(figsize=(16, 12), dpi=config.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], 
                            hspace=0.3, wspace=0.3)
        
        # Main highway map (top, spans 2 columns)
        ax_map = fig.add_subplot(gs[0, :])
        self._plot_stranding_map(ax_map, stranding_data, highway_length_km,
                                station_positions, station_names, config)
        
        # Distance from nearest station histogram
        ax_dist = fig.add_subplot(gs[1, 0])
        self._plot_distance_analysis(ax_dist, stranding_data, station_positions, config)
        
        # Temporal distribution
        ax_time = fig.add_subplot(gs[1, 1])
        self._plot_temporal_distribution(ax_time, stranding_data, config)
        
        # Driver type breakdown
        ax_driver = fig.add_subplot(gs[2, 0])
        self._plot_driver_breakdown(ax_driver, stranding_data, config)
        
        # Battery/SOC analysis
        ax_battery = fig.add_subplot(gs[2, 1])
        self._plot_battery_analysis(ax_battery, stranding_data, config)
        
        # Overall title with summary stats
        total = stranding_data.get('total_strandings', 0)
        fig.suptitle(f'Vehicle Stranding Analysis - Total: {total} vehicles', 
                    fontsize=config.title_fontsize + 2, fontweight='bold',
                    color=self.colors['text'], y=0.98)
        
        plt.tight_layout()
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_stranding_analysis.png", 
                    dpi=config.dpi, bbox_inches='tight',
                    facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig

    def _plot_stranding_map(self, ax: plt.Axes, data: Dict, highway_length: float, # pyright: ignore[reportPrivateImportUsage]
                        stations: List[float], station_names: Optional[List[str]],
                        config: ChartConfig):
        """Plot the main highway map with stranding locations."""
        ax.set_facecolor(self.colors['background'])
        
        # Draw highway as thick line
        ax.plot([0, highway_length], [0, 0], color=self.colors['neutral'], 
            linewidth=10, alpha=0.6, solid_capstyle='round', zorder=1)
        
        # Draw station markers
        names = station_names or [f"S{i+1}" for i in range(len(stations))]
        for i, (pos, name) in enumerate(zip(stations, names)):
            ax.scatter([pos], [0], s=500, c=self.colors['success'], 
                    marker='s', zorder=5, edgecolors='white', linewidth=2)
            ax.annotate(f'{name}\n{pos:.0f}km', xy=(pos, 0), xytext=(0, 30),
                    textcoords='offset points', ha='center', fontsize=9,
                    color=self.colors['text'], fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=self.colors['success'],
                                alpha=0.3, edgecolor='none'))
            
            # Draw coverage zone (approximate 80km range)
            ax.axvspan(pos-40, pos+40, alpha=0.05, color=self.colors['success'], zorder=0)
        
        # Plot strandings
        positions = data.get('positions_km', [])
        if positions:
            # Add slight vertical jitter for visibility
            jitter = np.random.uniform(-0.15, 0.15, len(positions))
            colors = plt.cm.Reds(np.linspace(0.4, 1, len(positions))) # pyright: ignore[reportAttributeAccessIssue]
            
            for i, (pos, color) in enumerate(zip(positions, colors)):
                ax.scatter([pos], [jitter[i]], s=150, c=[color], 
                        marker='X', edgecolors='darkred', linewidth=1.5, zorder=10)
            
            # Add density estimation if enough points
            if len(positions) >= 5:
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(positions)
                    x_range = np.linspace(0, highway_length, 500)
                    density = kde(x_range)
                    # Normalize and scale for visibility
                    density = density / density.max() * 0.4
                    ax.fill_between(x_range, -density, density, 
                                alpha=0.3, color=self.colors['danger'], 
                                label='Stranding Density')
                except Exception:
                    pass  # KDE can fail with some data patterns
        
        # Highlight gaps between stations
        if len(stations) >= 2:
            sorted_stations = sorted(stations)
            for i in range(len(sorted_stations)-1):
                gap = sorted_stations[i+1] - sorted_stations[i]
                mid = (sorted_stations[i] + sorted_stations[i+1]) / 2
                if gap > 100:  # Highlight large gaps
                    ax.axvspan(sorted_stations[i], sorted_stations[i+1], 
                            alpha=0.1, color=self.colors['warning'], zorder=0)
                    ax.annotate(f'Gap: {gap:.0f}km', xy=(mid, -0.4), 
                            ha='center', fontsize=8, color=self.colors['warning'],
                            fontweight='bold')
        
        ax.set_xlim(-5, highway_length + 5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Highway Position (km)', fontsize=config.label_fontsize, 
                    color=self.colors['text'])
        ax.set_title('Stranding Locations Along Highway', 
                    fontsize=config.title_fontsize, fontweight='bold',
                    color=self.colors['text'])
        ax.set_yticks([])
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=self.colors['success'],
                markersize=12, label='Charging Station'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor=self.colors['danger'],
                markersize=10, label='Stranding Location')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=config.tick_fontsize)
        ax.tick_params(colors=self.colors['text'])

    def _plot_distance_analysis(self, ax: plt.Axes, data: Dict,  # pyright: ignore[reportPrivateImportUsage]
                            stations: List[float], config: ChartConfig):
        """Analyze distance from strandings to nearest station."""
        ax.set_facecolor(self.colors['background'])
        
        positions = data.get('positions_km', [])
        if not positions or not stations:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
            return
        
        # Calculate distances to nearest station
        distances = []
        for pos in positions:
            min_dist = min(abs(pos - s) for s in stations)
            distances.append(min_dist)
        
        # Histogram
        n, bins, patches = ax.hist(distances, bins=15, color=self.colors['danger'],
                                alpha=0.7, edgecolor=self.colors['text'])
        
        # Color bars by severity
        for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])): # pyright: ignore[reportArgumentType, reportGeneralTypeIssues]
            if left_edge < 10:
                patch.set_color(self.colors['success'])  # Close to station
            elif left_edge < 30:
                patch.set_color(self.colors['warning'])  # Moderate
            else:
                patch.set_color(self.colors['danger'])   # Far from station
        
        # Statistics lines
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        ax.axvline(mean_dist, color=self.colors['primary'], linestyle='-',  # pyright: ignore[reportArgumentType]
                linewidth=2, label=f'Mean: {mean_dist:.1f} km')
        ax.axvline(median_dist, color=self.colors['secondary'], linestyle='--',
                linewidth=2, label=f'Median: {median_dist:.1f} km')
        
        ax.set_xlabel('Distance to Nearest Station (km)', fontsize=config.label_fontsize,
                    color=self.colors['text'])
        ax.set_ylabel('Number of Strandings', fontsize=config.label_fontsize,
                    color=self.colors['text'])
        ax.set_title('Distance from Stranding to Nearest Station',
                    fontsize=config.title_fontsize, color=self.colors['text'])
        ax.legend(fontsize=config.tick_fontsize)
        ax.tick_params(colors=self.colors['text'])

    def _plot_temporal_distribution(self, ax: plt.Axes, data: Dict, config: ChartConfig): # pyright: ignore[reportPrivateImportUsage]
        """Plot when strandings occur throughout the simulation."""
        ax.set_facecolor(self.colors['background'])
        
        times = data.get('times', [])
        if not times or times[0] is None:
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
            return
        
        # Parse times
        hours = []
        for t in times:
            if t:
                try:
                    dt = datetime.fromisoformat(t) if isinstance(t, str) else t
                    hours.append(dt.hour + dt.minute/60)
                except Exception:
                    continue
        
        if hours:
            ax.hist(hours, bins=24, color=self.colors['info'], alpha=0.7,
                edgecolor=self.colors['text'])
            ax.set_xlabel('Hour of Day', fontsize=config.label_fontsize,
                        color=self.colors['text'])
            ax.set_ylabel('Number of Strandings', fontsize=config.label_fontsize,
                        color=self.colors['text'])
            ax.set_title('Temporal Distribution of Strandings',
                        fontsize=config.title_fontsize, color=self.colors['text'])
            ax.set_xlim(0, 24)
            ax.set_xticks(range(0, 25, 4))
            ax.tick_params(colors=self.colors['text'])
        else:
            ax.text(0.5, 0.5, 'No valid time data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    def _plot_driver_breakdown(self, ax: plt.Axes, data: Dict, config: ChartConfig): # pyright: ignore[reportPrivateImportUsage]
        """Break down strandings by driver type."""
        ax.set_facecolor(self.colors['background'])
        
        driver_types = data.get('driver_types', [])
        if not driver_types:
            ax.text(0.5, 0.5, 'No driver data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
            return
        
        # Count by type
        type_counts = {}
        for dt in driver_types:
            type_counts[dt] = type_counts.get(dt, 0) + 1
        
        # Sort by count
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        types, counts = zip(*sorted_types)
        
        # Color by risk level
        color_map = {
            'conservative': self.colors['success'],
            'balanced': self.colors['info'],
            'aggressive': self.colors['warning'],
            'range_anxious': self.colors['danger']
        }
        colors = [color_map.get(t, self.colors['neutral']) for t in types]
        
        bars = ax.bar(range(len(types)), counts, color=colors, alpha=0.8,
                    edgecolor=self.colors['text'])
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=45, ha='right')
        ax.set_ylabel('Number of Strandings', fontsize=config.label_fontsize,
                    color=self.colors['text'])
        ax.set_title('Strandings by Driver Type',
                    fontsize=config.title_fontsize, color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9,
                    color=self.colors['text'])

    def _plot_battery_analysis(self, ax: plt.Axes, data: Dict, config: ChartConfig): # pyright: ignore[reportPrivateImportUsage]
        """Analyze battery characteristics of stranded vehicles."""
        ax.set_facecolor(self.colors['background'])
        
        initial_socs = data.get('initial_socs', [])
        capacities = data.get('battery_capacities', [])
        
        if not initial_socs:
            ax.text(0.5, 0.5, 'No battery data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
            return
        
        # Scatter plot: capacity vs initial SOC
        scatter = ax.scatter(capacities, initial_socs, s=100, c=self.colors['danger'],
                            alpha=0.6, edgecolors=self.colors['text'])
        
        # Add trend line if enough points
        if len(capacities) >= 3:
            z = np.polyfit(capacities, initial_socs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(capacities), max(capacities), 100)
            ax.plot(x_line, p(x_line), "--", color=self.colors['primary'], 
                alpha=0.8, label='Trend')
        
        ax.set_xlabel('Battery Capacity (kWh)', fontsize=config.label_fontsize,
                    color=self.colors['text'])
        ax.set_ylabel('Initial SOC', fontsize=config.label_fontsize,
                    color=self.colors['text'])
        ax.set_title('Battery Characteristics of Stranded Vehicles',
                    fontsize=config.title_fontsize, color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.set_ylim(0, 1)

    def plot_stranding_heatmap(self, highway_length_km: float = 300.0, **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Heatmap visualization of where vehicles get stranded along the highway.
        Shows hotspots where infrastructure may be insufficient.
        """
        if self.df is None or 'strandings' not in self.df.columns:
            return self._empty_figure("No stranding data available")
        
        config = self._merge_config(kwargs)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=config.dpi, 
                                gridspec_kw={'height_ratios': [3, 1]})
        fig.patch.set_facecolor(self.colors['background'])
        
        # Get stranding events with positions (need to extract from event log or vehicle tracker)
        # This would come from VehicleTracker's failed_trips or event log
        
        ax1 = axes[0]
        ax1.set_facecolor(self.colors['background'])
        
        # Create highway schematic
        ax1.axhline(y=0, color=self.colors['neutral'], linewidth=4, alpha=0.8, label='Highway')
        
        # Plot charging stations
        station_positions = kwargs.get('station_positions', [])  # km positions
        for pos in station_positions:
            ax1.axvline(x=pos, color=self.colors['success'], linewidth=2, 
                    alpha=0.6, linestyle='--')
            ax1.scatter([pos], [0], s=200, c=self.colors['success'], 
                    marker='s', zorder=5, label='Charging Station' if pos == station_positions[0] else "")
        
        # Plot stranding locations (scatter with jitter for visibility)
        stranding_positions = kwargs.get('stranding_positions', [])
        stranding_times = kwargs.get('stranding_times', [])
        
        if stranding_positions:
            # Add vertical jitter for visibility
            jitter = np.random.uniform(-0.1, 0.1, len(stranding_positions))
            colors = plt.cm.Reds(np.linspace(0.4, 1, len(stranding_positions))) # pyright: ignore[reportAttributeAccessIssue]
            
            scatter = ax1.scatter(stranding_positions, jitter, 
                                c=range(len(stranding_positions)), 
                                cmap='Reds', s=100, alpha=0.8, 
                                edgecolors='darkred', linewidth=1, zorder=10)
            
            # Add colorbar for temporal progression
            cbar = plt.colorbar(scatter, ax=ax1, pad=0.02)
            cbar.set_label('Time Progression', color=self.colors['text'])
        
        # KDE plot for density
        if len(stranding_positions) > 3:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(stranding_positions)
            x_range = np.linspace(0, highway_length_km, 500)
            density = kde(x_range)
            ax1.fill_between(x_range, -density*0.5, density*0.5, 
                            alpha=0.3, color=self.colors['danger'], label='Stranding Density')
        
        ax1.set_xlim(0, highway_length_km)
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('Highway Position (km)', fontsize=config.label_fontsize, 
                    color=self.colors['text'])
        ax1.set_title('Vehicle Stranding Locations Along Highway', 
                    fontsize=config.title_fontsize, fontweight='bold', 
                    color=self.colors['text'])
        ax1.set_yticks([])
        ax1.legend(loc='upper right', fontsize=config.tick_fontsize)
        ax1.tick_params(colors=self.colors['text'])
        
        # Bottom: Histogram of stranding distances from nearest station
        ax2 = axes[1]
        ax2.set_facecolor(self.colors['background'])
        
        if stranding_positions and station_positions:
            # Calculate distance to nearest station for each stranding
            distances_to_station = []
            for pos in stranding_positions:
                min_dist = min(abs(pos - s) for s in station_positions)
                distances_to_station.append(min_dist)
            
            ax2.hist(distances_to_station, bins=20, color=self.colors['danger'], 
                    alpha=0.7, edgecolor=self.colors['text'])
            ax2.axvline(x=np.mean(distances_to_station), color=self.colors['primary'], 
                    linestyle='-', linewidth=2, label=f'Mean: {np.mean(distances_to_station):.1f} km')
            ax2.axvline(x=np.median(distances_to_station), color=self.colors['secondary'], 
                    linestyle='--', linewidth=2, label=f'Median: {np.median(distances_to_station):.1f} km')
            
            ax2.set_xlabel('Distance to Nearest Charging Station (km)', 
                        fontsize=config.label_fontsize, color=self.colors['text'])
            ax2.set_ylabel('Number of Strandings', fontsize=config.label_fontsize, 
                        color=self.colors['text'])
            ax2.set_title('Stranding Distance from Nearest Station', 
                        fontsize=config.title_fontsize, color=self.colors['text'])
            ax2.legend(fontsize=config.tick_fontsize)
            ax2.tick_params(colors=self.colors['text'])
        
        plt.tight_layout()
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_stranding_heatmap.png", 
                    dpi=config.dpi, bbox_inches='tight',
                    facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig    
    
    def plot_required_kpis(self, **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Plot the two REQUIRED KPIs: cars quit waiting and cars with 0% battery.
        """
        if self.df is None or self.df.empty:
            return self._empty_figure("No data available")
        
        config = self._merge_config(kwargs)
        fig, axes = plt.subplots(2, 1, figsize=config.figure_size, dpi=config.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        time_index = pd.to_datetime(self.df['timestamp']) if 'timestamp' in self.df.columns else self.df.index
        
        # KPI 1: Cars quit waiting
        ax1 = axes[0]
        quit_data = self.df['cars_quit_waiting']
        
        # Bar plot for discrete events
        ax1.bar(time_index, quit_data, color=self.colors['danger'], 
                alpha=0.7, width=0.02, label='Step value')
        
        # Rolling average
        if len(quit_data) >= 10:
            rolling = quit_data.rolling(window=10, min_periods=1).mean()
            ax1.plot(time_index, rolling, color=self.colors['primary'], 
                    linewidth=2, label='10-step average')
        
        # Cumulative
        ax1_twin = ax1.twinx()
        cumulative = quit_data.cumsum()
        ax1_twin.plot(time_index, cumulative, color=self.colors['secondary'], 
                     linewidth=2, linestyle='--', alpha=0.8, label='Cumulative')
        ax1_twin.set_ylabel('Cumulative', color=self.colors['secondary'], 
                           fontsize=config.label_fontsize)
        ax1_twin.tick_params(axis='y', labelcolor=self.colors['secondary'])
        
        ax1.set_title('Cars Quit Waiting (Abandonments)',
                     fontsize=config.title_fontsize, fontweight='bold',
                     color=self.colors['text'])
        ax1.set_ylabel('Vehicles per Step', fontsize=config.label_fontsize, 
                      color=self.colors['text'])
        ax1.set_xlabel('Time', fontsize=config.label_fontsize, color=self.colors['text'])
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.tick_params(colors=self.colors['text'])
        
        # Add threshold line
        if quit_data.max() > 0:
            threshold = quit_data.mean() + 2 * quit_data.std()
            ax1.axhline(y=threshold, color=self.colors['warning'], 
                       linestyle=':', alpha=0.8, label=f'Alert threshold ({threshold:.1f})')
        
        # KPI 2: Cars with 0% battery (strandings)
        ax2 = axes[1]
        stranded_data = self.df['cars_zero_battery']
        
        # Highlight critical events
        colors = [self.colors['danger'] if x > 0 else self.colors['success'] 
                 for x in stranded_data]
        ax2.bar(time_index, stranded_data, color=colors, alpha=0.8, width=0.02)
        
        # Cumulative strandings
        ax2_twin = ax2.twinx()
        cum_stranded = stranded_data.cumsum()
        ax2_twin.fill_between(time_index, cum_stranded, alpha=0.3, 
                             color=self.colors['danger'], label='Total stranded')
        ax2_twin.plot(time_index, cum_stranded, color=self.colors['danger'], 
                     linewidth=2)
        ax2_twin.set_ylabel('Total Stranded', color=self.colors['danger'], 
                           fontsize=config.label_fontsize)
        
        # Annotate total
        if len(cum_stranded) > 0:
            final_total = cum_stranded.iloc[-1]
            ax2_twin.annotate(f'Total: {int(final_total)}', 
                            xy=(time_index.iloc[-1], final_total), # pyright: ignore[reportAttributeAccessIssue]
                            xytext=(10, 0), textcoords='offset points',
                            fontsize=config.annotation_fontsize,
                            color=self.colors['danger'], fontweight='bold')
        
        ax2.set_title('Cars with 0% Battery (Strandings)',
                     fontsize=config.title_fontsize, fontweight='bold',
                     color=self.colors['text'])
        ax2.set_ylabel('Vehicles per Step', fontsize=config.label_fontsize, 
                      color=self.colors['text'])
        ax2.set_xlabel('Time', fontsize=config.label_fontsize, color=self.colors['text'])
        ax2.tick_params(colors=self.colors['text'])
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_required_kpis.png", 
                       dpi=config.dpi, bbox_inches='tight', 
                       facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig
    
    def plot_system_overview(self, **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Comprehensive dashboard showing key system metrics.
        """
        if self.df is None or self.df.empty:
            return self._empty_figure("No data available")
        
        config = self._merge_config(kwargs)
        
        # Create grid layout
        fig = plt.figure(figsize=(16, 10), dpi=config.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        time_index = pd.to_datetime(self.df['timestamp']) if 'timestamp' in self.df.columns else self.df.index
        
        # 1. Vehicle counts (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_vehicle_counts(ax1, time_index, config) # pyright: ignore[reportArgumentType]
        
        # 2. Charger occupancy % (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_charger_occupancy(ax2, time_index, config) # pyright: ignore[reportArgumentType]
        
        # 3. Queue and charging (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_queue_metrics(ax3, time_index, config) # pyright: ignore[reportArgumentType]
        
        # 4. Wait times (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_wait_times(ax4, time_index, config) # pyright: ignore[reportArgumentType]
        
        # 5. Battery status (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_battery_status(ax5, time_index, config) # pyright: ignore[reportArgumentType]
        
        # 6. Service level and satisfaction (bottom, spans all)
        ax6 = fig.add_subplot(gs[2, :])
        self._plot_service_metrics(ax6, time_index, config) # pyright: ignore[reportArgumentType]
        
        plt.suptitle('EV Charging System Overview Dashboard', 
                    fontsize=config.title_fontsize + 2, fontweight='bold', 
                    color=self.colors['text'], y=0.98)
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_overview.png", 
                       dpi=config.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig
    
    def _plot_vehicle_counts(self, ax: plt.Axes, time_index: pd.DatetimeIndex,  # pyright: ignore[reportPrivateImportUsage]
                            config: ChartConfig):
        """Plot vehicle count time series with dual y-axis for better visibility."""
        ax.set_facecolor(self.colors['background'])

        # Primary axis: vehicles on road
        ax.fill_between(time_index, self.df['vehicles_on_road'],  # pyright: ignore[reportOptionalSubscript]
                       alpha=0.3, color=self.colors['primary'], label='On road')
        line1, = ax.plot(time_index, self.df['vehicles_on_road'],   # pyright: ignore[reportOptionalSubscript]
               color=self.colors['primary'], linewidth=2, label='On road')

        ax.set_ylabel('Vehicles on Road', fontsize=config.label_fontsize,
                     color=self.colors['primary'])
        ax.tick_params(axis='y', labelcolor=self.colors['primary'])

        # Secondary axis: charging and queued (for better visibility)
        ax2 = ax.twinx()
        line2, = ax2.plot(time_index, self.df['total_charging'],   # pyright: ignore[reportOptionalSubscript]
               color=self.colors['success'], linewidth=2.5, label='Charging')
        line3, = ax2.plot(time_index, self.df['total_queued'],   # pyright: ignore[reportOptionalSubscript]
               color=self.colors['warning'], linewidth=2.5, linestyle='--', label='Queued')

        ax2.set_ylabel('Charging / Queued', fontsize=config.label_fontsize,
                      color=self.colors['success'])
        ax2.tick_params(axis='y', labelcolor=self.colors['success'])

        # Ensure secondary axis starts at 0 and has reasonable range
        max_station_activity = max(self.df['total_charging'].max(), self.df['total_queued'].max(), 1)  # pyright: ignore[reportOptionalSubscript]
        ax2.set_ylim(0, max_station_activity * 1.2)

        ax.set_title('Vehicle Distribution', fontsize=config.title_fontsize,
                    color=self.colors['text'], fontweight='bold')

        # Combined legend
        lines = [line1, line2, line3]
        labels = ['On road', 'Charging', 'Queued']
        ax.legend(lines, labels, loc='upper left', fontsize=config.tick_fontsize)
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    def _plot_charger_occupancy(self, ax: plt.Axes, time_index: pd.DatetimeIndex, # pyright: ignore[reportPrivateImportUsage]
                               config: ChartConfig):
        """Plot charger occupancy percentage over time."""
        ax.set_facecolor(self.colors['background'])

        utilization = self.df['charging_utilization'] * 100 # pyright: ignore[reportOptionalSubscript]

        # Color based on severity
        colors = [self.colors['danger'] if x > 95 else
                 self.colors['warning'] if x > 80 else self.colors['success']
                 for x in utilization]

        ax.scatter(time_index, utilization, c=colors, alpha=0.6, s=20)
        ax.plot(time_index, utilization, color=self.colors['neutral'],
               alpha=0.5, linewidth=1)

        # Threshold lines
        ax.axhline(y=100, color=self.colors['danger'], linestyle='--',
                  alpha=0.7, label='Full')
        ax.axhline(y=80, color=self.colors['warning'], linestyle=':',
                  alpha=0.7, label='High')

        ax.set_title('Charger Occupancy %', fontsize=config.title_fontsize,
                    color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('Occupancy %', fontsize=config.label_fontsize, color=self.colors['text'])
        ax.set_ylim(0, max(110, utilization.max() * 1.1))
        ax.legend(fontsize=config.tick_fontsize, loc='lower right')
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    def _plot_queue_metrics(self, ax: plt.Axes, time_index: pd.DatetimeIndex,  # pyright: ignore[reportPrivateImportUsage]
                           config: ChartConfig):
        """Plot queue occupancy and utilization."""
        ax.set_facecolor(self.colors['background'])
        
        occupancy = self.df['queue_occupancy_rate'] * 100 # pyright: ignore[reportOptionalSubscript]
        
        # Color based on severity
        colors = [self.colors['danger'] if x > 90 else 
                 self.colors['warning'] if x > 70 else self.colors['success']
                 for x in occupancy]
        
        ax.scatter(time_index, occupancy, c=colors, alpha=0.6, s=20)
        ax.plot(time_index, occupancy, color=self.colors['neutral'], 
               alpha=0.5, linewidth=1)
        
        # Add threshold lines
        ax.axhline(y=100, color=self.colors['danger'], linestyle='--', 
                  alpha=0.7, label='Capacity')
        ax.axhline(y=70, color=self.colors['warning'], linestyle=':', 
                  alpha=0.7, label='Warning')
        
        ax.set_title('Queue Occupancy %', fontsize=config.title_fontsize, 
                    color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('Occupancy %', fontsize=config.label_fontsize, color=self.colors['text'])
        ax.set_ylim(0, max(110, occupancy.max() * 1.1))
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    def _plot_wait_times(self, ax: plt.Axes, time_index: pd.DatetimeIndex,  # pyright: ignore[reportPrivateImportUsage]
                        config: ChartConfig):
        """Plot wait time distribution."""
        ax.set_facecolor(self.colors['background'])
        
        wait_times = self.df['avg_wait_time_minutes'] # pyright: ignore[reportOptionalSubscript]
        
        # Gradient fill
        ax.fill_between(time_index, wait_times, alpha=0.4, 
                       color=self.colors['info'])
        ax.plot(time_index, wait_times, color=self.colors['info'], 
               linewidth=2, label='Average')
        
        if 'max_wait_time_minutes' in self.df.columns: # pyright: ignore[reportOptionalMemberAccess]
            ax.plot(time_index, self.df['max_wait_time_minutes'],  # pyright: ignore[reportOptionalSubscript]
                   color=self.colors['danger'], linewidth=1, 
                   alpha=0.7, linestyle='--', label='Maximum')
        
        # Target line
        ax.axhline(y=15, color=self.colors['success'], linestyle=':', 
                  alpha=0.7, label='Target (15 min)')
        
        ax.set_title('Wait Times', fontsize=config.title_fontsize, 
                    color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('Minutes', fontsize=config.label_fontsize, color=self.colors['text'])
        ax.legend(loc='upper left', fontsize=config.tick_fontsize)
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    def _plot_battery_status(self, ax: plt.Axes, time_index: pd.DatetimeIndex,  # pyright: ignore[reportPrivateImportUsage]
                            config: ChartConfig):
        """Plot battery SOC distribution over time."""
        ax.set_facecolor(self.colors['background'])
        
        soc = self.df['avg_soc_percent'] # pyright: ignore[reportOptionalSubscript]
        
        # Color gradient based on SOC level
        colors = plt.cm.RdYlGn(soc / 100) # pyright: ignore[reportAttributeAccessIssue]
        
        ax.scatter(time_index, soc, c=soc, cmap='RdYlGn', 
                  vmin=0, vmax=100, s=30, alpha=0.8)
        ax.plot(time_index, soc, color=self.colors['neutral'], 
               alpha=0.5, linewidth=1)
        
        # Critical zone
        ax.axhspan(0, 20, alpha=0.2, color=self.colors['danger'], 
                  label='Critical (<20%)')
        
        ax.set_title('Average Battery SOC', fontsize=config.title_fontsize, 
                    color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('SOC %', fontsize=config.label_fontsize, color=self.colors['text'])
        ax.set_ylim(0, 100)
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    def _plot_service_metrics(self, ax: plt.Axes, time_index: pd.DatetimeIndex,  # pyright: ignore[reportPrivateImportUsage]
                             config: ChartConfig):
        """Plot service level and customer satisfaction."""
        ax.set_facecolor(self.colors['background'])

        service = self.df['service_level'] * 100 # pyright: ignore[reportOptionalSubscript]
        satisfaction = self.df['customer_satisfaction_score'] * 100 # pyright: ignore[reportOptionalSubscript]

        ax.plot(time_index, service, color=self.colors['success'],
               linewidth=2.5, label='Service Level', marker='o', markersize=3)
        ax.plot(time_index, satisfaction, color=self.colors['secondary'],
               linewidth=2.5, label='Satisfaction', marker='s', markersize=3)

        # Calculate dynamic y-axis range to show variation better
        min_val = min(service.min(), satisfaction.min())

        # If values are clustered high (>80%), zoom in to show variation
        if min_val > 80:
            y_min = max(0, min_val - 5)
            y_max = 105
            # Target zones adjusted
            ax.axhspan(90, 100, alpha=0.15, color=self.colors['success'], label='Excellent')
            ax.axhline(90, color=self.colors['success'], linestyle='--', alpha=0.5)
        else:
            y_min = 0
            y_max = 105
            # Full target zones
            ax.axhspan(90, 100, alpha=0.1, color=self.colors['success'])
            ax.axhspan(70, 90, alpha=0.1, color=self.colors['warning'])
            ax.axhspan(0, 70, alpha=0.1, color=self.colors['danger'])

        ax.set_title('Service Quality Metrics', fontsize=config.title_fontsize,
                    color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('Score %', fontsize=config.label_fontsize, color=self.colors['text'])
        ax.set_ylim(y_min, y_max)
        ax.legend(loc='lower right', fontsize=config.label_fontsize)
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    # ========================================================================
    # DISTRIBUTION AND ANALYSIS PLOTS
    # ========================================================================
    
    def plot_distributions(self, **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Plot distribution histograms for key metrics.
        """
        if self.df is None or self.df.empty:
            return self._empty_figure("No data available")
        
        config = self._merge_config(kwargs)
        fig, axes = plt.subplots(2, 3, figsize=config.figure_size, dpi=config.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        metrics = [
            ('cars_quit_waiting', 'Cars Quit Waiting', self.colors['danger']),
            ('cars_zero_battery', 'Cars 0% Battery', self.colors['danger']),
            ('avg_wait_time_minutes', 'Wait Time (min)', self.colors['info']),
            ('queue_occupancy_rate', 'Queue Occupancy', self.colors['warning']),
            ('charging_utilization', 'Charger Utilization', self.colors['success']),
            ('avg_soc_percent', 'Average SOC %', self.colors['secondary'])
        ]
        
        for ax, (col, title, color) in zip(axes.flat, metrics):
            ax.set_facecolor(self.colors['background'])
            
            if col in self.df.columns:
                data = self.df[col].dropna()
                
                # Histogram with KDE
                ax.hist(data, bins=30, alpha=0.7, color=color, 
                       edgecolor=self.colors['text'], density=True)
                
                # Add statistics
                mean = data.mean()
                median = data.median()
                ax.axvline(mean, color=self.colors['primary'], 
                          linestyle='-', linewidth=2, label=f'Mean: {mean:.2f}')
                ax.axvline(median, color=self.colors['secondary'], 
                          linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
                
                ax.set_title(title, fontsize=config.title_fontsize, 
                          color=self.colors['text'], fontweight='bold')
                ax.set_xlabel('Value', fontsize=config.label_fontsize, color=self.colors['text'])
                ax.set_ylabel('Density', fontsize=config.label_fontsize, color=self.colors['text'])
                ax.legend(fontsize=config.tick_fontsize)
                ax.tick_params(colors=self.colors['text'])
        
        plt.tight_layout()
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_distributions.png", 
                       dpi=config.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig
    
    def plot_correlation_heatmap(self, **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Plot correlation heatmap of KPIs.
        """
        if self.df is None or self.df.empty:
            return self._empty_figure("No data available")
        
        config = self._merge_config(kwargs)
        
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Compute correlation
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=config.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        ax.set_facecolor(self.colors['background'])
        
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', 
                   cmap=self.cmap_diverging, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax, annot_kws={'size': config.tick_fontsize})
        
        ax.set_title('KPI Correlation Matrix', fontsize=config.title_fontsize + 2, 
                    color=self.colors['text'], fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right', color=self.colors['text'])
        plt.yticks(rotation=0, color=self.colors['text'])
        
        plt.tight_layout()
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_correlation.png", 
                       dpi=config.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig
    
    # ========================================================================
    # STATION-SPECIFIC VISUALIZATIONS
    # ========================================================================
    
    def plot_station_comparison(self, station_stats: Dict[str, Dict], **kwargs) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """
        Compare performance across multiple charging stations.
        """
        config = self._merge_config(kwargs)
        fig, axes = plt.subplots(2, 2, figsize=config.figure_size, dpi=config.dpi)
        fig.patch.set_facecolor(self.colors['background'])
        
        stations = list(station_stats.keys())
        if not stations:
            return self._empty_figure("No station data")
        
        # Extract metrics
        utilization = [station_stats[s]['current_utilization'] * 100 for s in stations]
        avg_wait = [station_stats[s]['avg_wait_time_minutes'] for s in stations]
        abandonment = [station_stats[s]['abandonment_rate'] * 100 for s in stations]
        completed = [station_stats[s]['completed_sessions'] for s in stations]
        
        # 1. Utilization comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(stations, utilization, color=self.colors['primary'], alpha=0.8)
        ax1.set_title('Charger Utilization %', fontsize=config.title_fontsize, 
                     color=self.colors['text'], fontweight='bold')
        ax1.set_ylabel('Utilization %', fontsize=config.label_fontsize, color=self.colors['text'])
        ax1.tick_params(axis='x', rotation=45, colors=self.colors['text'])
        ax1.tick_params(axis='y', colors=self.colors['text'])
        ax1.axhline(y=85, color=self.colors['warning'], linestyle='--', alpha=0.7)
        
        # Color bars by performance
        for bar, val in zip(bars1, utilization):
            if val > 90:
                bar.set_color(self.colors['danger'])
            elif val < 30:
                bar.set_color(self.colors['warning'])
        
        # 2. Wait time comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(stations, avg_wait, color=self.colors['info'], alpha=0.8)
        ax2.set_title('Average Wait Time', fontsize=config.title_fontsize, 
                     color=self.colors['text'], fontweight='bold')
        ax2.set_ylabel('Minutes', fontsize=config.label_fontsize, color=self.colors['text'])
        ax2.tick_params(axis='x', rotation=45, colors=self.colors['text'])
        ax2.tick_params(axis='y', colors=self.colors['text'])
        
        # 3. Abandonment rate
        ax3 = axes[1, 0]
        bars3 = ax3.bar(stations, abandonment, color=self.colors['danger'], alpha=0.8)
        ax3.set_title('Abandonment Rate %', fontsize=config.title_fontsize, 
                     color=self.colors['text'], fontweight='bold')
        ax3.set_ylabel('Abandonment %', fontsize=config.label_fontsize, color=self.colors['text'])
        ax3.tick_params(axis='x', rotation=45, colors=self.colors['text'])
        ax3.tick_params(axis='y', colors=self.colors['text'])
        
        # 4. Throughput
        ax4 = axes[1, 1]
        bars4 = ax4.bar(stations, completed, color=self.colors['success'], alpha=0.8)
        ax4.set_title('Completed Sessions', fontsize=config.title_fontsize, 
                     color=self.colors['text'], fontweight='bold')
        ax4.set_ylabel('Count', fontsize=config.label_fontsize, color=self.colors['text'])
        ax4.tick_params(axis='x', rotation=45, colors=self.colors['text'])
        ax4.tick_params(axis='y', colors=self.colors['text'])
        
        plt.suptitle('Station Performance Comparison', 
                    fontsize=config.title_fontsize + 2, fontweight='bold', 
                    color=self.colors['text'])
        plt.tight_layout()
        
        if config.save_path:
            plt.savefig(f"{config.save_path}_station_comparison.png", 
                       dpi=config.dpi, bbox_inches='tight',
                       facecolor=self.colors['background'])
        
        if config.show_plot:
            plt.show()
        
        return fig
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _merge_config(self, kwargs: Dict) -> ChartConfig:
        """Merge provided kwargs with default config."""
        defaults = {
            'figure_size': self.config.figure_size,
            'dpi': self.config.dpi,
            'save_path': self.config.save_path,
            'show_plot': self.config.show_plot,
            'title_fontsize': self.config.title_fontsize,
            'label_fontsize': self.config.label_fontsize,
            'tick_fontsize': self.config.tick_fontsize,
            'annotation_fontsize': self.config.annotation_fontsize
        }
        defaults.update(kwargs)
        return ChartConfig(**defaults)
    
    def _empty_figure(self, message: str) -> plt.Figure: # pyright: ignore[reportPrivateImportUsage]
        """Create empty figure with message."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    def generate_full_report(self, output_dir: str = "./visualization_output", 
                            station_data: Dict[str, pd.DataFrame] = None) -> List[str]: # pyright: ignore[reportArgumentType]
        """
        Generate all visualizations and save to directory.
        
        Args:
            output_dir: Directory to save visualizations
            station_data: Optional dict mapping station_id to DataFrame with occupancy data.
                         If provided, charging area visualizations will be generated.
            
        Returns:
            List of generated file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Use a clean prefix without UUID — the run directory name
        # already identifies the run.
        base_path = os.path.join(output_dir, "simulation")

        generated = []

        # Required KPIs
        try:
            self.plot_required_kpis(save_path=base_path, show_plot=False)
            generated.append(f"{base_path}_required_kpis.png")
        except Exception as e:
            print(f"Warning: Could not generate required KPIs plot: {e}")

        # System overview
        try:
            self.plot_system_overview(save_path=base_path, show_plot=False)
            generated.append(f"{base_path}_overview.png")
        except Exception as e:
            print(f"Warning: Could not generate system overview: {e}")

        # Distributions
        try:
            self.plot_distributions(save_path=base_path, show_plot=False)
            generated.append(f"{base_path}_distributions.png")
        except Exception as e:
            print(f"Warning: Could not generate distributions: {e}")

        # Correlation
        try:
            self.plot_correlation_heatmap(save_path=base_path, show_plot=False)
            generated.append(f"{base_path}_correlation.png")
        except Exception as e:
            print(f"Warning: Could not generate correlation heatmap: {e}")

        # NEW: Charging area occupancy visualizations
        if station_data is None:
            station_data = getattr(self, '_station_data', None)
        
        if station_data:
            try:
                self.plot_charging_area_occupancy(station_data, save_path=base_path, show_plot=False)
                generated.append(f"{base_path}_station_occupancy.png")
                print(f"  Generated station occupancy plot")
            except Exception as e:
                print(f"Warning: Could not generate station occupancy plot: {e}")
            
            try:
                self.plot_station_utilization_summary(station_data, save_path=base_path, show_plot=False)
                generated.append(f"{base_path}_station_utilization_heatmap.png")
                print(f"  Generated station utilization heatmap")
            except Exception as e:
                print(f"Warning: Could not generate station utilization heatmap: {e}")
        else:
            print("Note: No station data provided - skipping charging area visualizations")
            print("      Pass station_data to generate_full_report() or use set_station_data()")

        print(f"Generated {len(generated)} visualizations in {output_dir}")
        return generated
    
    def __repr__(self) -> str:
        return f"VisualizationTool({self.id}, records={len(self.df) if self.df is not None else 0})"
