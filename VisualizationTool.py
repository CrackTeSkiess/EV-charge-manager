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
    
    # ========================================================================
    # CORE TIME SERIES PLOTS
    # ========================================================================
    
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
        
        ax1.set_title('🚗 Cars Quit Waiting (Abandonments)', 
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
        
        ax2.set_title('🔋 Cars with 0% Battery (Strandings)', 
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
        
        # 2. Current statistics cards (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_stats_cards(ax2, config)
        
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
        """Plot vehicle count time series."""
        ax.set_facecolor(self.colors['background'])
        
        ax.fill_between(time_index, self.df['vehicles_on_road'],  # pyright: ignore[reportOptionalSubscript]
                       alpha=0.3, color=self.colors['primary'], label='On road')
        ax.plot(time_index, self.df['vehicles_on_road'],   # pyright: ignore[reportOptionalSubscript]
               color=self.colors['primary'], linewidth=2)
        
        ax.plot(time_index, self.df['total_charging'],   # pyright: ignore[reportOptionalSubscript]
               color=self.colors['success'], linewidth=2, label='Charging')
        ax.plot(time_index, self.df['total_queued'],   # pyright: ignore[reportOptionalSubscript]
               color=self.colors['warning'], linewidth=2, label='Queued')
        
        ax.set_title('Vehicle Distribution', fontsize=config.title_fontsize, 
                    color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('Count', fontsize=config.label_fontsize, color=self.colors['text'])
        ax.legend(loc='upper left')
        ax.tick_params(colors=self.colors['text'])
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    
    def _plot_stats_cards(self, ax: plt.Axes, config: ChartConfig): # pyright: ignore[reportPrivateImportUsage]
        """Display current statistics as text cards."""
        ax.set_facecolor(self.colors['background'])
        ax.axis('off')
        
        if self.df.empty: # pyright: ignore[reportOptionalMemberAccess]
            return
        
        latest = self.df.iloc[-1] # pyright: ignore[reportOptionalMemberAccess]
        
        stats = [
            ('🚗 Total Vehicles', f"{int(latest['vehicles_on_road'])}", self.colors['primary']),
            ('⚡ Charging', f"{int(latest['total_charging'])}", self.colors['success']),
            ('⏳ Queued', f"{int(latest['total_queued'])}", self.colors['warning']),
            ('⏱️ Avg Wait', f"{latest['avg_wait_time_minutes']:.1f} min", self.colors['info']),
            ('🔋 Avg SOC', f"{latest['avg_soc_percent']:.1f}%", self.colors['secondary']),
            ('✅ Service Level', f"{latest['service_level']*100:.1f}%", self.colors['success']),
        ]
        
        y_pos = 0.9
        for label, value, color in stats:
            # Card background
            rect = FancyBboxPatch((0.05, y_pos - 0.12), 0.9, 0.14,
                                 boxstyle="round,pad=0.01",
                                 facecolor=color, alpha=0.2,
                                 edgecolor=color, linewidth=2,
                                 transform=ax.transAxes)
            ax.add_patch(rect)
            
            # Text
            ax.text(0.1, y_pos, label, fontsize=config.label_fontsize,
                   transform=ax.transAxes, color=self.colors['text'], 
                   fontweight='bold', va='center')
            ax.text(0.9, y_pos, value, fontsize=config.title_fontsize,
                   transform=ax.transAxes, color=color, 
                   fontweight='bold', va='center', ha='right')
            
            y_pos -= 0.16
        
        ax.set_title('Current Status', fontsize=config.title_fontsize, 
                    color=self.colors['text'], fontweight='bold', y=0.98)
    
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
        
        # Target zones
        ax.axhspan(90, 100, alpha=0.1, color=self.colors['success'])
        ax.axhspan(70, 90, alpha=0.1, color=self.colors['warning'])
        ax.axhspan(0, 70, alpha=0.1, color=self.colors['danger'])
        
        ax.set_title('Service Quality Metrics', fontsize=config.title_fontsize, 
                    color=self.colors['text'], fontweight='bold')
        ax.set_ylabel('Score %', fontsize=config.label_fontsize, color=self.colors['text'])
        ax.set_ylim(0, 105)
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
    
    def generate_full_report(self, output_dir: str = "./visualization_output") -> List[str]:
        """
        Generate all visualizations and save to directory.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        base_path = f"{output_dir}/simulation_{self.id}"
        
        generated = []
        
        # Required KPIs
        self.plot_required_kpis(save_path=base_path, show_plot=False)
        generated.append(f"{base_path}_required_kpis.png")
        
        # System overview
        self.plot_system_overview(save_path=base_path, show_plot=False)
        generated.append(f"{base_path}_overview.png")
        
        # Distributions
        self.plot_distributions(save_path=base_path, show_plot=False)
        generated.append(f"{base_path}_distributions.png")
        
        # Correlation
        self.plot_correlation_heatmap(save_path=base_path, show_plot=False)
        generated.append(f"{base_path}_correlation.png")
        
        print(f"Generated {len(generated)} visualizations in {output_dir}")
        return generated
    
    def __repr__(self) -> str:
        return f"VisualizationTool({self.id}, records={len(self.df) if self.df is not None else 0})"
