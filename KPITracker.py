"""
KPI Tracker Module
Comprehensive performance monitoring and analytics for EV charging simulation.
"""

from __future__ import annotations

import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum, auto

import pandas as pd
import numpy as np


class KPIAlertLevel(Enum):
    """Severity levels for KPI threshold violations."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


@dataclass
class KPIThreshold:
    """Threshold configuration for alerting."""
    warning_min: Optional[float] = None
    warning_max: Optional[float] = None
    critical_min: Optional[float] = None
    critical_max: Optional[float] = None
    
    def check(self, value: float) -> Optional[KPIAlertLevel]:
        """Check if value violates thresholds."""
        if self.critical_min is not None and value < self.critical_min:
            return KPIAlertLevel.CRITICAL
        if self.critical_max is not None and value > self.critical_max:
            return KPIAlertLevel.CRITICAL
        if self.warning_min is not None and value < self.warning_min:
            return KPIAlertLevel.WARNING
        if self.warning_max is not None and value > self.warning_max:
            return KPIAlertLevel.WARNING
        return None


@dataclass
class KPIRecord:
    """Single timestep KPI record."""
    # Identification
    step: int
    timestamp: datetime
    simulation_id: str
    
    # REQUIRED KPIs
    cars_quit_waiting: int  # Number of cars that abandoned queue this step
    cars_zero_battery: int  # Number of cars stranded with 0% battery this step
    
    # Vehicle Flow KPIs
    vehicles_on_road: int
    vehicles_entered: int
    vehicles_exited: int
    vehicles_stranded: int
    completion_rate: float  # % of entered vehicles that completed trip
    
    # Charging Infrastructure KPIs
    total_charging: int
    total_queued: int
    total_waiting_spots: int
    queue_occupancy_rate: float  # queued / waiting_spots
    avg_wait_time_minutes: float
    max_wait_time_minutes: float
    charging_utilization: float  # % of chargers in use
    
    # Station-specific KPIs (flattened)
    num_stations: int
    stations_over_capacity: int  # Stations with queue >= waiting_spots
    stations_with_abandonments: int  # Stations with quitters this step
    
    # Battery Status KPIs
    avg_soc_percent: float
    critical_battery_count: int  # SOC < 10%
    low_battery_count: int  # SOC < 20%
    
    # Performance KPIs
    throughput_vehicles_per_hour: float
    avg_trip_duration_minutes: float
    energy_consumption_total_kwh: float
    
    # Efficiency KPIs
    charging_efficiency: float  # Energy delivered / energy requested
    station_turnaround_time: float  # Time from arrival to departure
    abandonment_rate: float  # quit_waiting / entered
    
    # Economic/Service KPIs
    estimated_revenue: float  # Based on energy delivered
    service_level: float  # % of vehicles served without excessive wait
    customer_satisfaction_score: float  # Composite metric


class KPITracker:
    """
    Tracks Key Performance Indicators for EV charging simulation.
    
    Core functionality:
    - Collects KPIs at each simulation step
    - Stores in pandas DataFrame for analysis
    - Provides real-time alerting on threshold violations
    - Computes rolling statistics and trends
    - Generates summary reports and visualizations
    """
    
    def __init__(
        self,
        simulation_id: Optional[str] = None,
        max_history_steps: int = 10000,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        self.simulation_id = simulation_id or str(uuid.uuid4())[:8]
        self.max_history = max_history_steps
        self.alert_callbacks = alert_callbacks or []
        
        # Data storage
        self.records: List[KPIRecord] = []
        self.df: Optional[pd.DataFrame] = None
        self._df_needs_update = False
        
        # Alerting configuration
        self.thresholds: Dict[str, KPIThreshold] = self._default_thresholds()
        
        # Cumulative counters (for computing step deltas)
        self._cumulative: Dict[str, int] = defaultdict(int)
        self._last_step_vehicles: Set[str] = set()
        
        # Rolling windows for trend analysis
        self._recent_abandonments: deque = deque(maxlen=60)  # 1 hour
        self._recent_strandings: deque = deque(maxlen=60)
        
        # Station state tracking (for quit detection)
        self._station_previous_queues: Dict[str, Dict[str, Any]] = {}
        
        # Cumulative tracking for rates
        self._total_entered: int = 0
        self._total_exited: int = 0
        
        # Alert log
        self.alerts: List[Dict] = []
        self.max_alerts = 1000
    
    def _default_thresholds(self) -> Dict[str, KPIThreshold]:
        """Default KPI thresholds for alerting."""
        return {
            'cars_quit_waiting': KPIThreshold(
                warning_max=5, critical_max=10
            ),
            'cars_zero_battery': KPIThreshold(
                warning_max=1, critical_max=3
            ),
            'queue_occupancy_rate': KPIThreshold(
                warning_max=0.8, critical_max=1.0
            ),
            'charging_utilization': KPIThreshold(
                warning_min=0.3, warning_max=0.95, critical_max=1.0
            ),
            'avg_wait_time_minutes': KPIThreshold(
                warning_max=15, critical_max=30
            ),
            'critical_battery_count': KPIThreshold(
                warning_max=3, critical_max=10
            ),
            'abandonment_rate': KPIThreshold(
                warning_max=0.1, critical_max=0.25
            ),
            'service_level': KPIThreshold(
                warning_min=0.85, critical_min=0.70
            )
        }
    
    # ========================================================================
    # CORE COLLECTION METHOD
    # ========================================================================
    
    def record_step(
        self,
        step: int,
        timestamp: datetime,
        environment: Any,  # Environment object
        step_events: Optional[Dict] = None
    ) -> KPIRecord:
        """
        Record KPIs for current simulation step.
        
        This is the main interface called after each Environment.step()
        """
        step_events = step_events or {}
        highway = environment.highway
        
        # Compute REQUIRED KPIs
        cars_quit_waiting = self._count_quit_waiting(highway, step_events)
        cars_zero_battery = len(step_events.get('strandings', []))
        
        # Update rolling windows
        self._recent_abandonments.append(cars_quit_waiting)
        self._recent_strandings.append(cars_zero_battery)
        
        # Vehicle counts - use current_metrics from environment
        vehicles_on_road = len(highway.vehicles)
        
        # Get step deltas from current_metrics (step-specific counts)
        vehicles_entered = environment.current_metrics.get('vehicles_spawned', 0)
        vehicles_exited = environment.current_metrics.get('vehicles_completed', 0)
        vehicles_stranded = environment.current_metrics.get('vehicles_stranded', 0)
        
        # Update cumulative totals
        self._total_entered += vehicles_entered
        self._total_exited += vehicles_exited
        
        # Compute completion rate (cumulative)
        completion_rate = self._total_exited / max(1, self._total_entered)
        
        # Charging infrastructure state
        highway_state = highway.get_highway_state()
        total_charging = highway_state['charging']
        total_queued = highway_state['queued']
        total_waiting_spots = sum(
            area.waiting_spots for area in highway.charging_areas.values()
        )
        queue_occupancy_rate = total_queued / max(1, total_waiting_spots)
        
        # Wait times
        wait_times = self._collect_wait_times(highway)
        avg_wait = np.mean(wait_times) if wait_times else 0.0
        max_wait = np.max(wait_times) if wait_times else 0.0
        
        # Charging utilization
        total_chargers = sum(len(area.chargers) for area in highway.charging_areas.values())
        charging_utilization = total_charging / max(1, total_chargers)
        
        # Station analysis
        num_stations = len(highway.charging_areas)
        stations_over_capacity = sum(
            1 for area in highway.charging_areas.values()
            if len(area.queue) >= area.waiting_spots and area.waiting_spots > 0
        )
        stations_with_abandonments = self._count_stations_with_abandonments(highway, step_events)
        
        # Battery status across all vehicles
        soc_values = [
            v.battery.current_soc for v in highway.vehicles.values()
        ]
        avg_soc = np.mean(soc_values) * 100 if soc_values else 0.0
        critical_battery = sum(1 for soc in soc_values if soc < 0.10)
        low_battery = sum(1 for soc in soc_values if soc < 0.20)
        
        # Performance metrics
        time_step_hours = environment.config.time_step_minutes / 60
        throughput = vehicles_exited / max(0.001, time_step_hours)
        
        # Trip duration (from completed trips)
        trip_durations = [
            t['duration_minutes'] for t in highway.completed_trips[-50:]
        ]
        avg_trip_duration = np.mean(trip_durations) if trip_durations else 0.0
        
        # Energy (from charging completions this step)
        energy_delivered = sum(
            c.get('energy_kwh', 0) 
            for c in step_events.get('charging_completions', [])
        )
        
        # Efficiency
        abandonment_rate = (
            cars_quit_waiting / max(1, vehicles_entered) 
            if step > 0 else 0.0
        )
        
        # Service level (vehicles served within acceptable wait)
        service_level = 1.0 - abandonment_rate
        
        # Estimated revenue (simplified: $0.40 per kWh + $0.10 per minute connected)
        estimated_revenue = energy_delivered * 0.40 + total_charging * time_step_hours * 6.0
        
        # Customer satisfaction (composite: 1.0 = perfect)
        # Factors: low wait, no abandonment, high SOC at exit, good service level
        satisfaction = (
            service_level * 0.4 +
            (1 - min(1, avg_wait / 30)) * 0.3 +
            (avg_soc / 100) * 0.2 +
            (1 - min(1, critical_battery / 5)) * 0.1
        )
        
        # Create record
        record = KPIRecord(
            step=step,
            timestamp=timestamp,
            simulation_id=self.simulation_id,
            cars_quit_waiting=cars_quit_waiting,
            cars_zero_battery=cars_zero_battery,
            vehicles_on_road=vehicles_on_road,
            vehicles_entered=vehicles_entered,
            vehicles_exited=vehicles_exited,
            vehicles_stranded=vehicles_stranded,
            completion_rate=completion_rate,
            total_charging=total_charging,
            total_queued=total_queued,
            total_waiting_spots=total_waiting_spots,
            queue_occupancy_rate=queue_occupancy_rate,
            avg_wait_time_minutes=avg_wait, # pyright: ignore[reportArgumentType]
            max_wait_time_minutes=max_wait,
            charging_utilization=charging_utilization,
            num_stations=num_stations,
            stations_over_capacity=stations_over_capacity,
            stations_with_abandonments=stations_with_abandonments,
            avg_soc_percent=avg_soc, # pyright: ignore[reportArgumentType]
            critical_battery_count=critical_battery,
            low_battery_count=low_battery,
            throughput_vehicles_per_hour=throughput,
            avg_trip_duration_minutes=avg_trip_duration, # pyright: ignore[reportArgumentType]
            energy_consumption_total_kwh=energy_delivered,
            charging_efficiency=0.9,  # Simplified
            station_turnaround_time=avg_wait + 30,  # Wait + charge time estimate # pyright: ignore[reportArgumentType]
            abandonment_rate=abandonment_rate,
            estimated_revenue=estimated_revenue,
            service_level=service_level,
            customer_satisfaction_score=satisfaction # pyright: ignore[reportArgumentType]
        )
        
        # Store and update
        self.records.append(record)
        self._df_needs_update = True
        
        # Trim history if needed
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history:]
        
        # Check alerts
        self._check_alerts(record)
        
        return record
    
    # ========================================================================
    # KPI COMPUTATION HELPERS
    # ========================================================================
    
    def _count_quit_waiting(self, highway: Any, step_events: Dict) -> int:
        """
        Count vehicles that quit waiting this step.
        Detects abandonments from queue or waiting area.
        """
        # Direct from events
        direct_quits = len(step_events.get('abandonments', []))
        
        # Also check for vehicles that disappeared from queues
        quit_count = direct_quits
        
        for area_id, area in highway.charging_areas.items():
            prev_state = self._station_previous_queues.get(area_id, {})
            prev_queue_ids = set(prev_state.get('queue_ids', []))
            current_queue_ids = {e.vehicle_id for e in area.queue if e.patience_score >= 0}
            
            # Find vehicles that left queue without starting charging
            left_queue = prev_queue_ids - current_queue_ids
            for vid in left_queue:
                # Check if vehicle is now charging (promotion) or gone (abandonment)
                vehicle = highway.vehicles.get(vid)
                if vehicle and vehicle.state.name != 'CHARGING':
                    quit_count += 1
            
            # Update stored state
            self._station_previous_queues[area_id] = {
                'queue_ids': list(current_queue_ids),
                'timestamp': datetime.now()
            }
        
        return quit_count
    
    def _count_stations_with_abandonments(self, highway: Any, step_events: Dict) -> int:
        """Count unique stations that had abandonments this step."""
        abandonment_events = step_events.get('abandonments', [])
        affected_stations = {e.get('area_id') for e in abandonment_events}
        return len(affected_stations)
    
    def _collect_wait_times(self, highway: Any) -> List[float]:
        """Collect current wait times for all queued vehicles."""
        wait_times = []
        current_time = datetime.now()  # Would use sim time
        
        for area in highway.charging_areas.values():
            for entry in area.queue:
                if entry.patience_score >= 0 and hasattr(entry, 'entry_time'):
                    wait = (current_time - entry.entry_time).total_seconds() / 60
                    wait_times.append(wait)
        
        return wait_times
    
    # ========================================================================
    # ALERTING SYSTEM
    # ========================================================================
    
    def _check_alerts(self, record: KPIRecord) -> List[Dict]:
        """Check record against thresholds and generate alerts."""
        triggered = []
        
        for kpi_name, threshold in self.thresholds.items():
            value = getattr(record, kpi_name, None)
            if value is None:
                continue
            
            level = threshold.check(value)
            if level:
                alert = {
                    'step': record.step,
                    'timestamp': record.timestamp,
                    'kpi': kpi_name,
                    'value': value,
                    'level': level.name,
                    'message': f"{kpi_name} = {value:.3f} ({level.name})"
                }
                self.alerts.append(alert)
                triggered.append(alert)
                
                # Trim alerts
                if len(self.alerts) > self.max_alerts:
                    self.alerts = self.alerts[-self.max_alerts:]
                
                # Call callbacks
                for callback in self.alert_callbacks:
                    callback(alert)
        
        return triggered
    
    def set_threshold(self, kpi_name: str, threshold: KPIThreshold) -> None:
        """Set or update threshold for a KPI."""
        self.thresholds[kpi_name] = threshold
    
    # ========================================================================
    # DATAFRAME INTERFACE
    # ========================================================================
    
    def get_dataframe(self, force_update: bool = False) -> pd.DataFrame:
        """Get pandas DataFrame of all records."""
        if self.df is None or self._df_needs_update or force_update:
            if not self.records:
                return pd.DataFrame()
            
            # Convert records to dicts
            data = [asdict(r) for r in self.records]
            self.df = pd.DataFrame(data)
            self._df_needs_update = False
        
        return self.df
    
    def get_recent_df(self, last_n_steps: int = 60) -> pd.DataFrame:
        """Get DataFrame of recent records."""
        df = self.get_dataframe()
        if df.empty:
            return df
        return df.tail(last_n_steps)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics from all records."""
        df = self.get_dataframe()
        if df.empty:
            return {}
        
        # REQUIRED KPIs aggregates
        total_quit = df['cars_quit_waiting'].sum()
        total_stranded = df['cars_zero_battery'].sum()
        
        return {
            'simulation_id': self.simulation_id,
            'total_steps': len(df),
            'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600,
            
            # Required KPIs
            'total_cars_quit_waiting': int(total_quit),
            'total_cars_zero_battery': int(total_stranded),
            'avg_quit_per_hour': total_quit / max(1, len(df) / 60),
            'avg_stranded_per_hour': total_stranded / max(1, len(df) / 60),
            
            # Key performance
            'avg_queue_occupancy': df['queue_occupancy_rate'].mean(),
            'peak_queue_occupancy': df['queue_occupancy_rate'].max(),
            'avg_wait_time': df['avg_wait_time_minutes'].mean(),
            'max_wait_time': df['max_wait_time_minutes'].max(),
            'avg_utilization': df['charging_utilization'].mean(),
            'avg_service_level': df['service_level'].mean(),
            'avg_satisfaction': df['customer_satisfaction_score'].mean(),
            'total_revenue': df['estimated_revenue'].sum(),
            
            # Trends (last hour vs first hour)
            'wait_trend': (
                df['avg_wait_time_minutes'].tail(60).mean() -
                df['avg_wait_time_minutes'].head(60).mean()
                if len(df) >= 120 else 0
            ),
            
            # Stability
            'wait_time_std': df['avg_wait_time_minutes'].std(),
            'utilization_std': df['charging_utilization'].std(),
            
            # Final state
            'final_completion_rate': df['completion_rate'].iloc[-1] if len(df) > 0 else 0,
            'final_abandonment_rate': df['abandonment_rate'].iloc[-1] if len(df) > 0 else 0
        }
    
    # ========================================================================
    # ANALYSIS METHODS
    # ========================================================================
    
    def get_time_series(self, kpi_name: str) -> pd.Series:
        """Get time series for a specific KPI."""
        df = self.get_dataframe()
        if df.empty or kpi_name not in df.columns:
            return pd.Series()
        return df.set_index('timestamp')[kpi_name]
    
    def get_rolling_average(self, kpi_name: str, window: int = 10) -> pd.Series:
        """Get rolling average for a KPI."""
        series = self.get_time_series(kpi_name)
        return series.rolling(window=window, min_periods=1).mean()
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix of numeric KPIs."""
        df = self.get_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].corr()
    
    def detect_anomalies(self, kpi_name: str, threshold_std: float = 3.0) -> pd.DataFrame:
        """Detect anomalous values using z-score."""
        df = self.get_dataframe()
        if kpi_name not in df.columns:
            return pd.DataFrame()
        
        series = df[kpi_name]
        z_scores = np.abs((series - series.mean()) / series.std())
        anomalies = df[z_scores > threshold_std].copy()
        anomalies['z_score'] = z_scores[z_scores > threshold_std]
        return anomalies
    
    def get_efficiency_score(self) -> float:
        """
        Compute overall efficiency score (0-100).
        Balances utilization, service level, and abandonment.
        """
        df = self.get_dataframe()
        if df.empty:
            return 0.0
        
        # Components
        utilization_score = min(100, df['charging_utilization'].mean() * 100)
        service_score = df['service_level'].mean() * 100
        abandonment_penalty = min(50, df['abandonment_rate'].mean() * 200)
        
        efficiency = (utilization_score * 0.3 + 
                     service_score * 0.5 - 
                     abandonment_penalty * 0.2)
        
        return max(0, min(100, efficiency))
    
    # ========================================================================
    # EXPORT AND REPORTING
    # ========================================================================
    
    def export_csv(self, filepath: str) -> None:
        """Export KPI data to CSV."""
        df = self.get_dataframe()
        df.to_csv(filepath, index=False)
    
    def export_json(self, filepath: str) -> None:
        """Export summary statistics to JSON."""
        stats = self.get_summary_stats()
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def generate_report(self) -> str:
        """Generate text report of KPIs."""
        stats = self.get_summary_stats()
        
        report = f"""
{'='*60}
KPI REPORT - Simulation {self.simulation_id}
{'='*60}

DURATION: {stats.get('duration_hours', 0):.1f} hours ({stats.get('total_steps', 0)} steps)

CRITICAL KPIS:
  • Cars quit waiting (total): {stats.get('total_cars_quit_waiting', 0)}
  • Cars with 0% battery (total): {stats.get('total_cars_zero_battery', 0)}
  • Avg quit rate: {stats.get('avg_quit_per_hour', 0):.2f}/hour
  • Avg stranded rate: {stats.get('avg_stranded_per_hour', 0):.2f}/hour

PERFORMANCE:
  • Completion rate: {stats.get('final_completion_rate', 0):.1%}
  • Service level: {stats.get('avg_service_level', 0):.1%}
  • Avg wait time: {stats.get('avg_wait_time', 0):.1f} min
  • Max wait time: {stats.get('max_wait_time', 0):.1f} min
  • Customer satisfaction: {stats.get('avg_satisfaction', 0):.1%}

INFRASTRUCTURE:
  • Avg utilization: {stats.get('avg_utilization', 0):.1%}
  • Avg queue occupancy: {stats.get('avg_queue_occupancy', 0):.1%}
  • Peak queue occupancy: {stats.get('peak_queue_occupancy', 0):.1%}

ECONOMICS:
  • Total revenue: ${stats.get('total_revenue', 0):.2f}
  • Efficiency score: {self.get_efficiency_score():.1f}/100

TRENDS:
  • Wait time trend: {stats.get('wait_trend', 0):+.2f} min
  • Wait stability (std): {stats.get('wait_time_std', 0):.2f}

ALERTS: {len(self.alerts)} total
{'='*60}
"""
        return report
    
    def __repr__(self) -> str:
        return (f"KPITracker({self.simulation_id}, "
                f"records={len(self.records)}, "
                f"alerts={len(self.alerts)})")

