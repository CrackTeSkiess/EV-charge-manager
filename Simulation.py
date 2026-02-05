"""
Simulation Orchestrator Module
Central controller for EV charging station simulation.
"""

from __future__ import annotations

from collections import deque
import uuid
import time
import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path

import pandas as pd
import numpy as np

from Environment import Environment, SimulationConfig
from VehicleGenerator import VehicleGenerator, GeneratorConfig, TemporalDistribution
from KPITracker import KPITracker, KPIRecord
from VisualizationTool import VisualizationTool, ChartConfig


class StopReason(Enum):
    """Reasons for simulation termination."""
    COMPLETED = auto()           # Reached max steps
    EARLY_STOP = auto()          # Early stop condition triggered
    CONVERGENCE = auto()         # Metrics converged
    INSTABILITY = auto()         # System became unstable
    EMPTY_SYSTEM = auto()        # No vehicles remaining
    TIME_LIMIT = auto()          # Wall clock time exceeded
    USER_INTERRUPT = auto()      # Manual stop requested


@dataclass
class EarlyStopCondition:
    """Configuration for early stopping criteria."""
    # Time-based
    max_simulation_steps: Optional[int] = None
    max_wall_clock_seconds: Optional[float] = None
    
    # Stability conditions (consecutive steps)
    max_consecutive_strandings: int = 10  # Stop if > N strandings/step for M steps
    stranding_window: int = 5
    
    max_consecutive_abandonments: int = 20  # Stop if abandonment rate > threshold
    abandonment_rate_threshold: float = 0.3
    abandonment_window: int = 10
    
    # Capacity conditions
    max_queue_occupancy: float = 1.5  # 150% of capacity (severe overflow)
    sustained_queue_occupancy_steps: int = 10
    
    # Performance conditions
    min_service_level: float = 0.5  # Stop if service level drops below 50%
    service_level_window: int = 20
    
    # Convergence conditions (for optimization)
    convergence_patience: int = 100  # Steps without significant improvement
    convergence_tolerance: float = 0.01
    
    # Empty system
    stop_if_empty: bool = False
    empty_grace_period: int = 60  # Steps to wait before stopping on empty


@dataclass
class SimulationParameters:
    """Complete parameter set for simulation configuration."""
    # Highway configuration
    highway_length_km: float = 300.0
    num_charging_areas: int = 3
    charging_area_positions: Optional[List[float]] = None
    chargers_per_area: int = 4
    charger_power_tiers: Dict[str, int] = field(default_factory=lambda: {
        'fast': 3, 'ultra': 1
    })
    waiting_spots_per_area: int = 6
    
    # Traffic configuration
    vehicles_per_hour: float = 60.0
    temporal_distribution: TemporalDistribution = TemporalDistribution.RANDOM_POISSON
    simulation_duration_hours: float = 24.0
    
    # Fleet composition
    vehicle_type_distribution: Dict[str, float] = field(default_factory=lambda: {
        'compact': 0.25, 'midsize': 0.40, 'premium': 0.25, 'truck': 0.10
    })
    driver_behavior_distribution: Dict[str, float] = field(default_factory=lambda: {
        'conservative': 0.25, 'balanced': 0.40, 'aggressive': 0.25, 'range_anxious': 0.10
    })
    
    # Time configuration
    time_step_minutes: float = 1.0
    start_time: datetime = field(default_factory=lambda: datetime(2024, 1, 15, 6, 0))
    
    # Random seed
    random_seed: Optional[int] = None
    
    # Early stopping
    early_stop: EarlyStopCondition = field(default_factory=EarlyStopCondition)
    
    # KPI tracking
    kpi_log_interval: int = 1  # Log every N steps
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for serialization)."""
        d = asdict(self)
        # Convert enums
        d['temporal_distribution'] = self.temporal_distribution.name
        d['start_time'] = self.start_time.isoformat()
        return d


@dataclass
class SimulationResult:
    """Complete result package from simulation run."""
    simulation_id: str
    stop_reason: StopReason
    stop_message: str
    final_step: int
    final_time: datetime
    wall_clock_time_seconds: float
    
    # Data
    kpi_dataframe: pd.DataFrame
    summary_statistics: Dict[str, Any]
    
    # Configuration
    parameters: SimulationParameters
    
    # State at end
    final_environment_state: Dict[str, Any]
    
    # Early stop details (if applicable)
    stop_trigger: Optional[str] = None
    stop_value: Optional[float] = None
    
    def save(self, output_dir: str = "./simulation_results"):
        """Save complete result to directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        base_path = f"{output_dir}/{self.simulation_id}"
        
        # Save KPI dataframe
        self.kpi_dataframe.to_csv(f"{base_path}_kpi.csv", index=False)
        
        # Save summary
        with open(f"{base_path}_summary.json", 'w') as f:
            summary = {
                'simulation_id': self.simulation_id,
                'stop_reason': self.stop_reason.name,
                'stop_message': self.stop_message,
                'final_step': self.final_step,
                'final_time': self.final_time.isoformat(),
                'wall_clock_time_seconds': self.wall_clock_time_seconds,
                'summary_statistics': self.summary_statistics,
                'parameters': self.parameters.to_dict(),
                'stop_trigger': self.stop_trigger,
                'stop_value': self.stop_value
            }
            json.dump(summary, f, indent=2, default=str)
        
        # Save final state
        with open(f"{base_path}_final_state.json", 'w') as f:
            json.dump(self.final_environment_state, f, indent=2, default=str)
        
        return base_path


class Simulation:
    """
    Main simulation orchestrator.
    
    Responsibilities:
    - Initialize Environment, VehicleGenerator, KPITracker
    - Execute time-stepped simulation loop
    - Check early stop conditions
    - Collect and return KPI DataFrame
    - Manage simulation state and logging
    """
    
    def __init__(self, parameters: Optional[SimulationParameters] = None):
        self.params = parameters or SimulationParameters()
        self.id = str(uuid.uuid4())[:8]
        
        # Components (initialized in run())
        self.environment: Optional[Environment] = None
        self.generator: Optional[VehicleGenerator] = None
        self.kpi_tracker: Optional[KPITracker] = None
        
        # State
        self.current_step: int = 0
        self.is_running: bool = False
        self.stop_reason: Optional[StopReason] = None
        self.stop_message: str = ""
        
        # Early stop tracking
        self._recent_strandings: deque = deque(maxlen=self.params.early_stop.stranding_window)
        self._recent_abandonments: deque = deque(maxlen=self.params.early_stop.abandonment_window)
        self._recent_service_levels: deque = deque(maxlen=self.params.early_stop.service_level_window)
        self._recent_queue_occupancy: deque = deque(maxlen=self.params.early_stop.sustained_queue_occupancy_steps)
        self._best_metric: float = float('inf')
        self._steps_without_improvement: int = 0
        self._empty_steps: int = 0
        
        # Timing
        self._start_time: Optional[datetime] = None
        self._step_times: List[float] = []  # For performance monitoring
        
        # Callbacks
        self.on_step: Optional[Callable[[int, Dict], None]] = None
        self.on_stop: Optional[Callable[[StopReason, str], None]] = None
        
        # Results
        self.result: Optional[SimulationResult] = None
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def _initialize(self) -> None:
        """Create and configure all simulation components."""
        # Set random seed if provided
        if self.params.random_seed:
            import random
            random.seed(self.params.random_seed)
            np.random.seed(self.params.random_seed)
        
        # Create environment config
        env_config = SimulationConfig(
            highway_length_km=self.params.highway_length_km,
            num_charging_areas=self.params.num_charging_areas,
            charging_area_positions=self.params.charging_area_positions,
            chargers_per_area=self.params.chargers_per_area,
            waiting_spots_per_area=self.params.waiting_spots_per_area,
            arrival_rate_per_minute=0,  # Controlled by generator
            simulation_start_time=self.params.start_time,
            time_step_minutes=self.params.time_step_minutes,
            random_seed=self.params.random_seed
        )
        
        self.environment = Environment(env_config)
        self.environment.id = f"ENV-{self.id}"
        
        # Create vehicle generator
        gen_config = GeneratorConfig(
            vehicles_per_hour=self.params.vehicles_per_hour,
            distribution_type=self.params.temporal_distribution,
            random_seed=self.params.random_seed
        )
        
        self.generator = VehicleGenerator(self.environment, gen_config)
        
        # Create KPI tracker
        self.kpi_tracker = KPITracker(
            simulation_id=self.id,
            max_history_steps=int(self.params.simulation_duration_hours * 60 * 1.5)
        )
        
        # Set up environment callbacks
        self.environment.on_vehicle_spawn = self._on_vehicle_spawn
        self.environment.on_vehicle_despawn = self._on_vehicle_despawn
        
        self.current_step = 0
        self.is_running = False
    
    def _on_vehicle_spawn(self, vehicle: Any) -> None:
        """Callback for vehicle spawn events."""
        pass  # Could log or modify
    
    def _on_vehicle_despawn(self, vehicle: Any, reason: str) -> None:
        """Callback for vehicle despawn events."""
        pass  # Could log or modify
    
    # ========================================================================
    # MAIN SIMULATION LOOP
    # ========================================================================
    
    def run(self, progress_interval: int = 60, verbose: bool = True) -> SimulationResult:
        """
        Execute complete simulation run.
        
        Args:
            progress_interval: Print progress every N steps
            verbose: Print progress messages
        
        Returns:
            SimulationResult with complete data and KPIs
        """
        # Initialize
        self._initialize()
        self._start_time = datetime.now()
        self.is_running = True
        
        max_steps = int(self.params.simulation_duration_hours * 60 / self.params.time_step_minutes)
        
        if verbose:
            print(f"{'='*70}")
            print(f"SIMULATION START: {self.id}")
            print(f"{'='*70}")
            print(f"Duration: {self.params.simulation_duration_hours}h ({max_steps} steps)")
            print(f"Highway: {self.params.highway_length_km}km, "
                  f"{self.params.num_charging_areas} stations")
            print(f"Traffic: {self.params.vehicles_per_hour}/h, "
                  f"{self.params.temporal_distribution.name}")
            print(f"Early stop: {self._early_stop_summary()}")
            print(f"{'='*70}\n")
        
        try:
            # Main loop
            for step in range(max_steps):
                self.current_step = step
                
                # Check wall clock time limit
                if self._check_time_limit():
                    break
                
                # Execute step
                step_start = time.time()
                should_stop, reason, message = self._execute_step()
                step_duration = time.time() - step_start
                self._step_times.append(step_duration)
                
                if should_stop:
                    self.stop_reason = reason
                    self.stop_message = message
                    break
                
                # Progress reporting
                if verbose and step % progress_interval == 0 and step > 0:
                    self._print_progress(step, max_steps)
                
                # Callback
                if self.on_step:
                    self.on_step(step, self._get_step_info())
            
            else:
                # Completed all steps
                self.stop_reason = StopReason.COMPLETED
                self.stop_message = f"Completed all {max_steps} steps"
        
        except KeyboardInterrupt:
            self.stop_reason = StopReason.USER_INTERRUPT
            self.stop_message = "User interrupted simulation"
        
        finally:
            self.is_running = False
            self._finalize()
        
        # Create result
        wall_time = (datetime.now() - self._start_time).total_seconds()
        
        self.result = SimulationResult(
            simulation_id=self.id,
            stop_reason=self.stop_reason or StopReason.COMPLETED,
            stop_message=self.stop_message,
            final_step=self.current_step,
            final_time=self.environment.current_time if self.environment else datetime.now(),
            wall_clock_time_seconds=wall_time,
            kpi_dataframe=self.kpi_tracker.get_dataframe(), # pyright: ignore[reportOptionalMemberAccess]
            summary_statistics=self.kpi_tracker.get_summary_stats(), # pyright: ignore[reportOptionalMemberAccess]
            parameters=self.params,
            final_environment_state=self._get_final_state(),
            stop_trigger=getattr(self, '_stop_trigger', None),
            stop_value=getattr(self, '_stop_value', None)
        )
        
        if verbose:
            self._print_summary()
        
        if self.on_stop:
            self.on_stop(self.stop_reason, self.stop_message) # pyright: ignore[reportArgumentType]
        
        return self.result
    
    def _execute_step(self) -> Tuple[bool, Optional[StopReason], str]:
        """
        Execute single simulation step.
        Returns (should_stop, reason, message)
        """
        # 1. Generate vehicles
        self.generator.step() # pyright: ignore[reportOptionalMemberAccess]
        
        # 2. Step environment (physics, charging, handoffs)
        events = self.environment.step() # pyright: ignore[reportOptionalMemberAccess]
        
        # 3. Record KPIs
        if self.current_step % self.params.kpi_log_interval == 0:
            record = self.kpi_tracker.record_step( # pyright: ignore[reportOptionalMemberAccess]
                step=self.current_step,
                timestamp=self.environment.current_time, # pyright: ignore[reportOptionalMemberAccess]
                environment=self.environment,
                step_events=events
            )
            
            # Update early stop tracking
            self._update_stop_tracking(record)
            
            # Check early stop conditions
            should_stop, reason, message = self._check_early_stop(record)
            if should_stop:
                return True, reason, message
        
        return False, None, ""
    
    def _update_stop_tracking(self, record: KPIRecord) -> None:
        """Update tracking data for early stop conditions."""
        # Stranding tracking
        self._recent_strandings.append(record.cars_zero_battery)
        
        # Abandonment tracking
        self._recent_abandonments.append(record.abandonment_rate)
        
        # Service level tracking
        self._recent_service_levels.append(record.service_level)
        
        # Queue occupancy tracking
        self._recent_queue_occupancy.append(record.queue_occupancy_rate)
        
        # Empty system tracking
        if record.vehicles_on_road == 0 and record.total_queued == 0:
            self._empty_steps += 1
        else:
            self._empty_steps = 0
        
        # Convergence tracking (using wait time as optimization target)
        current_metric = record.avg_wait_time_minutes
        if current_metric < self._best_metric - self.params.early_stop.convergence_tolerance:
            self._best_metric = current_metric
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1
    
    def _check_early_stop(self, record: KPIRecord) -> Tuple[bool, Optional[StopReason], str]:
        """Check all early stop conditions."""
        es = self.params.early_stop
        
        # 1. Consecutive strandings
        if len(self._recent_strandings) >= es.stranding_window:
            avg_strandings = np.mean(self._recent_strandings)
            if avg_strandings > es.max_consecutive_strandings / es.stranding_window:
                self._stop_trigger = "consecutive_strandings"
                self._stop_value = avg_strandings
                return True, StopReason.INSTABILITY, \
                       f"High stranding rate: {avg_strandings:.2f}/step over {es.stranding_window} steps"
        
        # 2. Abandonment rate
        if len(self._recent_abandonments) >= es.abandonment_window:
            avg_abandonment = np.mean(self._recent_abandonments)
            if avg_abandonment > es.abandonment_rate_threshold:
                self._stop_trigger = "abandonment_rate"
                self._stop_value = avg_abandonment
                return True, StopReason.EARLY_STOP, \
                       f"High abandonment rate: {avg_abandonment:.1%} over {es.abandonment_window} steps"
        
        # 3. Queue capacity
        if len(self._recent_queue_occupancy) >= es.sustained_queue_occupancy_steps:
            avg_occupancy = np.mean(self._recent_queue_occupancy)
            if avg_occupancy > es.max_queue_occupancy:
                self._stop_trigger = "queue_occupancy"
                self._stop_value = avg_occupancy
                return True, StopReason.INSTABILITY, \
                       f"Severe queue overflow: {avg_occupancy:.1%} over {es.sustained_queue_occupancy_steps} steps"
        
        # 4. Service level
        if len(self._recent_service_levels) >= es.service_level_window:
            avg_service = np.mean(self._recent_service_levels)
            if avg_service < es.min_service_level:
                self._stop_trigger = "service_level"
                self._stop_value = avg_service
                return True, StopReason.EARLY_STOP, \
                       f"Low service level: {avg_service:.1%} over {es.service_level_window} steps"
        
        # 5. Convergence (for optimization)
        if es.convergence_patience > 0 and \
           self._steps_without_improvement > es.convergence_patience:
            self._stop_trigger = "convergence"
            self._stop_value = self._best_metric
            return True, StopReason.CONVERGENCE, \
                   f"Converged: no improvement for {es.convergence_patience} steps"
        
        # 6. Empty system
        if es.stop_if_empty and self._empty_steps > es.empty_grace_period:
            self._stop_trigger = "empty_system"
            self._stop_value = self._empty_steps
            return True, StopReason.EMPTY_SYSTEM, \
                   f"System empty for {self._empty_steps} steps"
        
        return False, None, ""
    
    def _check_time_limit(self) -> bool:
        """Check if wall clock time limit exceeded."""
        if self.params.early_stop.max_wall_clock_seconds is None:
            return False
        
        elapsed = (datetime.now() - self._start_time).total_seconds() # pyright: ignore[reportOperatorIssue]
        if elapsed > self.params.early_stop.max_wall_clock_seconds:
            self.stop_reason = StopReason.TIME_LIMIT
            self.stop_message = f"Wall clock time limit: {elapsed:.1f}s > {self.params.early_stop.max_wall_clock_seconds}s"
            return True
        return False
    
    # ========================================================================
    # FINALIZATION
    # ========================================================================
    
    def _finalize(self) -> None:
        """Clean up after simulation completes."""
        # Ensure all remaining vehicles are accounted for
        if self.environment:
            # Force despawn of remaining vehicles
            for vid in list(self.environment.active_vehicle_ids):
                self.environment.despawn_vehicle(vid, "simulation_end")
    
    def _get_final_state(self) -> Dict[str, Any]:
        """Capture final environment state."""
        if not self.environment:
            return {}
        
        return {
            'vehicles_remaining': len(self.environment.active_vehicle_ids),
            'total_spawned': self.environment.stats['total_entered'], # pyright: ignore[reportAttributeAccessIssue]
            'total_completed': len(self.environment.completed_trips),
            'total_stranded': len(self.environment.failed_trips),
            'final_queue_states': {
                aid: {
                    'queue_length': len(area.queue),
                    'charging': sum(1 for c in area.chargers if c.status.name == 'OCCUPIED')
                }
                for aid, area in self.environment.highway.charging_areas.items()
            }
        }
    
    def _get_step_info(self) -> Dict[str, Any]:
        """Get current step information for callbacks."""
        return {
            'step': self.current_step,
            'time': self.environment.current_time if self.environment else None,
            'vehicles': len(self.environment.active_vehicle_ids) if self.environment else 0,
            'kpi': self.kpi_tracker.records[-1] if self.kpi_tracker.records else None # pyright: ignore[reportOptionalMemberAccess]
        }
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def _print_progress(self, step: int, max_steps: int) -> None:
        """Print progress update."""
        if not self.kpi_tracker.records: # pyright: ignore[reportOptionalMemberAccess]
            return
        
        latest = self.kpi_tracker.records[-1] # pyright: ignore[reportOptionalMemberAccess]
        progress = step / max_steps * 100
        
        # Calculate ETA
        avg_step_time = np.mean(self._step_times[-100:]) if self._step_times else 0
        remaining_steps = max_steps - step
        eta_seconds = avg_step_time * remaining_steps
        
        print(f"[{progress:5.1f}%] Step {step}/{max_steps} | "
              f"Time: {self.environment.current_time.strftime('%H:%M')} | " # pyright: ignore[reportOptionalMemberAccess]
              f"Vehicles: {latest.vehicles_on_road:3d} | "
              f"Queue: {latest.total_queued:2d} | "
              f"Wait: {latest.avg_wait_time_minutes:5.1f}min | "
              f"SOC: {latest.avg_soc_percent:5.1f}% | "
              f"ETA: {eta_seconds/60:.1f}min")
    
    def _print_summary(self) -> None:
        """Print final simulation summary."""
        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETE: {self.id}")
        print(f"{'='*70}")
        print(f"Stop reason: {self.stop_reason.name}") # pyright: ignore[reportOptionalMemberAccess]
        print(f"Message: {self.stop_message}")
        print(f"Steps completed: {self.current_step}")
        print(f"Wall clock time: {self.result.wall_clock_time_seconds:.2f}s") # pyright: ignore[reportOptionalMemberAccess]
        print(f"Avg step time: {np.mean(self._step_times)*1000:.2f}ms")
        
        if self.kpi_tracker:
            stats = self.kpi_tracker.get_summary_stats()
            print(f"\nKey Results:")
            print(f"  Total vehicles: {stats.get('total_spawned', 0)}")
            print(f"  Completed trips: {stats.get('completed', 0)}")
            print(f"  Stranded: {stats.get('stranded', 0)}")
            print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
            print(f"  Avg wait time: {stats.get('avg_wait_time', 0):.1f} min")
            print(f"  Total revenue: ${stats.get('total_revenue', 0):.2f}")
        
        print(f"{'='*70}\n")
    
    def _early_stop_summary(self) -> str:
        """Generate summary string of early stop conditions."""
        es = self.params.early_stop
        conditions = []
        
        if es.max_simulation_steps:
            conditions.append(f"max_steps={es.max_simulation_steps}")
        if es.max_wall_clock_seconds:
            conditions.append(f"max_time={es.max_wall_clock_seconds}s")
        if es.max_consecutive_strandings < float('inf'):
            conditions.append(f"strandings>{es.max_consecutive_strandings}")
        if es.abandonment_rate_threshold < 1.0:
            conditions.append(f"abandonment>{es.abandonment_rate_threshold:.0%}")
        if es.max_queue_occupancy < float('inf'):
            conditions.append(f"queue>{es.max_queue_occupancy:.0%}")
        if es.convergence_patience < float('inf'):
            conditions.append(f"convergence@{es.convergence_patience}")
        
        return ", ".join(conditions) if conditions else "disabled"
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def visualize(self, output_dir: str = "./simulation_output") -> List[str]:
        """
        Generate visualizations from completed simulation.
        Must be called after run().
        """
        if self.result is None:
            raise RuntimeError("Must run simulation before visualizing")
        
        viz = VisualizationTool(self.result.kpi_dataframe)
        return viz.generate_full_report(output_dir)
    
    def get_kpi_dataframe(self) -> pd.DataFrame:
        """Get KPI DataFrame (available after run())."""
        if self.result is None:
            raise RuntimeError("Must run simulation first")
        return self.result.kpi_dataframe
    
    def reset(self) -> None:
        """Reset simulation for new run."""
        self.__init__(self.params)
    
    def __repr__(self) -> str:
        status = "running" if self.is_running else "idle"
        if self.result:
            status = f"completed ({self.stop_reason.name})" # pyright: ignore[reportOptionalMemberAccess]
        return f"Simulation({self.id}, {status}, steps={self.current_step})"


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def demo():
    """Demonstrate Simulation functionality."""
    print("=" * 70)
    print("SIMULATION ORCHESTRATOR DEMO")
    print("=" * 70)
    
    # Create parameters
    params = SimulationParameters(
        highway_length_km=200.0,
        num_charging_areas=3,
        charging_area_positions=[50.0, 100.0, 150.0],
        chargers_per_area=4,
        waiting_spots_per_area=6,
        vehicles_per_hour=120.0,  # High traffic to trigger early stop
        temporal_distribution=TemporalDistribution.RUSH_HOUR, # pyright: ignore[reportAttributeAccessIssue]
        simulation_duration_hours=4.0,  # 4 hours
        time_step_minutes=1.0,
        random_seed=42,
        early_stop=EarlyStopCondition(
            max_consecutive_strandings=5,  # Stop if many strandings
            abandonment_rate_threshold=0.25,  # Stop if 25% abandon
            max_queue_occupancy=1.2,  # Stop if 120% capacity sustained
            convergence_patience=200  # Or if converged
        )
    )
    
    # Create and run simulation
    sim = Simulation(params)
    
    # Optional: Set callbacks
    def on_step(step, info):
        if step % 120 == 0:  # Every 2 hours simulated
            print(f"  [Callback] Step {step}: {info['vehicles']} vehicles active")
    
    def on_stop(reason, message):
        print(f"  [Callback] Stopped: {reason.name}")
    
    sim.on_step = on_step
    sim.on_stop = on_stop
    
    # Run simulation
    result = sim.run(progress_interval=30, verbose=True)
    
    # Access results
    print(f"\n{'='*70}")
    print("RESULT ACCESS DEMO")
    print(f"{'='*70}")
    
    print(f"\nResult object: {result}")
    print(f"KPI DataFrame shape: {result.kpi_dataframe.shape}")
    print(f"Columns: {list(result.kpi_dataframe.columns)[:5]}...")
    
    # Show required KPIs
    print(f"\nRequired KPIs (last 5 steps):")
    print(result.kpi_dataframe[['step', 'timestamp', 'cars_quit_waiting', 
                               'cars_zero_battery']].tail())
    
    # Summary statistics
    print(f"\nSummary statistics:")
    for key, value in list(result.summary_statistics.items())[:8]:
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    print(f"\nSaving results...")
    save_path = result.save("/tmp/simulation_demo")
    print(f"Saved to: {save_path}")
    
    # Visualize (optional - requires matplotlib)
    try:
        print(f"\nGenerating visualizations...")
        viz_files = sim.visualize("/tmp/simulation_demo")
        print(f"Generated: {len(viz_files)} visualizations")
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()