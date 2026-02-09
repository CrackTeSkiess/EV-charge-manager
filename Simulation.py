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

from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import numpy as np

from Environment import Environment, SimulationConfig
from Highway import Highway
from VehicleGenerator import VehicleGenerator, GeneratorConfig, TemporalDistribution
from VehicleTracker import VehicleTracker
from KPITracker import KPITracker, KPIRecord
from VisualizationTool import VisualizationTool, ChartConfig
from StationDataCollector import StationDataCollector
from EnergyManager import EnergyManagerConfig


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
    """
    Configuration for early stopping criteria.

    Use preset factory methods for common configurations:
        - EarlyStopCondition.disabled()     - No early stopping
        - EarlyStopCondition.lenient()      - Tolerant thresholds
        - EarlyStopCondition.default()      - Balanced defaults
        - EarlyStopCondition.strict()       - Strict thresholds
        - EarlyStopCondition.quick_test()   - For fast testing

    Or use builder methods to customize:
        config = (EarlyStopCondition.default()
            .with_stranding_limit(5, window=3)
            .with_abandonment_limit(0.2)
            .with_time_limit(seconds=300))
    """
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

    # ========================================================================
    # PRESET FACTORY METHODS
    # ========================================================================

    @classmethod
    def disabled(cls) -> 'EarlyStopCondition':
        """No early stopping - run until simulation completes."""
        return cls(
            max_consecutive_strandings=999999,
            abandonment_rate_threshold=1.0,
            max_queue_occupancy=999.0,
            min_service_level=0.0,
            convergence_patience=0,  # Disabled
            stop_if_empty=False
        )

    @classmethod
    def lenient(cls) -> 'EarlyStopCondition':
        """Lenient thresholds - only stop on severe problems."""
        return cls(
            max_consecutive_strandings=20,
            stranding_window=10,
            abandonment_rate_threshold=0.5,
            abandonment_window=20,
            max_queue_occupancy=2.0,
            sustained_queue_occupancy_steps=20,
            min_service_level=0.3,
            service_level_window=30,
            convergence_patience=0,  # Disabled
            stop_if_empty=False
        )

    @classmethod
    def default(cls) -> 'EarlyStopCondition':
        """Balanced default thresholds."""
        return cls()  # Uses dataclass defaults

    @classmethod
    def strict(cls) -> 'EarlyStopCondition':
        """Strict thresholds - stop early on any problems."""
        return cls(
            max_consecutive_strandings=5,
            stranding_window=3,
            abandonment_rate_threshold=0.15,
            abandonment_window=5,
            max_queue_occupancy=1.2,
            sustained_queue_occupancy_steps=5,
            min_service_level=0.7,
            service_level_window=10,
            convergence_patience=50,
            stop_if_empty=True,
            empty_grace_period=30
        )

    @classmethod
    def quick_test(cls, max_steps: int = 60, max_seconds: float = 30.0) -> 'EarlyStopCondition':
        """For quick testing - time-limited with lenient thresholds."""
        return cls(
            max_simulation_steps=max_steps,
            max_wall_clock_seconds=max_seconds,
            max_consecutive_strandings=999999,
            abandonment_rate_threshold=1.0,
            max_queue_occupancy=999.0,
            min_service_level=0.0,
            convergence_patience=0,
            stop_if_empty=False
        )

    # ========================================================================
    # BUILDER METHODS (return self for chaining)
    # ========================================================================

    def with_stranding_limit(self, max_strandings: int, window: int = 5) -> 'EarlyStopCondition':
        """Set stranding threshold. Stop if avg strandings > max/window."""
        self.max_consecutive_strandings = max_strandings
        self.stranding_window = window
        return self

    def with_abandonment_limit(self, rate: float, window: int = 10) -> 'EarlyStopCondition':
        """Set abandonment rate threshold (0.0-1.0)."""
        self.abandonment_rate_threshold = rate
        self.abandonment_window = window
        return self

    def with_queue_limit(self, occupancy: float, sustained_steps: int = 10) -> 'EarlyStopCondition':
        """Set queue occupancy threshold (1.0 = 100% capacity)."""
        self.max_queue_occupancy = occupancy
        self.sustained_queue_occupancy_steps = sustained_steps
        return self

    def with_service_level(self, min_level: float, window: int = 20) -> 'EarlyStopCondition':
        """Set minimum service level threshold (0.0-1.0)."""
        self.min_service_level = min_level
        self.service_level_window = window
        return self

    def with_convergence(self, patience: int, tolerance: float = 0.01) -> 'EarlyStopCondition':
        """Enable convergence detection. Set patience=0 to disable."""
        self.convergence_patience = patience
        self.convergence_tolerance = tolerance
        return self

    def with_time_limit(self, steps: Optional[int] = None, seconds: Optional[float] = None) -> 'EarlyStopCondition':
        """Set time limits (simulation steps and/or wall clock seconds)."""
        if steps is not None:
            self.max_simulation_steps = steps
        if seconds is not None:
            self.max_wall_clock_seconds = seconds
        return self

    def with_empty_stop(self, enabled: bool = True, grace_period: int = 60) -> 'EarlyStopCondition':
        """Stop when system becomes empty."""
        self.stop_if_empty = enabled
        self.empty_grace_period = grace_period
        return self

    def disable_stranding_check(self) -> 'EarlyStopCondition':
        """Disable stranding-based early stop."""
        self.max_consecutive_strandings = 999999
        return self

    def disable_abandonment_check(self) -> 'EarlyStopCondition':
        """Disable abandonment-based early stop."""
        self.abandonment_rate_threshold = 1.0
        return self

    def disable_queue_check(self) -> 'EarlyStopCondition':
        """Disable queue occupancy-based early stop."""
        self.max_queue_occupancy = 999.0
        return self

    def disable_service_check(self) -> 'EarlyStopCondition':
        """Disable service level-based early stop."""
        self.min_service_level = 0.0
        return self

    def disable_convergence(self) -> 'EarlyStopCondition':
        """Disable convergence-based early stop."""
        self.convergence_patience = 0
        return self

    def summary(self) -> str:
        """Get human-readable summary of conditions."""
        lines = ["Early Stop Conditions:"]
        if self.max_simulation_steps:
            lines.append(f"  - Max steps: {self.max_simulation_steps}")
        if self.max_wall_clock_seconds:
            lines.append(f"  - Max wall time: {self.max_wall_clock_seconds}s")
        if self.max_consecutive_strandings < 999999:
            lines.append(f"  - Strandings: >{self.max_consecutive_strandings} over {self.stranding_window} steps")
        if self.abandonment_rate_threshold < 1.0:
            lines.append(f"  - Abandonment: >{self.abandonment_rate_threshold:.0%} over {self.abandonment_window} steps")
        if self.max_queue_occupancy < 999:
            lines.append(f"  - Queue: >{self.max_queue_occupancy:.0%} over {self.sustained_queue_occupancy_steps} steps")
        if self.min_service_level > 0:
            lines.append(f"  - Service level: <{self.min_service_level:.0%} over {self.service_level_window} steps")
        if self.convergence_patience > 0:
            lines.append(f"  - Convergence: {self.convergence_patience} steps patience")
        if self.stop_if_empty:
            lines.append(f"  - Empty system: after {self.empty_grace_period} steps")
        if len(lines) == 1:
            lines.append("  (all checks disabled)")
        return "\n".join(lines)


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
    
    # Charger tracker
    enable_station_tracking: bool = False
    
    # vehicle tracker
    enable_vehicle_tracking: bool = False

    # Queue overflow (allow vehicles to join full queues in emergencies)
    allow_queue_overflow: bool = True

    # Energy management (optional — None means unlimited power for all areas)
    # If provided, list length must match num_charging_areas (one config per area)
    energy_manager_configs: Optional[List[EnergyManagerConfig]] = None

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
    station_data : Dict[str, pd.DataFrame]
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
    - Initialize Environment, VehicleGenerator, VehicleTracker, KPITracker
    - Wire all components together with shared VehicleTracker
    - Execute time-stepped simulation loop
    - Check early stop conditions
    - Collect and return KPI DataFrame
    - Manage simulation state and logging
    """

    def __init__(self, parameters: Optional[SimulationParameters] = None, highway: Optional[Highway] = None):
        self.params = parameters or SimulationParameters()
        self.id = str(uuid.uuid4())[:8]
        self.highway = highway

        # Components (initialized in run())
        self.vehicle_tracker: Optional[VehicleTracker] = None
        self.environment: Optional[Environment] = None
        self.generator: Optional[VehicleGenerator] = None
        self.kpi_tracker: Optional[KPITracker] = None
        self.collector: Optional[StationDataCollector] = None

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

        # Create centralized vehicle tracker FIRST
        self.vehicle_tracker = VehicleTracker()
        
        # Create station data collector if enabled
        if self.params.enable_station_tracking:
            self.collector = StationDataCollector()

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
            random_seed=self.params.random_seed,
            track_vehicle_history=self.params.enable_vehicle_tracking,
            allow_queue_overflow=self.params.allow_queue_overflow,
            energy_manager_configs=self.params.energy_manager_configs
        )

        # Create environment with tracker
        self.environment = Environment(env_config, vehicle_tracker=self.vehicle_tracker)
        self.environment.id = f"ENV-{self.id}"

        # Ensure highway also has the tracker (should be set via Environment constructor)
        if self.environment.highway.tracker is None:
            self.environment.highway.set_tracker(self.vehicle_tracker)
            
        if self.highway:
            self.environment.setHighway(self.highway)

        # Create vehicle generator
        gen_config = GeneratorConfig(
            vehicles_per_hour=self.params.vehicles_per_hour,
            distribution_type=self.params.temporal_distribution,
            random_seed=self.params.random_seed
        )

        self.generator = VehicleGenerator(self.environment, gen_config)

        # Create KPI tracker with shared vehicle tracker
        self.kpi_tracker = KPITracker(
            simulation_id=self.id,
            max_history_steps=int(self.params.simulation_duration_hours * 60 * 1.5),
            vehicle_tracker=self.vehicle_tracker
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
            print(f"Vehicle Tracker: ENABLED (centralized tracking)")
            print(f"Vehicle History: {'ENABLED' if self.params.enable_vehicle_tracking else 'DISABLED'}")
            print(f"Station Tracker: {'ENABLED' if self.collector else 'DISABLED'}")
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
                    
                if self.collector is not None:
                    self.collector.record_all_stations(self.environment.highway, self.environment.current_time)  # pyright: ignore[reportOptionalMemberAccess]

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
            kpi_dataframe=self.kpi_tracker.get_dataframe(),  # pyright: ignore[reportOptionalMemberAccess]
            station_data=self.collector.get_station_dataframes() if self.collector else None, # pyright: ignore[reportArgumentType]
            summary_statistics=self.kpi_tracker.get_summary_stats(),  # pyright: ignore[reportOptionalMemberAccess]
            parameters=self.params,
            final_environment_state=self._get_final_state(),
            stop_trigger=getattr(self, '_stop_trigger', None),
            stop_value=getattr(self, '_stop_value', None)
        )

        if verbose:
            self._print_summary()

        if self.on_stop:
            self.on_stop(self.stop_reason, self.stop_message)  # pyright: ignore[reportArgumentType]

        return self.result  

    def _execute_step(self) -> Tuple[bool, Optional[StopReason], str]:
        """
        Execute single simulation step.
        Returns (should_stop, reason, message)
        """
        # 0. Reset step metrics BEFORE anything spawns vehicles
        # This must happen before generator.step() which calls spawn_vehicle()
        self.environment.step_metrics = self.environment._reset_step_metrics()  # pyright: ignore[reportOptionalMemberAccess, reportPrivateUsage]
        if self.vehicle_tracker:
            self.vehicle_tracker.begin_step()

        # 1. Generate vehicles (spawns will be tracked in step_metrics)
        self.generator.step()  # pyright: ignore[reportOptionalMemberAccess]

        # 2. Step environment (physics, charging, handoffs) - skip_reset=True
        events = self.environment.step(skip_reset=True)  # pyright: ignore[reportOptionalMemberAccess]

        # 3. Record KPIs
        if self.current_step % self.params.kpi_log_interval == 0:
            record = self.kpi_tracker.record_step(  # pyright: ignore[reportOptionalMemberAccess]
                step=self.current_step,
                timestamp=self.environment.current_time,  # pyright: ignore[reportOptionalMemberAccess]
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
        # Stranding tracking - NOW USES ACTUAL DATA from tracker
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

        elapsed = (datetime.now() - self._start_time).total_seconds()  # pyright: ignore[reportOperatorIssue]
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

        # Validate tracker consistency if available
        if self.vehicle_tracker:
            issues = self.vehicle_tracker.validate_consistency()
            if issues:
                print(f"WARNING: VehicleTracker consistency issues: {issues}")
                
    def generate_stranding_report(self, output_dir: str = "./simulation_output") -> Dict[str, Any]:
        """
        Generate comprehensive stranding analysis report with visualizations.
        
        This method:
        1. Collects stranding data from VehicleTracker
        2. Generates multiple visualizations
        3. Saves data to JSON for further analysis
        4. Returns structured data for programmatic use
        
        Args:
            output_dir: Directory to save outputs (created if doesn't exist)
            
        Returns:
            Dictionary containing stranding analysis data and file paths
        """
        import os
        import json
        from pathlib import Path
        
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check prerequisites
        if not self.vehicle_tracker:
            raise RuntimeError("No VehicleTracker available. Run simulation first.")
        
        if not self.environment:
            raise RuntimeError("No Environment available. Run simulation first.")
        
        print(f"\n{'='*60}")
        print("GENERATING STRANDING REPORT")
        print(f"{'='*60}")
        
        # Step 1: Collect stranding data from VehicleTracker
        print("Collecting stranding data from VehicleTracker...")
        stranding_data = self.vehicle_tracker.get_detailed_stranding_data()
        
        total_strandings = stranding_data['total_strandings']
        print(f"Found {total_strandings} stranded vehicles")
        
        if total_strandings == 0:
            print("No strandings to report!")
            return {
                'total_strandings': 0,
                'output_files': [],
                'analysis': {}
            }
        
        # Step 2: Get station positions from highway
        print("Extracting charging station positions...")
        highway = self.environment.highway
        
        highway.plot_stranded_vehicle_trajectories(max_vehicles=10)
        highway.plot_stranding_summary_chart()
        station_positions = []
        station_names = []
        
        for area_id, area in highway.charging_areas.items():
            station_positions.append(area.location_km)
            station_names.append(area.name)
        
        print(f"Found {len(station_positions)} stations at positions: {station_positions}")
        
        # Step 3: Generate visualizations
        print("Generating visualizations...")
        
        # Create VisualizationTool instance
        viz = VisualizationTool(self.kpi_tracker.get_dataframe() if self.kpi_tracker else None)
        
        # Generate main stranding analysis plot
        viz.plot_stranding_analysis(
            stranding_data=stranding_data,
            highway_length_km=self.params.highway_length_km,
            station_positions=station_positions,
            station_names=station_names,
            save_path=str(output_path / f"stranding_report_{self.id}"),
            show_plot=False
        )
        
        # Generate additional KPI context if KPI data available
        if self.kpi_tracker and self.kpi_tracker.records:
            # Stranding over time plot
            self._plot_stranding_timeline(output_path, viz)
        
        # Step 4: Save raw data to JSON
        print("Saving data to JSON...")
        json_path = output_path / f"stranding_data_{self.id}.json"
        
        # Prepare serializable data
        export_data = {
            'simulation_id': self.id,
            'parameters': {
                'highway_length_km': self.params.highway_length_km,
                'num_stations': len(station_positions),
                'station_positions': station_positions,
                'station_names': station_names,
                'chargers_per_area': self.params.chargers_per_area,
                'waiting_spots_per_area': self.params.waiting_spots_per_area,
                'vehicles_per_hour': self.params.vehicles_per_hour,
                'simulation_duration_hours': self.params.simulation_duration_hours
            },
            'stranding_summary': {
                'total_strandings': total_strandings,
                'stranding_rate': total_strandings / max(1, self.vehicle_tracker.total_spawned),
                'strandings_per_100km': total_strandings / self.params.highway_length_km * 100
            },
            'detailed_data': stranding_data,
            'analysis': stranding_data.get('analysis', {}),
            'recommendations': self._generate_infrastructure_recommendations(
                stranding_data, station_positions
            )
        }
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Step 5: Generate text report
        print("Generating text report...")
        report_path = output_path / f"stranding_report_{self.id}.txt"
        self._write_text_report(report_path, export_data)
        
        # Collect output files
        output_files = [
            output_path / f"stranding_report_{self.id}_stranding_analysis.png",
            json_path,
            report_path
        ]
        
        print(f"\n{'='*60}")
        print("STRANDING REPORT COMPLETE")
        print(f"{'='*60}")
        print(f"Total strandings: {total_strandings}")
        print(f"Stranding rate: {export_data['stranding_summary']['stranding_rate']:.2%}")
        print(f"\nOutput files:")
        for f in output_files:
            print(f"  - {f}")
        print(f"{'='*60}\n")
        
        return {
            'total_strandings': total_strandings,
            'stranding_rate': export_data['stranding_summary']['stranding_rate'],
            'output_files': [str(f) for f in output_files],
            'analysis': export_data['analysis'],
            'recommendations': export_data['recommendations']
        }

    def _plot_stranding_timeline(self, output_path: Path, viz: VisualizationTool):
        """Generate timeline showing strandings over simulation duration."""
        if not self.kpi_tracker:
            return
        
        df = self.kpi_tracker.get_dataframe()
        if df.empty or 'cars_zero_battery' not in df.columns:
            return
        
        fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
        fig.patch.set_facecolor(viz.colors['background'])
        ax.set_facecolor(viz.colors['background'])
        
        time_index = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else df.index
        
        # Plot strandings as bars
        strandings = df['cars_zero_battery']
        ax.bar(time_index, strandings, color=viz.colors['danger'], 
            alpha=0.7, width=0.02, label='Strandings per step')
        
        # Cumulative line
        ax_twin = ax.twinx()
        cumulative = strandings.cumsum()
        ax_twin.plot(time_index, cumulative, color=viz.colors['danger'], 
                    linewidth=3, label='Cumulative')
        ax_twin.fill_between(time_index, cumulative, alpha=0.2, color=viz.colors['danger'])
        
        # Formatting
        ax.set_xlabel('Time', fontsize=11, color=viz.colors['text'])
        ax.set_ylabel('Strandings per Step', fontsize=11, color=viz.colors['text'])
        ax_twin.set_ylabel('Cumulative Strandings', fontsize=11, color=viz.colors['danger'])
        ax.set_title('Stranding Events Timeline', fontsize=14, fontweight='bold',
                    color=viz.colors['text'])
        
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / f"stranding_timeline_{self.id}.png",
                dpi=100, bbox_inches='tight', facecolor=viz.colors['background'])
        plt.close()

    def _generate_infrastructure_recommendations(self, stranding_data: Dict,
                                                station_positions: List[float]) -> List[Dict]:
        """Generate recommendations based on stranding patterns."""
        recommendations = []
        
        positions = stranding_data.get('positions_km', [])
        if not positions or len(station_positions) < 2:
            return recommendations
        
        analysis = stranding_data.get('analysis', {})
        
        # Check 1: Large gaps between stations
        sorted_stations = sorted(station_positions)
        for i in range(len(sorted_stations) - 1):
            gap = sorted_stations[i+1] - sorted_stations[i]
            if gap > 100:
                # Count strandings in this gap
                mid_point = (sorted_stations[i] + sorted_stations[i+1]) / 2
                strandings_in_gap = sum(1 for p in positions 
                                    if sorted_stations[i] < p < sorted_stations[i+1])
                
                if strandings_in_gap > 0:
                    recommendations.append({
                        'priority': 'HIGH' if strandings_in_gap > 2 else 'MEDIUM',
                        'type': 'station_gap',
                        'message': f'Large gap of {gap:.0f}km between stations at '
                                f'{sorted_stations[i]:.0f}km and {sorted_stations[i+1]:.0f}km',
                        'details': f'{strandings_in_gap} strandings in this gap. '
                                f'Consider adding intermediate station around {mid_point:.0f}km',
                        'estimated_impact': f'Could prevent {strandings_in_gap} strandings'
                    })
        
        # Check 2: Clustered strandings
        if 'clustering' in analysis:
            clusters = analysis['clustering']
            for cluster in clusters[:3]:  # Top 3 clusters
                if cluster['count'] >= 2:
                    recommendations.append({
                        'priority': 'HIGH' if cluster['count'] >= 3 else 'MEDIUM',
                        'type': 'cluster_hotspot',
                        'message': f"Stranding hotspot at {cluster['center_km']:.0f}km",
                        'details': f"{cluster['count']} vehicles stranded in "
                                f"{cluster['start_km']:.0f}-{cluster['end_km']:.0f}km range. "
                                f"Check if station at this location is functioning properly "
                                f"or if additional capacity is needed.",
                        'estimated_impact': f'Investigate infrastructure at this location'
                    })
        
        # Check 3: Distance analysis
        if 'mean_position_km' in analysis:
            mean_dist = analysis.get('mean_distance_to_station', 
                                self._calculate_mean_distance(positions, station_positions))
            if mean_dist > 20:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'type': 'coverage_issue',
                    'message': f'Vehicles stranding far from stations (avg {mean_dist:.1f}km)',
                    'details': 'Consider increasing station density or adding '
                            'mobile charging units for emergency coverage',
                    'estimated_impact': 'Improve emergency response capability'
                })
        
        # Check 4: Driver behavior
        driver_types = stranding_data.get('driver_types', [])
        if driver_types:
            type_counts = {}
            for dt in driver_types:
                type_counts[dt] = type_counts.get(dt, 0) + 1
            
            # Check if aggressive drivers are over-represented
            if type_counts.get('aggressive', 0) > len(driver_types) * 0.4:
                recommendations.append({
                    'priority': 'LOW',
                    'type': 'behavioral',
                    'message': 'High proportion of aggressive drivers among strandings',
                    'details': 'Consider driver education or dynamic pricing to '
                            'encourage more conservative driving behavior',
                    'estimated_impact': 'Reduce risk-taking behavior'
                })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return recommendations

    def _calculate_mean_distance(self, positions: List[float], stations: List[float]) -> float:
        """Calculate mean distance from strandings to nearest station."""
        if not positions or not stations:
            return 0.0
        
        distances = []
        for pos in positions:
            min_dist = min(abs(pos - s) for s in stations)
            distances.append(min_dist)
        
        return float(np.mean(distances))

    def _write_text_report(self, report_path: Path, data: Dict):
        """Write human-readable text report."""
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("VEHICLE STRANDING ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Simulation ID: {data['simulation_id']}\n")
            f.write(f"Report Generated: {datetime.now().isoformat()}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-"*70 + "\n")
            summary = data['stranding_summary']
            f.write(f"Total Vehicles Spawned: {self.vehicle_tracker.total_spawned if self.vehicle_tracker else 'N/A'}\n")
            f.write(f"Total Vehicles Stranded: {summary['total_strandings']}\n")
            f.write(f"Stranding Rate: {summary['stranding_rate']:.2%}\n")
            f.write(f"Strandings per 100km: {summary['strandings_per_100km']:.2f}\n\n")
            
            # Infrastructure
            f.write("INFRASTRUCTURE CONFIGURATION\n")
            f.write("-"*70 + "\n")
            params = data['parameters']
            f.write(f"Highway Length: {params['highway_length_km']:.0f} km\n")
            f.write(f"Number of Stations: {params['num_stations']}\n")
            f.write(f"Station Positions: {params['station_positions']}\n")
            f.write(f"Chargers per Station: {params['chargers_per_area']}\n")
            f.write(f"Waiting Spots per Station: {params['waiting_spots_per_area']}\n\n")
            
            # Analysis
            f.write("STRANDING ANALYSIS\n")
            f.write("-"*70 + "\n")
            analysis = data['analysis']
            if analysis:
                f.write(f"Mean Stranding Position: {analysis.get('mean_position_km', 'N/A'):.1f} km\n")
                f.write(f"Position Std Dev: {analysis.get('std_position_km', 'N/A'):.1f} km\n")
                f.write(f"Range: {analysis.get('min_position_km', 'N/A'):.1f} - "
                    f"{analysis.get('max_position_km', 'N/A'):.1f} km\n")
                f.write(f"Most Common Driver Type: {analysis.get('most_common_driver_type', 'N/A')}\n")
                f.write(f"Average Initial SOC: {analysis.get('avg_initial_soc', 'N/A'):.2%}\n\n")
            else:
                f.write("No detailed analysis available.\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*70 + "\n")
            recs = data['recommendations']
            if recs:
                for i, rec in enumerate(recs, 1):
                    f.write(f"\n{i}. [{rec['priority']}] {rec['type'].upper()}\n")
                    f.write(f"   {rec['message']}\n")
                    f.write(f"   Details: {rec['details']}\n")
                    f.write(f"   Expected Impact: {rec['estimated_impact']}\n")
            else:
                f.write("No specific recommendations. System performing adequately.\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

    def _get_final_state(self) -> Dict[str, Any]:
        """Capture final environment state."""
        if not self.environment:
            return {}

        # Get tracker summary if available
        tracker_summary = {}
        if self.vehicle_tracker:
            tracker_summary = self.vehicle_tracker.get_summary()

        return {
            'vehicles_remaining': len(self.environment.active_vehicle_ids),
            'total_spawned': self.environment.highway.stats['total_entered'],
            'total_completed': len(self.environment.completed_trips),
            'total_stranded': len(self.environment.failed_trips),
            'final_queue_states': {
                aid: {
                    'queue_length': len(area.queue),
                    'charging': sum(1 for c in area.chargers if c.status.name == 'OCCUPIED')
                }
                for aid, area in self.environment.highway.charging_areas.items()
            },
            'tracker_summary': tracker_summary
        }

    def _get_step_info(self) -> Dict[str, Any]:
        """Get current step information for callbacks."""
        return {
            'step': self.current_step,
            'time': self.environment.current_time if self.environment else None,
            'vehicles': len(self.environment.active_vehicle_ids) if self.environment else 0,
            'kpi': self.kpi_tracker.records[-1] if self.kpi_tracker and self.kpi_tracker.records else None
        }

    # ========================================================================
    # REPORTING
    # ========================================================================

    def _print_progress(self, step: int, max_steps: int) -> None:
        """Print progress update."""
        if not self.kpi_tracker or not self.kpi_tracker.records:
            return

        latest = self.kpi_tracker.records[-1]
        progress = step / max_steps * 100

        # Calculate ETA
        avg_step_time = np.mean(self._step_times[-100:]) if self._step_times else 0
        remaining_steps = max_steps - step
        eta_seconds = avg_step_time * remaining_steps

        # Get tracker stats
        tracker_info = ""
        if self.vehicle_tracker:
            tracker_info = f" | Stranded: {self.vehicle_tracker.total_stranded}"

        print(f"[{progress:5.1f}%] Step {step}/{max_steps} | "
              f"Time: {self.environment.current_time.strftime('%H:%M')} | "  # pyright: ignore[reportOptionalMemberAccess]
              f"Vehicles: {latest.vehicles_on_road:3d} | "
              f"Queue: {latest.total_queued:2d} | "
              f"Wait: {latest.avg_wait_time_minutes:5.1f}min | "
              f"SOC: {latest.avg_soc_percent:5.1f}%{tracker_info} | "
              f"ETA: {eta_seconds/60:.1f}min")

    def _print_summary(self) -> None:
        """Print final simulation summary."""
        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETE: {self.id}")
        print(f"{'='*70}")
        print(f"Stop reason: {self.stop_reason.name}")  # pyright: ignore[reportOptionalMemberAccess]
        print(f"Message: {self.stop_message}")
        print(f"Steps completed: {self.current_step}")
        print(f"Wall clock time: {self.result.wall_clock_time_seconds:.2f}s")  # pyright: ignore[reportOptionalMemberAccess]
        print(f"Avg step time: {np.mean(self._step_times)*1000:.2f}ms")

        if self.kpi_tracker:
            stats = self.kpi_tracker.get_summary_stats()
            print(f"\nKey Results:")
            print(f"  Total vehicles: {stats.get('total_spawned', 0)}")
            print(f"  Completed trips: {stats.get('completed', 0)}")
            print(f"  Stranded: {stats.get('total_cars_zero_battery', 0)}")
            print(f"  Abandonments: {stats.get('total_cars_quit_waiting', 0)}")
            print(f"  Avg wait time: {stats.get('avg_wait_time', 0):.1f} min")
            print(f"  Total revenue: ${stats.get('total_revenue', 0):.2f}")

        # Print tracker summary
        if self.vehicle_tracker:
            summary = self.vehicle_tracker.get_summary()
            print(f"\nVehicle Tracker Summary:")
            print(f"  Total spawned: {summary['total_spawned']}")
            print(f"  Total completed: {summary['total_completed']}")
            print(f"  Total stranded: {summary['total_stranded']}")
            print(f"  Total abandonments: {summary['total_abandonments']}")
            print(f"  Completion rate: {summary['completion_rate']:.1%}")
            print(f"  Stranding rate: {summary['stranding_rate']:.1%}")

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

    def visualize(self, output_dir: str = "./simulation_output") -> List[str]: # pyright: ignore[reportArgumentType]
        """
        Generate visualizations from completed simulation.
        Must be called after run().
        """

        if self.result is None:
            raise RuntimeError("Must run simulation before visualizing")

        viz = VisualizationTool(self.result.kpi_dataframe)
        return viz.generate_full_report(output_dir, self.collector.get_station_dataframes() if self.collector else None)  # pyright: ignore[reportArgumentType, reportOptionalMemberAccess]

    def get_kpi_dataframe(self) -> pd.DataFrame:
        """Get KPI DataFrame (available after run())."""
        if self.result is None:
            raise RuntimeError("Must run simulation first")
        return self.result.kpi_dataframe

    def get_vehicle_tracker(self) -> Optional[VehicleTracker]:
        """Get the vehicle tracker instance."""
        return self.vehicle_tracker

    def reset(self) -> None:
        """Reset simulation for new run."""
        self.__init__(self.params)

    def __repr__(self) -> str:
        status = "running" if self.is_running else "idle"
        if self.result:
            status = f"completed ({self.stop_reason.name})"  # pyright: ignore[reportOptionalMemberAccess]
        return f"Simulation({self.id}, {status}, steps={self.current_step})"