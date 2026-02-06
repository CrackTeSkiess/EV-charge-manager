"""
VehicleTracker Module
Centralized vehicle state management and lifecycle tracking.
Single source of truth for all vehicle states in the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Callable, Any, Tuple
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict
import numpy as np


class TrackedState(Enum):
    """Vehicle states tracked by the system."""
    DRIVING = auto()       # On highway, moving
    APPROACHING = auto()   # Near station, deciding
    QUEUED = auto()        # Waiting in queue
    CHARGING = auto()      # At charger
    EXITING = auto()       # Leaving station, returning to highway
    COMPLETED = auto()     # Successfully exited highway
    STRANDED = auto()      # Ran out of battery


@dataclass
class VehicleRecord:
    """Complete record for a single vehicle."""
    vehicle_id: str
    spawn_time: datetime
    initial_soc: float
    battery_capacity_kwh: float
    driver_type: str

    # Current state
    current_state: TrackedState = TrackedState.DRIVING
    position_km: float = 0.0
    current_soc: float = 0.0

    # Station interaction
    current_station_id: Optional[str] = None
    queue_entry_time: Optional[datetime] = None
    charging_start_time: Optional[datetime] = None

    # Trip metrics
    total_wait_time_minutes: float = 0.0
    total_charge_time_minutes: float = 0.0
    charging_stops: int = 0
    energy_received_kwh: float = 0.0

    # Exit info
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    final_position_km: float = 0.0
    final_soc: float = 0.0

    # State history
    state_history: List[Tuple[datetime, TrackedState, str]] = field(default_factory=list)


@dataclass
class StepMetrics:
    """Metrics for a single simulation step."""
    vehicles_spawned: int = 0
    vehicles_completed: int = 0
    vehicles_stranded: int = 0
    vehicles_started_queuing: int = 0
    vehicles_started_charging: int = 0
    vehicles_finished_charging: int = 0
    vehicles_abandoned_queue: int = 0

    # Lists for detailed tracking
    spawned_ids: List[str] = field(default_factory=list)
    completed_ids: List[str] = field(default_factory=list)
    stranded_ids: List[str] = field(default_factory=list)
    abandoned_ids: List[str] = field(default_factory=list)


class VehicleTracker:
    """
    Centralized vehicle tracking system.

    Single source of truth for:
    - Which vehicles are active and their states
    - State transitions (spawn, queue, charge, exit, strand)
    - Per-step metrics collection
    - Complete vehicle history

    All other components should query this tracker rather than
    maintaining their own vehicle lists.
    """

    def __init__(self):
        # Active vehicles by state
        self._vehicles_by_state: Dict[TrackedState, Set[str]] = {
            state: set() for state in TrackedState
        }

        # Master record for all vehicles (including completed/stranded)
        self._vehicle_records: Dict[str, VehicleRecord] = {}

        # Quick lookup for active vehicles
        self._active_vehicle_ids: Set[str] = set()

        # Station assignments
        self._vehicles_at_station: Dict[str, str] = {}  # vehicle_id -> station_id
        self._vehicles_in_queue: Dict[str, Tuple[str, datetime]] = {}  # vehicle_id -> (station_id, entry_time)

        # Step metrics (reset each step)
        self._current_step_metrics: StepMetrics = StepMetrics()

        # Cumulative counters
        self.total_spawned: int = 0
        self.total_completed: int = 0
        self.total_stranded: int = 0
        self.total_abandonments: int = 0

        # Event callbacks
        self.on_state_change: Optional[Callable[[str, TrackedState, TrackedState, str], None]] = None
        self.on_spawn: Optional[Callable[[str, VehicleRecord], None]] = None
        self.on_exit: Optional[Callable[[str, str, VehicleRecord], None]] = None

    # ========================================================================
    # STEP MANAGEMENT
    # ========================================================================

    def begin_step(self) -> None:
        """Reset step metrics at the start of each simulation step."""
        self._current_step_metrics = StepMetrics()

    def get_step_metrics(self) -> StepMetrics:
        """Get metrics for the current step."""
        return self._current_step_metrics

    # ========================================================================
    # SPAWN / REGISTER
    # ========================================================================

    def register_vehicle(
        self,
        vehicle_id: str,
        spawn_time: datetime,
        initial_soc: float,
        battery_capacity_kwh: float,
        driver_type: str,
        position_km: float = 0.0
    ) -> VehicleRecord:
        """
        Register a new vehicle entering the simulation.
        Called when a vehicle spawns at highway entry.
        """
        if vehicle_id in self._vehicle_records:
            raise ValueError(f"Vehicle {vehicle_id} already registered")

        record = VehicleRecord(
            vehicle_id=vehicle_id,
            spawn_time=spawn_time,
            initial_soc=initial_soc,
            battery_capacity_kwh=battery_capacity_kwh,
            driver_type=driver_type,
            current_state=TrackedState.DRIVING,
            position_km=position_km,
            current_soc=initial_soc
        )

        record.state_history.append((spawn_time, TrackedState.DRIVING, "spawned"))

        self._vehicle_records[vehicle_id] = record
        self._active_vehicle_ids.add(vehicle_id)
        self._vehicles_by_state[TrackedState.DRIVING].add(vehicle_id)

        # Update metrics
        self.total_spawned += 1
        self._current_step_metrics.vehicles_spawned += 1
        self._current_step_metrics.spawned_ids.append(vehicle_id)

        # Callback
        if self.on_spawn:
            self.on_spawn(vehicle_id, record)

        return record

    # ========================================================================
    # STATE TRANSITIONS
    # ========================================================================

    def transition_to_approaching(
        self,
        vehicle_id: str,
        timestamp: datetime,
        station_id: str,
        reason: str = ""
    ) -> bool:
        """Vehicle is approaching a charging station."""
        return self._transition_state(
            vehicle_id, TrackedState.APPROACHING, timestamp,
            reason or f"approaching {station_id}"
        )

    def transition_to_queued(
        self,
        vehicle_id: str,
        timestamp: datetime,
        station_id: str,
        reason: str = ""
    ) -> bool:
        """Vehicle enters a charging station queue."""
        success = self._transition_state(
            vehicle_id, TrackedState.QUEUED, timestamp,
            reason or f"queued at {station_id}"
        )

        if success:
            record = self._vehicle_records[vehicle_id]
            record.current_station_id = station_id
            record.queue_entry_time = timestamp

            self._vehicles_in_queue[vehicle_id] = (station_id, timestamp)
            self._current_step_metrics.vehicles_started_queuing += 1

        return success

    def transition_to_charging(
        self,
        vehicle_id: str,
        timestamp: datetime,
        station_id: str,
        reason: str = ""
    ) -> bool:
        """Vehicle starts charging (either from queue or immediate)."""
        # Update wait time if coming from queue
        record = self._vehicle_records.get(vehicle_id)
        if record and record.queue_entry_time:
            wait_minutes = (timestamp - record.queue_entry_time).total_seconds() / 60
            record.total_wait_time_minutes += wait_minutes

        success = self._transition_state(
            vehicle_id, TrackedState.CHARGING, timestamp,
            reason or f"charging at {station_id}"
        )

        if success:
            record = self._vehicle_records[vehicle_id]
            record.current_station_id = station_id
            record.charging_start_time = timestamp
            record.charging_stops += 1

            # Move from queue to station
            if vehicle_id in self._vehicles_in_queue:
                del self._vehicles_in_queue[vehicle_id]
            self._vehicles_at_station[vehicle_id] = station_id

            self._current_step_metrics.vehicles_started_charging += 1

        return success

    def transition_to_exiting(
        self,
        vehicle_id: str,
        timestamp: datetime,
        energy_received_kwh: float = 0.0,
        reason: str = ""
    ) -> bool:
        """Vehicle finishes charging and exits station."""
        record = self._vehicle_records.get(vehicle_id)
        if record and record.charging_start_time:
            charge_minutes = (timestamp - record.charging_start_time).total_seconds() / 60
            record.total_charge_time_minutes += charge_minutes
            record.energy_received_kwh += energy_received_kwh

        success = self._transition_state(
            vehicle_id, TrackedState.EXITING, timestamp,
            reason or "finished charging"
        )

        if success:
            record = self._vehicle_records[vehicle_id]
            record.charging_start_time = None
            record.queue_entry_time = None

            # Remove from station tracking
            if vehicle_id in self._vehicles_at_station:
                del self._vehicles_at_station[vehicle_id]

            self._current_step_metrics.vehicles_finished_charging += 1

        return success

    def transition_to_driving(
        self,
        vehicle_id: str,
        timestamp: datetime,
        reason: str = ""
    ) -> bool:
        """Vehicle returns to driving state (after exiting station or abandoning queue)."""
        # Check if abandoning queue
        was_queued = vehicle_id in self._vehicles_in_queue

        success = self._transition_state(
            vehicle_id, TrackedState.DRIVING, timestamp,
            reason or "resumed driving"
        )

        if success:
            record = self._vehicle_records[vehicle_id]
            record.current_station_id = None

            # Clean up station associations
            if vehicle_id in self._vehicles_in_queue:
                del self._vehicles_in_queue[vehicle_id]
            if vehicle_id in self._vehicles_at_station:
                del self._vehicles_at_station[vehicle_id]

        return success

    def record_abandonment(
        self,
        vehicle_id: str,
        timestamp: datetime,
        station_id: str,
        reason: str = ""
    ) -> bool:
        """Record that a vehicle abandoned the queue."""
        record = self._vehicle_records.get(vehicle_id)
        if not record:
            return False

        # Calculate wait time before abandoning
        if record.queue_entry_time:
            wait_minutes = (timestamp - record.queue_entry_time).total_seconds() / 60
            record.total_wait_time_minutes += wait_minutes

        success = self.transition_to_driving(
            vehicle_id, timestamp,
            reason or f"abandoned queue at {station_id}"
        )

        if success:
            self.total_abandonments += 1
            self._current_step_metrics.vehicles_abandoned_queue += 1
            self._current_step_metrics.abandoned_ids.append(vehicle_id)

        return success

    def transition_to_completed(
        self,
        vehicle_id: str,
        timestamp: datetime,
        final_position_km: float,
        final_soc: float,
        reason: str = ""
    ) -> bool:
        """Vehicle successfully completed the highway trip."""
        success = self._transition_state(
            vehicle_id, TrackedState.COMPLETED, timestamp,
            reason or "completed trip"
        )

        if success:
            record = self._vehicle_records[vehicle_id]
            record.exit_time = timestamp
            record.exit_reason = "completed"
            record.final_position_km = final_position_km
            record.final_soc = final_soc

            self._active_vehicle_ids.discard(vehicle_id)
            self._cleanup_vehicle_associations(vehicle_id)

            self.total_completed += 1
            self._current_step_metrics.vehicles_completed += 1
            self._current_step_metrics.completed_ids.append(vehicle_id)

            # Callback
            if self.on_exit:
                self.on_exit(vehicle_id, "completed", record)

        return success

    def transition_to_stranded(
        self,
        vehicle_id: str,
        timestamp: datetime,
        final_position_km: float,
        reason: str = ""
    ) -> bool:
        """Vehicle ran out of battery and is stranded."""
        success = self._transition_state(
            vehicle_id, TrackedState.STRANDED, timestamp,
            reason or "battery depleted"
        )

        if success:
            record = self._vehicle_records[vehicle_id]
            record.exit_time = timestamp
            record.exit_reason = "stranded"
            record.final_position_km = final_position_km
            record.final_soc = 0.0

            self._active_vehicle_ids.discard(vehicle_id)
            self._cleanup_vehicle_associations(vehicle_id)

            self.total_stranded += 1
            self._current_step_metrics.vehicles_stranded += 1
            self._current_step_metrics.stranded_ids.append(vehicle_id)

            # Callback
            if self.on_exit:
                self.on_exit(vehicle_id, "stranded", record)

        return success

    def _transition_state(
        self,
        vehicle_id: str,
        new_state: TrackedState,
        timestamp: datetime,
        reason: str
    ) -> bool:
        """Internal method to handle state transitions."""
        record = self._vehicle_records.get(vehicle_id)
        if not record:
            return False

        old_state = record.current_state

        # Don't transition if already in terminal state
        if old_state in (TrackedState.COMPLETED, TrackedState.STRANDED):
            return False

        # Update state tracking
        self._vehicles_by_state[old_state].discard(vehicle_id)
        self._vehicles_by_state[new_state].add(vehicle_id)

        record.current_state = new_state
        record.state_history.append((timestamp, new_state, reason))

        # Callback
        if self.on_state_change:
            self.on_state_change(vehicle_id, old_state, new_state, reason)

        return True

    def _cleanup_vehicle_associations(self, vehicle_id: str) -> None:
        """Remove vehicle from all station associations."""
        if vehicle_id in self._vehicles_at_station:
            del self._vehicles_at_station[vehicle_id]
        if vehicle_id in self._vehicles_in_queue:
            del self._vehicles_in_queue[vehicle_id]
            
            
    # Add to VehicleTracker class

    def get_stranding_positions(self) -> List[Tuple[float, datetime]]:
        """Get positions and times of all stranded vehicles."""
        stranded = self.get_failed_trips()
        return [(r.final_position_km, r.exit_time) for r in stranded if r.exit_time]

    def get_stranding_analysis(self) -> Dict[str, Any]:
        """Analyze stranding patterns."""
        positions = self.get_stranding_positions()
        if not positions:
            return {}
        
        pos_list, times = zip(*positions)
        
        return {
            'total_strandings': len(positions),
            'positions_km': list(pos_list),
            'times': list(times),
            'mean_position': np.mean(pos_list),
            'std_position': np.std(pos_list),
            'min_position': min(pos_list),
            'max_position': max(pos_list),
            'clustering': self._analyze_stranding_clusters(pos_list) # pyright: ignore[reportArgumentType]
        }
        
    def get_detailed_stranding_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data about all stranded vehicles for visualization.
        Returns positions, times, and context for each stranding event.
        """
        stranded_records = self.get_failed_trips()
        
        if not stranded_records:
            return {
                'total_strandings': 0,
                'positions_km': [],
                'times': [],
                'driver_types': [],
                'initial_socs': [],
                'battery_capacities': [],
                'analysis': {}
            }
        
        positions = []
        times = []
        driver_types = []
        initial_socs = []
        battery_caps = []
        
        for record in stranded_records:
            positions.append(record.final_position_km)
            times.append(record.exit_time.isoformat() if record.exit_time else None)
            driver_types.append(record.driver_type)
            initial_socs.append(record.initial_soc)
            battery_caps.append(record.battery_capacity_kwh)
        
        # Calculate distances between consecutive strandings
        sorted_positions = sorted(positions)
        gaps = [sorted_positions[i+1] - sorted_positions[i] 
                for i in range(len(sorted_positions)-1)]
        
        analysis = {
            'mean_position_km': float(np.mean(positions)),
            'median_position_km': float(np.median(positions)),
            'std_position_km': float(np.std(positions)),
            'min_position_km': float(min(positions)),
            'max_position_km': float(max(positions)),
            'most_common_driver_type': max(set(driver_types), key=driver_types.count),
            'avg_initial_soc': float(np.mean(initial_socs)),
            'position_gaps_km': gaps,
            'temporal_distribution': self._analyze_temporal_distribution(times)
        }
        
        return {
            'total_strandings': len(stranded_records),
            'positions_km': positions,
            'times': times,
            'driver_types': driver_types,
            'initial_socs': initial_socs,
            'battery_capacities': battery_caps,
            'analysis': analysis
        }

    def _analyze_temporal_distribution(self, times: List[str]) -> Dict[str, int]:
        """Analyze when strandings occur throughout the day."""
        if not times or times[0] is None:
            return {}
        
        hours = [datetime.fromisoformat(t).hour for t in times if t]
        hour_counts = {}
        for h in hours:
            hour_counts[h] = hour_counts.get(h, 0) + 1
        
        return hour_counts

    def _analyze_stranding_clusters(self, positions: List[float], 
                                highway_length: float = 300.0) -> List[Dict]:
        """Identify clusters of strandings (potential infrastructure gaps)."""
        if len(positions) < 3:
            return []
        
        # Simple clustering: 10km bins
        bin_size = 10
        bins = np.arange(0, highway_length + bin_size, bin_size)
        hist, edges = np.histogram(positions, bins=bins)
        
        clusters = []
        for i, count in enumerate(hist):
            if count > 0:
                clusters.append({
                    'start_km': edges[i],
                    'end_km': edges[i+1],
                    'count': int(count),
                    'center_km': (edges[i] + edges[i+1]) / 2
                })
        
        # Sort by count descending
        clusters.sort(key=lambda x: x['count'], reverse=True)
        return clusters

    # ========================================================================
    # POSITION / SOC UPDATES
    # ========================================================================

    def update_vehicle_position(self, vehicle_id: str, position_km: float) -> None:
        """Update vehicle's current position."""
        record = self._vehicle_records.get(vehicle_id)
        if record:
            record.position_km = position_km

    def update_vehicle_soc(self, vehicle_id: str, soc: float) -> None:
        """Update vehicle's current state of charge."""
        record = self._vehicle_records.get(vehicle_id)
        if record:
            record.current_soc = soc

    # ========================================================================
    # QUERIES
    # ========================================================================

    def get_active_vehicle_ids(self) -> Set[str]:
        """Get IDs of all active (non-completed, non-stranded) vehicles."""
        return self._active_vehicle_ids.copy()

    def get_vehicles_in_state(self, state: TrackedState) -> Set[str]:
        """Get IDs of vehicles in a specific state."""
        return self._vehicles_by_state[state].copy()

    def get_driving_vehicle_ids(self) -> Set[str]:
        """Get IDs of vehicles currently driving (including approaching/exiting)."""
        return (
            self._vehicles_by_state[TrackedState.DRIVING] |
            self._vehicles_by_state[TrackedState.APPROACHING] |
            self._vehicles_by_state[TrackedState.EXITING]
        )

    def get_vehicle_record(self, vehicle_id: str) -> Optional[VehicleRecord]:
        """Get the complete record for a vehicle."""
        return self._vehicle_records.get(vehicle_id)

    def get_vehicle_state(self, vehicle_id: str) -> Optional[TrackedState]:
        """Get current state of a vehicle."""
        record = self._vehicle_records.get(vehicle_id)
        return record.current_state if record else None

    def get_vehicles_at_station(self, station_id: str) -> Set[str]:
        """Get IDs of vehicles currently charging at a station."""
        return {vid for vid, sid in self._vehicles_at_station.items() if sid == station_id}

    def get_vehicles_in_queue_at_station(self, station_id: str) -> List[Tuple[str, datetime]]:
        """Get IDs and entry times of vehicles queued at a station."""
        return [
            (vid, entry_time)
            for vid, (sid, entry_time) in self._vehicles_in_queue.items()
            if sid == station_id
        ]

    def is_vehicle_active(self, vehicle_id: str) -> bool:
        """Check if a vehicle is still active in the simulation."""
        return vehicle_id in self._active_vehicle_ids

    def get_station_for_vehicle(self, vehicle_id: str) -> Optional[str]:
        """Get the station ID a vehicle is associated with (queued or charging)."""
        if vehicle_id in self._vehicles_at_station:
            return self._vehicles_at_station[vehicle_id]
        if vehicle_id in self._vehicles_in_queue:
            return self._vehicles_in_queue[vehicle_id][0]
        return None

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_state_counts(self) -> Dict[str, int]:
        """Get count of vehicles in each state."""
        return {state.name: len(ids) for state, ids in self._vehicles_by_state.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_spawned': self.total_spawned,
            'total_completed': self.total_completed,
            'total_stranded': self.total_stranded,
            'total_abandonments': self.total_abandonments,
            'currently_active': len(self._active_vehicle_ids),
            'currently_driving': len(self.get_driving_vehicle_ids()),
            'currently_queued': len(self._vehicles_in_queue),
            'currently_charging': len(self._vehicles_at_station),
            'completion_rate': self.total_completed / max(1, self.total_spawned),
            'stranding_rate': self.total_stranded / max(1, self.total_spawned)
        }

    def get_completed_trips(self) -> List[VehicleRecord]:
        """Get records of all completed trips."""
        return [
            r for r in self._vehicle_records.values()
            if r.current_state == TrackedState.COMPLETED
        ]

    def get_failed_trips(self) -> List[VehicleRecord]:
        """Get records of all stranded vehicles."""
        return [
            r for r in self._vehicle_records.values()
            if r.current_state == TrackedState.STRANDED
        ]

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate_consistency(self) -> List[str]:
        """
        Check internal consistency of tracking data.
        Returns list of any inconsistencies found.
        """
        issues = []

        # Check that all active vehicles are in exactly one state set
        all_state_vehicles = set()
        for state, vehicle_ids in self._vehicles_by_state.items():
            for vid in vehicle_ids:
                if vid in all_state_vehicles:
                    issues.append(f"Vehicle {vid} in multiple state sets")
                all_state_vehicles.add(vid)

        # Check that active vehicles match state tracking
        active_in_states = (
            self._vehicles_by_state[TrackedState.DRIVING] |
            self._vehicles_by_state[TrackedState.APPROACHING] |
            self._vehicles_by_state[TrackedState.QUEUED] |
            self._vehicles_by_state[TrackedState.CHARGING] |
            self._vehicles_by_state[TrackedState.EXITING]
        )

        if active_in_states != self._active_vehicle_ids:
            missing = self._active_vehicle_ids - active_in_states
            extra = active_in_states - self._active_vehicle_ids
            if missing:
                issues.append(f"Active IDs not in any state: {missing}")
            if extra:
                issues.append(f"State vehicles not in active set: {extra}")

        # Check station associations
        for vid in self._vehicles_at_station:
            record = self._vehicle_records.get(vid)
            if not record or record.current_state != TrackedState.CHARGING:
                issues.append(f"Vehicle {vid} at station but not in CHARGING state")

        for vid in self._vehicles_in_queue:
            record = self._vehicle_records.get(vid)
            if not record or record.current_state != TrackedState.QUEUED:
                issues.append(f"Vehicle {vid} in queue but not in QUEUED state")

        return issues

    def __repr__(self) -> str:
        return (f"VehicleTracker(active={len(self._active_vehicle_ids)}, "
                f"completed={self.total_completed}, stranded={self.total_stranded})")
