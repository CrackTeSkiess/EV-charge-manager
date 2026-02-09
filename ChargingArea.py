"""
Charging Area Module
Core component of the EV charging station simulation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Callable, TYPE_CHECKING
from collections import deque
import heapq
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from EnergyManager import EnergyManager


class ChargerStatus(Enum):
    AVAILABLE = auto()
    OCCUPIED = auto()
    MAINTENANCE = auto()
    RESERVED = auto()
    UNPOWERED = auto()  # No energy supply from EnergyManager


class ChargerType(Enum):
    SLOW = 22       # kW - AC Level 2
    FAST = 50       # kW - DC Fast
    ULTRA = 150     # kW - DC Ultra-fast
    MEGAWATT = 350  # kW - High-power for trucks


@dataclass
class ChargingSession:
    """Represents an active or completed charging session."""
    vehicle_id: str
    start_time: datetime
    requested_energy_kwh: float
    delivered_energy_kwh: float = 0.0
    expected_end_time: Optional[datetime] = None
    
    def remaining_time(self, current_time: datetime) -> timedelta:
        if self.expected_end_time:
            return max(timedelta(0), self.expected_end_time - current_time)
        return timedelta(0)
    
    def completion_percentage(self) -> float:
        if self.requested_energy_kwh > 0:
            return min(100.0, (self.delivered_energy_kwh / self.requested_energy_kwh) * 100)
        return 0.0


@dataclass 
class Charger:
    """
    Individual charging point within a ChargingArea.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    power_kw: float = 50.0
    charger_type: ChargerType = ChargerType.ULTRA
    status: ChargerStatus = ChargerStatus.AVAILABLE
    current_session: Optional[ChargingSession] = None
    efficiency_curve: Callable[[float], float] = field(default_factory=lambda: lambda soc: 0.9)
    total_sessions: int = 0
    total_energy_delivered_kwh: float = 0.0
    
    def is_available(self) -> bool:
        return self.status == ChargerStatus.AVAILABLE
    
    def start_session(self, vehicle_id: str, energy_needed_kwh: float, 
                     current_time: datetime, vehicle_soc: float = 0.5) -> ChargingSession:
        """Start a new charging session."""
        if not self.is_available():
            raise ValueError(f"Charger {self.id} is not available")
        
        # Calculate charging time based on power and efficiency at current SOC
        efficiency = self.efficiency_curve(vehicle_soc)
        effective_power = self.power_kw * efficiency
        duration_minutes = (energy_needed_kwh / effective_power) * 60
        
        session = ChargingSession(
            vehicle_id=vehicle_id,
            start_time=current_time,
            requested_energy_kwh=energy_needed_kwh,
            expected_end_time=current_time + timedelta(minutes=duration_minutes)
        )
        
        self.current_session = session
        self.status = ChargerStatus.OCCUPIED
        self.total_sessions += 1
        
        return session
    
    def update_session(self, time_elapsed_minutes: float) -> Optional[ChargingSession]:
        """Update session progress. Returns completed session if finished."""
        if self.current_session is None:
            return None
        
        # Calculate energy delivered in this time step
        efficiency = self.efficiency_curve(0.5)  # Simplified - would track actual SOC
        energy_delivered = (self.power_kw * efficiency) * (time_elapsed_minutes / 60)
        
        self.current_session.delivered_energy_kwh += energy_delivered
        
        # NOTE: We do NOT auto-complete based on requested_energy_kwh anymore
        # The Highway controls when to complete based on safety requirements
        # Return None to indicate session is still active
        # Highway will call complete_session() when vehicle can safely exit
        
        return None    
    
    def complete_session(self) -> ChargingSession:
        """Complete current session and free charger."""
        if self.current_session is None:
            raise ValueError("No active session to complete")
        
        completed = self.current_session
        self.total_energy_delivered_kwh += completed.delivered_energy_kwh
        self.current_session = None
        self.status = ChargerStatus.AVAILABLE
        
        return completed
    
    def estimate_availability(self, current_time: datetime) -> datetime:
        """Estimate when this charger will become available."""
        if self.is_available():
            return current_time
        
        if self.current_session and self.current_session.expected_end_time:
            # Add buffer for connection/disconnection
            return self.current_session.expected_end_time + timedelta(minutes=2)
        
        return current_time + timedelta(hours=1)  # Conservative default
    
    def interrupt_session(self, reason: str = "user_request") -> ChargingSession:
        """Interrupt ongoing session (user abandonment)."""
        if self.current_session is None:
            raise ValueError("No active session to interrupt")
        
        self.current_session.expected_end_time = None  # Mark as incomplete
        return self.complete_session()


@dataclass(order=True)
class QueueEntry:
    """Priority queue entry for waiting vehicles."""
    priority: float  # Lower = higher priority (for heapq)
    entry_time: datetime = field(compare=False)
    vehicle_id: str = field(compare=False)
    queue_position: int = field(compare=False, default=0)
    patience_score: float = field(compare=False, default=1.0)
    requested_power_preference: Optional[ChargerType] = field(compare=False, default=None)
    
    def __post_init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True


class ChargingArea:
    """
    A charging area containing multiple chargers and a waiting area.
    
    Tunable parameters:
    - num_chargers: Number of charging points
    - charger_power_config: Dict mapping ChargerType to count
    - waiting_spots: Maximum vehicles in waiting area
    """
    
    def __init__(
        self,
        area_id: Optional[str] = None,
        num_chargers: int = 4,
        charger_power_config: Optional[Dict[ChargerType, int]] = None,
        waiting_spots: int = 6,
        location_km: float = 0.0,
        name: str = "Charging Area",
        energy_manager: Optional[EnergyManager] = None
    ):
        self.id = area_id or str(uuid.uuid4())[:8]
        self.name = name
        self.location_km = location_km
        self.energy_manager = energy_manager
        
        # Tunable: Waiting area capacity
        self.waiting_spots = waiting_spots
        self.waiting_occupied = 0
        
        # Initialize chargers based on configuration
        self.chargers: List[Charger] = []
        self._initialize_chargers(num_chargers, charger_power_config)
        
        # Queue management
        self.queue: List[QueueEntry] = []  # Min-heap by priority
        self.queue_map: Dict[str, QueueEntry] = {}  # vehicle_id -> entry
        self.served_count = 0
        self.abandoned_count = 0
        
        # Statistics
        self.stats = {
            'total_arrivals': 0,
            'immediate_service': 0,
            'queued': 0,
            'abandoned_from_queue': 0,
            'abandoned_from_waiting': 0,
            'completed_sessions': 0,
            'total_wait_time_minutes': 0.0,
            'peak_queue_length': 0
        }
        
        self.current_time: Optional[datetime] = None

        # Energy management (optional — None means unlimited power)
        self.energy_manager: Optional[EnergyManager] = None

        # Cached available chargers (invalidated on status change)
        self._available_chargers_cache: Optional[List[Charger]] = None

        # Dirty flag for queue cleanup (set when abandon marks entries)
        self._queue_dirty = False
        
        self.rl_targets: Optional[Dict] = None
    
    def _initialize_chargers(
        self, 
        num_chargers: int, 
        config: Optional[Dict[ChargerType, int]]
    ) -> None:
        """Create charger instances based on configuration."""
        if config:
            # Explicit configuration: {ChargerType.FAST: 2, ChargerType.ULTRA: 2}
            for charger_type, count in config.items():
                power = self._get_power_for_type(charger_type)
                for _ in range(count):
                    self.chargers.append(Charger(
                        power_kw=power,
                        charger_type=charger_type
                    ))
        else:
            # Default: equal distribution
            default_type = ChargerType.ULTRA
            power = self._get_power_for_type(default_type)
            for _ in range(num_chargers):
                self.chargers.append(Charger(
                    power_kw=power,
                    charger_type=default_type
                ))
    
    def _get_power_for_type(self, charger_type: ChargerType) -> float:
        """Get default power rating for charger type."""
        return {
            ChargerType.SLOW: 22.0,
            ChargerType.FAST: 50.0,
            ChargerType.ULTRA: 150.0,
            ChargerType.MEGAWATT: 350.0
        }[charger_type]
    
    # =========================================================================
    # PUBLIC API: State Broadcasting
    # =========================================================================
    
    def get_broadcast_state(self) -> Dict:
        """
        Public information broadcast to approaching vehicles.
        """
        available_chargers = self.get_available_chargers()
        
        return {
            'area_id': self.id,
            'location_km': self.location_km,
            'name': self.name,
            'available_chargers': len(available_chargers),
            'total_chargers': len(self.chargers),
            'available_by_type': self._count_by_type(available_chargers),
            'queue_length': len(self.queue),
            'waiting_spots_available': self.waiting_spots - self.waiting_occupied,
            'total_waiting_spots': self.waiting_spots,
            'estimated_wait_minutes': self.estimate_wait_time(),
            'next_availability': self.estimate_next_availability(),
            'is_full': self.is_full(),
            'current_utilization': self.get_utilization_rate(),
            'chargers_unpowered': sum(1 for c in self.chargers if c.status == ChargerStatus.UNPOWERED)
        }
    
    def estimate_next_availability(self) -> Optional[datetime]:
        """
        Estimate when the next charger will become available.
        Returns None if chargers available now.
        """
        if self.get_available_chargers():
            return None
        
        earliest = None
        for charger in self.chargers:
            avail = charger.estimate_availability(self.current_time or datetime.now())
            if earliest is None or avail < earliest:
                earliest = avail
        
        return earliest
    
    def estimate_wait_time(self) -> float:
        """
        Estimate wait time in minutes for a new arrival.
        Uses fuzzy-inspired calculation based on queue and charging progress.
        """
        if self.get_available_chargers():
            return 0.0
        
        # Base: time until next charger free
        next_avail = self.estimate_next_availability()
        if next_avail is None:
            return 0.0
        
        base_wait = (next_avail - (self.current_time or datetime.now())).total_seconds() / 60
        
        # Add queue penalty: each queued vehicle adds estimated service time
        avg_service_time = 30.0  # minutes, would be tuned from data
        queue_penalty = len(self.queue) * (avg_service_time / len(self.chargers))
        
        return base_wait + queue_penalty
    
    # =========================================================================
    # PUBLIC API: Vehicle Entry and Queue Management
    # =========================================================================
    
    def request_entry(self, vehicle_id: str, priority_score: float = 1.0,
                     power_preference: Optional[ChargerType] = None) -> Dict:
        """
        Vehicle requests to enter the charging area.
        
        Returns:
            Dict with 'granted', 'action' (immediate/queue/wait/denied), 
            and 'estimated_wait'
        """
        self.stats['total_arrivals'] += 1
        
        # Check for immediate service
        available = self.get_available_chargers()
        if available and not self.queue:
            # Can start immediately
            self.stats['immediate_service'] += 1
            return {
                'granted': True,
                'action': 'immediate',
                'assigned_charger': available[0].id,
                'estimated_wait': 0.0
            }
        
        # Check if can join queue
        if len(self.queue) < self.waiting_spots:
            return self._join_queue(vehicle_id, priority_score, power_preference)
        
        # Waiting area full
        return {
            'granted': False,
            'action': 'denied',
            'reason': 'waiting_area_full',
            'estimated_wait': self.estimate_wait_time()
        }
    
    def _join_queue(self, vehicle_id: str, priority_score: float,
                    power_preference: Optional[ChargerType]) -> Dict:
        """Add vehicle to priority queue."""
        entry = QueueEntry(
            priority=priority_score,
            entry_time=self.current_time or datetime.now(),
            vehicle_id=vehicle_id,
            requested_power_preference=power_preference,
            queue_position=len(self.queue) + 1
        )
        
        heapq.heappush(self.queue, entry)
        self.queue_map[vehicle_id] = entry
        self.waiting_occupied += 1
        self.stats['queued'] += 1
        
        if len(self.queue) > self.stats['peak_queue_length']:
            self.stats['peak_queue_length'] = len(self.queue)
        
        return {
            'granted': True,
            'action': 'queued',
            'queue_position': entry.queue_position,
            'estimated_wait': self.estimate_wait_time()
        }
    
    def abandon_queue(self, vehicle_id: str, from_waiting_area: bool = True) -> bool:
        """
        Vehicle decides to leave the queue.
        
        Returns True if successfully removed, False if not found.
        """
        if vehicle_id not in self.queue_map:
            return False
        
        entry = self.queue_map[vehicle_id]
        
        # Mark as abandoned (can't easily remove from heap, filter later)
        entry.patience_score = -1.0  # Marker for abandonment
        self._queue_dirty = True

        del self.queue_map[vehicle_id]
        self.waiting_occupied -= 1
        
        if from_waiting_area:
            self.stats['abandoned_from_waiting'] += 1
        else:
            self.stats['abandoned_from_queue'] += 1
        
        self.abandoned_count += 1
        return True
    
    def update_queue_patience(self, vehicle_id: str, new_patience: float) -> None:
        """Update patience score for a queued vehicle (affects priority)."""
        if vehicle_id in self.queue_map:
            entry = self.queue_map[vehicle_id]
            entry.patience_score = new_patience
            # Note: In full implementation, would re-heapify if priority changes significantly
    
    # =========================================================================
    # PUBLIC API: Charging Operations
    # =========================================================================
    
    def start_charging(self, vehicle_id: str, energy_needed_kwh: float,
                      vehicle_soc: float = 0.2) -> Optional[Dict]:
        """
        Start charging session for vehicle at head of queue.
        """
        # Find best matching charger
        charger = self._select_charger_for_vehicle(vehicle_id, vehicle_soc)
        if not charger:
            return None
        
        # Remove from queue
        if vehicle_id in self.queue_map:
            entry = self.queue_map.pop(vehicle_id)
            self.waiting_occupied -= 1
            wait_time = (self.current_time - entry.entry_time).total_seconds() / 60 # pyright: ignore[reportOptionalOperand]
            self.stats['total_wait_time_minutes'] += wait_time
        
        # Start session
        session = charger.start_session(
            vehicle_id=vehicle_id,
            energy_needed_kwh=energy_needed_kwh,
            current_time=self.current_time, # pyright: ignore[reportArgumentType]
            vehicle_soc=vehicle_soc
        )
        self._invalidate_charger_cache()

        self.served_count += 1
        
        return {
            'charger_id': charger.id,
            'power_kw': charger.power_kw,
            'expected_duration_min': (session.expected_end_time - session.start_time).total_seconds() / 60, # pyright: ignore[reportOptionalOperand]
            'session': session
        }
    
    def _select_charger_for_vehicle(self, vehicle_id: str, 
                                    vehicle_soc: float) -> Optional[Charger]:
        """Select optimal available charger for vehicle."""
        available = self.get_available_chargers()
        if not available:
            return None
        
        # Simple selection: fastest available
        # In full version: match power_preference from queue entry
        return max(available, key=lambda c: c.power_kw)

    # =========================================================================
    # ENERGY MANAGEMENT
    # =========================================================================

    def set_energy_manager(self, energy_manager: EnergyManager) -> None:
        """Attach an energy manager to this charging area."""
        self.energy_manager = energy_manager

    def _requeue_vehicle(self, vehicle_id: str) -> None:
        """Re-add a vehicle to the queue after energy curtailment (high priority)."""
        entry = QueueEntry(
            priority=0.0,  # Highest priority — was already being served
            entry_time=self.current_time or datetime.now(),
            vehicle_id=vehicle_id,
            queue_position=0
        )
        heapq.heappush(self.queue, entry)
        self.queue_map[vehicle_id] = entry
        self.waiting_occupied += 1

    def _apply_energy_constraints(self, timestamp: datetime, time_step_minutes: float) -> Dict:
        """
        Apply energy constraints: determine which chargers get power.
        Called at the start of each step, before session updates.

        Returns dict of events (powered_down, powered_up, requeued).
        """
        if self.energy_manager is None or self.energy_manager.is_unlimited:
            # Power up any previously unpowered chargers (in case manager was removed)
            for charger in self.chargers:
                if charger.status == ChargerStatus.UNPOWERED:
                    charger.status = ChargerStatus.AVAILABLE
                    self._invalidate_charger_cache()
            return {'powered_down': [], 'powered_up': [], 'requeued': []}

        events: Dict[str, list] = {'powered_down': [], 'powered_up': [], 'requeued': []}

        # 1. Update energy sources
        self.energy_manager.step(timestamp, time_step_minutes)

        # 2. Calculate total demand: all chargers we want to keep powered
        #    (occupied + available + currently unpowered that could be restored)
        all_serviceable = [
            c for c in self.chargers
            if c.status in (ChargerStatus.OCCUPIED, ChargerStatus.AVAILABLE, ChargerStatus.UNPOWERED)
        ]
        total_demand_kw = sum(c.power_kw for c in all_serviceable)

        # 3. Request energy from manager
        available_kw = self.energy_manager.request_energy(total_demand_kw, time_step_minutes)

        # 4. Allocate power: prioritize OCCUPIED chargers (most-complete first),
        #    then AVAILABLE chargers, powering down the rest.
        occupied_chargers = [c for c in self.chargers if c.status == ChargerStatus.OCCUPIED]
        sorted_occupied = sorted(
            occupied_chargers,
            key=lambda c: c.current_session.completion_percentage() if c.current_session else 0,
            reverse=True
        )

        budget_kw = available_kw

        # 4a. Allocate to occupied chargers (keep most-complete running)
        for charger in sorted_occupied:
            if budget_kw >= charger.power_kw:
                budget_kw -= charger.power_kw
            else:
                # Power down this charger — not enough energy
                vehicle_id = None
                if charger.current_session:
                    vehicle_id = charger.current_session.vehicle_id
                    charger.interrupt_session(reason="energy_curtailment")
                    self._requeue_vehicle(vehicle_id)
                    events['requeued'].append(vehicle_id)
                charger.status = ChargerStatus.UNPOWERED
                self._invalidate_charger_cache()
                events['powered_down'].append({
                    'charger_id': charger.id,
                    'vehicle_id': vehicle_id,
                    'reason': 'energy_curtailment'
                })

        # 4b. Allocate remaining budget to AVAILABLE chargers
        for charger in self.chargers:
            if charger.status == ChargerStatus.AVAILABLE:
                if budget_kw >= charger.power_kw:
                    budget_kw -= charger.power_kw
                else:
                    charger.status = ChargerStatus.UNPOWERED
                    self._invalidate_charger_cache()

        # 5. Power up previously UNPOWERED chargers if budget remains
        for charger in self.chargers:
            if charger.status == ChargerStatus.UNPOWERED:
                if budget_kw >= charger.power_kw:
                    charger.status = ChargerStatus.AVAILABLE
                    budget_kw -= charger.power_kw
                    events['powered_up'].append({'charger_id': charger.id})
                    self._invalidate_charger_cache()

        return events

    # =========================================================================
    # SIMULATION STEP
    # =========================================================================

    def step(self, current_time: datetime, time_step_minutes: float = 1.0) -> Dict:
        """
        Advance simulation by one time step.

        Returns summary of events this step.
        """
        self.current_time = current_time
        events = {
            'completed_sessions': [],
            'new_assignments': [],
            'abandonments_cleaned': 0,
            'energy_events': {}
        }
        
        if hasattr(self, 'rl_targets') and self.rl_targets:
            # Override default energy dispatch with RL targets
            # This would modify how _manage_energy operates
            pass

        # Apply energy constraints FIRST (before session updates)
        energy_events = self._apply_energy_constraints(current_time, time_step_minutes)
        events['energy_events'] = energy_events

        # Update all active charging sessions AND vehicle batteries
        for charger in self.chargers:
            if charger.status == ChargerStatus.OCCUPIED:
                # Calculate energy delivered this step
                efficiency = charger.efficiency_curve(0.5)
                energy_delivered = (charger.power_kw * efficiency) * (time_step_minutes / 60)
                
                # Update session tracking
                if charger.current_session:
                    charger.current_session.delivered_energy_kwh += energy_delivered
                
                # CRITICAL: Update the vehicle's actual battery
                # Need to get vehicle reference - store it in session or look it up
                vehicle_id = charger.current_session.vehicle_id if charger.current_session else None
                if vehicle_id:
                    # Find vehicle in highway - need reference to highway
                    # For now, just track in session, Highway will apply it
                    pass
                
                # Check completion based on requested energy
                if charger.current_session and charger.current_session.delivered_energy_kwh >= charger.current_session.requested_energy_kwh:
                    completed = charger.complete_session()
                    self._invalidate_charger_cache()
                    if completed:
                        events['completed_sessions'].append({
                            'vehicle_id': completed.vehicle_id,
                            'energy_delivered': completed.delivered_energy_kwh,
                            'duration_min': (current_time - completed.start_time).total_seconds() / 60
                        })
                        self.stats['completed_sessions'] += 1
        
        # Clean abandoned entries from queue and assign new vehicles
        self._clean_queue()
        
        # Assign available chargers to queue
        while self.get_available_chargers() and self.queue:
            next_entry = self._get_next_valid_entry()
            if not next_entry:
                break
            
            # Get actual vehicle data - need highway reference
            result = self.start_charging(
                next_entry.vehicle_id,
                energy_needed_kwh=50.0,  # FIXME: Should come from vehicle
                vehicle_soc=0.2  # FIXME: Should come from vehicle
            )
            if result:
                events['new_assignments'].append({
                    'vehicle_id': next_entry.vehicle_id,
                    'charger_id': result['charger_id']
                })
        
        return events    
    
    def _clean_queue(self) -> int:
        """Remove abandoned entries marked with patience_score < 0."""
        if not self._queue_dirty:
            return 0

        cleaned = 0
        new_queue = []
        for entry in self.queue:
            if entry.patience_score < 0:
                cleaned += 1
                # Note: waiting_occupied already decremented in abandon_queue()
                # Only decrement if entry still in queue_map (not already processed)
                if entry.vehicle_id in self.queue_map:
                    del self.queue_map[entry.vehicle_id]
            else:
                # Update position
                entry.queue_position = len(new_queue) + 1
                new_queue.append(entry)

        self.queue = new_queue
        heapq.heapify(self.queue)  # Re-heapify
        self._queue_dirty = False
        return cleaned
    
    def _get_next_valid_entry(self) -> Optional[QueueEntry]:
        """Get next valid entry from queue, skipping abandoned."""
        while self.queue:
            entry = heapq.heappop(self.queue)
            if entry.patience_score >= 0 and entry.vehicle_id in self.queue_map:
                return entry
            # Abandoned entry - just clean up queue_map if still there
            # Note: waiting_occupied already handled in abandon_queue()
            if entry.vehicle_id in self.queue_map:
                del self.queue_map[entry.vehicle_id]
        return None
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_available_chargers(self) -> List[Charger]:
        """Return list of currently available chargers (cached)."""
        if self._available_chargers_cache is None:
            self._available_chargers_cache = [c for c in self.chargers if c.is_available()]
        return self._available_chargers_cache

    def _invalidate_charger_cache(self) -> None:
        """Invalidate cached available chargers list."""
        self._available_chargers_cache = None
    
    def get_utilization_rate(self) -> float:
        """Calculate current utilization (0.0 to 1.0). Excludes UNPOWERED chargers."""
        powered_chargers = [c for c in self.chargers if c.status != ChargerStatus.UNPOWERED]
        occupied = sum(1 for c in powered_chargers if c.status == ChargerStatus.OCCUPIED)
        return occupied / len(powered_chargers) if powered_chargers else 0.0
    
    def is_full(self) -> bool:
        """Check if waiting area is at capacity."""
        return self.waiting_occupied >= self.waiting_spots
    
    def _count_by_type(self, chargers: List[Charger]) -> Dict[str, int]:
        """Count chargers by type."""
        counts = {}
        for c in chargers:
            type_name = c.charger_type.name
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def get_statistics(self) -> Dict:
        """Return comprehensive statistics."""
        avg_wait = (self.stats['total_wait_time_minutes'] / max(1, self.served_count))
        
        return {
            **self.stats,
            'avg_wait_time_minutes': avg_wait,
            'service_rate': self.served_count / max(1, self.stats['total_arrivals']),
            'abandonment_rate': self.abandoned_count / max(1, self.stats['total_arrivals']),
            'current_queue_length': len(self.queue),
            'current_utilization': self.get_utilization_rate(),
            'total_energy_delivered_kwh': sum(c.total_energy_delivered_kwh for c in self.chargers)
        }
    
    def __repr__(self) -> str:
        return (f"ChargingArea(id={self.id}, chargers={len(self.chargers)}, "
                f"waiting={self.waiting_spots}, queue={len(self.queue)}, "
                f"utilization={self.get_utilization_rate():.1%})")


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def demo():
    """Demonstrate ChargingArea functionality."""
    print("=" * 60)
    print("CHARGING AREA DEMO")
    print("=" * 60)
    
    # Create charging area with mixed charger types
    area = ChargingArea(
        name="Highway Plaza North",
        location_km=125.5,
        num_chargers=6,
        charger_power_config={
            ChargerType.FAST: 4,
            ChargerType.ULTRA: 2
        },
        waiting_spots=8
    )
    
    print(f"\nCreated: {area}")
    print(f"Charger configuration:")
    for c in area.chargers:
        print(f"  - {c.id}: {c.charger_type.name} ({c.power_kw}kW)")
    
    # Simulate time progression
    start_time = datetime(2024, 1, 15, 14, 0)
    
    # Step 1: Check initial state
    print(f"\n--- {start_time.strftime('%H:%M')} ---")
    state = area.get_broadcast_state()
    print(f"Broadcast: {state['available_chargers']} available, "
          f"queue: {state['queue_length']}, "
          f"est. wait: {state['estimated_wait_minutes']:.1f}min")
    
    # Step 2: Vehicles arrive
    print(f"\n--- Vehicles arriving ---")
    
    # Vehicle 1: Immediate service
    result = area.request_entry("VEH-001", priority_score=1.0)
    print(f"VEH-001 request: {result['action']} (charger: {result.get('assigned_charger', 'N/A')})")
    
    # Vehicle 2: Immediate service
    result = area.request_entry("VEH-002", priority_score=1.0)
    print(f"VEH-002 request: {result['action']} (charger: {result.get('assigned_charger', 'N/A')})")
    
    # Vehicles 3-10: Fill queue
    for i in range(3, 11):
        result = area.request_entry(f"VEH-{i:03d}", priority_score=0.5 + i*0.05)
        print(f"VEH-{i:03d} request: {result['action']} (pos: {result.get('queue_position', 'N/A')})")
    
    # Check state
    print(f"\n--- State after arrivals ---")
    state = area.get_broadcast_state()
    print(f"Available: {state['available_chargers']}, "
          f"Queue: {state['queue_length']}, "
          f"Waiting spots free: {state['waiting_spots_available']}")
    print(f"Estimated wait for new arrival: {state['estimated_wait_minutes']:.1f} minutes")
    print(f"Next availability: {state['next_availability']}")
    
    # Step 3: Vehicle abandons
    print(f"\n--- VEH-005 abandons queue ---")
    success = area.abandon_queue("VEH-005")
    print(f"Abandonment successful: {success}")
    
    # Step 4: Run simulation steps
    print(f"\n--- Running simulation steps ---")
    for step in range(5):
        current_time = start_time + timedelta(minutes=step)
        events = area.step(current_time)
        
        print(f"\n{current_time.strftime('%H:%M')}:")
        if events['completed_sessions']:
            for cs in events['completed_sessions']:
                print(f"  ✓ Completed: {cs['vehicle_id']}, "
                      f"{cs['energy_delivered']:.1f}kWh")
        if events['new_assignments']:
            for na in events['new_assignments']:
                print(f"  → Started: {na['vehicle_id']} on {na['charger_id']}")
        
        state = area.get_broadcast_state()
        print(f"  Status: {state['available_chargers']} free, "
              f"{state['queue_length']} in queue")
    
    # Final statistics
    print(f"\n--- Final Statistics ---")
    stats = area.get_statistics()
    print(f"Total arrivals: {stats['total_arrivals']}")
    print(f"Immediate service: {stats['immediate_service']}")
    print(f"Queued: {stats['queued']}")
    print(f"Completed: {stats['completed_sessions']}")
    print(f"Abandoned: {stats['abandoned_from_queue'] + stats['abandoned_from_waiting']}")
    print(f"Average wait time: {stats['avg_wait_time_minutes']:.1f} min")
    print(f"Service rate: {stats['service_rate']:.1%}")
    print(f"Peak queue length: {stats['peak_queue_length']}")
    print(f"Total energy delivered: {stats['total_energy_delivered_kwh']:.1f} kWh")


if __name__ == "__main__":
    demo()