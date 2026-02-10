"""
Blackout Tracker
Tracks energy shortfall events and grid instability
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum, auto


class BlackoutType(Enum):
    """Types of energy shortfall events"""
    TOTAL_BLACKOUT = auto()  # Complete loss of charging capability
    PARTIAL_BROWNOUT = auto()  # Reduced power available
    GRID_OVERLOAD = auto()  # Exceeded grid connection
    BATTERY_DEPLETED = auto()  # Battery exhausted during peak


@dataclass
class EnergyShortfallEvent:
    """Record of a blackout/brownout event"""
    event_id: str
    timestamp: datetime
    station_id: str
    blackout_type: BlackoutType
    demand_kw: float
    available_kw: float
    shortfall_kw: float
    affected_vehicles: int
    duration_minutes: float
    root_cause: str  # Description of cause
    
    @property
    def severity_score(self) -> float:
        """Calculate severity based on shortfall magnitude and duration"""
        return (self.shortfall_kw / max(self.demand_kw, 1)) * self.duration_minutes


class BlackoutTracker:
    """
    Tracks energy shortfall events for optimization objective
    """
    
    def __init__(self, blackout_threshold: float = 0.1):
        """
        blackout_threshold: fraction of unmet demand to count as blackout
        """
        self.blackout_threshold = blackout_threshold
        
        # Event history
        self.events: List[EnergyShortfallEvent] = []
        
        # Current state tracking
        self.ongoing_events: Dict[str, EnergyShortfallEvent] = {}  # station_id -> event
        
        # Metrics
        self.total_blackouts: int = 0
        self.total_brownouts: int = 0
        self.total_shortfall_kwh: float = 0.0
        self.max_concurrent_blackouts: int = 0
        self.worst_shortfall_percentage: float = 0.0
        
        # Time-series tracking
        self.hourly_blackout_count: Dict[int, int] = {}
        self.station_blackout_count: Dict[str, int] = {}
    
    def check_energy_balance(self, station_id: str, timestamp: datetime,
                           total_demand_kw: float,
                           grid_available_kw: float,
                           battery_available_kw: float,
                           renewable_available_kw: float,
                           vehicles_charging: int) -> Optional[Dict]:
        """
        Check if energy demand can be met, record event if not
        
        Returns: Alert dict if blackout/brownout detected, None if OK
        """
        total_available = grid_available_kw + battery_available_kw + renewable_available_kw
        shortfall_kw = max(0, total_demand_kw - total_available)
        shortfall_ratio = shortfall_kw / max(total_demand_kw, 1)
        
        # Determine event type
        if shortfall_ratio >= self.blackout_threshold:
            if shortfall_ratio >= 0.9:
                btype = BlackoutType.TOTAL_BLACKOUT
            elif battery_available_kw <= 0 and renewable_available_kw <= 0:
                btype = BlackoutType.GRID_OVERLOAD
            elif battery_available_kw <= 0:
                btype = BlackoutType.BATTERY_DEPLETED
            else:
                btype = BlackoutType.PARTIAL_BROWNOUT
            
            # Start or continue event
            if station_id not in self.ongoing_events:
                event = EnergyShortfallEvent(
                    event_id=f"{station_id}_{timestamp.isoformat()}",
                    timestamp=timestamp,
                    station_id=station_id,
                    blackout_type=btype,
                    demand_kw=total_demand_kw,
                    available_kw=total_available,
                    shortfall_kw=shortfall_kw,
                    affected_vehicles=vehicles_charging,
                    duration_minutes=0,  # Will be updated
                    root_cause=self._determine_root_cause(
                        grid_available_kw, battery_available_kw, 
                        renewable_available_kw, total_demand_kw
                    )
                )
                self.ongoing_events[station_id] = event
                self._record_event_start(event)
            
            return {
                'type': 'ENERGY_SHORTFALL',
                'station_id': station_id,
                'shortfall_kw': shortfall_kw,
                'shortfall_ratio': shortfall_ratio,
                'blackout_type': btype.name,
                'affected_vehicles': vehicles_charging,
                'timestamp': timestamp,
                'severity': 'CRITICAL' if btype == BlackoutType.TOTAL_BLACKOUT else 'WARNING'
            }
        
        else:
            # No shortfall - close any ongoing event for this station
            if station_id in self.ongoing_events:
                self._close_event(station_id, timestamp)
            return None
    
    def _determine_root_cause(self, grid_kw: float, battery_kw: float,
                            renewable_kw: float, demand_kw: float) -> str:
        """Analyze root cause of shortfall"""
        available = grid_kw + battery_kw + renewable_kw
        
        if grid_kw <= 0:
            return "GRID_FAILURE"
        elif renewable_kw < 0.1 * demand_kw and battery_kw <= 0:
            return "RENEWABLE_INTERMITTENCY_WITH_DEPLETED_BATTERY"
        elif demand_kw > grid_kw * 1.5:
            return "DEMAND_SURGE"
        elif battery_kw <= 0:
            return "BATTERY_DEPLETED"
        else:
            return "COMPOUND_CONSTRAINTS"
    
    def _record_event_start(self, event: EnergyShortfallEvent) -> None:
        """Record new blackout event"""
        self.events.append(event)
        
        if event.blackout_type == BlackoutType.TOTAL_BLACKOUT:
            self.total_blackouts += 1
        else:
            self.total_brownouts += 1
        
        # Update station stats
        self.station_blackout_count[event.station_id] = \
            self.station_blackout_count.get(event.station_id, 0) + 1
        
        # Update hourly stats
        hour = event.timestamp.hour
        self.hourly_blackout_count[hour] = \
            self.hourly_blackout_count.get(hour, 0) + 1
        
        # Track worst case
        shortfall_pct = event.shortfall_kw / max(event.demand_kw, 1)
        self.worst_shortfall_percentage = max(
            self.worst_shortfall_percentage, 
            shortfall_pct
        )
        
        # Track concurrent
        concurrent = len(self.ongoing_events)
        self.max_concurrent_blackouts = max(
            self.max_concurrent_blackouts,
            concurrent
        )
    
    def _close_event(self, station_id: str, end_timestamp: datetime) -> None:
        """Close ongoing event and calculate duration"""
        if station_id not in self.ongoing_events:
            return
        
        event = self.ongoing_events.pop(station_id)
        duration = (end_timestamp - event.timestamp).total_seconds() / 60
        event.duration_minutes = duration
        
        # Accumulate energy shortfall
        self.total_shortfall_kwh += (event.shortfall_kw * duration / 60)
    
    def step(self, current_time: datetime):
        """Update ongoing events (call each timestep)"""
        # Update durations for ongoing events
        for station_id, event in self.ongoing_events.items():
            duration = (current_time - event.timestamp).total_seconds() / 60
            event.duration_minutes = duration
    
    def get_blackout_rate(self, simulation_duration_hours: float) -> float:
        """Calculate blackout events per year"""
        years = simulation_duration_hours / (24 * 365)
        total_events = self.total_blackouts + self.total_brownouts
        return total_events / max(years, 0.001)
    
    def get_metrics(self) -> Dict:
        """Get comprehensive blackout metrics"""
        total_events = len(self.events)
        
        return {
            'total_blackouts': self.total_blackouts,
            'total_brownouts': self.total_brownouts,
            'total_events': total_events,
            'total_shortfall_kwh': self.total_shortfall_kwh,
            'max_concurrent_blackouts': self.max_concurrent_blackouts,
            'worst_shortfall_percentage': self.worst_shortfall_percentage,
            'avg_shortfall_kw': sum(e.shortfall_kw for e in self.events) / max(total_events, 1),
            'avg_duration_min': sum(e.duration_minutes for e in self.events) / max(total_events, 1),
            'blackout_by_station': self.station_blackout_count.copy(),
            'blackout_by_hour': self.hourly_blackout_count.copy(),
            'ongoing_events': len(self.ongoing_events)
        }
    
    def reset(self):
        """Reset for new simulation run"""
        # Close any ongoing events
        for station_id in list(self.ongoing_events.keys()):
            self._close_event(station_id, datetime.now())
        
        self.events.clear()
        self.ongoing_events.clear()
        self.total_blackouts = 0
        self.total_brownouts = 0
        self.total_shortfall_kwh = 0.0
        self.max_concurrent_blackouts = 0
        self.worst_shortfall_percentage = 0.0
        self.hourly_blackout_count.clear()
        self.station_blackout_count.clear()