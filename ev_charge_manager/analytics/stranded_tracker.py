"""
Stranded Vehicle Tracker
Tracks vehicles at risk of stranding and actual stranded events
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum, auto


class VehicleStatus(Enum):
    """Status of vehicle in simulation"""
    DRIVING = auto()
    QUEUED = auto()
    CHARGING = auto()
    STRANDED = auto()
    COMPLETED = auto()


@dataclass
class VehicleState:
    """Current state of a vehicle"""
    vehicle_id: str
    soc: float  # State of charge [0, 1]
    battery_capacity_kwh: float
    consumption_rate_kwh_per_km: float
    current_position_km: float
    destination_km: float
    next_station_distance_km: float
    status: VehicleStatus
    entry_time: datetime
    
    @property
    def remaining_range_km(self) -> float:
        """Calculate remaining range based on current SOC"""
        remaining_energy_kwh = self.soc * self.battery_capacity_kwh
        return remaining_energy_kwh / self.consumption_rate_kwh_per_km
    
    def is_at_risk(self, safety_margin: float = 1.2) -> bool:
        """
        Check if vehicle is at risk of stranding
        safety_margin: multiplier on distance (e.g., 1.2 = 20% buffer)
        """
        return self.remaining_range_km < (self.next_station_distance_km * safety_margin)
    
    def will_strand(self) -> bool:
        """Check if vehicle will definitely strand before next station"""
        return self.remaining_range_km < self.next_station_distance_km


class StrandedVehicleTracker:
    """
    Tracks vehicles and identifies stranded events for optimization
    """
    
    def __init__(self, critical_soc: float = 0.05, safety_margin: float = 1.2):
        self.critical_soc = critical_soc
        self.safety_margin = safety_margin
        
        # Active vehicles
        self.active_vehicles: Dict[str, VehicleState] = {}
        
        # Tracking history
        self.stranded_vehicles: List[Dict] = []  # Detailed records
        self.at_risk_vehicles: Set[str] = set()  # Currently flagged
        
        # Metrics
        self.total_stranded_count: int = 0
        self.total_at_risk_warnings: int = 0
        self.stranding_events_by_location: Dict[float, int] = {}
        self.stranding_by_time_of_day: Dict[int, int] = {}  # Hour -> count
        
        # Early warning log
        self.early_warnings: List[Dict] = []
    
    def register_vehicle(self, vehicle_id: str, initial_soc: float,
                        battery_capacity_kwh: float, 
                        consumption_rate_kwh_per_km: float,
                        current_position_km: float,
                        destination_km: float,
                        next_station_distance_km: float,
                        timestamp: datetime) -> None:
        """Register new vehicle entering simulation"""
        self.active_vehicles[vehicle_id] = VehicleState(
            vehicle_id=vehicle_id,
            soc=initial_soc,
            battery_capacity_kwh=battery_capacity_kwh,
            consumption_rate_kwh_per_km=consumption_rate_kwh_per_km,
            current_position_km=current_position_km,
            destination_km=destination_km,
            next_station_distance_km=next_station_distance_km,
            status=VehicleStatus.DRIVING,
            entry_time=timestamp
        )
    
    def update_vehicle(self, vehicle_id: str, soc: float,
                      current_position_km: float,
                      next_station_distance_km: float,
                      status: VehicleStatus,
                      timestamp: datetime) -> Optional[Dict]:
        """
        Update vehicle state and check for stranding/at-risk conditions
        
        Returns: Alert dict if vehicle newly stranded or at risk, None otherwise
        """
        if vehicle_id not in self.active_vehicles:
            return None
        
        vehicle = self.active_vehicles[vehicle_id]
        old_status = vehicle.status
        
        # Update state
        vehicle.soc = soc
        vehicle.current_position_km = current_position_km
        vehicle.next_station_distance_km = next_station_distance_km
        vehicle.status = status
        
        alert = None
        
        # Check for stranding
        if status == VehicleStatus.STRANDED and old_status != VehicleStatus.STRANDED:
            # New stranding event
            self._record_stranding(vehicle, timestamp)
            alert = {
                'type': 'STRANDING',
                'vehicle_id': vehicle_id,
                'position_km': current_position_km,
                'soc': soc,
                'timestamp': timestamp,
                'severity': 'CRITICAL'
            }
        
        # Check for at-risk condition (early warning)
        elif vehicle.is_at_risk(self.safety_margin) and vehicle_id not in self.at_risk_vehicles:
            self.at_risk_vehicles.add(vehicle_id)
            self.total_at_risk_warnings += 1
            alert = {
                'type': 'AT_RISK',
                'vehicle_id': vehicle_id,
                'position_km': current_position_km,
                'soc': soc,
                'remaining_range_km': vehicle.remaining_range_km,
                'next_station_km': next_station_distance_km,
                'timestamp': timestamp,
                'severity': 'WARNING'
            }
            self.early_warnings.append(alert)
        
        # Remove from at-risk if no longer at risk
        elif not vehicle.is_at_risk(self.safety_margin) and vehicle_id in self.at_risk_vehicles:
            self.at_risk_vehicles.discard(vehicle_id)
        
        # Clean up completed vehicles
        if status == VehicleStatus.COMPLETED:
            del self.active_vehicles[vehicle_id]
        
        return alert
    
    def _record_stranding(self, vehicle: VehicleState, timestamp: datetime) -> None:
        """Record detailed stranding event"""
        record = {
            'vehicle_id': vehicle.vehicle_id,
            'timestamp': timestamp,
            'position_km': vehicle.current_position_km,
            'final_soc': vehicle.soc,
            'battery_capacity_kwh': vehicle.battery_capacity_kwh,
            'nearest_station_distance_km': vehicle.next_station_distance_km,
            'time_of_day': timestamp.hour
        }
        
        self.stranded_vehicles.append(record)
        self.total_stranded_count += 1
        
        # Update location-based stats
        loc_key = round(vehicle.current_position_km / 10) * 10  # Bin by 10km
        self.stranding_events_by_location[loc_key] = \
            self.stranding_events_by_location.get(loc_key, 0) + 1
        
        # Update time-based stats
        hour = timestamp.hour
        self.stranding_by_time_of_day[hour] = \
            self.stranding_by_time_of_day.get(hour, 0) + 1
        
        # Remove from active tracking
        if vehicle.vehicle_id in self.active_vehicles:
            del self.active_vehicles[vehicle.vehicle_id]
        self.at_risk_vehicles.discard(vehicle.vehicle_id)
    
    def get_stranding_rate(self, simulation_duration_hours: float) -> float:
        """Calculate stranded vehicles per day"""
        days = simulation_duration_hours / 24
        return self.total_stranded_count / max(days, 0.001)
    
    def get_metrics(self) -> Dict:
        """Get comprehensive stranding metrics"""
        return {
            'total_stranded': self.total_stranded_count,
            'total_at_risk_warnings': self.total_at_risk_warnings,
            'stranding_rate_per_day': self.get_stranding_rate(24),  # Assuming 24h for rate
            'current_active_vehicles': len(self.active_vehicles),
            'current_at_risk_vehicles': len(self.at_risk_vehicles),
            'stranding_by_location': self.stranding_events_by_location.copy(),
            'stranding_by_hour': self.stranding_by_time_of_day.copy(),
            'early_warning_count': len(self.early_warnings)
        }
    
    def reset(self):
        """Reset for new simulation run"""
        self.active_vehicles.clear()
        self.stranded_vehicles.clear()
        self.at_risk_vehicles.clear()
        self.early_warnings.clear()
        self.total_stranded_count = 0
        self.total_at_risk_warnings = 0
        self.stranding_events_by_location.clear()
        self.stranding_by_time_of_day.clear()