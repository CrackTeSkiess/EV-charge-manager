"""
Highway Module
Road infrastructure managing vehicle movement and charging area coordination.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import math

from Vehicle import Vehicle, VehicleState, DriverBehavior
from ChargingArea import ChargingArea, QueueEntry


@dataclass
class HighwaySegment:
    """
    Subdivision of highway for traffic management.
    """
    start_km: float
    end_km: float
    speed_limit_kmh: float = 130.0
    lanes: int = 2
    gradient_percent: float = 0.0  # Affects consumption
    nearby_station_ids: List[str] = field(default_factory=list)


class Highway:
    """
    Highway infrastructure with charging areas.
    
    Responsibilities:
    - Update vehicle positions (1-minute time steps)
    - Spatial indexing for efficient station proximity queries
    - Handoff vehicles to charging areas when they decide to enter
    - Manage vehicle lifecycle (entry to exit)
    """
    
    # Spatial indexing parameters
    TILE_SIZE_KM = 5.0  # Spatial bucket size for queries
    
    def __init__(
        self,
        highway_id: Optional[str] = None,
        length_km: float = 300.0,
        charging_areas: Optional[List[ChargingArea]] = None,
        segments: Optional[List[HighwaySegment]] = None
    ):
        self.id = highway_id or str(uuid.uuid4())[:8]
        self.length_km = length_km
        
        # Charging infrastructure
        self.charging_areas: Dict[str, ChargingArea] = {}
        self.station_positions: Dict[float, str] = {}  # km -> area_id
        self._register_charging_areas(charging_areas or [])
        
        # Road segmentation (for variable speed limits, etc.)
        self.segments = segments or self._create_default_segments()
        
        # Vehicle management
        self.vehicles: Dict[str, Vehicle] = {}  # All active vehicles
        self.vehicles_by_tile: Dict[int, Set[str]] = defaultdict(set)
        self.vehicles_exiting: List[Vehicle] = []  # Reached destination
        self.vehicles_stranded: List[Vehicle] = []  # Ran out of battery
        
        # Charging area assignments
        self.vehicles_at_station: Dict[str, str] = {}  # vehicle_id -> area_id
        self.vehicles_in_queue: Dict[str, Tuple[str, datetime]] = {}  # vehicle_id -> (area_id, entry_time)
        
        # Statistics
        self.stats = {
            'total_entered': 0,
            'total_exited': 0,
            'total_stranded': 0,
            'total_charging_stops': 0,
            'total_abandonments': 0
        }
        
        self.current_time: Optional[datetime] = None
    
    def _register_charging_areas(self, areas: List[ChargingArea]) -> None:
        """Index charging areas by position."""
        for area in areas:
            if area.location_km > self.length_km:
                raise ValueError(f"Station {area.id} at km {area.location_km} "
                               f"exceeds highway length {self.length_km}")
            
            self.charging_areas[area.id] = area
            self.station_positions[area.location_km] = area.id
    
    def _create_default_segments(self) -> List[HighwaySegment]:
        """Create default uniform segmentation."""
        segment_length = 50.0  # km
        segments = []
        for start in range(0, int(self.length_km), int(segment_length)):
            end = min(start + segment_length, self.length_km)
            segments.append(HighwaySegment(
                start_km=start,
                end_km=end,
                speed_limit_kmh=130.0
            ))
        return segments
    
    # ========================================================================
    # SPATIAL INDEXING
    # ========================================================================
    
    def _get_tile(self, position_km: float) -> int:
        """Convert position to spatial tile index."""
        return int(position_km / self.TILE_SIZE_KM)
    
    def _get_tile_range(self, position_km: float, radius_km: float) -> range:
        """Get tile indices within radius of position."""
        center_tile = self._get_tile(position_km)
        radius_tiles = int(radius_km / self.TILE_SIZE_KM) + 1
        return range(center_tile - radius_tiles, center_tile + radius_tiles + 1)
    
    def _update_vehicle_spatial_index(self, vehicle: Vehicle) -> None:
        """Update which tile a vehicle belongs to."""
        old_tiles = [t for t, ids in self.vehicles_by_tile.items() 
                    if vehicle.id in ids]
        
        new_tile = self._get_tile(vehicle.position_km)
        
        # Remove from old tiles
        for tile in old_tiles:
            self.vehicles_by_tile[tile].discard(vehicle.id)
        
        # Add to new tile
        self.vehicles_by_tile[new_tile].add(vehicle.id)
    
    def get_vehicles_near_position(self, position_km: float, 
                                   radius_km: float = 10.0) -> List[Vehicle]:
        """Get all vehicles within radius of position."""
        nearby = []
        for tile in self._get_tile_range(position_km, radius_km):
            for vehicle_id in self.vehicles_by_tile.get(tile, []):
                vehicle = self.vehicles[vehicle_id]
                if abs(vehicle.position_km - position_km) <= radius_km:
                    nearby.append(vehicle)
        return nearby
    
    def get_stations_in_range(self, position_km: float, 
                             lookahead_km: float = 50.0) -> List[Dict]:
        """Get charging stations ahead within range."""
        stations = []
        for km, area_id in self.station_positions.items():
            if position_km < km <= position_km + lookahead_km:
                area = self.charging_areas[area_id]
                stations.append(area.get_broadcast_state())
        return sorted(stations, key=lambda s: s['location_km'])
    
    def get_segment_at(self, position_km: float) -> HighwaySegment:
        """Get road segment for position."""
        for seg in self.segments:
            if seg.start_km <= position_km < seg.end_km:
                return seg
        return self.segments[-1] if self.segments else HighwaySegment(0, self.length_km)
    
    # ========================================================================
    # VEHICLE LIFECYCLE
    # ========================================================================
    
    def spawn_vehicle(self, vehicle: Optional[Vehicle] = None,
                     position_km: Optional[float] = None,
                     entry_time: Optional[datetime] = None) -> Vehicle:
        """
        Add new vehicle to highway at entry point.
        """
        if vehicle is None:
            # Create default vehicle
            vehicle = Vehicle(
                initial_position_km=position_km or 0.0,
                initial_soc=None  # Randomized
            )
        else:
            vehicle.position_km = position_km or vehicle.position_km
        
        self.vehicles[vehicle.id] = vehicle # pyright: ignore[reportOptionalMemberAccess]
        self._update_vehicle_spatial_index(vehicle)
        self.stats['total_entered'] += 1
        
        vehicle._log_state("entered_highway", f"at km {vehicle.position_km:.1f}") # pyright: ignore[reportOptionalMemberAccess]
        
        return vehicle
    
    def remove_vehicle(self, vehicle_id: str, reason: str = "exit") -> Optional[Vehicle]:
        """
        Remove vehicle from highway (reached destination or failed).
        """
        if vehicle_id not in self.vehicles:
            return None
        
        vehicle = self.vehicles.pop(vehicle_id)
        
        # Clean up spatial index
        for tile in list(self.vehicles_by_tile.keys()):
            self.vehicles_by_tile[tile].discard(vehicle_id)
        
        # Clean up station associations
        if vehicle_id in self.vehicles_at_station:
            del self.vehicles_at_station[vehicle_id]
        if vehicle_id in self.vehicles_in_queue:
            del self.vehicles_in_queue[vehicle_id]
        
        # Track outcome
        if reason == "stranded":
            self.vehicles_stranded.append(vehicle)
            self.stats['total_stranded'] += 1
        else:
            self.vehicles_exiting.append(vehicle)
            self.stats['total_exited'] += 1
        
        return vehicle
    
    # ========================================================================
    # CHARGING AREA COORDINATION
    # ========================================================================
    
    def _check_station_approach(self, vehicle: Vehicle) -> Optional[str]:
        """
        Check if vehicle is at a charging area location and wants to enter.
        Returns area_id if handoff should occur, None otherwise.
        """
        # Only interested vehicles in right states
        if vehicle.state not in [VehicleState.APPROACHING, VehicleState.DECIDING]:
            return None
        
        # Find if at station location (within 0.5 km threshold)
        for km, area_id in self.station_positions.items():
            if abs(vehicle.position_km - km) < 0.5:
                # Vehicle is at station location
                if vehicle.target_station_id == area_id:
                    return area_id
                elif vehicle.needs_charging():
                    # Auto-assign if needs charging but no target
                    return area_id
        
        return None
    
    def _handoff_to_station(self, vehicle_id: str, area_id: str, 
                           timestamp: datetime) -> Dict:
        """
        Transfer vehicle from highway to charging area.
        """
        vehicle = self.vehicles[vehicle_id]
        area = self.charging_areas[area_id]
        
        # Request entry to charging area
        result = area.request_entry(
            vehicle_id=vehicle_id,
            priority_score=vehicle.current_patience,
            power_preference=None  # Could be based on vehicle capability
        )
        
        if result['granted']:
            if result['action'] == 'immediate':
                # Start charging immediately
                self.vehicles_at_station[vehicle_id] = area_id
                vehicle.set_state(VehicleState.CHARGING, timestamp,
                                f"immediate at {area_id}")
                self.stats['total_charging_stops'] += 1
                
                # Start session (simplified - would get energy need from vehicle)
                session_result = area.start_charging(
                    vehicle_id=vehicle_id,
                    energy_needed_kwh=vehicle.battery.capacity_kwh * 
                                     (vehicle.driver.target_soc - vehicle.battery.current_soc),
                    vehicle_soc=vehicle.battery.current_soc
                )
                
                vehicle.start_charging(
                    session_result['charger_id'], # pyright: ignore[reportOptionalSubscript]
                    session_result['power_kw'], # pyright: ignore[reportOptionalSubscript]
                    timestamp
                )
                
                return {'status': 'charging', 'charger': session_result['charger_id']} # pyright: ignore[reportOptionalSubscript]
            
            elif result['action'] == 'queued':
                # Join queue
                self.vehicles_in_queue[vehicle_id] = (area_id, timestamp)
                vehicle.set_state(VehicleState.QUEUED, timestamp, 
                                f"queued at {area_id}")
                vehicle.queue_entry_time = timestamp
                vehicle.target_station_id = area_id
                
                return {
                    'status': 'queued', 
                    'position': result['queue_position'],
                    'estimated_wait': result['estimated_wait']
                }
            
            else:
                # Denied - continue driving
                vehicle.set_state(VehicleState.CRUISING, timestamp, 
                                f"entry denied at {area_id}")
                return {'status': 'denied', 'reason': result.get('reason')}
        else:
            # Entry rejected
            vehicle.set_state(VehicleState.CRUISING, timestamp, 
                            f"rejected at {area_id}")
            return {'status': 'rejected', 'reason': result.get('reason')}
    
    def _update_queued_vehicles(self, timestamp: datetime) -> List[Dict]:
        """
        Update vehicles waiting in charging area queues.
        Check for abandonments and promotions to charging.
        """
        events = []
        
        for vehicle_id, (area_id, entry_time) in list(self.vehicles_in_queue.items()):
            vehicle = self.vehicles[vehicle_id]
            area = self.charging_areas[area_id]
            
            # Update patience
            wait_minutes = (timestamp - entry_time).total_seconds() / 60
            comfort = 0.5  # Would come from area amenities
            vehicle.update_patience(wait_minutes, comfort)
            
            # Check for abandonment decision
            should_abandon = vehicle.driver.decide_to_abort(
                vehicle.current_patience,
                len(area.queue),  # Position approx
                area.estimate_wait_time(),
                alternative_available=len(self.get_stations_in_range(area.location_km)) > 1
            )
            
            if should_abandon:
                # Remove from queue
                area.abandon_queue(vehicle_id, from_waiting_area=True)
                del self.vehicles_in_queue[vehicle_id]
                
                vehicle.abandon_charging(timestamp, "patience depleted")
                self.stats['total_abandonments'] += 1
                
                events.append({
                    'type': 'abandonment',
                    'vehicle_id': vehicle_id,
                    'area_id': area_id,
                    'wait_minutes': wait_minutes
                })
                continue
            
            # Check if promoted to charging (by area step)
            # This is handled in area.step() which calls start_charging
            # We just need to update vehicle state if that happened
            if vehicle.state == VehicleState.CHARGING:
                # Promoted!
                del self.vehicles_in_queue[vehicle_id]
                self.vehicles_at_station[vehicle_id] = area_id
                self.stats['total_charging_stops'] += 1
                
                events.append({
                    'type': 'charging_started',
                    'vehicle_id': vehicle_id,
                    'area_id': area_id,
                    'wait_minutes': wait_minutes
                })
        
        return events
    
    def _update_charging_vehicles(self, timestamp: datetime) -> List[Dict]:
        """
        Update vehicles actively charging.
        Check for completion and release back to highway.
        """
        events = []
        
        for vehicle_id, area_id in list(self.vehicles_at_station.items()):
            vehicle = self.vehicles[vehicle_id]
            area = self.charging_areas[area_id]
            
            # Find which charger has this vehicle
            charger = None
            for c in area.chargers:
                if c.current_session and c.current_session.vehicle_id == vehicle_id:
                    charger = c
                    break
            
            if not charger or vehicle.state != VehicleState.CHARGING:
                # Already released or state mismatch
                continue
            
            # Check if vehicle wants to stop (reached target SOC)
            session = charger.current_session
            if session:
                # Calculate if charged enough - use driver's target_soc directly
                # The 0.95 factor was a workaround for slow final charging, but
                # drivers should decide when they're satisfied with their charge level
                if vehicle.battery.current_soc >= vehicle.driver.target_soc:
                    # Complete charging
                    completed = charger.complete_session()
                    duration_min = (timestamp - session.start_time).total_seconds() / 60
                    
                    vehicle.finish_charging(timestamp, duration_min)
                    del self.vehicles_at_station[vehicle_id]
                    
                    # Back to highway
                    vehicle.set_state(VehicleState.EXITING, timestamp, 
                                    "charging complete")
                    
                    events.append({
                        'type': 'charging_complete',
                        'vehicle_id': vehicle_id,
                        'area_id': area_id,
                        'duration_min': duration_min,
                        'energy_kwh': completed.delivered_energy_kwh
                    })
        
        return events
    
    # ========================================================================
    # MAIN SIMULATION STEP
    # ========================================================================
    
    def step(self, timestamp: datetime, time_step_minutes: float = 1.0) -> Dict:
        """
        Execute one simulation step for entire highway.
        
        1. Update charging areas (process charging, queue management)
        2. Update queued vehicles (patience, abandonments)
        3. Update charging vehicles (completions)
        4. Update driving vehicles (physics, decisions)
        5. Handoffs to stations
        6. Clean up exits/stranded
        """
        self.current_time = timestamp
        step_events = {
            'handoffs': [],
            'abandonments': [],
            'charging_completions': [],
            'new_charging': [],
            'exits': [],
            'strandings': []
        }
        
        # 1. Update all charging areas
        for area in self.charging_areas.values():
            area.step(timestamp, time_step_minutes)
        
        # 2. Update queued vehicles
        queue_events = self._update_queued_vehicles(timestamp)
        for evt in queue_events:
            if evt['type'] == 'abandonment':
                step_events['abandonments'].append(evt)
            elif evt['type'] == 'charging_started':
                step_events['new_charging'].append(evt)
        
        # 3. Update actively charging vehicles
        charging_events = self._update_charging_vehicles(timestamp)
        step_events['charging_completions'].extend(charging_events)
        
        # 4. Update driving vehicles
        vehicles_to_remove = []
        
        for vehicle_id, vehicle in self.vehicles.items():
            # Skip if at station
            if vehicle_id in self.vehicles_at_station or \
               vehicle_id in self.vehicles_in_queue:
                continue
            
            # Get environment for this vehicle
            segment = self.get_segment_at(vehicle.position_km)
            # Lookahead should cover vehicle's estimated range to plan charging
            vehicle_range = vehicle.get_range_estimate()
            lookahead = max(100.0, min(vehicle_range * 1.5, 300.0))  # 100-300km lookahead
            upcoming = self.get_stations_in_range(vehicle.position_km, lookahead)
            
            # Traffic speed (simplified - average of nearby vehicles)
            nearby = self.get_vehicles_near_position(vehicle.position_km, 2.0)
            avg_speed = sum(v.speed_kmh for v in nearby) / max(1, len(nearby))
            traffic_speed = avg_speed if nearby else None
            
            # Execute vehicle step
            result = vehicle.step(
                timestamp=timestamp,
                time_step_minutes=time_step_minutes,
                environment={
                    'speed_limit_kmh': segment.speed_limit_kmh,
                    'traffic_speed_kmh': traffic_speed,
                    'upcoming_stations': upcoming,
                    'station_comfort': 0.5  # Would be dynamic
                }
            )
            
            # Update spatial index
            self._update_vehicle_spatial_index(vehicle)
            
            # Check for handoff to station
            if result.get('needs_decision') or vehicle.state in [
                VehicleState.APPROACHING, VehicleState.DECIDING
            ]:
                target_area = self._check_station_approach(vehicle)
                if target_area:
                    handoff_result = self._handoff_to_station(
                        vehicle_id, target_area, timestamp
                    )
                    step_events['handoffs'].append({
                        'vehicle_id': vehicle_id,
                        'area_id': target_area,
                        'result': handoff_result
                    })
            
            # Check for exit or stranding
            if vehicle.position_km >= self.length_km:
                vehicles_to_remove.append((vehicle_id, "exit"))
            elif result.get('stranded'):
                vehicles_to_remove.append((vehicle_id, "stranded"))
        
        # 5. Remove exited/stranded vehicles
        for vid, reason in vehicles_to_remove:
            self.remove_vehicle(vid, reason)
            if reason == "exit":
                step_events['exits'].append({'vehicle_id': vid})
            else:
                step_events['strandings'].append({'vehicle_id': vid})
        
        return step_events
    
    # ========================================================================
    # BATCH OPERATIONS
    # ========================================================================
    
    def spawn_traffic(self, count: int, time: datetime, 
                     distribution: Optional[Callable] = None) -> List[Vehicle]:
        """
        Spawn multiple vehicles with distributed properties.
        """
        spawned = []
        for i in range(count):
            # Distribute entry points (mostly at start, some along highway)
            import random
            if random.random() < 0.8:
                pos = 0.0
            else:
                pos = random.uniform(0, self.length_km * 0.5)
            
            # Create varied driver behaviors
            behavior_types = ['conservative', 'balanced', 'aggressive', 'range_anxious']
            weights = [0.25, 0.40, 0.25, 0.10]
            btype = random.choices(behavior_types, weights)[0]
            
            behavior = DriverBehavior(
                behavior_type=btype,
                patience_base=random.uniform(0.3, 0.9),
                risk_tolerance=random.uniform(0.1, 0.9)
            )
            
            vehicle = Vehicle(
                battery_capacity_kwh=random.choice([50, 60, 75, 90, 100]),
                initial_soc=None,  # Randomized
                driver_behavior=behavior,
                initial_position_km=pos
            )
            
            self.spawn_vehicle(vehicle, entry_time=time)
            spawned.append(vehicle)
        
        return spawned
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def get_highway_state(self) -> Dict:
        """Current state summary."""
        driving = [v for v in self.vehicles.values() 
                  if v.id not in self.vehicles_at_station 
                  and v.id not in self.vehicles_in_queue]
        
        return {
            'highway_id': self.id,
            'length_km': self.length_km,
            'current_time': self.current_time,
            'active_vehicles': len(self.vehicles),
            'driving': len(driving),
            'charging': len(self.vehicles_at_station),
            'queued': len(self.vehicles_in_queue),
            'charging_areas': {
                aid: area.get_broadcast_state() 
                for aid, area in self.charging_areas.items()
            },
            'stats': self.stats.copy()
        }
    
    def get_vehicle_positions(self) -> List[Dict]:
        """Get all vehicle positions for visualization."""
        return [
            {
                'id': vid,
                'position_km': v.position_km,
                'speed_kmh': v.speed_kmh,
                'soc': v.battery.current_soc,
                'state': v.state.name
            }
            for vid, v in self.vehicles.items()
        ]
    
    def __repr__(self) -> str:
        return (f"Highway({self.id}, {self.length_km}km, "
                f"stations={len(self.charging_areas)}, "
                f"vehicles={len(self.vehicles)})")