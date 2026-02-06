"""
Highway Module - WITH DEBUGGING
Road infrastructure managing vehicle movement and charging area coordination.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Callable, TYPE_CHECKING, Any
from collections import defaultdict
from datetime import datetime, timedelta
import math

from Vehicle import Vehicle, VehicleState, DriverBehavior
from ChargingArea import ChargingArea, QueueEntry

if TYPE_CHECKING:
    from VehicleTracker import VehicleTracker


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
    - Report state transitions to VehicleTracker
    """

    # Spatial indexing parameters
    TILE_SIZE_KM = 5.0  # Spatial bucket size for queries

    def __init__(
        self,
        highway_id: Optional[str] = None,
        length_km: float = 300.0,
        charging_areas: Optional[List[ChargingArea]] = None,
        segments: Optional[List[HighwaySegment]] = None,
        vehicle_tracker: Optional[VehicleTracker] = None
    ):
        self.id = highway_id or str(uuid.uuid4())[:8]
        self.length_km = length_km

        # Vehicle tracker (set externally or None)
        self.tracker: Optional[VehicleTracker] = vehicle_tracker

        # Charging infrastructure
        self.charging_areas: Dict[str, ChargingArea] = {}
        self.station_positions: Dict[float, str] = {}  # km -> area_id
        self._register_charging_areas(charging_areas or [])

        # Road segmentation (for variable speed limits, etc.)
        self.segments = segments or self._create_default_segments()

        # Vehicle management - Highway still owns Vehicle objects for physics
        self.vehicles: Dict[str, Vehicle] = {}  # All active vehicles
        self.vehicles_by_tile: Dict[int, Set[str]] = defaultdict(set)

        # Legacy lists - now just for reference, tracker is source of truth
        self.vehicles_exiting: List[Vehicle] = []
        self.vehicles_stranded: List[Vehicle] = []

        # Charging area assignments - synced with tracker
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
        
        # DEBUG: Track all events for analysis
        self.event_log: List[Dict[str, Any]] = []

    def _log_highway_event(self, event_type: str, details: str, 
                          vehicle_id: Optional[str] = None,
                          extra_data: Optional[Dict] = None) -> None:
        """Log highway-level events for debugging."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'vehicle_id': vehicle_id,
            'sim_time': self.current_time.isoformat() if self.current_time else None
        }
        if extra_data:
            entry.update(extra_data)
        self.event_log.append(entry)
        
        # Print for real-time debugging
        prefix = f"[HIGHWAY {self.id}]"
        if vehicle_id:
            prefix += f" [Vehicle {vehicle_id}]"
        #print(f"{prefix} {event_type}: {details}")

    def set_tracker(self, tracker: VehicleTracker) -> None:
        """Set the vehicle tracker after construction."""
        self.tracker = tracker

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
                vehicle = self.vehicles.get(vehicle_id)
                if vehicle and abs(vehicle.position_km - position_km) <= radius_km:
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

        self.vehicles[vehicle.id] = vehicle
        self._update_vehicle_spatial_index(vehicle)
        self.stats['total_entered'] += 1

        vehicle._log_state("entered_highway", f"at km {vehicle.position_km:.1f}")
        self._log_highway_event("VEHICLE_ENTER", 
                               f"Spawned at km {vehicle.position_km:.1f}, "
                               f"SOC={vehicle.battery.current_soc:.1%}",
                               vehicle.id)

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
            self._log_highway_event("VEHICLE_STRANDED", 
                                   f"Stranded at km {vehicle.position_km:.1f}, "
                                   f"SOC={vehicle.battery.current_soc:.1%}",
                                   vehicle_id)
            # Print debug report immediately
            print("\n" + vehicle.get_debug_report())
        else:
            self.vehicles_exiting.append(vehicle)
            self.stats['total_exited'] += 1
            self._log_highway_event("VEHICLE_EXIT", 
                                   f"Exited at km {vehicle.position_km:.1f}",
                                   vehicle_id)

        return vehicle

    # ========================================================================
    # CHARGING AREA COORDINATION
    # ========================================================================

    def _check_station_approach(self, vehicle: Vehicle) -> Optional[str]:
        """
        Check if vehicle is at a charging area location and wants to enter.
        Returns area_id if handoff should occur, None otherwise.

        A rational driver MUST stop if they cannot reach the next station.
        """
        # Only interested vehicles in right states
        if vehicle.state not in [VehicleState.APPROACHING, VehicleState.DECIDING]:
            return None

        # Get sorted list of station positions
        sorted_stations = sorted(self.station_positions.keys())

        # Find if at station location (within detection threshold)
        # INCREASED from 0.5 km to 3.0 km to prevent vehicles from jumping over
        # At 120 km/h, vehicles move 2 km per minute, so 3 km ensures detection
        DETECTION_THRESHOLD_KM = 3.0
        
        for km, area_id in self.station_positions.items():
            if abs(vehicle.position_km - km) < DETECTION_THRESHOLD_KM:
                # Vehicle is at station location
                # Find the next station after this one
                next_station_km = None
                for station_km in sorted_stations:
                    if station_km > km + 0.5:  # Must be meaningfully ahead
                        next_station_km = station_km
                        break

                # Check if vehicle MUST stop here (can't reach next station or highway end)
                must_stop = vehicle.must_stop_at_station(km, next_station_km, self.length_km)

                if vehicle.target_station_id == area_id:
                    self._log_highway_event("STATION_APPROACH", 
                                           f"Target station {area_id} at km {km}",
                                           vehicle.id,
                                           {'must_stop': must_stop})
                    return area_id
                elif must_stop:
                    # MUST stop - rational driver won't skip and strand
                    self._log_highway_event("STATION_MUST_STOP", 
                                           f"Forced stop at {area_id}, km {km}",
                                           vehicle.id)
                    return area_id
                elif vehicle.needs_charging(next_station_km):
                    # Wants to charge based on comfort/range anxiety
                    self._log_highway_event("STATION_WANTS_CHARGE", 
                                           f"Optional stop at {area_id}, km {km}",
                                           vehicle.id)
                    return area_id

        return None

    def _handoff_to_station(self, vehicle_id: str, area_id: str,
                           timestamp: datetime) -> Dict:
        """
        Transfer vehicle from highway to charging area.

        IMPORTANT: If a vehicle MUST stop (can't reach next station), they cannot
        be rejected. They will join the queue even if over capacity (emergency overflow).
        """
        vehicle = self.vehicles[vehicle_id]
        area = self.charging_areas[area_id]

        # Check if this is a must-stop situation
        sorted_stations = sorted(self.station_positions.keys())
        current_station_km = area.location_km
        next_station_km = None
        for station_km in sorted_stations:
            if station_km > current_station_km + 0.5:
                next_station_km = station_km
                break

        must_stop = vehicle.must_stop_at_station(current_station_km, next_station_km, self.length_km)

        # Request entry to charging area
        result = area.request_entry(
            vehicle_id=vehicle_id,
            priority_score=vehicle.current_patience,
            power_preference=None  # Could be based on vehicle capability
        )

        self._log_highway_event("HANDOFF_REQUEST", 
                               f"Area {area_id}, granted={result['granted']}, "
                               f"action={result.get('action')}, must_stop={must_stop}",
                               vehicle_id)

        if result['granted']:
            if result['action'] == 'immediate':
                # Start charging immediately
                self.vehicles_at_station[vehicle_id] = area_id
                vehicle.set_state(VehicleState.CHARGING, timestamp,
                                f"immediate at {area_id}")
                self.stats['total_charging_stops'] += 1

                # Notify tracker
                if self.tracker:
                    self.tracker.transition_to_charging(
                        vehicle_id, timestamp, area_id, "immediate entry"
                    )

                # Start session (simplified - would get energy need from vehicle)
                # Calculate energy needed - SAFETY FIRST
                # Must have enough to reach next station or highway end
                sorted_stations = sorted(self.station_positions.keys())
                current_station_km = area.location_km
                next_station_km = None
                for station_km in sorted_stations:
                    if station_km > current_station_km + 0.5:
                        next_station_km = station_km
                        break
                
                # Calculate energy needed for safety (to reach next destination)
                if next_station_km is not None:
                    distance_to_next = abs(next_station_km - current_station_km)
                else:
                    distance_to_next = abs(self.length_km - current_station_km)
                
                # Add 30km buffer for safety
                safety_distance = distance_to_next + 30.0
                highway_speed = vehicle.driver.speed_preference_kmh
                consumption_rate = vehicle.calculate_consumption_rate(highway_speed)
                safety_energy_needed = consumption_rate * safety_distance
                
                # Current energy needed to reach target SOC
                target_energy_needed = vehicle.battery.capacity_kwh * (vehicle.driver.target_soc - vehicle.battery.current_soc)
                
                # Use the MAXIMUM of safety requirement and driver preference
                energy_needed_kwh = max(safety_energy_needed, target_energy_needed)
                
                # Cap at battery capacity
                max_energy = vehicle.battery.capacity_kwh * (1.0 - vehicle.battery.current_soc)
                energy_needed_kwh = min(energy_needed_kwh, max_energy)

                # Start session (simplified - would get energy need from vehicle)
                session_result = area.start_charging(
                    vehicle_id=vehicle_id,
                    energy_needed_kwh=energy_needed_kwh,
                    vehicle_soc=vehicle.battery.current_soc
                )                
                '''
                session_result = area.start_charging(
                    vehicle_id=vehicle_id,
                    energy_needed_kwh=vehicle.battery.capacity_kwh *
                                     (vehicle.driver.target_soc - vehicle.battery.current_soc),
                    vehicle_soc=vehicle.battery.current_soc
                )
                '''

                if session_result:
                    vehicle.start_charging(
                        session_result['charger_id'],
                        session_result['power_kw'],
                        timestamp
                    )
                    self._log_highway_event("CHARGING_START", 
                                           f"Charger {session_result['charger_id']}, "
                                           f"power={session_result['power_kw']}kW",
                                           vehicle_id)
                    return {'status': 'charging', 'charger': session_result['charger_id']}
                else:
                    return {'status': 'charging', 'charger': None}

            elif result['action'] == 'queued':
                # Join queue
                self.vehicles_in_queue[vehicle_id] = (area_id, timestamp)
                vehicle.set_state(VehicleState.QUEUED, timestamp,
                                f"queued at {area_id}")
                vehicle.queue_entry_time = timestamp
                vehicle.target_station_id = area_id

                # Notify tracker
                if self.tracker:
                    self.tracker.transition_to_queued(
                        vehicle_id, timestamp, area_id, f"queue position {result['queue_position']}"
                    )

                self._log_highway_event("QUEUE_JOIN", 
                                       f"Position {result['queue_position']}, "
                                       f"wait={result['estimated_wait']:.1f}min",
                                       vehicle_id)

                return {
                    'status': 'queued',
                    'position': result['queue_position'],
                    'estimated_wait': result['estimated_wait']
                }

            else:
                # Denied - but check if must_stop
                if must_stop:
                    # EMERGENCY: Vehicle cannot be turned away - force into overflow queue
                    self.vehicles_in_queue[vehicle_id] = (area_id, timestamp)
                    vehicle.set_state(VehicleState.QUEUED, timestamp,
                                    f"emergency queued at {area_id} (overflow)")
                    vehicle.queue_entry_time = timestamp
                    vehicle.target_station_id = area_id
                    area.waiting_occupied += 1  # Track overflow

                    if self.tracker:
                        self.tracker.transition_to_queued(
                            vehicle_id, timestamp, area_id, "emergency overflow queue"
                        )

                    self._log_highway_event("QUEUE_EMERGENCY", 
                                           "Forced into overflow queue",
                                           vehicle_id)

                    return {
                        'status': 'queued',
                        'position': len(area.queue) + 1,
                        'estimated_wait': area.estimate_wait_time(),
                        'overflow': True
                    }
                else:
                    # Can be denied - continue driving
                    vehicle.set_state(VehicleState.CRUISING, timestamp,
                                    f"entry denied at {area_id}")

                    if self.tracker:
                        self.tracker.transition_to_driving(
                            vehicle_id, timestamp, f"entry denied at {area_id}"
                        )

                    return {'status': 'denied', 'reason': result.get('reason')}
        else:
            # Entry rejected - but check if must_stop
            if must_stop:
                # EMERGENCY: Vehicle cannot be turned away - force into overflow queue
                self.vehicles_in_queue[vehicle_id] = (area_id, timestamp)
                vehicle.set_state(VehicleState.QUEUED, timestamp,
                                f"emergency queued at {area_id} (overflow)")
                vehicle.queue_entry_time = timestamp
                vehicle.target_station_id = area_id
                area.waiting_occupied += 1  # Track overflow

                if self.tracker:
                    self.tracker.transition_to_queued(
                        vehicle_id, timestamp, area_id, "emergency overflow queue"
                    )

                self._log_highway_event("QUEUE_EMERGENCY", 
                                       "Forced into overflow queue (rejected)",
                                       vehicle_id)

                return {
                    'status': 'queued',
                    'position': len(area.queue) + 1,
                    'estimated_wait': area.estimate_wait_time(),
                    'overflow': True
                }
            else:
                # Can be rejected - continue driving
                vehicle.set_state(VehicleState.CRUISING, timestamp,
                                f"rejected at {area_id}")

                if self.tracker:
                    self.tracker.transition_to_driving(
                        vehicle_id, timestamp, f"rejected at {area_id}"
                    )

                return {'status': 'rejected', 'reason': result.get('reason')}

    def _update_queued_vehicles(self, timestamp: datetime) -> List[Dict]:
        """
        Update vehicles waiting in charging area queues.
        Check for abandonments and promotions to charging.

        CRITICAL FIX: A rational driver will NEVER abandon if they cannot reach 
        the next station. This is now enforced with multiple safety checks.
        """
        events = []

        # Get sorted station positions for next-station lookups
        sorted_stations = sorted(self.station_positions.keys())

        for vehicle_id, (area_id, entry_time) in list(self.vehicles_in_queue.items()):
            vehicle = self.vehicles.get(vehicle_id)
            if not vehicle:
                # Vehicle no longer exists - clean up
                del self.vehicles_in_queue[vehicle_id]
                continue

            area = self.charging_areas[area_id]
            current_station_km = area.location_km

            # Find the next station after this one
            next_station_km = None
            for station_km in sorted_stations:
                if station_km > current_station_km + 0.5:
                    next_station_km = station_km
                    break

            # === CRITICAL SAFETY CHECKS ===
            # Check 1: If battery is critical (< 20%), CANNOT abandon regardless of range calculation
            if vehicle.battery.current_soc < 0.20:
                vehicle._log_event("QUEUE_SAFETY_BLOCK", 
                                 f"Critical battery {vehicle.battery.current_soc:.1%}, "
                                 f"forced to stay in queue")
                # Still update patience for logging, but cap it to prevent abandonment
                wait_minutes = (timestamp - entry_time).total_seconds() / 60
                vehicle.update_patience(wait_minutes, 0.5)
                vehicle.current_patience = max(vehicle.current_patience, 0.3)  # Force minimum patience
                continue  # Skip all abandonment logic

            # Check 2: Can the vehicle reach the next station (or highway end) if it abandons?
            can_reach_next = False
            distance_to_target = 1000
            
            if next_station_km is not None:
                # Distance from current position to next station
                distance_to_target = abs(next_station_km - vehicle.position_km)
                can_reach_next = vehicle.can_physically_reach(next_station_km, at_highway_speed=True)
            else:
                # No next station - check if can reach highway end
                distance_to_target = abs(self.length_km - vehicle.position_km)
                can_reach_next = vehicle.can_physically_reach(self.length_km, at_highway_speed=True)

            # Check 3: Additional safety margin - if range is close to distance, don't risk it
            if can_reach_next:
                # Get the estimated range
                consumption = vehicle._get_conservative_consumption(vehicle.driver.speed_preference_kmh)
                estimated_range = vehicle.battery.estimate_range_km(consumption)
                
                # If range is less than 1.5x the distance, it's too risky
                safety_margin = 1.5
                if estimated_range < distance_to_target * safety_margin:
                    vehicle._log_event("QUEUE_SAFETY_MARGIN", 
                                     f"Range {estimated_range:.1f}km too close to "
                                     f"distance {distance_to_target:.1f}km "
                                     f"(need {safety_margin}x, have {estimated_range/max(1,distance_to_target):.2f}x)")
                    can_reach_next = False

            # Update patience (for tracking purposes)
            wait_minutes = (timestamp - entry_time).total_seconds() / 60
            comfort = 0.5  # Would come from area amenities
            vehicle.update_patience(wait_minutes, comfort)

            # FINAL CHECK: Only consider abandonment if ALL safety checks pass
            if not can_reach_next:
                # Cannot abandon - would strand. Vehicle must stay in queue.
                vehicle._log_event("QUEUE_BLOCKED", 
                                 f"Cannot safely reach next station at "
                                 f"{next_station_km if next_station_km else 'END'} "
                                 f"(distance {distance_to_target:.1f}km)")
                # Force patience to stay above abandonment threshold
                vehicle.current_patience = max(vehicle.current_patience, 0.3)
                continue  # Skip abandonment - stay in queue

            # Only if we passed all safety checks, check for abandonment decision
            alternative_stations = self.get_stations_in_range(current_station_km)
            alternative_available = len(alternative_stations) > 1

            should_abandon = vehicle.driver.decide_to_abort(
                vehicle.current_patience,
                len(area.queue),  # Position approx
                area.estimate_wait_time(),
                alternative_available=alternative_available
            )

            if should_abandon:
                # Remove from queue
                area.abandon_queue(vehicle_id, from_waiting_area=True)
                del self.vehicles_in_queue[vehicle_id]

                vehicle.abandon_charging(timestamp, "patience depleted")
                self.stats['total_abandonments'] += 1

                self._log_highway_event("QUEUE_ABANDON", 
                                       f"Abandoned after {wait_minutes:.1f}min, "
                                       f"patience={vehicle.current_patience:.2f}, "
                                       f"SOC={vehicle.battery.current_soc:.1%}, "
                                       f"can_reach_next={can_reach_next}",
                                       vehicle_id)

                # Notify tracker
                if self.tracker:
                    self.tracker.record_abandonment(
                        vehicle_id, timestamp, area_id, "patience depleted"
                    )

                events.append({
                    'type': 'abandonment',
                    'vehicle_id': vehicle_id,
                    'area_id': area_id,
                    'wait_minutes': wait_minutes
                })
                continue

            # Check if promoted to charging (by area step)
            if vehicle.state == VehicleState.CHARGING:
                # Promoted!
                del self.vehicles_in_queue[vehicle_id]
                self.vehicles_at_station[vehicle_id] = area_id
                self.stats['total_charging_stops'] += 1

                self._log_highway_event("QUEUE_PROMOTED", 
                                       "Promoted to charging",
                                       vehicle_id)

                # Notify tracker
                if self.tracker:
                    self.tracker.transition_to_charging(
                        vehicle_id, timestamp, area_id, "promoted from queue"
                    )

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
        
        CRITICAL FIX: Vehicle will NOT exit until it has enough charge to reach
        the next station or the highway end.
        """
        events = []
        
        # Get sorted station positions for next-station lookups
        sorted_stations = sorted(self.station_positions.keys())

        for vehicle_id, area_id in list(self.vehicles_at_station.items()):
            vehicle = self.vehicles.get(vehicle_id)
            if not vehicle:
                # Vehicle no longer exists - clean up
                del self.vehicles_at_station[vehicle_id]
                continue

            area = self.charging_areas[area_id]
            current_station_km = area.location_km
            
            # Find the next station after this one (or None if this is the last)
            next_station_km = None
            for station_km in sorted_stations:
                if station_km > current_station_km + 0.5:
                    next_station_km = station_km
                    break

            # Find which charger has this vehicle
            charger = None
            for c in area.chargers:
                if c.current_session and c.current_session.vehicle_id == vehicle_id:
                    charger = c
                    break

            if not charger or vehicle.state != VehicleState.CHARGING:
                # Already released or state mismatch
                continue

            # === CRITICAL: Update vehicle battery with energy delivered ===
            # Calculate energy delivered in this time step (1 minute default)
            time_step_minutes = 1.0  # Or get from config
            efficiency = charger.efficiency_curve(0.5)
            energy_delivered = (charger.power_kw * efficiency) * (time_step_minutes / 60)
            
            # Actually charge the vehicle's battery!
            vehicle.battery.charge(energy_delivered)
            
            # Update tracker with new SOC
            if self.tracker:
                self.tracker.update_vehicle_soc(vehicle_id, vehicle.battery.current_soc)

            # Check if vehicle wants to stop (reached target SOC)
            session = charger.current_session
            if session:
                # === CRITICAL SAFETY CHECK ===
                # Vehicle must have enough charge to reach next station OR highway end
                can_exit_safely = False
                
                if next_station_km is not None:
                    # Must be able to reach next station (at highway speed)
                    can_exit_safely = vehicle.can_physically_reach(next_station_km, at_highway_speed=True)
                else:
                    # No next station - must be able to reach highway end
                    can_exit_safely = vehicle.can_physically_reach(self.length_km, at_highway_speed=True)
                
                # Vehicle can only complete charging if:
                # 1. It has reached target SOC, AND
                # 2. It can safely reach the next destination (next station or highway end)
                reached_target_soc = vehicle.battery.current_soc >= vehicle.driver.target_soc
                
                if reached_target_soc and can_exit_safely:
                    # Complete charging - safe to exit
                    completed = charger.complete_session()
                    duration_min = (timestamp - session.start_time).total_seconds() / 60

                    vehicle.finish_charging(timestamp, duration_min)
                    del self.vehicles_at_station[vehicle_id]

                    self._log_highway_event("CHARGING_COMPLETE", 
                                           f"Duration={duration_min:.1f}min, "
                                           f"energy={completed.delivered_energy_kwh:.1f}kWh, "
                                           f"SOC={vehicle.battery.current_soc:.1%}, "
                                           f"can_reach_next={can_exit_safely}",
                                           vehicle_id)

                    # Notify tracker
                    if self.tracker:
                        self.tracker.transition_to_exiting(
                            vehicle_id, timestamp,
                            energy_received_kwh=completed.delivered_energy_kwh,
                            reason="charging complete"
                        )

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
                elif reached_target_soc and not can_exit_safely:
                    # Vehicle reached target SOC but CANNOT safely exit
                    # Continue charging beyond target until safe to exit
                    # This overrides the driver's normal target SOC preference
                    # for safety reasons
                    vehicle._log_event("CHARGING_CONTINUE", 
                                     f"Target SOC reached but cannot reach next station. "
                                     f"Continuing to charge...")
                    pass  # Continue charging - do not release vehicle
                # else: Haven't reached target SOC yet, keep charging
                
                # REMOVED: Old logic that completed based only on requested_energy_kwh
                # if session.delivered_energy_kwh >= session.requested_energy_kwh:
                #     # Only complete if safety check passes
                #     pass            

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
        
        # FIRST: Check for vehicles that passed stations without stopping
        # This catches vehicles that might have "jumped over" the detection zone
        for vehicle_id, vehicle in list(self.vehicles.items()):
            if vehicle_id in self.vehicles_at_station or \
               vehicle_id in self.vehicles_in_queue:
                continue
                
            # Check if vehicle just passed a station (within last step)
            for station_km, area_id in self.station_positions.items():
                # Vehicle is just past station (within 2 km behind)
                if 0 < vehicle.position_km - station_km <= 2.0:
                    # Check if this is a must-stop situation
                    sorted_stations = sorted(self.station_positions.keys())
                    next_station_km = None
                    for sk in sorted_stations:
                        if sk > station_km + 0.5:
                            next_station_km = sk
                            break
                    
                    # If must stop but didn't, force them back!
                    if vehicle.must_stop_at_station(station_km, next_station_km, self.length_km):
                        self._log_highway_event("MISSED_STATION", 
                                               f"Forced back to station at km {station_km}",
                                               vehicle_id)
                        vehicle.position_km = station_km  # Move back to station
                        vehicle.set_state(VehicleState.APPROACHING, timestamp, f"forced stop at {area_id}")
                        handoff_result = self._handoff_to_station(vehicle_id, area_id, timestamp)
                        step_events['handoffs'].append({
                            'vehicle_id': vehicle_id,
                            'area_id': area_id,
                            'result': handoff_result,
                            'forced': True
                        })
                        break  # Only handle one station per vehicle

        for vehicle_id, vehicle in list(self.vehicles.items()):
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

            # Execute vehicle step with full environment info
            result = vehicle.step(
                timestamp=timestamp,
                time_step_minutes=time_step_minutes,
                environment={
                    'speed_limit_kmh': segment.speed_limit_kmh,
                    'traffic_speed_kmh': traffic_speed,
                    'upcoming_stations': upcoming,
                    'station_comfort': 0.5,  # Would be dynamic
                    'highway_end_km': self.length_km  # For must-stop calculations
                }
            )

            # Update spatial index
            self._update_vehicle_spatial_index(vehicle)

            # Update tracker with position/SOC
            if self.tracker:
                self.tracker.update_vehicle_position(vehicle_id, vehicle.position_km)
                self.tracker.update_vehicle_soc(vehicle_id, vehicle.battery.current_soc)

            # Check for handoff to station
            if result.get('needs_decision') or vehicle.state in [
                VehicleState.APPROACHING, VehicleState.DECIDING
            ]:
                # Notify tracker of approaching
                if self.tracker and vehicle.state == VehicleState.APPROACHING:
                    self.tracker.transition_to_approaching(
                        vehicle_id, timestamp,
                        vehicle.target_station_id or "unknown",
                        "approaching station"
                    )

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
            vehicle = self.vehicles.get(vid)
            if vehicle:
                # Notify tracker BEFORE removing
                if self.tracker:
                    if reason == "exit":
                        self.tracker.transition_to_completed(
                            vid, timestamp,
                            final_position_km=vehicle.position_km,
                            final_soc=vehicle.battery.current_soc,
                            reason="reached highway end"
                        )
                    else:
                        self.tracker.transition_to_stranded(
                            vid, timestamp,
                            final_position_km=vehicle.position_km,
                            reason="battery depleted"
                        )

            self.remove_vehicle(vid, reason)
            if reason == "exit":
                step_events['exits'].append({'vehicle_id': vid})
            else:
                step_events['strandings'].append({'vehicle_id': vid})

        return step_events

    def run_simulation_with_debug(self, duration_hours: float = 8.0, 
                                 spawn_rate_per_hour: float = 60.0) -> None:
        """
        Run simulation and automatically generate debug plots at the end.
        """
        from datetime import datetime, timedelta
        
        start_time = datetime(2024, 1, 15, 8, 0)  # 8 AM
        end_time = start_time + timedelta(hours=duration_hours)
        current_time = start_time
        
        time_step = timedelta(minutes=1)
        
        print(f"Starting simulation from {start_time} to {end_time}")
        print(f"Highway length: {self.length_km} km")
        print(f"Stations at: {sorted(self.station_positions.keys())} km")
        print("-" * 70)
        
        step_count = 0
        while current_time < end_time:
            # Spawn new vehicles
            import random
            if random.random() < (spawn_rate_per_hour / 60.0):
                self.spawn_traffic(1, current_time)
            
            # Step simulation
            self.step(current_time, time_step_minutes=1.0)
            
            # Progress report every hour
            step_count += 1
            if step_count % 60 == 0:
                hours_elapsed = step_count // 60
                state = self.get_highway_state()
                print(f"[{hours_elapsed}h] Active: {state['active_vehicles']}, "
                      f"Stranded: {self.stats['total_stranded']}, "
                      f"Exited: {self.stats['total_exited']}")
            
            current_time += time_step
        
        print("-" * 70)
        print("Simulation complete!")
        print(f"Total entered: {self.stats['total_entered']}")
        print(f"Total exited: {self.stats['total_exited']}")
        print(f"Total stranded: {self.stats['total_stranded']}")
        
        # Generate debug visualizations
        if self.vehicles_stranded:
            print("\nGenerating stranded vehicle analysis...")
            self.plot_stranded_vehicle_trajectories(max_vehicles=10)
            self.plot_stranding_summary_chart()
            print(self.generate_stranding_report())
        else:
            print("\nNo vehicles stranded - no debug plots generated")

    # ========================================================================
    # DEBUGGING: Generate comprehensive report and plots for stranded vehicles
    # ========================================================================
    
    def generate_stranding_report(self) -> str:
        """Generate a comprehensive report of all stranded vehicles."""
        lines = [
            f"\n{'='*80}",
            f"STRANDING ANALYSIS REPORT",
            f"{'='*80}",
            f"Total vehicles entered: {self.stats['total_entered']}",
            f"Total vehicles exited: {self.stats['total_exited']}",
            f"Total vehicles stranded: {self.stats['total_stranded']}",
            f"Stranding rate: {100*self.stats['total_stranded']/max(1,self.stats['total_entered']):.1f}%",
            f"\nDetailed reports for each stranded vehicle:",
            f"{'-'*80}"
        ]
        
        for vehicle in self.vehicles_stranded:
            lines.append(vehicle.get_debug_report())
        
        # Summary of common patterns
        if self.vehicles_stranded:
            lines.extend(self._analyze_stranding_patterns())
        
        lines.append(f"{'='*80}\n")
        return "\n".join(lines)
    
    def _analyze_stranding_patterns(self) -> List[str]:
        """Analyze common patterns among stranded vehicles."""
        analysis = [
            f"\n{'#'*80}",
            "PATTERN ANALYSIS:",
            f"{'#'*80}"
        ]
        
        # Group by driver type
        driver_types = {}
        for v in self.vehicles_stranded:
            dtype = v.driver.behavior_type
            driver_types[dtype] = driver_types.get(dtype, 0) + 1
        
        analysis.append("\nStranding by driver type:")
        for dtype, count in sorted(driver_types.items(), key=lambda x: -x[1]):
            analysis.append(f"  {dtype}: {count}")
        
        # Check for abandonment correlation
        abandon_count = sum(1 for v in self.vehicles_stranded 
                          if any(e['event_type'] == 'ABANDON' for e in v.detailed_history))
        analysis.append(f"\nVehicles that abandoned queue before stranding: {abandon_count}")
        
        # Check charging stop correlation
        avg_charging_stops = sum(v.charging_stops for v in self.vehicles_stranded) / len(self.vehicles_stranded)
        analysis.append(f"Average charging stops before stranding: {avg_charging_stops:.1f}")
        
        # Position analysis
        positions = [v.position_km for v in self.vehicles_stranded]
        avg_pos = sum(positions) / len(positions)
        analysis.append(f"Average stranding position: {avg_pos:.1f} km")
        
        # Check if strandings cluster near stations
        station_kms = sorted(self.station_positions.keys())
        near_station_count = 0
        for pos in positions:
            for sk in station_kms:
                if abs(pos - sk) < 20:  # Within 20km of a station
                    near_station_count += 1
                    break
        
        analysis.append(f"Vehicles stranded near stations (<20km): {near_station_count}/{len(positions)}")
        
        return analysis

    def plot_stranded_vehicle_trajectories(self, max_vehicles: int = 10, 
                                          output_dir: str = "./simulation_output") -> None:
        """
        Plot detailed trajectories of stranded vehicles to analyze failure modes.
        
        Creates a multi-panel plot for each vehicle showing:
        - Position vs time (with station markers)
        - SOC vs time (with critical levels marked)
        - State timeline
        - Key events annotated
        
        Args:
            max_vehicles: Maximum number of stranded vehicles to plot (default 10)
            save_path: If provided, save plots to this file path
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
        import numpy as np
        
        if not self.vehicles_stranded:
            print("No stranded vehicles to plot")
            return
        
        # Select up to max_vehicles most recent strandings
        vehicles_to_plot = self.vehicles_stranded[-max_vehicles:]
        n_vehicles = len(vehicles_to_plot)
        
        # Create figure with subplots for each vehicle
        # Each vehicle gets 3 rows: position, SOC, state timeline
        fig, axes = plt.subplots(n_vehicles, 3, figsize=(18, 4*n_vehicles))
        if n_vehicles == 1:
            axes = axes.reshape(1, -1)
        
        # Color map for states
        state_colors = {
            'CRUISING': '#2ecc71',      # Green
            'APPROACHING': '#f39c12',   # Orange
            'DECIDING': '#9b59b6',      # Purple
            'QUEUED': '#3498db',        # Blue
            'CHARGING': '#1abc9c',      # Teal
            'EXITING': '#e74c3c',       # Red
            'STRANDED': '#000000',      # Black
            'DESTINATION': '#95a5a6'    # Gray
        }
        
        # Station positions for reference
        station_positions = sorted(self.station_positions.keys())
        
        for idx, vehicle in enumerate(vehicles_to_plot):
            ax_pos = axes[idx, 0]  # Position vs time
            ax_soc = axes[idx, 1]  # SOC vs time
            ax_state = axes[idx, 2]  # State timeline
            
            # Extract time series data from history
            times = []
            positions = []
            socs = []
            states = []
            speeds = []
            
            for entry in vehicle.detailed_history:
                try:
                    # Parse timestamp
                    if isinstance(entry['timestamp'], str):
                        t = datetime.fromisoformat(entry['timestamp'])
                    else:
                        t = entry['timestamp']
                    times.append(t)
                    positions.append(entry.get('position_km', 0))
                    socs.append(entry.get('soc', 0) * 100)  # Convert to percentage
                    states.append(entry.get('state', 'UNKNOWN'))
                    speeds.append(entry.get('speed_kmh', 0))
                except (KeyError, ValueError):
                    continue
            
            if not times:
                continue
            
            # Convert times to minutes from start
            start_time = times[0]
            time_minutes = [(t - start_time).total_seconds() / 60 for t in times]
            
            # --- PLOT 1: Position vs Time ---
            ax_pos.plot(time_minutes, positions, 'b-', linewidth=2, label='Position')
            ax_pos.set_ylabel('Position (km)', fontsize=10)
            ax_pos.set_xlabel('Time (min)', fontsize=10)
            ax_pos.set_title(f'Vehicle {vehicle.id} - Position Track\n'
                           f'Driver: {vehicle.driver.behavior_type}, '
                           f'Stops: {vehicle.charging_stops}', fontsize=11)
            ax_pos.grid(True, alpha=0.3)
            
            # Add station markers
            for station_km in station_positions:
                ax_pos.axhline(y=station_km, color='gray', linestyle='--', alpha=0.5)
                ax_pos.text(time_minutes[-1]*0.02, station_km+2, f'Station {station_km:.0f}km', 
                          fontsize=8, color='gray')
            
            # Mark stranding point
            if vehicle.state == VehicleState.STRANDED and positions:
                ax_pos.scatter([time_minutes[-1]], [positions[-1]], 
                             color='red', s=200, marker='X', zorder=5, label='Stranded')
            
            # Add event annotations
            for i, entry in enumerate(vehicle.detailed_history):
                if entry.get('event_type') in ['ABANDON', 'CHARGE_START', 'CHARGE_COMPLETE', 
                                               'MISSED_STATION', 'QUEUE_BLOCKED']:
                    if i < len(time_minutes):
                        ax_pos.annotate(entry['event_type'], 
                                      xy=(time_minutes[i], positions[i]),
                                      xytext=(10, 10), textcoords='offset points',
                                      fontsize=7, color='red',
                                      bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor='yellow', alpha=0.7),
                                      arrowprops=dict(arrowstyle='->', color='red'))
            
            ax_pos.legend(loc='upper left', fontsize=8)
            
            # --- PLOT 2: SOC vs Time ---
            ax_soc.plot(time_minutes, socs, 'g-', linewidth=2, label='SOC')
            ax_soc.set_ylabel('SOC (%)', fontsize=10)
            ax_soc.set_xlabel('Time (min)', fontsize=10)
            ax_soc.set_title('Battery State of Charge', fontsize=11)
            ax_soc.set_ylim(0, 100)
            ax_soc.grid(True, alpha=0.3)
            
            # Add critical SOC lines
            ax_soc.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Low (20%)')
            ax_soc.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Critical (10%)')
            ax_soc.axhline(y=5, color='darkred', linestyle='--', alpha=0.7, label='Emergency (5%)')
            
            # Fill area below critical levels
            ax_soc.fill_between(time_minutes, 0, socs, where=[s <= 10 for s in socs], 
                              color='red', alpha=0.3, label='Critical zone')
            
            # Mark charging periods
            in_charging = False
            charge_start = None
            for i, state in enumerate(states):
                if state == 'CHARGING' and not in_charging:
                    in_charging = True
                    charge_start = time_minutes[i]
                elif state != 'CHARGING' and in_charging:
                    in_charging = False
                    if charge_start is not None:
                        ax_soc.axvspan(charge_start, time_minutes[i], 
                                     color='green', alpha=0.2, label='Charging')
            
            # Mark stranding point
            if socs:
                ax_soc.scatter([time_minutes[-1]], [socs[-1]], 
                             color='red', s=200, marker='X', zorder=5)
            
            ax_soc.legend(loc='upper right', fontsize=8)
            
            # --- PLOT 3: State Timeline ---
            # Create a timeline showing state transitions
            state_changes = []
            current_state = states[0] if states else 'UNKNOWN'
            state_start_time = time_minutes[0] if time_minutes else 0
            
            for i in range(1, len(states)):
                if states[i] != current_state:
                    state_changes.append({
                        'state': current_state,
                        'start': state_start_time,
                        'end': time_minutes[i-1],
                        'duration': time_minutes[i-1] - state_start_time
                    })
                    current_state = states[i]
                    state_start_time = time_minutes[i]
            
            # Add final state
            if time_minutes:
                state_changes.append({
                    'state': current_state,
                    'start': state_start_time,
                    'end': time_minutes[-1],
                    'duration': time_minutes[-1] - state_start_time
                })
            
            # Plot state bars
            y_pos = 0
            for change in state_changes:
                color = state_colors.get(change['state'], '#95a5a6')
                ax_state.barh(y_pos, change['duration'], left=change['start'], 
                            color=color, height=0.6, edgecolor='black')
                # Add state label if duration is long enough
                if change['duration'] > (time_minutes[-1] - time_minutes[0]) * 0.05:
                    ax_state.text(change['start'] + change['duration']/2, y_pos,
                                change['state'], ha='center', va='center',
                                fontsize=8, fontweight='bold')
            
            ax_state.set_yticks([])
            ax_state.set_xlabel('Time (min)', fontsize=10)
            ax_state.set_title('State Timeline', fontsize=11)
            ax_state.set_xlim(time_minutes[0] if time_minutes else 0, 
                            time_minutes[-1] if time_minutes else 1)
            
            # Add legend for states
            legend_elements = [mpatches.Patch(color=color, label=state) 
                             for state, color in state_colors.items() 
                             if state in [s['state'] for s in state_changes]]
            ax_state.legend(handles=legend_elements, loc='upper left', 
                          fontsize=7, ncol=2)
        
        plt.tight_layout()
        
        if output_dir:
            save_path = f"{output_dir}/stranded_vehicles_trajectories.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved stranded vehicle trajectories to {save_path}")
        else:
            plt.savefig('./simulation_output/stranded_vehicles_trajectories.png', 
                       dpi=150, bbox_inches='tight')
            print("Saved stranded vehicle trajectories to /simulation_output/stranded_vehicles_trajectories.png")
        
        plt.show()
    
    def plot_stranding_summary_chart(self, output_dir: str = "./simulation_output") -> None:
        """
        Create a summary chart showing aggregate statistics about stranded vehicles.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not self.vehicles_stranded:
            print("No stranded vehicles to analyze")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Stranding position histogram
        ax = axes[0, 0]
        positions = [v.position_km for v in self.vehicles_stranded]
        ax.hist(positions, bins=20, color='red', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Stranding Position (km)')
        ax.set_ylabel('Number of Vehicles')
        ax.set_title('Where Do Vehicles Strand?')
        ax.grid(True, alpha=0.3)
        
        # Add station markers
        for station_km in sorted(self.station_positions.keys()):
            ax.axvline(station_km, color='blue', linestyle='--', alpha=0.5)
        
        # 2. SOC at stranding
        ax = axes[0, 1]
        final_socs = [v.battery.current_soc * 100 for v in self.vehicles_stranded]
        ax.hist(final_socs, bins=15, color='orange', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Final SOC (%)')
        ax.set_ylabel('Number of Vehicles')
        ax.set_title('Battery Level When Stranded')
        ax.axvline(5, color='red', linestyle='--', label='Critical (5%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Driver type distribution
        ax = axes[0, 2]
        driver_types = {}
        for v in self.vehicles_stranded:
            dtype = v.driver.behavior_type
            driver_types[dtype] = driver_types.get(dtype, 0) + 1
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        ax.pie(driver_types.values(), labels=driver_types.keys(), autopct='%1.1f%%',
               colors=colors[:len(driver_types)])
        ax.set_title('Stranded by Driver Type')
        
        # 4. Charging stops before stranding
        ax = axes[1, 0]
        charging_stops = [v.charging_stops for v in self.vehicles_stranded]
        stop_counts = {}
        for stops in charging_stops:
            stop_counts[stops] = stop_counts.get(stops, 0) + 1
        ax.bar(stop_counts.keys(), stop_counts.values(), color='green', alpha=0.7)
        ax.set_xlabel('Number of Charging Stops')
        ax.set_ylabel('Number of Vehicles')
        ax.set_title('Charging Stops Before Stranding')
        ax.grid(True, alpha=0.3)
        
        # 5. Distance traveled
        ax = axes[1, 1]
        distances = [v.total_distance_km for v in self.vehicles_stranded]
        ax.hist(distances, bins=15, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Total Distance Traveled (km)')
        ax.set_ylabel('Number of Vehicles')
        ax.set_title('Distance Before Stranding')
        ax.grid(True, alpha=0.3)
        
        # 6. Abandonment correlation
        ax = axes[1, 2]
        abandoned = []
        for v in self.vehicles_stranded:
            has_abandon = any(e.get('event_type') == 'ABANDON' 
                            for e in v.detailed_history)
            abandoned.append('Abandoned Queue' if has_abandon else 'Did Not Abandon')
        abandon_counts = {}
        for a in abandoned:
            abandon_counts[a] = abandon_counts.get(a, 0) + 1
        ax.pie(abandon_counts.values(), labels=abandon_counts.keys(), autopct='%1.1f%%',
               colors=['#e74c3c', '#2ecc71'])
        ax.set_title('Queue Abandonment Before Stranding')
        
        plt.suptitle(f'Stranded Vehicle Analysis (n={len(self.vehicles_stranded)})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_dir:
            save_path = f"{output_dir}/stranding_summary.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig('./simulation_output/stranding_summary.png', dpi=150, bbox_inches='tight')
        
        plt.show()

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
            
            while not vehicle.can_physically_reach(vehicle.position_km):
                vehicle.position_km += 1

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