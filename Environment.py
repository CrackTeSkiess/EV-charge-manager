"""
Environment Module
Top-level simulation controller managing highway, vehicles, and time progression.
"""

from __future__ import annotations

import uuid
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from datetime import datetime, timedelta
from collections import deque

from Highway import Highway, HighwaySegment
from Vehicle import Vehicle, VehicleState, DriverBehavior
from ChargingArea import ChargingArea, ChargerType


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation environment."""
    # Highway parameters
    highway_length_km: float = 300.0
    
    # Charging infrastructure
    num_charging_areas: int = 3
    charging_area_positions: Optional[List[float]] = None  # km marks, or auto-spaced
    chargers_per_area: int = 4
    waiting_spots_per_area: int = 6
    
    # Traffic parameters
    arrival_rate_per_minute: float = 2.0  # Poisson rate
    ev_penetration_rate: float = 0.3  # 30% of traffic is EV (for future hybrid support)
    
    # Vehicle fleet distribution
    battery_capacity_distribution: List[Tuple[float, float]] = field(
        default_factory=lambda: [(50, 0.2), (60, 0.3), (75, 0.3), (90, 0.15), (100, 0.05)]
    )
    driver_behavior_distribution: List[Tuple[str, float]] = field(
        default_factory=lambda: [
            ('conservative', 0.25),
            ('balanced', 0.40),
            ('aggressive', 0.25),
            ('range_anxious', 0.10)
        ]
    )
    
    # Time parameters
    simulation_start_time: datetime = field(
        default_factory=lambda: datetime(2024, 1, 15, 8, 0)
    )
    time_step_minutes: float = 1.0
    
    # Random seed
    random_seed: Optional[int] = None


class Environment:
    """
    Simulation environment managing the complete EV highway charging ecosystem.
    
    Responsibilities:
    - Initialize and own the Highway instance
    - Manage vehicle lifecycle (spawn at entry, despawn at exit)
    - Orchestrate time-stepped simulation
    - Collect global statistics and metrics
    - Provide observation interface for ML optimization
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.id = str(uuid.uuid4())[:8]
        
        # Initialize random state
        if self.config.random_seed:
            random.seed(self.config.random_seed)
        
        # Time management
        self.current_time = self.config.simulation_start_time
        self.step_count = 0
        self.is_running = False
        
        # Create highway with charging infrastructure
        self.highway = self._initialize_highway()
        
        # Vehicle tracking (beyond highway's tracking)
        self.all_vehicles_history: Dict[str, Dict] = {}  # Complete history by ID
        self.active_vehicle_ids: set = set()
        self.completed_trips: List[Dict] = []
        self.failed_trips: List[Dict] = []  # Stranded vehicles
        
        # Arrival process state
        self._arrival_accumulator = 0.0  # For fractional arrivals
        
        # Metrics collection
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours of minute data
        self.current_metrics = self._reset_metrics()
        
        # Event log
        self.event_log: List[Dict] = []
        self.max_log_size = 10000
        
        # Callbacks for external integration
        self.on_vehicle_spawn: Optional[Callable[[Vehicle], None]] = None
        self.on_vehicle_despawn: Optional[Callable[[Vehicle, str], None]] = None
        self.on_step_complete: Optional[Callable[[int, Dict], None]] = None
    
    def _initialize_highway(self) -> Highway:
        """Create highway with configured charging areas."""
        # Determine charging area positions
        if self.config.charging_area_positions:
            positions = self.config.charging_area_positions
        else:
            # Auto-space evenly along highway
            n = self.config.num_charging_areas
            spacing = self.config.highway_length_km / (n + 1)
            positions = [spacing * (i + 1) for i in range(n)]
        
        # Create charging areas
        charging_areas = []
        for i, pos in enumerate(positions):
            area = ChargingArea(
                area_id=f"AREA-{i+1:02d}",
                name=f"Charging Area {i+1}",
                location_km=pos,
                num_chargers=self.config.chargers_per_area,
                waiting_spots=self.config.waiting_spots_per_area
            )
            charging_areas.append(area)
        
        # Create highway
        highway = Highway(
            highway_id="HWY-MAIN",
            length_km=self.config.highway_length_km,
            charging_areas=charging_areas
        )
        
        return highway
    
    # ========================================================================
    # VEHICLE LIFECYCLE MANAGEMENT
    # ========================================================================
    
    def spawn_vehicle(self, custom_vehicle: Optional[Vehicle] = None) -> Optional[Vehicle]:
        """
        Insert new vehicle at start of highway (position 0).
        Returns created vehicle or None if spawn failed.
        """
        if custom_vehicle:
            vehicle = custom_vehicle
            vehicle.position_km = 0.0
        else:
            vehicle = self._generate_random_vehicle()
        
        # Add to highway
        self.highway.spawn_vehicle(vehicle, position_km=0.0, entry_time=self.current_time)
        self.active_vehicle_ids.add(vehicle.id)
        
        # Track in environment history
        self.all_vehicles_history[vehicle.id] = {
            'spawn_time': self.current_time,
            'initial_soc': vehicle.battery.current_soc,
            'battery_capacity': vehicle.battery.capacity_kwh,
            'driver_type': vehicle.driver.behavior_type,
            'events': []
        }
        
        # Update metrics
        self.current_metrics['vehicles_spawned'] += 1
        self._log_event('spawn', vehicle.id, {'soc': vehicle.battery.current_soc})
        
        # Callback
        if self.on_vehicle_spawn:
            self.on_vehicle_spawn(vehicle)
        
        return vehicle
    
    def _generate_random_vehicle(self) -> Vehicle:
        """Generate vehicle with random properties from distributions."""
        # Battery capacity
        capacities, weights = zip(*self.config.battery_capacity_distribution)
        capacity = random.choices(capacities, weights)[0]
        
        # Driver behavior
        behaviors, b_weights = zip(*self.config.driver_behavior_distribution)
        behavior_type = random.choices(behaviors, b_weights)[0]
        
        # Create driver with type-specific parameters
        driver = self._create_driver_by_type(behavior_type)
        
        # Initial SOC: mostly mid-range, some edge cases
        rand = random.random()
        if rand < 0.15:
            initial_soc = random.uniform(0.20, 0.40)  # Low
        elif rand < 0.85:
            initial_soc = random.uniform(0.40, 0.75)  # Mid
        else:
            initial_soc = random.uniform(0.75, 0.95)  # High
        
        return Vehicle(
            battery_capacity_kwh=capacity,
            initial_soc=initial_soc,
            driver_behavior=driver,
            initial_position_km=0.0,
            initial_speed_kmh=random.uniform(100, 130)
        )
    
    def _create_driver_by_type(self, behavior_type: str) -> DriverBehavior:
        """Create driver behavior profile based on type."""
        base_params = {
            'conservative': {
                'patience_base': random.uniform(0.7, 0.9),
                'risk_tolerance': random.uniform(0.1, 0.3),
                'speed_preference_kmh': random.uniform(100, 115),
                'buffer_margin_km': random.uniform(60, 100),
                'target_soc': random.uniform(0.85, 0.95)
            },
            'balanced': {
                'patience_base': random.uniform(0.5, 0.7),
                'risk_tolerance': random.uniform(0.3, 0.6),
                'speed_preference_kmh': random.uniform(110, 125),
                'buffer_margin_km': random.uniform(40, 70),
                'target_soc': random.uniform(0.75, 0.85)
            },
            'aggressive': {
                'patience_base': random.uniform(0.3, 0.5),
                'risk_tolerance': random.uniform(0.6, 0.9),
                'speed_preference_kmh': random.uniform(125, 140),
                'acceleration_aggression': random.uniform(0.6, 0.9),
                'buffer_margin_km': random.uniform(20, 40),
                'target_soc': random.uniform(0.60, 0.75)
            },
            'range_anxious': {
                'patience_base': random.uniform(0.5, 0.7),
                'risk_tolerance': random.uniform(0.05, 0.2),
                'speed_preference_kmh': random.uniform(90, 110),
                'buffer_margin_km': random.uniform(80, 120),
                'min_charge_to_continue': random.uniform(0.35, 0.50),
                'social_influence': random.uniform(0.5, 0.8),
                'target_soc': random.uniform(0.90, 1.0)
            }
        }
        
        params = base_params.get(behavior_type, base_params['balanced'])
        return DriverBehavior(behavior_type=behavior_type, **params) # pyright: ignore[reportArgumentType]
    
    def despawn_vehicle(self, vehicle_id: str, reason: str = "exit") -> Optional[Vehicle]:
        """
        Remove vehicle from simulation when it reaches highway end or fails.
        Automatically called by step() for vehicles at position >= length.
        """
        # Get from highway (this removes it from all highway tracking)
        vehicle = self.highway.remove_vehicle(vehicle_id, reason)
        if not vehicle:
            return None
        
        # Update environment tracking
        self.active_vehicle_ids.discard(vehicle_id)
        
        # Record trip summary
        trip_data = vehicle.get_trip_summary()
        trip_data['exit_time'] = self.current_time
        trip_data['exit_reason'] = reason
        trip_data['duration_minutes'] = (
            (self.current_time - self.all_vehicles_history[vehicle_id]['spawn_time'])
            .total_seconds() / 60
        )
        
        if reason == "stranded":
            self.failed_trips.append(trip_data)
            self.current_metrics['vehicles_stranded'] += 1
        else:
            self.completed_trips.append(trip_data)
            self.current_metrics['vehicles_completed'] += 1
        
        # Update history
        self.all_vehicles_history[vehicle_id]['completion'] = trip_data
        
        # Update metrics
        self._log_event('despawn', vehicle_id, {'reason': reason, 'trip': trip_data})
        
        # Callback
        if self.on_vehicle_despawn:
            self.on_vehicle_despawn(vehicle, reason)
        
        return vehicle
    
    def _process_automatic_despawns(self) -> List[str]:
        """
        Check for vehicles at highway end and despawn them.
        Returns list of despawned vehicle IDs.
        """
        despawned = []
        
        for vid in list(self.active_vehicle_ids):
            vehicle = self.highway.vehicles.get(vid)
            if not vehicle:
                continue
            
            # Check if reached end
            if vehicle.position_km >= self.config.highway_length_km:
                self.despawn_vehicle(vid, "exit")
                despawned.append(vid)
            
            # Check if stranded
            elif vehicle.state == VehicleState.STRANDED:
                self.despawn_vehicle(vid, "stranded")
                despawned.append(vid)
        
        return despawned
    
    # ========================================================================
    # ARRIVAL PROCESS
    # ========================================================================
    
    def _process_arrivals(self) -> int:
        """
        Generate new vehicles based on arrival rate.
        Uses Poisson process with accumulation for fractional rates.
        Returns number of vehicles spawned.
        """
        # Add expected arrivals to accumulator
        self._arrival_accumulator += self.config.arrival_rate_per_minute
        
        # Determine integer arrivals
        num_arrivals = int(self._arrival_accumulator)
        self._arrival_accumulator -= num_arrivals
        
        # Add randomness (Poisson variation)
        if random.random() < (self._arrival_accumulator % 1):
            num_arrivals += 1
            self._arrival_accumulator = 0
        
        # Spawn vehicles
        spawned = 0
        for _ in range(num_arrivals):
            vehicle = self.spawn_vehicle()
            if vehicle:
                spawned += 1
        
        return spawned
    
    # ========================================================================
    # MAIN SIMULATION STEP
    # ========================================================================
    
    def step(self) -> Dict:
        """
        Execute one simulation step (default 1 minute).
        
        Sequence:
        1. Process new vehicle arrivals
        2. Step highway (update all vehicles, charging areas, handoffs)
        3. Automatic despawn at highway end
        4. Collect metrics
        5. Advance time
        
        Returns step summary with key events.
        """
        if not self.is_running:
            self.is_running = True
        
        step_summary = {
            'step': self.step_count,
            'time': self.current_time,
            'arrivals': 0,
            'despawns': [],
            'highway_events': {},
            'metrics': {}
        }
        
        # 1. Process arrivals
        step_summary['arrivals'] = self._process_arrivals()
        
        # 2. Step highway (this updates all vehicle physics, charging, etc.)
        highway_events = self.highway.step(self.current_time, self.config.time_step_minutes)
        step_summary['highway_events'] = highway_events
        
        # Update metrics from highway events
        self.current_metrics['abandonments'] += len(highway_events.get('abandonments', []))
        self.current_metrics['charging_starts'] += len(highway_events.get('new_charging', []))
        self.current_metrics['charging_completions'] += len(highway_events.get('charging_completions', []))
        
        # 3. Automatic despawn
        despawned = self._process_automatic_despawns()
        step_summary['despawns'] = despawned
        
        # 4. Collect current state metrics
        highway_state = self.highway.get_highway_state()
        self.current_metrics.update({
            'active_vehicles': len(self.active_vehicle_ids),
            'driving': highway_state['driving'],
            'charging': highway_state['charging'],
            'queued': highway_state['queued'],
            'avg_speed': self._calculate_avg_speed(),
            'avg_soc': self._calculate_avg_soc()
        })
        
        # Store metrics history
        self.metrics_history.append(self.current_metrics.copy())
        step_summary['metrics'] = self.current_metrics.copy()
        
        # 5. Advance time
        self.step_count += 1
        self.current_time += timedelta(minutes=self.config.time_step_minutes)
        
        # Callback
        if self.on_step_complete:
            self.on_step_complete(self.step_count, step_summary)
        
        return step_summary
    
    def run(self, duration_minutes: int, progress_interval: int = 60) -> Dict:
        """
        Run simulation for specified duration.
        
        Args:
            duration_minutes: Total simulation time to run
            progress_interval: Print progress every N minutes
        
        Returns:
            Final simulation statistics
        """
        print(f"Starting simulation {self.id} for {duration_minutes} minutes...")
        print(f"Highway: {self.config.highway_length_km}km, "
              f"{len(self.highway.charging_areas)} charging areas")
        
        start_real_time = datetime.now()
        
        for minute in range(duration_minutes):
            summary = self.step()
            
            # Progress reporting
            if minute % progress_interval == 0 and minute > 0:
                self._print_progress(summary)
        
        elapsed = (datetime.now() - start_real_time).total_seconds()
        print(f"\nSimulation complete in {elapsed:.1f}s real time")
        
        return self.get_final_statistics()
    
    def _print_progress(self, summary: Dict):
        """Print current simulation status."""
        m = summary['metrics']
        print(f"\n[{summary['time'].strftime('%H:%M')}] "
              f"Step {summary['step']}: "
              f"Vehicles: {m['active_vehicles']} "
              f"(Driving: {m['driving']}, Charging: {m['charging']}, Queued: {m['queued']}) | "
              f"Completed: {m['vehicles_completed']}, "
              f"Stranded: {m['vehicles_stranded']}")
    
    # ========================================================================
    # METRICS AND STATISTICS
    # ========================================================================
    
    def _reset_metrics(self) -> Dict:
        """Initialize metrics dictionary."""
        return {
            'vehicles_spawned': 0,
            'vehicles_completed': 0,
            'vehicles_stranded': 0,
            'abandonments': 0,
            'charging_starts': 0,
            'charging_completions': 0,
            'active_vehicles': 0,
            'driving': 0,
            'charging': 0,
            'queued': 0,
            'avg_speed': 0.0,
            'avg_soc': 0.0
        }
    
    def _calculate_avg_speed(self) -> float:
        """Calculate average speed of driving vehicles."""
        driving = [
            v for v in self.highway.vehicles.values()
            if v.id not in self.highway.vehicles_at_station
            and v.id not in self.highway.vehicles_in_queue
        ]
        if not driving:
            return 0.0
        return sum(v.speed_kmh for v in driving) / len(driving)
    
    def _calculate_avg_soc(self) -> float:
        """Calculate average SOC of all active vehicles."""
        if not self.highway.vehicles:
            return 0.0
        return sum(v.battery.current_soc for v in self.highway.vehicles.values()) / len(self.highway.vehicles)
    
    def _log_event(self, event_type: str, vehicle_id: str, data: Dict):
        """Log simulation event."""
        event = {
            'step': self.step_count,
            'time': self.current_time,
            'type': event_type,
            'vehicle_id': vehicle_id,
            'data': data
        }
        self.event_log.append(event)
        
        # Trim log if too large
        if len(self.event_log) > self.max_log_size:
            self.event_log = self.event_log[-self.max_log_size//2:]
    
    def get_final_statistics(self) -> Dict:
        """Compile comprehensive simulation statistics."""
        total_vehicles = len(self.all_vehicles_history)
        
        if not total_vehicles:
            return {"error": "No vehicles processed"}
        
        # Trip statistics
        trip_durations = [t['duration_minutes'] for t in self.completed_trips]
        trip_consumption = [t['avg_consumption_kwh_per_100km'] for t in self.completed_trips]
        
        # Charging statistics
        total_charging_time = sum(t['total_charging_time_min'] for t in self.completed_trips)
        total_charging_stops = sum(t['charging_stops'] for t in self.completed_trips)
        
        # Station statistics
        station_stats = {
            aid: area.get_statistics()
            for aid, area in self.highway.charging_areas.items()
        }
        
        return {
            'simulation_id': self.id,
            'config': {
                'highway_length_km': self.config.highway_length_km,
                'arrival_rate': self.config.arrival_rate_per_minute,
                'duration_steps': self.step_count,
                'simulated_duration_hours': self.step_count * self.config.time_step_minutes / 60
            },
            'vehicles': {
                'total_spawned': total_vehicles,
                'completed': len(self.completed_trips),
                'stranded': len(self.failed_trips),
                'success_rate': len(self.completed_trips) / total_vehicles if total_vehicles else 0,
                'in_progress': len(self.active_vehicle_ids)
            },
            'trips': {
                'avg_duration_min': sum(trip_durations) / len(trip_durations) if trip_durations else 0,
                'avg_consumption_kwh_per_100km': sum(trip_consumption) / len(trip_consumption) if trip_consumption else 0,
                'avg_charging_stops': total_charging_stops / len(self.completed_trips) if self.completed_trips else 0,
                'avg_charging_time_min': total_charging_time / len(self.completed_trips) if self.completed_trips else 0
            },
            'charging_infrastructure': {
                'num_stations': len(self.highway.charging_areas),
                'total_chargers': sum(len(a.chargers) for a in self.highway.charging_areas.values()),
                'total_waiting_spots': sum(a.waiting_spots for a in self.highway.charging_areas.values()),
                'station_statistics': station_stats
            },
            'final_state': self.highway.get_highway_state(),
            'metrics_timeseries': list(self.metrics_history)
        }
    
    def get_observation_for_ml(self) -> Dict:
        """
        Get current state observation for machine learning optimization.
        Flattened state suitable for RL or optimization algorithms.
        """
        highway_state = self.highway.get_highway_state()
        
        # Vehicle distribution by SOC bins
        soc_distribution = [0, 0, 0, 0, 0]  # <10%, 10-30%, 30-60%, 60-85%, >85%
        for v in self.highway.vehicles.values():
            soc = v.battery.current_soc
            if soc < 0.10:
                soc_distribution[0] += 1
            elif soc < 0.30:
                soc_distribution[1] += 1
            elif soc < 0.60:
                soc_distribution[2] += 1
            elif soc < 0.85:
                soc_distribution[3] += 1
            else:
                soc_distribution[4] += 1
        
        # Station utilization vector
        station_utils = [
            s['current_utilization'] 
            for s in highway_state['charging_areas'].values()
        ]
        
        return {
            'time_of_day': self.current_time.hour + self.current_time.minute / 60,
            'step_count': self.step_count,
            'total_vehicles': highway_state['active_vehicles'],
            'driving_vehicles': highway_state['driving'],
            'queued_vehicles': highway_state['queued'],
            'charging_vehicles': highway_state['charging'],
            'soc_distribution': soc_distribution,
            'avg_soc': self._calculate_avg_soc(),
            'avg_speed': self._calculate_avg_speed(),
            'station_utilizations': station_utils,
            'mean_station_utilization': sum(station_utils) / len(station_utils) if station_utils else 0,
            'max_station_utilization': max(station_utils) if station_utils else 0,
            'recent_abandonment_rate': (
                sum(m['abandonments'] for m in list(self.metrics_history)[-10:]) / 10
                if len(self.metrics_history) >= 10 else 0
            )
        }
    
    def reset(self):
        """Reset environment to initial state."""
        self.__init__(self.config)
    
    def __repr__(self) -> str:
        return (f"Environment({self.id}, step={self.step_count}, "
                f"time={self.current_time.strftime('%H:%M')}, "
                f"vehicles={len(self.active_vehicle_ids)})")
