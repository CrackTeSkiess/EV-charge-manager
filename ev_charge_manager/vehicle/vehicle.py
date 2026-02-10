"""
Vehicle Module - WITH DEBUGGING
Electric vehicle agent with physical dynamics and driver behavior.
"""

from __future__ import annotations

import uuid
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Callable, Tuple, List, Any
from datetime import datetime, timedelta


class VehicleState(Enum):
    """Physical state of the vehicle on the highway."""
    CRUISING = auto()       # Normal highway driving
    APPROACHING = auto()    # Within decision range of station
    DECIDING = auto()       # Evaluating charging options
    QUEUED = auto()         # In charging area queue
    CHARGING = auto()       # Actively charging
    EXITING = auto()        # Leaving station, merging back
    DESTINATION = auto()    # Trip complete
    STRANDED = auto()       # Battery depleted (failed)


@dataclass
class Battery:
    """
    Electric vehicle battery with state-of-charge management.
    """
    capacity_kwh: float = 70.0          # Total capacity
    current_soc: float = 0.8             # Current state of charge (0-1)
    min_operating_soc: float = 0.05      # Absolute minimum before damage
    usable_soc_range: Tuple[float, float] = (0.1, 1.0)  # Usable window
    
    # Physical characteristics
    degradation_factor: float = 0.95     # Capacity retention (age)
    temperature_c: float = 25.0          # Current battery temp
    thermal_resistance: float = 0.5      # Heating/cooling rate
    
    def __post_init__(self):
        self.current_kwh = self.capacity_kwh * self.current_soc * self.degradation_factor
        self.max_usable_kwh = self.capacity_kwh * (self.usable_soc_range[1] - self.usable_soc_range[0])
    
    @property
    def usable_soc(self) -> float:
        """SOC within usable window (0-1 scale relative to usable range)."""
        usable_range = self.usable_soc_range[1] - self.usable_soc_range[0]
        if usable_range <= 0:
            return 0.0
        current_in_range = self.current_soc - self.usable_soc_range[0]
        return max(0.0, min(1.0, current_in_range / usable_range))
    
    @property
    def is_critical(self) -> bool:
        """Battery critically low."""
        return self.current_soc <= self.usable_soc_range[0] + 0.05
    
    @property
    def is_depleted(self) -> bool:
        """Battery empty (stranded)."""
        return self.current_soc <= self.min_operating_soc
    
    def consume(self, energy_kwh: float) -> bool:
        """
        Consume energy from battery. Returns False if insufficient.
        """
        available = self.current_kwh
        if energy_kwh > available:
            self.current_kwh = 0
            self.current_soc = 0
            return False
        
        self.current_kwh -= energy_kwh
        self.current_soc = self.current_kwh / (self.capacity_kwh * self.degradation_factor)
        return True
    
    def charge(self, energy_kwh: float) -> float:
        """
        Add energy to battery. Returns actual energy accepted.
        """
        max_kwh = self.capacity_kwh * self.degradation_factor * self.usable_soc_range[1]
        available_space = max_kwh - self.current_kwh
        actual = min(energy_kwh, available_space)
        
        self.current_kwh += actual
        self.current_soc = self.current_kwh / (self.capacity_kwh * self.degradation_factor)
        return actual
    
    def estimate_range_km(self, consumption_rate_kwh_per_km: float) -> float:
        """Estimate remaining range at given consumption rate."""
        if consumption_rate_kwh_per_km <= 0:
            return float('inf')
        return self.current_kwh / consumption_rate_kwh_per_km


@dataclass
class DriverBehavior:
    """
    Driver behavior profile - determines decision making and driving style.
    """
    
    # Identification
    driver_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    behavior_type: str = "balanced"  # conservative, balanced, aggressive, range_anxious
    
    # Fuzzy behavioral parameters (0-1 scales, interpreted via fuzzy logic later)
    patience_base: float = 0.6           # Base patience level
    patience_decay_rate: float = 0.3     # How quickly patience degrades
    risk_tolerance: float = 0.5          # Willingness to push battery low
    price_sensitivity: float = 0.5       # Sensitivity to charging costs
    comfort_preference: float = 0.5      # Preference for amenities
    time_sensitivity: float = 0.5        # How much delay matters
    social_influence: float = 0.3        # Susceptibility to herd behavior
    tech_savviness: float = 0.5          # Trust in/app usage of apps/predictions
    
    # Charging preferences
    target_soc: float = 0.8              # Desired SOC when leaving station
    min_charge_to_continue: float = 0.3  # Minimum acceptable to skip station
    buffer_margin_km: float = 50.0       # Safety buffer for range anxiety
    
    # Driving style (affects consumption)
    acceleration_aggression: float = 0.5  # 0=eco, 1=sport
    speed_preference_kmh: float = 120.0   # Preferred cruising speed
    drafting_behavior: float = 0.3        # Willingness to follow closely (reduce drag)
    
    # Decision thresholds (crisp for now, fuzzy later)
    max_wait_acceptable_min: float = 30.0
    max_price_per_kwh: float = 0.60
    min_station_rating: float = 3.0
    
    def calculate_desired_speed(self, current_speed_limit: float, 
                               traffic_flow: float = 1.0) -> float:
        """
        Determine actual desired speed based on behavior and conditions.
        """
        # Base: preferred speed vs limit
        base = min(self.speed_preference_kmh, current_speed_limit + 10)
        
        # Adjust for traffic (fuzzy: heavy flow = slower)
        traffic_adjustment = 1.0 - (1.0 - traffic_flow) * 0.3
        
        # Aggressive drivers push harder
        aggression_bonus = 1.0 + (self.acceleration_aggression - 0.5) * 0.2
        
        desired = base * traffic_adjustment * aggression_bonus
        return max(20.0, min(current_speed_limit + 20, desired))
    
    def evaluate_station_attractiveness(self, station_info: Dict, 
                                       current_soc: float,
                                       estimated_wait: float) -> float:
        """
        Calculate attractiveness score for a charging station.
        Returns 0-1 score (higher = more attractive).
        """
        # Urgency factor (linear for now, fuzzy later)
        urgency = max(0, 1 - (current_soc / self.min_charge_to_continue))
        
        # Wait tolerance (fuzzy: short wait = good, long = bad)
        wait_ratio = estimated_wait / self.max_wait_acceptable_min
        wait_score = max(0, 1 - wait_ratio ** 2)  # Quadratic penalty
        
        # Price evaluation
        price = station_info.get('price_per_kwh', 0.40)
        price_score = max(0, 1 - (price / self.max_price_per_kwh))
        
        # Availability preference (herd behavior)
        availability = station_info.get('available_chargers', 0)
        total = station_info.get('total_chargers', 1)
        availability_score = availability / total
        
        # Weighted combination (will become fuzzy rules)
        score = (
            urgency * 0.4 +
            wait_score * 0.3 +
            price_score * 0.1 +
            availability_score * 0.2
        )
        
        # Social influence: boost if others are going there
        if station_info.get('trending', False):
            score *= (1 + self.social_influence * 0.3)
        
        return min(1.0, score)
    
    def update_patience(self, current_patience: float, wait_minutes: float, comfort_level: float, urgency: float) -> float:
        """
        Update patience based on waiting experience.
        Returns new patience level (0-1).
        """
        # Base decay
        decay = wait_minutes * self.patience_decay_rate
        
        # Comfort adjustment (fuzzy: better amenities reduce decay)
        comfort_adjustment = (1 - comfort_level) * 0.2
        
        # Urgency adjustment (fuzzy: more urgent = less patience loss)
        urgency_adjustment = (1 - urgency) * 0.3
        
        new_patience = current_patience - decay * (1 + comfort_adjustment) * (1 - urgency_adjustment)
        return max(0.0, min(1.0, new_patience))
    
    def decide_to_abort(self, current_patience: float, queue_position: int,
                       estimated_wait: float, alternative_available: bool) -> bool:
        """
        Decide whether to abandon queue.
        Probabilistic based on patience threshold.
        """
        if current_patience <= 0.1:
            return True  # Critical patience
        
        # Fuzzy probability (simplified as linear for now)
        abandonment_prob = (1 - current_patience) * 0.8
        
        if alternative_available:
            abandonment_prob *= 1.3  # 30% more likely if alternative exists
        
        # Queue position effect (worse position = more likely to leave)
        position_penalty = min(0.3, queue_position * 0.05)
        abandonment_prob += position_penalty
        
        return random.random() < min(0.95, abandonment_prob)
    
    def __repr__(self) -> str:
        return (f"DriverBehavior({self.behavior_type}, "
                f"patience={self.patience_base:.1f}, "
                f"risk={self.risk_tolerance:.1f})")


class Vehicle:
    """
    Electric vehicle agent with physical dynamics and driver behavior.
    """
    
    # Physics constants
    MAX_ACCELERATION_MS2 = 3.0       # m/s² (0-100 in ~9s)
    MAX_DECELERATION_MS2 = -6.0      # m/s² (emergency braking)
    COMFORT_DECELERATION_MS2 = -2.5  # Normal braking
    DRAG_COEFFICIENT = 0.23          # Typical EV
    FRONTAL_AREA_M2 = 2.5            # m²
    AIR_DENSITY = 1.225              # kg/m³ at sea level
    ROLLING_RESISTANCE = 0.01        # Coefficient
    VEHICLE_MASS_KG = 2000.0         # kg (with driver)
    GRAVITY = 9.81                   # m/s²
    
    # Safety margins
    SAFETY_MARGIN_KM = 20.0          # Minimum km buffer for "must stop" decisions
    CONSUMPTION_SAFETY_FACTOR = 1.15  # 15% buffer on consumption
    
    def __init__(
        self,
        vehicle_id: Optional[str] = None,
        battery_capacity_kwh: float = 60.0,
        initial_soc: Optional[float] = None,
        driver_behavior: Optional[DriverBehavior] = None,
        initial_position_km: float = 0.0,
        initial_speed_kmh: float = 0.0,
        track_history: bool = False
    ):
        self.id = vehicle_id or str(uuid.uuid4())[:12]
        
        # Battery initialization with random SOC if not specified
        if initial_soc is None:
            # Distribution: mostly mid-range, some low, some high
            rand = random.random()
            if rand < 0.2:
                initial_soc = random.uniform(0.15, 0.35)  # Low
            elif rand < 0.8:
                initial_soc = random.uniform(0.4, 0.7)    # Mid
            else:
                initial_soc = random.uniform(0.75, 0.95)  # High
        
        self.battery = Battery(
            capacity_kwh=battery_capacity_kwh,
            current_soc=initial_soc
        )
        
        # Driver behavior (create default if none provided)
        self.driver = driver_behavior or DriverBehavior()
        
        # Kinematic state
        self.position_km = initial_position_km
        self.speed_kmh = initial_speed_kmh
        self.acceleration_ms2 = 0.0
        self.target_speed_kmh = initial_speed_kmh
        
        # Simulation state
        self.state = VehicleState.CRUISING
        self.state_entry_time: Optional[datetime] = None
        
        # Charging interaction
        self.target_station_id: Optional[str] = None
        self.assigned_charger_id: Optional[str] = None
        self.queue_entry_time: Optional[datetime] = None
        self.current_patience: float = self.driver.patience_base
        
        # Trip tracking
        self.total_distance_km = 0.0
        self.total_energy_consumed_kwh = 0.0
        self.charging_stops = 0
        self.total_charging_time_min = 0.0
        
        # Consumption rate cache: discretized speed (int km/h) -> kWh/km
        self._consumption_cache: Dict[int, float] = {}

        # DEBUG: Detailed history for stranded vehicle analysis (optional)
        self.track_history = track_history
        self.detailed_history: List[Dict[str, Any]] = []
        self.state_history: List[Dict[str, Any]] = []
        self.position_history: List[Dict[str, Any]] = []
        self._log_event("INIT", f"Created at km {initial_position_km:.1f}, SOC={initial_soc:.1%}")
    
    # ========================================================================
    # DEBUGGING / HISTORY
    # ========================================================================
    
    def _log_event(self, event_type: str, details: str,
                   extra_data: Optional[Dict] = None) -> None:
        """Log a detailed event for debugging stranded vehicles."""
        if not self.track_history:
            return
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details,
            'position_km': self.position_km,
            'soc': self.battery.current_soc,
            'state': self.state.name,
            'speed_kmh': self.speed_kmh,
            'range_estimate_km': self.get_range_estimate(),
            'patience': self.current_patience
        }
        if extra_data:
            entry.update(extra_data)
        self.detailed_history.append(entry)
        
        # Also print for real-time debugging
        #print(f"[VEHICLE {self.id}] {event_type}: {details} "
        #      f"(pos={self.position_km:.1f}km, SOC={self.battery.current_soc:.1%}, "
        #      f"state={self.state.name})")
    
    def get_debug_report(self) -> str:
        """Generate a detailed debug report for this vehicle."""
        lines = [
            f"\n{'='*70}",
            f"DEBUG REPORT FOR VEHICLE {self.id}",
            f"{'='*70}",
            f"Final Status: {self.state.name}",
            f"Final Position: {self.position_km:.1f} km",
            f"Final SOC: {self.battery.current_soc:.1%}",
            f"Total Distance: {self.total_distance_km:.1f} km",
            f"Charging Stops: {self.charging_stops}",
            f"Driver Type: {self.driver.behavior_type}",
            f"\nDetailed Event History:",
            f"{'-'*70}"
        ]
        
        for i, entry in enumerate(self.detailed_history, 1):
            lines.append(
                f"{i:3d}. [{entry['event_type']:12s}] {entry['details'][:50]:50s} "
                f"| pos={entry['position_km']:6.1f}km "
                f"| SOC={entry['soc']:5.1%} "
                f"| state={entry['state']}"
            )
        
        # Analyze why stranded
        if self.state == VehicleState.STRANDED:
            lines.extend(self._analyze_stranding())
        
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)
    
    def _analyze_stranding(self) -> List[str]:
        """Analyze why this vehicle got stranded."""
        analysis = [
            f"\n{'!'*70}",
            "STRANDING ANALYSIS:",
            f"{'!'*70}"
        ]
        
        # Find last charging stop
        last_charge_idx = None
        for i, entry in enumerate(self.detailed_history):
            if entry['event_type'] in ['CHARGE_START', 'CHARGE_COMPLETE']:
                last_charge_idx = i
        
        if last_charge_idx:
            charge_entry = self.detailed_history[last_charge_idx]
            analysis.append(f"Last charging event: {charge_entry['event_type']} "
                          f"at {charge_entry['position_km']:.1f}km, "
                          f"SOC={charge_entry['soc']:.1%}")
            
            # Check what happened after charging
            if last_charge_idx + 1 < len(self.detailed_history):
                next_events = self.detailed_history[last_charge_idx+1:last_charge_idx+6]
                analysis.append("Events after charging:")
                for evt in next_events:
                    analysis.append(f"  - {evt['event_type']}: {evt['details'][:60]}")
        
        # Check for abandonment events
        abandon_events = [e for e in self.detailed_history if 'abandon' in e['event_type'].lower()]
        if abandon_events:
            analysis.append(f"\nWARNING: Vehicle abandoned queue {len(abandon_events)} times!")
            for evt in abandon_events:
                analysis.append(f"  Abandoned at {evt['position_km']:.1f}km, "
                              f"SOC={evt['soc']:.1%}, reason={evt['details']}")
        
        # Check for missed stations
        approach_events = [e for e in self.detailed_history if e['event_type'] == 'APPROACH']
        if len(approach_events) > self.charging_stops:
            analysis.append(f"\nWARNING: Approached {len(approach_events)} stations "
                          f"but only charged {self.charging_stops} times!")
        
        # Calculate consumption pattern
        if len(self.detailed_history) > 1:
            total_dist = self.detailed_history[-1]['position_km'] - self.detailed_history[0]['position_km']
            total_soc_drop = self.detailed_history[0]['soc'] - self.detailed_history[-1]['soc']
            if total_dist > 0:
                avg_consumption = (total_soc_drop * self.battery.capacity_kwh) / (total_dist / 100)
                analysis.append(f"\nAverage consumption: {avg_consumption:.1f} kWh/100km")
        
        return analysis
    
    # ========================================================================
    # PHYSICAL DYNAMICS
    # ========================================================================
    
    def calculate_consumption_rate(self, speed_kmh: float) -> float:
        """
        Calculate energy consumption rate in kWh per km at given speed.
        Speed-dependent discharge rate using physics model.
        Results are cached by integer speed (driver aggression is constant).
        """
        if speed_kmh <= 0:
            return 0.0

        # Cache lookup by integer speed (sufficient precision)
        speed_key = int(speed_kmh)
        cached = self._consumption_cache.get(speed_key)
        if cached is not None:
            return cached

        speed_ms = speed_kmh / 3.6  # Convert to m/s

        # Aerodynamic drag (dominant at high speed)
        # P_drag = 0.5 * rho * Cd * A * v³
        drag_power = (0.5 * self.AIR_DENSITY * self.DRAG_COEFFICIENT *
                     self.FRONTAL_AREA_M2 * speed_ms ** 3)

        # Rolling resistance (constant)
        # P_roll = Crr * m * g * v
        rolling_power = (self.ROLLING_RESISTANCE * self.VEHICLE_MASS_KG *
                        self.GRAVITY * speed_ms)

        # Drivetrain efficiency (speed-dependent, peak around 50-80 km/h)
        efficiency = self._motor_efficiency(speed_kmh)

        # Total power in watts
        total_power_w = (drag_power + rolling_power) / efficiency

        # Add auxiliary loads (climate, electronics) - ~1-3 kW depending on weather
        aux_power_w = 1500  # Simplified constant

        total_power_kw = (total_power_w + aux_power_w) / 1000

        # Convert to kWh per km
        time_per_km_h = 1.0 / speed_kmh  # hours per km
        consumption_kwh_per_km = total_power_kw * time_per_km_h

        # Apply driver behavior modifier (aggressive driving)
        aggression_penalty = 1.0 + (self.driver.acceleration_aggression - 0.5) * 0.3

        result = consumption_kwh_per_km * aggression_penalty
        self._consumption_cache[speed_key] = result
        return result
    
    def _motor_efficiency(self, speed_kmh: float) -> float:
        """Electric motor efficiency curve by speed."""
        # Peak efficiency around 60 km/h, drops at extremes
        if speed_kmh < 10:
            return 0.75
        elif speed_kmh < 60:
            return 0.90 + (speed_kmh - 10) / 50 * 0.05  # 0.90 -> 0.95
        elif speed_kmh < 120:
            return 0.95 - (speed_kmh - 60) / 60 * 0.10  # 0.95 -> 0.85
        else:
            return 0.85 - (speed_kmh - 120) / 60 * 0.15  # 0.85 -> 0.70
    
    def update_physics(self, time_step_minutes: float, 
                      speed_limit_kmh: float = 130.0,
                      traffic_speed_kmh: Optional[float] = None) -> bool:
        """
        Update vehicle physics for one time step.
        Returns False if vehicle stranded (battery depleted).
        """
        dt_hours = time_step_minutes / 60.0
        dt_seconds = time_step_minutes * 60.0
        
        # Determine target speed
        env_limit = traffic_speed_kmh if traffic_speed_kmh else speed_limit_kmh
        self.target_speed_kmh = self.driver.calculate_desired_speed(env_limit)
        
        # Calculate acceleration (speed control)
        speed_diff = self.target_speed_kmh - self.speed_kmh
        max_accel = self.MAX_ACCELERATION_MS2 * 3.6 * dt_hours  # km/h per step
        max_decel = abs(self.COMFORT_DECELERATION_MS2) * 3.6 * dt_hours
        
        if speed_diff > 0:
            self.acceleration_ms2 = min(speed_diff / dt_hours / 3.6, 
                                       self.MAX_ACCELERATION_MS2)
        else:
            self.acceleration_ms2 = max(speed_diff / dt_hours / 3.6, 
                                       self.COMFORT_DECELERATION_MS2)
        
        # Update speed
        speed_change = self.acceleration_ms2 * dt_seconds / 3.6  # km/h
        old_speed = self.speed_kmh
        self.speed_kmh = max(0.0, self.speed_kmh + speed_change)
        
        # Update position using average of old and new speed (trapezoidal integration)
        avg_speed = (old_speed + self.speed_kmh) / 2.0
        distance_km = avg_speed * dt_hours
        old_position = self.position_km
        self.position_km += distance_km
        self.total_distance_km += distance_km

        # Calculate and apply energy consumption (use average speed for accuracy)
        consumption_rate = self.calculate_consumption_rate(avg_speed)
        energy_consumed = consumption_rate * distance_km
        
        old_soc = self.battery.current_soc
        success = self.battery.consume(energy_consumed)
        self.total_energy_consumed_kwh += energy_consumed
        
        # Log significant events
        if old_soc > 0.3 and self.battery.current_soc <= 0.3:
            self._log_event("LOW_BATTERY", f"SOC dropped to {self.battery.current_soc:.1%}")
        if old_soc > 0.15 and self.battery.current_soc <= 0.15:
            self._log_event("CRITICAL_BATTERY", f"SOC dropped to {self.battery.current_soc:.1%}")
        
        # Check for stranded condition
        if not success or self.battery.is_depleted:
            self.state = VehicleState.STRANDED
            self._log_event("STRANDED", 
                          f"Battery depleted at position {self.position_km:.1f}km, "
                          f"last speed={self.speed_kmh:.1f}km/h")
            return False
        
        return True
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    def set_state(self, new_state: VehicleState, timestamp: datetime, 
                  reason: str = ""):
        """Transition to new state with logging."""
        if self.state != new_state:
            old_state = self.state
            self._log_event("STATE_CHANGE", 
                          f"{old_state.name} -> {new_state.name}: {reason}",
                          {'old_state': old_state.name, 'new_state': new_state.name})
            if self.track_history:
                self.state_history.append({
                    'time': timestamp,
                    'from': old_state,
                    'to': new_state,
                    'reason': reason,
                    'position': self.position_km,
                    'soc': self.battery.current_soc
                })
            self.state = new_state
            self.state_entry_time = timestamp
    
    def _log_state(self, event: str, details: str = ""):
        """Internal state logging."""
        if not self.track_history:
            return
        self.state_history.append({
            'event': event,
            'details': details,
            'soc': self.battery.current_soc,
            'position': self.position_km
        })
    
    # ========================================================================
    # DECISION SUPPORT
    # ========================================================================
    
    def get_range_estimate(self) -> float:
        """Estimate remaining range at current speed."""
        current_consumption = self.calculate_consumption_rate(self.speed_kmh)
        return self.battery.estimate_range_km(current_consumption)
    
    def _get_conservative_consumption(self, speed_kmh: float) -> float:
        """
        Get consumption rate with safety margin for planning.
        Uses standardized CONSUMPTION_SAFETY_FACTOR instead of arbitrary multipliers.
        """
        base_consumption = self.calculate_consumption_rate(speed_kmh)
        return base_consumption * self.CONSUMPTION_SAFETY_FACTOR
    
    def _estimate_range_at_speed(self, speed_kmh: float, apply_safety: bool = True) -> float:
        """
        Unified range estimation with optional safety margin.
        
        Args:
            speed_kmh: Speed for estimation
            apply_safety: If True, applies CONSUMPTION_SAFETY_FACTOR (for planning)
                         If False, uses raw consumption (for actual physics)
        """
        if apply_safety:
            consumption = self._get_conservative_consumption(speed_kmh)
        else:
            consumption = self.calculate_consumption_rate(speed_kmh)
        
        return self.battery.estimate_range_km(consumption)
    
    def can_reach_station(self, station_position_km: float,
                         buffer_km: Optional[float] = None) -> bool:
        """Check if vehicle can reach a station with safety buffer."""
        buffer = buffer_km or self.driver.buffer_margin_km
        distance = abs(station_position_km - self.position_km)
        required_range = distance + buffer
        return self.get_range_estimate() >= required_range
    
    def can_physically_reach(self, position_km: float, at_highway_speed: bool = True) -> bool:
        """
        Check if vehicle can physically reach a position.
        
        Args:
            position_km: Target position to reach
            at_highway_speed: If True, calculate range at driver's preferred highway speed
        """
        distance = abs(position_km - self.position_km)
        
        if at_highway_speed:
            # Use driver's preferred speed for realistic range estimate
            speed_for_estimate = self.driver.speed_preference_kmh
        else:
            speed_for_estimate = self.speed_kmh
            
        # Use standardized method with 15% safety margin
        estimated_range = self._estimate_range_at_speed(speed_for_estimate, apply_safety=True)
        
        can_reach = estimated_range >= distance
        
        # Debug logging for critical decisions
        if not can_reach and self.battery.current_soc < 0.4:
            self._log_event("CANNOT_REACH", 
                          f"Cannot reach {position_km:.1f}km (dist={distance:.1f}km, "
                          f"range={estimated_range:.1f}km)",
                          {'target': position_km, 'distance': distance, 
                           'estimated_range': estimated_range})
        
        return can_reach

    def must_stop_at_station(self, current_station_km: float,
                             next_station_km: Optional[float],
                             highway_end_km: Optional[float] = None) -> bool:
        """
        Determine if vehicle MUST stop at this station to avoid stranding.
        
        Returns True if skipping this station would result in stranding.
        """
        # First, can we even reach this station?
        if not self.can_physically_reach(current_station_km, at_highway_speed=False):
            return True  # Must try to stop here, already in trouble

        # Use standardized range estimation with safety margin
        # This ensures consistency between "can I reach" and "must I stop" decisions
        consumption_at_highway = self._get_conservative_consumption(self.driver.speed_preference_kmh)
        range_at_highway = self.battery.estimate_range_km(consumption_at_highway)

        # If there's a next station, check if we can reach it after passing this one
        if next_station_km is not None:
            # Calculate remaining range if we drive to current station and pass it
            distance_to_current = abs(current_station_km - self.position_km)
            remaining_range_after_current = range_at_highway - distance_to_current
            distance_current_to_next = abs(next_station_km - current_station_km)

            # Add mandatory SAFETY_MARGIN_KM (20km) to required distance
            # This prevents the "15km short" problem
            required_range_with_safety = distance_current_to_next + self.SAFETY_MARGIN_KM

            # Can we make it from current station to next station with safety margin?
            if remaining_range_after_current < required_range_with_safety:
                self._log_event("MUST_STOP", 
                              f"Cannot reach next station at {next_station_km:.1f}km "
                              f"(need {required_range_with_safety:.1f}km, "
                              f"have {remaining_range_after_current:.1f}km)",
                              {'current_station': current_station_km,
                               'next_station': next_station_km,
                               'required_range': required_range_with_safety,
                               'available_range': remaining_range_after_current})
                return True  # Must stop here or we strand before next station

        # If no next station, check if we can reach highway end
        if next_station_km is None and highway_end_km is not None:
            distance_to_current = abs(current_station_km - self.position_km)
            remaining_range_after_current = range_at_highway - distance_to_current
            distance_current_to_end = abs(highway_end_km - current_station_km)

            # Add safety margin for highway end too
            required_range_with_safety = distance_current_to_end + self.SAFETY_MARGIN_KM

            if remaining_range_after_current < required_range_with_safety:
                self._log_event("MUST_STOP_END", 
                              f"Cannot reach highway end at {highway_end_km:.1f}km",
                              {'required_range': required_range_with_safety,
                               'available_range': remaining_range_after_current})
                return True  # Must stop here or we strand before highway end

        return False  # Safe to skip if we want to

    def needs_charging(self, next_station_km: Optional[float] = None) -> bool:
        """Determine if vehicle needs to charge now."""
        # Critical battery
        if self.battery.is_critical:
            return True
        
        # Check if can reach next station
        if next_station_km:
            if not self.can_reach_station(next_station_km):
                return True
        
        # Comfort threshold (driver behavior)
        if self.battery.current_soc < self.driver.min_charge_to_continue:
            return True
        
        return False
    
    def evaluate_charging_decision(self, station_options: list) -> Optional[Dict]:
        """
        Evaluate whether and where to charge.
        Returns chosen station or None to skip.
        """
        if not self.needs_charging():
            return None
        
        best_option = None
        best_score = 0.0
        
        for station in station_options:
            score = self.driver.evaluate_station_attractiveness(
                station,
                self.battery.current_soc,
                station.get('estimated_wait', 0)
            )
            
            # Range check: can we actually get there?
            if not self.can_reach_station(station['location_km']):
                score *= 0.1  # Penalty but not zero (tow truck possibility)
            
            if score > best_score:
                best_score = score
                best_option = station
        
        # Threshold to decide (fuzzy: if best is too low, risk continuing)
        if best_score < 0.3 and not self.battery.is_critical:
            return None  # Driver chooses to continue
        
        return best_option
    
    # ========================================================================
    # CHARGING INTERACTION
    # ========================================================================
    
    def start_charging(self, charger_id: str, power_kw: float, 
                      timestamp: datetime):
        """Begin charging session."""
        self.assigned_charger_id = charger_id
        self.charging_stops += 1
        self.set_state(VehicleState.CHARGING, timestamp, f"at {charger_id}")
        self._log_event("CHARGE_START", f"Charger {charger_id}, power={power_kw}kW")
    
    def charge_step(self, energy_kwh: float, timestamp: datetime) -> bool:
        """
        Add energy during charging step.
        Returns True if target SOC reached.
        """
        actual = self.battery.charge(energy_kwh)
        
        if self.battery.current_soc >= self.driver.target_soc:
            self._log_event("CHARGE_TARGET_REACHED", 
                          f"SOC={self.battery.current_soc:.1%}, target={self.driver.target_soc:.1%}")
            return True  # Charged enough
        
        return False
    
    def finish_charging(self, timestamp: datetime, 
                       total_time_min: float):
        """Complete charging session."""
        self.total_charging_time_min += total_time_min
        self.assigned_charger_id = None
        self.target_station_id = None
        self.current_patience = self.driver.patience_base  # Reset patience
        self.set_state(VehicleState.EXITING, timestamp, "charging complete")
        self._log_event("CHARGE_COMPLETE", 
                      f"Duration={total_time_min:.1f}min, final SOC={self.battery.current_soc:.1%}")
    
    def abandon_charging(self, timestamp: datetime, reason: str):
        """Leave queue or charging early."""
        self._log_event("ABANDON", f"Reason: {reason}")
        self.set_state(VehicleState.EXITING, timestamp, f"abandoned: {reason}")
        self.target_station_id = None
        self.queue_entry_time = None
    
    def update_patience(self, wait_minutes: float, comfort_level: float):
        """Update patience while waiting."""
        urgency = 1.0 - self.battery.usable_soc
        self.current_patience = self.driver.update_patience(
            self.current_patience,
            wait_minutes,
            comfort_level,
            urgency
        )
    
    # ========================================================================
    # SIMULATION STEP
    # ========================================================================
    
    def step(self, timestamp: datetime, time_step_minutes: float = 1.0,
             environment: Optional[Dict] = None, skip_physics: bool = False) -> Dict:
        """
        Execute one simulation step.
        """
        result = {
            'state_changed': False,
            'stranded': False,
            'needs_decision': False,
            'decision_options': [],
            'abandoned': False,
            'charging_complete': False
        }

        # Use empty dict if None, with safe defaults for all accessed keys
        env = environment or {}
        # Define defaults for commonly accessed keys
        env.setdefault('speed_limit_kmh', 130.0)
        env.setdefault('upcoming_stations', [])
        env.setdefault('station_comfort', 0.5)
        env.setdefault('queue_position', 99)
        env.setdefault('estimated_wait', 30)
        env.setdefault('alternative_available', False)
        env.setdefault('highway_end_km', None)
        env.setdefault('can_reach_next_station', True)

        # Physics update (unless charging/queued or already done in batch)
        if self.state in [VehicleState.CRUISING, VehicleState.APPROACHING,
                         VehicleState.EXITING]:

            if not skip_physics:
                speed_limit = env['speed_limit_kmh']
                traffic_speed = env.get('traffic_speed_kmh')  # Optional, can be None

                success = self.update_physics(time_step_minutes, speed_limit, traffic_speed)

                if not success:
                    result['stranded'] = True
                    return result

            # Check for station approach and proactive charging decision
            upcoming_stations = env['upcoming_stations']
            highway_end_km = env.get('highway_end_km')

            # Filter stations still ahead after physics update (already sorted by location)
            stations_ahead = [s for s in upcoming_stations
                            if s['location_km'] > self.position_km]

            # Proactive charging decision
            if self.state == VehicleState.CRUISING and stations_ahead:
                for i, station in enumerate(stations_ahead):
                    station_km = station['location_km']
                    # Find the next station after this one (if any)
                    next_station_km = stations_ahead[i + 1]['location_km'] if i + 1 < len(stations_ahead) else None

                    # Check if we can physically reach this station
                    if not self.can_physically_reach(station_km):
                        continue  # Can't reach this one, try further ones (shouldn't happen)

                    # MUST STOP: Check if skipping this station would strand us
                    must_stop = self.must_stop_at_station(station_km, next_station_km, highway_end_km)

                    # WANT TO STOP: Check comfort-based charging need
                    wants_to_stop = self.needs_charging(next_station_km)

                    if must_stop or wants_to_stop:
                        self.target_station_id = station.get('area_id', station.get('id'))
                        reason = "must charge (no alternative)" if must_stop else "seeking charging"
                        self.set_state(VehicleState.APPROACHING, timestamp,
                                     f"{reason} at {self.target_station_id}")
                        self._log_event("APPROACH", 
                                      f"Station at {station_km:.1f}km, reason={reason}, "
                                      f"next_station={next_station_km}")
                        result['state_changed'] = True
                        result['needs_decision'] = True
                        result['decision_options'] = upcoming_stations
                        result['must_stop'] = must_stop  # Signal to Highway that this is mandatory
                        break

            # Check if within 10km of any station (normal approach trigger)
            for i, station in enumerate(stations_ahead):
                distance = station['location_km'] - self.position_km
                if 0 < distance <= 10.0:  # Within 10km
                    if self.state != VehicleState.APPROACHING:
                        station_id = station.get('area_id', station.get('id', 'unknown'))
                        next_station_km = stations_ahead[i + 1]['location_km'] if i + 1 < len(stations_ahead) else None

                        # Check if this is a must-stop situation
                        must_stop = self.must_stop_at_station(station['location_km'], next_station_km, highway_end_km)
                        if must_stop:
                            self.target_station_id = station_id

                        self.set_state(VehicleState.APPROACHING, timestamp,
                                     f"approaching {station_id}")
                        self._log_event("APPROACH_NEAR", 
                                      f"Within {distance:.1f}km of {station_id}, must_stop={must_stop}")
                        result['state_changed'] = True
                    break

            # Check if needs charging decision
            if self.state == VehicleState.APPROACHING:
                # Find current target station info
                for i, station in enumerate(stations_ahead):
                    if station.get('area_id', station.get('id')) == self.target_station_id:
                        next_station_km = stations_ahead[i + 1]['location_km'] if i + 1 < len(stations_ahead) else None
                        if self.needs_charging(next_station_km) or \
                           self.must_stop_at_station(station['location_km'], next_station_km, highway_end_km):
                            result['needs_decision'] = True
                            result['decision_options'] = upcoming_stations
                        break
                else:
                    # Target station not in list, check general need
                    if self.needs_charging():
                        result['needs_decision'] = True
                        result['decision_options'] = upcoming_stations
        
        elif self.state == VehicleState.QUEUED:
            # Update waiting experience
            if self.queue_entry_time:
                wait_min = (timestamp - self.queue_entry_time).total_seconds() / 60
                comfort = env['station_comfort']
                self.update_patience(wait_min, comfort)

                # CRITICAL FIX: Check if we can reach next station before allowing abandonment
                can_abandon_safely = env.get('can_reach_next_station', True)
                
                # If we cannot reach next station, we MUST stay in queue
                if not can_abandon_safely:
                    # Force patience to stay above critical level
                    self.current_patience = max(self.current_patience, 0.15)
                    self._log_event("QUEUE_STUCK", 
                                  "Cannot abandon - would strand. Staying in queue.")
                    # Skip abandonment check entirely
                else:
                    # Safe to abandon, check patience
                    should_abandon = self.driver.decide_to_abort(
                        self.current_patience,
                        env['queue_position'],
                        env['estimated_wait'],
                        env['alternative_available']
                    )

                    if should_abandon:
                        self._log_event("ABANDON_DECISION", 
                                      f"Patience={self.current_patience:.2f}, "
                                      f"queue_pos={env['queue_position']}, "
                                      f"wait={env['estimated_wait']:.1f}min")
                        self.abandon_charging(timestamp, "patience depleted")
                        result['abandoned'] = True
                        result['state_changed'] = True
        
        elif self.state == VehicleState.CHARGING:
            # Charging handled by station, but check completion
            if env.get('charging_complete', False):
                self.finish_charging(timestamp, env.get('session_duration', 0))
                result['charging_complete'] = True
                result['state_changed'] = True
        
        # Record history (every step, if enabled)
        if self.track_history:
            self.position_history.append({
                'time': timestamp,
                'position': self.position_km,
                'speed': self.speed_kmh,
                'soc': self.battery.current_soc,
                'state': self.state
            })
        
        return result
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def get_status(self) -> Dict:
        """Current vehicle status."""
        return {
            'id': self.id,
            'state': self.state.name,
            'position_km': self.position_km,
            'speed_kmh': self.speed_kmh,
            'soc': self.battery.current_soc,
            'usable_soc': self.battery.usable_soc,
            'range_estimate_km': self.get_range_estimate(),
            'current_patience': self.current_patience,
            'target_station': self.target_station_id,
            'driver_type': self.driver.behavior_type
        }
    
    def get_trip_summary(self) -> Dict:
        """Summary statistics for completed trip."""
        return {
            'vehicle_id': self.id,
            'total_distance_km': self.total_distance_km,
            'total_energy_consumed_kwh': self.total_energy_consumed_kwh,
            'avg_consumption_kwh_per_100km': 
                (self.total_energy_consumed_kwh / max(1, self.total_distance_km)) * 100,
            'charging_stops': self.charging_stops,
            'total_charging_time_min': self.total_charging_time_min,
            'final_soc': self.battery.current_soc,
            'was_stranded': self.state == VehicleState.STRANDED,
            'state_transitions': len(self.state_history)
        }
    
    def __repr__(self) -> str:
        return (f"Vehicle({self.id}, SOC={self.battery.current_soc:.1%}, "
                f"pos={self.position_km:.1f}km, {self.state.name}, "
                f"driver={self.driver.behavior_type})")