"""
Vehicle Module
Electric vehicle agent with physical dynamics and driver behavior.
"""

from __future__ import annotations

import uuid
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Callable, Tuple
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
    capacity_kwh: float = 60.0          # Total capacity
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
    This is a separate class as requested, encapsulating all human factors.
    """
    
    # Identification
    driver_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
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
        This will integrate with fuzzy logic system later.
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
    
    def update_patience(self, current_patience: float, wait_minutes: float,
                       comfort_level: float, battery_urgency: float) -> float:
        """
        Update patience based on waiting experience.
        Returns new patience level (0-1).
        """
        # Base decay
        decay = self.patience_decay_rate * (wait_minutes / 10)  # Per 10 min
        
        # Comfort modifier (better comfort = slower decay)
        comfort_factor = 1.0 - comfort_level * 0.5
        
        # Urgency modifier (urgent = faster patience loss)
        urgency_factor = 1.0 + battery_urgency * 2.0
        
        new_patience = current_patience - (decay * comfort_factor * urgency_factor)
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
        
        import random
        return random.random() < min(0.95, abandonment_prob)
    
    def __repr__(self) -> str:
        return (f"DriverBehavior({self.behavior_type}, "
                f"patience={self.patience_base:.1f}, "
                f"risk={self.risk_tolerance:.1f})")


class Vehicle:
    """
    Electric vehicle agent with physical dynamics and driver behavior.
    
    Physical attributes:
    - Battery: charge state, capacity, consumption
    - Speed: current velocity with acceleration limits
    - Position: location on highway (km from origin)
    - Discharge rate: speed-dependent energy consumption
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
    
    def __init__(
        self,
        vehicle_id: Optional[str] = None,
        battery_capacity_kwh: float = 60.0,
        initial_soc: Optional[float] = None,
        driver_behavior: Optional[DriverBehavior] = None,
        initial_position_km: float = 0.0,
        initial_speed_kmh: float = 0.0
    ):
        self.id = vehicle_id or str(uuid.uuid4())[:8]
        
        # Battery initialization with random SOC if not specified
        if initial_soc is None:
            # Distribution: mostly mid-range, some low, some high
            import random
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
        
        # History for analysis
        self.position_history: list = []  # [(time, pos, speed, soc)]
        self.state_history: list = []     # [(time, state, reason)]
        
        self._log_state("initialized")
    
    # ========================================================================
    # PHYSICAL DYNAMICS
    # ========================================================================
    
    def calculate_consumption_rate(self, speed_kmh: float) -> float:
        """
        Calculate energy consumption rate in kWh per km at given speed.
        Speed-dependent discharge rate using physics model.
        """
        if speed_kmh <= 0:
            return 0.0
        
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
        
        return consumption_kwh_per_km * aggression_penalty
    
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
        self.speed_kmh = max(0.0, self.speed_kmh + speed_change)
        
        # Update position
        distance_km = self.speed_kmh * dt_hours
        self.position_km += distance_km
        self.total_distance_km += distance_km
        
        # Calculate and apply energy consumption
        consumption_rate = self.calculate_consumption_rate(self.speed_kmh)
        energy_consumed = consumption_rate * distance_km
        
        success = self.battery.consume(energy_consumed)
        self.total_energy_consumed_kwh += energy_consumed
        
        # Check for stranded condition
        if not success or self.battery.is_depleted:
            self.state = VehicleState.STRANDED
            self._log_state("stranded", f"Battery depleted at {self.position_km:.1f}km")
            return False
        
        return True
    
    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================
    
    def set_state(self, new_state: VehicleState, timestamp: datetime, 
                  reason: str = ""):
        """Transition to new state with logging."""
        if self.state != new_state:
            self.state_history.append({
                'time': timestamp,
                'from': self.state,
                'to': new_state,
                'reason': reason,
                'position': self.position_km,
                'soc': self.battery.current_soc
            })
            self.state = new_state
            self.state_entry_time = timestamp
    
    def _log_state(self, event: str, details: str = ""):
        """Internal state logging."""
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
    
    def can_reach_station(self, station_position_km: float, 
                         buffer_km: Optional[float] = None) -> bool:
        """Check if vehicle can reach a station with safety buffer."""
        buffer = buffer_km or self.driver.buffer_margin_km
        distance = abs(station_position_km - self.position_km)
        required_range = distance + buffer
        return self.get_range_estimate() >= required_range
    
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
    
    def charge_step(self, energy_kwh: float, timestamp: datetime) -> bool:
        """
        Add energy during charging step.
        Returns True if target SOC reached.
        """
        actual = self.battery.charge(energy_kwh)
        
        if self.battery.current_soc >= self.driver.target_soc:
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
    
    def abandon_charging(self, timestamp: datetime, reason: str):
        """Leave queue or charging early."""
        self.set_state(VehicleState.CRUISING, timestamp, f"abandoned: {reason}")
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
             environment: Optional[Dict] = None) -> Dict:
        """
        Execute one simulation step.
        
        Returns dict with state changes and alerts.
        """
        result = {
            'state_changed': False,
            'stranded': False,
            'needs_decision': False,
            'decision_options': [],
            'abandoned': False,
            'charging_complete': False
        }
        
        env = environment or {}
        
        # Physics update (unless charging/queued)
        if self.state in [VehicleState.CRUISING, VehicleState.APPROACHING, 
                         VehicleState.EXITING]:
            
            speed_limit = env.get('speed_limit_kmh', 130.0)
            traffic_speed = env.get('traffic_speed_kmh')
            
            success = self.update_physics(time_step_minutes, speed_limit, traffic_speed)
            
            if not success:
                result['stranded'] = True
                return result
            
            # Check for station approach
            upcoming_stations = env.get('upcoming_stations', [])
            for station in upcoming_stations:
                distance = station['location_km'] - self.position_km
                if 0 < distance <= 10.0:  # Within 10km
                    if self.state != VehicleState.APPROACHING:
                        self.set_state(VehicleState.APPROACHING, timestamp, 
                                     f"approaching {station['id']}")
                        result['state_changed'] = True
                    break
            
            # Check if needs charging decision
            if self.state == VehicleState.APPROACHING and self.needs_charging():
                result['needs_decision'] = True
                result['decision_options'] = upcoming_stations
        
        elif self.state == VehicleState.QUEUED:
            # Update waiting experience
            if self.queue_entry_time:
                wait_min = (timestamp - self.queue_entry_time).total_seconds() / 60
                comfort = env.get('station_comfort', 0.5)
                self.update_patience(wait_min, comfort)
                
                # Check abandonment
                should_abandon = self.driver.decide_to_abort(
                    self.current_patience,
                    env.get('queue_position', 99),
                    env.get('estimated_wait', 30),
                    env.get('alternative_available', False)
                )
                
                if should_abandon:
                    self.abandon_charging(timestamp, "patience depleted")
                    result['abandoned'] = True
                    result['state_changed'] = True
        
        elif self.state == VehicleState.CHARGING:
            # Charging handled by station, but check completion
            if env.get('charging_complete', False):
                self.finish_charging(timestamp, env.get('session_duration', 0))
                result['charging_complete'] = True
                result['state_changed'] = True
        
        # Record history (every step)
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


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def demo():
    """Demonstrate Vehicle and DriverBehavior functionality."""
    print("=" * 70)
    print("VEHICLE & DRIVER BEHAVIOR DEMO")
    print("=" * 70)
    
    # Create different driver types
    drivers = {
        'conservative': DriverBehavior(
            behavior_type='conservative',
            patience_base=0.8,
            risk_tolerance=0.2,
            buffer_margin_km=80.0,
            target_soc=0.9
        ),
        'aggressive': DriverBehavior(
            behavior_type='aggressive',
            patience_base=0.4,
            risk_tolerance=0.8,
            speed_preference_kmh=140.0,
            acceleration_aggression=0.8,
            buffer_margin_km=20.0,
            target_soc=0.6
        ),
        'range_anxious': DriverBehavior(
            behavior_type='range_anxious',
            patience_base=0.6,
            risk_tolerance=0.1,
            buffer_margin_km=100.0,
            min_charge_to_continue=0.4,
            social_influence=0.7
        )
    }
    
    # Create vehicles with different drivers
    vehicles = []
    start_time = datetime(2024, 1, 15, 9, 0)
    
    print("\n--- Creating Vehicles ---")
    for driver_type, driver in drivers.items():
        vehicle = Vehicle(
            battery_capacity_kwh=75.0,
            initial_soc=0.45,  # Start with low battery to force decisions
            driver_behavior=driver,
            initial_position_km=100.0,
            initial_speed_kmh=120.0
        )
        vehicles.append((driver_type, vehicle))
        print(f"\n{driver_type.upper()}:")
        print(f"  Driver: {driver}")
        print(f"  Vehicle: {vehicle}")
        print(f"  Range estimate: {vehicle.get_range_estimate():.1f} km")
        print(f"  Needs charging: {vehicle.needs_charging(next_station_km=150.0)}")
    
    # Simulate driving
    print("\n--- Simulating 10 minutes of driving ---")
    for minute in range(10):
        timestamp = start_time + timedelta(minutes=minute)
        print(f"\n{timestamp.strftime('%H:%M')}:")
        
        for driver_type, vehicle in vehicles:
            result = vehicle.step(
                timestamp=timestamp,
                time_step_minutes=1.0,
                environment={
                    'speed_limit_kmh': 130.0,
                    'upcoming_stations': [
                        {'id': 'STA-001', 'location_km': 125.0, 
                         'estimated_wait': 15.0, 'price_per_kwh': 0.45}
                    ]
                }
            )
            
            status = vehicle.get_status()
            print(f"  {driver_type:12s}: pos={status['position_km']:6.1f}km, "
                  f"SOC={status['soc']:.1%}, {status['state'][:4]}, "
                  f"pat={status['current_patience']:.2f}")
            
            if result['needs_decision']:
                print(f"    *** Decision needed! ***")
    
    # Test consumption rates
    print("\n--- Consumption vs Speed ---")
    test_vehicle = Vehicle(battery_capacity_kwh=60.0, initial_soc=0.8)
    for speed in [30, 50, 80, 100, 120, 150]:
        rate = test_vehicle.calculate_consumption_rate(speed)
        print(f"  {speed:3d} km/h: {rate*100:.2f} kWh/100km")
    
    # Test patience decay
    print("\n--- Patience Decay Simulation ---")
    anxious_driver = drivers['range_anxious']
    patience = anxious_driver.patience_base
    for wait in [0, 5, 10, 15, 20, 25, 30]:
        new_patience = anxious_driver.update_patience(
            patience, wait, comfort_level=0.5, battery_urgency=0.8
        )
        print(f"  Wait {wait:2d}min: patience {patience:.2f} → {new_patience:.2f}")
        patience = new_patience
    
    print("\n--- Final Status ---")
    for driver_type, vehicle in vehicles:
        print(f"\n{driver_type}:")
        print(f"  {vehicle.get_status()}")


if __name__ == "__main__":
    demo()