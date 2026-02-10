"""
Vehicle Generator Module
Sophisticated vehicle generation with non-uniform temporal distribution.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque

import numpy as np

from ev_charge_manager.simulation.environment import Environment, SimulationConfig
from ev_charge_manager.vehicle import Vehicle, DriverBehavior
from ev_charge_manager.charging import ChargingArea, ChargerType


class TemporalDistribution(Enum):
    """Distribution patterns for vehicle arrivals over time."""
    UNIFORM = auto()           # Evenly spread
    RANDOM_POISSON = auto()    # Poisson process (memoryless)
    RUSH_HOUR = auto()      # Morning and evening peaks (bimodal)
    SINGLE_PEAK = auto()       # One peak period
    PERIODIC = auto()          # Regular oscillation
    BURSTY = auto()            # Clusters with gaps


@dataclass
class VehicleTypeSpec:
    """Specification for a vehicle type in the fleet."""
    name: str
    battery_capacity_kwh: float
    weight: float = 1.0  # Relative frequency
    soc_distribution: Callable[[], float] = field(
        default_factory=lambda: lambda: random.uniform(0.5, 0.9)
    )


@dataclass
class BehaviorSpec:
    """Specification for a driver behavior type."""
    name: str
    weight: float = 1.0
    parameter_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratorConfig:
    """Configuration for the vehicle generator."""
    # Base rate
    vehicles_per_hour: float = 60.0
    
    # Temporal distribution
    distribution_type: TemporalDistribution = TemporalDistribution.RANDOM_POISSON
    
    # Rush hour parameters (for双峰 distribution)
    morning_rush_peak: float = 8.0      # 8 AM peak
    morning_rush_width: float = 2.0     # 2 hours std dev
    evening_rush_peak: float = 18.0     # 6 PM peak
    evening_rush_width: float = 2.5     # 2.5 hours std dev
    rush_hour_ratio: float = 3.0        # Peak is 3x base rate
    
    # Single peak parameters
    peak_hour: float = 12.0             # Noon peak
    peak_width: float = 3.0             # 3 hours width
    
    # Burst parameters
    burst_probability: float = 0.3      # 30% chance of burst
    burst_size_multiplier: float = 5.0  # 5x normal during burst
    burst_duration_minutes: float = 10.0
    
    # Fleet composition
    vehicle_types: List[VehicleTypeSpec] = field(default_factory=lambda: [
        VehicleTypeSpec("compact", 40.0, 0.25, lambda: random.uniform(0.4, 0.9)),
        VehicleTypeSpec("midsize", 60.0, 0.40, lambda: random.triangular(0.3, 0.9, 0.5)),
        VehicleTypeSpec("premium", 80.0, 0.25, lambda: random.uniform(0.5, 0.95)),
        VehicleTypeSpec("truck", 100.0, 0.10, lambda: random.uniform(0.6, 0.9))
    ])
    
    behavior_types: List[BehaviorSpec] = field(default_factory=lambda: [
        BehaviorSpec("conservative", 0.25),
        BehaviorSpec("balanced", 0.40),
        BehaviorSpec("aggressive", 0.25),
        BehaviorSpec("range_anxious", 0.10)
    ])
    
    # Random seed
    random_seed: Optional[int] = None


class VehicleGenerator:
    """
    Generates vehicles with controlled temporal distribution and fleet composition.
    
    Key features:
    - Non-uniform temporal distribution (peaks, bursts, etc.)
    - Maintains exact hourly totals on average
    - Configurable fleet composition
    - Integrates with Environment for spawning
    """
    
    def __init__(
        self,
        environment: Environment,
        config: Optional[GeneratorConfig] = None
    ):
        self.env = environment
        self.config = config or GeneratorConfig()
        
        # Initialize random state
        if self.config.random_seed:
            random.seed(self.config.random_seed)
        
        # Time tracking
        self.current_hour: int = environment.current_time.hour
        self.hourly_target: float = self.config.vehicles_per_hour
        self.hourly_generated: int = 0
        self.hourly_schedule: List[int] = []  # Minute-by-minute schedule for current hour
        
        # Accumulator for fractional vehicles
        self._accumulator: float = 0.0
        
        # Burst state
        self._in_burst: bool = False
        self._burst_end_time: Optional[datetime] = None
        
        # Statistics
        self.generation_history: deque = deque(maxlen=1440)  # 24 hours
        self.total_generated: int = 0
        self.generation_log: List[Dict] = []
        
        # Pre-compute first hour's schedule
        self._generate_hourly_schedule()
    
    # ========================================================================
    # SCHEDULE GENERATION
    # ========================================================================
    
    def _generate_hourly_schedule(self) -> None:
        """
        Generate minute-by-minute vehicle counts for the upcoming hour.
        Ensures total matches target while allowing non-uniform distribution.
        """
        self.hourly_generated = 0
        minutes = 60
        target = self.hourly_target
        
        if self.config.distribution_type == TemporalDistribution.UNIFORM:
            # Even distribution: target/60 per minute with rounding
            base = int(target // 60)
            remainder = target % 60
            schedule = [base] * minutes
            # Distribute remainder randomly
            for _ in range(int(remainder)):
                schedule[random.randint(0, 59)] += 1
                
        elif self.config.distribution_type == TemporalDistribution.RANDOM_POISSON:
            # Poisson: each minute is independent draw with mean target/60
            mean_per_minute = target / 60
            schedule = list(np.random.poisson(mean_per_minute, minutes))
            
        elif self.config.distribution_type == TemporalDistribution.RUSH_HOUR:
            # Bimodal: morning and evening peaks
            schedule = self._generate_bimodal_schedule(target)
            
        elif self.config.distribution_type == TemporalDistribution.SINGLE_PEAK:
            # Single peak at specified hour
            schedule = self._generate_single_peak_schedule(target)
            
        elif self.config.distribution_type == TemporalDistribution.PERIODIC:
            # Sinusoidal variation
            schedule = self._generate_periodic_schedule(target)
            
        elif self.config.distribution_type == TemporalDistribution.BURSTY:
            # Random bursts
            schedule = self._generate_bursty_schedule(target)
            
        else:
            # Default to uniform
            schedule = [int(target // 60)] * minutes
        
        self.hourly_schedule = schedule
        
        # Verify total (may need adjustment due to randomness)
        current_total = sum(schedule)
        if abs(current_total - target) > 1 and target > 0:
            # Adjust to hit target more precisely
            diff = int(target) - current_total
            self._adjust_schedule(diff)
    
    def _generate_bimodal_schedule(self, target: float) -> List[int]:
        """Generate schedule with morning and evening rush hour peaks."""
        minutes = 60
        current_hour = self.current_hour
        
        # Calculate base rate (trough rate)
        peak_ratio = self.config.rush_hour_ratio
        base_rate = target / (60 * ((peak_ratio - 1) * 0.3 + 1))  # Approximate normalization
        
        schedule = []
        for minute in range(minutes):
            hour_frac = current_hour + minute / 60
            
            # Morning peak contribution
            morning = math.exp(
                -0.5 * ((hour_frac - self.config.morning_rush_peak) / 
                       self.config.morning_rush_width) ** 2
            )
            
            # Evening peak contribution
            evening = math.exp(
                -0.5 * ((hour_frac - self.config.evening_rush_peak) / 
                       self.config.evening_rush_width) ** 2
            )
            
            # Combined intensity (0 to 1 scale, scaled to peak_ratio)
            intensity = 1 + (peak_ratio - 1) * max(morning, evening)
            minute_rate = base_rate * intensity
            
            # Poisson draw
            count = int(np.random.poisson(minute_rate)) if minute_rate > 0 else 0
            schedule.append(count)
        
        return schedule
    
    def _generate_single_peak_schedule(self, target: float) -> List[int]:
        """Generate schedule with single peak period."""
        minutes = 60
        current_hour = self.current_hour
        
        peak_ratio = self.config.rush_hour_ratio
        base_rate = target / (60 * ((peak_ratio - 1) * 0.4 + 1))
        
        schedule = []
        for minute in range(minutes):
            hour_frac = current_hour + minute / 60
            
            # Gaussian peak
            peak = math.exp(
                -0.5 * ((hour_frac - self.config.peak_hour) / 
                       self.config.peak_width) ** 2
            )
            
            intensity = 1 + (peak_ratio - 1) * peak
            minute_rate = base_rate * intensity
            count = int(np.random.poisson(minute_rate)) if minute_rate > 0 else 0
            schedule.append(count)
        
        return schedule
    
    def _generate_periodic_schedule(self, target: float) -> List[int]:
        """Generate schedule with periodic oscillation."""
        minutes = 60
        period = 2.0  # 2-hour cycle
        
        schedule = []
        for minute in range(minutes):
            hour_frac = self.current_hour + minute / 60
            # Sinusoidal variation ±30% around mean
            variation = 0.3 * math.sin(2 * math.pi * hour_frac / period)
            minute_rate = (target / 60) * (1 + variation)
            count = max(0, int(np.random.poisson(max(0, minute_rate))))
            schedule.append(count)
        
        return schedule
    
    def _generate_bursty_schedule(self, target: float) -> List[int]:
        """Generate schedule with random burst periods."""
        minutes = 60
        base_rate = target / 60
        schedule = []
        
        burst_active = False
        burst_remaining = 0
        
        for minute in range(minutes):
            # Check for burst start
            if not burst_active and random.random() < self.config.burst_probability / 6:
                # Start burst (divide prob by 6 to get ~prob per 10-min block)
                burst_active = True
                burst_remaining = int(self.config.burst_duration_minutes)
            
            # Determine rate
            if burst_active:
                rate = base_rate * self.config.burst_size_multiplier
                burst_remaining -= 1
                if burst_remaining <= 0:
                    burst_active = False
            else:
                rate = base_rate
            
            count = int(np.random.poisson(rate)) if rate > 0 else 0
            schedule.append(count)
        
        return schedule
    
    def _adjust_schedule(self, diff: int) -> None:
        """
        Adjust schedule to hit exact target.
        diff > 0: need to add vehicles, diff < 0: need to remove
        """
        if diff == 0:
            return
        
        # Add/remove randomly, preferring minutes with fewer vehicles
        # to maintain distribution shape
        minutes = list(range(60))
        
        if diff > 0:
            # Add vehicles to random minutes
            for _ in range(int(diff)):
                idx = random.choice(minutes)
                self.hourly_schedule[idx] += 1
        else:
            # Remove from minutes that have vehicles
            available = [i for i, v in enumerate(self.hourly_schedule) if v > 0]
            for _ in range(min(-diff, len(available))):
                if not available:
                    break
                idx = random.choice(available)
                self.hourly_schedule[idx] -= 1
                if self.hourly_schedule[idx] == 0:
                    available.remove(idx)
    
    # ========================================================================
    # VEHICLE CREATION
    # ========================================================================
    
    def _select_vehicle_type(self) -> VehicleTypeSpec:
        """Select vehicle type based on weights."""
        types = self.config.vehicle_types
        weights = [t.weight for t in types]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(types, probs)[0]
    
    def _select_behavior_type(self) -> BehaviorSpec:
        """Select behavior type based on weights."""
        behaviors = self.config.behavior_types
        weights = [b.weight for b in behaviors]
        total = sum(weights)
        probs = [w / total for w in weights]
        return random.choices(behaviors, probs)[0]
    
    def _create_driver_behavior(self, spec: BehaviorSpec) -> DriverBehavior:
        """Create DriverBehavior from specification."""
        # Base parameters by type
        base_params = {
            'conservative': {
                'patience_base': random.uniform(0.7, 0.9),
                'risk_tolerance': random.uniform(0.1, 0.3),
                'speed_preference_kmh': random.uniform(100, 115),
                'buffer_margin_km': random.uniform(60, 100),
                'target_soc': random.uniform(0.85, 0.95),
                'price_sensitivity': random.uniform(0.4, 0.7)
            },
            'balanced': {
                'patience_base': random.uniform(0.5, 0.7),
                'risk_tolerance': random.uniform(0.3, 0.6),
                'speed_preference_kmh': random.uniform(110, 125),
                'buffer_margin_km': random.uniform(40, 70),
                'target_soc': random.uniform(0.75, 0.85),
                'price_sensitivity': random.uniform(0.3, 0.5)
            },
            'aggressive': {
                'patience_base': random.uniform(0.3, 0.5),
                'risk_tolerance': random.uniform(0.6, 0.9),
                'speed_preference_kmh': random.uniform(125, 140),
                'acceleration_aggression': random.uniform(0.6, 0.9),
                'buffer_margin_km': random.uniform(20, 40),
                'target_soc': random.uniform(0.60, 0.75),
                'price_sensitivity': random.uniform(0.1, 0.3)
            },
            'range_anxious': {
                'patience_base': random.uniform(0.5, 0.7),
                'risk_tolerance': random.uniform(0.05, 0.2),
                'speed_preference_kmh': random.uniform(90, 110),
                'buffer_margin_km': random.uniform(80, 120),
                'min_charge_to_continue': random.uniform(0.35, 0.50),
                'social_influence': random.uniform(0.5, 0.8),
                'target_soc': random.uniform(0.90, 1.0),
                'price_sensitivity': random.uniform(0.2, 0.4)
            }
        }
        
        params = base_params.get(spec.name, base_params['balanced']).copy()
        params.update(spec.parameter_overrides)
        
        return DriverBehavior(behavior_type=spec.name, **params) # pyright: ignore[reportArgumentType]
    
    def generate_vehicle(self) -> Vehicle:
        """Create a single vehicle with randomized properties."""
        # Select type and behavior
        vtype = self._select_vehicle_type()
        bspec = self._select_behavior_type()
        
        # Generate SOC using type-specific distribution
        initial_soc = vtype.soc_distribution()
        
        # Create driver
        driver = self._create_driver_behavior(bspec)
        
        # Create vehicle
        vehicle = Vehicle(
            battery_capacity_kwh=vtype.battery_capacity_kwh,
            initial_soc=initial_soc,
            driver_behavior=driver,
            initial_position_km=0.0,
            initial_speed_kmh=random.uniform(
                max(80, driver.speed_preference_kmh - 20),
                driver.speed_preference_kmh
            ),
            track_history=self.env.config.track_vehicle_history
        )
        
        return vehicle
    
    # ========================================================================
    # MAIN GENERATION INTERFACE
    # ========================================================================
    
    def step(self) -> List[Vehicle]:
        """
        Generate vehicles for the current minute.
        
        Returns list of generated vehicles (may be empty).
        """
        current_minute = self.env.current_time.minute
        
        # Check for hour rollover
        if self.env.current_time.hour != self.current_hour:
            self.current_hour = self.env.current_time.hour
            self._generate_hourly_schedule()
        
        # Get scheduled count for this minute
        if current_minute < len(self.hourly_schedule):
            target_count = self.hourly_schedule[current_minute]
        else:
            target_count = 0
        
        # Generate vehicles
        generated = []
        for _ in range(int(target_count)):
            vehicle = self.generate_vehicle()
            
            # Spawn into environment
            self.env.spawn_vehicle(vehicle)
            generated.append(vehicle)
            
            # Update tracking
            self.hourly_generated += 1
            self.total_generated += 1
        
        # Log generation event
        if generated:
            self.generation_log.append({
                'time': self.env.current_time,
                'count': len(generated),
                'cumulative_hour': self.hourly_generated,
                'cumulative_total': self.total_generated
            })
        
        return generated
    
    def generate_batch(self, count: int) -> List[Vehicle]:
        """
        Generate a specific number of vehicles immediately.
        Bypasses temporal distribution (for initialization or special events).
        """
        generated = []
        for _ in range(count):
            vehicle = self.generate_vehicle()
            self.env.spawn_vehicle(vehicle)
            generated.append(vehicle)
            self.total_generated += 1
        
        return generated
    
    # ========================================================================
    # STATISTICS AND MONITORING
    # ========================================================================
    
    def get_generation_stats(self) -> Dict:
        """Get generation statistics."""
        if not self.generation_log:
            return {}
        
        recent = [g for g in self.generation_log 
                 if (self.env.current_time - g['time']).total_seconds() < 3600]
        
        return {
            'total_generated': self.total_generated,
            'current_hour_generated': self.hourly_generated,
            'current_hour_target': self.hourly_target,
            'generation_rate_achieved': (
                self.hourly_generated / max(1, self.env.current_time.minute + 
                                          (self.env.current_time.hour - self.current_hour) * 60)
            ),
            'recent_minute_average': sum(g['count'] for g in recent) / max(1, len(recent)),
            'distribution_type': self.config.distribution_type.name
        }
    
    def get_hourly_distribution(self, hours: int = 24) -> List[Tuple[int, int]]:
        """
        Get actual generation counts by hour for recent history.
        """
        distribution = []
        current = self.env.current_time.replace(minute=0, second=0, microsecond=0)
        
        for h in range(hours):
            hour_start = current - timedelta(hours=h)
            hour_end = hour_start + timedelta(hours=1)
            
            count = sum(
                g['count'] for g in self.generation_log
                if hour_start <= g['time'] < hour_end
            )
            distribution.append((hour_start.hour, count))
        
        return list(reversed(distribution))
    
    def preview_schedule(self, hours: int = 2) -> List[Dict]:
        """
        Preview upcoming generation schedule.
        """
        # Save current state
        saved_hour = self.current_hour
        saved_schedule = self.hourly_schedule.copy()
        
        preview = []
        for h in range(hours):
            test_hour = (self.current_hour + h) % 24
            self.current_hour = test_hour
            self._generate_hourly_schedule()
            
            preview.append({
                'hour': test_hour,
                'total': sum(self.hourly_schedule),
                'by_minute': self.hourly_schedule.copy(),
                'peak_minute': max(range(60), key=lambda i: self.hourly_schedule[i]),
                'peak_count': max(self.hourly_schedule)
            })
        
        # Restore state
        self.current_hour = saved_hour
        self.hourly_schedule = saved_schedule
        
        return preview
    
    def __repr__(self) -> str:
        return (f"VehicleGenerator({self.config.distribution_type.name}, "
                f"{self.hourly_target}/hr, "
                f"generated={self.total_generated})")

