"""
Unit tests for VehicleGenerator (Phase 2).

Covers:
- Initialisation: schedule generated, stats zeroed
- generate_vehicle(): returns Vehicle, capacity in fleet range, driver type set
- generate_batch(): exact count, each is a Vehicle, batch of zero
- _select_vehicle_type() / _select_behavior_type(): weighted selection,
  single-item fleet always returns that item, reproducible with seed
- Fleet composition: with enough samples the weights are approximately respected
- GeneratorConfig: default fleet specs, behavior specs
"""

import sys
import random
import pytest
from datetime import datetime
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Break the circular import between generator.py → simulation/__init__.py
# → simulation.py → generator.py.
#
# generator.py does:
#   from ev_charge_manager.simulation.environment import Environment, SimulationConfig
# which triggers simulation/__init__.py, which triggers simulation.py,
# which tries to re-import VehicleGenerator — circular.
#
# We pre-populate sys.modules with lightweight stubs so generator.py
# satisfies its own top-level imports without pulling in the full
# simulation package.
# ---------------------------------------------------------------------------
def _stub_simulation_modules():
    sim_stub = MagicMock()
    sim_stub.Environment = MagicMock
    sim_stub.SimulationConfig = MagicMock
    for name in [
        'ev_charge_manager.simulation',
        'ev_charge_manager.simulation.environment',
        'ev_charge_manager.simulation.simulation',
        'ev_charge_manager.simulation.stop_conditions',
        'ev_charge_manager.simulation.parameters',
    ]:
        sys.modules.setdefault(name, sim_stub)

_stub_simulation_modules()

from ev_charge_manager.vehicle.vehicle import Vehicle, DriverBehavior  # noqa: E402
from ev_charge_manager.vehicle.generator import (                       # noqa: E402
    VehicleGenerator,
    GeneratorConfig,
    TemporalDistribution,
    VehicleTypeSpec,
    BehaviorSpec,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_mock_env(hour=12, minute=0, track_history=False):
    env = MagicMock()
    env.current_time = datetime(2025, 1, 1, hour, minute, 0)
    env.config.track_vehicle_history = track_history
    env.spawn_vehicle = MagicMock()
    return env


KNOWN_CAPACITIES = {40.0, 60.0, 80.0, 100.0}   # Default fleet spec kWh values
KNOWN_BEHAVIORS  = {"conservative", "balanced", "aggressive", "range_anxious"}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestVehicleGeneratorInit:
    def test_uses_provided_config(self):
        env = make_mock_env()
        cfg = GeneratorConfig(vehicles_per_hour=120.0)
        gen = VehicleGenerator(env, cfg)
        assert gen.config.vehicles_per_hour == 120.0

    def test_creates_default_config_when_none(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        assert isinstance(gen.config, GeneratorConfig)

    def test_total_generated_starts_at_zero(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        assert gen.total_generated == 0

    def test_hourly_schedule_has_60_entries(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        assert len(gen.hourly_schedule) == 60

    def test_hourly_schedule_entries_non_negative(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        assert all(v >= 0 for v in gen.hourly_schedule)

    def test_random_seed_makes_generation_reproducible(self):
        """Same seed → same schedule."""
        cfg1 = GeneratorConfig(random_seed=42, distribution_type=TemporalDistribution.UNIFORM)
        cfg2 = GeneratorConfig(random_seed=42, distribution_type=TemporalDistribution.UNIFORM)
        gen1 = VehicleGenerator(make_mock_env(), cfg1)
        gen2 = VehicleGenerator(make_mock_env(), cfg2)
        assert gen1.hourly_schedule == gen2.hourly_schedule


# ---------------------------------------------------------------------------
# generate_vehicle()
# ---------------------------------------------------------------------------

class TestGenerateVehicle:
    def test_returns_vehicle_instance(self):
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=0))
        v = gen.generate_vehicle()
        assert isinstance(v, Vehicle)

    def test_battery_capacity_in_fleet_range(self):
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=7))
        for _ in range(50):
            v = gen.generate_vehicle()
            assert v.battery.capacity_kwh in KNOWN_CAPACITIES

    def test_driver_behavior_type_in_known_types(self):
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=3))
        for _ in range(50):
            v = gen.generate_vehicle()
            assert v.driver.behavior_type in KNOWN_BEHAVIORS

    def test_initial_soc_within_valid_range(self):
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=1))
        for _ in range(50):
            v = gen.generate_vehicle()
            assert 0.0 <= v.battery.current_soc <= 1.0

    def test_vehicle_starts_at_position_zero(self):
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=2))
        v = gen.generate_vehicle()
        assert v.position_km == pytest.approx(0.0)

    def test_vehicle_has_positive_initial_speed(self):
        """Generated vehicles start with non-zero speed."""
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=5))
        speeds = [gen.generate_vehicle().speed_kmh for _ in range(20)]
        # All speeds should be positive (driver.speed_preference - 20 >= 70)
        assert all(s > 0 for s in speeds)


# ---------------------------------------------------------------------------
# generate_batch()
# ---------------------------------------------------------------------------

class TestGenerateBatch:
    def test_returns_exact_count(self):
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=10))
        batch = gen.generate_batch(5)
        assert len(batch) == 5

    def test_each_item_is_vehicle(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        batch = gen.generate_batch(3)
        assert all(isinstance(v, Vehicle) for v in batch)

    def test_batch_of_zero_returns_empty_list(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        batch = gen.generate_batch(0)
        assert batch == []

    def test_batch_increments_total_generated(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        gen.generate_batch(7)
        assert gen.total_generated == 7

    def test_batch_spawns_each_vehicle_into_env(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        gen.generate_batch(4)
        assert env.spawn_vehicle.call_count == 4

    def test_vehicles_in_batch_have_unique_ids(self):
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=99))
        batch = gen.generate_batch(10)
        ids = [v.id for v in batch]
        assert len(set(ids)) == 10


# ---------------------------------------------------------------------------
# _select_vehicle_type()
# ---------------------------------------------------------------------------

class TestSelectVehicleType:
    def test_single_type_fleet_always_returns_that_type(self):
        only_type = VehicleTypeSpec("only", 55.0, weight=1.0,
                                    soc_distribution=lambda: 0.7)
        cfg = GeneratorConfig(vehicle_types=[only_type])
        env = make_mock_env()
        gen = VehicleGenerator(env, cfg)
        for _ in range(20):
            selected = gen._select_vehicle_type()
            assert selected.name == "only"

    def test_selection_respects_weights_statistically(self):
        """With 9:1 weight split and 500 samples, majority should be type_a."""
        type_a = VehicleTypeSpec("a", 40.0, weight=9.0)
        type_b = VehicleTypeSpec("b", 60.0, weight=1.0)
        cfg = GeneratorConfig(vehicle_types=[type_a, type_b], random_seed=42)
        env = make_mock_env()
        gen = VehicleGenerator(env, cfg)
        random.seed(42)
        counts = {"a": 0, "b": 0}
        for _ in range(500):
            t = gen._select_vehicle_type()
            counts[t.name] += 1
        # With 9:1 ratio, type_a should win ~90% of the time
        assert counts["a"] > counts["b"] * 5

    def test_returns_vehicletypespec(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        selected = gen._select_vehicle_type()
        assert isinstance(selected, VehicleTypeSpec)


# ---------------------------------------------------------------------------
# _select_behavior_type()
# ---------------------------------------------------------------------------

class TestSelectBehaviorType:
    def test_single_behavior_fleet_always_returns_it(self):
        only = BehaviorSpec("only_behavior", weight=1.0)
        cfg = GeneratorConfig(behavior_types=[only])
        env = make_mock_env()
        gen = VehicleGenerator(env, cfg)
        for _ in range(20):
            assert gen._select_behavior_type().name == "only_behavior"

    def test_returns_behaviorspec(self):
        env = make_mock_env()
        gen = VehicleGenerator(env)
        selected = gen._select_behavior_type()
        assert isinstance(selected, BehaviorSpec)

    def test_selection_covers_all_behavior_names(self):
        """Over many samples, all four default behavior types should appear."""
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=0))
        random.seed(0)
        seen = set()
        for _ in range(200):
            seen.add(gen._select_behavior_type().name)
        assert seen == KNOWN_BEHAVIORS


# ---------------------------------------------------------------------------
# Fleet composition (statistical)
# ---------------------------------------------------------------------------

class TestFleetComposition:
    def test_midsize_most_common_vehicle(self):
        """Midsize has weight 0.40 — should appear most often."""
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=123))
        random.seed(123)
        capacity_counts: dict = {}
        for _ in range(400):
            v = gen.generate_vehicle()
            cap = v.battery.capacity_kwh
            capacity_counts[cap] = capacity_counts.get(cap, 0) + 1
        # 60 kWh (midsize, weight=0.40) should be the most common
        assert capacity_counts.get(60.0, 0) == max(capacity_counts.values())

    def test_balanced_most_common_behavior(self):
        """Balanced has weight 0.40 — should appear most often."""
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=456))
        random.seed(456)
        behavior_counts: dict = {}
        for _ in range(400):
            v = gen.generate_vehicle()
            btype = v.driver.behavior_type
            behavior_counts[btype] = behavior_counts.get(btype, 0) + 1
        assert behavior_counts.get("balanced", 0) == max(behavior_counts.values())

    def test_truck_least_common_vehicle(self):
        """Truck has weight 0.10 — should appear least often."""
        env = make_mock_env()
        gen = VehicleGenerator(env, GeneratorConfig(random_seed=789))
        random.seed(789)
        capacity_counts: dict = {}
        for _ in range(400):
            v = gen.generate_vehicle()
            cap = v.battery.capacity_kwh
            capacity_counts[cap] = capacity_counts.get(cap, 0) + 1
        # 100 kWh (truck, weight=0.10) should be the least common
        assert capacity_counts.get(100.0, 0) == min(capacity_counts.values())


# ---------------------------------------------------------------------------
# GeneratorConfig defaults
# ---------------------------------------------------------------------------

class TestGeneratorConfigDefaults:
    def test_default_vehicles_per_hour(self):
        cfg = GeneratorConfig()
        assert cfg.vehicles_per_hour == pytest.approx(60.0)

    def test_default_distribution_is_random_poisson(self):
        cfg = GeneratorConfig()
        assert cfg.distribution_type == TemporalDistribution.RANDOM_POISSON

    def test_default_vehicle_types_count(self):
        cfg = GeneratorConfig()
        assert len(cfg.vehicle_types) == 4

    def test_default_behavior_types_count(self):
        cfg = GeneratorConfig()
        assert len(cfg.behavior_types) == 4

    def test_default_vehicle_capacities(self):
        cfg = GeneratorConfig()
        caps = {t.battery_capacity_kwh for t in cfg.vehicle_types}
        assert caps == KNOWN_CAPACITIES

    def test_default_behavior_names(self):
        cfg = GeneratorConfig()
        names = {b.name for b in cfg.behavior_types}
        assert names == KNOWN_BEHAVIORS

    def test_weights_all_positive(self):
        cfg = GeneratorConfig()
        for t in cfg.vehicle_types:
            assert t.weight > 0
        for b in cfg.behavior_types:
            assert b.weight > 0

    def test_no_random_seed_by_default(self):
        cfg = GeneratorConfig()
        assert cfg.random_seed is None
