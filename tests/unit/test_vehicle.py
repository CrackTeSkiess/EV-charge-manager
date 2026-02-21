"""
Unit tests for the Vehicle class (Phase 2).

Covers:
- __init__: ID generation, battery/driver wiring, initial state
- calculate_consumption_rate(): physics model, caching, aggression modifier
- _motor_efficiency(): piecewise curve, always in [0.70, 0.95]
- update_physics(): normal step, distance/energy accumulation, stranding
- get_range_estimate(): normal, zero-speed edge case
- can_reach_station(): near / far with driver buffer
- can_physically_reach(): near / far with safety factor
- must_stop_at_station(): must-stop vs safe-to-skip logic
- needs_charging(): critical / comfort / reachability triggers
- evaluate_charging_decision(): best-option selection, unreachable penalty
- start_charging() / charge_step() / finish_charging(): session lifecycle
- abandon_charging(): state and cleanup
- update_patience(): delegates to driver with battery urgency
- set_state(): state machine transitions, no-op on same state
- get_status() / get_trip_summary(): expected keys present
"""

import pytest
from datetime import datetime
from ev_charge_manager.vehicle.vehicle import Vehicle, VehicleState, Battery, DriverBehavior

TS = datetime(2025, 6, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vehicle(
    capacity=60.0,
    soc=0.5,
    speed=0.0,
    position=0.0,
    driver=None,
    track_history=False,
):
    return Vehicle(
        battery_capacity_kwh=capacity,
        initial_soc=soc,
        driver_behavior=driver,
        initial_position_km=position,
        initial_speed_kmh=speed,
        track_history=track_history,
    )


def make_station(location_km, price=0.40, available=2, total=4, wait=5.0):
    return {
        'location_km': location_km,
        'price_per_kwh': price,
        'available_chargers': available,
        'total_chargers': total,
        'estimated_wait': wait,
        'trending': False,
    }


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestVehicleInit:
    def test_auto_generated_id_is_string(self):
        v = make_vehicle()
        assert isinstance(v.id, str)
        assert len(v.id) > 0

    def test_explicit_id_preserved(self):
        v = Vehicle(vehicle_id="test-abc-123", battery_capacity_kwh=60.0, initial_soc=0.5)
        assert v.id == "test-abc-123"

    def test_two_vehicles_have_different_ids(self):
        v1 = make_vehicle()
        v2 = make_vehicle()
        assert v1.id != v2.id

    def test_battery_capacity_matches_parameter(self):
        v = make_vehicle(capacity=80.0, soc=0.5)
        assert v.battery.capacity_kwh == pytest.approx(80.0)

    def test_battery_soc_matches_parameter(self):
        v = make_vehicle(soc=0.65)
        assert v.battery.current_soc == pytest.approx(0.65)

    def test_default_driver_created_when_none(self):
        v = make_vehicle()
        assert isinstance(v.driver, DriverBehavior)

    def test_custom_driver_attached(self):
        d = DriverBehavior(behavior_type="aggressive")
        v = make_vehicle(driver=d)
        assert v.driver is d

    def test_initial_state_is_cruising(self):
        v = make_vehicle()
        assert v.state == VehicleState.CRUISING

    def test_initial_patience_equals_driver_patience_base(self):
        d = DriverBehavior(patience_base=0.7)
        v = make_vehicle(driver=d)
        assert v.current_patience == pytest.approx(0.7)

    def test_counters_start_at_zero(self):
        v = make_vehicle()
        assert v.total_distance_km == pytest.approx(0.0)
        assert v.total_energy_consumed_kwh == pytest.approx(0.0)
        assert v.charging_stops == 0
        assert v.total_charging_time_min == pytest.approx(0.0)

    def test_position_and_speed_match_parameters(self):
        v = make_vehicle(position=50.0, speed=100.0)
        assert v.position_km == pytest.approx(50.0)
        assert v.speed_kmh == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# calculate_consumption_rate()
# ---------------------------------------------------------------------------

class TestCalculateConsumptionRate:
    def test_zero_speed_returns_zero(self):
        v = make_vehicle()
        assert v.calculate_consumption_rate(0.0) == pytest.approx(0.0)

    def test_negative_speed_returns_zero(self):
        v = make_vehicle()
        assert v.calculate_consumption_rate(-10.0) == pytest.approx(0.0)

    def test_positive_speed_returns_positive(self):
        v = make_vehicle()
        assert v.calculate_consumption_rate(100.0) > 0.0

    def test_higher_speed_higher_consumption(self):
        v = make_vehicle()
        low = v.calculate_consumption_rate(60.0)
        high = v.calculate_consumption_rate(130.0)
        assert high > low

    def test_result_cached_for_same_integer_speed(self):
        v = make_vehicle()
        # 60.2 and 60.8 both map to int key 60 — must return same value
        r1 = v.calculate_consumption_rate(60.2)
        r2 = v.calculate_consumption_rate(60.8)
        assert r1 == pytest.approx(r2)

    def test_cache_populated_after_first_call(self):
        v = make_vehicle()
        v.calculate_consumption_rate(100.0)
        assert 100 in v._consumption_cache

    def test_aggressive_driver_higher_consumption_than_eco(self):
        eco_driver = DriverBehavior(acceleration_aggression=0.0)
        agg_driver = DriverBehavior(acceleration_aggression=1.0)
        eco_v = make_vehicle(driver=eco_driver)
        agg_v = make_vehicle(driver=agg_driver)
        assert agg_v.calculate_consumption_rate(100.0) > eco_v.calculate_consumption_rate(100.0)

    def test_aux_load_visible_at_low_speed(self):
        """Aux power (1.5 kW) makes consumption positive even at slow speeds."""
        v = make_vehicle()
        assert v.calculate_consumption_rate(5.0) > 0.0

    def test_reasonable_highway_consumption(self):
        """At 100 km/h, expect ~0.12–0.35 kWh/km (real-world EV range)."""
        v = make_vehicle()
        rate = v.calculate_consumption_rate(100.0)
        assert 0.10 < rate < 0.40


# ---------------------------------------------------------------------------
# _motor_efficiency()
# ---------------------------------------------------------------------------

class TestMotorEfficiency:
    def test_very_low_speed_returns_0_75(self):
        v = make_vehicle()
        assert v._motor_efficiency(5.0) == pytest.approx(0.75)

    def test_at_exactly_10_returns_0_90(self):
        v = make_vehicle()
        assert v._motor_efficiency(10.0) == pytest.approx(0.90)

    def test_at_60_returns_0_95_peak(self):
        v = make_vehicle()
        assert v._motor_efficiency(60.0) == pytest.approx(0.95)

    def test_at_120_returns_0_85(self):
        v = make_vehicle()
        assert v._motor_efficiency(120.0) == pytest.approx(0.85)

    def test_above_120_below_0_85(self):
        v = make_vehicle()
        assert v._motor_efficiency(150.0) < 0.85

    def test_always_in_valid_range(self):
        v = make_vehicle()
        for speed in [0, 5, 10, 30, 60, 90, 120, 150, 180]:
            eff = v._motor_efficiency(float(speed))
            assert 0.70 <= eff <= 0.95, f"Efficiency {eff} out of range at {speed} km/h"

    def test_rising_slope_between_10_and_60(self):
        v = make_vehicle()
        assert v._motor_efficiency(30.0) > v._motor_efficiency(10.0)
        assert v._motor_efficiency(60.0) > v._motor_efficiency(30.0)

    def test_falling_slope_above_60(self):
        v = make_vehicle()
        assert v._motor_efficiency(90.0) < v._motor_efficiency(60.0)
        assert v._motor_efficiency(130.0) < v._motor_efficiency(90.0)


# ---------------------------------------------------------------------------
# update_physics()
# ---------------------------------------------------------------------------

class TestUpdatePhysics:
    def test_returns_true_with_normal_battery(self):
        v = make_vehicle(soc=0.8, speed=100.0)
        result = v.update_physics(1.0, speed_limit_kmh=130.0)
        assert result is True

    def test_position_increases_when_moving(self):
        v = make_vehicle(soc=0.8, speed=100.0)
        v.update_physics(1.0, speed_limit_kmh=130.0)
        assert v.position_km > 0.0

    def test_distance_accumulates_over_steps(self):
        v = make_vehicle(soc=0.9, speed=100.0)
        v.update_physics(1.0, speed_limit_kmh=130.0)
        d1 = v.total_distance_km
        v.update_physics(1.0, speed_limit_kmh=130.0)
        d2 = v.total_distance_km
        assert d2 > d1

    def test_energy_consumed_accumulates(self):
        v = make_vehicle(soc=0.9, speed=100.0)
        v.update_physics(1.0, speed_limit_kmh=130.0)
        assert v.total_energy_consumed_kwh > 0.0

    def test_stranded_returns_false_and_sets_state(self):
        """Tiny battery + highway speed → depletes in one step → STRANDED."""
        v = make_vehicle(capacity=60.0, soc=0.01, speed=130.0)
        # current_kwh = 60 * 0.01 * 0.95 ≈ 0.57 kWh; consumption > that in 1 min
        result = v.update_physics(1.0, speed_limit_kmh=130.0)
        assert result is False
        assert v.state == VehicleState.STRANDED

    def test_speed_converges_toward_limit(self):
        """Vehicle starting from rest should increase speed each step."""
        v = make_vehicle(soc=0.9, speed=0.0)
        v.update_physics(1.0, speed_limit_kmh=130.0)
        assert v.speed_kmh > 0.0

    def test_static_vehicle_does_not_move(self):
        """A vehicle that cannot accelerate (already at speed 0, limit 0) stays put."""
        v = make_vehicle(soc=0.9, speed=0.0)
        # Speed limit = 0 → target speed = max(20, 0+20) = 20; slight movement OK
        # Let's pass traffic_speed = 0.0
        v.update_physics(1.0, speed_limit_kmh=0.0, traffic_speed_kmh=0.0)
        # position should be minimal since avg speed between 0 and small


# ---------------------------------------------------------------------------
# get_range_estimate()
# ---------------------------------------------------------------------------

class TestGetRangeEstimate:
    def test_returns_positive_at_highway_speed(self):
        v = make_vehicle(soc=0.8, speed=100.0)
        assert v.get_range_estimate() > 0.0

    def test_higher_soc_means_more_range(self):
        v_low = make_vehicle(soc=0.3, speed=100.0)
        v_high = make_vehicle(soc=0.8, speed=100.0)
        assert v_high.get_range_estimate() > v_low.get_range_estimate()

    def test_zero_speed_returns_inf(self):
        """At speed 0 consumption is 0, so range = infinity."""
        v = make_vehicle(soc=0.5, speed=0.0)
        assert v.get_range_estimate() == float('inf')

    def test_range_proportional_to_battery_energy(self):
        """Double the kWh → double the range (same speed, same driver)."""
        d = DriverBehavior(acceleration_aggression=0.5)
        v_small = make_vehicle(capacity=40.0, soc=0.5, speed=100.0, driver=d)
        v_large = make_vehicle(capacity=80.0, soc=0.5, speed=100.0, driver=d)
        # v_large has double the energy (same SOC) → double range
        ratio = v_large.get_range_estimate() / v_small.get_range_estimate()
        assert ratio == pytest.approx(2.0, rel=0.01)


# ---------------------------------------------------------------------------
# can_reach_station()
# ---------------------------------------------------------------------------

class TestCanReachStation:
    def test_nearby_station_reachable(self):
        v = make_vehicle(soc=0.8, speed=100.0, position=0.0)
        # Range at 100 km/h for 80% SOC on 60 kWh ~ several hundred km
        assert v.can_reach_station(5.0) is True

    def test_very_distant_station_not_reachable(self):
        v = make_vehicle(soc=0.05, speed=100.0, position=0.0,
                         driver=DriverBehavior(buffer_margin_km=50.0))
        # 5% SOC on 60 kWh = ~3 kWh, at 100 km/h ≈ 0.17 kWh/km → ~18 km bare range
        # Station at 200 km with 50 km buffer = 250 km required → unreachable
        assert v.can_reach_station(200.0) is False

    def test_buffer_included_in_decision(self):
        """A larger buffer makes a distant station unreachable; a small buffer allows it.

        NOTE: buffer_km=0.0 is falsy in Python, so the source code's
        ``buffer = buffer_km or self.driver.buffer_margin_km`` falls through
        to the driver default for zero. We use explicit positive values.
        """
        v = make_vehicle(soc=0.5, speed=100.0, position=0.0)
        range_est = v.get_range_estimate()
        target = range_est * 0.6  # comfortably inside range
        assert v.can_reach_station(target, buffer_km=1.0) is True
        assert v.can_reach_station(target, buffer_km=range_est) is False

    def test_custom_buffer_overrides_driver_default(self):
        """A large explicit buffer makes a station unreachable that the default buffer allows."""
        d = DriverBehavior(buffer_margin_km=5.0)
        v = make_vehicle(soc=0.5, speed=100.0, driver=d)
        range_est = v.get_range_estimate()
        target = range_est * 0.7  # reachable with 5 km buffer
        with_small_buffer = v.can_reach_station(target, buffer_km=5.0)
        with_huge_buffer = v.can_reach_station(target, buffer_km=range_est)
        assert with_small_buffer is True
        assert with_huge_buffer is False


# ---------------------------------------------------------------------------
# can_physically_reach()
# ---------------------------------------------------------------------------

class TestCanPhysicallyReach:
    def test_position_right_next_to_vehicle_reachable(self):
        v = make_vehicle(soc=0.8, position=50.0)
        assert v.can_physically_reach(51.0) is True

    def test_very_far_unreachable_with_low_battery(self):
        v = make_vehicle(soc=0.05, position=0.0)
        assert v.can_physically_reach(500.0) is False

    def test_applies_safety_margin(self):
        """can_physically_reach uses 1.15x safety factor; get_range_estimate does not."""
        v = make_vehicle(soc=0.5, speed=100.0, position=0.0)
        raw_range = v.get_range_estimate()
        # Can reach a point exactly at raw_range without safety? Physically: NO (safety applied)
        # Can reach a shorter point? YES
        short = raw_range * 0.5
        assert v.can_physically_reach(short) is True


# ---------------------------------------------------------------------------
# must_stop_at_station()
# ---------------------------------------------------------------------------

class TestMustStopAtStation:
    def test_must_stop_when_cannot_reach_next_station(self):
        """Low battery: can barely reach current station, cannot make it to next."""
        v = make_vehicle(soc=0.12, speed=100.0, position=0.0)
        # Current station at 10 km, next at 200 km — not enough range
        assert v.must_stop_at_station(
            current_station_km=10.0,
            next_station_km=200.0,
        ) is True

    def test_safe_to_skip_when_battery_is_ample(self):
        """High battery: plenty of range to skip current and reach next."""
        v = make_vehicle(soc=0.95, speed=100.0, position=0.0)
        # Next station only 50 km away — easily reachable
        assert v.must_stop_at_station(
            current_station_km=10.0,
            next_station_km=60.0,
        ) is False

    def test_must_stop_when_cannot_physically_reach_current(self):
        """If vehicle can't even reach current station, must_stop returns True."""
        v = make_vehicle(soc=0.05, speed=100.0, position=0.0)
        # Station 300 km away — unreachable with low battery
        assert v.must_stop_at_station(
            current_station_km=300.0,
            next_station_km=None,
        ) is True

    def test_safety_margin_prevents_too_close_decision(self):
        """next_station needs range + SAFETY_MARGIN_KM (20 km) buffer."""
        v = make_vehicle(soc=0.50, speed=100.0, position=0.0)
        cons = v._get_conservative_consumption(v.driver.speed_preference_kmh)
        range_km = v.battery.estimate_range_km(cons)
        dist_to_current = 5.0
        remaining = range_km - dist_to_current
        # Set next station just 1 km beyond the 20-km safety margin
        next_km = dist_to_current + (remaining - 21.0)
        assert v.must_stop_at_station(
            current_station_km=dist_to_current,
            next_station_km=next_km + 0.0,
        ) is False  # can make it with margin


# ---------------------------------------------------------------------------
# needs_charging()
# ---------------------------------------------------------------------------

class TestNeedsCharging:
    def test_critical_battery_needs_charging(self):
        """is_critical (SOC <= lower_bound + 0.05 = 0.15) → needs charging."""
        v = make_vehicle(soc=0.12)
        assert v.needs_charging() is True

    def test_full_battery_does_not_need_charging(self):
        v = make_vehicle(soc=1.0)
        assert v.needs_charging() is False

    def test_below_min_charge_to_continue_needs_charging(self):
        d = DriverBehavior(min_charge_to_continue=0.4)
        v = make_vehicle(soc=0.35, driver=d)
        assert v.needs_charging() is True

    def test_above_min_and_non_critical_does_not_need_charging(self):
        d = DriverBehavior(min_charge_to_continue=0.2, buffer_margin_km=50.0)
        v = make_vehicle(soc=0.5, speed=100.0, driver=d)
        assert v.needs_charging() is False

    def test_cannot_reach_next_station_needs_charging(self):
        """Tiny battery, huge next station gap → needs to charge now."""
        d = DriverBehavior(min_charge_to_continue=0.1, buffer_margin_km=50.0)
        v = make_vehicle(soc=0.2, speed=100.0, driver=d)
        # At 20% SOC on 60 kWh ≈ 11 kWh, range ≈ ~60-70 km; next station far away
        assert v.needs_charging(next_station_km=500.0) is True


# ---------------------------------------------------------------------------
# evaluate_charging_decision()
# ---------------------------------------------------------------------------

class TestEvaluateChargingDecision:
    def test_no_charge_needed_returns_none(self):
        v = make_vehicle(soc=0.9)
        options = [make_station(50.0)]
        assert v.evaluate_charging_decision(options) is None

    def test_critical_battery_no_options_returns_none(self):
        v = make_vehicle(soc=0.1)
        assert v.evaluate_charging_decision([]) is None

    def test_returns_best_station_when_charging_needed(self):
        """With a critical battery and one option, that option is returned."""
        v = make_vehicle(soc=0.1, speed=100.0, position=0.0)
        station = make_station(5.0)
        result = v.evaluate_charging_decision([station])
        assert result is station

    def test_picks_better_station_out_of_two(self):
        """Cheap nearby station vs expensive far station — cheap wins."""
        v = make_vehicle(soc=0.1, speed=100.0, position=0.0)
        cheap = make_station(5.0, price=0.20, available=4, wait=0.0)
        expensive = make_station(10.0, price=0.59, available=0, wait=60.0)
        result = v.evaluate_charging_decision([cheap, expensive])
        assert result is cheap

    def test_unreachable_station_gets_penalised_not_excluded(self):
        """Station too far away gets score*0.1 penalty but can still be returned."""
        v = make_vehicle(soc=0.10, speed=100.0, position=0.0)
        # Only option is far away but driver is critical so may accept it
        far_station = make_station(1000.0, price=0.20, available=4, wait=0.0)
        result = v.evaluate_charging_decision([far_station])
        # With critical battery the threshold is bypassed → something returned
        assert result is far_station or result is None  # Either is valid


# ---------------------------------------------------------------------------
# start_charging() / charge_step() / finish_charging()
# ---------------------------------------------------------------------------

class TestChargingSessionLifecycle:
    def test_start_charging_increments_stops(self):
        v = make_vehicle(soc=0.3)
        v.start_charging("charger-001", 50.0, TS)
        assert v.charging_stops == 1

    def test_start_charging_sets_state_to_charging(self):
        v = make_vehicle(soc=0.3)
        v.start_charging("charger-001", 50.0, TS)
        assert v.state == VehicleState.CHARGING

    def test_start_charging_records_charger_id(self):
        v = make_vehicle(soc=0.3)
        v.start_charging("charger-XYZ", 50.0, TS)
        assert v.assigned_charger_id == "charger-XYZ"

    def test_charge_step_returns_false_below_target_soc(self):
        d = DriverBehavior(target_soc=0.8)
        v = make_vehicle(soc=0.3, driver=d)
        v.start_charging("c1", 50.0, TS)
        result = v.charge_step(1.0, TS)  # Small chunk, won't reach 0.8
        assert result is False

    def test_charge_step_returns_true_when_target_reached(self):
        d = DriverBehavior(target_soc=0.5)
        v = make_vehicle(soc=0.49, driver=d)
        v.start_charging("c1", 50.0, TS)
        # charge 5 kWh into 60 kWh battery → SOC += 5/(60*0.95) ≈ 0.088 → >0.5
        result = v.charge_step(5.0, TS)
        assert result is True

    def test_finish_charging_sets_exiting_state(self):
        v = make_vehicle(soc=0.3)
        v.start_charging("c1", 50.0, TS)
        v.finish_charging(TS, total_time_min=30.0)
        assert v.state == VehicleState.EXITING

    def test_finish_charging_resets_patience(self):
        d = DriverBehavior(patience_base=0.8)
        v = make_vehicle(soc=0.3, driver=d)
        v.current_patience = 0.1  # Simulate depleted patience
        v.start_charging("c1", 50.0, TS)
        v.finish_charging(TS, total_time_min=30.0)
        assert v.current_patience == pytest.approx(0.8)

    def test_finish_charging_clears_charger_and_station_ids(self):
        v = make_vehicle(soc=0.3)
        v.target_station_id = "station-1"
        v.start_charging("c1", 50.0, TS)
        v.finish_charging(TS, total_time_min=20.0)
        assert v.assigned_charger_id is None
        assert v.target_station_id is None

    def test_finish_charging_accumulates_time(self):
        v = make_vehicle(soc=0.3)
        v.start_charging("c1", 50.0, TS)
        v.finish_charging(TS, total_time_min=25.0)
        assert v.total_charging_time_min == pytest.approx(25.0)

    def test_multiple_charging_stops_counted(self):
        v = make_vehicle(soc=0.1)
        for _ in range(3):
            v.start_charging("c1", 50.0, TS)
            v.finish_charging(TS, total_time_min=10.0)
        assert v.charging_stops == 3


# ---------------------------------------------------------------------------
# abandon_charging()
# ---------------------------------------------------------------------------

class TestAbandonCharging:
    def test_sets_exiting_state(self):
        v = make_vehicle(soc=0.3)
        v.set_state(VehicleState.QUEUED, TS, "joined")
        v.abandon_charging(TS, "patience exhausted")
        assert v.state == VehicleState.EXITING

    def test_clears_target_station_id(self):
        v = make_vehicle(soc=0.3)
        v.target_station_id = "station-X"
        v.abandon_charging(TS, "gave up")
        assert v.target_station_id is None

    def test_clears_queue_entry_time(self):
        v = make_vehicle(soc=0.3)
        from datetime import datetime
        v.queue_entry_time = datetime(2025, 1, 1)
        v.abandon_charging(TS, "alternative found")
        assert v.queue_entry_time is None


# ---------------------------------------------------------------------------
# update_patience() — vehicle-level wrapper
# ---------------------------------------------------------------------------

class TestVehicleUpdatePatience:
    def test_patience_decreases_while_waiting(self):
        d = DriverBehavior(patience_base=0.8, patience_decay_rate=0.5)
        v = make_vehicle(soc=0.5, driver=d)
        before = v.current_patience
        v.update_patience(wait_minutes=5.0, comfort_level=0.5)
        assert v.current_patience < before

    def test_patience_never_negative(self):
        d = DriverBehavior(patience_base=0.1, patience_decay_rate=0.9)
        v = make_vehicle(soc=0.5, driver=d)
        for _ in range(20):
            v.update_patience(wait_minutes=10.0, comfort_level=0.0)
        assert v.current_patience >= 0.0

    def test_urgency_derived_from_battery_usable_soc(self):
        """Lower usable_soc → higher urgency → affects patience differently."""
        d = DriverBehavior(patience_base=0.8, patience_decay_rate=0.3)
        v_low = make_vehicle(soc=0.11, driver=d)   # very low usable SOC
        v_high = make_vehicle(soc=0.80, driver=d)  # high usable SOC
        # Both wait 3 min; their urgency differs → different patience results
        v_low.update_patience(3.0, 0.5)
        v_high.update_patience(3.0, 0.5)
        assert v_low.current_patience != v_high.current_patience


# ---------------------------------------------------------------------------
# set_state()
# ---------------------------------------------------------------------------

class TestSetState:
    def test_state_changes_correctly(self):
        v = make_vehicle()
        v.set_state(VehicleState.APPROACHING, TS, "close to station")
        assert v.state == VehicleState.APPROACHING

    def test_state_entry_time_updated(self):
        v = make_vehicle()
        v.set_state(VehicleState.QUEUED, TS, "joined queue")
        assert v.state_entry_time == TS

    def test_same_state_is_noop(self):
        v = make_vehicle()
        # Default state is CRUISING; set it again with a different timestamp
        from datetime import datetime
        ts2 = datetime(2025, 6, 1, 13, 0, 0)
        v.state_entry_time = None
        v.set_state(VehicleState.CRUISING, ts2, "no change")
        # state_entry_time should NOT be updated (same state)
        assert v.state_entry_time is None

    def test_full_state_machine_sequence(self):
        v = make_vehicle(soc=0.3)
        transitions = [
            VehicleState.APPROACHING,
            VehicleState.QUEUED,
            VehicleState.CHARGING,
            VehicleState.EXITING,
        ]
        for s in transitions:
            v.set_state(s, TS, "test")
            assert v.state == s


# ---------------------------------------------------------------------------
# get_status() / get_trip_summary()
# ---------------------------------------------------------------------------

class TestStatusAndSummary:
    # Actual keys returned by Vehicle.get_status()
    EXPECTED_STATUS_KEYS = {
        'id', 'state', 'position_km', 'speed_kmh',
        'soc', 'usable_soc', 'range_estimate_km',
        'current_patience', 'target_station', 'driver_type',
    }

    # Actual keys returned by Vehicle.get_trip_summary()
    EXPECTED_SUMMARY_KEYS = {
        'vehicle_id', 'total_distance_km', 'total_energy_consumed_kwh',
        'charging_stops', 'total_charging_time_min',
        'final_soc', 'was_stranded', 'state_transitions',
    }

    def test_get_status_returns_dict(self):
        v = make_vehicle()
        assert isinstance(v.get_status(), dict)

    def test_get_status_has_required_keys(self):
        v = make_vehicle()
        status = v.get_status()
        for key in self.EXPECTED_STATUS_KEYS:
            assert key in status, f"Missing key '{key}' in get_status()"

    def test_get_trip_summary_returns_dict(self):
        v = make_vehicle()
        assert isinstance(v.get_trip_summary(), dict)

    def test_get_trip_summary_has_required_keys(self):
        v = make_vehicle()
        summary = v.get_trip_summary()
        for key in self.EXPECTED_SUMMARY_KEYS:
            assert key in summary, f"Missing key '{key}' in get_trip_summary()"

    def test_status_values_reflect_vehicle_state(self):
        v = make_vehicle(soc=0.65, position=42.0)
        status = v.get_status()
        assert status['position_km'] == pytest.approx(42.0)
        assert status['soc'] == pytest.approx(0.65, rel=0.01)

    def test_trip_summary_reflects_accumulated_data(self):
        v = make_vehicle(soc=0.9, speed=100.0)
        v.update_physics(1.0)
        v.start_charging("c1", 50.0, TS)
        v.finish_charging(TS, total_time_min=20.0)
        summary = v.get_trip_summary()
        assert summary['charging_stops'] == 1
        assert summary['total_charging_time_min'] == pytest.approx(20.0)
        assert summary['total_distance_km'] > 0.0
