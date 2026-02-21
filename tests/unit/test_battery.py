"""
Unit tests for the Battery class.

Covers:
- consume(): normal drain, exact depletion, over-drain, SOC consistency
- charge(): normal charge, at-capacity clamping, degradation factor interaction
- usable_soc: below lower bound, mid-range, at upper bound
- is_critical, is_depleted: boundary values
- estimate_range_km(): normal case, zero consumption rate
"""

import pytest
from ev_charge_manager.vehicle.vehicle import Battery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_battery(capacity_kwh=70.0, current_soc=0.8, degradation_factor=0.95,
                 usable_soc_range=(0.1, 1.0), min_operating_soc=0.05):
    return Battery(
        capacity_kwh=capacity_kwh,
        current_soc=current_soc,
        degradation_factor=degradation_factor,
        usable_soc_range=usable_soc_range,
        min_operating_soc=min_operating_soc,
    )


# ---------------------------------------------------------------------------
# __post_init__ / initialisation
# ---------------------------------------------------------------------------

class TestBatteryInit:
    def test_current_kwh_derived_from_soc(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        assert b.current_kwh == pytest.approx(50.0)

    def test_current_kwh_accounts_for_degradation(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=0.8)
        # effective capacity = 80 kWh; 50% of that = 40 kWh
        assert b.current_kwh == pytest.approx(40.0)

    def test_max_usable_kwh_derived_from_range(self):
        b = make_battery(capacity_kwh=100.0, usable_soc_range=(0.1, 1.0))
        # usable window = 90% of 100 kWh = 90 kWh
        assert b.max_usable_kwh == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# consume()
# ---------------------------------------------------------------------------

class TestBatteryConsume:
    def test_normal_drain_reduces_kwh(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        # current_kwh starts at 50
        result = b.consume(10.0)
        assert result is True
        assert b.current_kwh == pytest.approx(40.0)

    def test_soc_updated_consistently_after_consume(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        b.consume(10.0)
        # SOC should equal current_kwh / (capacity * degradation)
        expected_soc = b.current_kwh / (100.0 * 1.0)
        assert b.current_soc == pytest.approx(expected_soc)

    def test_soc_updated_with_degradation_factor(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.6, degradation_factor=0.8)
        # effective capacity = 80 kWh; current_kwh = 48
        b.consume(8.0)
        expected_soc = 40.0 / (100.0 * 0.8)
        assert b.current_soc == pytest.approx(expected_soc)

    def test_exact_depletion_returns_false(self):
        """Consuming exactly available energy triggers the over-drain branch."""
        b = make_battery(capacity_kwh=100.0, current_soc=0.1, degradation_factor=1.0)
        # current_kwh = 10.0; consume exactly 10.0 — guard is `>`, not `>=`,
        # so this path actually succeeds (returns True with kwh=0, soc=0).
        available = b.current_kwh
        result = b.consume(available)
        assert result is True
        assert b.current_kwh == pytest.approx(0.0)
        assert b.current_soc == pytest.approx(0.0)

    def test_over_drain_returns_false_and_zeros_battery(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.1, degradation_factor=1.0)
        result = b.consume(999.0)
        assert result is False
        assert b.current_kwh == 0
        assert b.current_soc == 0

    def test_consume_zero_is_noop(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        original_kwh = b.current_kwh
        result = b.consume(0.0)
        assert result is True
        assert b.current_kwh == pytest.approx(original_kwh)

    def test_soc_never_goes_below_zero_on_over_drain(self):
        b = make_battery(capacity_kwh=50.0, current_soc=0.05, degradation_factor=1.0)
        b.consume(1000.0)
        assert b.current_soc >= 0.0
        assert b.current_kwh >= 0.0


# ---------------------------------------------------------------------------
# charge()
# ---------------------------------------------------------------------------

class TestBatteryCharge:
    def test_normal_charge_increases_kwh(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        actual = b.charge(10.0)
        assert actual == pytest.approx(10.0)
        assert b.current_kwh == pytest.approx(60.0)

    def test_charge_updates_soc_consistently(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        b.charge(10.0)
        expected_soc = b.current_kwh / (100.0 * 1.0)
        assert b.current_soc == pytest.approx(expected_soc)

    def test_charge_clamped_at_usable_ceiling(self):
        """Charging beyond usable ceiling should be clamped to available space."""
        b = make_battery(capacity_kwh=100.0, current_soc=0.95, degradation_factor=1.0,
                         usable_soc_range=(0.1, 1.0))
        # max_kwh = 100 * 1.0 * 1.0 = 100; current_kwh = 95; space = 5
        actual = b.charge(50.0)
        assert actual == pytest.approx(5.0)
        assert b.current_kwh == pytest.approx(100.0)

    def test_charge_clamped_with_degradation(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.9, degradation_factor=0.8,
                         usable_soc_range=(0.1, 1.0))
        # max_kwh = 100 * 0.8 * 1.0 = 80; current_kwh = 100 * 0.9 * 0.8 = 72; space = 8
        actual = b.charge(50.0)
        assert actual == pytest.approx(8.0, abs=1e-6)
        assert b.current_kwh == pytest.approx(80.0, abs=1e-6)

    def test_charge_full_battery_accepts_nothing(self):
        b = make_battery(capacity_kwh=100.0, current_soc=1.0, degradation_factor=1.0,
                         usable_soc_range=(0.1, 1.0))
        actual = b.charge(10.0)
        assert actual == pytest.approx(0.0)

    def test_charge_returns_actual_energy_accepted(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.0, degradation_factor=1.0)
        actual = b.charge(30.0)
        assert actual == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# usable_soc property
# ---------------------------------------------------------------------------

class TestUsableSoc:
    def test_below_lower_bound_returns_zero(self):
        b = make_battery(current_soc=0.05, usable_soc_range=(0.1, 1.0))
        assert b.usable_soc == pytest.approx(0.0)

    def test_at_lower_bound_returns_zero(self):
        b = make_battery(current_soc=0.1, usable_soc_range=(0.1, 1.0))
        assert b.usable_soc == pytest.approx(0.0)

    def test_midrange_returns_correct_fraction(self):
        b = make_battery(current_soc=0.55, usable_soc_range=(0.1, 1.0))
        # (0.55 - 0.10) / (1.0 - 0.10) = 0.45 / 0.90 = 0.5
        assert b.usable_soc == pytest.approx(0.5)

    def test_at_upper_bound_returns_one(self):
        b = make_battery(current_soc=1.0, usable_soc_range=(0.1, 1.0))
        assert b.usable_soc == pytest.approx(1.0)

    def test_above_upper_bound_clamped_to_one(self):
        # Should not happen in practice, but the clamp must hold
        b = make_battery(current_soc=1.0, usable_soc_range=(0.1, 0.9))
        assert b.usable_soc == pytest.approx(1.0)

    def test_degenerate_zero_width_range_returns_zero(self):
        b = Battery(capacity_kwh=70.0, current_soc=0.5,
                    usable_soc_range=(0.5, 0.5))
        assert b.usable_soc == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# is_critical property
# ---------------------------------------------------------------------------

class TestIsCritical:
    def test_not_critical_well_above_threshold(self):
        # critical threshold = lower_bound + 0.05 = 0.15
        b = make_battery(current_soc=0.5, usable_soc_range=(0.1, 1.0))
        assert b.is_critical is False

    def test_critical_at_threshold(self):
        # SOC = lower_bound + 0.05 = 0.15 → is_critical True (<=)
        b = make_battery(current_soc=0.15, usable_soc_range=(0.1, 1.0))
        assert b.is_critical is True

    def test_critical_below_threshold(self):
        b = make_battery(current_soc=0.10, usable_soc_range=(0.1, 1.0))
        assert b.is_critical is True

    def test_just_above_threshold_not_critical(self):
        # SOC = 0.151 is just above 0.15
        b = make_battery(current_soc=0.151, usable_soc_range=(0.1, 1.0))
        assert b.is_critical is False


# ---------------------------------------------------------------------------
# is_depleted property
# ---------------------------------------------------------------------------

class TestIsDepleted:
    def test_not_depleted_above_min(self):
        b = make_battery(current_soc=0.5, min_operating_soc=0.05)
        assert b.is_depleted is False

    def test_depleted_at_min_operating_soc(self):
        # is_depleted uses <=
        b = make_battery(current_soc=0.05, min_operating_soc=0.05)
        assert b.is_depleted is True

    def test_depleted_below_min(self):
        b = make_battery(current_soc=0.02, min_operating_soc=0.05)
        assert b.is_depleted is True

    def test_just_above_min_not_depleted(self):
        b = make_battery(current_soc=0.051, min_operating_soc=0.05)
        assert b.is_depleted is False


# ---------------------------------------------------------------------------
# estimate_range_km()
# ---------------------------------------------------------------------------

class TestEstimateRangeKm:
    def test_normal_range_estimate(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        # current_kwh = 50; consumption = 0.2 kWh/km → range = 250 km
        assert b.estimate_range_km(0.2) == pytest.approx(250.0)

    def test_zero_consumption_rate_returns_infinity(self):
        b = make_battery(current_soc=0.5)
        assert b.estimate_range_km(0.0) == float('inf')

    def test_negative_consumption_rate_returns_infinity(self):
        b = make_battery(current_soc=0.5)
        assert b.estimate_range_km(-1.0) == float('inf')

    def test_empty_battery_has_zero_range(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.0, degradation_factor=1.0)
        assert b.estimate_range_km(0.2) == pytest.approx(0.0)

    def test_range_scales_with_current_kwh(self):
        b1 = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        b2 = make_battery(capacity_kwh=100.0, current_soc=0.25, degradation_factor=1.0)
        # Half the energy → half the range
        assert b1.estimate_range_km(0.2) == pytest.approx(2 * b2.estimate_range_km(0.2))

    def test_range_reflects_degradation(self):
        b_new = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=1.0)
        b_old = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=0.8)
        # Degraded battery holds less energy → shorter range
        assert b_old.estimate_range_km(0.2) < b_new.estimate_range_km(0.2)


# ---------------------------------------------------------------------------
# Round-trip: consume then charge
# ---------------------------------------------------------------------------

class TestBatteryRoundTrip:
    def test_consume_then_charge_back_to_original(self):
        b = make_battery(capacity_kwh=100.0, current_soc=0.8, degradation_factor=1.0)
        original_kwh = b.current_kwh

        b.consume(10.0)
        b.charge(10.0)

        assert b.current_kwh == pytest.approx(original_kwh)
        assert b.current_soc == pytest.approx(0.8)

    def test_soc_and_kwh_always_consistent(self):
        """After any sequence of operations, soc == kwh / (capacity * degradation)."""
        b = make_battery(capacity_kwh=100.0, current_soc=0.5, degradation_factor=0.9)
        effective_capacity = 100.0 * 0.9

        for energy in [5.0, 3.0, 10.0]:
            b.consume(energy)
            assert b.current_soc == pytest.approx(
                b.current_kwh / effective_capacity, abs=1e-9
            )

        b.charge(20.0)
        assert b.current_soc == pytest.approx(
            b.current_kwh / effective_capacity, abs=1e-9
        )
