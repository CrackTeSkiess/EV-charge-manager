"""
Unit tests for Charger, ChargingSession, ChargerStatus, ChargerType (Phase 2).

Covers:
- ChargingSession: remaining_time(), completion_percentage()
- Charger.is_available(): status-based predicate
- Charger.start_session(): creates session, sets OCCUPIED, raises on re-use
- Charger.update_session(): increments delivered_energy_kwh
- Charger.complete_session(): frees charger, raises on empty
- Charger.interrupt_session(): marks incomplete, delegates to complete
- Charger.estimate_availability(): returns now or end_time+2min
- Cumulative stats: total_sessions, total_energy_delivered_kwh
"""

import pytest
from datetime import datetime, timedelta

from ev_charge_manager.charging.area import (
    Charger,
    ChargerStatus,
    ChargerType,
    ChargingSession,
)

TS = datetime(2025, 6, 1, 12, 0, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_charger(power_kw=50.0, charger_type=ChargerType.FAST,
                 status=ChargerStatus.AVAILABLE):
    c = Charger(power_kw=power_kw, charger_type=charger_type, status=status)
    return c


def start_session_on(charger, vehicle_id="v001", energy=30.0,
                     ts=TS, soc=0.5) -> ChargingSession:
    return charger.start_session(vehicle_id, energy, ts, vehicle_soc=soc)


# ---------------------------------------------------------------------------
# ChargingSession
# ---------------------------------------------------------------------------

class TestChargingSession:
    def test_remaining_time_future_end(self):
        session = ChargingSession(
            vehicle_id="v1",
            start_time=TS,
            requested_energy_kwh=30.0,
            expected_end_time=TS + timedelta(minutes=30),
        )
        remaining = session.remaining_time(TS)
        assert remaining == timedelta(minutes=30)

    def test_remaining_time_past_end_returns_zero(self):
        session = ChargingSession(
            vehicle_id="v1",
            start_time=TS,
            requested_energy_kwh=30.0,
            expected_end_time=TS - timedelta(minutes=5),
        )
        assert session.remaining_time(TS) == timedelta(0)

    def test_remaining_time_no_end_time_returns_zero(self):
        session = ChargingSession(
            vehicle_id="v1",
            start_time=TS,
            requested_energy_kwh=30.0,
            expected_end_time=None,
        )
        assert session.remaining_time(TS) == timedelta(0)

    def test_completion_percentage_zero_at_start(self):
        session = ChargingSession(
            vehicle_id="v1",
            start_time=TS,
            requested_energy_kwh=30.0,
            delivered_energy_kwh=0.0,
        )
        assert session.completion_percentage() == pytest.approx(0.0)

    def test_completion_percentage_halfway(self):
        session = ChargingSession(
            vehicle_id="v1",
            start_time=TS,
            requested_energy_kwh=30.0,
            delivered_energy_kwh=15.0,
        )
        assert session.completion_percentage() == pytest.approx(50.0)

    def test_completion_percentage_capped_at_100(self):
        """Over-delivery is capped at 100%."""
        session = ChargingSession(
            vehicle_id="v1",
            start_time=TS,
            requested_energy_kwh=30.0,
            delivered_energy_kwh=50.0,
        )
        assert session.completion_percentage() == pytest.approx(100.0)

    def test_completion_percentage_zero_requested_returns_zero(self):
        session = ChargingSession(
            vehicle_id="v1",
            start_time=TS,
            requested_energy_kwh=0.0,
        )
        assert session.completion_percentage() == pytest.approx(0.0)

    def test_remaining_time_at_exact_end_time(self):
        session = ChargingSession(
            vehicle_id="v1",
            start_time=TS,
            requested_energy_kwh=30.0,
            expected_end_time=TS + timedelta(minutes=10),
        )
        # current_time == expected_end_time → remaining = 0
        assert session.remaining_time(TS + timedelta(minutes=10)) == timedelta(0)


# ---------------------------------------------------------------------------
# Charger.is_available()
# ---------------------------------------------------------------------------

class TestChargerIsAvailable:
    def test_available_status_returns_true(self):
        c = make_charger(status=ChargerStatus.AVAILABLE)
        assert c.is_available() is True

    def test_occupied_status_returns_false(self):
        c = make_charger(status=ChargerStatus.OCCUPIED)
        assert c.is_available() is False

    def test_maintenance_status_returns_false(self):
        c = make_charger(status=ChargerStatus.MAINTENANCE)
        assert c.is_available() is False

    def test_unpowered_status_returns_false(self):
        c = make_charger(status=ChargerStatus.UNPOWERED)
        assert c.is_available() is False


# ---------------------------------------------------------------------------
# Charger.start_session()
# ---------------------------------------------------------------------------

class TestChargerStartSession:
    def test_start_session_returns_charging_session(self):
        c = make_charger()
        session = start_session_on(c)
        assert isinstance(session, ChargingSession)

    def test_start_session_sets_occupied(self):
        c = make_charger()
        start_session_on(c)
        assert c.status == ChargerStatus.OCCUPIED

    def test_start_session_stores_vehicle_id(self):
        c = make_charger()
        session = start_session_on(c, vehicle_id="vehicle-42")
        assert session.vehicle_id == "vehicle-42"

    def test_start_session_stores_requested_energy(self):
        c = make_charger()
        session = start_session_on(c, energy=45.0)
        assert session.requested_energy_kwh == pytest.approx(45.0)

    def test_start_session_increments_total_sessions(self):
        c = make_charger()
        assert c.total_sessions == 0
        start_session_on(c)
        assert c.total_sessions == 1

    def test_start_session_sets_expected_end_time(self):
        """end_time must be after start_time."""
        c = make_charger(power_kw=50.0)
        session = start_session_on(c, energy=30.0, ts=TS, soc=0.5)
        assert session.expected_end_time is not None
        assert session.expected_end_time > TS

    def test_start_session_duration_reflects_power(self):
        """Higher power charger → shorter estimated duration."""
        slow = make_charger(power_kw=22.0)
        fast = make_charger(power_kw=150.0)
        s_slow = start_session_on(slow, energy=30.0)
        s_fast = start_session_on(fast, energy=30.0)
        # fast charger should finish sooner
        assert s_fast.expected_end_time < s_slow.expected_end_time

    def test_start_session_on_occupied_raises_valueerror(self):
        c = make_charger()
        start_session_on(c)
        with pytest.raises(ValueError, match="not available"):
            start_session_on(c, vehicle_id="v002")

    def test_start_session_sets_current_session(self):
        c = make_charger()
        session = start_session_on(c)
        assert c.current_session is session

    def test_delivered_energy_starts_at_zero(self):
        c = make_charger()
        session = start_session_on(c)
        assert session.delivered_energy_kwh == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Charger.update_session()
# ---------------------------------------------------------------------------

class TestChargerUpdateSession:
    def test_update_increases_delivered_energy(self):
        c = make_charger(power_kw=50.0)
        start_session_on(c)
        c.update_session(1.0)  # 1 minute
        # delivered = 50 * 0.9 * (1/60) ≈ 0.75 kWh
        assert c.current_session.delivered_energy_kwh > 0.0

    def test_update_delivered_proportional_to_time(self):
        c = make_charger(power_kw=50.0)
        start_session_on(c)
        c.update_session(1.0)
        energy_1min = c.current_session.delivered_energy_kwh

        c2 = make_charger(power_kw=50.0)
        start_session_on(c2)
        c2.update_session(2.0)
        energy_2min = c2.current_session.delivered_energy_kwh

        assert energy_2min == pytest.approx(2 * energy_1min, rel=1e-6)

    def test_update_does_not_auto_complete_session(self):
        """update_session never auto-completes — always returns None."""
        c = make_charger(power_kw=150.0)
        start_session_on(c, energy=1.0)  # Tiny requested energy
        result = c.update_session(60.0)  # 1 hour — more than enough to deliver
        assert result is None
        # Session is still active
        assert c.current_session is not None
        assert c.status == ChargerStatus.OCCUPIED

    def test_update_with_no_session_returns_none(self):
        c = make_charger()
        result = c.update_session(1.0)
        assert result is None

    def test_multiple_updates_accumulate_energy(self):
        c = make_charger(power_kw=50.0)
        start_session_on(c)
        c.update_session(1.0)
        c.update_session(1.0)
        c.update_session(1.0)
        # 3 minutes total
        single = make_charger(power_kw=50.0)
        start_session_on(single)
        single.update_session(3.0)
        assert c.current_session.delivered_energy_kwh == pytest.approx(
            single.current_session.delivered_energy_kwh, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Charger.complete_session()
# ---------------------------------------------------------------------------

class TestChargerCompleteSession:
    def test_complete_session_returns_completed_session(self):
        c = make_charger()
        session = start_session_on(c)
        completed = c.complete_session()
        assert completed is session

    def test_complete_session_sets_available(self):
        c = make_charger()
        start_session_on(c)
        c.complete_session()
        assert c.status == ChargerStatus.AVAILABLE

    def test_complete_session_clears_current_session(self):
        c = make_charger()
        start_session_on(c)
        c.complete_session()
        assert c.current_session is None

    def test_complete_session_accumulates_energy_delivered(self):
        c = make_charger(power_kw=50.0)
        start_session_on(c, energy=30.0)
        c.update_session(30.0)  # 30 minutes
        expected = c.current_session.delivered_energy_kwh
        c.complete_session()
        assert c.total_energy_delivered_kwh == pytest.approx(expected)

    def test_complete_session_no_active_session_raises(self):
        c = make_charger()
        with pytest.raises(ValueError, match="No active session"):
            c.complete_session()

    def test_complete_session_makes_charger_available_for_reuse(self):
        c = make_charger()
        start_session_on(c, vehicle_id="v1")
        c.complete_session()
        # Must be able to start a new session
        session2 = start_session_on(c, vehicle_id="v2")
        assert session2.vehicle_id == "v2"
        assert c.total_sessions == 2


# ---------------------------------------------------------------------------
# Charger.interrupt_session()
# ---------------------------------------------------------------------------

class TestChargerInterruptSession:
    def test_interrupt_returns_the_session(self):
        c = make_charger()
        session = start_session_on(c)
        interrupted = c.interrupt_session("user_request")
        assert interrupted is session

    def test_interrupt_clears_expected_end_time(self):
        """Interrupt marks the session as incomplete by nulling expected_end_time."""
        c = make_charger()
        session = start_session_on(c)
        assert session.expected_end_time is not None
        c.interrupt_session()
        assert session.expected_end_time is None

    def test_interrupt_frees_charger(self):
        c = make_charger()
        start_session_on(c)
        c.interrupt_session()
        assert c.status == ChargerStatus.AVAILABLE

    def test_interrupt_with_no_session_raises(self):
        c = make_charger()
        with pytest.raises(ValueError, match="No active session"):
            c.interrupt_session()

    def test_partial_energy_preserved_after_interrupt(self):
        """Delivered energy up to interrupt point is preserved in the returned session."""
        c = make_charger(power_kw=50.0)
        start_session_on(c, energy=30.0)
        c.update_session(5.0)  # 5 minutes of charging
        partial_energy = c.current_session.delivered_energy_kwh
        interrupted = c.interrupt_session()
        assert interrupted.delivered_energy_kwh == pytest.approx(partial_energy)


# ---------------------------------------------------------------------------
# Charger.estimate_availability()
# ---------------------------------------------------------------------------

class TestChargerEstimateAvailability:
    def test_available_charger_returns_current_time(self):
        c = make_charger()
        result = c.estimate_availability(TS)
        assert result == TS

    def test_occupied_charger_returns_end_time_plus_buffer(self):
        c = make_charger()
        session = start_session_on(c)
        end_time = session.expected_end_time
        result = c.estimate_availability(TS)
        expected = end_time + timedelta(minutes=2)
        assert result == expected

    def test_occupied_no_end_time_returns_conservative_default(self):
        """No expected_end_time → returns current_time + 1 hour (conservative)."""
        c = make_charger()
        session = start_session_on(c)
        session.expected_end_time = None  # Simulate unknown end
        result = c.estimate_availability(TS)
        assert result == TS + timedelta(hours=1)

    def test_availability_always_in_future_or_now(self):
        c = make_charger()
        result = c.estimate_availability(TS)
        assert result >= TS


# ---------------------------------------------------------------------------
# Cumulative stats
# ---------------------------------------------------------------------------

class TestChargerCumulativeStats:
    def test_total_sessions_increments_per_session(self):
        c = make_charger()
        for i in range(3):
            start_session_on(c, vehicle_id=f"v{i}")
            c.complete_session()
        assert c.total_sessions == 3

    def test_total_energy_accumulates_across_sessions(self):
        c = make_charger(power_kw=50.0)
        for i in range(3):
            start_session_on(c, vehicle_id=f"v{i}")
            c.update_session(10.0)
            c.complete_session()
        # Each session: 50 * 0.9 * (10/60) ≈ 7.5 kWh → 3x ≈ 22.5 kWh
        assert c.total_energy_delivered_kwh == pytest.approx(3 * 50 * 0.9 * (10/60))

    def test_id_is_string(self):
        c = make_charger()
        assert isinstance(c.id, str)
        assert len(c.id) > 0
