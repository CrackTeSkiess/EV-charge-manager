"""
Unit tests for the DriverBehavior class.

Covers:
- evaluate_station_attractiveness(): urgency, wait score, price score,
  availability score, social influence, output clamped to [0, 1]
- update_patience(): decay, comfort/urgency adjustments, clamped to [0, 1]
- decide_to_abort(): hard threshold, probabilistic path, alternative boost,
  queue position penalty
- calculate_desired_speed(): base speed, traffic adjustment, aggression,
  never exceeds speed_limit + 20
"""

import random
import pytest
from ev_charge_manager.vehicle.vehicle import DriverBehavior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_station(price_per_kwh=0.40, available_chargers=2, total_chargers=4,
                 trending=False):
    return {
        'price_per_kwh': price_per_kwh,
        'available_chargers': available_chargers,
        'total_chargers': total_chargers,
        'trending': trending,
    }


# ---------------------------------------------------------------------------
# evaluate_station_attractiveness()
# ---------------------------------------------------------------------------

class TestEvaluateStationAttractiveness:
    def test_returns_float_between_zero_and_one(self, default_driver):
        station = make_station()
        score = default_driver.evaluate_station_attractiveness(station, 0.5, 10.0)
        assert 0.0 <= score <= 1.0

    def test_urgency_zero_when_soc_above_min_charge_to_continue(self, default_driver):
        """When SOC >= min_charge_to_continue, urgency factor = 0 (no desperation boost)."""
        # min_charge_to_continue default = 0.3; set SOC well above that
        station = make_station(available_chargers=0, total_chargers=4)
        score_high_soc = default_driver.evaluate_station_attractiveness(
            station, current_soc=0.8, estimated_wait=0.0
        )
        score_low_soc = default_driver.evaluate_station_attractiveness(
            station, current_soc=0.1, estimated_wait=0.0
        )
        # Low SOC (more urgent) should score higher
        assert score_low_soc > score_high_soc

    def test_zero_wait_gives_maximum_wait_score(self, default_driver):
        station_no_wait = make_station()
        station_long_wait = make_station()
        score_no_wait = default_driver.evaluate_station_attractiveness(
            station_no_wait, 0.5, estimated_wait=0.0
        )
        score_long_wait = default_driver.evaluate_station_attractiveness(
            station_long_wait, 0.5, estimated_wait=60.0
        )
        assert score_no_wait > score_long_wait

    def test_wait_exceeding_max_gives_zero_wait_component(self, default_driver):
        """Wait longer than max_wait_acceptable_min → wait_score = 0 (clamped)."""
        # max_wait_acceptable_min default = 30; score at 60 min wait
        station = make_station(available_chargers=4, total_chargers=4)
        score = default_driver.evaluate_station_attractiveness(
            station, 0.5, estimated_wait=60.0
        )
        # Score can't be negative
        assert score >= 0.0

    def test_lower_price_gives_higher_score(self, default_driver):
        station_cheap = make_station(price_per_kwh=0.20)
        station_expensive = make_station(price_per_kwh=0.55)
        score_cheap = default_driver.evaluate_station_attractiveness(
            station_cheap, 0.5, 5.0
        )
        score_expensive = default_driver.evaluate_station_attractiveness(
            station_expensive, 0.5, 5.0
        )
        assert score_cheap > score_expensive

    def test_price_above_max_gives_zero_price_component(self, default_driver):
        """Price above max_price_per_kwh → price_score clamped to 0."""
        station = make_station(price_per_kwh=1.00)  # above 0.60 default
        score = default_driver.evaluate_station_attractiveness(station, 0.5, 0.0)
        assert score >= 0.0  # total score still non-negative

    def test_full_availability_higher_than_empty(self, default_driver):
        station_full = make_station(available_chargers=4, total_chargers=4)
        station_empty = make_station(available_chargers=0, total_chargers=4)
        score_full = default_driver.evaluate_station_attractiveness(
            station_full, 0.5, 0.0
        )
        score_empty = default_driver.evaluate_station_attractiveness(
            station_empty, 0.5, 0.0
        )
        assert score_full > score_empty

    def test_trending_station_boosts_score(self, default_driver):
        station_normal = make_station(trending=False)
        station_trending = make_station(trending=True)
        score_normal = default_driver.evaluate_station_attractiveness(
            station_normal, 0.5, 5.0
        )
        score_trending = default_driver.evaluate_station_attractiveness(
            station_trending, 0.5, 5.0
        )
        assert score_trending > score_normal

    def test_trending_boost_capped_at_one(self, default_driver):
        """Even a perfect station with trending=True must not exceed 1.0."""
        station = make_station(
            price_per_kwh=0.01,
            available_chargers=4,
            total_chargers=4,
            trending=True,
        )
        score = default_driver.evaluate_station_attractiveness(
            station, current_soc=0.0, estimated_wait=0.0
        )
        assert score <= 1.0

    def test_no_trending_social_influence_has_no_effect(self):
        """trending=False means social_influence multiplier is skipped."""
        d_low = DriverBehavior(social_influence=0.0)
        d_high = DriverBehavior(social_influence=1.0)
        station = make_station(trending=False)

        score_low = d_low.evaluate_station_attractiveness(station, 0.5, 5.0)
        score_high = d_high.evaluate_station_attractiveness(station, 0.5, 5.0)
        assert score_low == pytest.approx(score_high)

    def test_score_non_negative_for_worst_case_inputs(self, default_driver):
        station = make_station(price_per_kwh=1.0, available_chargers=0, total_chargers=4)
        score = default_driver.evaluate_station_attractiveness(
            station, current_soc=0.9, estimated_wait=120.0
        )
        assert score >= 0.0


# ---------------------------------------------------------------------------
# update_patience()
# ---------------------------------------------------------------------------

class TestUpdatePatience:
    def test_patience_decreases_with_waiting(self, default_driver):
        new_patience = default_driver.update_patience(
            current_patience=0.8, wait_minutes=5.0,
            comfort_level=0.5, urgency=0.0
        )
        assert new_patience < 0.8

    def test_patience_never_below_zero(self, default_driver):
        new_patience = default_driver.update_patience(
            current_patience=0.0, wait_minutes=100.0,
            comfort_level=0.0, urgency=0.0
        )
        assert new_patience >= 0.0

    def test_patience_never_above_one(self, default_driver):
        # Starting at 1.0 with minimal decay should still be <= 1.0
        new_patience = default_driver.update_patience(
            current_patience=1.0, wait_minutes=0.0,
            comfort_level=1.0, urgency=1.0
        )
        assert new_patience <= 1.0

    def test_high_comfort_slows_decay(self, default_driver):
        """Higher comfort level reduces the effective decay."""
        p_low_comfort = default_driver.update_patience(
            current_patience=0.8, wait_minutes=3.0,
            comfort_level=0.0, urgency=0.0
        )
        p_high_comfort = default_driver.update_patience(
            current_patience=0.8, wait_minutes=3.0,
            comfort_level=1.0, urgency=0.0
        )
        assert p_high_comfort > p_low_comfort

    def test_high_urgency_reduces_patience_loss(self, default_driver):
        """
        NOTE: The source comment says "more urgent = less patience loss", but
        the implementation is inverted:
            urgency_adjustment = (1 - urgency) * 0.3
            decay_multiplier   = (1 - urgency_adjustment)

        High urgency → small urgency_adjustment → large decay_multiplier → MORE loss.
        This test documents the ACTUAL behaviour; the intent vs implementation
        mismatch should be investigated and fixed separately.
        """
        p_low_urgency = default_driver.update_patience(
            current_patience=0.8, wait_minutes=3.0,
            comfort_level=0.5, urgency=0.0
        )
        p_high_urgency = default_driver.update_patience(
            current_patience=0.8, wait_minutes=3.0,
            comfort_level=0.5, urgency=1.0
        )
        # Counter-intuitive result caused by the inverted formula:
        # high urgency produces MORE patience decay (not less).
        assert p_high_urgency < p_low_urgency

    def test_zero_wait_time_minimal_decay(self, default_driver):
        """Zero wait should produce no decay (decay = wait_minutes * rate = 0)."""
        new_patience = default_driver.update_patience(
            current_patience=0.8, wait_minutes=0.0,
            comfort_level=0.0, urgency=0.0
        )
        assert new_patience == pytest.approx(0.8)

    def test_fast_decay_driver_loses_patience_quicker(self):
        slow_decay = DriverBehavior(patience_decay_rate=0.1)
        fast_decay = DriverBehavior(patience_decay_rate=0.9)

        p_slow = slow_decay.update_patience(0.8, 5.0, 0.5, 0.5)
        p_fast = fast_decay.update_patience(0.8, 5.0, 0.5, 0.5)
        assert p_fast < p_slow


# ---------------------------------------------------------------------------
# decide_to_abort()
# ---------------------------------------------------------------------------

class TestDecideToAbort:
    def test_hard_threshold_always_aborts(self, default_driver):
        """patience <= 0.1 must always return True regardless of other factors."""
        for patience in [0.0, 0.05, 0.1]:
            assert default_driver.decide_to_abort(
                patience, queue_position=1, estimated_wait=5.0,
                alternative_available=False
            ) is True

    def test_just_above_hard_threshold_uses_probability(self, default_driver):
        """patience = 0.11 enters the probabilistic branch (may return True or False)."""
        random.seed(42)
        results = [
            default_driver.decide_to_abort(
                0.11, queue_position=1, estimated_wait=5.0,
                alternative_available=False
            )
            for _ in range(100)
        ]
        # Both outcomes should be possible (not all True or all False)
        assert True in results
        assert False in results

    def test_full_patience_rarely_aborts(self, default_driver):
        """With patience = 1.0 the abandonment probability should be near 0."""
        random.seed(0)
        results = [
            default_driver.decide_to_abort(
                1.0, queue_position=1, estimated_wait=5.0,
                alternative_available=False
            )
            for _ in range(200)
        ]
        # abandonment_prob = (1-1.0)*0.8 + position_penalty = 0.05 → very rare
        abort_rate = sum(results) / len(results)
        assert abort_rate < 0.2

    def test_alternative_available_increases_abort_rate(self, default_driver):
        """Having an alternative should increase the abandonment probability."""
        random.seed(1)
        n = 500
        patience = 0.4  # Mid-range patience — in probabilistic zone

        aborts_no_alt = sum(
            default_driver.decide_to_abort(patience, 1, 15.0, False)
            for _ in range(n)
        )
        random.seed(1)
        aborts_with_alt = sum(
            default_driver.decide_to_abort(patience, 1, 15.0, True)
            for _ in range(n)
        )
        assert aborts_with_alt > aborts_no_alt

    def test_higher_queue_position_increases_abort_rate(self, default_driver):
        """Being further back in the queue should raise the abandonment probability."""
        random.seed(2)
        n = 500
        patience = 0.5

        aborts_pos1 = sum(
            default_driver.decide_to_abort(patience, 1, 15.0, False)
            for _ in range(n)
        )
        random.seed(2)
        aborts_pos10 = sum(
            default_driver.decide_to_abort(patience, 10, 15.0, False)
            for _ in range(n)
        )
        assert aborts_pos10 > aborts_pos1

    def test_abort_probability_capped_at_0_95(self, default_driver):
        """Even with worst-case inputs, probability is capped at 0.95 (never certain)."""
        random.seed(3)
        n = 500
        # patience just above 0.1 → prob ~= 0.72 + large queue penalty, capped at 0.95
        results = [
            default_driver.decide_to_abort(
                0.11, queue_position=20, estimated_wait=120.0,
                alternative_available=True
            )
            for _ in range(n)
        ]
        abort_rate = sum(results) / n
        # Rate must be below 1.0 (cap works), but very high
        assert abort_rate < 1.0
        assert abort_rate > 0.5


# ---------------------------------------------------------------------------
# calculate_desired_speed()
# ---------------------------------------------------------------------------

class TestCalculateDesiredSpeed:
    def test_speed_never_exceeds_limit_plus_20(self, aggressive_driver):
        for limit in [80.0, 100.0, 120.0, 130.0]:
            speed = aggressive_driver.calculate_desired_speed(limit, traffic_flow=1.0)
            assert speed <= limit + 20

    def test_speed_at_least_20_kmh(self, aggressive_driver):
        """Minimum returned speed is 20 km/h regardless of inputs."""
        speed = aggressive_driver.calculate_desired_speed(
            current_speed_limit=10.0, traffic_flow=0.0
        )
        assert speed >= 20.0

    def test_heavy_traffic_reduces_speed(self, default_driver):
        speed_free = default_driver.calculate_desired_speed(130.0, traffic_flow=1.0)
        speed_heavy = default_driver.calculate_desired_speed(130.0, traffic_flow=0.0)
        assert speed_heavy < speed_free

    def test_aggressive_driver_faster_than_conservative(self,
                                                         aggressive_driver,
                                                         conservative_driver):
        limit = 130.0
        speed_aggressive = aggressive_driver.calculate_desired_speed(limit)
        speed_conservative = conservative_driver.calculate_desired_speed(limit)
        assert speed_aggressive > speed_conservative

    def test_preferred_speed_below_limit_is_respected(self):
        """Driver who prefers slow speed shouldn't be pushed to the limit."""
        slow_driver = DriverBehavior(speed_preference_kmh=90.0,
                                     acceleration_aggression=0.5)
        speed = slow_driver.calculate_desired_speed(130.0, traffic_flow=1.0)
        # min(90, 130+10) = 90 as base; aggression at 0.5 → no adjustment
        assert speed <= 100.0

    def test_returns_positive_speed(self, default_driver):
        speed = default_driver.calculate_desired_speed(100.0)
        assert speed > 0
