"""
Shared pytest fixtures for the EV Charge Manager test suite.
"""

import pytest
from ev_charge_manager.vehicle.vehicle import Battery, DriverBehavior


@pytest.fixture
def default_battery():
    """A battery with default parameters: 70 kWh capacity, 80% SOC."""
    return Battery()


@pytest.fixture
def full_battery():
    """A battery at maximum usable SOC (100%)."""
    return Battery(capacity_kwh=70.0, current_soc=1.0)


@pytest.fixture
def low_battery():
    """A battery just above the critical threshold."""
    return Battery(capacity_kwh=70.0, current_soc=0.16)


@pytest.fixture
def depleted_battery():
    """A battery at exactly the minimum operating SOC."""
    return Battery(capacity_kwh=70.0, current_soc=0.05)


@pytest.fixture
def default_driver():
    """A balanced driver with default parameters."""
    return DriverBehavior(behavior_type="balanced")


@pytest.fixture
def conservative_driver():
    """A conservative driver with high patience and low risk tolerance."""
    return DriverBehavior(
        behavior_type="conservative",
        patience_base=0.9,
        patience_decay_rate=0.1,
        risk_tolerance=0.1,
        max_wait_acceptable_min=60.0,
        speed_preference_kmh=110.0,
    )


@pytest.fixture
def aggressive_driver():
    """An aggressive driver with low patience and high risk tolerance."""
    return DriverBehavior(
        behavior_type="aggressive",
        patience_base=0.3,
        patience_decay_rate=0.8,
        risk_tolerance=0.9,
        max_wait_acceptable_min=10.0,
        speed_preference_kmh=135.0,
        acceleration_aggression=0.9,
    )
