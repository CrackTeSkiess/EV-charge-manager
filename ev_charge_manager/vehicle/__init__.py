"""vehicle – EV agent, battery, driver behaviour and vehicle tracker.

VehicleGenerator is intentionally NOT eagerly imported here because it has a
top-level dependency on Environment, which would create a circular import chain
(Environment → Highway → Vehicle → VehicleGenerator → Environment).
Import it directly when needed:
    from ev_charge_manager.vehicle.generator import VehicleGenerator
"""

from .vehicle import Vehicle, VehicleState, Battery, DriverBehavior
from .tracker import VehicleTracker, TrackedState, VehicleRecord, StepMetrics

__all__ = [
    "Vehicle", "VehicleState", "Battery", "DriverBehavior",
    "VehicleTracker", "TrackedState", "VehicleRecord", "StepMetrics",
]
