"""analytics – KPI tracking, station data collection and event trackers.

KPITracker, StationDataCollector, BlackoutTracker and StrandedVehicleTracker
are always available. EarlyWarningExtractor requires PyTorch.
"""

from .kpi_tracker import KPIAlertLevel, KPIThreshold, KPIRecord, KPITracker
from .station_collector import StationSnapshot, StationDataCollector
from .blackout_tracker import BlackoutType, EnergyShortfallEvent, BlackoutTracker
from .stranded_tracker import VehicleStatus, StrandedVehicleTracker

__all__ = [
    "KPIAlertLevel", "KPIThreshold", "KPIRecord", "KPITracker",
    "StationSnapshot", "StationDataCollector",
    "BlackoutType", "EnergyShortfallEvent", "BlackoutTracker",
    "VehicleStatus", "StrandedVehicleTracker",
]

try:
    from .early_warning import WarningEvent, EarlyWarningExtractor
    __all__ += ["WarningEvent", "EarlyWarningExtractor"]
except ModuleNotFoundError:
    pass  # pip install torch  to enable early warning system
