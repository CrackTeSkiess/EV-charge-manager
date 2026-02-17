"""energy – energy sources, manager, hierarchical manager and micro-RL agent.

PyTorch is required for HierarchicalEnergyManager and EnergyManagerAgent.
Install RL support with:  pip install torch gymnasium
"""

from .manager import (
    CHARGER_RATED_POWER_KW, CHARGER_AVG_DRAW_FACTOR,
    EnergySourceType, EnergySourceConfig,
    GridSourceConfig, SolarSourceConfig, WindSourceConfig, BatteryStorageConfig,
    EnergyManagerConfig, EnergySource, EnergyManager,
)

__all__ = [
    "CHARGER_RATED_POWER_KW", "CHARGER_AVG_DRAW_FACTOR",
    "EnergySourceType", "EnergySourceConfig",
    "GridSourceConfig", "SolarSourceConfig", "WindSourceConfig", "BatteryStorageConfig",
    "EnergyManagerConfig", "EnergySource", "EnergyManager",
]

try:
    from .hierarchical_manager import HierarchicalEnergyManager
    from .agent import GridPricingSchedule, EnergyManagerNetwork, EnergyManagerAgent
    __all__ += ["HierarchicalEnergyManager", "GridPricingSchedule", "EnergyManagerNetwork", "EnergyManagerAgent"]
except ModuleNotFoundError:
    pass  # pip install torch  to enable RL components
