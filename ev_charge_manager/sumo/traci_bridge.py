"""
TraCI bridge for connecting SUMO traffic simulation to micro-RL energy agents.

Manages the TraCI connection lifecycle and translates per-vehicle
charging state into aggregate station-level demand_kw suitable for
HierarchicalEnergyManager.step().
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Ensure traci is importable via SUMO_HOME
_sumo_home = os.environ.get("SUMO_HOME")
if _sumo_home:
    _tools = os.path.join(_sumo_home, "tools")
    if _tools not in sys.path:
        sys.path.append(_tools)


@dataclass
class ChargingVehicleInfo:
    """Snapshot of a vehicle currently at or near a charging station."""
    vehicle_id: str
    station_index: int
    battery_capacity_wh: float
    current_soc: float
    charging_power_kw: float
    is_charging: bool
    is_waiting: bool


@dataclass
class StationSnapshot:
    """Aggregate state of a single charging station at a point in time."""
    station_index: int
    demand_kw: float
    n_vehicles_charging: int
    n_vehicles_waiting: int
    vehicles: List[ChargingVehicleInfo] = field(default_factory=list)


@dataclass
class TrafficSnapshot:
    """Global traffic statistics at a point in time."""
    simulation_time_s: float
    active_vehicles: int
    arrived_vehicles: int
    departed_vehicles: int
    teleported_vehicles: int


class TraCIBridge:
    """
    Bridge between a running SUMO simulation and the EV charging
    energy management system.

    Typical usage::

        bridge = TraCIBridge(
            sumo_config_path="output/highway.sumocfg",
            charging_edge_ids=["charging_station_0", ...],
            n_chargers=[6, 8, 4],
        )
        bridge.start()

        for hour in range(24):
            bridge.advance_to((hour + 1) * 3600)
            demands = bridge.get_all_station_demands()
            # feed demands[i] to HierarchicalEnergyManager.step()

        bridge.close()
    """

    # Default charger power used when calculating demand from vehicle count
    DEFAULT_CHARGER_POWER_KW = 150.0

    def __init__(
        self,
        sumo_config_path: str,
        charging_edge_ids: List[str],
        n_chargers: List[int],
        sumo_binary: str = "sumo",
        step_length_s: float = 1.0,
        port: Optional[int] = None,
    ):
        self.sumo_config_path = sumo_config_path
        self.charging_edge_ids = list(charging_edge_ids)
        self.n_chargers = list(n_chargers)
        self.n_stations = len(charging_edge_ids)
        self.sumo_binary = sumo_binary
        self.step_length_s = step_length_s
        self.port = port
        self._traci = None       # traci module reference
        self._started = False

        # Cumulative stats
        self.total_vehicles_served: List[int] = [0] * self.n_stations
        self.total_charging_energy_kwh: List[float] = [0.0] * self.n_stations

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch SUMO and establish a TraCI connection."""
        import traci
        self._traci = traci

        sumo_home = os.environ.get("SUMO_HOME", "")
        binary = os.path.join(sumo_home, "bin", self.sumo_binary) if sumo_home else self.sumo_binary

        cmd = [
            binary,
            "-c", self.sumo_config_path,
            "--step-length", str(self.step_length_s),
            "--no-step-log", "true",
        ]

        if self.port is not None:
            traci.start(cmd, port=self.port)
        else:
            traci.start(cmd)

        self._started = True

    def close(self) -> None:
        """Close the TraCI connection and shut down SUMO."""
        if self._started and self._traci is not None:
            try:
                self._traci.close()
            except Exception:
                pass
            self._started = False

    @property
    def is_running(self) -> bool:
        """Whether the simulation is currently active."""
        if not self._started or self._traci is None:
            return False
        try:
            self._traci.simulation.getTime()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def advance_to(self, target_time_s: float) -> None:
        """
        Step SUMO forward until simulation time reaches *target_time_s*.

        This may involve many 1-second SUMO steps.  The method blocks
        until the target time is reached or the simulation ends.
        """
        traci = self._traci
        while traci.simulation.getTime() < target_time_s:
            traci.simulationStep()
            # Check if the simulation has ended (no more vehicles)
            if (traci.simulation.getMinExpectedNumber() <= 0
                    and traci.simulation.getTime() > 3600):
                break

    def get_simulation_time(self) -> float:
        """Current SUMO simulation time in seconds."""
        return self._traci.simulation.getTime()

    # ------------------------------------------------------------------
    # Demand queries
    # ------------------------------------------------------------------

    def get_station_demand(self, station_index: int) -> StationSnapshot:
        """
        Query demand at a specific charging station.

        Vehicles at SUMO charging stations are tracked via
        ``traci.chargingstation`` (they become invisible to edge queries
        while plugged in).  We also check the edge and parking area for
        any vehicles that are approaching or waiting in the queue.
        """
        traci = self._traci
        edge_id = self.charging_edge_ids[station_index]
        # Derive charging-station ID from edge naming convention
        # Edge: "charging_station_X"  →  cs: "cs_station_X"
        station_suffix = edge_id.replace("charging_", "")
        cs_id = f"cs_{station_suffix}"

        # --- Vehicles actively at the SUMO charging station ---
        cs_vehicle_count = 0
        try:
            cs_vehicle_count = traci.chargingstation.getVehicleCount(cs_id)
        except Exception:
            pass

        # --- Vehicles on the edge (approaching or queuing) ---
        edge_vehicle_ids: List[str] = []
        try:
            edge_vehicle_ids = list(
                traci.edge.getLastStepVehicleIDs(edge_id)
            )
        except Exception:
            pass

        # Build vehicle info for vehicles visible on the edge
        vehicles: List[ChargingVehicleInfo] = []
        n_waiting = 0
        for vid in edge_vehicle_ids:
            try:
                max_cap_wh = float(traci.vehicle.getParameter(
                    vid, "device.battery.maximumBatteryCapacity"))
                actual_cap_wh = float(traci.vehicle.getParameter(
                    vid, "device.battery.actualBatteryCapacity"))
                soc = actual_cap_wh / max(1.0, max_cap_wh)
            except Exception:
                max_cap_wh = 60000.0
                soc = 0.5

            try:
                is_stopped = traci.vehicle.getSpeed(vid) < 0.1
            except Exception:
                is_stopped = False

            if is_stopped:
                n_waiting += 1

            vehicles.append(ChargingVehicleInfo(
                vehicle_id=vid,
                station_index=station_index,
                battery_capacity_wh=max_cap_wh,
                current_soc=soc,
                charging_power_kw=0.0,
                is_charging=False,
                is_waiting=is_stopped,
            ))

        # --- Compute demand from the charging station vehicle count ---
        # We use the count (not individual vehicle data, since they are
        # not visible via traci.vehicle while at a charging station).
        # Assume average SOC of ~0.5 → full power charging.
        max_chargers = self.n_chargers[station_index]
        n_charging = min(cs_vehicle_count, max_chargers)
        n_cs_waiting = max(0, cs_vehicle_count - max_chargers)

        # Average power per charging vehicle (SOC-weighted estimate)
        avg_power_kw = self.DEFAULT_CHARGER_POWER_KW * 0.85  # ~85% avg
        total_power_kw = n_charging * avg_power_kw

        # Minimum station load (base load from lighting, HVAC, etc.)
        demand_kw = max(10.0, total_power_kw)

        return StationSnapshot(
            station_index=station_index,
            demand_kw=demand_kw,
            n_vehicles_charging=n_charging,
            n_vehicles_waiting=n_waiting + n_cs_waiting,
            vehicles=vehicles,
        )

    def get_all_station_demands(self) -> Tuple[List[float], List[StationSnapshot]]:
        """
        Get demand_kw for every station.

        Returns:
            (demands, snapshots) where demands[i] is the total kW demand
            at station i, and snapshots[i] contains full details.
        """
        demands = []
        snapshots = []
        for i in range(self.n_stations):
            snap = self.get_station_demand(i)
            demands.append(snap.demand_kw)
            snapshots.append(snap)
            # Update cumulative stats
            self.total_vehicles_served[i] += snap.n_vehicles_charging
        return demands, snapshots

    # ------------------------------------------------------------------
    # Traffic statistics
    # ------------------------------------------------------------------

    def get_traffic_stats(self) -> TrafficSnapshot:
        """Get global traffic statistics from SUMO."""
        traci = self._traci
        return TrafficSnapshot(
            simulation_time_s=traci.simulation.getTime(),
            active_vehicles=traci.vehicle.getIDCount(),
            arrived_vehicles=traci.simulation.getArrivedNumber(),
            departed_vehicles=traci.simulation.getDepartedNumber(),
            teleported_vehicles=traci.simulation.getStartingTeleportNumber(),
        )

    def get_stranded_vehicle_count(self) -> int:
        """
        Count vehicles that have been teleported (ran out of battery
        or got stuck), which approximates 'stranded' vehicles.
        """
        return self._traci.simulation.getStartingTeleportNumber()
