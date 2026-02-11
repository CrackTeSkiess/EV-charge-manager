"""
SUMO network generator for EV charging infrastructure validation.

Translates the macro agent's best infrastructure configuration
(station positions, charger counts, energy sources) into a complete
set of SUMO simulation files: highway network, EV routes, charging
stations, and a ready-to-run .sumocfg.
"""

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional

from ev_charge_manager.data.traffic_profiles import (
    WEEKDAY_HOURLY_FRACTIONS,
    SATURDAY_HOURLY_FRACTIONS,
    SUNDAY_HOURLY_FRACTIONS,
    MONTHLY_MULTIPLIERS,
    DAY_OF_WEEK_MULTIPLIERS,
)


# ---------------------------------------------------------------------------
# Vehicle type specs (mirroring ev_charge_manager/vehicle/generator.py)
# ---------------------------------------------------------------------------

@dataclass
class _EVTypeSpec:
    """SUMO EV type derived from project VehicleTypeSpec."""
    type_id: str
    battery_capacity_kwh: float
    weight: float          # frequency weight
    initial_soc_low: float
    initial_soc_high: float
    length_m: float
    max_speed_ms: float    # m/s
    accel: float           # m/s^2
    decel: float           # m/s^2


EV_TYPES: List[_EVTypeSpec] = [
    _EVTypeSpec("ev_compact",  40.0, 0.25, 0.40, 0.90, 4.5, 36.1, 2.6, 4.5),
    _EVTypeSpec("ev_midsize",  60.0, 0.40, 0.30, 0.90, 4.8, 38.9, 2.8, 4.5),
    _EVTypeSpec("ev_premium",  80.0, 0.25, 0.50, 0.95, 5.0, 41.7, 3.0, 5.0),
    _EVTypeSpec("ev_truck",   100.0, 0.10, 0.60, 0.90, 7.0, 33.3, 1.5, 4.0),
]


class SUMONetworkGenerator:
    """
    Generate a complete SUMO simulation from a macro-agent infrastructure config.

    Produces:
        .nod.xml   – junction definitions
        .edg.xml   – edge (road) definitions
        .con.xml   – explicit connections for ramp merges
        .net.xml   – compiled network (via netconvert)
        .rou.xml   – EV vehicle types and traffic flows
        .add.xml   – parking areas and charging stations
        .sumocfg   – ready-to-run configuration
    """

    # Network geometry constants
    HIGHWAY_SPEED_MS = 36.11        # 130 km/h
    HIGHWAY_LANES = 2
    RAMP_LENGTH_M = 300.0
    RAMP_SPEED_MS = 8.33            # 30 km/h
    CHARGING_EDGE_LENGTH_M = 200.0
    CHARGING_EDGE_SPEED_MS = 5.56   # 20 km/h

    def __init__(
        self,
        highway_length_km: float,
        station_positions: List[float],
        n_chargers: List[int],
        n_waiting: List[int],
        output_dir: str,
        base_aadt: float = 50000.0,
        ev_penetration: float = 0.15,
        simulation_seconds: int = 86400,
        seed: int = 42,
    ):
        self.highway_length_km = highway_length_km
        self.highway_length_m = highway_length_km * 1000.0
        self.station_positions = list(station_positions)
        self.n_chargers = list(n_chargers)
        self.n_waiting = list(n_waiting)
        self.output_dir = output_dir
        self.base_aadt = base_aadt
        self.ev_penetration = ev_penetration
        self.simulation_seconds = simulation_seconds
        self.seed = seed
        self.n_stations = len(station_positions)

        # Edge IDs for external reference (populated during generate)
        self.charging_edge_ids: List[str] = []

        # File paths (set by generate)
        self.file_paths: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Dict[str, str]:
        """
        Generate all SUMO files and return their paths.

        Returns:
            dict with keys: "net", "routes", "additional", "config"
        """
        os.makedirs(self.output_dir, exist_ok=True)

        nod_path = self._generate_nodes()
        edg_path = self._generate_edges()
        con_path = self._generate_connections()
        net_path = self._run_netconvert(nod_path, edg_path, con_path)
        rou_path = self._generate_routes()
        add_path = self._generate_additional()
        cfg_path = self._generate_sumocfg(net_path, rou_path, add_path)

        self.file_paths = {
            "net": net_path,
            "routes": rou_path,
            "additional": add_path,
            "config": cfg_path,
        }
        return self.file_paths

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _generate_nodes(self) -> str:
        """Create .nod.xml with highway junctions and station ramp nodes."""
        root = ET.Element("nodes")

        # Highway start and end
        ET.SubElement(root, "node", id="highway_start", x="0.0", y="0.0",
                      type="priority")
        ET.SubElement(root, "node", id="highway_end",
                      x=f"{self.highway_length_m:.1f}", y="0.0",
                      type="priority")

        # Segment break points at each station + ramp nodes
        sorted_positions = sorted(enumerate(self.station_positions),
                                  key=lambda t: t[1])

        for idx, pos_km in sorted_positions:
            pos_m = pos_km * 1000.0
            sid = f"station_{idx}"

            # Highway split/merge nodes
            ET.SubElement(root, "node", id=f"hw_before_{sid}",
                          x=f"{pos_m - 50:.1f}", y="0.0", type="priority")
            ET.SubElement(root, "node", id=f"hw_after_{sid}",
                          x=f"{pos_m + 50:.1f}", y="0.0", type="priority")

            # Off-ramp / charging / on-ramp nodes (offset 80m south).
            # The charging edge must be long enough to hold multiple
            # parked vehicles (each ~5-7m).  We use 500m to comfortably
            # fit up to ~30 vehicles at any given time.
            ET.SubElement(root, "node", id=f"offramp_end_{sid}",
                          x=f"{pos_m - 300:.1f}", y="-80.0", type="priority")
            ET.SubElement(root, "node", id=f"charging_start_{sid}",
                          x=f"{pos_m - 250:.1f}", y="-80.0", type="priority")
            ET.SubElement(root, "node", id=f"charging_end_{sid}",
                          x=f"{pos_m + 250:.1f}", y="-80.0", type="priority")
            ET.SubElement(root, "node", id=f"onramp_start_{sid}",
                          x=f"{pos_m + 300:.1f}", y="-80.0", type="priority")

        path = os.path.join(self.output_dir, "highway.nod.xml")
        self._write_xml(root, path)
        return path

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def _generate_edges(self) -> str:
        """Create .edg.xml with highway segments and station ramp edges."""
        root = ET.Element("edges")

        sorted_positions = sorted(enumerate(self.station_positions),
                                  key=lambda t: t[1])

        # Build the highway edge sequence: start → station_0 before → after → station_1 before → ... → end
        prev_node = "highway_start"
        for i, (idx, _pos_km) in enumerate(sorted_positions):
            sid = f"station_{idx}"
            before = f"hw_before_{sid}"
            after = f"hw_after_{sid}"

            # Highway segment from previous node to this station's before-node
            ET.SubElement(root, "edge", id=f"highway_seg_{i}",
                          attrib={"from": prev_node, "to": before,
                                  "numLanes": str(self.HIGHWAY_LANES),
                                  "speed": f"{self.HIGHWAY_SPEED_MS:.2f}",
                                  "type": "highway"})

            # Through segment past the station
            ET.SubElement(root, "edge", id=f"highway_through_{idx}",
                          attrib={"from": before, "to": after,
                                  "numLanes": str(self.HIGHWAY_LANES),
                                  "speed": f"{self.HIGHWAY_SPEED_MS:.2f}",
                                  "type": "highway"})

            # Off-ramp: highway before → offramp end
            ET.SubElement(root, "edge", id=f"offramp_{sid}",
                          attrib={"from": before, "to": f"offramp_end_{sid}",
                                  "numLanes": "1",
                                  "speed": f"{self.RAMP_SPEED_MS:.2f}",
                                  "type": "ramp"})

            # Connector: offramp end → charging start
            ET.SubElement(root, "edge", id=f"connector_in_{sid}",
                          attrib={"from": f"offramp_end_{sid}",
                                  "to": f"charging_start_{sid}",
                                  "numLanes": "1",
                                  "speed": f"{self.CHARGING_EDGE_SPEED_MS:.2f}",
                                  "type": "internal"})

            # Charging edge
            charging_edge_id = f"charging_{sid}"
            ET.SubElement(root, "edge", id=charging_edge_id,
                          attrib={"from": f"charging_start_{sid}",
                                  "to": f"charging_end_{sid}",
                                  "numLanes": "1",
                                  "speed": f"{self.CHARGING_EDGE_SPEED_MS:.2f}",
                                  "type": "internal"})
            self.charging_edge_ids.append(charging_edge_id)

            # Connector: charging end → onramp start
            ET.SubElement(root, "edge", id=f"connector_out_{sid}",
                          attrib={"from": f"charging_end_{sid}",
                                  "to": f"onramp_start_{sid}",
                                  "numLanes": "1",
                                  "speed": f"{self.CHARGING_EDGE_SPEED_MS:.2f}",
                                  "type": "internal"})

            # On-ramp: onramp start → highway after
            ET.SubElement(root, "edge", id=f"onramp_{sid}",
                          attrib={"from": f"onramp_start_{sid}",
                                  "to": after,
                                  "numLanes": "1",
                                  "speed": f"{self.RAMP_SPEED_MS:.2f}",
                                  "type": "ramp"})

            prev_node = after

        # Final highway segment to end
        ET.SubElement(root, "edge", id=f"highway_seg_{len(sorted_positions)}",
                      attrib={"from": prev_node, "to": "highway_end",
                              "numLanes": str(self.HIGHWAY_LANES),
                              "speed": f"{self.HIGHWAY_SPEED_MS:.2f}",
                              "type": "highway"})

        path = os.path.join(self.output_dir, "highway.edg.xml")
        self._write_xml(root, path)
        return path

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------

    def _generate_connections(self) -> str:
        """
        Create .con.xml with explicit ramp diverge/merge connections.

        At each station, the node ``hw_before_station_X`` has one
        incoming edge (``highway_seg_X``) and two outgoing edges
        (``highway_through_X`` for through-traffic and ``offramp_station_X``
        for vehicles exiting to charge).  We explicitly connect the
        incoming edge to both outgoing edges so netconvert creates the
        correct lane topology.
        """
        root = ET.Element("connections")

        sorted_positions = sorted(enumerate(self.station_positions),
                                  key=lambda t: t[1])

        for seg_i, (idx, _pos_km) in enumerate(sorted_positions):
            sid = f"station_{idx}"

            # --- Diverge at hw_before node ---
            # highway_seg → highway_through (lanes 0..N)
            # highway_seg → offramp (lane 0 → 0)
            for lane in range(self.HIGHWAY_LANES):
                ET.SubElement(root, "connection",
                              attrib={"from": f"highway_seg_{seg_i}",
                                      "to": f"highway_through_{idx}",
                                      "fromLane": str(lane),
                                      "toLane": str(lane)})
            ET.SubElement(root, "connection",
                          attrib={"from": f"highway_seg_{seg_i}",
                                  "to": f"offramp_{sid}",
                                  "fromLane": "0", "toLane": "0"})

            # --- Ramp chain: offramp → connector_in → charging → connector_out → onramp ---
            ET.SubElement(root, "connection",
                          attrib={"from": f"offramp_{sid}",
                                  "to": f"connector_in_{sid}",
                                  "fromLane": "0", "toLane": "0"})
            ET.SubElement(root, "connection",
                          attrib={"from": f"connector_in_{sid}",
                                  "to": f"charging_{sid}",
                                  "fromLane": "0", "toLane": "0"})
            ET.SubElement(root, "connection",
                          attrib={"from": f"charging_{sid}",
                                  "to": f"connector_out_{sid}",
                                  "fromLane": "0", "toLane": "0"})
            ET.SubElement(root, "connection",
                          attrib={"from": f"connector_out_{sid}",
                                  "to": f"onramp_{sid}",
                                  "fromLane": "0", "toLane": "0"})

            # --- Merge at hw_after node ---
            # highway_through → next highway_seg (lanes 0..N)
            # onramp → next highway_seg (lane 0 → 0)
            next_seg = f"highway_seg_{seg_i + 1}"
            for lane in range(self.HIGHWAY_LANES):
                ET.SubElement(root, "connection",
                              attrib={"from": f"highway_through_{idx}",
                                      "to": next_seg,
                                      "fromLane": str(lane),
                                      "toLane": str(lane)})
            ET.SubElement(root, "connection",
                          attrib={"from": f"onramp_{sid}",
                                  "to": next_seg,
                                  "fromLane": "0", "toLane": "0"})

        path = os.path.join(self.output_dir, "highway.con.xml")
        self._write_xml(root, path)
        return path

    # ------------------------------------------------------------------
    # netconvert
    # ------------------------------------------------------------------

    def _run_netconvert(self, nod_path: str, edg_path: str,
                        con_path: str) -> str:
        """Run netconvert to compile .net.xml from nodes/edges/connections."""
        net_path = os.path.join(self.output_dir, "highway.net.xml")

        sumo_home = os.environ.get("SUMO_HOME", "")
        netconvert = os.path.join(sumo_home, "bin", "netconvert") if sumo_home else "netconvert"

        cmd = [
            netconvert,
            "--node-files", nod_path,
            "--edge-files", edg_path,
            "--connection-files", con_path,
            "--output-file", net_path,
            "--no-turnarounds", "true",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                f"netconvert not found. Ensure SUMO_HOME is set "
                f"(tried: {netconvert}). "
                f"Install SUMO: https://sumo.dlr.de/docs/Installing/index.html"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"netconvert failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
            )

        return net_path

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _generate_routes(self) -> str:
        """
        Generate .rou.xml with EV vehicle types and hourly flows.

        Uses BASt-based hourly traffic fractions for realistic patterns.
        """
        root = ET.Element("routes")

        # --- Vehicle types with battery device ---
        for ev in EV_TYPES:
            vtype = ET.SubElement(root, "vType",
                                 id=ev.type_id,
                                 vClass="passenger",
                                 length=f"{ev.length_m:.1f}",
                                 maxSpeed=f"{ev.max_speed_ms:.2f}",
                                 accel=f"{ev.accel:.1f}",
                                 decel=f"{ev.decel:.1f}",
                                 sigma="0.5")
            # Battery device parameters (capacity in Wh for SUMO)
            ET.SubElement(vtype, "param",
                          key="has.battery.device", value="true")
            ET.SubElement(vtype, "param",
                          key="device.battery.maximumBatteryCapacity",
                          value=f"{ev.battery_capacity_kwh * 1000:.0f}")
            mid_soc = (ev.initial_soc_low + ev.initial_soc_high) / 2.0
            ET.SubElement(vtype, "param",
                          key="device.battery.initialSOC",
                          value=f"{mid_soc:.2f}")
            # Stopping threshold: vehicle seeks charging when SOC below this
            ET.SubElement(vtype, "param",
                          key="device.battery.stoppingThreshold",
                          value="0.20")

        # --- Build the main highway route (through-traffic, no station stops) ---
        sorted_positions = sorted(enumerate(self.station_positions),
                                  key=lambda t: t[1])
        main_edges = []
        for i, (idx, _pos_km) in enumerate(sorted_positions):
            main_edges.append(f"highway_seg_{i}")
            main_edges.append(f"highway_through_{idx}")
        main_edges.append(f"highway_seg_{len(sorted_positions)}")

        route_elem = ET.SubElement(root, "route", id="highway_through",
                                   edges=" ".join(main_edges))

        # --- Build routes that detour through each station ---
        for station_order, (idx, _pos_km) in enumerate(sorted_positions):
            sid = f"station_{idx}"
            # Route up to the station, take off-ramp, charge, on-ramp, continue
            detour_edges = []
            for j in range(station_order + 1):
                jdx = sorted_positions[j][0]
                detour_edges.append(f"highway_seg_{j}")
                if j == station_order:
                    # Take the off-ramp at this station
                    detour_edges.append(f"offramp_{sid}")
                    detour_edges.append(f"connector_in_{sid}")
                    detour_edges.append(f"charging_{sid}")
                    detour_edges.append(f"connector_out_{sid}")
                    detour_edges.append(f"onramp_{sid}")
                else:
                    detour_edges.append(f"highway_through_{jdx}")

            # Continue past remaining stations on highway
            for j in range(station_order + 1, len(sorted_positions)):
                jdx = sorted_positions[j][0]
                # After our station's on-ramp merges at hw_after, we continue
                # on the next segment
                if j == station_order + 1:
                    # The on-ramp merges at hw_after which connects to next seg
                    pass
                detour_edges.append(f"highway_seg_{j}")
                detour_edges.append(f"highway_through_{jdx}")
            detour_edges.append(f"highway_seg_{len(sorted_positions)}")

            ET.SubElement(root, "route", id=f"route_via_{sid}",
                          edges=" ".join(detour_edges))

        # --- Hourly flows using BASt traffic fractions ---
        hourly_fracs = WEEKDAY_HOURLY_FRACTIONS  # Default weekday

        # Daily EV traffic = AADT × EV penetration
        daily_ev_total = self.base_aadt * self.ev_penetration

        flow_id = 0
        for hour in range(min(24, self.simulation_seconds // 3600)):
            fraction = hourly_fracs[hour]
            hourly_evs = daily_ev_total * fraction
            begin_s = hour * 3600
            end_s = (hour + 1) * 3600

            if hourly_evs < 0.5:
                continue

            # Split traffic: ~70% through-traffic (no stop), ~30% needs charging
            through_rate = hourly_evs * 0.70
            charging_rate = hourly_evs * 0.30

            # Through-traffic flow (distributed across EV types)
            for ev in EV_TYPES:
                type_rate = through_rate * ev.weight
                if type_rate < 0.1:
                    continue
                ET.SubElement(root, "flow",
                              id=f"through_{flow_id}",
                              type=ev.type_id,
                              route="highway_through",
                              begin=str(begin_s),
                              end=str(end_s),
                              vehsPerHour=f"{type_rate:.1f}",
                              departSpeed="max",
                              departLane="best")
                flow_id += 1

            # Charging-bound flows (distributed across stations and EV types)
            # Each flow includes a <stop> at the station's charging station
            # so vehicles actually park and charge rather than driving through.
            per_station_rate = charging_rate / max(1, self.n_stations)
            for station_order, (idx, _pos_km) in enumerate(sorted_positions):
                sid = f"station_{idx}"
                for ev in EV_TYPES:
                    type_rate = per_station_rate * ev.weight
                    if type_rate < 0.1:
                        continue
                    flow_elem = ET.SubElement(
                        root, "flow",
                        id=f"charging_{flow_id}",
                        type=ev.type_id,
                        route=f"route_via_{sid}",
                        begin=str(begin_s),
                        end=str(end_s),
                        vehsPerHour=f"{type_rate:.1f}",
                        departSpeed="max",
                        departLane="best",
                    )
                    # Stop at the charging station for 20-40 min (1200-2400s)
                    ET.SubElement(
                        flow_elem, "stop",
                        chargingStation=f"cs_{sid}",
                        duration="1800",
                        parking="true",
                    )
                    flow_id += 1

        path = os.path.join(self.output_dir, "highway.rou.xml")
        self._write_xml(root, path)
        return path

    # ------------------------------------------------------------------
    # Additional (parking areas, charging stations, rerouters)
    # ------------------------------------------------------------------

    def _generate_additional(self) -> str:
        """Generate .add.xml with parking areas, charging stations, and rerouters."""
        root = ET.Element("additional")

        sorted_positions = sorted(enumerate(self.station_positions),
                                  key=lambda t: t[1])

        for _order, (idx, _pos_km) in enumerate(sorted_positions):
            sid = f"station_{idx}"
            charging_edge = f"charging_{sid}"
            total_capacity = self.n_chargers[idx] + self.n_waiting[idx]

            # Parking area on the charging edge (500m long edge)
            ET.SubElement(root, "parkingArea",
                          id=f"parking_{sid}",
                          lane=f"{charging_edge}_0",
                          startPos="10",
                          endPos="480",
                          roadsideCapacity=str(total_capacity),
                          friendlyPos="true")

            # Charging station overlaid on same edge
            # Power per charger: 150 kW = 150000 W
            charger_power_w = 150000.0 * self.n_chargers[idx]
            ET.SubElement(root, "chargingStation",
                          id=f"cs_{sid}",
                          lane=f"{charging_edge}_0",
                          startPos="10",
                          endPos="480",
                          chargingPower=f"{charger_power_w:.0f}",
                          efficiency="0.90",
                          friendlyPos="true")

            # Parking rerouter on the off-ramp so vehicles stop at the
            # parking area.  Vehicles on the charging route will be
            # rerouted into the parking area when they reach the off-ramp.
            rerouter = ET.SubElement(root, "rerouter",
                                    id=f"rerouter_{sid}",
                                    edges=f"offramp_{sid}",
                                    pos=f"{_pos_km * 1000 - 20:.1f},-25.0")
            interval = ET.SubElement(rerouter, "interval",
                                     begin="0",
                                     end=str(self.simulation_seconds))
            ET.SubElement(interval, "parkingAreaReroute",
                          id=f"parking_{sid}")

        path = os.path.join(self.output_dir, "highway.add.xml")
        self._write_xml(root, path)
        return path

    # ------------------------------------------------------------------
    # SUMO configuration
    # ------------------------------------------------------------------

    def _generate_sumocfg(self, net_path: str, rou_path: str,
                          add_path: str) -> str:
        """Generate .sumocfg tying all files together."""
        root = ET.Element("configuration")

        inp = ET.SubElement(root, "input")
        ET.SubElement(inp, "net-file",
                      value=os.path.basename(net_path))
        ET.SubElement(inp, "route-files",
                      value=os.path.basename(rou_path))
        ET.SubElement(inp, "additional-files",
                      value=os.path.basename(add_path))

        time_elem = ET.SubElement(root, "time")
        ET.SubElement(time_elem, "begin", value="0")
        ET.SubElement(time_elem, "end", value=str(self.simulation_seconds))
        ET.SubElement(time_elem, "step-length", value="1.0")

        processing = ET.SubElement(root, "processing")
        ET.SubElement(processing, "device.battery.probability", value="1.0")

        report = ET.SubElement(root, "report")
        ET.SubElement(report, "no-step-log", value="true")
        ET.SubElement(report, "no-warnings", value="true")

        rand = ET.SubElement(root, "random_number")
        ET.SubElement(rand, "seed", value=str(self.seed))

        path = os.path.join(self.output_dir, "highway.sumocfg")
        self._write_xml(root, path)
        return path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_xml(root: ET.Element, path: str) -> None:
        """Write an ElementTree to file with XML declaration."""
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        with open(path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
