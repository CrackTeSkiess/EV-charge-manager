"""
Highway Selector — main user-facing class for the Highway Selection Module.

Typical usage::

    from ev_charge_manager.highway_selection import HighwaySelector

    selector = HighwaySelector()
    result = selector.select("A5", start_km=0, end_km=300, show_map=True)
    print(result.summary())

    # Pre-filled by real OSM data:
    params = result.simulation_params
    # params.highway_length_km  → 300.0
    # params.num_charging_areas → 8
    # params.charging_area_positions → [12.3, 45.1, …]
    # params.charging_area_names     → ["Rasthof Gräfenhausen", …]

    # Tune freely:
    params.chargers_per_area = 6
    params.vehicles_per_hour = 80.0

    from ev_charge_manager.simulation import Simulation
    Simulation(params).run()
"""

from __future__ import annotations

import math
import tempfile
import warnings
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ev_charge_manager.simulation.simulation import SimulationParameters

from .geo_utils import (
    LatLon,
    build_polyline_km_index,
    filter_nodes_by_km_range,
    haversine_km,
    project_point_to_polyline,
)
from .osm_client import HighwayGeometry, OverpassClient, ServiceArea


# ---------------------------------------------------------------------------
# Highway registry
# ---------------------------------------------------------------------------

# Pre-registered highways with approximate bounding boxes (south, west, north, east)
# and approximate total length in km. Used to resolve highway names to OSM queries
# without requiring the user to specify coordinates.
HIGHWAY_REGISTRY: Dict[str, Dict] = {
    # --- German Autobahn ---
    "A1":    {"country": "DE", "type": "motorway", "bbox": (47.5,  6.7, 54.0, 10.5), "approx_km": 750},
    "A2":    {"country": "DE", "type": "motorway", "bbox": (51.4,  7.0, 52.6, 14.5), "approx_km": 502},
    "A3":    {"country": "DE", "type": "motorway", "bbox": (47.8,  7.5, 51.5, 13.0), "approx_km": 760},
    "A4":    {"country": "DE", "type": "motorway", "bbox": (50.7,  6.0, 51.2, 15.1), "approx_km": 492},
    "A5":    {"country": "DE", "type": "motorway", "bbox": (47.5,  7.5, 50.2,  9.5), "approx_km": 445},
    "A6-DE": {"country": "DE", "type": "motorway", "bbox": (49.2,  7.7, 49.6, 13.0), "approx_km": 447},
    "A7":    {"country": "DE", "type": "motorway", "bbox": (47.6,  9.0, 55.0, 10.2), "approx_km": 963},
    "A8":    {"country": "DE", "type": "motorway", "bbox": (47.5,  8.0, 48.9, 12.9), "approx_km": 393},
    "A9":    {"country": "DE", "type": "motorway", "bbox": (48.3, 11.3, 51.6, 12.5), "approx_km": 529},
    # --- UK Motorways ---
    "M1":    {"country": "GB", "type": "motorway", "bbox": (51.5, -1.4, 53.8,  0.2), "approx_km": 311},
    "M6":    {"country": "GB", "type": "motorway", "bbox": (52.4, -2.7, 55.0, -1.5), "approx_km": 373},
    "M25":   {"country": "GB", "type": "motorway", "bbox": (51.3, -0.6, 51.8,  0.4), "approx_km": 188},
    # --- French Autoroutes (use select_custom for the OSM ref, e.g. "A6") ---
    "A6-FR": {"country": "FR", "type": "motorway", "bbox": (43.3,  4.8, 48.9,  5.4), "approx_km": 450},
    "A7-FR": {"country": "FR", "type": "motorway", "bbox": (43.3,  4.4, 45.8,  5.0), "approx_km": 301},
    # --- US Interstates ---
    "I-95":  {"country": "US", "type": "motorway", "bbox": (25.8, -80.2, 47.5, -67.0), "approx_km": 3118},
    "I-80":  {"country": "US", "type": "motorway", "bbox": (37.7, -122.4, 41.2, -75.0), "approx_km": 4666},
    "I-10":  {"country": "US", "type": "motorway", "bbox": (29.7, -106.5, 34.0, -79.2), "approx_km": 3957},
    "I-90":  {"country": "US", "type": "motorway", "bbox": (41.8, -122.4, 47.6, -71.0), "approx_km": 4862},
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StopAreaInfo:
    """A processed service area with its km position along the selected segment."""
    name: str
    km_position: float      # Distance from segment start in km (0 = segment start)
    lat: float
    lon: float
    osm_id: Optional[int] = None
    is_synthetic: bool = False  # True if generated as fallback (not from OSM)


@dataclass
class HighwaySelectionResult:
    """
    Complete result of a highway selection operation.

    The ``simulation_params`` field contains a ``SimulationParameters`` instance
    with three fields pre-filled from real-world data:

    * ``highway_length_km``
    * ``num_charging_areas``
    * ``charging_area_positions``
    * ``charging_area_names``

    All other ``SimulationParameters`` fields are left at their defaults so the
    user can tune them freely before passing to ``Simulation()``.
    """
    # Identification
    highway_ref: str
    country: str

    # Segment geometry
    segment_start_km: float
    segment_end_km: float
    segment_length_km: float

    # Stop areas along the segment, sorted by km_position ascending
    stop_areas: List[StopAreaInfo]

    # Pre-configured simulation parameters
    simulation_params: SimulationParameters

    # Quality metadata
    data_source: str            # "osm" | "osm_expanded" | "synthetic"
    total_osm_areas_found: int  # Total areas found in the bounding box (before km filter)
    areas_in_segment: int       # Areas after km filtering and deduplication

    def summary(self) -> str:
        """Return a human-readable summary of the selection result."""
        lines = [
            f"Highway {self.highway_ref} ({self.country})",
            f"  Segment : {self.segment_start_km:.1f} – {self.segment_end_km:.1f} km "
            f"({self.segment_length_km:.1f} km total)",
            f"  Source  : {self.data_source}",
            f"  Areas   : {self.areas_in_segment} stop areas "
            f"(of {self.total_osm_areas_found} found in bounding box)",
            "",
        ]
        for area in self.stop_areas:
            marker = " [synthetic]" if area.is_synthetic else ""
            lines.append(f"    km {area.km_position:6.1f}  |  {area.name}{marker}")
        lines.append("")
        lines.append(
            f"  SimulationParameters ready: highway_length_km={self.simulation_params.highway_length_km}, "
            f"num_charging_areas={self.simulation_params.num_charging_areas}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main selector class
# ---------------------------------------------------------------------------

class HighwaySelector:
    """
    Select a real highway segment and pre-configure SimulationParameters
    with real-world service area positions from OpenStreetMap.

    Args:
        cache_dir: Directory for OSM API response cache files.
        overpass_timeout_sec: HTTP timeout for Overpass API requests.
        use_cache: If False, always fetch fresh data from the API.
    """

    MIN_STOP_AREAS = 1          # Trigger fallback if fewer areas found after filtering
    MAX_BBOX_EXPANSION = 0.20   # 20% bbox expansion on retry

    def __init__(
        self,
        cache_dir: str = "data/cache/osm",
        overpass_timeout_sec: int = 90,
        use_cache: bool = True,
    ) -> None:
        self.osm_client = OverpassClient(
            cache_dir=cache_dir,
            timeout_sec=overpass_timeout_sec,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def list_available_highways(cls) -> List[Dict]:
        """
        Return the list of pre-registered highways with their metadata.

        Each entry is a dict with keys: ref, country, type, approx_km, bbox.
        """
        return [{"ref": ref, **info} for ref, info in HIGHWAY_REGISTRY.items()]

    def select(
        self,
        highway_ref: str,
        start_km: float = 0.0,
        end_km: Optional[float] = None,
        show_map: bool = False,
        map_output_path: Optional[str] = None,
        min_area_spacing_km: float = 10.0,
    ) -> HighwaySelectionResult:
        """
        Select a highway segment and fetch real stop areas from OpenStreetMap.

        Args:
            highway_ref: Highway reference string, e.g. "A5", "M1", "I-95".
                         Must be a key in HIGHWAY_REGISTRY.  For custom highways
                         use ``select_custom()`` instead.
            start_km: Start of the desired segment in km from the highway origin.
            end_km: End of the desired segment in km.  Defaults to the full highway
                    length (from HIGHWAY_REGISTRY approx_km).
            show_map: If True, render an interactive map and open it in a browser.
            map_output_path: Optional path for the HTML map file.  If None and
                             ``show_map`` is True, a temporary file is used.
            min_area_spacing_km: Minimum km between included stop areas; closer
                                 areas are deduplicated (longer name is kept).

        Returns:
            HighwaySelectionResult with pre-filled SimulationParameters.

        Raises:
            ValueError: If highway_ref is not in HIGHWAY_REGISTRY.
            ValueError: If start_km >= end_km.
        """
        ref_upper = highway_ref.upper()
        if ref_upper not in HIGHWAY_REGISTRY:
            # Try case-insensitive match
            match = next(
                (k for k in HIGHWAY_REGISTRY if k.upper() == ref_upper), None
            )
            if match is None:
                available = ", ".join(sorted(HIGHWAY_REGISTRY.keys()))
                raise ValueError(
                    f"Highway '{highway_ref}' is not in the built-in registry. "
                    f"Available: {available}. "
                    f"For custom highways, use select_custom()."
                )
            highway_ref = match
        else:
            highway_ref = ref_upper

        info = HIGHWAY_REGISTRY[highway_ref]
        return self._fetch_and_process(
            highway_ref=highway_ref,
            bounding_box=info["bbox"],
            highway_type=info["type"],
            start_km=start_km,
            end_km=end_km,
            min_area_spacing_km=min_area_spacing_km,
            country=info["country"],
            approx_total_km=info["approx_km"],
            show_map=show_map,
            map_output_path=map_output_path,
        )

    def select_custom(
        self,
        highway_ref: str,
        bounding_box: Tuple[float, float, float, float],
        highway_type: str = "motorway",
        start_km: float = 0.0,
        end_km: Optional[float] = None,
        show_map: bool = False,
        map_output_path: Optional[str] = None,
        min_area_spacing_km: float = 10.0,
        country: str = "unknown",
    ) -> HighwaySelectionResult:
        """
        Select a highway segment for a highway not in the built-in registry.

        Args:
            highway_ref: Highway reference string, e.g. "A96".
            bounding_box: (south, west, north, east) in decimal degrees.
            highway_type: OSM highway tag value (default "motorway").
            start_km: Start of segment in km.
            end_km: End of segment in km.  Defaults to full geometry length.
            show_map: Render an interactive map.
            map_output_path: Path for the HTML map file.
            min_area_spacing_km: Minimum spacing between stop areas in km.
            country: Country code string (informational only).

        Returns:
            HighwaySelectionResult with pre-filled SimulationParameters.
        """
        return self._fetch_and_process(
            highway_ref=highway_ref,
            bounding_box=bounding_box,
            highway_type=highway_type,
            start_km=start_km,
            end_km=end_km,
            min_area_spacing_km=min_area_spacing_km,
            country=country,
            approx_total_km=None,
            show_map=show_map,
            map_output_path=map_output_path,
        )

    # ------------------------------------------------------------------
    # Core processing pipeline
    # ------------------------------------------------------------------

    def _fetch_and_process(
        self,
        highway_ref: str,
        bounding_box: Tuple[float, float, float, float],
        highway_type: str,
        start_km: float,
        end_km: Optional[float],
        min_area_spacing_km: float,
        country: str,
        approx_total_km: Optional[float],
        show_map: bool,
        map_output_path: Optional[str],
    ) -> HighwaySelectionResult:
        """
        Internal implementation shared by select() and select_custom().

        Steps:
          1. Fetch highway geometry via Overpass API
          2. Build cumulative km index
          3. Determine segment bounds
          4. Fetch service areas via Overpass API
          5. Project each area onto the polyline → km position
          6. Filter to [start_km, end_km], re-zero relative to start
          7. Deduplicate areas closer than min_area_spacing_km
          8. Apply fallback if insufficient areas found
          9. Build SimulationParameters (only highway geometry fields set)
         10. Build and return HighwaySelectionResult
         11. Optionally render map
        """
        if start_km < 0:
            start_km = 0.0

        # --- Step 1: Fetch highway geometry ---
        geometry = self.osm_client.fetch_highway_geometry(
            highway_ref, bounding_box, highway_type
        )

        # Determine effective end_km
        if geometry is not None:
            total_highway_km = geometry.total_length_km
        elif approx_total_km is not None:
            total_highway_km = float(approx_total_km)
        else:
            total_highway_km = 300.0  # Absolute fallback
            warnings.warn(
                "Could not determine highway length; defaulting to 300 km.",
                RuntimeWarning,
            )

        if end_km is None:
            end_km = total_highway_km
        end_km = min(end_km, total_highway_km)

        if start_km >= end_km:
            raise ValueError(
                f"start_km ({start_km}) must be less than end_km ({end_km})."
            )

        segment_length_km = end_km - start_km

        # --- Step 2: Fetch service areas ---
        raw_areas = self.osm_client.fetch_service_areas(bounding_box)
        total_osm_found = len(raw_areas)

        # --- Step 3: Project and filter service areas ---
        stop_areas: List[StopAreaInfo] = []
        data_source = "osm"

        if geometry is not None and geometry.nodes:
            cumulative_km, _ = build_polyline_km_index(geometry.nodes)

            for area in raw_areas:
                raw_km = project_point_to_polyline(
                    (area.lat, area.lon), geometry.nodes, cumulative_km
                )
                # Convert to segment-relative position
                segment_km = raw_km - start_km
                if 0.0 <= segment_km <= segment_length_km:
                    stop_areas.append(StopAreaInfo(
                        name=area.name,
                        km_position=round(segment_km, 2),
                        lat=area.lat,
                        lon=area.lon,
                        osm_id=area.osm_id,
                        is_synthetic=False,
                    ))

            # Sort by km position
            stop_areas.sort(key=lambda a: a.km_position)

            # --- Step 4: Deduplicate ---
            stop_areas = self._deduplicate_areas(stop_areas, min_area_spacing_km)
        else:
            # No geometry available — try direct distance projection on raw areas
            if raw_areas:
                stop_areas, data_source = self._project_areas_without_geometry(
                    raw_areas, bounding_box, start_km, segment_length_km
                )
            else:
                data_source = "osm"

        # --- Step 5: Fallback if too few areas found ---
        if len(stop_areas) < self.MIN_STOP_AREAS:
            # Expand bounding box by 20% and retry once
            if data_source == "osm" and total_osm_found == 0:
                expanded_bbox = self._expand_bbox(bounding_box, self.MAX_BBOX_EXPANSION)
                raw_areas_retry = self.osm_client.fetch_service_areas(expanded_bbox)
                if raw_areas_retry and geometry is not None and geometry.nodes:
                    cumulative_km, _ = build_polyline_km_index(geometry.nodes)
                    for area in raw_areas_retry:
                        raw_km = project_point_to_polyline(
                            (area.lat, area.lon), geometry.nodes, cumulative_km
                        )
                        segment_km = raw_km - start_km
                        if 0.0 <= segment_km <= segment_length_km:
                            stop_areas.append(StopAreaInfo(
                                name=area.name,
                                km_position=round(segment_km, 2),
                                lat=area.lat,
                                lon=area.lon,
                                osm_id=area.osm_id,
                                is_synthetic=False,
                            ))
                    stop_areas.sort(key=lambda a: a.km_position)
                    stop_areas = self._deduplicate_areas(stop_areas, min_area_spacing_km)
                    data_source = "osm_expanded"

            # If still not enough, use synthetic fallback
            if len(stop_areas) < self.MIN_STOP_AREAS:
                warnings.warn(
                    f"No real OSM stop areas found for {highway_ref} in the selected "
                    f"segment ({start_km:.0f}–{end_km:.0f} km). "
                    "Generating synthetic evenly-spaced areas as fallback. "
                    "You can still tune all charging parameters freely.",
                    RuntimeWarning,
                )
                stop_areas = self._apply_fallback_areas(segment_length_km)
                data_source = "synthetic"

        areas_in_segment = len(stop_areas)

        # --- Step 6: Build SimulationParameters ---
        sim_params = self._build_simulation_params(segment_length_km, stop_areas)

        # --- Step 7: Assemble result ---
        result = HighwaySelectionResult(
            highway_ref=highway_ref,
            country=country,
            segment_start_km=start_km,
            segment_end_km=end_km,
            segment_length_km=segment_length_km,
            stop_areas=stop_areas,
            simulation_params=sim_params,
            data_source=data_source,
            total_osm_areas_found=total_osm_found,
            areas_in_segment=areas_in_segment,
        )

        # --- Step 8: Optional map rendering ---
        if show_map:
            highway_nodes = geometry.nodes if geometry is not None else []
            self._render_map(
                result=result,
                highway_nodes=highway_nodes,
                start_km=start_km,
                end_km=end_km,
                output_path=map_output_path,
                open_browser=True,
            )

        return result

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _project_areas_without_geometry(
        self,
        raw_areas: List[ServiceArea],
        bounding_box: Tuple[float, float, float, float],
        start_km: float,
        segment_length_km: float,
    ) -> Tuple[List[StopAreaInfo], str]:
        """
        Approximate km positions when no highway geometry is available.

        Projects areas by computing their fractional position along the
        bounding box's major axis (lat or lon span), then scaling to km.
        This is a rough approximation — geometry-based projection is preferred.
        """
        south, west, north, east = bounding_box
        lat_span = north - south
        lon_span = east - west

        # Determine dominant axis
        # Use haversine to convert spans to km for comparison
        center_lat = (south + north) / 2
        lat_km = haversine_km((south, west), (north, west))
        lon_km = haversine_km((center_lat, west), (center_lat, east))

        stop_areas: List[StopAreaInfo] = []
        for area in raw_areas:
            if lon_km >= lat_km:
                # E-W highway: project on longitude axis
                frac = (area.lon - west) / lon_span if lon_span > 0 else 0.5
                approx_km = frac * (lat_km + lon_km) / 2
            else:
                # N-S highway: project on latitude axis
                frac = (area.lat - south) / lat_span if lat_span > 0 else 0.5
                approx_km = frac * lat_km

            segment_km = approx_km - start_km
            if 0.0 <= segment_km <= segment_length_km:
                stop_areas.append(StopAreaInfo(
                    name=area.name,
                    km_position=round(segment_km, 2),
                    lat=area.lat,
                    lon=area.lon,
                    osm_id=area.osm_id,
                    is_synthetic=False,
                ))

        stop_areas.sort(key=lambda a: a.km_position)
        return stop_areas, "osm"

    def _deduplicate_areas(
        self,
        areas: List[StopAreaInfo],
        min_spacing_km: float,
    ) -> List[StopAreaInfo]:
        """
        Remove duplicate / very-close service areas.

        For each cluster of areas within min_spacing_km of each other,
        keeps the entry with the longest name (usually most informative).
        Input must be sorted by km_position.
        """
        if not areas:
            return []

        result: List[StopAreaInfo] = [areas[0]]
        for area in areas[1:]:
            prev = result[-1]
            if (area.km_position - prev.km_position) < min_spacing_km:
                # Keep the one with the longer name
                if len(area.name) > len(prev.name):
                    result[-1] = area
            else:
                result.append(area)

        return result

    def _apply_fallback_areas(
        self,
        segment_length_km: float,
        n_areas: int = 3,
    ) -> List[StopAreaInfo]:
        """
        Generate synthetic evenly-spaced stop areas as fallback.

        Mirrors the auto-spacing logic in Environment._initialize_highway().
        Names follow: "Service Area 1", "Service Area 2", etc.
        """
        if n_areas <= 0 or segment_length_km <= 0:
            return []

        # Cap to one station per 100 km minimum spacing to be realistic
        max_areas = max(1, int(segment_length_km / 80.0))
        n = min(n_areas, max_areas)

        spacing = segment_length_km / (n + 1)
        areas = []
        for i in range(n):
            km_pos = spacing * (i + 1)
            areas.append(StopAreaInfo(
                name=f"Service Area {i + 1}",
                km_position=round(km_pos, 2),
                lat=0.0,
                lon=0.0,
                osm_id=None,
                is_synthetic=True,
            ))
        return areas

    def _build_simulation_params(
        self,
        segment_length_km: float,
        stop_areas: List[StopAreaInfo],
    ) -> SimulationParameters:
        """
        Build a SimulationParameters instance with the highway geometry fields
        pre-filled from real (or synthetic fallback) data.

        Only these three fields are set here:
            highway_length_km, num_charging_areas, charging_area_positions,
            charging_area_names

        All other fields keep their defaults so the user can tune them freely.
        """
        positions = [a.km_position for a in stop_areas]
        names = [a.name for a in stop_areas]

        return SimulationParameters(
            highway_length_km=round(segment_length_km, 2),
            num_charging_areas=len(positions),
            charging_area_positions=positions if positions else None,
            charging_area_names=names if names else None,
        )

    @staticmethod
    def _expand_bbox(
        bbox: Tuple[float, float, float, float],
        fraction: float,
    ) -> Tuple[float, float, float, float]:
        """Expand a bounding box by fraction on each side."""
        south, west, north, east = bbox
        lat_delta = (north - south) * fraction
        lon_delta = (east - west) * fraction
        return (
            south - lat_delta,
            west - lon_delta,
            north + lat_delta,
            east + lon_delta,
        )

    # ------------------------------------------------------------------
    # Map rendering
    # ------------------------------------------------------------------

    def _render_map(
        self,
        result: HighwaySelectionResult,
        highway_nodes: List[LatLon],
        start_km: float,
        end_km: float,
        output_path: Optional[str],
        open_browser: bool,
    ) -> Optional[str]:
        """
        Render an interactive map.  Tries folium first, falls back to matplotlib.
        """
        try:
            return self.render_map_folium(
                result=result,
                highway_nodes=highway_nodes,
                output_path=output_path,
                open_browser=open_browser,
            )
        except ImportError:
            warnings.warn(
                "folium is not installed — falling back to matplotlib map. "
                "Install folium for interactive HTML maps: pip install folium",
                RuntimeWarning,
            )
            return self.render_map_matplotlib(
                result=result,
                highway_nodes=highway_nodes,
                output_path=output_path,
                show=open_browser,
            )

    def render_map_folium(
        self,
        result: HighwaySelectionResult,
        highway_nodes: List[LatLon],
        output_path: Optional[str] = None,
        open_browser: bool = True,
    ) -> str:
        """
        Render an interactive folium HTML map.

        The map shows:
        - The full highway polyline in blue
        - The selected segment highlighted in green
        - Red circle markers for each stop area with name and km tooltips

        Args:
            result: HighwaySelectionResult from select().
            highway_nodes: Full ordered list of highway nodes (lat, lon).
            output_path: Path to save the HTML file.  Uses a temp file if None.
            open_browser: If True, open the map in the default browser.

        Returns:
            Absolute path of the saved HTML file.

        Raises:
            ImportError: If folium is not installed.
        """
        import folium  # Intentional late import — optional dependency

        # Determine map centre
        if highway_nodes:
            center_lat = sum(n[0] for n in highway_nodes) / len(highway_nodes)
            center_lon = sum(n[1] for n in highway_nodes) / len(highway_nodes)
        elif result.stop_areas:
            center_lat = sum(a.lat for a in result.stop_areas) / len(result.stop_areas)
            center_lon = sum(a.lon for a in result.stop_areas) / len(result.stop_areas)
        else:
            center_lat, center_lon = 51.0, 10.0  # Rough centre of Europe

        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

        # Full highway polyline (blue, semi-transparent)
        if len(highway_nodes) >= 2:
            folium.PolyLine(
                locations=[[n[0], n[1]] for n in highway_nodes],
                color="#3388ff",
                weight=3,
                opacity=0.5,
                tooltip=f"Highway {result.highway_ref}",
            ).add_to(m)

        # Selected segment (green, bold)
        from ev_charge_manager.highway_selection.geo_utils import (
            build_polyline_km_index,
            filter_nodes_by_km_range,
        )
        if len(highway_nodes) >= 2:
            cum_km, _ = build_polyline_km_index(highway_nodes)
            seg_nodes, _ = filter_nodes_by_km_range(
                highway_nodes, cum_km, result.segment_start_km, result.segment_end_km
            )
            if len(seg_nodes) >= 2:
                folium.PolyLine(
                    locations=[[n[0], n[1]] for n in seg_nodes],
                    color="#22aa44",
                    weight=5,
                    opacity=0.85,
                    tooltip=(
                        f"Selected segment: {result.segment_start_km:.0f}–"
                        f"{result.segment_end_km:.0f} km"
                    ),
                ).add_to(m)

        # Stop area markers
        for area in result.stop_areas:
            if area.lat == 0.0 and area.lon == 0.0:
                continue  # synthetic areas have no coordinates
            popup_html = (
                f"<b>{area.name}</b><br>"
                f"km {area.km_position:.1f}<br>"
                f"{'(synthetic)' if area.is_synthetic else 'OSM'}"
            )
            folium.CircleMarker(
                location=[area.lat, area.lon],
                radius=8,
                color="#cc2200",
                fill=True,
                fill_color="#ff4422",
                fill_opacity=0.85,
                tooltip=f"{area.name} — km {area.km_position:.1f}",
                popup=folium.Popup(popup_html, max_width=200),
            ).add_to(m)

        # Title overlay
        title_html = (
            f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
            f'background:white;padding:6px 14px;border-radius:4px;'
            f'box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:14px;font-weight:bold;'
            f'z-index:9999;">'
            f'Highway {result.highway_ref} — {result.segment_length_km:.0f} km segment — '
            f'{result.areas_in_segment} stop areas ({result.data_source})'
            f'</div>'
        )
        m.get_root().html.add_child(folium.Element(title_html))

        # Save to file
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, prefix=f"highway_{result.highway_ref}_"
            )
            output_path = tmp.name
            tmp.close()

        output_path = str(Path(output_path).resolve())
        m.save(output_path)

        if open_browser:
            webbrowser.open(f"file://{output_path}")

        return output_path

    def render_map_matplotlib(
        self,
        result: HighwaySelectionResult,
        highway_nodes: List[LatLon],
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[str]:
        """
        Render a static matplotlib map as a fallback when folium is unavailable.

        The map shows longitude on the x-axis and latitude on the y-axis, with:
        - The highway corridor as a thin blue line
        - The selected segment highlighted in green
        - Red dots for stop areas with name annotations

        Args:
            result: HighwaySelectionResult from select().
            highway_nodes: Full ordered list of highway nodes (lat, lon).
            output_path: Optional path to save a PNG file.
            show: If True, display the plot interactively.

        Returns:
            Absolute path of the saved PNG file, or None if not saved.
        """
        import matplotlib.pyplot as plt
        from ev_charge_manager.highway_selection.geo_utils import (
            build_polyline_km_index,
            filter_nodes_by_km_range,
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        # Full highway
        if len(highway_nodes) >= 2:
            lons = [n[1] for n in highway_nodes]
            lats = [n[0] for n in highway_nodes]
            ax.plot(lons, lats, color="#3388ff", linewidth=1.5, alpha=0.4,
                    label=f"Highway {result.highway_ref}")

        # Selected segment
        if len(highway_nodes) >= 2:
            cum_km, _ = build_polyline_km_index(highway_nodes)
            seg_nodes, _ = filter_nodes_by_km_range(
                highway_nodes, cum_km, result.segment_start_km, result.segment_end_km
            )
            if len(seg_nodes) >= 2:
                seg_lons = [n[1] for n in seg_nodes]
                seg_lats = [n[0] for n in seg_nodes]
                ax.plot(seg_lons, seg_lats, color="#22aa44", linewidth=3,
                        label=f"Selected segment ({result.segment_length_km:.0f} km)")

        # Stop areas
        for area in result.stop_areas:
            if area.lat == 0.0 and area.lon == 0.0:
                continue
            ax.scatter(area.lon, area.lat, s=60, color="#cc2200", zorder=5)
            ax.annotate(
                f"{area.name}\nkm {area.km_position:.1f}",
                xy=(area.lon, area.lat),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#880000",
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"Highway {result.highway_ref} — {result.segment_length_km:.0f} km segment — "
            f"{result.areas_in_segment} stop areas ({result.data_source})"
        )
        ax.legend(loc="best", fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        plt.tight_layout()

        saved_path: Optional[str] = None
        if output_path:
            saved_path = str(Path(output_path).resolve())
            fig.savefig(saved_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return saved_path
