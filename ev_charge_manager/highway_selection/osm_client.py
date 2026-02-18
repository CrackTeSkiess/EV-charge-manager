"""
OpenStreetMap Overpass API client for the Highway Selection Module.

Fetches real highway geometry and motorway service area locations from OSM.
All responses are cached to disk as JSON files (same pattern as pvgis_client.py).
No API key is required — Overpass is a free public service.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .geo_utils import LatLon, sort_nodes_by_direction, build_polyline_km_index

# ---------------------------------------------------------------------------
# Overpass API endpoints (primary + fallback mirror)
# ---------------------------------------------------------------------------
OVERPASS_PRIMARY_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACK_URL = "https://lz4.overpass-api.de/api/interpreter"

_USER_AGENT = "EV-charge-manager/1.0 (https://github.com/EV-charge-manager)"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServiceArea:
    """A real-world motorway service area or rest stop from OpenStreetMap."""
    osm_id: int
    name: str                          # OSM 'name' tag, e.g. "Rasthof Garbsen"
    lat: float
    lon: float
    osm_type: str                      # "node", "way", or "relation"
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Provide a readable fallback name when OSM has none
        if not self.name:
            self.name = f"Service Area ({self.osm_type}/{self.osm_id})"


@dataclass
class HighwayGeometry:
    """Raw geometry of a highway fetched from OpenStreetMap."""
    ref: str                           # e.g. "A5", "M1", "I-95"
    nodes: List[LatLon]               # Directionally-ordered (lat, lon) waypoints
    total_length_km: float
    bounding_box: Tuple[float, float, float, float]   # (south, west, north, east)
    osm_way_ids: List[int]            # OSM way element IDs that form this highway


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OverpassClient:
    """
    Client for the OpenStreetMap Overpass API.

    Fetches highway geometry and motorway service area locations.
    All results are cached to disk as JSON files keyed by a SHA-256 hash
    of the query string.  On a cache miss the primary Overpass endpoint
    is tried first, then the LZ4 mirror on timeout or server error.

    Args:
        cache_dir: Directory for JSON cache files.
        timeout_sec: HTTP request timeout in seconds.
        use_cache: If False, always fetch fresh data (ignores cache on read,
                   but still writes results to cache).
    """

    DEFAULT_CACHE_DIR = "data/cache/osm"
    DEFAULT_TIMEOUT_SEC = 90

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        use_cache: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_sec = timeout_sec
        self.use_cache = use_cache
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            warnings.warn(
                f"OSM cache directory '{cache_dir}' could not be created: {exc}. "
                "Caching disabled for this session.",
                RuntimeWarning,
            )
            self.use_cache = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_highway_geometry(
        self,
        highway_ref: str,
        bounding_box: Tuple[float, float, float, float],
        highway_type: str = "motorway",
    ) -> Optional[HighwayGeometry]:
        """
        Fetch the ordered lat/lon nodes of a named highway from Overpass.

        Args:
            highway_ref: Highway reference string, e.g. "A5", "M1", "I-95".
            bounding_box: (south, west, north, east) in decimal degrees.
            highway_type: OSM highway tag value (default "motorway").

        Returns:
            HighwayGeometry if successful, None on API failure.
        """
        query = self._build_geometry_query(highway_ref, bounding_box, highway_type)
        cache_key = f"geo_{self._hash(query)}"
        data = self._execute_query(query, cache_key)
        if data is None:
            return None
        return self._parse_geometry_response(data, highway_ref, bounding_box)

    def fetch_service_areas(
        self,
        bounding_box: Tuple[float, float, float, float],
    ) -> List[ServiceArea]:
        """
        Fetch motorway service areas / rest stops within the bounding box.

        Queries for both European-style 'highway=services' and
        US-style 'highway=rest_area' nodes and ways.

        Args:
            bounding_box: (south, west, north, east) in decimal degrees.

        Returns:
            List of ServiceArea objects, may be empty on API failure or if
            no stop areas exist in the region.
        """
        query = self._build_services_query(bounding_box)
        cache_key = f"svc_{self._hash(query)}"
        data = self._execute_query(query, cache_key)
        if data is None:
            return []
        return self._parse_services_response(data)

    # ------------------------------------------------------------------
    # Query builders
    # ------------------------------------------------------------------

    def _build_geometry_query(
        self,
        highway_ref: str,
        bounding_box: Tuple[float, float, float, float],
        highway_type: str,
    ) -> str:
        south, west, north, east = bounding_box
        return (
            f'[out:json][timeout:{self.timeout_sec}]'
            f'[bbox:{south},{west},{north},{east}];\n'
            f'(\n'
            f'  way["highway"="{highway_type}"]["ref"="{highway_ref}"];\n'
            f');\n'
            f'(._;>;);\n'
            f'out body;'
        )

    def _build_services_query(
        self,
        bounding_box: Tuple[float, float, float, float],
    ) -> str:
        south, west, north, east = bounding_box
        timeout = min(self.timeout_sec, 60)
        return (
            f'[out:json][timeout:{timeout}]'
            f'[bbox:{south},{west},{north},{east}];\n'
            f'(\n'
            f'  node["highway"="services"];\n'
            f'  way["highway"="services"];\n'
            f'  node["highway"="rest_area"];\n'
            f'  way["highway"="rest_area"];\n'
            f'  node["amenity"="fuel"]["name"]["highway"~"motorway|trunk"];\n'
            f');\n'
            f'out center;'
        )

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    def _parse_geometry_response(
        self,
        data: Dict[str, Any],
        highway_ref: str,
        bounding_box: Tuple[float, float, float, float],
    ) -> Optional[HighwayGeometry]:
        """
        Parse Overpass JSON into HighwayGeometry.

        Extracts all node coordinates and stitches way segments into a single
        directional polyline via sort_nodes_by_direction().
        """
        elements = data.get("elements", [])
        if not elements:
            warnings.warn(
                f"Overpass returned no elements for highway '{highway_ref}'. "
                "Check that the highway ref and bounding box are correct.",
                RuntimeWarning,
            )
            return None

        # Build a node_id → (lat, lon) lookup from all node elements
        node_coords: Dict[int, LatLon] = {}
        way_elements = []

        for elem in elements:
            if elem.get("type") == "node":
                node_coords[elem["id"]] = (elem["lat"], elem["lon"])
            elif elem.get("type") == "way":
                way_elements.append(elem)

        if not way_elements:
            warnings.warn(
                f"No way elements found for highway '{highway_ref}'.",
                RuntimeWarning,
            )
            return None

        # Collect all node coordinates referenced by any way
        all_node_coords: List[LatLon] = []
        osm_way_ids: List[int] = []

        for way in way_elements:
            osm_way_ids.append(way["id"])
            for node_id in way.get("nodes", []):
                if node_id in node_coords:
                    coord = node_coords[node_id]
                    # Avoid exact duplicates at way junctions
                    if not all_node_coords or all_node_coords[-1] != coord:
                        all_node_coords.append(coord)

        if len(all_node_coords) < 2:
            warnings.warn(
                f"Insufficient node data for highway '{highway_ref}' "
                f"(only {len(all_node_coords)} unique nodes).",
                RuntimeWarning,
            )
            return None

        # Sort into a directional polyline
        ordered_nodes = sort_nodes_by_direction(all_node_coords)
        _, total_length_km = build_polyline_km_index(ordered_nodes)

        return HighwayGeometry(
            ref=highway_ref,
            nodes=ordered_nodes,
            total_length_km=total_length_km,
            bounding_box=bounding_box,
            osm_way_ids=osm_way_ids,
        )

    def _parse_services_response(
        self,
        data: Dict[str, Any],
    ) -> List[ServiceArea]:
        """
        Parse Overpass JSON into a list of ServiceArea objects.

        Handles both node elements (direct lat/lon) and way/relation
        elements (uses Overpass 'center' lat/lon). Deduplicates by OSM ID.
        """
        elements = data.get("elements", [])
        seen_ids: set = set()
        areas: List[ServiceArea] = []

        for elem in elements:
            osm_id = elem.get("id")
            if osm_id in seen_ids:
                continue
            seen_ids.add(osm_id)

            osm_type = elem.get("type", "node")
            tags = elem.get("tags", {})
            name = tags.get("name", "").strip()

            # Resolve coordinates — nodes have direct lat/lon; ways use center
            if osm_type == "node":
                lat = elem.get("lat")
                lon = elem.get("lon")
            else:
                center = elem.get("center", {})
                lat = center.get("lat")
                lon = center.get("lon")

            if lat is None or lon is None:
                continue

            areas.append(ServiceArea(
                osm_id=osm_id,
                name=name,
                lat=float(lat),
                lon=float(lon),
                osm_type=osm_type,
                tags=tags,
            ))

        return areas

    # ------------------------------------------------------------------
    # HTTP execution with caching and failover
    # ------------------------------------------------------------------

    def _execute_query(
        self,
        query: str,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an Overpass QL query.

        Checks cache first (if use_cache is True), then hits the primary
        endpoint, then falls back to the secondary mirror.  Returns parsed
        JSON dict or None on total failure.
        """
        if self.use_cache:
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached

        data = self._http_post(OVERPASS_PRIMARY_URL, query)
        if data is None:
            warnings.warn(
                "Primary Overpass endpoint failed, trying fallback mirror...",
                RuntimeWarning,
            )
            data = self._http_post(OVERPASS_FALLBACK_URL, query)

        if data is not None:
            self._save_to_cache(cache_key, data)

        return data

    def _http_post(
        self,
        url: str,
        query: str,
    ) -> Optional[Dict[str, Any]]:
        """POST a query to the Overpass API and return parsed JSON, or None."""
        try:
            response = self._session.post(
                url,
                data={"data": query},
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            warnings.warn(
                f"Overpass API request to {url} timed out after {self.timeout_sec}s.",
                RuntimeWarning,
            )
        except requests.exceptions.HTTPError as exc:
            warnings.warn(
                f"Overpass API HTTP error from {url}: {exc}",
                RuntimeWarning,
            )
        except requests.exceptions.RequestException as exc:
            warnings.warn(
                f"Overpass API request to {url} failed: {exc}",
                RuntimeWarning,
            )
        except (ValueError, KeyError) as exc:
            warnings.warn(
                f"Failed to parse Overpass API response from {url}: {exc}",
                RuntimeWarning,
            )
        return None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        """SHA-256 hex digest of a string (used as cache filename)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load a cached JSON response. Returns None on cache miss or read error."""
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Write an API response to the disk cache."""
        if not self.use_cache:
            return
        path = self._cache_path(cache_key)
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f)
        except OSError as exc:
            warnings.warn(f"Could not write OSM cache file '{path}': {exc}", RuntimeWarning)
