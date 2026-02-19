"""
Geographic utilities for the Highway Selection Module.

Pure-math functions — no external dependencies beyond stdlib math.
All distance calculations use the haversine formula.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

# Type alias for a geographic coordinate pair (latitude, longitude) in decimal degrees
LatLon = Tuple[float, float]

EARTH_RADIUS_KM = 6371.0


def haversine_km(point_a: LatLon, point_b: LatLon) -> float:
    """
    Compute the great-circle distance in kilometres between two (lat, lon) points.

    Uses the haversine formula. Accurate to within ~0.5% for distances under 500 km,
    which is sufficient for highway segment positioning.

    Args:
        point_a: (latitude, longitude) in decimal degrees
        point_b: (latitude, longitude) in decimal degrees

    Returns:
        Distance in kilometres.
    """
    lat1, lon1 = math.radians(point_a[0]), math.radians(point_a[1])
    lat2, lon2 = math.radians(point_b[0]), math.radians(point_b[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_KM * c


def bearing_deg(point_a: LatLon, point_b: LatLon) -> float:
    """
    Compute compass bearing from point_a to point_b in degrees [0, 360).

    Args:
        point_a: (latitude, longitude) in decimal degrees — origin
        point_b: (latitude, longitude) in decimal degrees — destination

    Returns:
        Bearing in degrees, clockwise from north.
    """
    lat1, lon1 = math.radians(point_a[0]), math.radians(point_a[1])
    lat2, lon2 = math.radians(point_b[0]), math.radians(point_b[1])
    dlon = lon2 - lon1

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def build_polyline_km_index(nodes: List[LatLon]) -> Tuple[List[float], float]:
    """
    Convert an ordered list of lat/lon nodes into a cumulative km distance index.

    Args:
        nodes: Ordered list of (lat, lon) coordinates forming the polyline.

    Returns:
        Tuple of:
            cumulative_km: List of cumulative km values, same length as nodes.
                           cumulative_km[0] == 0.0, cumulative_km[-1] == total length.
            total_length_km: Total length of the polyline in km.
    """
    if not nodes:
        return [], 0.0

    cumulative = [0.0]
    for i in range(1, len(nodes)):
        d = haversine_km(nodes[i - 1], nodes[i])
        cumulative.append(cumulative[-1] + d)

    return cumulative, cumulative[-1]


def project_point_to_polyline(
    point: LatLon,
    nodes: List[LatLon],
    cumulative_km: List[float],
) -> float:
    """
    Find the km position of a point along a polyline via nearest-node projection.

    Finds the node in nodes[] that is closest (haversine distance) to point,
    then returns cumulative_km[closest_node_index].

    Args:
        point: (lat, lon) of the point to project.
        nodes: Ordered list of polyline nodes.
        cumulative_km: Cumulative km index (same length as nodes).

    Returns:
        km position of the projected point from the start of the polyline.
        Returns 0.0 if nodes is empty.
    """
    if not nodes:
        return 0.0

    min_dist = math.inf
    nearest_idx = 0
    for i, node in enumerate(nodes):
        d = haversine_km(point, node)
        if d < min_dist:
            min_dist = d
            nearest_idx = i

    return cumulative_km[nearest_idx]


def project_point_validated(
    point: LatLon,
    nodes: List[LatLon],
    cumulative_km: List[float],
    max_distance_km: float = 3.0,
) -> Optional[float]:
    """
    Project a point onto a polyline and return its km position, or None if
    the point lies farther than ``max_distance_km`` from the nearest node.

    This filters out service areas that happen to be inside the highway's
    bounding box but are not actually on this specific highway (e.g. a
    fuel station on a parallel road).

    Args:
        point: (lat, lon) to project.
        nodes: Ordered polyline nodes.
        cumulative_km: Cumulative km index (same length as nodes).
        max_distance_km: Projection is rejected if the nearest highway node
                         is farther than this distance.  Default 3 km covers
                         the longest motorway slip roads.

    Returns:
        km position from the start of the polyline, or None if too far away.
    """
    if not nodes:
        return None

    min_dist = math.inf
    nearest_idx = 0
    for i, node in enumerate(nodes):
        d = haversine_km(point, node)
        if d < min_dist:
            min_dist = d
            nearest_idx = i

    if min_dist > max_distance_km:
        return None

    return cumulative_km[nearest_idx]


def filter_nodes_by_km_range(
    nodes: List[LatLon],
    cumulative_km: List[float],
    start_km: float,
    end_km: float,
) -> Tuple[List[LatLon], List[float]]:
    """
    Trim a polyline to the [start_km, end_km] range and re-zero the km index.

    Args:
        nodes: Ordered list of polyline nodes.
        cumulative_km: Cumulative km index (same length as nodes).
        start_km: Start of the desired range in km.
        end_km: End of the desired range in km.

    Returns:
        Tuple of:
            filtered_nodes: Subset of nodes within the range.
            re_zeroed_km: Cumulative km values re-zeroed to 0 at start_km.
    """
    filtered_nodes: List[LatLon] = []
    filtered_km: List[float] = []

    for node, km in zip(nodes, cumulative_km):
        if start_km <= km <= end_km:
            filtered_nodes.append(node)
            filtered_km.append(km - start_km)

    return filtered_nodes, filtered_km


def sort_nodes_by_direction(nodes: List[LatLon]) -> List[LatLon]:
    """
    Sort an unordered set of highway nodes into a directional sequence.

    Uses a greedy nearest-neighbour chain starting from the westernmost node
    (smallest longitude). This handles the common case where OSM returns
    multiple disconnected way segments that need to be stitched together.

    For highways running predominantly N-S (bearing close to 0° or 180°),
    starts from the southernmost node instead.

    Args:
        nodes: Unordered list of (lat, lon) coordinates.

    Returns:
        Ordered list forming a continuous polyline.
    """
    if len(nodes) <= 1:
        return list(nodes)

    # Detect dominant highway direction by comparing extreme points
    westmost = min(nodes, key=lambda n: n[1])
    eastmost = max(nodes, key=lambda n: n[1])
    northmost = max(nodes, key=lambda n: n[0])
    southmost = min(nodes, key=lambda n: n[0])

    lon_span = abs(eastmost[1] - westmost[1])
    lat_span = abs(northmost[0] - southmost[0])

    # Start from the western (or southern for N-S highways) extremity
    if lon_span >= lat_span:
        start_node = westmost
    else:
        start_node = southmost

    remaining = list(nodes)
    ordered: List[LatLon] = []

    # Find and remove start node
    start_idx = min(range(len(remaining)), key=lambda i: haversine_km(remaining[i], start_node))
    ordered.append(remaining.pop(start_idx))

    # Greedy nearest-neighbour chain
    while remaining:
        current = ordered[-1]
        nearest_idx = min(range(len(remaining)), key=lambda i: haversine_km(remaining[i], current))
        ordered.append(remaining.pop(nearest_idx))

    return ordered
