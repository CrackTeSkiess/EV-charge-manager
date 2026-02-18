"""
Highway Selection Module
========================
Select a real highway segment from OpenStreetMap and pre-configure
SimulationParameters with real-world service area positions.

The module fetches data from the OpenStreetMap Overpass API (free, no key
required) and caches all responses to disk.  Service area positions are
converted to km along the highway, fixing ``highway_length_km``,
``num_charging_areas``, and ``charging_area_positions`` in the returned
``SimulationParameters``.  All other simulation parameters (chargers,
power tiers, queue capacity, energy configuration, traffic) remain at
their defaults and are free for the user to tune.

Quick start::

    from ev_charge_manager.highway_selection import HighwaySelector
    from ev_charge_manager.simulation import Simulation

    selector = HighwaySelector()

    # List pre-registered highways
    for hw in HighwaySelector.list_available_highways():
        print(hw["ref"], hw["country"], hw["approx_km"], "km")

    # Select a 300 km segment of the German A5 and view on an interactive map
    result = selector.select("A5", start_km=0, end_km=300, show_map=True)
    print(result.summary())

    # Tune the remaining parameters freely
    params = result.simulation_params
    params.chargers_per_area = 6
    params.vehicles_per_hour = 80.0

    # Run the simulation
    sim_result = Simulation(params).run()

For highways not in the built-in registry use ``select_custom()``::

    result = selector.select_custom(
        highway_ref="A96",
        bounding_box=(47.5, 8.8, 48.5, 11.5),  # (south, west, north, east)
        end_km=200,
    )
"""

from .selector import (
    HIGHWAY_REGISTRY,
    HighwaySelectionResult,
    HighwaySelector,
    StopAreaInfo,
)
from .osm_client import (
    HighwayGeometry,
    OverpassClient,
    ServiceArea,
)
from .geo_utils import (
    build_polyline_km_index,
    haversine_km,
    project_point_to_polyline,
)

__all__ = [
    # Main API
    "HighwaySelector",
    "HighwaySelectionResult",
    "StopAreaInfo",
    "HIGHWAY_REGISTRY",
    # OSM client (advanced use)
    "OverpassClient",
    "ServiceArea",
    "HighwayGeometry",
    # Geo utilities (advanced use)
    "haversine_km",
    "build_polyline_km_index",
    "project_point_to_polyline",
]
