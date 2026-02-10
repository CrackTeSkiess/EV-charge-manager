"""
Station Data Collector Module
Collects per-station occupancy and queue data during simulation for visualization.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
from collections import defaultdict


@dataclass
class StationSnapshot:
    """Single timestep snapshot of a charging station's state."""
    timestamp: datetime
    station_id: str
    station_name: str
    occupancy: int  # Number of chargers in use
    queue_length: int  # Number of vehicles waiting
    total_chargers: int
    waiting_spots: int
    chargers_unpowered: int = 0  # Chargers without energy supply
    energy_available_kw: float = 0.0  # Total available power from energy manager


class StationDataCollector:
    """
    Collects time-series data for charging stations during simulation.
    
    Usage:
        collector = StationDataCollector()
        
        # During each simulation step:
        for area in highway.charging_areas.values():
            collector.record_snapshot(area, current_time)
        
        # At end of simulation:
        station_data = collector.get_station_dataframes()
        viz.set_station_data(station_data)
        viz.generate_full_report(output_dir, station_data=station_data)
    """
    
    def __init__(self):
        self.snapshots: Dict[str, List[StationSnapshot]] = defaultdict(list)
        self.station_names: Dict[str, str] = {}
    
    def record_snapshot(self, charging_area, timestamp: datetime) -> None:
        """
        Record a snapshot of a charging area's current state.
        
        Args:
            charging_area: ChargingArea instance
            timestamp: Current simulation time
        """
        area_id = charging_area.id
        
        # Count occupied chargers
        occupied = sum(1 for c in charging_area.chargers
                      if c.status.name == 'OCCUPIED')

        # Count queue length (only valid entries)
        queue_len = sum(1 for e in charging_area.queue if e.patience_score >= 0)

        # Energy management data
        unpowered = sum(1 for c in charging_area.chargers
                       if c.status.name == 'UNPOWERED')
        energy_available = 0.0
        if hasattr(charging_area, 'energy_manager') and charging_area.energy_manager and not charging_area.energy_manager.is_unlimited:
            energy_available = charging_area.energy_manager.total_available_kw

        snapshot = StationSnapshot(
            timestamp=timestamp,
            station_id=area_id,
            station_name=charging_area.name,
            occupancy=occupied,
            queue_length=queue_len,
            total_chargers=len(charging_area.chargers),
            waiting_spots=charging_area.waiting_spots,
            chargers_unpowered=unpowered,
            energy_available_kw=energy_available
        )
        
        self.snapshots[area_id].append(snapshot)
        self.station_names[area_id] = charging_area.name
    
    def record_all_stations(self, highway, timestamp: datetime) -> None:
        """
        Record snapshots for all charging areas on a highway.
        
        Args:
            highway: Highway instance with charging_areas
            timestamp: Current simulation time
        """
        for area_id, area in highway.charging_areas.items():
            self.record_snapshot(area, timestamp)
    
    def get_station_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Convert collected snapshots to DataFrames for visualization.
        
        Returns:
            Dict mapping station_id to DataFrame with columns:
            ['timestamp', 'occupancy', 'queue_length', 'total_chargers', 'waiting_spots']
        """
        result = {}
        
        for station_id, snapshots in self.snapshots.items():
            if not snapshots:
                continue
            
            # Create DataFrame
            data = {
                'timestamp': [s.timestamp for s in snapshots],
                'occupancy': [s.occupancy for s in snapshots],
                'queue_length': [s.queue_length for s in snapshots],
                'total_chargers': [s.total_chargers for s in snapshots],
                'waiting_spots': [s.waiting_spots for s in snapshots],
                'chargers_unpowered': [s.chargers_unpowered for s in snapshots],
                'energy_available_kw': [s.energy_available_kw for s in snapshots]
            }
            
            df = pd.DataFrame(data)
            df.attrs['name'] = self.station_names.get(station_id, f'Station {station_id}')
            result[station_id] = df
        
        return result
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for all stations."""
        stats = {}
        
        for station_id, snapshots in self.snapshots.items():
            if not snapshots:
                continue
            
            occupancies = [s.occupancy for s in snapshots]
            queues = [s.queue_length for s in snapshots]
            total_chargers = snapshots[0].total_chargers if snapshots else 1
            
            stats[station_id] = {
                'name': self.station_names.get(station_id, station_id),
                'avg_occupancy': sum(occupancies) / len(occupancies) if occupancies else 0,
                'max_occupancy': max(occupancies) if occupancies else 0,
                'avg_occupancy_rate': (sum(occupancies) / len(occupancies)) / total_chargers * 100 if occupancies else 0,
                'avg_queue': sum(queues) / len(queues) if queues else 0,
                'max_queue': max(queues) if queues else 0,
                'total_snapshots': len(snapshots)
            }
        
        return stats
    
    def clear(self) -> None:
        """Clear all collected data."""
        self.snapshots.clear()
        self.station_names.clear()