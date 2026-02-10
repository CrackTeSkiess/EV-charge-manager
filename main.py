"""
EV Charge Manager - Main Entry Point
Run this file to start a simulation of the EV charging ecosystem.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from ev_charge_manager.simulation import Simulation, SimulationParameters, EarlyStopCondition
from ev_charge_manager.vehicle.generator import TemporalDistribution
from ev_charge_manager.visualization import VisualizationTool
from ev_charge_manager.analytics import StationDataCollector
from ev_charge_manager.energy import (
    EnergyManagerConfig, GridSourceConfig, SolarSourceConfig,
    WindSourceConfig, BatteryStorageConfig
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EV Charging Station Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Highway configuration
    parser.add_argument(
        "--highway-length", type=float, default=3000.0,
        help="Highway length in kilometers"
    )
    parser.add_argument(
        "--num-stations", type=int, default=5,
        help="Number of charging stations along the highway"
    )
    parser.add_argument(
        "--chargers-per-station", type=int, default=4,
        help="Number of chargers at each station"
    )
    parser.add_argument(
        "--waiting-spots", type=int, default=6,
        help="Waiting spots per station"
    )

    # Traffic configuration
    parser.add_argument(
        "--vehicles-per-hour", type=float, default=60.0,
        help="Average vehicle arrival rate per hour"
    )
    parser.add_argument(
        "--traffic-pattern", type=str, default="RANDOM_POISSON",
        choices=["UNIFORM", "RANDOM_POISSON", "RUSH_HOUR", "SINGLE_PEAK", "PERIODIC", "BURSTY"],
        help="Temporal distribution pattern for vehicle arrivals"
    )

    # Simulation configuration
    parser.add_argument(
        "--duration", type=float, default=4.0,
        help="Simulation duration in hours"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", type=str, default="./simulation_output",
        help="Directory to save results and visualizations"
    )
    parser.add_argument(
        "--no-visualize", action="store_true",
        help="Skip generating visualizations"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--progress-interval", type=int, default=30,
        help="Print progress every N steps"
    )
    parser.add_argument(
        "--enable-station-tracker", action="store_true",
        help="Enable station tracking output"
    )
    parser.add_argument(
        "--enable-vehicle-tracker", action="store_true",
        help="Enable per-vehicle history tracking (detailed/state/position)"
    )
    parser.add_argument(
        "--no-queue-overflow", action="store_true",
        help="Disable emergency queue overflow (vehicles rejected even if they can't reach next station)"
    )

    # Energy management
    parser.add_argument(
        "--energy-grid-max-kw", type=float, default=None,
        help="Grid power limit per station in kW (default: unlimited/disabled)"
    )
    parser.add_argument(
        "--energy-solar-peak-kw", type=float, default=None,
        help="Solar panel peak power per station in kW (default: disabled)"
    )
    parser.add_argument(
        "--energy-wind-base-kw", type=float, default=None,
        help="Wind turbine base power per station in kW (default: disabled)"
    )
    parser.add_argument(
        "--energy-battery-kwh", type=float, default=None,
        help="Battery storage capacity per station in kWh (default: disabled)"
    )

    return parser.parse_args()


def main():
    """Main entry point for EV charging simulation."""
    args = parse_args()

    # Map traffic pattern string to enum
    traffic_pattern = TemporalDistribution[args.traffic_pattern]

    # Build energy manager configs if any energy source is specified
    energy_configs = None
    has_energy = any([
        args.energy_grid_max_kw, args.energy_solar_peak_kw,
        args.energy_wind_base_kw, args.energy_battery_kwh
    ])
    if has_energy:
        source_configs = []
        if args.energy_grid_max_kw is not None:
            source_configs.append(GridSourceConfig(max_power_kw=args.energy_grid_max_kw))
        if args.energy_solar_peak_kw is not None:
            source_configs.append(SolarSourceConfig(peak_power_kw=args.energy_solar_peak_kw))
        if args.energy_wind_base_kw is not None:
            source_configs.append(WindSourceConfig(base_power_kw=args.energy_wind_base_kw))
        if args.energy_battery_kwh is not None:
            source_configs.append(BatteryStorageConfig(capacity_kwh=args.energy_battery_kwh))

        # Same config for every station
        single_config = EnergyManagerConfig(source_configs=source_configs)
        energy_configs = [single_config] * args.num_stations

    # Create simulation parameters
    params = SimulationParameters(
        highway_length_km=args.highway_length,
        num_charging_areas=args.num_stations,
        chargers_per_area=args.chargers_per_station,
        waiting_spots_per_area=args.waiting_spots,
        vehicles_per_hour=args.vehicles_per_hour,
        temporal_distribution=traffic_pattern,
        simulation_duration_hours=args.duration,
        random_seed=args.seed,
        enable_station_tracking=args.enable_station_tracker,
        enable_vehicle_tracking=args.enable_vehicle_tracker,
        allow_queue_overflow=not args.no_queue_overflow,
        early_stop=EarlyStopCondition.disabled(),
        energy_manager_configs=energy_configs
    )

    # Create and run simulation
    print("=" * 70)
    print("EV CHARGE MANAGER SIMULATION")
    print("=" * 70)
    print(f"Highway: {args.highway_length} km with {args.num_stations} charging stations")
    print(f"Traffic: {args.vehicles_per_hour} vehicles/hour ({args.traffic_pattern})")
    print(f"Duration: {args.duration} hours")
    if args.seed:
        print(f"Random seed: {args.seed}")
    if has_energy:
        sources = []
        if args.energy_grid_max_kw is not None:
            sources.append(f"Grid({args.energy_grid_max_kw}kW)")
        if args.energy_solar_peak_kw is not None:
            sources.append(f"Solar({args.energy_solar_peak_kw}kW peak)")
        if args.energy_wind_base_kw is not None:
            sources.append(f"Wind({args.energy_wind_base_kw}kW base)")
        if args.energy_battery_kwh is not None:
            sources.append(f"Battery({args.energy_battery_kwh}kWh)")
        print(f"Energy: {' + '.join(sources)} per station")
    print("=" * 70)
    print()

    sim = Simulation(params)
        
    result = sim.run(
        progress_interval=args.progress_interval,
        verbose=not args.quiet
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to {output_dir}...")
    result.save(str(output_dir))

    # Generate visualizations
    if not args.no_visualize:
        try:
            print("Generating visualizations...")
            viz_files = sim.visualize(str(output_dir))
            print(f"Generated {len(viz_files)} visualization files")
        except ImportError as e:
            print(f"Visualization skipped (missing dependency): {e}")
        except Exception as e:
            print(f"Visualization failed: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)

    stats = result.summary_statistics
    print(f"Stop reason: {result.stop_reason.name}")
    print(f"Steps completed: {result.final_step}")
    print(f"Wall clock time: {result.wall_clock_time_seconds:.2f} seconds")

    print(f"\nVehicles:")
    print(f"  Total quit waiting: {stats.get('total_cars_quit_waiting', 0)}")
    print(f"  Total stranded (0% battery): {stats.get('total_cars_zero_battery', 0)}")
    print(f"  Avg quit rate: {stats.get('avg_quit_per_hour', 0):.2f}/hour")

    print(f"\nPerformance:")
    print(f"  Avg wait time: {stats.get('avg_wait_time', 0):.1f} min")
    print(f"  Max wait time: {stats.get('max_wait_time', 0):.1f} min")
    print(f"  Avg utilization: {stats.get('avg_utilization', 0):.1%}")
    print(f"  Service level: {stats.get('avg_service_level', 0):.1%}")

    print(f"\nEconomics:")
    print(f"  Total revenue: ${stats.get('total_revenue', 0):.2f}")

    print("=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    if not args.no_visualize:
        report_data = sim.generate_stranding_report(output_dir=str(output_dir))
        if report_data["total_strandings"] > 0:
            print(f"\nStranding Report Summary:")
            print(f"  Total strandings: {report_data['total_strandings']}")
            print(f"  Stranding rate: {report_data['stranding_rate']:.2%}")
            
            print(f"\nTop Recommendations:")
            for rec in report_data['recommendations'][:3]:
                print(f"  [{rec['priority']}] {rec['message']}")
    
    return 0


def run_quick_demo():
    """Run a quick demonstration with default settings."""
    print("Running quick demo simulation...")
    print()

    params = SimulationParameters(
        highway_length_km=200.0,
        num_charging_areas=3,
        chargers_per_area=4,
        waiting_spots_per_area=6,
        vehicles_per_hour=80.0,
        temporal_distribution=TemporalDistribution.RUSH_HOUR,
        simulation_duration_hours=2.0,
        random_seed=42
    )

    sim = Simulation(params)
    result = sim.run(progress_interval=30, verbose=True)

    print("\nQuick demo completed!")
    print(f"Total vehicles quit waiting: {result.summary_statistics.get('total_cars_quit_waiting', 0)}")
    print(f"Total vehicles stranded: {result.summary_statistics.get('total_cars_zero_battery', 0)}")

    return result


if __name__ == "__main__":
    # Check if running demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_quick_demo()
    else:
        sys.exit(main())
