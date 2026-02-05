"""
EV Charge Manager - Main Entry Point
Run this file to start a simulation of the EV charging ecosystem.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from Simulation import Simulation, SimulationParameters, EarlyStopCondition
from VehicleGenerator import TemporalDistribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EV Charging Station Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Highway configuration
    parser.add_argument(
        "--highway-length", type=float, default=300.0,
        help="Highway length in kilometers"
    )
    parser.add_argument(
        "--num-stations", type=int, default=3,
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

    return parser.parse_args()


def main():
    """Main entry point for EV charging simulation."""
    args = parse_args()

    # Map traffic pattern string to enum
    traffic_pattern = TemporalDistribution[args.traffic_pattern]

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
        early_stop=EarlyStopCondition(
            max_consecutive_strandings=10,
            abandonment_rate_threshold=0.3,
            max_queue_occupancy=1.5,
            convergence_patience=200
        )
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
