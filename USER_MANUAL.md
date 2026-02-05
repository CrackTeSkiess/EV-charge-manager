# EV Charge Manager - User Manual

## 1. Introduction

EV Charge Manager is a discrete-event simulation framework that models electric vehicle (EV) charging ecosystems on highways. It simulates vehicle behavior, charging station operations, and driver decision-making to help analyze charging infrastructure performance.

### Key Features
- Realistic physics-based energy consumption modeling
- Multiple driver behavior profiles (conservative, balanced, aggressive, range-anxious)
- Configurable charging station layouts with mixed charger types
- Non-uniform traffic patterns (rush hour, bursts, periodic)
- Comprehensive KPI tracking and visualization
- Early stopping conditions for optimization scenarios

### Use Cases
- Infrastructure planning: Determine optimal station placement and capacity
- Performance analysis: Identify bottlenecks and service quality issues
- Policy evaluation: Test pricing strategies and queue management
- Research: Study driver behavior and charging patterns

---

## 2. Installation

### Requirements
- Python 3.8 or higher
- Windows, macOS, or Linux

### Step 1: Install Dependencies

```bash
cd EV-charge-manager
pip install -r requirements.txt
```

This installs:
- `numpy` - Numerical computations
- `pandas` - Data analysis
- `matplotlib` - Visualization
- `seaborn` - Statistical plots

### Step 2: Verify Installation

```bash
python main.py demo
```

If successful, you'll see a 2-hour simulation run with progress updates.

---

## 3. Running Simulations

### Quick Start (Demo Mode)

```bash
python main.py demo
```

Runs a pre-configured 2-hour simulation with rush hour traffic.

### Custom Simulation

```bash
python main.py [OPTIONS]
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--highway-length` | 3000.0 | Highway length in kilometers |
| `--num-stations` | 10 | Number of charging stations |
| `--chargers-per-station` | 4 | Chargers at each station |
| `--waiting-spots` | 6 | Queue capacity per station |
| `--vehicles-per-hour` | 60.0 | Average vehicle arrival rate |
| `--traffic-pattern` | RANDOM_POISSON | Arrival distribution pattern |
| `--duration` | 4.0 | Simulation duration (hours) |
| `--seed` | None | Random seed for reproducibility |
| `--output-dir` | ./simulation_output | Results directory |
| `--progress-interval` | 30 | Progress update frequency (steps) |
| `--no-visualize` | False | Skip generating charts |
| `--quiet` | False | Reduce console output |

### Traffic Patterns

| Pattern | Description |
|---------|-------------|
| `UNIFORM` | Constant arrival rate |
| `RANDOM_POISSON` | Random arrivals (memoryless) |
| `RUSH_HOUR` | Morning (8 AM) and evening (6 PM) peaks |
| `SINGLE_PEAK` | One peak period (noon default) |
| `PERIODIC` | Sinusoidal oscillation |
| `BURSTY` | Random clusters with gaps |

### Examples

**High-traffic rush hour simulation:**
```bash
python main.py --vehicles-per-hour 150 --traffic-pattern RUSH_HOUR --duration 8
```

**Reproducible experiment:**
```bash
python main.py --seed 42 --duration 24 --output-dir ./experiment_1
```

**Quick test with minimal output:**
```bash
python main.py --duration 1 --quiet --no-visualize
```

---

## 4. Understanding the Output

### Console Output

During simulation, you'll see progress updates:

```
[ 50.0%] Step 60/120 | Time: 07:01 | Vehicles: 81 | Queue: 0 | Wait: 0.0min | SOC: 40.5%
```

- **Step**: Current simulation step / total steps
- **Time**: Simulated time of day
- **Vehicles**: Active vehicles on highway
- **Queue**: Total vehicles waiting at all stations
- **Wait**: Average wait time in minutes
- **SOC**: Average state-of-charge (battery level)

### Output Files

Results are saved to the output directory:

| File | Description |
|------|-------------|
| `{id}_kpi.csv` | Time-series of all KPIs (one row per step) |
| `{id}_summary.json` | Aggregate statistics and configuration |
| `{id}_final_state.json` | System state at simulation end |

### Key Performance Indicators (KPIs)

**Critical KPIs:**
- `cars_quit_waiting` - Vehicles that abandoned the queue (per step)
- `cars_zero_battery` - Vehicles stranded with depleted battery (per step)

**Performance KPIs:**
- `avg_wait_time_minutes` - Average time spent in queue
- `max_wait_time_minutes` - Longest wait time observed
- `charging_utilization` - Fraction of chargers in use (0-1)
- `queue_occupancy_rate` - Queue fullness relative to capacity
- `abandonment_rate` - Fraction of arrivals that quit
- `service_level` - 1 minus abandonment rate

**Vehicle KPIs:**
- `vehicles_on_road` - Active vehicles in simulation
- `vehicles_entered` - New arrivals this step
- `vehicles_exited` - Completed trips this step
- `avg_soc_percent` - Average battery level

**Economic KPIs:**
- `estimated_revenue` - Revenue from charging ($0.40/kWh + connection fees)
- `customer_satisfaction_score` - Composite quality metric (0-1)

### Visualizations

If `--no-visualize` is not set, charts are generated:

1. **Required KPIs** - Bar charts of abandonments and strandings
2. **System Overview** - 6-panel dashboard with key metrics
3. **Distributions** - Histograms of wait times and SOC
4. **Correlation Heatmap** - KPI relationships
5. **Station Comparison** - Per-station performance

---

## 5. Configuration Reference

### Simulation Parameters

The simulation can be configured programmatically:

```python
from Simulation import Simulation, SimulationParameters, EarlyStopCondition
from VehicleGenerator import TemporalDistribution

params = SimulationParameters(
    # Highway
    highway_length_km=300.0,
    num_charging_areas=3,
    charging_area_positions=[75.0, 150.0, 225.0],  # Or None for auto-spacing
    chargers_per_area=4,
    waiting_spots_per_area=6,

    # Traffic
    vehicles_per_hour=80.0,
    temporal_distribution=TemporalDistribution.RUSH_HOUR,

    # Time
    simulation_duration_hours=24.0,
    time_step_minutes=1.0,

    # Reproducibility
    random_seed=42,

    # Early stopping
    early_stop=EarlyStopCondition(
        max_consecutive_strandings=10,
        abandonment_rate_threshold=0.3,
        max_queue_occupancy=1.5,
        convergence_patience=200
    )
)

sim = Simulation(params)
result = sim.run(verbose=True)
```

### Early Stop Conditions

The simulation can stop early if:

| Condition | Default | Description |
|-----------|---------|-------------|
| `max_consecutive_strandings` | 10 | Stop if strandings exceed threshold |
| `abandonment_rate_threshold` | 0.3 | Stop if 30%+ vehicles abandon |
| `max_queue_occupancy` | 1.5 | Stop if queues at 150% capacity |
| `min_service_level` | 0.5 | Stop if service drops below 50% |
| `convergence_patience` | 100 | Stop if no improvement for N steps |
| `max_wall_clock_seconds` | None | Real-time limit |

### Vehicle Fleet Configuration

Default fleet composition:

| Battery Size | Weight |
|--------------|--------|
| 40 kWh (compact) | 25% |
| 60 kWh (midsize) | 40% |
| 80 kWh (premium) | 25% |
| 100 kWh (truck) | 10% |

Default driver behaviors:

| Type | Weight | Characteristics |
|------|--------|-----------------|
| Conservative | 25% | High patience, low risk, charges early |
| Balanced | 40% | Moderate in all aspects |
| Aggressive | 25% | Low patience, high speed, charges late |
| Range Anxious | 10% | Very cautious, charges frequently |

---

## 6. Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'numpy'"**
```bash
pip install -r requirements.txt
```

**Simulation stops immediately with CONVERGENCE**

The default convergence patience may be too low. Increase it:
```bash
python main.py --duration 4  # Uses internal convergence_patience=2000
```

**No vehicles completing trips**

Check if highway is long enough for vehicles to reach the end within simulation duration. At 120 km/h, a 300km highway takes ~2.5 hours.

**High abandonment rates**

- Increase `chargers_per_area`
- Increase `waiting_spots`
- Reduce `vehicles_per_hour`

### Performance Tips

- Use `--quiet` for faster execution
- Use `--no-visualize` if you only need CSV data
- Set `--seed` for reproducible experiments
- Start with shorter durations (1-2 hours) for testing

### Getting Help

- Check console output for error messages
- Review the summary statistics for system health
- Examine KPI CSV for detailed time-series analysis

---

## 7. Architecture Overview

```
Simulation (orchestrator)
    ├── Environment (manages time and lifecycle)
    │   ├── Highway (road infrastructure)
    │   │   ├── ChargingArea[] (stations)
    │   │   │   ├── Charger[] (charging points)
    │   │   │   └── Queue (waiting vehicles)
    │   │   └── Vehicle[] (on-road vehicles)
    │   │       ├── Battery (energy state)
    │   │       └── DriverBehavior (decisions)
    │   └── VehicleGenerator (traffic creation)
    ├── KPITracker (metrics collection)
    └── VisualizationTool (charts)
```

### Simulation Loop (per step)

1. Generate new vehicle arrivals based on traffic pattern
2. Update all charging areas (process sessions, manage queues)
3. Update queued vehicles (patience decay, abandonment decisions)
4. Update charging vehicles (check completion)
5. Update driving vehicles (physics, energy consumption)
6. Handle station handoffs (vehicles entering stations)
7. Despawn completed/stranded vehicles
8. Record KPIs and check early stop conditions

---

*EV Charge Manager v1.0*
