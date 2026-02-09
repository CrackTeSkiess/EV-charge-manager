# EV Charge Manager - User Manual

## 1. Introduction

EV Charge Manager is a discrete-event simulation framework that models electric vehicle (EV) charging ecosystems on highways. It simulates vehicle behavior, charging station operations, and driver decision-making to help analyze charging infrastructure performance.

### Key Features
- Realistic physics-based energy consumption modeling
- Multiple driver behavior profiles (conservative, balanced, aggressive, range-anxious)
- Configurable charging station layouts with mixed charger types
- **Energy management** with configurable sources (grid, solar, wind, battery storage)
- Non-uniform traffic patterns (rush hour, bursts, periodic)
- Comprehensive KPI tracking and visualization
- Early stopping conditions for optimization scenarios

### Use Cases
- Infrastructure planning: Determine optimal station placement and capacity
- **Energy planning**: Evaluate renewable energy mixes and battery storage sizing
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
| `--enable-station-tracker` | False | Enable per-station time-series output |
| `--enable-vehicle-tracker` | False | Enable per-vehicle history tracking |
| `--no-queue-overflow` | False | Disable emergency queue overflow |

#### Energy Management Options

| Option | Default | Description |
|--------|---------|-------------|
| `--energy-grid-max-kw` | None | Grid power limit per station (kW). When set, enables energy management. |
| `--energy-solar-peak-kw` | None | Solar panel peak power per station (kW). Output follows a day/night bell curve. |
| `--energy-wind-base-kw` | None | Wind turbine base power per station (kW). Output varies stochastically. |
| `--energy-battery-kwh` | None | Battery storage capacity per station (kWh). Charges from surplus, discharges during deficit. |

When any energy option is specified, each charging area gets its own **Energy Manager** that controls which chargers receive power. If the total available energy is less than what the chargers need, some chargers are powered down. Occupied chargers with active sessions are prioritized (most-complete sessions kept running); if a charger with an active session must be powered down, the vehicle is returned to the queue with highest priority. Idle chargers are simply marked unavailable until power returns.

If no energy options are specified, all chargers have unlimited power (original behavior).

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

**Grid-constrained stations (200 kW limit per station):**
```bash
python main.py --energy-grid-max-kw 200 --vehicles-per-hour 100 --duration 4
```

**Solar + battery storage (off-grid scenario):**
```bash
python main.py --energy-solar-peak-kw 300 --energy-battery-kwh 500 --duration 24
```

**Full renewable mix with grid backup:**
```bash
python main.py --energy-grid-max-kw 150 --energy-solar-peak-kw 100 --energy-wind-base-kw 80 --energy-battery-kwh 400 --duration 12
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

**Energy Management KPIs** (populated when energy options are active):
- `energy_available_kw` - Total power available across all stations
- `energy_demand_kw` - Total power demanded by all chargers
- `energy_curtailment_kw` - Power deficit (demand minus available)
- `chargers_unpowered` - Number of chargers without power supply
- `vehicles_requeued_energy` - Vehicles sent back to queue due to power loss

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

### Energy Management Configuration

Each charging area can have its own energy manager with a mix of sources. When no energy configuration is provided, chargers have unlimited power.

```python
from EnergyManager import (
    EnergyManagerConfig, GridSourceConfig, SolarSourceConfig,
    WindSourceConfig, BatteryStorageConfig
)

# Define energy sources for each station
energy_cfg = EnergyManagerConfig(source_configs=[
    GridSourceConfig(max_power_kw=200),          # 200 kW grid connection
    SolarSourceConfig(peak_power_kw=150),        # 150 kW peak solar
    WindSourceConfig(base_power_kw=60),           # 60 kW average wind
    BatteryStorageConfig(capacity_kwh=500),       # 500 kWh battery
])

params = SimulationParameters(
    highway_length_km=300.0,
    num_charging_areas=3,
    chargers_per_area=4,
    vehicles_per_hour=80.0,
    simulation_duration_hours=24.0,
    random_seed=42,
    # One config per area (here the same config for all 3)
    energy_manager_configs=[energy_cfg] * 3
)
```

#### Energy Source Types

| Source | Config Class | Behavior |
|--------|-------------|----------|
| **Grid** | `GridSourceConfig` | Constant output at `max_power_kw` |
| **Solar** | `SolarSourceConfig` | Gaussian bell curve: 0 at night, peaks at `peak_hour` (default 1 PM) |
| **Wind** | `WindSourceConfig` | Stochastic output around `base_power_kw` with mean reversion |
| **Battery** | `BatteryStorageConfig` | Finite storage: charges from surplus, discharges during deficit |

#### Solar Source Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `peak_power_kw` | 100.0 | Maximum output at solar noon |
| `sunrise_hour` | 6.0 | Hour when output begins |
| `sunset_hour` | 20.0 | Hour when output ends |
| `peak_hour` | 13.0 | Hour of peak output |

#### Wind Source Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_power_kw` | 50.0 | Average expected output |
| `min_power_kw` | 10.0 | Minimum output floor |
| `max_power_kw` | 100.0 | Maximum output ceiling |
| `variability` | 0.3 | Noise amplitude (0-1 scale) |

#### Battery Storage Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity_kwh` | 500.0 | Total storage capacity |
| `max_charge_rate_kw` | 100.0 | Maximum charging power |
| `max_discharge_rate_kw` | 100.0 | Maximum discharging power |
| `initial_soc` | 0.5 | Starting state of charge (0-1) |
| `min_soc` | 0.1 | Minimum allowed SOC |
| `max_soc` | 0.95 | Maximum allowed SOC |
| `round_trip_efficiency` | 0.90 | Round-trip energy efficiency |

#### Per-Area Configuration

You can give different energy configurations to each area:

```python
area1_cfg = EnergyManagerConfig(source_configs=[
    GridSourceConfig(max_power_kw=300),
    SolarSourceConfig(peak_power_kw=200),
])
area2_cfg = EnergyManagerConfig(source_configs=[
    WindSourceConfig(base_power_kw=150),
    BatteryStorageConfig(capacity_kwh=800),
])
area3_cfg = EnergyManagerConfig(source_configs=[
    GridSourceConfig(max_power_kw=500),  # Well-connected station
])

params = SimulationParameters(
    num_charging_areas=3,
    energy_manager_configs=[area1_cfg, area2_cfg, area3_cfg],
    # ... other params
)
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
- If using energy management, increase grid power or add battery storage

**Many chargers unpowered / high curtailment**

- The energy supply is too low for the number of chargers. Increase `--energy-grid-max-kw` or add more sources.
- Add battery storage (`--energy-battery-kwh`) to buffer intermittent solar/wind.
- For solar-only setups, expect all chargers to be unpowered at night. Pair with grid or battery.

**Vehicles repeatedly requeued**

- This happens when energy supply fluctuates (e.g., passing clouds on solar). Add battery storage to smooth output, or increase grid capacity.

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
    │   │   │   ├── EnergyManager (power supply control)
    │   │   │   │   ├── GridSource
    │   │   │   │   ├── SolarSource
    │   │   │   │   ├── WindSource
    │   │   │   │   └── BatteryStorage
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
2. **Apply energy constraints** per charging area (update sources, power down/up chargers, requeue interrupted vehicles)
3. Update all charging areas (process sessions, manage queues)
4. Update queued vehicles (patience decay, abandonment decisions)
5. Update charging vehicles (check completion)
6. Update driving vehicles (physics, energy consumption)
7. Handle station handoffs (vehicles entering stations)
8. Despawn completed/stranded vehicles
9. Record KPIs and check early stop conditions

### Energy Management Flow (per area, per step)

When an energy manager is attached to a charging area, the following happens at the start of each step:

1. **Update sources**: Solar output recalculated from time-of-day, wind output varies stochastically
2. **Calculate demand**: Total power needed for all chargers (occupied + available + unpowered)
3. **Request energy**: Manager sums non-battery sources; if surplus, charges battery; if deficit, discharges battery
4. **Allocate power budget**: Occupied chargers (most-complete first) get priority, then available chargers
5. **Power down excess**: Chargers that don't fit in the budget are set to UNPOWERED
   - If an occupied charger is powered down, its vehicle is interrupted and returned to the queue with highest priority
   - If an available charger is powered down, it simply cannot accept new vehicles
6. **Power up**: If previously unpowered chargers now fit in the budget, they are restored to AVAILABLE

---

*EV Charge Manager v1.0*
