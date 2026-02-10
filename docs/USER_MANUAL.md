# EV Charge Manager — User Manual

## 1. Introduction

EV Charge Manager is a discrete-event simulation framework for electric vehicle (EV) charging infrastructure on highways. It combines physics-based simulation with hierarchical multi-agent reinforcement learning (RL) for dual-level optimization.

### Key Features

- Realistic physics-based energy consumption modelling
- Multiple driver behaviour profiles (conservative, balanced, aggressive, range-anxious)
- Configurable charging station layouts with mixed charger types
- **Energy management** with configurable sources (grid, solar, wind, battery storage)
- Non-uniform traffic patterns (rush hour, bursts, periodic)
- Comprehensive KPI tracking and visualization
- Early stopping conditions for optimization scenarios
- **Hierarchical multi-agent RL** for infrastructure planning and energy arbitrage

### Use Cases

- **Infrastructure planning**: Determine optimal station placement and capacity
- **Energy planning**: Evaluate renewable energy mixes and battery storage sizing
- **Performance analysis**: Identify bottlenecks and service quality issues
- **Policy evaluation**: Test pricing strategies and queue management
- **RL training**: Train PPO and hierarchical agents for autonomous optimization

---

## 2. Installation

### Requirements

- Python 3.8 or higher
- Windows, macOS, or Linux

### Step 1: Install Core Dependencies

```bash
cd EV-charge-manager
pip install -r requirements.txt
```

This installs: `numpy`, `pandas`, `matplotlib`, `seaborn`.

### Step 2: Install RL Dependencies (optional)

Required only for reinforcement learning features:

```bash
pip install torch gymnasium
```

### Step 3: Verify Installation

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

Runs a pre-configured 2-hour simulation with rush-hour traffic.

### Custom Simulation

```bash
python main.py [OPTIONS]
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--highway-length` | 3000.0 | Highway length in kilometres |
| `--num-stations` | 10 | Number of charging stations |
| `--chargers-per-station` | 4 | Chargers at each station |
| `--waiting-spots` | 6 | Queue capacity per station |
| `--vehicles-per-hour` | 60.0 | Average vehicle arrival rate |
| `--traffic-pattern` | POISSON | Arrival distribution pattern |
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
| `--energy-grid-max-kw` | None | Grid power limit per station (kW). Enables energy management when set. |
| `--energy-solar-peak-kw` | None | Solar panel peak power per station (kW). Output follows a day/night bell curve. |
| `--energy-wind-base-kw` | None | Wind turbine base power per station (kW). Output varies stochastically. |
| `--energy-battery-kwh` | None | Battery storage capacity per station (kWh). Charges from surplus, discharges during deficit. |

When any energy option is specified, each charging area gets its own **EnergyManager** that controls which chargers receive power. If total available energy is less than what chargers need, some chargers are powered down. Occupied chargers (most-complete sessions first) are prioritised; if an occupied charger must be powered down, the vehicle is returned to the queue with highest priority.

If no energy options are specified, all chargers have unlimited power.

### Traffic Patterns

| Pattern | Description |
|---------|-------------|
| `UNIFORM` | Constant arrival rate |
| `POISSON` | Random arrivals (memoryless) |
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
python main.py --energy-grid-max-kw 150 --energy-solar-peak-kw 100 \
               --energy-wind-base-kw 80 --energy-battery-kwh 400 --duration 12
```

---

## 4. Reinforcement Learning Training

### Multi-Agent PPO Training

Train PPO agents to optimize infrastructure placement and capacity:

```bash
python train.py --n-agents 3 --episodes 1000 --highway-length 300 --save-dir ./models
python evaluate.py --model ./models/ppo_agent.pt --episodes 20 --render
```

### Hierarchical Dual-RL Training

The hierarchical system trains two levels simultaneously:
- **Macro-RL** (PPO): one agent per station, optimises position, charger count, and waiting spots
- **Micro-RL** (Actor-Critic): one agent per station, handles energy arbitrage (battery charge/discharge timing)

#### Training Modes

| Mode | Stability | Typical Duration | Recommended |
|------|-----------|------------------|-------------|
| `SEQUENTIAL` | High | 4–6 h | **Yes** |
| `SIMULTANEOUS` | Low | 8–12 h | No |
| `CURRICULUM` | Highest | 12–24 h | Production |

```bash
# RECOMMENDED – sequential: micro first → macro
python hierarchical.py                                         # 3 agents, defaults
python hierarchical.py --n-agents 5 --micro-episodes 2000     # custom scale
python hierarchical.py --robust                                # high-episode preset

# Simultaneous (both levels train together)
python hierarchical.py --mode simultaneous

# Curriculum (progressive 1-day → 3-day → 7-day complexity)
python hierarchical.py --mode curriculum --robust

# Frozen-micro – load pre-trained micro-agents, train macro only
python hierarchical.py --mode frozen_micro \
    --micro-load ./models/hierarchical/micro_pretrained
```

All outputs land in `--save-dir` (default `./models/hierarchical/`):

| File | Written by | Description |
|------|-----------|-------------|
| `run_config.json` | `hierarchical.py` | All CLI parameters used for this run |
| `results.json` | `hierarchical.py` | Final training metrics summary |
| `training_result.json` | `HierarchicalTrainer` | Full trainer output (best config, rewards) |
| `training_history.jsonl` | `MultiAgentPPO` | One JSON record per PPO update (macro metrics) |
| `micro_history.json` | `MicroRLTrainer` | Per-station episode rewards and grid costs |
| `micro_final_day.json` | `MicroRLTrainer` | Hourly breakdown from the last micro episode |
| `micro_pretrained/` | `MicroRLTrainer` | Saved micro-agent weights (reusable) |
| `training_plots/` | `TrainingVisualizer` | Auto-generated PNG plots (see below) |

Pass `--no-visualize` to skip plot generation.

### Training Visualisation

Training plots are generated automatically at the end of every `hierarchical.py` run and saved to `{save_dir}/training_plots/`. You can also generate or regenerate them at any time with `visualize.py`:

```bash
# Re-generate all plots from a completed run
python visualize.py training --save-dir ./models/hierarchical

# Override the highway length used for position evolution bands
python visualize.py training --save-dir ./models/hierarchical --highway-length 400

# Compare several runs side-by-side
python visualize.py compare --dirs ./models/sequential ./models/simultaneous ./models/curriculum
```

#### Generated Plot Files

| File | Content |
|------|---------|
| `macro_learning_curves.png` | Episode reward (raw + rolling mean) and PPO losses over training |
| `config_evolution.png` | How station positions, charger counts, and waiting spots evolved |
| `micro_convergence.png` | Per-station daily reward and grid cost over micro-RL training |
| `battery_strategy.png` | Learned SOC profile and charge/discharge actions across 24 h |
| `cost_breakdown.png` | Per-station cost decomposition (grid, other, arbitrage) + waterfall |
| `curriculum_stages.png` | Reward, cost, and arbitrage per curriculum stage *(curriculum mode only)* |

#### Programmatic API

```python
from ev_charge_manager.visualization.training_plots import TrainingVisualizer

tv = TrainingVisualizer()

# Generate everything in one call
plots = tv.generate_full_report("./models/hierarchical")

# Or call individual methods
tv.plot_macro_learning_curves("./models/hierarchical/training_history.jsonl",
                              output_dir="./plots")
tv.plot_micro_convergence("./models/hierarchical/micro_history.json",
                          output_dir="./plots")
tv.plot_battery_strategy("./models/hierarchical/micro_final_day.json",
                         output_dir="./plots")

# Compare runs programmatically
tv.plot_mode_comparison(["./models/sequential", "./models/curriculum"],
                        output_dir="./plots/comparison")
```

### Pre-trained Models

Pre-trained models are provided in `models/sequential/`:
- 3 stations at positions [75.4, 148.2, 226.9] km on a 300 km highway
- 2 chargers per station, 10–12 waiting spots
- Trained: 500 micro + 300 macro episodes (SEQUENTIAL mode)

---

## 5. Analysis Tools

```bash
# Traffic vs stranding rate analysis
python tools/traffic_stranding.py

# Pareto-front infrastructure grid search
python tools/infrastructure_optimizer.py
```

---

## 6. Understanding the Output

### Console Output

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

| File | Description |
|------|-------------|
| `{id}_kpi.csv` | Time-series of all KPIs (one row per step) |
| `{id}_summary.json` | Aggregate statistics and configuration |
| `{id}_final_state.json` | System state at simulation end |

### Key Performance Indicators (KPIs)

**Critical KPIs:**
- `cars_quit_waiting` — vehicles that abandoned the queue (per step)
- `cars_zero_battery` — vehicles stranded with depleted battery (per step)

**Performance KPIs:**
- `avg_wait_time_minutes` — average time spent in queue
- `max_wait_time_minutes` — longest wait time observed
- `charging_utilization` — fraction of chargers in use (0–1)
- `queue_occupancy_rate` — queue fullness relative to capacity
- `abandonment_rate` — fraction of arrivals that quit
- `service_level` — 1 minus abandonment rate

**Vehicle KPIs:**
- `vehicles_on_road` — active vehicles in simulation
- `vehicles_entered` — new arrivals this step
- `vehicles_exited` — completed trips this step
- `avg_soc_percent` — average battery level

**Economic KPIs:**
- `estimated_revenue` — revenue from charging ($0.40/kWh + connection fees)
- `customer_satisfaction_score` — composite quality metric (0–1)

**Energy Management KPIs** (populated when energy options are active):
- `energy_available_kw` — total power available across all stations
- `energy_demand_kw` — total power demanded by all chargers
- `energy_curtailment_kw` — power deficit (demand minus available)
- `chargers_unpowered` — number of chargers without power supply
- `vehicles_requeued_energy` — vehicles sent back to queue due to power loss

KPI alert levels escalate through: `WARNING → CRITICAL → EMERGENCY`

### Visualisations

Unless `--no-visualize` is set, charts are generated:

1. **Required KPIs** — bar charts of abandonments and strandings
2. **System Overview** — 6-panel dashboard with key metrics
3. **Distributions** — histograms of wait times and SOC
4. **Correlation Heatmap** — KPI relationships
5. **Station Comparison** — per-station performance

---

## 7. Configuration Reference

### Programmatic API

Import paths follow the `ev_charge_manager` package structure:

```python
from ev_charge_manager.simulation import Simulation, SimulationParameters, EarlyStopCondition
from ev_charge_manager.vehicle.generator import VehicleGenerator, GeneratorConfig, TemporalDistribution

params = SimulationParameters(
    # Highway
    highway_length_km=300.0,
    num_charging_areas=3,
    charging_area_positions=[75.0, 150.0, 225.0],  # or None for auto-spacing
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

> **Note:** `VehicleGenerator` must be imported directly from `ev_charge_manager.vehicle.generator`, not from `ev_charge_manager.vehicle`, to avoid a circular import.

### Energy Management Configuration

```python
from ev_charge_manager.energy import (
    EnergyManager, EnergyManagerConfig, GridSourceConfig,
    SolarSourceConfig, WindSourceConfig, BatteryStorageConfig
)

energy_cfg = EnergyManagerConfig(source_configs=[
    GridSourceConfig(max_power_kw=200),       # 200 kW grid connection
    SolarSourceConfig(peak_power_kw=150),     # 150 kW peak solar
    WindSourceConfig(base_power_kw=60),       # 60 kW average wind
    BatteryStorageConfig(capacity_kwh=500),   # 500 kWh battery
])

params = SimulationParameters(
    highway_length_km=300.0,
    num_charging_areas=3,
    chargers_per_area=4,
    vehicles_per_hour=80.0,
    simulation_duration_hours=24.0,
    random_seed=42,
    energy_manager_configs=[energy_cfg] * 3   # one config per area
)
```

#### Energy Source Types

| Source | Config Class | Behaviour |
|--------|-------------|-----------|
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
| `variability` | 0.3 | Noise amplitude (0–1) |

#### Battery Storage Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity_kwh` | 500.0 | Total storage capacity |
| `max_charge_rate_kw` | 100.0 | Maximum charging power |
| `max_discharge_rate_kw` | 100.0 | Maximum discharging power |
| `initial_soc` | 0.5 | Starting state of charge (0–1) |
| `min_soc` | 0.1 | Minimum allowed SOC |
| `max_soc` | 0.95 | Maximum allowed SOC |
| `round_trip_efficiency` | 0.90 | Round-trip energy efficiency |

#### Per-Area Configuration

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
    GridSourceConfig(max_power_kw=500),
])

params = SimulationParameters(
    num_charging_areas=3,
    energy_manager_configs=[area1_cfg, area2_cfg, area3_cfg],
    # ... other params
)
```

### Early Stop Conditions

| Condition | Default | Description |
|-----------|---------|-------------|
| `max_consecutive_strandings` | 10 | Stop if strandings exceed threshold |
| `abandonment_rate_threshold` | 0.3 | Stop if 30%+ vehicles abandon |
| `max_queue_occupancy` | 1.5 | Stop if queues at 150% capacity |
| `min_service_level` | 0.5 | Stop if service drops below 50% |
| `convergence_patience` | 100 | Stop if no improvement for N steps |
| `max_wall_clock_seconds` | None | Real-time limit |

Use `EarlyStopCondition.disabled()` for deterministic runs where early halting is undesirable.

### Vehicle Fleet Composition

Default fleet:

| Battery Size | Weight |
|--------------|--------|
| 40 kWh (compact) | 25% |
| 60 kWh (midsize) | 40% |
| 80 kWh (premium) | 25% |
| 100 kWh (truck) | 10% |

Default driver behaviours:

| Type | Weight | Characteristics |
|------|--------|-----------------|
| Conservative | 25% | High patience, low risk, charges early |
| Balanced | 40% | Moderate in all aspects |
| Aggressive | 25% | Low patience, high speed, charges late |
| Range Anxious | 10% | Very cautious, charges frequently |

---

## 8. Troubleshooting

### Common Issues

**`ModuleNotFoundError: No module named 'numpy'`**
```bash
pip install -r requirements.txt
```

**`ModuleNotFoundError: No module named 'torch'`**
```bash
pip install torch gymnasium
```

**Simulation stops immediately with CONVERGENCE**

Increase the convergence patience or disable early stopping:
```python
early_stop=EarlyStopCondition.disabled()
```

**No vehicles completing trips**

Check that highway length is appropriate for simulation duration. At 120 km/h, a 300 km highway takes ~2.5 hours.

**High abandonment rates**
- Increase `--chargers-per-station`
- Increase `--waiting-spots`
- Reduce `--vehicles-per-hour`
- If using energy management, increase grid power or add battery storage

**Many chargers unpowered / high curtailment**
- Increase `--energy-grid-max-kw` or add more sources
- Add battery storage (`--energy-battery-kwh`) to buffer intermittent solar/wind
- For solar-only setups, expect chargers to be unpowered at night; pair with grid or battery

**Vehicles repeatedly requeued**

This occurs when energy supply fluctuates (e.g., clouds on solar). Add battery storage to smooth output, or increase grid capacity.

**SIMULTANEOUS training mode crashes or diverges**

Use `SEQUENTIAL` mode (default). See [docs/TrainingComparison.md](docs/TrainingComparison.md) for details.

### Performance Tips

- Use `--quiet` for faster execution
- Use `--no-visualize` if you only need CSV data
- Set `--seed` for reproducible experiments
- Start with short durations (1–2 hours) for testing

---

## 9. Architecture Overview

### Package Structure

```
ev_charge_manager/
├── simulation/     ← orchestrator, parameters, stop conditions
├── highway/        ← road infrastructure, spatial index, vehicle movement
├── vehicle/        ← EV agents, battery, driver behaviour, tracker
├── charging/       ← station infrastructure (areas, chargers, queues)
├── energy/         ← energy management (basic + hierarchical RL)
├── rl/             ← reinforcement learning (PPO, hierarchical trainer)
├── analytics/      ← KPI tracking, blackout/stranding trackers
└── visualization/  ← charts and dashboards
```

### Simulation Flow

```
Simulation (orchestrator)
    └── Environment (tick loop, vehicle lifecycle)
            ├── VehicleGenerator (spawn vehicles)
            ├── Highway (move vehicles, spatial indexing)
            │       └── ChargingArea (queue, charge, sessions)
            │               └── EnergyManager (dispatch power)
            ├── VehicleTracker (state machine per vehicle)
            └── KPITracker (metrics per tick)
```

### Vehicle State Machine

```
DRIVING → APPROACHING → QUEUED → CHARGING → EXITING → COMPLETED
                                                     → STRANDED  (battery = 0)
                              → ABANDONED  (patience exceeded)
```

### Simulation Loop (per 1-minute step)

1. Generate new vehicle arrivals based on traffic pattern
2. Apply energy constraints per charging area (update sources, power down/up chargers, requeue interrupted vehicles)
3. Update all charging areas (process sessions, manage queues)
4. Update queued vehicles (patience decay, abandonment decisions)
5. Update charging vehicles (check completion)
6. Update driving vehicles (physics, energy consumption)
7. Handle station handoffs (vehicles entering stations)
8. Despawn completed/stranded vehicles
9. Record KPIs and check early stop conditions

### Energy Management Flow (per area, per step)

1. **Update sources** — solar output recalculated from time-of-day; wind varies stochastically
2. **Calculate demand** — total power needed for all chargers
3. **Request energy** — sum non-battery sources; if surplus, charge battery; if deficit, discharge battery
4. **Allocate power budget** — occupied chargers (most-complete first) get priority, then available chargers
5. **Power down excess** — chargers outside budget set to `UNPOWERED`; occupied vehicles interrupted and requeued with highest priority
6. **Power up** — previously unpowered chargers restored when budget allows

### Hierarchical RL Architecture

**Macro-RL (Infrastructure, PPO)**
- Agents: one per charging station
- Actions: 8-dim continuous (position, charger count, waiting spots, energy capacities)
- Reward: minimise cost (infrastructure + stranding penalty + blackout penalty)

**Micro-RL (Energy Arbitrage, Actor-Critic)**
- Agents: one per station, embedded in `HierarchicalEnergyManager`
- Actions: battery charge/discharge rate
- Reward: minimise grid cost through time-of-use arbitrage

---

## 10. Extending the Project

| Goal | File to modify |
|------|---------------|
| Add a new traffic pattern | `ev_charge_manager/vehicle/generator.py` → `TemporalDistribution` |
| Add a new energy source | `ev_charge_manager/energy/sources.py` |
| Add a new KPI | `ev_charge_manager/analytics/kpi_definitions.py` → `KPIRecord` |
| Add a new RL reward | `ev_charge_manager/rl/env_configs.py` → `CostParameters` |
| Add a new visualisation | `ev_charge_manager/visualization/` |
| Add a new driver profile | `ev_charge_manager/vehicle/driver_behavior.py` |

---

## Further Reading

- [docs/TrainingComparison.md](docs/TrainingComparison.md) — detailed comparison of RL training modes
- [AGENT.md](AGENT.md) — developer onboarding, key class reference, import paths

---

*EV Charge Manager v2.0*
