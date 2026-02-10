# EV Charge Manager — Agent Onboarding Guide

## Project Overview

**EV Charge Manager** is a discrete-event simulation framework for electric vehicle (EV) charging infrastructure on highways. It combines physics-based simulation with hierarchical multi-agent reinforcement learning (RL) for dual-level optimization:

- **Infrastructure Planning**: Optimal station placement, charger counts, waiting spots
- **Energy Management**: Renewable energy mixes, battery storage arbitrage
- **Performance Analysis**: Bottleneck detection, service quality, stranding risk
- **Policy Evaluation**: Pricing strategies, queue management, traffic patterns

---

## Repository Layout

```
EV-charge-manager/                  ← root: entry points + docs only
│
├── ev_charge_manager/              ← main importable package
│   ├── simulation/                 ← orchestrator, parameters, stop conditions
│   │   ├── simulation.py           ↳ Simulation class (~1260 lines)
│   │   ├── environment.py          ↳ Environment, SimulationConfig (~760 lines)
│   │   ├── parameters.py           ↳ re-exports SimulationParameters, SimulationResult
│   │   └── stop_conditions.py      ↳ re-exports StopReason, EarlyStopCondition
│   │
│   ├── highway/                    ← road infra, spatial index, vehicle movement
│   │   ├── highway.py              ↳ Highway class (~1730 lines)
│   │   ├── segment.py              ↳ re-exports HighwaySegment dataclass
│   │   ├── spatial_index.py        ↳ tile-based spatial index (5 km tiles)
│   │   └── vehicle_movement.py     ↳ position updates & station handoffs
│   │
│   ├── vehicle/                    ← EV agent, battery, driver behaviour, tracker
│   │   ├── vehicle.py              ↳ Vehicle, Battery, DriverBehavior, VehicleState (~1030 lines)
│   │   ├── battery.py              ↳ re-exports Battery
│   │   ├── driver_behavior.py      ↳ re-exports DriverBehavior
│   │   ├── physics.py              ↳ re-exports Vehicle (consumption model)
│   │   ├── states.py               ↳ re-exports VehicleState
│   │   ├── generator.py            ↳ VehicleGenerator, GeneratorConfig, TemporalDistribution
│   │   │                             ⚠ import directly – not via vehicle package (circular)
│   │   └── tracker.py              ↳ VehicleTracker, TrackedState, VehicleRecord, StepMetrics
│   │
│   ├── charging/                   ← station infrastructure
│   │   ├── area.py                 ↳ ChargingArea, Charger, ChargingSession, QueueEntry, enums (~840 lines)
│   │   ├── charger.py              ↳ re-exports Charger, ChargingSession
│   │   ├── queue.py                ↳ re-exports QueueEntry
│   │   └── enums.py                ↳ re-exports ChargerStatus, ChargerType
│   │
│   ├── energy/                     ← energy management (core always available; RL needs torch)
│   │   ├── manager.py              ↳ EnergyManager + all source configs (~575 lines)
│   │   ├── sources.py              ↳ re-exports source config dataclasses
│   │   ├── hierarchical_manager.py ↳ HierarchicalEnergyManager  [requires torch]
│   │   └── agent.py                ↳ EnergyManagerAgent, GridPricingSchedule  [requires torch]
│   │
│   ├── rl/                         ← reinforcement learning  [all require torch + gymnasium]
│   │   ├── environment.py          ↳ MultiAgentChargingEnv (Gymnasium)
│   │   ├── env_configs.py          ↳ WeatherProfile, CostParameters, ChargingAreaAgent
│   │   ├── ppo_trainer.py          ↳ MultiAgentPPO (CTDE)
│   │   ├── hierarchical_trainer.py ↳ HierarchicalTrainer, MicroRLTrainer, TrainingMode, TrainingConfig
│   │   ├── micro_trainer.py        ↳ re-exports MicroRLTrainer
│   │   └── training_config.py      ↳ re-exports TrainingMode, TrainingConfig
│   │
│   ├── analytics/                  ← KPI tracking & event trackers
│   │   ├── kpi_tracker.py          ↳ KPITracker, KPIAlertLevel, KPIThreshold, KPIRecord (~710 lines)
│   │   ├── kpi_definitions.py      ↳ re-exports KPIRecord, KPIThreshold, KPIAlertLevel
│   │   ├── station_collector.py    ↳ StationDataCollector
│   │   ├── blackout_tracker.py     ↳ BlackoutTracker, BlackoutType, EnergyShortfallEvent
│   │   ├── stranded_tracker.py     ↳ StrandedVehicleTracker, VehicleStatus
│   │   └── early_warning.py        ↳ EarlyWarningExtractor  [requires torch + ppo_agent]
│   │
│   └── visualization/              ← charts and dashboards
│       ├── tool.py                 ↳ VisualizationTool, ChartConfig (~1570 lines)
│       ├── training_plots.py       ↳ TrainingVisualizer — learning curves, config evolution, battery strategy
│       ├── config.py               ↳ re-exports ChartConfig
│       ├── time_series.py          ↳ re-exports VisualizationTool (time-series plots)
│       ├── station_plots.py        ↳ re-exports VisualizationTool (per-station charts)
│       ├── stranding_plots.py      ↳ re-exports VisualizationTool (stranding analysis)
│       └── dashboard.py            ↳ re-exports VisualizationTool (6-panel overview)
│
├── tools/                          ← standalone analysis & optimisation scripts
│   ├── traffic_stranding.py        ↳ TrafficStrandingAnalyzer (traffic vs stranding rate)
│   └── infrastructure_optimizer.py ↳ InfrastructureOptimizer (Pareto-front grid search)
│
├── examples/                       ← runnable scenario scripts
│   ├── ppo_example.py              ↳ PPO training scenarios
│   └── hierarchical_example.py     ↳ hierarchical RL training scenarios
│
├── docs/                           ← documentation
│   ├── USER_MANUAL.md
│   ├── TrainingComparison.md
│   └── AGENT.md
│
├── models/                         ← saved model weights
│   └── sequential/                 ↳ pre-trained micro + macro agents
│
├── main.py                         ← CLI entry point  (python main.py --help)
├── train.py                        ← multi-agent PPO training
├── hierarchical.py                 ← hierarchical dual-RL training (micro + macro)
├── evaluate.py                     ← model evaluation
├── visualize.py                    ← visualisation & analysis
├── AGENT.md                        ← this file
├── USER_MANUAL.md
├── TrainingComparison.md
├── README.md
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Numerical | NumPy ≥ 1.21 |
| Data | Pandas ≥ 1.3 |
| Visualization | Matplotlib ≥ 3.4, Seaborn ≥ 0.11 |
| Deep Learning | PyTorch *(install separately — required for RL)* |
| RL Environment | Gymnasium *(install separately — required for RL)* |
| Simulation | Custom discrete-event engine (1-minute time steps) |
| RL Algorithm | PPO, Actor-Critic |
| Multi-Agent | CTDE — Centralized Training, Decentralized Execution |

---

## Installation

```bash
cd EV-charge-manager
pip install -r requirements.txt

# Required for RL features (not in requirements.txt):
pip install torch gymnasium
```

---

## Running the Project

### Quick demo (2-hour rush-hour simulation)
```bash
python main.py demo
```

### Custom simulation
```bash
python main.py --duration 4 --vehicles-per-hour 80

python main.py --energy-grid-max-kw 200 --energy-solar-peak-kw 100 \
               --energy-battery-kwh 500 --duration 24
```

### CLI flags reference
```
Highway:    --highway-length (km), --num-stations, --chargers-per-station, --waiting-spots
Traffic:    --vehicles-per-hour, --traffic-pattern, --duration (hours)
            Patterns: UNIFORM | POISSON | RUSH_HOUR | SINGLE_PEAK | PERIODIC | BURSTY
Energy:     --energy-grid-max-kw, --energy-solar-peak-kw, --energy-wind-base-kw, --energy-battery-kwh
Output:     --output-dir, --no-visualize, --quiet, --seed
Trackers:   --enable-station-tracker, --enable-vehicle-tracker
```

### Multi-agent PPO training
```bash
python train.py --n-agents 3 --episodes 1000 --highway-length 300 --save-dir ./models
python evaluate.py --model ./models/ppo_agent.pt --episodes 20 --render
```

### Hierarchical dual-RL training
```bash
# RECOMMENDED – sequential: micro first → macro (most stable)
python hierarchical.py                                         # 3 agents, defaults
python hierarchical.py --n-agents 5 --micro-episodes 2000     # custom scale
python hierarchical.py --robust                                # high-episode preset

# Simultaneous (both levels train together – faster, less stable)
python hierarchical.py --mode simultaneous

# Curriculum (progressive 1-day → 3-day → 7-day complexity)
python hierarchical.py --mode curriculum --robust

# Frozen-micro – load pre-trained micro-agents, train macro only
python hierarchical.py --mode frozen_micro \
    --micro-load ./models/hierarchical/micro_pretrained
```

All outputs land in `--save-dir` (default `./models/hierarchical/`):
- `run_config.json` — CLI parameters used
- `results.json` / `training_result.json` — final training metrics
- `training_history.jsonl` — one JSON record per PPO update (streaming)
- `micro_history.json` — per-station episode rewards and grid costs
- `micro_final_day.json` — hourly breakdown from the last micro episode
- `micro_pretrained/` — trained energy-arbitrage agents (reusable)
- `training_plots/` — auto-generated PNG plots (pass `--no-visualize` to skip)

Regenerate plots at any time:
```bash
python visualize.py training --save-dir ./models/hierarchical
python visualize.py compare  --dirs ./models/sequential ./models/curriculum
```

### Analysis tools
```bash
python tools/traffic_stranding.py
python tools/infrastructure_optimizer.py
```

---

## Using the New Package

```python
from ev_charge_manager.simulation import Simulation, SimulationParameters, EarlyStopCondition
from ev_charge_manager.simulation import SimulationConfig, Environment
from ev_charge_manager.highway import Highway, HighwaySegment
from ev_charge_manager.vehicle import Vehicle, Battery, DriverBehavior, VehicleTracker
from ev_charge_manager.charging import ChargingArea, ChargerStatus, ChargerType
from ev_charge_manager.energy import EnergyManager, EnergyManagerConfig, GridSourceConfig
from ev_charge_manager.analytics import KPITracker, StationDataCollector, BlackoutTracker
from ev_charge_manager.visualization import VisualizationTool, ChartConfig

# VehicleGenerator must be imported from its sub-module directly —
# NOT from ev_charge_manager.vehicle — it has a top-level dependency on
# Environment which would create a circular import through the package.
from ev_charge_manager.vehicle.generator import VehicleGenerator, GeneratorConfig, TemporalDistribution

# RL + hierarchical energy (requires torch + gymnasium)
from ev_charge_manager.rl import MultiAgentPPO, HierarchicalTrainer, TrainingMode
from ev_charge_manager.energy import HierarchicalEnergyManager, EnergyManagerAgent
```

---

## Architecture Deep Dive

### Simulation Flow

```
main.py
    └── ev_charge_manager.simulation.simulation     (Simulation orchestrator)
            └── ev_charge_manager.simulation.environment  (tick loop, vehicle lifecycle)
                    ├── ev_charge_manager.vehicle.generator   (spawn vehicles)
                    ├── ev_charge_manager.highway.highway     (move vehicles, spatial indexing)
                    │       └── ev_charge_manager.charging.area  (queue, charge, sessions)
                    │               └── ev_charge_manager.energy.manager (dispatch power)
                    ├── ev_charge_manager.vehicle.tracker     (state machine per vehicle)
                    └── ev_charge_manager.analytics.kpi_tracker  (metrics per tick)
```

Each simulation tick (1 minute):
1. Generate new vehicles (VehicleGenerator)
2. Update vehicle positions and battery state (Highway)
3. Route vehicles needing charge to nearest available station
4. Dispatch power across chargers (EnergyManager)
5. Update queues: serve, abandon, complete charging (ChargingArea)
6. Record KPIs and check early stop conditions

### Vehicle State Machine
```
DRIVING → APPROACHING → QUEUED → CHARGING → EXITING → COMPLETED
                                                     → STRANDED  (battery = 0)
                              → ABANDONED  (patience exceeded)
```

### Hierarchical RL Architecture

**Macro-RL (Infrastructure, PPO)**
- Agents: one per charging station
- Actions: 8-dim continuous (position, charger count, waiting spots, energy capacities)
- Reward: minimize cost (infrastructure + stranding penalty + blackout penalty)

**Micro-RL (Energy Arbitrage, Actor-Critic)**
- One agent per station, embedded in HierarchicalEnergyManager
- Actions: battery charge/discharge rate
- Reward: minimize grid cost through time-of-use arbitrage

### Training Modes

| Mode | Stability | Time | Recommended |
|---|---|---|---|
| `SEQUENTIAL` | High | 4–6 h | **Yes** |
| `SIMULTANEOUS` | Low | 8–12 h | No |
| `CURRICULUM` | Highest | 12–24 h | For production |

---

## Key Classes Quick Reference

| Class | New import path | Original file |
|---|---|---|
| `Simulation` | `ev_charge_manager.simulation` | Simulation.py |
| `SimulationParameters` | `ev_charge_manager.simulation.parameters` | Simulation.py |
| `EarlyStopCondition` | `ev_charge_manager.simulation.stop_conditions` | Simulation.py |
| `Environment` | `ev_charge_manager.simulation.environment` | Environment.py |
| `Highway` | `ev_charge_manager.highway` | Highway.py |
| `HighwaySegment` | `ev_charge_manager.highway.segment` | Highway.py |
| `Vehicle` | `ev_charge_manager.vehicle` | Vehicle.py |
| `Battery` | `ev_charge_manager.vehicle.battery` | Vehicle.py |
| `DriverBehavior` | `ev_charge_manager.vehicle.driver_behavior` | Vehicle.py |
| `VehicleState` | `ev_charge_manager.vehicle.states` | Vehicle.py |
| `VehicleGenerator` | `ev_charge_manager.vehicle.generator` | VehicleGenerator.py |
| `VehicleTracker` | `ev_charge_manager.vehicle.tracker` | VehicleTracker.py |
| `ChargingArea` | `ev_charge_manager.charging` | ChargingArea.py |
| `Charger` | `ev_charge_manager.charging.charger` | ChargingArea.py |
| `ChargerStatus` | `ev_charge_manager.charging.enums` | ChargingArea.py |
| `EnergyManager` | `ev_charge_manager.energy` | EnergyManager.py |
| `HierarchicalEnergyManager` | `ev_charge_manager.energy` | HierarchicalEnergyManager.py |
| `EnergyManagerAgent` | `ev_charge_manager.energy` | EnergyManagerAgent.py |
| `MultiAgentChargingEnv` | `ev_charge_manager.rl` | ChargingAreaEnv.py |
| `MultiAgentPPO` | `ev_charge_manager.rl` | PPOTrainer.py |
| `HierarchicalTrainer` | `ev_charge_manager.rl` | HierarchicalTrainer.py |
| `KPITracker` | `ev_charge_manager.analytics` | KPITracker.py |
| `StationDataCollector` | `ev_charge_manager.analytics` | StationDataCollector.py |
| `BlackoutTracker` | `ev_charge_manager.analytics` | blackout_tracker.py |
| `StrandedVehicleTracker` | `ev_charge_manager.analytics` | stranded_vehicle_tracker.py |
| `EarlyWarningExtractor` | `ev_charge_manager.analytics` | early_warning_system.py |
| `VisualizationTool` | `ev_charge_manager.visualization` | VisualizationTool.py |
| `TrainingVisualizer` | `ev_charge_manager.visualization` | visualization/training_plots.py |

---

## KPIs Tracked

**Required (always tracked)**
- `cars_quit_waiting` — abandonments per step
- `cars_zero_battery` — strandeds per step

**Performance** — wait time (avg/max/p95), charger utilisation %, queue occupancy, service level

**Economic** — estimated revenue ($0.40/kWh + connection fees), customer satisfaction score

**Energy** *(when energy management enabled)* — available vs. demand power, curtailment, unpowered charger counts, vehicles requeued

Alert levels: WARNING → CRITICAL → EMERGENCY

---

## Pre-trained Models

Located in `models/sequential/` (mirrored from `models_sequential/`):
- 3 stations at positions [75.4, 148.2, 226.9] km
- 2 chargers per station, 10–12 waiting spots
- Trained: 500 micro + 300 macro episodes (SEQUENTIAL mode)

---

## Known Issues Fixed During Reorganisation

- **Circular import in EnergyManager.py** — `EnergyManager` imported `HierarchicalEnergyManager` at module level while `HierarchicalEnergyManager` imports `EnergyManager`, breaking `main.py`. Fixed by converting to a lazy import inside `_create_hierarchical_manager()`. This was a pre-existing bug that caused `python main.py` to fail.

---

## Common Gotchas

1. **PyTorch / Gymnasium not in requirements.txt** — install manually for any RL functionality (`pip install torch gymnasium`). The package handles their absence gracefully.
2. **`SIMULTANEOUS` training mode is unstable** — use `SEQUENTIAL`.
3. **`VehicleTracker` is the single source of truth** — always read vehicle state through the tracker.
4. **Spatial indexing tiles are 5 km** — changing `Highway` tile size affects query performance significantly.
5. **`EarlyStopCondition` defaults** — may halt short simulations early; use `EarlyStopCondition.disabled()` for deterministic runs.
6. **Energy manager requeues vehicles** — when grid power drops, vehicles may be moved back to queue; this is intentional.

---

## Extending the Project

| Goal | New location | Original file |
|---|---|---|
| Add a new traffic pattern | `ev_charge_manager/vehicle/generator.py` → `TemporalDistribution` | VehicleGenerator.py |
| Add a new energy source | `ev_charge_manager/energy/sources.py` | EnergyManager.py |
| Add a new KPI | `ev_charge_manager/analytics/kpi_definitions.py` → `KPIRecord` | KPITracker.py |
| Add a new RL reward | `ev_charge_manager/rl/env_configs.py` → `CostParameters` | ChargingAreaEnv.py |
| Add a new visualisation | `ev_charge_manager/visualization/` | VisualizationTool.py |
| New driver profile | `ev_charge_manager/vehicle/driver_behavior.py` | Vehicle.py |

---

## Further Reading

- [docs/USER_MANUAL.md](docs/USER_MANUAL.md) — full parameter reference, configuration examples, troubleshooting
- [docs/TrainingComparison.md](docs/TrainingComparison.md) — detailed comparison of RL training modes
