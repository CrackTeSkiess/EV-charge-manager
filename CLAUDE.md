# CLAUDE.md — EV Charge Manager

AI assistant guide for the **EV Charge Manager** codebase. Read this before making any changes.

---

## Project Overview

EV Charge Manager is a discrete-event simulation framework for highway EV charging infrastructure, with hierarchical multi-agent reinforcement learning (RL) for autonomous optimisation. The simulation runs at 1-minute time steps.

**Two primary use cases:**
1. **Pure simulation** — study traffic, charging behaviour, KPIs without RL
2. **RL training** — learn optimal infrastructure placement and energy arbitrage policies

---

## Repository Structure

```
EV-charge-manager/
├── ev_charge_manager/          ← main importable package
│   ├── simulation/             ← orchestrator, parameters, stop conditions
│   ├── highway/                ← road, spatial index (5 km tiles), vehicle movement
│   ├── vehicle/                ← EV physics, battery, driver behaviour, generator, tracker
│   ├── charging/               ← station infrastructure (areas, chargers, queues)
│   ├── energy/                 ← energy sources, manager, hierarchical RL manager
│   ├── rl/                     ← Gymnasium env, PPO trainer, hierarchical trainer
│   ├── analytics/              ← KPI tracking, blackout/stranding trackers
│   ├── visualization/          ← charts and dashboards
│   ├── utils/                  ← RunDirectory (per-run output management)
│   ├── sumo/                   ← SUMO/TraCI integration (optional, needs SUMO installed)
│   └── data/                   ← real-world data providers (PVGIS weather, BASt traffic)
│
├── tools/                      ← standalone analysis scripts
│   ├── traffic_stranding.py    ← traffic vs stranding rate analysis
│   └── infrastructure_optimizer.py ← Pareto-front grid search
│
├── scripts/                    ← validation scripts
│   ├── download_real_data.py
│   ├── validate_real_world.py  ← full-year RL validation vs rule-based baseline
│   └── validate_sumo.py        ← SUMO traffic simulation validation
│
├── examples/                   ← runnable scenario scripts
│   ├── ppo_example.py
│   └── hierarchical_example.py
│
├── models/                     ← saved model weights (gitignored)
│   ├── sequential/             ← pre-trained micro + macro agents
│   └── hierarchical/
│
├── main.py                     ← simulation CLI entry point
├── train.py                    ← multi-agent PPO training
├── hierarchical.py             ← hierarchical dual-RL training
├── evaluate.py                 ← model evaluation
├── visualize.py                ← visualisation utilities
├── ui.py                       ← interactive terminal launcher (ANSI UI)
└── requirements.txt
```

**Gitignored output directories:** `simulation_output/`, `models/`, `validation_output/`

---

## Installation

```bash
pip install -r requirements.txt

# Required for ALL RL features (not in requirements.txt):
pip install torch gymnasium
```

**Python 3.8+ required.** Core deps: NumPy ≥ 1.21, Pandas ≥ 1.3, Matplotlib ≥ 3.4, Seaborn ≥ 0.11, requests ≥ 2.25.

---

## Development Commands

### Simulation
```bash
python main.py demo                                    # 2-hour rush-hour demo
python main.py --duration 8 --vehicles-per-hour 100 --traffic-pattern RUSH_HOUR
python main.py --energy-grid-max-kw 200 --energy-solar-peak-kw 100 \
               --energy-battery-kwh 500 --duration 24
python ui.py                                           # interactive terminal launcher
```

### RL Training
```bash
# Multi-agent PPO
python train.py --n-agents 3 --episodes 1000 --save-dir ./models

# Hierarchical (RECOMMENDED: sequential mode)
python hierarchical.py                                 # defaults: 3 agents, sequential
python hierarchical.py --n-agents 5 --micro-episodes 2000
python hierarchical.py --robust                        # high-episode preset
python hierarchical.py --mode curriculum --robust
python hierarchical.py --mode frozen_micro --micro-load ./models/hierarchical/micro_pretrained
```

### Evaluation
```bash
python evaluate.py --model ./models/ppo_agent.pt --episodes 20 --render
```

### Visualisation
```bash
python visualize.py training --save-dir ./models/hierarchical
python visualize.py compare  --dirs ./models/sequential ./models/curriculum
```

### Analysis Tools
```bash
python tools/traffic_stranding.py
python tools/infrastructure_optimizer.py
```

### Real-world Validation
```bash
python scripts/download_real_data.py
python scripts/validate_real_world.py --model-dir models/hierarchical --days 365
python scripts/validate_real_world.py --days 7    # quick test
python scripts/validate_sumo.py                   # needs SUMO installed
```

---

## CLI Reference

| Flag group | Flags |
|---|---|
| Highway | `--highway-length` (km), `--num-stations`, `--chargers-per-station`, `--waiting-spots` |
| Traffic | `--vehicles-per-hour`, `--traffic-pattern`, `--duration` |
| Traffic patterns | `UNIFORM`, `RANDOM_POISSON`, `RUSH_HOUR`, `SINGLE_PEAK`, `PERIODIC`, `BURSTY` |
| Energy | `--energy-grid-max-kw`, `--energy-solar-peak-kw`, `--energy-wind-base-kw`, `--energy-battery-kwh` |
| Output | `--output-dir`, `--run-name`, `--no-visualize`, `--quiet`, `--seed` |
| Trackers | `--enable-station-tracker`, `--enable-vehicle-tracker` |

---

## Architecture Overview

### Simulation Flow (1 tick = 1 minute)

```
main.py → Simulation → Environment (tick loop)
                          ├── VehicleGenerator   (spawn new vehicles)
                          ├── Highway            (move vehicles, spatial indexing)
                          │     └── ChargingArea (queue, charge, sessions)
                          │           └── EnergyManager (dispatch power)
                          ├── VehicleTracker     (state machine per vehicle)
                          └── KPITracker         (record metrics per tick)
```

### Vehicle State Machine

```
DRIVING → APPROACHING → QUEUED → CHARGING → EXITING → COMPLETED
                                                     → STRANDED  (battery = 0)
                              → ABANDONED  (patience exceeded)
```

### Hierarchical RL

| Level | Class | Algorithm | Actions |
|---|---|---|---|
| **Macro** (infrastructure) | `MultiAgentPPO` + `MultiAgentChargingEnv` | PPO, CTDE | Position, charger count, waiting spots, energy capacities |
| **Micro** (energy arbitrage) | `EnergyManagerAgent` + `HierarchicalEnergyManager` | Actor-Critic | Battery charge/discharge rate |

**CTDE** = Centralized Training, Decentralized Execution (one agent per charging station).

### Training Modes

| Mode | Stability | Recommended |
|---|---|---|
| `sequential` | High | **Yes — default** |
| `simultaneous` | Low | No |
| `curriculum` | Highest | Production use |
| `frozen_micro` | High | Re-use pre-trained micro |

---

## Output Directory Structure

Every run writes to a timestamped folder via `RunDirectory`:

```
simulation_output/
  2026-02-17_143052_my-run-name/
    metadata.json        ← CLI args + timestamp
    data/                ← KPIs, summary stats, final state JSON/CSV
    plots/               ← PNG visualisations
    reports/             ← stranding reports

models/hierarchical/
  run_config.json
  results.json / training_result.json
  training_history.jsonl    ← one JSON per PPO update (streaming)
  micro_history.json
  micro_final_day.json
  micro_pretrained/         ← reusable micro-agent weights
  training_plots/           ← auto-generated PNGs
```

Use `--run-name` to give human-readable names; they become URL-safe slugs.

---

## Key Import Patterns

```python
# Core simulation
from ev_charge_manager.simulation import Simulation, SimulationParameters, EarlyStopCondition
from ev_charge_manager.simulation import SimulationConfig, Environment

# Highway + vehicles
from ev_charge_manager.highway import Highway, HighwaySegment
from ev_charge_manager.vehicle import Vehicle, Battery, DriverBehavior, VehicleTracker

# ⚠ VehicleGenerator MUST be imported from its sub-module directly (circular import risk)
from ev_charge_manager.vehicle.generator import VehicleGenerator, GeneratorConfig, TemporalDistribution

# Charging infrastructure
from ev_charge_manager.charging import ChargingArea, ChargerStatus, ChargerType

# Energy management
from ev_charge_manager.energy import EnergyManager, EnergyManagerConfig
from ev_charge_manager.energy import GridSourceConfig, SolarSourceConfig, WindSourceConfig, BatteryStorageConfig

# RL (requires torch + gymnasium)
from ev_charge_manager.rl import MultiAgentPPO, HierarchicalTrainer, TrainingMode
from ev_charge_manager.energy import HierarchicalEnergyManager, EnergyManagerAgent, GridPricingSchedule

# Analytics
from ev_charge_manager.analytics import KPITracker, StationDataCollector, BlackoutTracker, StrandedVehicleTracker

# Visualisation
from ev_charge_manager.visualization import VisualizationTool, ChartConfig
from ev_charge_manager.visualization.training_plots import TrainingVisualizer

# Utilities
from ev_charge_manager.utils import RunDirectory

# Real-world data (optional)
from ev_charge_manager.data import RealWeatherProvider, RealTrafficProvider

# SUMO integration (requires SUMO installed + SUMO_HOME set)
from ev_charge_manager.sumo import check_sumo_available, require_sumo
```

---

## Critical Gotchas

1. **`VehicleGenerator` circular import** — always import from `ev_charge_manager.vehicle.generator`, never from `ev_charge_manager.vehicle`. The package `__init__.py` intentionally excludes it.

2. **PyTorch/Gymnasium not in requirements.txt** — `pip install torch gymnasium` is required for any RL functionality. The package handles their absence gracefully in non-RL paths.

3. **`SIMULTANEOUS` training mode is unstable** — always prefer `SEQUENTIAL` unless you have a specific reason.

4. **`VehicleTracker` is the single source of truth** — always read vehicle state through the tracker, not directly from `Vehicle` objects.

5. **`EarlyStopCondition` halts short simulations** — use `EarlyStopCondition.disabled()` for deterministic/reproducible runs.

6. **Energy manager re-queues vehicles** — when grid power drops, charging vehicles may be moved back to queue intentionally. This is expected behaviour.

7. **Spatial tile size is 5 km** — changing tile size in `Highway` significantly affects query performance across the whole simulation.

8. **Energy manager lazy import** — `EnergyManager._create_hierarchical_manager()` uses a lazy import to avoid the circular import between `EnergyManager` and `HierarchicalEnergyManager`. Do not convert this back to a top-level import.

9. **SUMO integration** — requires SUMO installed and `SUMO_HOME` env var set. Call `check_sumo_available()` before using any `ev_charge_manager.sumo` features. On Ubuntu: `sudo apt-get install -y sumo sumo-tools && export SUMO_HOME=/usr/share/sumo`.

10. **Independent energy configs per station** — use `copy.deepcopy()` when creating per-station `EnergyManagerConfig` instances to avoid shared reference mutations (see `main.py`).

---

## KPIs Tracked

| Category | KPIs |
|---|---|
| **Required** (always) | `cars_quit_waiting`, `cars_zero_battery` |
| **Performance** | avg/max/p95 wait time, charger utilisation %, queue occupancy, service level |
| **Economic** | revenue ($0.40/kWh + connection fees), customer satisfaction score |
| **Energy** *(when enabled)* | available vs demand power, curtailment, unpowered charger count, vehicles requeued |

Alert levels: `WARNING` → `CRITICAL` → `EMERGENCY`

---

## Extending the Project

| Goal | File to edit |
|---|---|
| New traffic pattern | `ev_charge_manager/vehicle/generator.py` → `TemporalDistribution` enum |
| New energy source | `ev_charge_manager/energy/sources.py` + `manager.py` |
| New KPI | `ev_charge_manager/analytics/kpi_definitions.py` → `KPIRecord` |
| New RL reward term | `ev_charge_manager/rl/env_configs.py` → `CostParameters` |
| New visualisation | `ev_charge_manager/visualization/` |
| New driver profile | `ev_charge_manager/vehicle/driver_behavior.py` |
| New stop condition | `ev_charge_manager/simulation/stop_conditions.py` |

---

## Pre-trained Models

Located in `models/sequential/`:
- 3 stations at positions [75.4, 148.2, 226.9] km on a 300 km highway
- 2 chargers per station, 10–12 waiting spots
- Trained: 500 micro + 300 macro episodes (SEQUENTIAL mode)

---

## Git Workflow

Active development branch: `claude/add-claude-documentation-GNfaU`

Push with:
```bash
git push -u origin <branch-name>
```

Branch naming: branches must start with `claude/` and end with the session ID.

---

## Further Reading

- [README.md](README.md) — quick start, overview
- [AGENT.md](AGENT.md) — detailed architecture deep-dive, full class reference table
- [docs/USER_MANUAL.md](docs/USER_MANUAL.md) — full parameter reference, configuration examples, troubleshooting
- [docs/TrainingComparison.md](docs/TrainingComparison.md) — detailed comparison of RL training modes
