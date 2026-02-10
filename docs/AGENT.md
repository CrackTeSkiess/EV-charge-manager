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
EV-charge-manager/
├── CORE SIMULATION
│   ├── Simulation.py              # Orchestrator, results, KPI tracking (~1400 lines)
│   ├── Environment.py             # Vehicle lifecycle, time stepping (~1000 lines)
│   ├── Highway.py                 # Road infra, spatial indexing (~2100 lines)
│   ├── ChargingArea.py            # Station queues & chargers (~1000 lines)
│   ├── Vehicle.py                 # EV physics, battery, driver behavior (~1200 lines)
│   ├── VehicleGenerator.py        # Arrival generation, fleet composition (~600 lines)
│   └── VehicleTracker.py          # Centralized vehicle state (single source of truth)
│
├── ENERGY MANAGEMENT
│   ├── EnergyManager.py           # Grid/solar/wind/battery coordination (~1500 lines)
│   ├── HierarchicalEnergyManager.py # Energy manager with embedded micro-RL
│   └── EnergyManagerAgent.py      # Micro-RL agent for energy arbitrage
│
├── REINFORCEMENT LEARNING
│   ├── PPOTrainer.py              # Multi-agent PPO (macro-level infra)
│   ├── HierarchicalTrainer.py     # Two-level training orchestrator (~1000 lines)
│   ├── ChargingAreaEnv.py         # Gymnasium RL environment (~1500 lines)
│   └── EnergyManagerAgent.py      # Micro-RL actor-critic network
│
├── ANALYTICS & MONITORING
│   ├── KPITracker.py              # KPI monitoring with alert thresholds (~1400 lines)
│   ├── VisualizationTool.py       # Plots, heatmaps, dashboards (~1800 lines)
│   ├── StationDataCollector.py    # Per-station time-series collection
│   ├── TrafficStrandingAnalysis.py # Stranding rate vs. traffic analysis
│   ├── InfrastructureOptimization.py # Pareto front grid search
│   ├── blackout_tracker.py        # Energy shortfall event tracking
│   ├── early_warning_system.py    # RL policy-derived early warnings
│   └── stranded_vehicle_tracker.py # Stranding risk detection
│
├── ENTRY POINTS
│   ├── main.py                    # CLI for simulation runs
│   ├── train.py                   # Multi-agent PPO training
│   ├── evaluate.py                # Evaluate trained models
│   ├── visualize.py               # Visualization & analysis
│   ├── ppo_use_case_example.py    # PPO scenario examples
│   └── hierarchical_use_example.py # Hierarchical RL examples
│
├── DOCUMENTATION
│   ├── USER_MANUAL.md             # Full user guide (17 KB)
│   └── TrainingComparison.md      # Training mode comparison
│
├── MODELS
│   └── models_sequential/         # Pre-trained hierarchical models
│       ├── micro_pretrained/      # Pre-trained micro-RL agents
│       └── training_result.json   # Training metadata
│
└── requirements.txt               # Python dependencies
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| Numerical | NumPy ≥ 1.21 |
| Data | Pandas ≥ 1.3 |
| Visualization | Matplotlib ≥ 3.4, Seaborn ≥ 0.11 |
| Deep Learning | PyTorch (install separately) |
| RL Environment | Gymnasium (install separately) |
| Simulation | Custom discrete-event engine (1-minute time steps) |
| RL Algorithm | Proximal Policy Optimization (PPO), Actor-Critic |
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
# Basic run
python main.py --duration 4 --vehicles-per-hour 80

# With energy management
python main.py --energy-grid-max-kw 200 --energy-solar-peak-kw 100 \
               --energy-battery-kwh 500 --duration 24

# Full renewable mix
python main.py --energy-grid-max-kw 150 --energy-solar-peak-kw 100 \
               --energy-wind-base-kw 80 --energy-battery-kwh 400 --duration 12
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

### Hierarchical RL training (recommended mode: SEQUENTIAL)
```bash
python hierarchical_use_example.py scenario_1
```

### Analysis tools
```bash
python visualize.py               # Analyze training results
python TrafficStrandingAnalysis.py # Traffic vs. stranding rate
python InfrastructureOptimization.py # Find Pareto-optimal infrastructure
```

---

## Architecture Deep Dive

### Simulation Flow

```
main.py / Simulation.py
    └── Environment.py          (tick loop, vehicle lifecycle)
            ├── VehicleGenerator.py  (spawn vehicles)
            ├── Highway.py           (move vehicles, spatial indexing)
            │       └── ChargingArea.py  (queue, charge, track sessions)
            │               └── EnergyManager.py (dispatch power)
            ├── VehicleTracker.py    (state machine per vehicle)
            └── KPITracker.py        (metrics per tick)
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

Two decoupled optimization levels:

**Macro-RL (Infrastructure, PPO)**
- Agents: one per charging station
- Observation: traffic density, queue lengths, stranding events, energy state
- Actions: 8-dim continuous (station position, charger count, waiting spots, energy capacities)
- Reward: minimize cost (infrastructure capex + stranding penalty + blackout penalty)

**Micro-RL (Energy Arbitrage, Actor-Critic)**
- One agent per station, embedded in HierarchicalEnergyManager
- Observation: hour of day, battery SOC, grid price, renewable availability
- Actions: battery charge/discharge rate (continuous)
- Reward: minimize grid energy cost through time-of-use arbitrage
- Grid pricing: off-peak $0.08/kWh (night), peak $0.35/kWh (morning/evening)

### Training Modes (HierarchicalTrainer)

| Mode | Description | Stability | Recommended |
|---|---|---|---|
| `SEQUENTIAL` | Micro → freeze → Macro | High | **Yes** |
| `SIMULTANEOUS` | Micro + Macro together | Low | No |
| `CURRICULUM` | Staged complexity | Highest | For production |

**SEQUENTIAL steps:**
1. Train micro-agents (500 episodes) on fixed average infrastructure
2. Freeze micro policies
3. Train macro-RL (300 episodes) with frozen micro-agents

### Energy Sources (EnergyManager)

| Source | Behavior |
|---|---|
| Grid | Constant power at max capacity |
| Solar | Bell curve 6 AM–8 PM (time-of-day) |
| Wind | Stochastic with mean reversion |
| Battery | Finite storage, charge/discharge limits |

Power allocation priority: occupied chargers (most complete first) → available chargers → unpowered chargers.

### Spatial Indexing (Highway)

Tile-based bucketing with 5 km tiles. Enables O(1) nearest-neighbor queries for station proximity checks instead of scanning all vehicles/stations.

---

## Key Classes Quick Reference

| Class | File | Role |
|---|---|---|
| `Simulation` | Simulation.py | Top-level orchestrator, run entry point |
| `SimulationParameters` | Simulation.py | Complete config dataclass |
| `EarlyStopCondition` | Simulation.py | Stopping criteria (disabled/lenient/default/strict) |
| `Environment` | Environment.py | Ecosystem controller, time stepping |
| `SimulationConfig` | Environment.py | Environment parameters |
| `Highway` | Highway.py | Road with spatial indexing |
| `ChargingArea` | ChargingArea.py | Station with queues and chargers |
| `Charger` | ChargingArea.py | Individual charging point |
| `Vehicle` | Vehicle.py | EV agent with battery physics |
| `Battery` | Vehicle.py | SOC, degradation, thermal model |
| `DriverBehavior` | Vehicle.py | conservative / balanced / aggressive / range-anxious |
| `VehicleGenerator` | VehicleGenerator.py | Fleet composition and arrival patterns |
| `VehicleTracker` | VehicleTracker.py | Centralized state tracking |
| `EnergyManager` | EnergyManager.py | Multi-source power dispatch |
| `HierarchicalEnergyManager` | HierarchicalEnergyManager.py | Energy manager + micro-RL |
| `EnergyManagerAgent` | EnergyManagerAgent.py | Micro-RL actor-critic |
| `MultiAgentPPO` | PPOTrainer.py | CTDE PPO trainer |
| `HierarchicalTrainer` | HierarchicalTrainer.py | Dual-level training orchestrator |
| `MultiAgentChargingEnv` | ChargingAreaEnv.py | Gymnasium RL environment |
| `KPITracker` | KPITracker.py | Metrics, alerting, aggregation |
| `VisualizationTool` | VisualizationTool.py | Charts, dashboards, heatmaps |

---

## KPIs Tracked

**Required (always tracked)**
- `cars_quit_waiting` — abandonments per step
- `cars_zero_battery` — strandeds per step

**Performance**
- Wait time (avg, max, p95)
- Charger utilization %
- Queue occupancy rates
- Service level / abandonment rate

**Economic**
- Estimated revenue ($0.40/kWh + connection fees)
- Customer satisfaction score

**Energy** (when energy management enabled)
- Available vs. demand power
- Curtailment, unpowered charger counts
- Vehicles requeued due to power loss

Alert levels: WARNING → CRITICAL → EMERGENCY

---

## Pre-trained Models

Located in `models_sequential/`:
- 3 stations at positions [75.4, 148.2, 226.9] km on a ~300 km highway
- 2 chargers per station, 10–12 waiting spots per station
- Trained: 500 micro episodes + 300 macro episodes
- Mode: SEQUENTIAL

---

## No Test Suite

There is no formal test suite. Validation approaches:
- Run `python main.py demo` to verify basic simulation works
- Use example scripts: `ppo_use_case_example.py`, `hierarchical_use_example.py`
- Load pre-trained models from `models_sequential/` and evaluate

---

## Common Gotchas

1. **PyTorch / Gymnasium not in requirements.txt** — must be installed manually for any RL functionality.
2. **`SIMULTANEOUS` training mode is unstable** — use `SEQUENTIAL` unless you have a specific reason.
3. **`VehicleTracker` is the single source of truth** — do not read vehicle state from multiple places; always go through the tracker.
4. **Spatial indexing tiles are 5 km** — changing `Highway` tile size affects query performance significantly.
5. **EarlyStopCondition defaults** — the default stop condition may halt short simulations early; use `EarlyStopCondition.disabled()` for deterministic runs.
6. **Energy manager requeues vehicles** — when grid power drops, vehicles may be moved back to queue; this is intentional behavior, not a bug.

---

## Extending the Project

| Goal | Where to start |
|---|---|
| Add a new traffic pattern | `VehicleGenerator.py` — `TemporalDistribution` enum + generation logic |
| Add a new energy source | `EnergyManager.py` — add config dataclass + dispatch logic |
| Add a new KPI | `KPITracker.py` — add field to `KPIRecord`, update aggregation |
| Add a new RL reward signal | `ChargingAreaEnv.py` — `CostParameters` + reward calculation |
| Add a new visualization | `VisualizationTool.py` — add method returning a matplotlib figure |
| Change training curriculum | `HierarchicalTrainer.py` — `TrainingConfig` + phase logic |
| New driver profile | `Vehicle.py` — `DriverBehavior` dataclass |

---

## Further Reading

- [USER_MANUAL.md](USER_MANUAL.md) — Full parameter reference, configuration examples, troubleshooting
- [TrainingComparison.md](TrainingComparison.md) — Detailed comparison of RL training modes
