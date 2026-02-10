# EV Charge Manager

A discrete-event simulation framework for electric vehicle (EV) charging infrastructure on highways, with hierarchical multi-agent reinforcement learning for autonomous optimisation.

## Overview

EV Charge Manager simulates vehicle behaviour, charging station operations, and driver decision-making to help analyse and optimise highway charging infrastructure. It supports both pure simulation studies and RL-driven policy search.

**Core capabilities:**
- Physics-based EV energy consumption and battery modelling
- Configurable traffic patterns (rush hour, Poisson, bursty, etc.)
- Per-station energy management: grid, solar, wind, and battery storage
- Hierarchical multi-agent RL (PPO + Actor-Critic) for infrastructure and energy optimisation
- Comprehensive KPI tracking, analytics, and visualisation
- CLI entry point plus a fully importable Python package

## Quick Start

```bash
# Install core dependencies
pip install -r requirements.txt

# Run a 2-hour demo simulation
python main.py demo

# Custom simulation
python main.py --vehicles-per-hour 100 --duration 8 --traffic-pattern RUSH_HOUR

# With renewable energy constraints
python main.py --energy-grid-max-kw 200 --energy-solar-peak-kw 100 \
               --energy-battery-kwh 500 --duration 24
```

## Installation

```bash
git clone <repo-url>
cd EV-charge-manager
pip install -r requirements.txt

# Optional: required for RL features
pip install torch gymnasium
```

**Requirements:** Python 3.8+, NumPy ≥ 1.21, Pandas ≥ 1.3, Matplotlib ≥ 3.4, Seaborn ≥ 0.11

## Reinforcement Learning

### Multi-Agent PPO

```bash
python train.py --n-agents 3 --episodes 1000 --save-dir ./models
python evaluate.py --model ./models/ppo_agent.pt --episodes 20 --render
```

### Hierarchical Dual-RL (recommended)

Trains macro-agents (infrastructure placement, PPO) and micro-agents (energy arbitrage, Actor-Critic) together:

```bash
# Sequential training — micro first, then macro (most stable)
python hierarchical.py --n-agents 3

# Load pre-trained micro-agents and train macro only
python hierarchical.py --mode frozen_micro \
    --micro-load ./models/hierarchical/micro_pretrained
```

Pre-trained models are available in `models/sequential/`.

## Analysis Tools

```bash
python tools/traffic_stranding.py        # traffic vs stranding rate
python tools/infrastructure_optimizer.py # Pareto-front grid search
```

## Package Structure

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

### Programmatic Usage

```python
from ev_charge_manager.simulation import Simulation, SimulationParameters, EarlyStopCondition
from ev_charge_manager.vehicle.generator import VehicleGenerator, TemporalDistribution
from ev_charge_manager.energy import EnergyManagerConfig, GridSourceConfig, SolarSourceConfig

params = SimulationParameters(
    highway_length_km=300.0,
    num_charging_areas=3,
    chargers_per_area=4,
    vehicles_per_hour=80.0,
    temporal_distribution=TemporalDistribution.RUSH_HOUR,
    simulation_duration_hours=24.0,
    random_seed=42,
)
result = Simulation(params).run(verbose=True)
```

> `VehicleGenerator` must be imported from `ev_charge_manager.vehicle.generator` directly (not from `ev_charge_manager.vehicle`) to avoid a circular import.

## CLI Reference

```
Highway:    --highway-length (km), --num-stations, --chargers-per-station, --waiting-spots
Traffic:    --vehicles-per-hour, --traffic-pattern, --duration (hours)
            Patterns: UNIFORM | POISSON | RUSH_HOUR | SINGLE_PEAK | PERIODIC | BURSTY
Energy:     --energy-grid-max-kw, --energy-solar-peak-kw, --energy-wind-base-kw, --energy-battery-kwh
Output:     --output-dir, --no-visualize, --quiet, --seed
Trackers:   --enable-station-tracker, --enable-vehicle-tracker
```

## Documentation

- [USER_MANUAL.md](USER_MANUAL.md) — full parameter reference, configuration examples, troubleshooting
- [docs/TrainingComparison.md](docs/TrainingComparison.md) — comparison of RL training modes
- [AGENT.md](AGENT.md) — developer onboarding, architecture deep-dive, key class reference
