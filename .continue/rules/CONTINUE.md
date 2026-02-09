# EV Simulation & Optimization System - CONTINUE.md
---
## Project Overview
This system models electric vehicle behavior focusing on stranded vehicle tracking, energy optimization, and traffic flow analysis through reinforcement learning agents.

## Key Features
- Real-time stranded vehicle tracking (KPITracker.py)
- Energy management with blackout event handling
- PPO-based RL decision making (ppo_agent.py)
- Parallelizable simulation engine

### Key Technologies Used:
- Python 3.x with type hints
- Reinforcement Learning (PPO-based agents)
- Custom simulation engine
- Optuna hyperparameter optimization
- Data visualization tools

---

## Project Architecture Analysis

### Core Components and Their Relationships:

```
┌───────────────────────┐    ┌───────────────────────┐    ┌───────────────────────┐
│ Vehicle Simulation   │    │ Energy Management     │    │ Optimization System  │
│ (Simulation.py)      │    │ (EnergyManager.py)    │    │ (main_optimization.py)│
└─────────┬─────────────┘    └─────────┬─────────────┘    └─────────┬─────────────┘
          │                          │                          │
┌──────────▼───────────┐ ┌───────────▼───────────────┐ ┌───────────▼───────────────┐
│ Vehicle Tracker     │ │ Charging Infrastructure  │ │ RL Decision Engine        │
│ (VehicleTracker.py) │ │ (ChargingArea.py)       │ │ (ppo_agent.py)             │
└──────────┬───────────┘ └───────────┬───────────────┘ └───────────┬───────────────┘
          │                          │                          │
┌──────────▼───────────┐ ┌───────────▼───────────────┐ ┌───────────▼───────────────┐
│ Stranded Vehicle    │ │ Data Collection           │ │ Simulation Output         │
│ Tracking System      │ │ (StationDataCollector.py)│ │ Visualization              │
│ (KPITracker.py)     │ └───────────┬───────────────┘ │ (VisualizationTool.py)   │
└─────────────────────┘             │                   └─────────────────────────┘
                                      │
                                      ▼
                                     Data Pipeline
```

---

## Getting Started

### Prerequisites:
```bash
# Install dependencies with version requirements
pip install -r requirements.txt
```

### Installation Instructions:

1. Clone the repository
2. Set up virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .\.\.venv\Scripts\activate # Windows
   ```

3. Install additional dependencies:
   ```bash
   pip install numpy matplotlib pandas optuna
   ```

---

## Project Structure

### Core Directories and Files:

```
├── core_simulation/
│   ├── simulation.py          # Main simulation engine with parallelization support
│   └── vehicle_generator.py    # Vehicle population and movement logic

├── energy_system/
│   ├── EnergyManager.py        # EV battery/charging dynamics
│   ├── blackout_tracker.py     # Blackout event handling
│   └── ChargingArea.py         # Charging infrastructure management

├── tracking/
│   ├── KPITracker.py           # Key Performance Indicator system
│   └── stranded_vehicle_tracker.py # Real-time tracking of stranded vehicles

├── optimization/
│   ├── main_optimization.py     # Optimization workflow
│   ├── optuna_optimizer.py      # Hyperparameter tuning
│   └── optimization_config.py   # Configuration parameters

└── visualization/
    └── VisualizationTool.py      # Data visualization utilities
```

---

## Development Workflow and Best Practices

### Coding Standards (PEP 8 Compliance):
```python
# Example of properly formatted code with docstrings
def calculate_battery_drain(distance: float, speed: float) -> float:
    """
    Calculate battery drain based on distance traveled and vehicle speed.

    Args:
        distance: Total distance traveled in kilometers
        speed: Average speed in km/h

    Returns:
        Battery drain percentage

    Raises:
        ValueError: If inputs are negative or zero
    """
    if distance <= 0 or speed <= 0:
        raise ValueError("Distance and speed must be positive values")
    return (distance * speed) / 1000
```

### Testing Recommendations:

1. **Unit Tests**: Add comprehensive tests for critical components:
   ```python
   # Example test case structure
   def test_battery_drain_calculation():
       assert calculate_battery_drain(5, 60) == 3.0  # Should return 3.0%
   ```

2. **Edge Case Testing**: Include tests for:
   - Maximum blackout events (100% coverage)
   - Minimum vehicle populations
   - Extreme speed/distance combinations

---

## Common Tasks and Potential Pitfalls

### Running a Simulation:

```bash
# Basic simulation command with warnings about scalability
python main.py \
    --sim-duration 60 \
    --env highway_simulation \
    --parallel True  # Enable parallel processing for large simulations
```

**Warning**: For simulations with >10,000 vehicles:
- Consider reducing `--sim-duration` to prevent memory overload
- Use `--parallel True` flag for better performance

### Optimization Process:

```bash
# Run optimization with versioned results
python main_optimization.py \
    --config optimization_config.json \
    --output_prefix ev_opt_$(date +%Y%m%d_%H%M%S)
```

---

## Troubleshooting Guide

| Issue Category               | Common Problems                          | Solutions                                  |
|------------------------------|-------------------------------------------|--------------------------------------------|
| **Scalability Issues**       | Simulation crashes with large datasets    | Increase memory allocation or split workload |
| **Energy Management Failures** | Blackout events not properly handled     | Add validation in `blackout_tracker.py`   |
| **Data Corruption**          | Missing/duplicate station data            | Use pandas for robust data handling         |
| **Visualization Errors**     | Missing results files                    | Check simulation output directory           |

---

## Optimization and Scalability Recommendations

### Current Limitations:

1. **Parallel Processing**:
   - The simulation engine currently lacks explicit parallelization
   - For large-scale simulations (>5,000 vehicles), consider:
     ```python
     # Example of potential multiprocessing implementation
     from multiprocessing import Pool

     def run_simulation_chunk(chunk):
         # Process a subset of vehicles in parallel
         pass

     with Pool() as p:
         results = p.map(run_simulation_chunk, vehicle_chunks)
     ```

2. **Blackout Event Handling**:
   - Current implementation may not properly integrate blackouts into RL decisions
   - Recommend adding dynamic adaptation in `ppo_agent.py`

3. **Data Versioning**:
   - Optimization results lack version control
   - Suggest using commit hashes or timestamps in filenames

---

## Domain-Specific Terminology

| Term                | Definition                                                                 |
|---------------------|---------------------------------------------------------------------------|
| Stranded Vehicle    | EV unable to complete journey due to charging failures                    |
| KPI Tracker         | System monitoring stranded vehicle metrics (e.g., % of EVs stuck)          |
| Blackout Event      | Power outage affecting charging infrastructure                            |
| Parallel Processing | Distributing simulation workload across multiple CPU cores                 |

---

## References and Additional Resources

1. **Documentation**:
   - [USER_MANUAL.md](USER_MANUAL.md)
   - Optimization configuration: `optimization_config.py`

2. **Dependencies**:
   ```bash
   pip install numpy==1.23.5  # Specific version for compatibility
   ```

3. **Visualization Examples**:
   ```python
   from VisualizationTool import plot_stranded_vehicles
   plot_stranded_vehicles("simulation_output/results.csv")
