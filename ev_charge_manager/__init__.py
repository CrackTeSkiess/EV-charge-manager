"""
EV Charge Manager
=================
Discrete-event simulation framework for EV charging infrastructure
with hierarchical multi-agent reinforcement learning.

Package layout
--------------
ev_charge_manager/
    simulation/     – orchestrator, parameters, stop conditions
    highway/        – road, spatial index, vehicle movement
    vehicle/        – EV physics, battery, driver behaviour, generator, tracker
    charging/       – station, chargers, queue
    energy/         – energy sources, manager, hierarchical manager, micro-RL agent
    rl/             – Gymnasium env, PPO trainer, hierarchical trainer
    analytics/      – KPI tracker, station collector, blackout/stranding trackers
    visualization/  – charts, dashboards
"""

