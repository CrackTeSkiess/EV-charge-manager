"""
Hierarchical Dual-RL Training Script
=====================================
Trains both Micro-RL (energy arbitrage per station) and Macro-RL
(infrastructure optimisation across all charging areas) using one of
three coordinated strategies:

  sequential    Micro first → Macro (RECOMMENDED, most stable)
  simultaneous  Both together (faster, potentially unstable)
  curriculum    Progressive complexity: 1-day → 3-day → 7-day
  frozen_micro  Load pre-trained micro-agents, train macro only

Usage
-----
  python hierarchical.py                         # sequential, 3 stations
  python hierarchical.py --mode simultaneous
  python hierarchical.py --mode curriculum --robust
  python hierarchical.py --mode frozen_micro --micro-load ./models/micro_pretrained
  python hierarchical.py --n-agents 5 --micro-episodes 2000 --macro-episodes 1000
"""

import argparse
import json
import os
import sys
from datetime import datetime

from ev_charge_manager.rl.environment import (
    MultiAgentChargingEnv,
    WeatherProfile,
    CostParameters,
)
from ev_charge_manager.rl.hierarchical_trainer import (
    HierarchicalTrainer,
    MicroRLTrainer,
    TrainingConfig,
    TrainingMode,
)
from ev_charge_manager.energy import GridPricingSchedule


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hierarchical Dual-RL trainer for EV charging infrastructure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Training mode ────────────────────────────────────────────────────────
    mode_group = parser.add_argument_group("Training mode")
    mode_group.add_argument(
        "--mode",
        choices=["sequential", "simultaneous", "curriculum", "frozen_micro"],
        default="sequential",
        help="Hierarchical training strategy",
    )
    mode_group.add_argument(
        "--robust",
        action="store_true",
        help="Use robust preset (more episodes, 3-stage curriculum). "
             "Overrides --micro-episodes, --macro-episodes and curriculum stages.",
    )

    # ── Environment ──────────────────────────────────────────────────────────
    env_group = parser.add_argument_group("Environment")
    env_group.add_argument("--n-agents", type=int, default=3,
                           help="Number of charging areas / macro agents")
    env_group.add_argument("--highway-length", type=float, default=300.0,
                           help="Highway length in km")
    env_group.add_argument("--traffic-min", type=float, default=30.0,
                           help="Min traffic flow (veh/hr)")
    env_group.add_argument("--traffic-max", type=float, default=80.0,
                           help="Max traffic flow (veh/hr)")
    env_group.add_argument("--sim-duration", type=float, default=24.0,
                           help="Simulation duration in hours (macro env)")
    env_group.add_argument("--collab-weight", type=float, default=0.5,
                           help="Collaboration vs individual reward weighting (0-1)")

    # ── Weather ──────────────────────────────────────────────────────────────
    wx_group = parser.add_argument_group("Weather profile")
    wx_group.add_argument("--solar-min", type=float, default=0.0)
    wx_group.add_argument("--solar-max", type=float, default=1.0)
    wx_group.add_argument("--wind-min",  type=float, default=0.0)
    wx_group.add_argument("--wind-max",  type=float, default=1.0)

    # ── Cost / penalties ─────────────────────────────────────────────────────
    cost_group = parser.add_argument_group("Cost parameters")
    cost_group.add_argument("--stranded-penalty", type=float, default=10_000.0,
                            help="Penalty per stranded vehicle ($)")
    cost_group.add_argument("--blackout-penalty", type=float, default=50_000.0,
                            help="Penalty per blackout event ($)")
    cost_group.add_argument("--grid-peak-price",     type=float, default=0.25,
                            help="Grid electricity price during peak hours ($/kWh)")
    cost_group.add_argument("--grid-shoulder-price", type=float, default=0.15,
                            help="Grid price during shoulder hours ($/kWh)")
    cost_group.add_argument("--grid-offpeak-price",  type=float, default=0.08,
                            help="Grid price during off-peak hours ($/kWh)")

    # ── Micro-RL (Phase 1 / per-station energy agents) ───────────────────────
    micro_group = parser.add_argument_group("Micro-RL (energy arbitrage)")
    micro_group.add_argument("--micro-episodes",  type=int,   default=1000,
                             help="Training episodes for micro-agents")
    micro_group.add_argument("--micro-lr",        type=float, default=3e-4,
                             help="Learning rate for micro-agents")
    micro_group.add_argument("--micro-load",      type=str,   default=None,
                             help="Path to pre-trained micro-agents (skips micro training)")
    micro_group.add_argument("--battery-kwh",     type=float, default=500.0,
                             help="Station battery capacity (kWh)")
    micro_group.add_argument("--battery-kw",      type=float, default=100.0,
                             help="Station battery max charge/discharge power (kW)")
    micro_group.add_argument("--solar-kw",        type=float, default=300.0,
                             help="Installed solar capacity per station (kW)")
    micro_group.add_argument("--wind-kw",         type=float, default=150.0,
                             help="Installed wind capacity per station (kW)")

    # ── Macro-RL (infrastructure / PPO) ─────────────────────────────────────
    macro_group = parser.add_argument_group("Macro-RL (infrastructure optimisation)")
    macro_group.add_argument("--macro-episodes",          type=int,   default=500,
                             help="PPO training episodes for macro-agents")
    macro_group.add_argument("--macro-episodes-per-update", type=int, default=10,
                             help="Episodes collected before each PPO update")
    macro_group.add_argument("--macro-lr",                type=float, default=3e-4,
                             help="Learning rate for macro PPO")
    macro_group.add_argument("--gamma",                   type=float, default=0.99,
                             help="Discount factor")
    macro_group.add_argument("--eval-frequency",          type=int,   default=50,
                             help="Evaluate every N episodes")
    macro_group.add_argument("--checkpoint-frequency",    type=int,   default=100,
                             help="Save checkpoint every N episodes")

    # ── Simultaneous mode extras ─────────────────────────────────────────────
    sim_group = parser.add_argument_group("Simultaneous mode extras")
    sim_group.add_argument("--freeze-micro-after", type=int, default=1000,
                           help="Freeze micro-agents after N episodes "
                                "(simultaneous mode only)")

    # ── System ───────────────────────────────────────────────────────────────
    sys_group = parser.add_argument_group("System")
    sys_group.add_argument("--device",   type=str, default="auto",
                           help="Compute device: auto / cpu / cuda")
    sys_group.add_argument("--seed",     type=int, default=None,
                           help="Global random seed")
    sys_group.add_argument("--save-dir", type=str, default="./models/hierarchical",
                           help="Root directory for checkpoints and final models")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(args: argparse.Namespace) -> None:
    mode_label = {
        "sequential":   "Sequential  (micro → macro, RECOMMENDED)",
        "simultaneous": "Simultaneous (micro + macro together, may be unstable)",
        "curriculum":   "Curriculum  (1-day → 3-day → 7-day complexity)",
        "frozen_micro": "Frozen-Micro (load pre-trained micro, train macro only)",
    }[args.mode]
    print("=" * 70)
    print("  Hierarchical Dual-RL Charging Infrastructure Trainer")
    print("=" * 70)
    print(f"  Mode        : {mode_label}")
    print(f"  Agents      : {args.n_agents}")
    print(f"  Highway     : {args.highway_length} km")
    print(f"  Traffic     : {args.traffic_min}–{args.traffic_max} veh/hr")
    if args.mode not in ("frozen_micro",):
        print(f"  Micro eps   : {args.micro_episodes}")
    print(f"  Macro eps   : {args.macro_episodes}")
    print(f"  Save dir    : {args.save_dir}")
    if args.micro_load:
        print(f"  Load micro  : {args.micro_load}")
    print("=" * 70)


def _make_env(args: argparse.Namespace) -> MultiAgentChargingEnv:
    weather = WeatherProfile(
        min_solar=args.solar_min,
        max_solar=args.solar_max,
        min_wind=args.wind_min,
        max_wind=args.wind_max,
    )
    cost_params = CostParameters(
        stranded_vehicle_penalty=args.stranded_penalty,
        blackout_penalty=args.blackout_penalty,
        grid_peak_price=args.grid_peak_price,
        grid_shoulder_price=args.grid_shoulder_price,
        grid_offpeak_price=args.grid_offpeak_price,
    )
    return MultiAgentChargingEnv(
        n_agents=args.n_agents,
        highway_length_km=args.highway_length,
        traffic_range=(args.traffic_min, args.traffic_max),
        weather_profile=weather,
        cost_params=cost_params,
        simulation_duration_hours=args.sim_duration,
        collaboration_weight=args.collab_weight,
        enable_micro_rl=True,
    )


def _make_config(args: argparse.Namespace) -> TrainingConfig:
    mode_map = {
        "sequential":   TrainingMode.SEQUENTIAL,
        "simultaneous": TrainingMode.SIMULTANEOUS,
        "curriculum":   TrainingMode.CURRICULUM,
        "frozen_micro": TrainingMode.FROZEN_MICRO,
    }

    curriculum_stages = [
        {"days": 1, "price_variance": 0.0},
        {"days": 3, "price_variance": 0.3},
        {"days": 7, "price_variance": 0.6},
    ] if args.robust else [
        {"days": 1, "price_variance": 0.0},
        {"days": 3, "price_variance": 0.2},
        {"days": 7, "price_variance": 0.5},
    ]

    micro_eps  = 2000 if args.robust else args.micro_episodes
    macro_eps  = 1000 if args.robust else args.macro_episodes

    return TrainingConfig(
        mode=mode_map[args.mode],
        micro_episodes=micro_eps,
        micro_lr=args.micro_lr,
        macro_episodes=macro_eps,
        macro_episodes_per_update=args.macro_episodes_per_update,
        macro_lr=args.macro_lr,
        simultaneous_freeze_micro_after=args.freeze_micro_after,
        curriculum_stages=curriculum_stages,
        eval_frequency=args.eval_frequency,
        checkpoint_frequency=args.checkpoint_frequency,
        output_dir=args.save_dir,
    )


# ---------------------------------------------------------------------------
# frozen_micro shortcut – load pre-trained micro-agents then train macro
# ---------------------------------------------------------------------------

def _run_frozen_micro(args: argparse.Namespace, env: MultiAgentChargingEnv,
                      config: TrainingConfig) -> dict:
    """Load pre-trained micro-agents and train only the macro layer."""
    if not args.micro_load:
        print("ERROR: --micro-load <path> is required for frozen_micro mode.")
        sys.exit(1)

    if not os.path.isdir(args.micro_load):
        print(f"ERROR: Micro-agent directory not found: {args.micro_load}")
        sys.exit(1)

    pricing = GridPricingSchedule(
        off_peak_price=args.grid_offpeak_price,
        shoulder_price=args.grid_shoulder_price,
        peak_price=args.grid_peak_price,
    )
    base_config = {
        "battery_kwh": args.battery_kwh,
        "battery_kw":  args.battery_kw,
        "solar_kw":    args.solar_kw,
        "wind_kw":     args.wind_kw,
    }

    micro = MicroRLTrainer(
        base_config=base_config,
        n_stations=args.n_agents,
        pricing=pricing,
        device=args.device,
    )
    micro.load(args.micro_load)
    print(f"Loaded {args.n_agents} micro-agents from {args.micro_load}")

    # Evaluate loaded micro-agents before macro training
    micro_eval = micro.evaluate(n_episodes=5)
    print(f"Pre-loaded micro-agent performance:")
    print(f"  Avg Reward    : {micro_eval['avg_reward']:.2f}")
    print(f"  Avg Grid Cost : ${micro_eval['avg_grid_cost']:.2f}")

    # Hand off to hierarchical trainer (macro phase only).
    # Inject trained micro-agent weights into the environment so every
    # HierarchicalEnergyManager created during macro training starts from the
    # loaded policy rather than a random initialisation.
    trainer = HierarchicalTrainer(env=env, config=config, device=args.device)
    trainer.micro_trainer = micro  # kept for interrupt-save in caller
    env.enable_micro_rl = True
    env.set_micro_agents(micro.agents)
    from ev_charge_manager.rl.ppo_trainer import MultiAgentPPO
    trainer.macro_trainer = MultiAgentPPO(
        env=env,
        lr=config.macro_lr,
        device=args.device,
    )
    print(f"\n--- Macro-RL training ({config.macro_episodes} episodes) ---")
    print("Using pre-loaded micro-agents (frozen)")

    try:
        trainer.macro_trainer.train(
            total_episodes=config.macro_episodes,
            episodes_per_update=config.macro_episodes_per_update,
            eval_interval=config.eval_frequency,
            save_interval=config.checkpoint_frequency,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user – saving checkpoint...")
        trainer.macro_trainer.save(
            os.path.join(args.save_dir, "macro_interrupted.pt")
        )

    final_eval = trainer.macro_trainer.evaluate(n_episodes=10)
    result = {
        "mode": "frozen_micro",
        "micro_source": args.micro_load,
        "macro_episodes": config.macro_episodes,
        "macro_final_reward": final_eval["avg_reward"],
        "best_config": env.get_best_config(),
    }
    result_path = os.path.join(args.save_dir, "training_result.json")
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Seed
    if args.seed is not None:
        import random, numpy as np, torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    _banner(args)
    os.makedirs(args.save_dir, exist_ok=True)

    # Persist run configuration
    run_cfg = vars(args).copy()
    run_cfg["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(args.save_dir, "run_config.json"), "w") as fh:
        json.dump(run_cfg, fh, indent=2)

    env    = _make_env(args)
    config = _make_config(args)

    # ── Run ──────────────────────────────────────────────────────────────────
    try:
        if args.mode == "frozen_micro":
            result = _run_frozen_micro(args, env, config)
        else:
            trainer = HierarchicalTrainer(
                env=env,
                config=config,
                device=args.device,
            )

            # Optional: pre-load micro-agents before sequential/curriculum run
            if args.micro_load and args.mode in ("sequential", "curriculum"):
                pricing = GridPricingSchedule(
                    off_peak_price=args.grid_offpeak_price,
                    shoulder_price=args.grid_shoulder_price,
                    peak_price=args.grid_peak_price,
                )
                base_config = {
                    "battery_kwh": args.battery_kwh,
                    "battery_kw":  args.battery_kw,
                    "solar_kw":    args.solar_kw,
                    "wind_kw":     args.wind_kw,
                }
                micro = MicroRLTrainer(
                    base_config=base_config,
                    n_stations=args.n_agents,
                    pricing=pricing,
                    device=args.device,
                )
                micro.load(args.micro_load)
                trainer.micro_trainer = micro  # kept for interrupt-save
                trainer.env.set_micro_agents(micro.agents)
                print(f"Pre-loaded micro-agents from {args.micro_load} "
                      f"(will skip Phase 1 micro training)")

            try:
                result = trainer.train()
            except KeyboardInterrupt:
                print("\nTraining interrupted by user.")
                if trainer.macro_trainer is not None:
                    trainer.macro_trainer.save(
                        os.path.join(args.save_dir, "macro_interrupted.pt")
                    )
                if trainer.micro_trainer is not None:
                    trainer.micro_trainer.save(
                        os.path.join(args.save_dir, "micro_interrupted")
                    )
                sys.exit(0)

    except ImportError as exc:
        print(f"\nERROR: Missing dependency – {exc}")
        print("Install RL support with:  pip install torch gymnasium")
        sys.exit(1)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)

    results_path = os.path.join(args.save_dir, "results.json")
    with open(results_path, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    # Pretty-print key metrics
    mode = result.get("mode", args.mode)
    if mode == "sequential":
        print(f"  Micro final reward : {result.get('micro_final_reward', 'N/A')}")
        print(f"  Macro final reward : {result.get('macro_final_reward', 'N/A')}")
    elif mode == "simultaneous":
        print(f"  Episodes trained   : {result.get('episodes', args.macro_episodes)}")
    elif mode == "curriculum":
        stages = result.get("stages", [])
        for s in stages:
            print(f"  Stage {s['stage']+1}: {s['config']}")
    elif mode == "frozen_micro":
        print(f"  Macro final reward : {result.get('macro_final_reward', 'N/A')}")

    best = result.get("best_config")
    if best:
        print(f"\n  Best infrastructure config found:")
        for k, v in (best.items() if isinstance(best, dict) else []):
            print(f"    {k}: {v}")

    print(f"\nAll outputs saved to: {args.save_dir}/")
    print(f"  run_config.json   – CLI parameters used")
    print(f"  results.json      – final training metrics")
    print(f"  training_result.json – full result (written by HierarchicalTrainer)")
    if mode in ("sequential", "curriculum"):
        micro_path = os.path.join(args.save_dir, "micro_pretrained")
        print(f"  micro_pretrained/ – trained energy-arbitrage agents")
        print(f"    Reuse with:  --mode frozen_micro --micro-load {micro_path}")


if __name__ == "__main__":
    main()
