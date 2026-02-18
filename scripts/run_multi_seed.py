"""
Multi-Seed Training Runner
===========================
Runs hierarchical RL training across multiple random seeds and aggregates
results into a single JSON with mean ± std statistics for use in the
scientific article's results table.

Usage
-----
  python scripts/run_multi_seed.py
  python scripts/run_multi_seed.py --seeds 0 1 2 3 4 --mode sequential
  python scripts/run_multi_seed.py --seeds 0 1 2 --output results/multi_seed
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hierarchical RL training across multiple seeds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="List of random seeds to use")
    parser.add_argument("--mode", choices=["sequential", "simultaneous", "curriculum"],
                        default="sequential", help="Training mode")
    parser.add_argument("--micro-episodes", type=int, default=1000)
    parser.add_argument("--macro-episodes", type=int, default=500)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/multi_seed",
                        help="Directory to store per-seed outputs and aggregate")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Skip plot generation per seed (faster)")
    return parser.parse_args()


def run_single_seed(seed: int, args: argparse.Namespace, seed_dir: str) -> dict:
    """Run hierarchical.py for a single seed and return its result JSON."""
    cmd = [
        sys.executable, "hierarchical.py",
        "--seed", str(seed),
        "--mode", args.mode,
        "--micro-episodes", str(args.micro_episodes),
        "--macro-episodes", str(args.macro_episodes),
        "--n-agents", str(args.n_agents),
        "--save-dir", seed_dir,
        "--run-name", f"seed_{seed}",
    ]
    if args.no_visualize:
        cmd.append("--no-visualize")

    print(f"\n{'='*60}")
    print(f"  Running seed {seed}")
    print(f"  Output: {seed_dir}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent.parent),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"WARNING: Seed {seed} exited with code {result.returncode}")

    # Locate training_result.json written by hierarchical.py
    # Per RunDirectory layout: <save_dir>/<run_name>/data/training_result.json
    result_path = os.path.join(seed_dir, f"seed_{seed}", "data", "training_result.json")
    if not os.path.exists(result_path):
        # Fallback: older layout wrote directly in save_dir
        result_path = os.path.join(seed_dir, "training_result.json")

    if os.path.exists(result_path):
        with open(result_path) as fh:
            return json.load(fh)
    else:
        print(f"WARNING: training_result.json not found for seed {seed}")
        return {}


def extract_metrics(result: dict) -> dict:
    """Pull key scalar metrics from a training_result dict."""
    best = result.get("best_config") or {}
    return {
        "micro_final_reward": result.get("micro_final_reward"),
        "macro_final_reward": result.get("macro_final_reward"),
        "training_time_seconds": result.get("training_time_seconds"),
        "best_cost": best.get("cost"),
        "best_grid_cost": best.get("grid_cost"),
        "best_arbitrage_profit": best.get("arbitrage_profit"),
    }


def aggregate(seed_metrics: list) -> dict:
    """Compute mean ± std for all numeric metrics across seeds."""
    all_keys = set()
    for m in seed_metrics:
        all_keys.update(m.keys())

    stats = {}
    for key in sorted(all_keys):
        values = [m[key] for m in seed_metrics if m.get(key) is not None]
        if not values:
            stats[key] = {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        else:
            arr = np.array(values, dtype=float)
            stats[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(arr),
            }
    return stats


def print_table(stats: dict) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("  Multi-Seed Aggregate Results")
    print("=" * 70)
    print(f"  {'Metric':<35} {'Mean':>12} {'±Std':>12} {'Min':>10} {'Max':>10}")
    print("-" * 70)
    for key, s in stats.items():
        if s["mean"] is None:
            continue
        mean = s["mean"]
        std = s["std"]
        lo = s["min"]
        hi = s["max"]
        print(f"  {key:<35} {mean:>12.3f} {std:>12.3f} {lo:>10.3f} {hi:>10.3f}")
    print("=" * 70)


def print_latex_table(stats: dict, mode: str) -> None:
    """Print a LaTeX-ready table row for each metric."""
    print("\n% LaTeX table (mean \\pm std) — paste into article:")
    print("\\begin{tabular}{lcc}")
    print("\\hline")
    print("Metric & Mean & Std \\\\ \\hline")
    for key, s in stats.items():
        if s["mean"] is None:
            continue
        latex_key = key.replace("_", "\\_")
        print(f"{latex_key} & {s['mean']:.3f} & {s['std']:.3f} \\\\")
    print("\\hline\\end{tabular}")
    print(f"% Mode: {mode}, n seeds = {list(stats.values())[0]['n'] if stats else '?'}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    seed_results = []
    seed_metrics_list = []

    for seed in args.seeds:
        seed_dir = os.path.join(args.output, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        result = run_single_seed(seed, args, seed_dir)
        seed_results.append(result)
        seed_metrics_list.append(extract_metrics(result))

    # Aggregate
    stats = aggregate(seed_metrics_list)

    # Save
    aggregate_path = os.path.join(args.output, "aggregate_training.json")
    with open(aggregate_path, "w") as fh:
        json.dump({
            "seeds": args.seeds,
            "mode": args.mode,
            "micro_episodes": args.micro_episodes,
            "macro_episodes": args.macro_episodes,
            "per_seed_metrics": seed_metrics_list,
            "aggregate_stats": stats,
        }, fh, indent=2)

    print_table(stats)
    print_latex_table(stats, args.mode)
    print(f"\nAggregate results saved to: {aggregate_path}")


if __name__ == "__main__":
    main()
