"""
Training Mode Comparison: SEQUENTIAL vs SIMULTANEOUS vs CURRICULUM
===================================================================
Runs hierarchical training for each of the three modes with multiple seeds
and compares: final reward, convergence episode, training wall-clock time,
and best arbitrage profit.

Usage
-----
  python scripts/run_mode_comparison.py
  python scripts/run_mode_comparison.py --seeds 0 1 2 --micro-episodes 500
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
        description="Compare SEQUENTIAL, SIMULTANEOUS, and CURRICULUM training modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="Random seeds per mode")
    parser.add_argument("--micro-episodes", type=int, default=500)
    parser.add_argument("--macro-episodes", type=int, default=300)
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--output", type=str, default="results/mode_comparison")
    return parser.parse_args()


def run_mode(mode: str, seed: int, args: argparse.Namespace, out_dir: str) -> dict:
    """Run one training mode for one seed."""
    run_name = f"{mode}_seed{seed}"
    cmd = [
        sys.executable, "hierarchical.py",
        "--seed", str(seed),
        "--mode", mode,
        "--micro-episodes", str(args.micro_episodes),
        "--macro-episodes", str(args.macro_episodes),
        "--n-agents", str(args.n_agents),
        "--save-dir", out_dir,
        "--run-name", run_name,
        "--no-visualize",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent.parent),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  WARNING: mode={mode} seed={seed} exited with {result.returncode}")

    result_path = os.path.join(out_dir, run_name, "data", "training_result.json")
    if not os.path.exists(result_path):
        result_path = os.path.join(out_dir, "training_result.json")
    if os.path.exists(result_path):
        with open(result_path) as fh:
            data = json.load(fh)
            data["mode"] = mode
            data["seed"] = seed
            return data
    return {"mode": mode, "seed": seed}


def extract_mode_metrics(result: dict) -> dict:
    """Pull key metrics from a training_result dict."""
    best = result.get("best_config") or {}
    return {
        "mode": result.get("mode"),
        "seed": result.get("seed"),
        "micro_final_reward": result.get("micro_final_reward"),
        "macro_final_reward": result.get("macro_final_reward"),
        "training_time_seconds": result.get("training_time_seconds"),
        "best_cost": best.get("cost"),
        "best_arbitrage_profit": best.get("arbitrage_profit"),
    }


def aggregate_mode(metrics_list: list) -> dict:
    """Compute mean ± std for a list of per-seed metrics."""
    def stats(key):
        values = [m[key] for m in metrics_list if m.get(key) is not None]
        if not values:
            return None
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": len(arr),
        }

    return {
        "micro_final_reward": stats("micro_final_reward"),
        "macro_final_reward": stats("macro_final_reward"),
        "training_time_seconds": stats("training_time_seconds"),
        "best_cost": stats("best_cost"),
        "best_arbitrage_profit": stats("best_arbitrage_profit"),
    }


def print_comparison_table(aggregated: dict) -> None:
    modes = list(aggregated.keys())
    print("\n" + "=" * 80)
    print("  Mode Comparison Results (mean ± std)")
    print("=" * 80)

    metrics_to_show = [
        ("training_time_seconds", "Train Time (s)"),
        ("macro_final_reward", "Macro Final Reward"),
        ("micro_final_reward", "Micro Final Reward"),
        ("best_arbitrage_profit", "Best Arbitrage ($)"),
        ("best_cost", "Best Total Cost ($)"),
    ]

    print(f"  {'Metric':<28}", end="")
    for mode in modes:
        print(f"  {mode:<22}", end="")
    print()
    print("-" * 80)

    for key, label in metrics_to_show:
        print(f"  {label:<28}", end="")
        for mode in modes:
            s = aggregated[mode].get(key)
            if s and s["mean"] is not None:
                print(f"  {s['mean']:>10.2f}±{s['std']:>8.2f}  ", end="")
            else:
                print(f"  {'N/A':>21}  ", end="")
        print()
    print("=" * 80)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    modes = ["sequential", "simultaneous", "curriculum"]
    all_results: dict = {m: [] for m in modes}
    all_metrics: dict = {m: [] for m in modes}

    print("=" * 70)
    print("  Training Mode Comparison")
    print(f"  Modes: {modes}")
    print(f"  Seeds: {args.seeds}")
    print("=" * 70)

    for mode in modes:
        print(f"\n--- Mode: {mode} ---")
        for seed in args.seeds:
            print(f"  Seed {seed}...")
            try:
                result = run_mode(mode, seed, args, args.output)
                all_results[mode].append(result)
                metrics = extract_mode_metrics(result)
                all_metrics[mode].append(metrics)
                print(f"    time={result.get('training_time_seconds', '?'):.1f}s  "
                      f"macro_reward={result.get('macro_final_reward', '?')}")
            except Exception as exc:
                print(f"    FAILED: {exc}")

    aggregated = {mode: aggregate_mode(all_metrics[mode]) for mode in modes}
    print_comparison_table(aggregated)

    out_path = os.path.join(args.output, "mode_comparison_results.json")
    with open(out_path, "w") as fh:
        json.dump({
            "config": vars(args),
            "per_mode_results": all_results,
            "per_mode_metrics": all_metrics,
            "aggregated": aggregated,
        }, fh, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # LaTeX table hint
    print("\n% LaTeX table hint (paste into article):")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Metric & SEQUENTIAL & SIMULTANEOUS & CURRICULUM \\\\ \\hline")
    for key, label in [("training_time_seconds", "Train Time (s)"),
                       ("macro_final_reward", "Macro Reward"),
                       ("best_arbitrage_profit", "Arbitrage (\\$)")]:
        row = [label.replace("$", "\\$")]
        for mode in modes:
            s = aggregated[mode].get(key)
            if s and s["mean"] is not None:
                row.append(f"${s['mean']:.1f}\\pm{s['std']:.1f}$")
            else:
                row.append("N/A")
        print(" & ".join(row) + " \\\\")
    print("\\hline\\end{tabular}")


if __name__ == "__main__":
    main()
