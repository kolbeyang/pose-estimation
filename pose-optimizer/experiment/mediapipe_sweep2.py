"""MediaPipe hyperparameter sweep round 2: refine top candidates.

Usage:
    cd pose-optimizer
    uv run python experiment/mediapipe_sweep2.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from config import OptimizationConfig
from evaluate import evaluate
from optimize import optimize

# Reuse data loader from sweep1
from experiment.mediapipe_sweep import load_example, run_optimization, BASELINE, make_config


def define_sweeps() -> list[tuple[str, OptimizationConfig]]:
    """Refined sweep based on round 1 findings."""
    sweeps: list[tuple[str, OptimizationConfig]] = []

    # sigma=4 was surprisingly good and fast. Try combos.
    sweeps.append(("sigma=4+anchor=500", make_config(
        heatmap_blur_sigma=4.0, anchor_weight=500.0,
    )))
    sweeps.append(("sigma=4+anchor=500+steps=50", make_config(
        heatmap_blur_sigma=4.0, anchor_weight=500.0, num_steps=50,
    )))
    sweeps.append(("sigma=4+anchor=500+steps=75", make_config(
        heatmap_blur_sigma=4.0, anchor_weight=500.0, num_steps=75,
    )))
    sweeps.append(("sigma=4+anchor=300", make_config(
        heatmap_blur_sigma=4.0, anchor_weight=300.0,
    )))
    sweeps.append(("sigma=4+anchor=200", make_config(
        heatmap_blur_sigma=4.0, anchor_weight=200.0,
    )))

    # Anneal 8->2 was good + fast. Try combos.
    sweeps.append(("anneal=8->2+anchor=500", make_config(
        heatmap_blur_sigma=0.0, heatmap_blur_sigma_start=8.0,
        heatmap_blur_sigma_end=2.0, anchor_weight=500.0,
    )))
    sweeps.append(("anneal=8->2+anchor=500+steps=50", make_config(
        heatmap_blur_sigma=0.0, heatmap_blur_sigma_start=8.0,
        heatmap_blur_sigma_end=2.0, anchor_weight=500.0, num_steps=50,
    )))

    # Top winner from round 1 with variations
    sweeps.append(("steps=50+anneal=16->4+anchor=300", make_config(
        num_steps=50, heatmap_blur_sigma=0.0,
        heatmap_blur_sigma_start=16.0, heatmap_blur_sigma_end=4.0,
        anchor_weight=300.0,
    )))
    sweeps.append(("steps=50+anneal=16->4+anchor=500+lr=0.001", make_config(
        num_steps=50, learning_rate=0.001, heatmap_blur_sigma=0.0,
        heatmap_blur_sigma_start=16.0, heatmap_blur_sigma_end=4.0,
        anchor_weight=500.0,
    )))

    # Try anneal 8->1
    sweeps.append(("anneal=8->1+anchor=500+steps=50", make_config(
        heatmap_blur_sigma=0.0, heatmap_blur_sigma_start=8.0,
        heatmap_blur_sigma_end=1.0, anchor_weight=500.0, num_steps=50,
    )))

    # sigma=4, lower anchor, more steps
    sweeps.append(("sigma=4+anchor=500+steps=50+lr=0.001", make_config(
        heatmap_blur_sigma=4.0, anchor_weight=500.0, num_steps=50, learning_rate=0.001,
    )))

    return sweeps


def main():
    print("=" * 70)
    print("MediaPipe Hyperparameter Sweep Round 2")
    print("=" * 70)

    print("\nLoading example data (one-time cost)...")
    data = load_example()
    print("  Done.\n")

    sweeps = define_sweeps()
    results: list[dict] = []

    for i, (name, cfg) in enumerate(sweeps):
        print(f"[{i+1}/{len(sweeps)}] {name}")
        res = run_optimization(data, cfg)
        res["name"] = name
        results.append(res)
        print(
            f"  Raw: {res['raw_vw_si_mpjpe_cm']:.2f} cm  "
            f"Opt: {res['opt_vw_si_mpjpe_cm']:.2f} cm  "
            f"Improvement: {res['improvement_pct']:.1f}%  "
            f"Time: {res['elapsed_s']:.1f}s"
        )

    results.sort(key=lambda r: r["improvement_pct"], reverse=True)

    print("\n" + "=" * 70)
    print("RESULTS (sorted by improvement)")
    print("=" * 70)
    print(f"{'Config':<50} {'Raw':>7} {'Opt':>7} {'Impr%':>7} {'Time':>6}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['name']:<50} "
            f"{r['raw_vw_si_mpjpe_cm']:>6.2f}  "
            f"{r['opt_vw_si_mpjpe_cm']:>6.2f}  "
            f"{r['improvement_pct']:>6.1f}%  "
            f"{r['elapsed_s']:>5.1f}s"
        )

    out_path = os.path.join(os.path.dirname(__file__), "mediapipe_sweep2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
