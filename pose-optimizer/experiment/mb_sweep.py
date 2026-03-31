"""Sweep hyperparameters for MotionBert optimizer on a few representative examples.

Usage:
    uv run python experiment/mb_sweep.py
"""

import json
import os
import sys
import time
from itertools import product

import numpy as np

# Add parent dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RunConfig, ExampleConfig, OptimizationConfig
from run_motionbert import process_example


# Representative examples: 2 that improve, 2 that get worse, 1 borderline
TEST_EXAMPLES = [
    # Good improvement
    ("171204_pose1_sample", 0, 100),    # -3.84cm (big win)
    ("171204_pose1", 5000, 150),         # -1.63cm (good win)
    # Gets worse
    ("171204_pose1", 16000, 150),        # +0.31cm (bad)
    ("171204_pose3", 3000, 150),         # +0.51cm (bad)
    # Borderline
    ("171204_pose1", 12000, 150),        # +0.11cm (slight worse)
]


def run_sweep(sweep_name, param_grid):
    """Run a parameter sweep and return results."""
    results = {}

    # Generate all combos
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(product(*param_values))

    print(f"\n{'='*60}")
    print(f"  Sweep: {sweep_name}")
    print(f"  {len(combos)} configurations x {len(TEST_EXAMPLES)} examples")
    print(f"{'='*60}")

    for combo in combos:
        params = dict(zip(param_names, combo))
        config_label = "_".join(f"{k}={v}" for k, v in params.items())
        print(f"\n  Config: {config_label}")

        opt_config = OptimizationConfig(**params)
        config = RunConfig(
            examples=[],
            data_root="data/panoptic-toolbox",
            target_fps=10.0,
            optimization=opt_config,
        )

        example_results = {}
        for seq, start, nframes in TEST_EXAMPLES:
            name = f"{seq}_{start}"
            run_dir = f"/tmp/mb_sweep_{sweep_name}"
            os.makedirs(run_dir, exist_ok=True)

            try:
                t0 = time.time()
                metrics = process_example(
                    config, seq, "00_00", start, nframes, 0, run_dir,
                )
                elapsed = time.time() - t0

                if metrics:
                    det_vw = metrics.get("det_vw_si_mpjpe", 0) * 100
                    opt_vw = metrics.get("opt_vw_si_mpjpe", 0) * 100
                    det_vel = metrics.get("det_vw_si_mpjve", 0) * 100
                    opt_vel = metrics.get("opt_vw_si_mpjve", 0) * 100
                    example_results[name] = {
                        "det_vw_si_mpjpe": det_vw,
                        "opt_vw_si_mpjpe": opt_vw,
                        "delta_pos": opt_vw - det_vw,
                        "det_vw_si_mpjve": det_vel,
                        "opt_vw_si_mpjve": opt_vel,
                        "delta_vel": opt_vel - det_vel,
                        "time": elapsed,
                    }
                    print(f"    {name}: pos {det_vw:.2f}->{opt_vw:.2f} ({opt_vw-det_vw:+.2f}), vel {det_vel:.2f}->{opt_vel:.2f} ({opt_vel-det_vel:+.2f}), {elapsed:.1f}s")
            except Exception as e:
                print(f"    {name}: ERROR {e}")

        # Summary for this config
        if example_results:
            deltas = [v["delta_pos"] for v in example_results.values()]
            mean_delta = np.mean(deltas)
            improved = sum(1 for d in deltas if d < 0)
            vel_deltas = [v["delta_vel"] for v in example_results.values()]
            mean_vel_delta = np.mean(vel_deltas)

            results[config_label] = {
                "params": params,
                "examples": example_results,
                "mean_delta_pos_cm": float(mean_delta),
                "mean_delta_vel_cm": float(mean_vel_delta),
                "improved_count": improved,
                "total_count": len(example_results),
            }
            print(f"  => Mean pos delta: {mean_delta:+.3f}cm, vel delta: {mean_vel_delta:+.3f}cm, improved: {improved}/{len(example_results)}")

    # Save results
    out_path = f"experiment/sweep_results_{sweep_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print ranking
    print(f"\n{'='*60}")
    print(f"  Ranking by mean VW-SI-MPJPE improvement (negative = better):")
    print(f"{'='*60}")
    ranked = sorted(results.items(), key=lambda x: x[1]["mean_delta_pos_cm"])
    for label, data in ranked:
        print(f"  {label:60s} pos={data['mean_delta_pos_cm']:+.3f}cm vel={data['mean_delta_vel_cm']:+.3f}cm ({data['improved_count']}/{data['total_count']})")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep", choices=["lr", "blur", "steps", "combined", "penalty"])
    args = parser.parse_args()

    if args.sweep == "lr":
        run_sweep("lr", {
            "num_steps": [5],
            "learning_rate": [0.0005, 0.001, 0.002, 0.004],
            "heatmap_blur_sigma": [4.0],
            "position_penalty_weight": [500.0],
        })

    elif args.sweep == "blur":
        run_sweep("blur", {
            "num_steps": [5],
            "learning_rate": [0.002],
            "heatmap_blur_sigma": [2.0, 4.0, 8.0, 12.0],
            "position_penalty_weight": [500.0],
        })

    elif args.sweep == "steps":
        run_sweep("steps", {
            "num_steps": [2, 3, 5, 8, 10],
            "learning_rate": [0.002],
            "heatmap_blur_sigma": [4.0],
            "position_penalty_weight": [500.0],
        })

    elif args.sweep == "penalty":
        run_sweep("penalty", {
            "num_steps": [5],
            "learning_rate": [0.002],
            "heatmap_blur_sigma": [4.0],
            "position_penalty_weight": [500.0, 1000.0, 2000.0],
            "rotation_penalty_scalar": [10.0, 20.0, 40.0],
        })

    elif args.sweep == "combined":
        # Test the most promising combinations
        run_sweep("combined", {
            "num_steps": [3, 5],
            "learning_rate": [0.001, 0.002],
            "heatmap_blur_sigma": [4.0, 8.0],
            "position_penalty_weight": [500.0],
        })
