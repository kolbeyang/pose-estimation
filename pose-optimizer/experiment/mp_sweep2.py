"""MediaPipe sweep round 2: combine lower LR with more steps.

Usage:
    uv run python experiment/mp_sweep2.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RunConfig, OptimizationConfig
from run_mediapipe import process_example


TEST_EXAMPLES = [
    ("171204_pose1", 8000, 150),
    ("171204_pose2", 2000, 150),
    ("171204_pose1", 16000, 150),
    ("171204_pose1", 22000, 150),
    ("171204_pose3", 4000, 150),
]


def run_config(label, params):
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
        run_dir = f"/tmp/mp_sweep2"
        os.makedirs(run_dir, exist_ok=True)

        try:
            metrics = process_example(
                config, seq, "00_00", start, nframes, 0, run_dir,
            )
            if metrics:
                det_vw = metrics.get("det_vw_si_mpjpe", 0) * 100
                opt_vw = metrics.get("opt_vw_si_mpjpe", 0) * 100
                det_vel = metrics.get("det_vw_si_mpjve", 0) * 100
                opt_vel = metrics.get("opt_vw_si_mpjve", 0) * 100
                example_results[name] = {
                    "delta_pos": opt_vw - det_vw,
                    "delta_vel": opt_vel - det_vel,
                }
        except Exception as e:
            print(f"    {name}: ERROR {e}")

    if example_results:
        deltas = [v["delta_pos"] for v in example_results.values()]
        mean_delta = np.mean(deltas)
        improved = sum(1 for d in deltas if d < 0)
        vel_deltas = [v["delta_vel"] for v in example_results.values()]
        mean_vel_delta = np.mean(vel_deltas)
        print(f"  {label}: pos={mean_delta:+.3f}cm vel={mean_vel_delta:+.3f}cm ({improved}/{len(example_results)})")
        for ex, r in example_results.items():
            print(f"    {ex:35s} pos={r['delta_pos']:+.3f}  vel={r['delta_vel']:+.3f}")
        return {"mean_delta_pos_cm": mean_delta, "mean_delta_vel_cm": mean_vel_delta, "improved": improved}
    return None


if __name__ == "__main__":
    configs = {
        "baseline": {
            "num_steps": 50, "learning_rate": 0.002,
            "heatmap_blur_sigma": 4.0, "position_penalty_weight": 500.0,
        },
        "lr0.001_steps50": {
            "num_steps": 50, "learning_rate": 0.001,
            "heatmap_blur_sigma": 4.0, "position_penalty_weight": 500.0,
        },
        "lr0.0005_steps50": {
            "num_steps": 50, "learning_rate": 0.0005,
            "heatmap_blur_sigma": 4.0, "position_penalty_weight": 500.0,
        },
        "lr0.001_steps80": {
            "num_steps": 80, "learning_rate": 0.001,
            "heatmap_blur_sigma": 4.0, "position_penalty_weight": 500.0,
        },
        "lr0.001_steps50_pen1000": {
            "num_steps": 50, "learning_rate": 0.001,
            "heatmap_blur_sigma": 4.0, "position_penalty_weight": 1000.0,
        },
    }

    all_results = {}
    for label, params in configs.items():
        print(f"\n  Config: {label}")
        r = run_config(label, params)
        if r:
            all_results[label] = r

    print(f"\n  Ranking:")
    ranked = sorted(all_results.items(), key=lambda x: x[1]["mean_delta_pos_cm"])
    for label, data in ranked:
        print(f"  {label:40s} pos={data['mean_delta_pos_cm']:+.3f}cm vel={data['mean_delta_vel_cm']:+.3f}cm ({data['improved']}/5)")
