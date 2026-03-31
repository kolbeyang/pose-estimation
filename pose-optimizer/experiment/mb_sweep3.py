"""Third sweep: test higher rotation penalties for distal joints (wrists/elbows).

Usage:
    uv run python experiment/mb_sweep3.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RunConfig, OptimizationConfig
from run_motionbert import process_example


TEST_EXAMPLES = [
    ("171204_pose1_sample", 0, 100),
    ("171204_pose1", 5000, 150),
    ("171204_pose1", 16000, 150),
    ("171204_pose3", 3000, 150),
    ("171204_pose1", 12000, 150),
]


def run_config(label, params, rotation_multipliers=None):
    opt_config = OptimizationConfig(**params)
    if rotation_multipliers:
        opt_config.rotation_penalty_multipliers = rotation_multipliers
    config = RunConfig(
        examples=[],
        data_root="data/panoptic-toolbox",
        target_fps=10.0,
        optimization=opt_config,
    )

    example_results = {}
    for seq, start, nframes in TEST_EXAMPLES:
        name = f"{seq}_{start}"
        run_dir = f"/tmp/mb_sweep_rot"
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
        print(f"\n  {label}: pos={mean_delta:+.3f}cm vel={mean_vel_delta:+.3f}cm ({improved}/{len(example_results)})")
        for ex, r in example_results.items():
            print(f"    {ex:35s} pos={r['delta_pos']:+.3f}  vel={r['delta_vel']:+.3f}")
        return {"mean_delta_pos_cm": mean_delta, "mean_delta_vel_cm": mean_vel_delta, "improved": improved}
    return None


if __name__ == "__main__":
    # Base params from best config
    base_params = {
        "num_steps": 5,
        "learning_rate": 0.0005,
        "heatmap_blur_sigma": 12.0,
        "position_penalty_weight": 500.0,
        "rotation_penalty_scalar": 10.0,
    }

    # Default multipliers:
    # [3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2, 1.0, 1.0, 0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1]
    # Indices: 0=Pelvis, 1=RHip, 2=RKnee, 3=RAnkle, 4=LHip, 5=LKnee, 6=LAnkle,
    #          7=Spine, 8=Neck, 9=Head, 10=LShoulder, 11=LElbow, 12=LWrist, 13=RShoulder, 14=RElbow, 15=RWrist

    print("Testing rotation penalty multiplier variants on best config (lr=0.0005, blur=12)")

    # Baseline best config
    run_config("baseline_best", base_params)

    # Increase wrist/elbow penalties (dampen arm jitter)
    higher_arm = [3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.3]
    run_config("higher_arm_rot", base_params, higher_arm)

    # Much higher wrist/elbow penalties
    much_higher_arm = [3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5]
    run_config("much_higher_arm_rot", base_params, much_higher_arm)

    # Uniform penalties (all 1.0)
    uniform = [1.0] * 16
    run_config("uniform_rot", base_params, uniform)
