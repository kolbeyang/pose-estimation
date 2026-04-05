"""Test MediaPipe configs on multiple examples to find a robust config.

Usage:
    cd pose-optimizer
    uv run python experiment/mediapipe_multi_sweep.py
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
from experiment.mediapipe_sweep import load_example, ExampleData


EXAMPLES = [
    {"seq": "171204_pose1_sample", "camera": "00_00", "start_frame": 0, "num_frames": 100},
    {"seq": "171204_pose2", "camera": "00_00", "start_frame": 200, "num_frames": 150},
    {"seq": "171204_pose3", "camera": "00_00", "start_frame": 200, "num_frames": 150},
    {"seq": "171204_pose1", "camera": "00_00", "start_frame": 5000, "num_frames": 150},
    {"seq": "171204_pose2", "camera": "00_00", "start_frame": 5000, "num_frames": 150},
]


def run_opt(data: ExampleData, config: OptimizationConfig) -> dict:
    opt_3d, _, _ = optimize(
        raw_3d=data.det_cam_positions, camera=data.camera, config=config,
        heatmaps=data.heatmaps, affine=data.affine, visibility=data.visibility,
        verbose=False,
    )
    gt_indices = [i for i, g in enumerate(data.gt_cam) if g is not None]
    gt_arr = np.array([data.gt_cam[i] for i in gt_indices])
    det_arr = np.array([data.det_cam_positions[i] for i in gt_indices])
    opt_arr = np.array([opt_3d[i] for i in gt_indices])
    det_m = evaluate(det_arr, gt_arr, data.camera)
    opt_m = evaluate(opt_arr, gt_arr, data.camera)
    raw = det_m["vw_si_mpjpe"] * 100
    opt = opt_m["vw_si_mpjpe"] * 100
    return {"raw": raw, "opt": opt, "pct": (raw - opt) / raw * 100}


CONFIGS = {
    "sigma=4+anchor=300": OptimizationConfig(
        num_steps=25, learning_rate=0.0005, heatmap_blur_sigma=4.0, anchor_weight=300.0,
    ),
    "sigma=4+anchor=200": OptimizationConfig(
        num_steps=25, learning_rate=0.0005, heatmap_blur_sigma=4.0, anchor_weight=200.0,
    ),
    "sigma=4+anchor=100": OptimizationConfig(
        num_steps=25, learning_rate=0.0005, heatmap_blur_sigma=4.0, anchor_weight=100.0,
    ),
    "anneal=8->1+anchor=500+steps=50": OptimizationConfig(
        num_steps=50, learning_rate=0.0005, heatmap_blur_sigma=0.0,
        heatmap_blur_sigma_start=8.0, heatmap_blur_sigma_end=1.0, anchor_weight=500.0,
    ),
    "anneal=8->2+anchor=300+steps=50": OptimizationConfig(
        num_steps=50, learning_rate=0.0005, heatmap_blur_sigma=0.0,
        heatmap_blur_sigma_start=8.0, heatmap_blur_sigma_end=2.0, anchor_weight=300.0,
    ),
    "anneal=8->2+anchor=200+steps=50": OptimizationConfig(
        num_steps=50, learning_rate=0.0005, heatmap_blur_sigma=0.0,
        heatmap_blur_sigma_start=8.0, heatmap_blur_sigma_end=2.0, anchor_weight=200.0,
    ),
    "sigma=4+anchor=100+steps=50": OptimizationConfig(
        num_steps=50, learning_rate=0.0005, heatmap_blur_sigma=4.0, anchor_weight=100.0,
    ),
    "sigma=4+anchor=200+steps=50": OptimizationConfig(
        num_steps=50, learning_rate=0.0005, heatmap_blur_sigma=4.0, anchor_weight=200.0,
    ),
}


def main():
    print("=" * 80)
    print("MediaPipe Multi-Example Config Comparison")
    print("=" * 80)

    # Load all examples
    all_data: list[tuple[str, ExampleData]] = []
    for ex in EXAMPLES:
        name = f"{ex['seq']}_{ex['start_frame']}"
        print(f"\nLoading {name}...")
        data = load_example(
            seq=ex["seq"], camera_name=ex["camera"],
            start_frame=ex["start_frame"], num_frames=ex["num_frames"],
        )
        all_data.append((name, data))

    # Run each config on each example
    all_results: dict[str, list[dict]] = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f"\n--- Config: {cfg_name} ---")
        results = []
        for ex_name, data in all_data:
            r = run_opt(data, cfg)
            results.append(r)
            print(f"  {ex_name:<30} Raw:{r['raw']:>6.2f}  Opt:{r['opt']:>6.2f}  {r['pct']:>5.1f}%")
        all_results[cfg_name] = results

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Average improvement % across 5 examples")
    print("=" * 80)
    print(f"{'Config':<45} {'Avg Impr%':>10} {'Avg Opt':>10} {'Min Impr%':>10}")
    print("-" * 80)
    ranked = []
    for cfg_name, results in all_results.items():
        avg_pct = np.mean([r["pct"] for r in results])
        avg_opt = np.mean([r["opt"] for r in results])
        min_pct = min(r["pct"] for r in results)
        ranked.append((cfg_name, avg_pct, avg_opt, min_pct))
    ranked.sort(key=lambda x: x[1], reverse=True)
    for cfg_name, avg_pct, avg_opt, min_pct in ranked:
        print(f"  {cfg_name:<43} {avg_pct:>8.1f}%  {avg_opt:>8.2f}cm  {min_pct:>8.1f}%")


if __name__ == "__main__":
    main()
