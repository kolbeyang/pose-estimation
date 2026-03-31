"""Rotation penalty scalar sweep for MotionBert optimization.

Runs detection ONCE per example, then re-runs optimization with different
rotation_penalty_scalar values to find the best setting.

Usage:
    uv run python experiment/rotation_penalty_sweep.py
"""

import json
import os
import sys
import time

# Add parent dir to path
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np

from camera import Camera
from cmu_data import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
)
from config import OptimizationConfig
from evaluate import evaluate
from optimize import optimize
from run_motionbert.detect import detect_poses, motionbert_to_camera_space

# 5 representative examples: 2 worst, 1 middle, 2 best
EXAMPLES = [
    # (sequence, camera, start_frame, num_frames, person_idx)
    ("171204_pose3_2000", "171204_pose3", "00_00", 2000, 150, 0),   # worst: -55.8%
    ("171204_pose3_4000", "171204_pose3", "00_00", 4000, 150, 0),   # bad: -49.3%
    ("171204_pose2_8000", "171204_pose2", "00_00", 8000, 150, 0),   # middle: -7.0%
    ("171204_pose1_5000", "171204_pose1", "00_00", 5000, 150, 0),   # good: +8.0%
    ("171204_pose2_10000", "171204_pose2", "00_00", 10000, 150, 0), # best: +9.2%
]

SCALARS = [10, 50, 100, 200, 500]
DATA_ROOT = "data/panoptic-toolbox"
TARGET_FPS = 10.0


def load_detection_data(seq_name, camera_name, start_frame, num_frames, person_idx):
    """Run detection and return all intermediates needed for optimization."""
    seq_dir = get_sequence_dir(DATA_ROOT, seq_name)
    video_path = get_video_path(DATA_ROOT, seq_name, camera_name)

    # Camera
    cameras = load_calibration(seq_dir)
    cam_calib = cameras[camera_name]
    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    resolution = cam_calib["resolution"]
    camera = Camera.from_panoptic_calibration(K, R, t, resolution)

    # Frames
    video_fps = 30.0
    frame_step = max(1, int(round(video_fps / TARGET_FPS)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    frames_rgb = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[:len(frames_rgb)]

    # Detection
    kp_2d, visibility, heatmaps, mpii_kp_2d, affine, positions_3d_norm = detect_poses(
        frames_rgb, sh_batch_size=32, conf_threshold=0.0,
    )

    # Camera-space positions
    det_cam_positions = []
    for i in range(len(frames_rgb)):
        pos_cam = motionbert_to_camera_space(
            positions_3d_norm[i], kp_2d[i], fx, fy, cx, cy,
        )
        det_cam_positions.append(pos_cam)

    # Ground truth
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam = []
    for gt in gt_world:
        if gt is not None:
            gt_cam.append(camera.world_to_camera(gt) * 0.01)
        else:
            gt_cam.append(None)

    return {
        "camera": camera,
        "det_cam_positions": det_cam_positions,
        "heatmaps": heatmaps,
        "affine": affine,
        "visibility": visibility,
        "gt_cam": gt_cam,
    }


def run_optimization(data, rotation_penalty_scalar):
    """Run optimization with a given rotation_penalty_scalar."""
    config = OptimizationConfig(
        num_steps=50,
        learning_rate=0.002,
        bone_length_lr=0.0001,
        position_penalty_weight=500.0,
        rotation_penalty_scalar=rotation_penalty_scalar,
        heatmap_blur_sigma=4.0,
        heatmap_sigma=50.0,
    )

    optimized_3d, bone_lengths, loss_history = optimize(
        raw_3d=data["det_cam_positions"],
        camera=data["camera"],
        config=config,
        heatmaps=data["heatmaps"],
        affine=data["affine"],
        visibility=data["visibility"],
        verbose=False,
    )
    return optimized_3d


def evaluate_predictions(data, predictions):
    """Evaluate predictions against ground truth."""
    gt_cam = data["gt_cam"]
    camera = data["camera"]

    gt_indices = [i for i, g in enumerate(gt_cam) if g is not None]
    if len(gt_indices) < 2:
        return None

    gt_arr = np.array([gt_cam[i] for i in gt_indices])
    pred_arr = np.array([predictions[i] for i in gt_indices])

    return evaluate(pred_arr, gt_arr, camera)


def main():
    print("=" * 80)
    print("  Rotation Penalty Scalar Sweep for MotionBert")
    print("=" * 80)

    # Phase 1: Run detection for all examples (cached)
    all_data = {}
    for name, seq, cam, start, nf, pidx in EXAMPLES:
        print(f"\n--- Detecting: {name} ---")
        t0 = time.time()
        all_data[name] = load_detection_data(seq, cam, start, nf, pidx)
        print(f"    Detection done in {time.time()-t0:.1f}s")

    # Phase 2: Evaluate raw detector
    print("\n--- Evaluating raw detector ---")
    raw_metrics = {}
    for name in all_data:
        data = all_data[name]
        metrics = evaluate_predictions(data, data["det_cam_positions"])
        raw_metrics[name] = metrics
        print(f"  {name}: VW-SI-MPJPE = {metrics['vw_si_mpjpe']*100:.2f} cm")

    # Phase 3: Sweep rotation_penalty_scalar
    print("\n--- Sweeping rotation_penalty_scalar ---")
    results = {}  # scalar -> {name -> metrics}
    for scalar in SCALARS:
        print(f"\n  scalar={scalar}")
        results[scalar] = {}
        for name in all_data:
            t0 = time.time()
            optimized = run_optimization(all_data[name], scalar)
            metrics = evaluate_predictions(all_data[name], optimized)
            results[scalar][name] = metrics
            dt = time.time() - t0
            pct = (raw_metrics[name]["vw_si_mpjpe"] - metrics["vw_si_mpjpe"]) / raw_metrics[name]["vw_si_mpjpe"] * 100
            print(f"    {name}: VW-SI-MPJPE = {metrics['vw_si_mpjpe']*100:.2f} cm ({pct:+.1f}%) [{dt:.1f}s]")

    # Phase 4: Print results table
    print("\n" + "=" * 120)
    print("  RESULTS TABLE: VW-SI-MPJPE (cm) and % change from raw detector")
    print("=" * 120)

    # Header
    header = f"{'Example':<25} {'Raw':>8}"
    for s in SCALARS:
        header += f" | s={s:>3}    %chg"
    print(header)
    print("-" * 120)

    # Per-example rows
    for name, seq, cam, start, nf, pidx in EXAMPLES:
        raw_val = raw_metrics[name]["vw_si_mpjpe"] * 100
        row = f"{name:<25} {raw_val:8.2f}"
        for s in SCALARS:
            opt_val = results[s][name]["vw_si_mpjpe"] * 100
            pct = (raw_val - opt_val) / raw_val * 100
            row += f" | {opt_val:6.2f} {pct:+6.1f}%"
        print(row)

    # Average row
    print("-" * 120)
    avg_raw = np.mean([raw_metrics[n]["vw_si_mpjpe"] * 100 for n, *_ in EXAMPLES])
    row = f"{'AVERAGE':<25} {avg_raw:8.2f}"
    for s in SCALARS:
        avg_opt = np.mean([results[s][n]["vw_si_mpjpe"] * 100 for n, *_ in EXAMPLES])
        avg_pct = (avg_raw - avg_opt) / avg_raw * 100
        row += f" | {avg_opt:6.2f} {avg_pct:+6.1f}%"
    print(row)

    # Improved count
    row = f"{'IMPROVED':<25} {'':>8}"
    for s in SCALARS:
        count = sum(1 for n, *_ in EXAMPLES
                    if results[s][n]["vw_si_mpjpe"] < raw_metrics[n]["vw_si_mpjpe"])
        row += f" |  {count}/{len(EXAMPLES)}       "
    print(row)

    # MPJVE table
    print("\n" + "=" * 120)
    print("  RESULTS TABLE: MPJVE (cm/frame)")
    print("=" * 120)
    header = f"{'Example':<25} {'Raw':>8}"
    for s in SCALARS:
        header += f" | s={s:>3}    %chg"
    print(header)
    print("-" * 120)

    for name, seq, cam, start, nf, pidx in EXAMPLES:
        raw_val = raw_metrics[name]["mpjve"] * 100
        row = f"{name:<25} {raw_val:8.2f}"
        for s in SCALARS:
            opt_val = results[s][name]["mpjve"] * 100
            pct = (raw_val - opt_val) / raw_val * 100
            row += f" | {opt_val:6.2f} {pct:+6.1f}%"
        print(row)

    print("-" * 120)
    avg_raw_v = np.mean([raw_metrics[n]["mpjve"] * 100 for n, *_ in EXAMPLES])
    row = f"{'AVERAGE':<25} {avg_raw_v:8.2f}"
    for s in SCALARS:
        avg_opt_v = np.mean([results[s][n]["mpjve"] * 100 for n, *_ in EXAMPLES])
        avg_pct_v = (avg_raw_v - avg_opt_v) / avg_raw_v * 100
        row += f" | {avg_opt_v:6.2f} {avg_pct_v:+6.1f}%"
    print(row)

    # Save results as JSON
    out_data = {
        "scalars": SCALARS,
        "examples": [name for name, *_ in EXAMPLES],
        "raw": {n: {k: float(v) for k, v in raw_metrics[n].items()} for n, *_ in EXAMPLES},
        "results": {
            str(s): {
                n: {k: float(v) for k, v in results[s][n].items()}
                for n, *_ in EXAMPLES
            }
            for s in SCALARS
        },
    }
    os.makedirs("output", exist_ok=True)
    out_path = "output/rotation_penalty_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
