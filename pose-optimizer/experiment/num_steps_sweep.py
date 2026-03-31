"""Num steps + rotation penalty scalar sweep for MotionBert optimization.

Tests combinations of num_steps and rotation_penalty_scalar.

Usage:
    uv run python experiment/num_steps_sweep.py
"""

import json
import os
import sys
import time

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

# Same 5 representative examples
EXAMPLES = [
    ("171204_pose3_2000", "171204_pose3", "00_00", 2000, 150, 0),
    ("171204_pose3_4000", "171204_pose3", "00_00", 4000, 150, 0),
    ("171204_pose2_8000", "171204_pose2", "00_00", 8000, 150, 0),
    ("171204_pose1_5000", "171204_pose1", "00_00", 5000, 150, 0),
    ("171204_pose2_10000", "171204_pose2", "00_00", 10000, 150, 0),
]

# Grid: num_steps x rotation_penalty_scalar
NUM_STEPS_VALUES = [10, 20, 30, 50]
SCALAR_VALUES = [10, 50, 100, 500]

DATA_ROOT = "data/panoptic-toolbox"
TARGET_FPS = 10.0


def load_detection_data(seq_name, camera_name, start_frame, num_frames, person_idx):
    """Run detection and return intermediates."""
    seq_dir = get_sequence_dir(DATA_ROOT, seq_name)
    video_path = get_video_path(DATA_ROOT, seq_name, camera_name)

    cameras = load_calibration(seq_dir)
    cam_calib = cameras[camera_name]
    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    resolution = cam_calib["resolution"]
    camera = Camera.from_panoptic_calibration(K, R, t, resolution)

    video_fps = 30.0
    frame_step = max(1, int(round(video_fps / TARGET_FPS)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    frames_rgb = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[:len(frames_rgb)]

    kp_2d, visibility, heatmaps, mpii_kp_2d, affine, positions_3d_norm = detect_poses(
        frames_rgb, sh_batch_size=32, conf_threshold=0.0,
    )

    det_cam_positions = []
    for i in range(len(frames_rgb)):
        pos_cam = motionbert_to_camera_space(
            positions_3d_norm[i], kp_2d[i], fx, fy, cx, cy,
        )
        det_cam_positions.append(pos_cam)

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


def run_optimization(data, num_steps, rotation_penalty_scalar):
    """Run optimization with given params."""
    config = OptimizationConfig(
        num_steps=num_steps,
        learning_rate=0.002,
        bone_length_lr=0.0001,
        position_penalty_weight=500.0,
        rotation_penalty_scalar=rotation_penalty_scalar,
        heatmap_blur_sigma=4.0,
        heatmap_sigma=50.0,
    )

    optimized_3d, _, _ = optimize(
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
    print("  Num Steps + Rotation Penalty Scalar Grid Sweep")
    print("=" * 80)

    # Detection
    all_data = {}
    for name, seq, cam, start, nf, pidx in EXAMPLES:
        print(f"\n--- Detecting: {name} ---")
        t0 = time.time()
        all_data[name] = load_detection_data(seq, cam, start, nf, pidx)
        print(f"    Done in {time.time()-t0:.1f}s")

    # Raw detector metrics
    raw_metrics = {}
    for name in all_data:
        metrics = evaluate_predictions(all_data[name], all_data[name]["det_cam_positions"])
        raw_metrics[name] = metrics

    # Grid sweep
    print("\n--- Grid Sweep ---")
    results = {}
    for ns in NUM_STEPS_VALUES:
        for scalar in SCALAR_VALUES:
            key = (ns, scalar)
            results[key] = {}
            for name in all_data:
                optimized = run_optimization(all_data[name], ns, scalar)
                metrics = evaluate_predictions(all_data[name], optimized)
                results[key][name] = metrics

            # Print summary
            avg_raw = np.mean([raw_metrics[n]["vw_si_mpjpe"] * 100 for n, *_ in EXAMPLES])
            avg_opt = np.mean([results[key][n]["vw_si_mpjpe"] * 100 for n, *_ in EXAMPLES])
            avg_pct = (avg_raw - avg_opt) / avg_raw * 100
            improved = sum(1 for n, *_ in EXAMPLES
                          if results[key][n]["vw_si_mpjpe"] < raw_metrics[n]["vw_si_mpjpe"])
            print(f"  steps={ns:3d}, scalar={scalar:4d}: avg VW-SI-MPJPE={avg_opt:.2f} cm ({avg_pct:+.1f}%), improved={improved}/5")

    # Summary table
    print("\n" + "=" * 100)
    print("  VW-SI-MPJPE Average % Change (positive = optimizer helped)")
    print("=" * 100)
    avg_raw = np.mean([raw_metrics[n]["vw_si_mpjpe"] * 100 for n, *_ in EXAMPLES])
    print(f"  Raw average: {avg_raw:.2f} cm\n")

    header = f"{'':>12}"
    for scalar in SCALAR_VALUES:
        header += f" | scalar={scalar:>4}"
    print(header)
    print("-" * 80)

    for ns in NUM_STEPS_VALUES:
        row = f"  steps={ns:>3}"
        for scalar in SCALAR_VALUES:
            key = (ns, scalar)
            avg_opt = np.mean([results[key][n]["vw_si_mpjpe"] * 100 for n, *_ in EXAMPLES])
            avg_pct = (avg_raw - avg_opt) / avg_raw * 100
            improved = sum(1 for n, *_ in EXAMPLES
                           if results[key][n]["vw_si_mpjpe"] < raw_metrics[n]["vw_si_mpjpe"])
            row += f" | {avg_pct:+6.1f}% {improved}/5"
        print(row)

    # MPJVE table
    print("\n" + "=" * 100)
    print("  MPJVE Average % Change (positive = optimizer helped)")
    print("=" * 100)
    avg_raw_v = np.mean([raw_metrics[n]["mpjve"] * 100 for n, *_ in EXAMPLES])
    print(f"  Raw average: {avg_raw_v:.2f} cm/frame\n")

    header = f"{'':>12}"
    for scalar in SCALAR_VALUES:
        header += f" | scalar={scalar:>4}"
    print(header)
    print("-" * 80)

    for ns in NUM_STEPS_VALUES:
        row = f"  steps={ns:>3}"
        for scalar in SCALAR_VALUES:
            key = (ns, scalar)
            avg_opt_v = np.mean([results[key][n]["mpjve"] * 100 for n, *_ in EXAMPLES])
            avg_pct_v = (avg_raw_v - avg_opt_v) / avg_raw_v * 100
            row += f" | {avg_pct_v:+6.1f}%    "
        print(row)

    # Per-example detail for best config
    best_key = min(results.keys(),
                   key=lambda k: np.mean([results[k][n]["vw_si_mpjpe"] for n, *_ in EXAMPLES]))
    print(f"\n  Best config: steps={best_key[0]}, scalar={best_key[1]}")
    print(f"  {'Example':<25} {'Raw':>8} {'Opt':>8} {'%Chg':>8}")
    print(f"  {'-'*55}")
    for name, *_ in EXAMPLES:
        raw_val = raw_metrics[name]["vw_si_mpjpe"] * 100
        opt_val = results[best_key][name]["vw_si_mpjpe"] * 100
        pct = (raw_val - opt_val) / raw_val * 100
        print(f"  {name:<25} {raw_val:8.2f} {opt_val:8.2f} {pct:+8.1f}%")


if __name__ == "__main__":
    main()
