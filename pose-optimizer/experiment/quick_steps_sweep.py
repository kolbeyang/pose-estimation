"""Quick sweep of very low num_steps values to see if fewer steps helps more.

Usage:
    uv run python experiment/quick_steps_sweep.py
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

# Focus on examples that still regress at steps=10
EXAMPLES = [
    ("171204_pose3_2000", "171204_pose3", "00_00", 2000, 150, 0),   # -12.8% at steps=10
    ("171204_pose1_16000", "171204_pose1", "00_00", 16000, 150, 0), # -13.9% at steps=10
    ("171204_pose2_8000", "171204_pose2", "00_00", 8000, 150, 0),   # +1.6% at steps=10
    ("171204_pose1_5000", "171204_pose1", "00_00", 5000, 150, 0),   # +7.6% at steps=10
    ("171204_pose1_sample_0", "171204_pose1_sample", "00_00", 0, 100, 0),  # +19.3% at steps=10
]

STEPS_VALUES = [3, 5, 7, 10, 15]
DATA_ROOT = "data/panoptic-toolbox"
TARGET_FPS = 10.0


def load_detection_data(seq_name, camera_name, start_frame, num_frames, person_idx):
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
        pos_cam = motionbert_to_camera_space(positions_3d_norm[i], kp_2d[i], fx, fy, cx, cy)
        det_cam_positions.append(pos_cam)
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam = []
    for gt in gt_world:
        if gt is not None:
            gt_cam.append(camera.world_to_camera(gt) * 0.01)
        else:
            gt_cam.append(None)
    return {
        "camera": camera, "det_cam_positions": det_cam_positions,
        "heatmaps": heatmaps, "affine": affine, "visibility": visibility, "gt_cam": gt_cam,
    }


def run_opt(data, num_steps):
    config = OptimizationConfig(
        num_steps=num_steps, learning_rate=0.002, bone_length_lr=0.0001,
        position_penalty_weight=500.0, rotation_penalty_scalar=10.0,
        heatmap_blur_sigma=4.0, heatmap_sigma=50.0,
    )
    optimized_3d, _, _ = optimize(
        raw_3d=data["det_cam_positions"], camera=data["camera"], config=config,
        heatmaps=data["heatmaps"], affine=data["affine"], visibility=data["visibility"],
        verbose=False,
    )
    return optimized_3d


def eval_pred(data, predictions):
    gt_cam = data["gt_cam"]
    gt_indices = [i for i, g in enumerate(gt_cam) if g is not None]
    if len(gt_indices) < 2:
        return None
    gt_arr = np.array([gt_cam[i] for i in gt_indices])
    pred_arr = np.array([predictions[i] for i in gt_indices])
    return evaluate(pred_arr, gt_arr, data["camera"])


def main():
    print("=" * 80)
    print("  Quick Low-Steps Sweep")
    print("=" * 80)

    all_data = {}
    for name, seq, cam, start, nf, pidx in EXAMPLES:
        print(f"\n--- Detecting: {name} ---")
        all_data[name] = load_detection_data(seq, cam, start, nf, pidx)

    raw_metrics = {}
    for name in all_data:
        raw_metrics[name] = eval_pred(all_data[name], all_data[name]["det_cam_positions"])

    print("\n--- Sweep ---")
    results = {}
    for ns in STEPS_VALUES:
        results[ns] = {}
        for name in all_data:
            optimized = run_opt(all_data[name], ns)
            results[ns][name] = eval_pred(all_data[name], optimized)

    # Table
    print("\n" + "=" * 100)
    print("  VW-SI-MPJPE (cm) and % change")
    print("=" * 100)
    header = f"{'Example':<28} {'Raw':>6}"
    for ns in STEPS_VALUES:
        header += f" | s={ns:>2}   %chg"
    print(header)
    print("-" * 100)

    for name, *_ in EXAMPLES:
        raw_val = raw_metrics[name]["vw_si_mpjpe"] * 100
        row = f"{name:<28} {raw_val:6.2f}"
        for ns in STEPS_VALUES:
            opt_val = results[ns][name]["vw_si_mpjpe"] * 100
            pct = (raw_val - opt_val) / raw_val * 100
            row += f" | {opt_val:5.2f} {pct:+5.1f}%"
        print(row)

    print("-" * 100)
    avg_raw = np.mean([raw_metrics[n]["vw_si_mpjpe"] * 100 for n, *_ in EXAMPLES])
    row = f"{'AVERAGE':<28} {avg_raw:6.2f}"
    for ns in STEPS_VALUES:
        avg_opt = np.mean([results[ns][n]["vw_si_mpjpe"] * 100 for n, *_ in EXAMPLES])
        avg_pct = (avg_raw - avg_opt) / avg_raw * 100
        improved = sum(1 for n, *_ in EXAMPLES if results[ns][n]["vw_si_mpjpe"] < raw_metrics[n]["vw_si_mpjpe"])
        row += f" | {avg_opt:5.2f} {avg_pct:+5.1f}%"
    print(row)

    row = f"{'IMPROVED':<28} {'':>6}"
    for ns in STEPS_VALUES:
        improved = sum(1 for n, *_ in EXAMPLES if results[ns][n]["vw_si_mpjpe"] < raw_metrics[n]["vw_si_mpjpe"])
        row += f" |  {improved}/5      "
    print(row)


if __name__ == "__main__":
    main()
