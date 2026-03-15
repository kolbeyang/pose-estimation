"""Quick single-example experiment runner with diagnostics.

Usage:
    uv run python experiment.py [--name ATTEMPT_NAME]

Runs the simplest example (171204_pose1_sample), saves:
  - Predictions JSON
  - Mid-video overlay frame (PNG)
  - Raw scoring diagnostics (JSON)
  - Per-frame MPJPE data (JSON)
"""

import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

import config as cfg
from detect import detect_poses, mediapipe_3d_to_camera
from evaluate import compute_comparison
from model.camera import Camera
from optimize import run_optimization
from overlay_video import _project_3d_to_2d, _render_heatmap, _blend_heatmap_additive, _draw_skeleton_2d
from panoptic import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
    world_to_camera,
)
from skeleton import JOINT_NAMES, NUM_JOINTS, BONES


def run_experiment(attempt_name="test"):
    """Run single example and save detailed diagnostics."""
    run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, f"experiment-{attempt_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Use simplest example
    seq_name, camera_name, start_frame, num_frames, person_idx = cfg.EXAMPLES[0]
    name = f"{seq_name}_{start_frame}"

    print(f"Running experiment '{attempt_name}' on {name}")
    print(f"Config: SIGMA={cfg.SIGMA}, POS_W={cfg.POSITION_PENALTY_WEIGHT}, "
          f"BL_LR={cfg.BONE_LENGTH_LR}, "
          f"LR={cfg.LEARNING_RATE}, STEPS={cfg.NUM_STEPS}")
    print(f"Rot per-joint weights: {cfg.ROTATION_PENALTY_PER_JOINT.tolist()}")

    # --- Setup ---
    seq_dir = get_sequence_dir(cfg.PANOPTIC_ROOT, seq_name)
    video_path = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)

    cameras = load_calibration(seq_dir)
    cam_calib = None
    for cname in cameras:
        if cname == camera_name or cname.startswith("00_00"):
            cam_calib = cameras[cname]
            break
    assert cam_calib is not None

    K = cam_calib["K"]
    R_cam = cam_calib["R"]
    t_cam = cam_calib["t"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    camera = Camera(fx=fx, fy=fy, cx=cx, cy=cy,
                    image_size=(cam_calib["resolution"][1], cam_calib["resolution"][0]))

    video_fps = 30.0
    frame_step = max(1, int(round(video_fps / cfg.TARGET_FPS)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))

    frames_rgb = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[:len(frames_rgb)]

    # --- Detect ---
    kp_2d, kp_3d, visibility = detect_poses(frames_rgb)

    # --- Camera coords ---
    mp_cam = []
    for i in range(len(frames_rgb)):
        pos_cam = mediapipe_3d_to_camera(kp_3d[i], kp_2d[i], fx, fy, cx, cy)
        mp_cam.append(pos_cam)

    # --- Ground truth ---
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam = []
    for gt in gt_world:
        if gt is not None:
            gt_cam.append(world_to_camera(gt, R_cam, t_cam) * 0.01)
        else:
            gt_cam.append(None)

    # --- Optimize ---
    result = run_optimization(
        initial_positions_cam=mp_cam,
        target_2d=kp_2d,
        visibility=visibility,
        camera=camera,
    )

    # --- Evaluate ---
    metrics = compute_comparison(result.mediapipe_3d, result.optimized_3d, gt_cam, camera)

    print(f"\n  MP  MPJPE:   {metrics.get('mp_mpjpe', 0)*100:.2f} cm")
    print(f"  Opt MPJPE:   {metrics.get('opt_mpjpe', 0)*100:.2f} cm")
    print(f"  MP  P-MPJPE: {metrics.get('mp_p_mpjpe', 0)*100:.2f} cm")
    print(f"  Opt P-MPJPE: {metrics.get('opt_p_mpjpe', 0)*100:.2f} cm")
    if "mp_mpjve" in metrics:
        print(f"  MP  MPJVE:   {metrics['mp_mpjve']*100:.2f} cm/frame")
        print(f"  Opt MPJVE:   {metrics['opt_mpjve']*100:.2f} cm/frame")

    # --- Save diagnostics JSON ---
    diag = {
        "attempt": attempt_name,
        "example": name,
        "config": {
            "sigma": cfg.SIGMA,
            "position_penalty_weight": cfg.POSITION_PENALTY_WEIGHT,
            "bone_length_lr": cfg.BONE_LENGTH_LR,
            "learning_rate": cfg.LEARNING_RATE,
            "num_steps": cfg.NUM_STEPS,
            "rotation_per_joint_weights": cfg.ROTATION_PENALTY_PER_JOINT.tolist(),
        },
        "metrics": {k: v for k, v in metrics.items()
                    if isinstance(v, (int, float))},
        "loss_history": result.loss_history,
        "score_details_history": result.score_details_history,
        "per_frame_mpjpe_mp": metrics.get("mp_per_frame_mpjpe", []),
        "per_frame_mpjpe_opt": metrics.get("opt_per_frame_mpjpe", []),
        "per_joint_mp": metrics.get("mp_per_joint", []),
        "per_joint_opt": metrics.get("opt_per_joint", []),
        "per_frame_mpjve_mp": metrics.get("mp_per_frame_mpjve", []),
        "per_frame_mpjve_opt": metrics.get("opt_per_frame_mpjve", []),
        "per_joint_mpjve_mp": metrics.get("mp_mpjve_per_joint", []),
        "per_joint_mpjve_opt": metrics.get("opt_mpjve_per_joint", []),
        "bone_lengths_final": result.bone_lengths_final.tolist(),
    }
    diag_path = os.path.join(run_dir, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"  Saved diagnostics: {diag_path}")

    # --- Save mid-video overlay frame ---
    mid_idx = len(frames_rgb) // 2
    frame_rgb = frames_rgb[mid_idx]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]

    mp_2d = kp_2d[mid_idx]
    mp_3d_cam = result.mediapipe_3d[mid_idx]
    opt_3d_cam = result.optimized_3d[mid_idx]
    vis = visibility[mid_idx]

    # Heatmap
    heatmap = _render_heatmap(h, w, mp_2d, vis, cfg.SIGMA)
    frame_bgr = _blend_heatmap_additive(frame_bgr, heatmap)

    # Project 3D to 2D
    mp_proj = _project_3d_to_2d(mp_3d_cam, fx, fy, cx, cy)
    opt_proj = _project_3d_to_2d(opt_3d_cam, fx, fy, cx, cy)

    # GT projection
    gt_proj = None
    if gt_cam[mid_idx] is not None:
        gt_proj = _project_3d_to_2d(gt_cam[mid_idx], fx, fy, cx, cy)

    # White dots for MediaPipe 2D
    for j in range(NUM_JOINTS):
        pt = (int(mp_2d[j, 0]), int(mp_2d[j, 1]))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame_bgr, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)

    # Green skeleton: MediaPipe 3D projected
    _draw_skeleton_2d(frame_bgr, mp_proj, color=(0, 255, 0), thickness=2)

    # Red skeleton: Optimised 3D projected
    _draw_skeleton_2d(frame_bgr, opt_proj, color=(0, 0, 255), thickness=2)

    # Blue skeleton: Ground truth projected
    if gt_proj is not None:
        _draw_skeleton_2d(frame_bgr, gt_proj, color=(255, 100, 0), thickness=2)

    # Labels
    cv2.putText(frame_bgr, f"Frame {frame_indices[mid_idx]} | sigma={cfg.SIGMA}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame_bgr, f"MP MPJPE={metrics.get('mp_mpjpe',0)*100:.1f}cm  Opt={metrics.get('opt_mpjpe',0)*100:.1f}cm",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Legend
    cv2.putText(frame_bgr, "Green=MediaPipe  Red=Optimised  Blue=GT  White=2D target",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    overlay_path = os.path.join(run_dir, "mid_frame_overlay.png")
    cv2.imwrite(overlay_path, frame_bgr)
    print(f"  Saved overlay: {overlay_path}")

    return diag


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="test", help="Attempt name")
    args = parser.parse_args()
    run_experiment(args.name)
