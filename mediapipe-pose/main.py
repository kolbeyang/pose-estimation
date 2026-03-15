"""Full-body FK pose optimisation pipeline on CMU Panoptic examples.

For each example:
  1. Extract video frames from CMU HD video.
  2. Run MediaPipe to get 2D + 3D poses.
  3. Load CMU 3D ground truth + camera calibration.
  4. Convert everything to camera coordinates.
  5. Optimise FK parameters against 2D detections.
  6. Evaluate against ground truth.
  7. Save predictions JSON + graphs.
"""

import json
import os
from datetime import datetime

import numpy as np

import config as cfg
from detect import detect_poses, mediapipe_3d_to_camera
from evaluate import compute_comparison
from fk import positions_to_fk_params
from graphs import (
    generate_aggregate_summary,
    generate_bone_lengths_graph,
    generate_loss_curve,
    generate_per_frame_mpjpe,
    generate_per_joint_error_bar,
    generate_summary,
    generate_trajectory_graphs,
)
from overlay_video import generate_overlay_video
from model.camera import Camera
from optimize import run_optimization
from panoptic import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
    world_to_camera,
)
from skeleton import JOINT_NAMES, NUM_JOINTS


def _example_name(seq: str, start: int) -> str:
    return f"{seq}_{start}"


def process_example(
    seq_name: str,
    camera_name: str,
    start_frame: int,
    num_frames: int,
    person_idx: int,
    run_dir: str,
) -> dict:
    """Process one CMU Panoptic example end-to-end."""
    name = _example_name(seq_name, start_frame)
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"  Sequence: {seq_name}, Camera: {camera_name}")
    print(f"  Frames: {start_frame}–{start_frame + num_frames - 1}, Person: {person_idx}")
    print(f"{'='*60}")

    seq_dir = get_sequence_dir(cfg.PANOPTIC_ROOT, seq_name)
    video_path = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)

    # --- 1. Load camera calibration ---
    print("\n  [1/7] Loading camera calibration...")
    cameras = load_calibration(seq_dir)
    cam_name_full = f"00_{camera_name.split('_')[1]}" if "_" in camera_name else camera_name
    # HD cameras use panel 00
    cam_calib = None
    for cname in cameras:
        if cname == camera_name or cname == cam_name_full:
            cam_calib = cameras[cname]
            break
    if cam_calib is None:
        # Try matching by panel/node
        for cname, cal in cameras.items():
            if cname.startswith("00_00"):
                cam_calib = cal
                break
    if cam_calib is None:
        raise ValueError(f"Camera {camera_name} not found in calibration")

    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    resolution = cam_calib["resolution"]
    print(f"    fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} res={resolution}")

    camera = Camera(fx=fx, fy=fy, cx=cx, cy=cy, image_size=(resolution[1], resolution[0]))

    # --- 2. Extract video frames ---
    print("\n  [2/7] Extracting video frames...")
    # Subsample: pick every Nth frame to match TARGET_FPS
    video_fps = 30.0  # CMU HD is ~30fps
    frame_step = max(1, int(round(video_fps / cfg.TARGET_FPS)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    print(f"    {len(frame_indices)} frames (step={frame_step}, target {cfg.TARGET_FPS} fps)")

    frames_rgb = extract_video_frames(video_path, frame_indices)
    if len(frames_rgb) < len(frame_indices):
        print(f"    WARNING: Only got {len(frames_rgb)}/{len(frame_indices)} frames from video")
        frame_indices = frame_indices[:len(frames_rgb)]
    if len(frames_rgb) < 2:
        print("    ERROR: Need at least 2 frames. Skipping.")
        return {}

    # --- 3. Run MediaPipe ---
    print(f"\n  [3/7] Running MediaPipe on {len(frames_rgb)} frames...")
    kp_2d, kp_3d, visibility = detect_poses(frames_rgb)
    n_detected = sum(1 for v in visibility if v.mean() > 0.3)
    print(f"    Detected poses in {n_detected}/{len(frames_rgb)} frames")

    # --- 4. Convert to camera coordinates ---
    print("\n  [4/7] Converting to camera coordinates...")
    # MediaPipe 3D → camera space
    mp_cam_positions = []
    for i in range(len(frames_rgb)):
        pos_cam = mediapipe_3d_to_camera(
            kp_3d[i], kp_2d[i], fx, fy, cx, cy,
        )
        mp_cam_positions.append(pos_cam)

    # Ground truth → camera space
    print("    Loading ground truth...")
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam = []
    for gt in gt_world:
        if gt is not None:
            # CMU GT is in centimeters; R,t calibration also uses centimeters.
            # Transform to camera coords first, then convert cm → meters.
            gt_cam_cm = world_to_camera(gt, R, t)
            gt_cam_m = gt_cam_cm * 0.01  # cm → meters
            gt_cam.append(gt_cam_m)
        else:
            gt_cam.append(None)
    n_gt = sum(1 for g in gt_cam if g is not None)
    print(f"    Ground truth available for {n_gt}/{len(frame_indices)} frames")

    # --- 5. Optimise ---
    print("\n  [5/7] Running FK optimisation...")
    result = run_optimization(
        initial_positions_cam=mp_cam_positions,
        target_2d=kp_2d,
        visibility=visibility,
        camera=camera,
    )

    # --- 6. Evaluate ---
    print("\n  [6/7] Evaluating...")
    metrics = compute_comparison(result.mediapipe_3d, result.optimized_3d, gt_cam, camera)
    metrics["name"] = name

    if "mp_mpjpe" in metrics:
        print(f"    MP  MPJPE:   {metrics['mp_mpjpe']*100:.2f} cm")
        print(f"    Opt MPJPE:   {metrics['opt_mpjpe']*100:.2f} cm")
        print(f"    MP  P-MPJPE: {metrics['mp_p_mpjpe']*100:.2f} cm")
        print(f"    Opt P-MPJPE: {metrics['opt_p_mpjpe']*100:.2f} cm")
    if "mp_2d_mpjpe" in metrics:
        print(f"    MP  2D MPJPE: {metrics['mp_2d_mpjpe']:.1f} px")
        print(f"    Opt 2D MPJPE: {metrics['opt_2d_mpjpe']:.1f} px")
    if "mp_mpjpe" not in metrics:
        print("    No ground truth available for evaluation.")

    # --- 7. Save results ---
    print("\n  [7/7] Saving results...")
    predictions_dir = os.path.join(run_dir, "predictions")
    example_graph_dir = os.path.join(run_dir, "graphs", name)

    # Prediction JSON
    os.makedirs(predictions_dir, exist_ok=True)
    pred_data = {
        "sequence": seq_name,
        "camera": camera_name,
        "start_frame": start_frame,
        "num_frames": len(frames_rgb),
        "frame_indices": frame_indices[:len(frames_rgb)],
        "joint_names": JOINT_NAMES,
        "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "metrics": {k: v for k, v in metrics.items() if k != "name"},
        "frames": [],
    }
    for i in range(len(frames_rgb)):
        frame_data = {
            "frame_idx": frame_indices[i] if i < len(frame_indices) else i,
            "mediapipe_3d": result.mediapipe_3d[i].tolist(),
            "optimized_3d": result.optimized_3d[i].tolist(),
            "mediapipe_2d": kp_2d[i].tolist(),
            "visibility": visibility[i].tolist(),
        }
        if gt_cam[i] is not None:
            frame_data["ground_truth_3d"] = gt_cam[i].tolist()
        else:
            frame_data["ground_truth_3d"] = None
        pred_data["frames"].append(frame_data)

    json_path = os.path.join(predictions_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(pred_data, f, indent=2)
    print(f"    Saved predictions: {json_path}")

    # Graphs
    generate_trajectory_graphs(
        result.mediapipe_3d, result.optimized_3d, gt_cam, example_graph_dir,
    )
    generate_loss_curve(result.loss_history, example_graph_dir)
    generate_bone_lengths_graph(
        result.bone_lengths_final, example_graph_dir,
        gt_bone_lengths=metrics.get("gt_bone_lengths"),
        mp_bone_lengths=metrics.get("mp_bone_lengths"),
    )

    if "mp_per_joint" in metrics:
        generate_per_joint_error_bar(
            metrics["mp_per_joint"], metrics["opt_per_joint"], example_graph_dir,
        )
    if "mp_per_frame_mpjpe" in metrics:
        generate_per_frame_mpjpe(
            metrics["mp_per_frame_mpjpe"], metrics["opt_per_frame_mpjpe"],
            example_graph_dir,
        )

    generate_summary(
        result.mediapipe_3d, result.optimized_3d, gt_cam,
        result.loss_history, metrics, result.bone_lengths_final,
        example_graph_dir, title=name,
    )
    print(f"    Saved graphs: {example_graph_dir}")

    # Overlay video
    overlay_path = os.path.join(example_graph_dir, f"{name}_overlay.mp4")
    generate_overlay_video(json_path, overlay_path, sigma=cfg.SIGMA)

    return metrics


def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, f"mediapipe-run-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("  Full-Body FK Pose Optimisation Pipeline")
    print(f"  {len(cfg.EXAMPLES)} examples to process")
    print(f"  Run: {run_dir}")
    print("=" * 60)

    all_metrics = []
    for example in cfg.EXAMPLES:
        seq_name, camera_name, start_frame, num_frames, person_idx = example
        try:
            metrics = process_example(
                seq_name, camera_name, start_frame, num_frames, person_idx,
                run_dir,
            )
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"\n  ERROR processing {seq_name}_{start_frame}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate summary
    if all_metrics:
        graphs_dir = os.path.join(run_dir, "graphs")
        generate_aggregate_summary(all_metrics, graphs_dir)

    print(f"\n{'='*60}")
    print(f"  Done. {len(all_metrics)}/{len(cfg.EXAMPLES)} examples processed.")
    print(f"  Results: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
