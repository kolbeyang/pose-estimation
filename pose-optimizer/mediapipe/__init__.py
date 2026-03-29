"""MediaPipe pipeline entrypoint.

Full pipeline: config -> load data -> MediaPipe -> synthetic heatmaps -> optimizer -> evaluate -> output.

IMPORTANT: This package is named 'mediapipe' which conflicts with the pip
mediapipe package. The detect.py submodule handles this by importing the
pip package before the local package is registered.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np

# Add parent dir to path
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from camera import Camera
from cmu_data import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
)
from config import RunConfig, load_config
from evaluate import (
    evaluate,
    compute_visibility_weights,
    mpjpe_per_joint,
    mpjve_per_joint,
    root_relative,
)
from graphs import (
    generate_aggregate_summary,
    generate_bone_lengths_graph,
    generate_loss_curve,
    generate_per_frame_mpjpe,
    generate_per_frame_mpjve,
    generate_per_joint_error_bar,
    generate_summary,
    generate_trajectory_graphs,
)
from mediapipe.detect import detect_poses, mediapipe_3d_to_camera
from optimize import optimize
from overlay_video import generate_overlay_video
from scoring import generate_synthetic_heatmaps
from skeleton import JOINT_NAMES, EVAL_JOINTS, EVAL_JOINT_NAMES, NUM_JOINTS, PARENTS, DEFAULT_BONE_LENGTHS


def _example_name(seq: str, start: int) -> str:
    return f"{seq}_{start}"


def _find_camera_calibration(
    cameras: dict[str, dict[str, np.ndarray]],
    camera_name: str,
) -> dict[str, np.ndarray]:
    """Find camera calibration by name with fallbacks."""
    if camera_name in cameras:
        return cameras[camera_name]
    cam_name_full = f"00_{camera_name.split('_')[1]}" if "_" in camera_name else camera_name
    if cam_name_full in cameras:
        return cameras[cam_name_full]
    for cname, cal in cameras.items():
        if cname.startswith("00_00"):
            print(f"    WARNING: Camera {camera_name} not found, falling back to {cname}")
            return cal
    raise ValueError(f"Camera {camera_name} not found in calibration")


def _compute_bone_lengths(positions: np.ndarray) -> np.ndarray:
    """Compute bone lengths from positions (K, 3)."""
    bl = np.zeros(NUM_JOINTS)
    for j in range(1, NUM_JOINTS):
        p = int(PARENTS[j])
        bl[j] = float(np.linalg.norm(positions[j] - positions[p]))
    return bl


def process_example(
    config: RunConfig,
    seq_name: str,
    camera_name: str,
    start_frame: int,
    num_frames: int,
    person_idx: int,
    run_dir: str,
) -> dict[str, Any]:
    """Process one CMU Panoptic example end-to-end with MediaPipe pipeline."""
    name = _example_name(seq_name, start_frame)
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"  Sequence: {seq_name}, Camera: {camera_name}")
    print(f"  Frames: {start_frame}-{start_frame + num_frames - 1}, Person: {person_idx}")
    print(f"{'='*60}")

    data_root = config.data_root
    seq_dir = get_sequence_dir(data_root, seq_name)
    video_path = get_video_path(data_root, seq_name, camera_name)

    # --- 1. Load camera calibration ---
    print("\n  [1/7] Loading camera calibration...")
    cameras = load_calibration(seq_dir)
    cam_calib = _find_camera_calibration(cameras, camera_name)

    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    resolution = cam_calib["resolution"]
    print(f"    fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} res={resolution}")

    camera = Camera.from_panoptic_calibration(K, R, t, resolution)

    # --- 2. Extract video frames ---
    print("\n  [2/7] Extracting video frames...")
    video_fps = 30.0
    frame_step = max(1, int(round(video_fps / config.target_fps)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    print(f"    {len(frame_indices)} frames (step={frame_step}, target {config.target_fps} fps)")

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
    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        pos_cam = mediapipe_3d_to_camera(
            kp_3d[i], kp_2d[i], fx, fy, cx, cy,
        )
        det_cam_positions.append(pos_cam)

    z_vals = [float(pos[0, 2]) for pos in det_cam_positions]
    print(f"    Detector root Z range: {min(z_vals):.2f} to {max(z_vals):.2f} m")

    # Ground truth -> camera space
    print("    Loading ground truth...")
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam: list[np.ndarray | None] = []
    for gt in gt_world:
        if gt is not None:
            gt_cam_pts = camera.world_to_camera(gt) * 0.01
            gt_cam.append(gt_cam_pts)
        else:
            gt_cam.append(None)
    n_gt = sum(1 for g in gt_cam if g is not None)
    print(f"    Ground truth available for {n_gt}/{len(frame_indices)} frames")

    # --- 5. Optimize ---
    print(f"\n  [5/7] Running FK optimization...")

    # Generate synthetic heatmaps for overlay video (we pass target_2d to optimizer
    # which generates its own, but we also need them for overlay)
    target_2d_arr = np.array(kp_2d)
    synthetic_heatmaps, synthetic_affine = generate_synthetic_heatmaps(
        target_2d_arr,
        image_size=camera.image_size,
        heatmap_size=64,
        sigma=config.gaussian_sigma,
    )

    optimized_3d, bone_lengths_final, loss_history = optimize(
        raw_3d=det_cam_positions,
        camera=camera,
        config=config.optimization,
        target_2d=kp_2d,
        visibility=visibility,
    )

    # --- 6. Evaluate ---
    print(f"\n  [6/7] Evaluating...")
    metrics: dict[str, Any] = {"name": name}

    gt_indices = [i for i, g in enumerate(gt_cam) if g is not None]
    if len(gt_indices) >= 2:
        gt_arr = np.array([gt_cam[i] for i in gt_indices])
        det_arr = np.array([det_cam_positions[i] for i in gt_indices])
        opt_arr = np.array([optimized_3d[i] for i in gt_indices])

        det_metrics = evaluate(det_arr, gt_arr, camera)
        for k, v in det_metrics.items():
            metrics[f"det_{k}"] = v

        opt_metrics = evaluate(opt_arr, gt_arr, camera)
        for k, v in opt_metrics.items():
            metrics[f"opt_{k}"] = v

        det_rr = root_relative(det_arr)[:, EVAL_JOINTS, :]
        gt_rr = root_relative(gt_arr)[:, EVAL_JOINTS, :]
        opt_rr = root_relative(opt_arr)[:, EVAL_JOINTS, :]
        metrics["det_per_joint"] = mpjpe_per_joint(det_rr, gt_rr).tolist()
        metrics["opt_per_joint"] = mpjpe_per_joint(opt_rr, gt_rr).tolist()

        metrics["det_per_frame_mpjpe"] = [
            float(np.mean(np.linalg.norm(det_rr[i] - gt_rr[i], axis=-1)))
            for i in range(len(gt_indices))
        ]
        metrics["opt_per_frame_mpjpe"] = [
            float(np.mean(np.linalg.norm(opt_rr[i] - gt_rr[i], axis=-1)))
            for i in range(len(gt_indices))
        ]

        gt_bl = _compute_bone_lengths(gt_arr[0])
        det_bl = _compute_bone_lengths(det_arr[0])
        metrics["gt_bone_lengths"] = gt_bl.tolist()
        metrics["det_bone_lengths"] = det_bl.tolist()

        print(f"    Det MPJPE:       {metrics['det_mpjpe']*100:.2f} cm")
        print(f"    Opt MPJPE:       {metrics['opt_mpjpe']*100:.2f} cm")
        print(f"    Det SI-MPJPE:    {metrics['det_si_mpjpe']*100:.2f} cm")
        print(f"    Opt SI-MPJPE:    {metrics['opt_si_mpjpe']*100:.2f} cm")
        print(f"    Det VW-SI-MPJPE: {metrics['det_vw_si_mpjpe']*100:.2f} cm")
        print(f"    Opt VW-SI-MPJPE: {metrics['opt_vw_si_mpjpe']*100:.2f} cm")
        if "det_mpjve" in metrics:
            print(f"    Det MPJVE:       {metrics['det_mpjve']*100:.2f} cm/f")
            print(f"    Opt MPJVE:       {metrics['opt_mpjve']*100:.2f} cm/f")
    else:
        print("    No ground truth available for evaluation.")

    # --- 7. Save results ---
    print(f"\n  [7/7] Saving results...")
    example_dir = os.path.join(run_dir, name)
    graphs_dir = os.path.join(example_dir, "graphs")
    os.makedirs(example_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # results.json
    results_data: dict[str, Any] = {
        "model": "mediapipe",
        "example": name,
        "num_frames": len(frames_rgb),
        "metrics": {k: v for k, v in metrics.items() if k != "name"},
        "config": config.model_dump(),
    }
    results_path = os.path.join(example_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"    Saved results: {results_path}")

    # trajectories.json
    traj_data = {
        "joint_names": JOINT_NAMES,
        "ground_truth": [g.tolist() if g is not None else None for g in gt_cam],
        "raw_prediction": [d.tolist() for d in det_cam_positions],
        "optimized_prediction": [o.tolist() for o in optimized_3d],
        "camera": camera.to_dict(),
    }
    traj_path = os.path.join(example_dir, "trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(traj_data, f, indent=2)
    print(f"    Saved trajectories: {traj_path}")

    # Graphs
    generate_trajectory_graphs(det_cam_positions, optimized_3d, gt_cam, graphs_dir)
    generate_loss_curve(loss_history, graphs_dir)
    generate_bone_lengths_graph(
        bone_lengths_final, graphs_dir,
        gt_bone_lengths=np.array(metrics["gt_bone_lengths"]) if "gt_bone_lengths" in metrics else None,
        det_bone_lengths=np.array(metrics["det_bone_lengths"]) if "det_bone_lengths" in metrics else None,
    )
    if "det_per_joint" in metrics:
        generate_per_joint_error_bar(
            metrics["det_per_joint"], graphs_dir,
            opt_per_joint=metrics.get("opt_per_joint"),
        )
    if "det_per_frame_mpjpe" in metrics:
        generate_per_frame_mpjpe(
            metrics["det_per_frame_mpjpe"], graphs_dir,
            opt_per_frame=metrics.get("opt_per_frame_mpjpe"),
        )
    generate_summary(
        det_cam_positions, optimized_3d, gt_cam,
        loss_history, metrics, bone_lengths_final,
        graphs_dir, title=name,
    )
    print(f"    Saved graphs: {graphs_dir}")

    # Overlay video -- use synthetic heatmaps for visualization
    overlay_path = os.path.join(example_dir, "overlay_video.mp4")

    # For overlay, build per-frame 2D detection arrays with visibility as 3rd column
    detector_2d_with_vis = []
    for i in range(len(frames_rgb)):
        kp_with_vis = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
        kp_with_vis[:, :2] = kp_2d[i]
        kp_with_vis[:, 2] = visibility[i]
        detector_2d_with_vis.append(kp_with_vis)

    generate_overlay_video(
        output_path=overlay_path,
        frames_rgb=frames_rgb,
        heatmaps=[synthetic_heatmaps[i] for i in range(len(frames_rgb))],
        detector_2d=detector_2d_with_vis,
        detector_3d=det_cam_positions,
        optimized_3d=optimized_3d,
        camera_fx=fx,
        camera_fy=fy,
        camera_cx=cx,
        camera_cy=cy,
        affine=synthetic_affine,
        frame_indices=frame_indices[:len(frames_rgb)],
        gt_3d=gt_cam,
        visibility=visibility,
        pipeline_name="MediaPipe",
    )

    # summary.png copy
    summary_src = os.path.join(graphs_dir, "summary.png")
    summary_dst = os.path.join(example_dir, "summary.png")
    if os.path.exists(summary_src) and not os.path.exists(summary_dst):
        import shutil
        shutil.copy2(summary_src, summary_dst)

    return metrics


def run_pipeline(config: RunConfig) -> None:
    """Run the full MediaPipe pipeline on all examples in config."""
    if config.output_dir:
        run_dir = config.output_dir
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        run_dir = os.path.join("output", f"mediapipe_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("  MediaPipe Pose Estimation Pipeline")
    print(f"  {len(config.examples)} examples to process")
    print(f"  Run: {run_dir}")
    print("=" * 60)

    all_metrics: list[dict[str, Any]] = []
    for example in config.examples:
        try:
            metrics = process_example(
                config,
                example.sequence,
                example.camera,
                example.start_frame,
                example.num_frames,
                example.person_idx,
                run_dir,
            )
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"\n  ERROR processing {example.sequence}_{example.start_frame}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_metrics:
        generate_aggregate_summary(all_metrics, run_dir)

    print(f"\n{'='*60}")
    print(f"  Done. {len(all_metrics)}/{len(config.examples)} examples processed.")
    print(f"  Results: {run_dir}")
    print(f"{'='*60}")
