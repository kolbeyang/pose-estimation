"""Full MotionBERT pose estimation pipeline on CMU Panoptic examples.

For each example:
  1. Extract video frames from CMU HD video.
  2. Run MotionBERT to get 2D + 3D poses.
  3. Load CMU 3D ground truth + camera calibration.
  4. Convert to camera coordinates.
  5. Evaluate against ground truth.
  6. Save predictions JSON + graphs.
"""

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

import config as cfg
from camera import Camera
from detect import detect_poses, motionbert_to_camera_space
from evaluate import compute_comparison, compute_comparison_with_optimization
from graphs import (
    generate_aggregate_summary,
    generate_per_frame_mpjpe,
    generate_per_joint_error_bar,
)
from models import CameraParams, ExampleResult
from optimize import run_optimization
from panoptic import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
    world_to_camera,
)
from skeleton import JOINT_NAMES, EVAL_JOINT_NAMES


def _example_name(seq: str, start: int) -> str:
    """Generate a name for an example."""
    return f"{seq}_{start}"


def _find_camera_calibration(
    cameras: dict[str, dict[str, np.ndarray]],
    camera_name: str,
) -> dict[str, np.ndarray]:
    """Find camera calibration by name with fallbacks.

    Args:
        cameras: Dict of camera calibrations.
        camera_name: Requested camera name (e.g. "00_00").

    Returns:
        Camera calibration dict.

    Raises:
        ValueError: If camera not found.
    """
    # Try exact match
    if camera_name in cameras:
        return cameras[camera_name]

    # Try "00_XX" format
    cam_name_full: str = f"00_{camera_name.split('_')[1]}" if "_" in camera_name else camera_name
    if cam_name_full in cameras:
        return cameras[cam_name_full]

    # Fallback: first HD camera starting with "00_00"
    for cname, cal in cameras.items():
        if cname.startswith("00_00"):
            print(f"    WARNING: Camera {camera_name} not found, falling back to {cname}")
            return cal

    raise ValueError(f"Camera {camera_name} not found in calibration")


def process_example(
    seq_name: str,
    camera_name: str,
    start_frame: int,
    num_frames: int,
    person_idx: int,
    run_dir: str,
) -> dict[str, Any]:
    """Process one CMU Panoptic example end-to-end.

    Args:
        seq_name: Sequence name.
        camera_name: Camera identifier.
        start_frame: Start frame index.
        num_frames: Number of frames.
        person_idx: Person index.
        run_dir: Output directory for this run.

    Returns:
        Metrics dict.
    """
    name: str = _example_name(seq_name, start_frame)
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"  Sequence: {seq_name}, Camera: {camera_name}")
    print(f"  Frames: {start_frame}-{start_frame + num_frames - 1}, Person: {person_idx}")
    print(f"{'='*60}")

    seq_dir: str = get_sequence_dir(cfg.PANOPTIC_ROOT, seq_name)
    video_path: str = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)

    # --- 1. Load camera calibration ---
    print("\n  [1/5] Loading camera calibration...")
    cameras: dict[str, dict[str, np.ndarray]] = load_calibration(seq_dir)
    cam_calib: dict[str, np.ndarray] = _find_camera_calibration(cameras, camera_name)

    K: np.ndarray = cam_calib["K"]
    R: np.ndarray = cam_calib["R"]
    t: np.ndarray = cam_calib["t"]
    fx: float = float(K[0, 0])
    fy: float = float(K[1, 1])
    cx: float = float(K[0, 2])
    cy: float = float(K[1, 2])
    resolution: tuple[int, int] = cam_calib["resolution"]
    print(f"    fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} res={resolution}")

    camera: Camera = Camera.from_panoptic_calibration(K, resolution)
    camera_params: CameraParams = CameraParams(
        fx=fx, fy=fy, cx=cx, cy=cy,
        image_width=int(resolution[0]),
        image_height=int(resolution[1]),
    )

    # --- 2. Extract video frames ---
    print("\n  [2/5] Extracting video frames...")
    video_fps: float = 30.0
    frame_step: int = max(1, int(round(video_fps / cfg.TARGET_FPS)))
    frame_indices: list[int] = list(range(start_frame, start_frame + num_frames, frame_step))
    print(f"    {len(frame_indices)} frames (step={frame_step}, target {cfg.TARGET_FPS} fps)")

    frames_rgb: list[np.ndarray] = extract_video_frames(video_path, frame_indices)
    if len(frames_rgb) < len(frame_indices):
        print(f"    WARNING: Only got {len(frames_rgb)}/{len(frame_indices)} frames from video")
        frame_indices = frame_indices[:len(frames_rgb)]
    if len(frames_rgb) < 2:
        print("    ERROR: Need at least 2 frames. Skipping.")
        return {}

    # --- 3. Run MotionBERT ---
    print(f"\n  [3/5] Running MotionBERT on {len(frames_rgb)} frames...")
    kp_2d: list[np.ndarray]
    kp_3d: list[np.ndarray]
    visibility: list[np.ndarray]
    heatmaps: list[np.ndarray]
    affine: np.ndarray
    positions_3d_norm: np.ndarray
    cs_params: dict[str, float]
    kp_2d, kp_3d, visibility, heatmaps, affine, positions_3d_norm, cs_params = detect_poses(frames_rgb)

    # --- 4. Convert to camera coordinates ---
    print("\n  [4/5] Converting to camera coordinates...")
    scale: float = cs_params["scale"]
    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        dist_coeffs: np.ndarray | None = cam_calib.get("distCoef")
        pos_cam: np.ndarray = motionbert_to_camera_space(
            positions_3d_norm[i], kp_2d[i], scale, fx, fy, cx, cy,
            dist_coeffs=dist_coeffs,
            visibility=visibility[i],
        )
        det_cam_positions.append(pos_cam)

    # Sanity check
    z_vals: list[float] = [float(pos[0, 2]) for pos in det_cam_positions]
    print(f"    Detector root Z range: {min(z_vals):.2f} to {max(z_vals):.2f} m")

    # Ground truth -> camera space
    print("    Loading ground truth...")
    gt_world: list[np.ndarray | None] = load_ground_truth_sequence(
        seq_dir, frame_indices, person_idx
    )
    gt_cam: list[np.ndarray | None] = []
    for gt in gt_world:
        if gt is not None:
            gt_cam_cm: np.ndarray = world_to_camera(gt, R, t)
            gt_cam_m: np.ndarray = gt_cam_cm * 0.01  # cm -> meters
            gt_cam.append(gt_cam_m)
        else:
            gt_cam.append(None)
    n_gt: int = sum(1 for g in gt_cam if g is not None)
    print(f"    Ground truth available for {n_gt}/{len(frame_indices)} frames")

    if n_gt > 0:
        gt_z_vals: list[float] = [
            float(g[0, 2]) for g in gt_cam if g is not None
        ]
        print(f"    GT root Z range: {min(gt_z_vals):.2f} to {max(gt_z_vals):.2f} m")

    # --- 5. Optimize (Phase 2) ---
    print(f"\n  [5/7] Running FK optimization...")

    # For joints where Stacked Hourglass is unreliable (below confidence threshold),
    # use MotionBERT's projected 2D as the optimization target instead of garbage 2D.
    improved_target_2d: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        target: np.ndarray = kp_2d[i].copy()
        mb_projected: np.ndarray = camera.world_to_image(det_cam_positions[i])
        for j in range(17):
            if visibility[i][j] < cfg.FK_TARGET_CONF_THRESHOLD:
                target[j] = mb_projected[j]
        improved_target_2d.append(target)

    optimized_3d: list[np.ndarray]
    bone_lengths_final: np.ndarray
    loss_history: list[float]
    optimized_3d, bone_lengths_final, loss_history = run_optimization(
        initial_positions_cam=det_cam_positions,
        target_2d=improved_target_2d,
        visibility=visibility,
        camera=camera,
    )

    # --- 6. Evaluate ---
    print(f"\n  [6/7] Evaluating...")
    metrics: dict[str, Any] = compute_comparison_with_optimization(
        detector_3d=det_cam_positions,
        optimized_3d=optimized_3d,
        gt_3d=gt_cam,
    )
    metrics["name"] = name

    if "det_mpjpe" in metrics:
        print(f"    Det MPJPE:   {metrics['det_mpjpe']*100:.2f} cm")
        print(f"    Det P-MPJPE: {metrics['det_p_mpjpe']*100:.2f} cm")
    if "det_mpjpe_no_ankles" in metrics:
        print(f"    Det MPJPE (no ankles): {metrics['det_mpjpe_no_ankles']*100:.2f} cm")
    if "opt_mpjpe" in metrics:
        print(f"    Opt MPJPE:   {metrics['opt_mpjpe']*100:.2f} cm")
        print(f"    Opt P-MPJPE: {metrics['opt_p_mpjpe']*100:.2f} cm")
    if "opt_mpjpe_no_ankles" in metrics:
        print(f"    Opt MPJPE (no ankles): {metrics['opt_mpjpe_no_ankles']*100:.2f} cm")
    if "improvement" in metrics:
        improv_cm: float = metrics["improvement"] * 100
        print(f"    Improvement: {improv_cm:+.2f} cm")
    if "det_mpjpe" not in metrics:
        print("    No ground truth available for evaluation.")

    # --- 7. Save results ---
    print(f"\n  [7/7] Saving results...")
    predictions_dir: str = os.path.join(run_dir, "predictions")
    example_graph_dir: str = os.path.join(run_dir, "graphs", name)

    # Prediction JSON
    os.makedirs(predictions_dir, exist_ok=True)
    pred_data: dict[str, Any] = {
        "sequence": seq_name,
        "camera": camera_name,
        "start_frame": start_frame,
        "num_frames": len(frames_rgb),
        "frame_indices": frame_indices[:len(frames_rgb)],
        "joint_names": JOINT_NAMES,
        "eval_joint_names": EVAL_JOINT_NAMES,
        "camera_intrinsics": camera_params.model_dump(),
        "metrics": {k: v for k, v in metrics.items() if k != "name"},
        "bone_lengths_final": bone_lengths_final.tolist(),
        "loss_history": loss_history,
        "frames": [],
    }
    for i in range(len(frames_rgb)):
        frame_data: dict[str, Any] = {
            "frame_idx": frame_indices[i] if i < len(frame_indices) else i,
            "detector_3d": det_cam_positions[i].tolist(),
            "optimized_3d": optimized_3d[i].tolist(),
            "detector_2d": kp_2d[i].tolist(),
            "visibility": visibility[i].tolist(),
        }
        if i < len(gt_cam) and gt_cam[i] is not None:
            frame_data["ground_truth_3d"] = gt_cam[i].tolist()
        else:
            frame_data["ground_truth_3d"] = None
        pred_data["frames"].append(frame_data)

    json_path: str = os.path.join(predictions_dir, f"{name}.json")
    with open(json_path, "w") as f:
        json.dump(pred_data, f, indent=2)
    print(f"    Saved predictions: {json_path}")

    # Graphs
    if "det_per_joint" in metrics:
        generate_per_joint_error_bar(
            metrics["det_per_joint"],
            example_graph_dir,
            opt_per_joint=metrics.get("opt_per_joint"),
        )
    if "det_per_frame_mpjpe" in metrics:
        generate_per_frame_mpjpe(
            metrics["det_per_frame_mpjpe"],
            example_graph_dir,
            opt_per_frame=metrics.get("opt_per_frame_mpjpe"),
        )
    print(f"    Saved graphs: {example_graph_dir}")

    # Build ExampleResult for summary
    example_result: ExampleResult = ExampleResult(
        name=name,
        sequence=seq_name,
        camera=camera_name,
        start_frame=start_frame,
        num_frames=len(frames_rgb),
        camera_params=camera_params,
        mpjpe=metrics.get("det_mpjpe"),
        p_mpjpe=metrics.get("det_p_mpjpe"),
        mpjpe_cm=metrics.get("det_mpjpe", 0) * 100 if "det_mpjpe" in metrics else None,
        p_mpjpe_cm=metrics.get("det_p_mpjpe", 0) * 100 if "det_p_mpjpe" in metrics else None,
        opt_mpjpe=metrics.get("opt_mpjpe"),
        opt_p_mpjpe=metrics.get("opt_p_mpjpe"),
        opt_mpjpe_cm=metrics.get("opt_mpjpe", 0) * 100 if "opt_mpjpe" in metrics else None,
        opt_p_mpjpe_cm=metrics.get("opt_p_mpjpe", 0) * 100 if "opt_p_mpjpe" in metrics else None,
        improvement_cm=metrics.get("improvement", 0) * 100 if "improvement" in metrics else None,
    )
    print(f"    {example_result.model_dump_json(indent=2)}")

    return metrics


def main() -> None:
    """Process all 10 examples, save aggregate results."""
    timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir: str = os.path.join(cfg.TRAINING_RUNS_DIR, f"motionbert-run-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("  MotionBERT Pose Estimation Pipeline")
    print(f"  {len(cfg.EXAMPLES)} examples to process")
    print(f"  Run: {run_dir}")
    print("=" * 60)

    all_metrics: list[dict[str, Any]] = []
    for example in cfg.EXAMPLES:
        seq_name: str
        camera_name: str
        start_frame: int
        num_frames: int
        person_idx: int
        seq_name, camera_name, start_frame, num_frames, person_idx = example
        try:
            metrics: dict[str, Any] = process_example(
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
        graphs_dir: str = os.path.join(run_dir, "graphs")
        generate_aggregate_summary(all_metrics, graphs_dir)

    print(f"\n{'='*60}")
    print(f"  Done. {len(all_metrics)}/{len(cfg.EXAMPLES)} examples processed.")
    print(f"  Results: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
