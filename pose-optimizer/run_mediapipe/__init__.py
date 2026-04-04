"""MediaPipe pipeline entrypoint.

Full pipeline: config -> load data -> MediaPipe -> synthetic heatmaps -> optimizer -> evaluate -> output.
"""

import json
import logging
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
    discover_examples,
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
)
from config import RunConfig
from evaluate import (
    evaluate,
    mpjpe_per_joint,
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
from run_mediapipe.detect import detect_poses, load_landmarker, mediapipe_3d_to_camera
from optimize import optimize
from overlay_video import generate_overlay_video
from scoring import generate_synthetic_heatmaps
from skeleton import JOINT_NAMES, EVAL_JOINTS, NUM_JOINTS, PARENTS

logger = logging.getLogger(__name__)


def _example_name(seq: str, start: int) -> str:
    """Build a short example name from sequence name and start frame."""
    return f"{seq}_{start}"


def _find_camera_calibration(
    cameras: dict[str, dict[str, np.ndarray]],
    camera_name: str,
) -> dict[str, np.ndarray]:
    """Find camera calibration by name with fallbacks.

    Args:
        cameras: Dict mapping camera name to calibration dict (K, R, t, etc.).
        camera_name: Desired camera name (e.g. "00_00").

    Returns:
        Calibration dict for the matched camera.

    Raises:
        ValueError: If no matching camera is found.
    """
    if camera_name in cameras:
        return cameras[camera_name]
    cam_name_full = f"00_{camera_name.split('_')[1]}" if "_" in camera_name else camera_name
    if cam_name_full in cameras:
        return cameras[cam_name_full]
    for cname, cal in cameras.items():
        if cname.startswith("00_00"):
            logger.warning("Camera %s not found, falling back to %s", camera_name, cname)
            return cal
    raise ValueError(f"Camera {camera_name} not found in calibration")


def _compute_bone_lengths(positions: np.ndarray) -> np.ndarray:
    """Compute bone lengths from joint positions.

    Args:
        positions: (16, 3) joint positions. [3D:SKELETON_16]

    Returns:
        (16,) bone lengths in meters (index 0 = root, always 0).
    """
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
    landmarker,
) -> dict[str, Any]:
    """Process one CMU Panoptic example end-to-end with MediaPipe pipeline.

    Flow: load calibration -> extract frames -> MediaPipe detection ->
    convert to camera-space [3D:SKELETON_16] -> synthetic heatmaps ->
    FK optimization -> evaluation.

    Args:
        config: Run configuration with optimization hyperparameters.
        seq_name: CMU Panoptic sequence name.
        camera_name: Camera name (e.g. "00_00").
        start_frame: First frame index in the sequence.
        num_frames: Number of frames to process.
        person_idx: Which person to track (0 = first).
        run_dir: Output directory for results.
        landmarker: Pre-loaded MediaPipe PoseLandmarker from load_landmarker().

    Returns:
        Dict of evaluation metrics, or empty dict on failure.
    """
    name = _example_name(seq_name, start_frame)
    logger.info("=" * 60)
    logger.info("Processing: %s", name)
    logger.info("Sequence: %s, Camera: %s", seq_name, camera_name)
    logger.info("Frames: %d-%d, Person: %d", start_frame, start_frame + num_frames - 1, person_idx)
    logger.info("=" * 60)

    data_root = config.data_root
    seq_dir = get_sequence_dir(data_root, seq_name)
    video_path = get_video_path(data_root, seq_name, camera_name)

    # --- 1. Load camera calibration ---
    logger.info("[1/7] Loading camera calibration...")
    cameras = load_calibration(seq_dir)
    cam_calib = _find_camera_calibration(cameras, camera_name)

    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    resolution = cam_calib["resolution"]
    logger.info("    fx=%.1f fy=%.1f cx=%.1f cy=%.1f res=%s", fx, fy, cx, cy, resolution)

    camera = Camera.from_panoptic_calibration(K, R, t, resolution)

    # --- 2. Extract video frames ---
    logger.info("[2/7] Extracting video frames...")
    video_fps = 30.0
    frame_step = max(1, int(round(video_fps / config.target_fps)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    logger.info("    %d frames (step=%d, target %.1f fps)", len(frame_indices), frame_step, config.target_fps)

    frames_rgb = extract_video_frames(video_path, frame_indices)
    if len(frames_rgb) < len(frame_indices):
        logger.warning("Only got %d/%d frames from video", len(frames_rgb), len(frame_indices))
        frame_indices = frame_indices[:len(frames_rgb)]
    if len(frames_rgb) < 2:
        logger.error("Need at least 2 frames. Skipping.")
        return {}

    # --- 3. Run MediaPipe ---
    logger.info("[3/7] Running MediaPipe on %d frames...", len(frames_rgb))
    kp_2d, kp_3d, visibility = detect_poses(frames_rgb, landmarker=landmarker)  # [2D:SKELETON_16], [3D:SKELETON_16], [VIS:SKELETON_16]
    n_detected = sum(1 for v in visibility if v.mean() > 0.3)
    logger.info("    Detected poses in %d/%d frames", n_detected, len(frames_rgb))

    # --- 4. Convert to camera coordinates ---
    logger.info("[4/7] Converting to camera coordinates...")
    det_cam_positions: list[np.ndarray] = []  # list of [3D:SKELETON_16] (16, 3), camera-space meters
    for i in range(len(frames_rgb)):
        pos_cam = mediapipe_3d_to_camera(  # [3D:SKELETON_16]
            kp_3d[i], kp_2d[i], fx, fy, cx, cy,
        )
        det_cam_positions.append(pos_cam)

    z_vals = [float(pos[0, 2]) for pos in det_cam_positions]
    logger.info("    Detector root Z range: %.2f to %.2f m", min(z_vals), max(z_vals))

    # Ground truth -> camera space
    logger.info("    Loading ground truth...")
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam: list[np.ndarray | None] = []  # list of [3D:SKELETON_16] (16, 3), camera-space meters
    for gt in gt_world:
        if gt is not None:
            gt_cam_pts = camera.world_to_camera(gt) * 0.01  # cm -> meters [3D:SKELETON_16]
            gt_cam.append(gt_cam_pts)
        else:
            gt_cam.append(None)
    n_gt = sum(1 for g in gt_cam if g is not None)
    logger.info("    Ground truth available for %d/%d frames", n_gt, len(frame_indices))

    # --- 5. Optimize ---
    logger.info("[5/7] Running FK optimization...")

    optimized_3d, bone_lengths_final, loss_history = optimize(
        raw_3d=det_cam_positions,
        camera=camera,
        config=config.optimization,
        target_2d=kp_2d,
        visibility=visibility,
        verbose=False,
    )

    # --- 6. Evaluate ---
    logger.info("[6/7] Evaluating...")
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

        # Per-joint breakdown (camera coordinates, not root-relative)
        det_eval = det_arr[:, EVAL_JOINTS, :]  # [3D:SKELETON_16_EVAL]
        gt_eval = gt_arr[:, EVAL_JOINTS, :]  # [3D:SKELETON_16_EVAL]
        opt_eval = opt_arr[:, EVAL_JOINTS, :]  # [3D:SKELETON_16_EVAL]
        metrics["det_per_joint"] = mpjpe_per_joint(det_eval, gt_eval).tolist()
        metrics["opt_per_joint"] = mpjpe_per_joint(opt_eval, gt_eval).tolist()

        metrics["det_per_frame_mpjpe"] = [
            float(np.mean(np.linalg.norm(det_eval[i] - gt_eval[i], axis=-1)))
            for i in range(len(gt_indices))
        ]
        metrics["opt_per_frame_mpjpe"] = [
            float(np.mean(np.linalg.norm(opt_eval[i] - gt_eval[i], axis=-1)))
            for i in range(len(gt_indices))
        ]

        # Per-frame MPJVE (velocity error)
        if len(gt_indices) >= 3:
            det_vel = np.diff(det_eval, axis=0)
            gt_vel = np.diff(gt_eval, axis=0)
            opt_vel = np.diff(opt_eval, axis=0)
            metrics["det_per_frame_mpjve"] = [
                float(np.mean(np.linalg.norm(det_vel[i] - gt_vel[i], axis=-1)))
                for i in range(len(det_vel))
            ]
            metrics["opt_per_frame_mpjve"] = [
                float(np.mean(np.linalg.norm(opt_vel[i] - gt_vel[i], axis=-1)))
                for i in range(len(opt_vel))
            ]

        gt_bl = _compute_bone_lengths(gt_arr[0])
        det_bl = _compute_bone_lengths(det_arr[0])
        metrics["gt_bone_lengths"] = gt_bl.tolist()
        metrics["det_bone_lengths"] = det_bl.tolist()

        # Log secondary metrics
        logger.info("    Det MPJPE:    %.2f cm", metrics['det_mpjpe'] * 100)
        logger.info("    Opt MPJPE:    %.2f cm", metrics['opt_mpjpe'] * 100)
        logger.info("    Det SI-MPJPE: %.2f cm", metrics['det_si_mpjpe'] * 100)
        logger.info("    Opt SI-MPJPE: %.2f cm", metrics['opt_si_mpjpe'] * 100)
        if "det_mpjve" in metrics:
            logger.info("    Det MPJVE:    %.2f cm/f", metrics['det_mpjve'] * 100)
            logger.info("    Opt MPJVE:    %.2f cm/f", metrics['opt_mpjve'] * 100)

    else:
        logger.info("    No ground truth available for evaluation.")

    # Per-example summary (always printed regardless of GT availability)
    det_vw = metrics.get("det_vw_si_mpjpe")
    opt_vw = metrics.get("opt_vw_si_mpjpe")
    det_str = f"{det_vw * 100:.2f} cm" if det_vw is not None else "N/A"
    opt_str = f"{opt_vw * 100:.2f} cm" if opt_vw is not None else "N/A"
    print(f"[{name}] Det VW-SI-MPJPE: {det_str}  Opt VW-SI-MPJPE: {opt_str}")

    # --- 7. Save results ---
    logger.info("[7/7] Saving results...")
    example_dir = os.path.join(run_dir, name)
    os.makedirs(example_dir, exist_ok=True)

    # results.json -- structured per plan Phase 6 spec
    _METRIC_KEYS = [
        "mpjpe", "p_mpjpe", "si_mpjpe", "vw_mpjpe", "vw_si_mpjpe",
        "mpjve", "si_mpjve", "vw_mpjve", "vw_si_mpjve",
    ]
    results_data: dict[str, Any] = {
        "model": "mediapipe",
        "example": name,
        "num_frames": len(frames_rgb),
        "metrics": {k: metrics[f"opt_{k}"] for k in _METRIC_KEYS if f"opt_{k}" in metrics},
        "raw_metrics": {k: metrics[f"det_{k}"] for k in _METRIC_KEYS if f"det_{k}" in metrics},
        "per_joint": {
            "det_per_joint": metrics.get("det_per_joint", []),
            "opt_per_joint": metrics.get("opt_per_joint", []),
            "det_per_frame_mpjpe": metrics.get("det_per_frame_mpjpe", []),
            "opt_per_frame_mpjpe": metrics.get("opt_per_frame_mpjpe", []),
            "gt_bone_lengths": metrics.get("gt_bone_lengths", []),
            "det_bone_lengths": metrics.get("det_bone_lengths", []),
        },
        "config": config.model_dump(),
    }
    results_path = os.path.join(example_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    logger.info("    Saved results: %s", results_path)

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
    logger.info("    Saved trajectories: %s", traj_path)

    # Graphs
    if config.generate_graphs:
        graphs_dir = os.path.join(example_dir, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
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
        if "det_per_frame_mpjve" in metrics:
            generate_per_frame_mpjve(
                metrics["det_per_frame_mpjve"],
                metrics.get("opt_per_frame_mpjve"),
                graphs_dir,
            )
        generate_summary(
            det_cam_positions, optimized_3d, gt_cam,
            loss_history, metrics, bone_lengths_final,
            graphs_dir, title=name,
        )
        logger.info("    Saved graphs: %s", graphs_dir)

        # summary.png copy to example root
        summary_src = os.path.join(graphs_dir, "summary.png")
        summary_dst = os.path.join(example_dir, "summary.png")
        if os.path.exists(summary_src) and not os.path.exists(summary_dst):
            import shutil
            shutil.copy2(summary_src, summary_dst)

    # Overlay video -- use synthetic heatmaps for visualization
    if config.generate_video:
        target_2d_arr = np.array(kp_2d)  # (F, 16, 2) [2D:SKELETON_16]
        synthetic_heatmaps, synthetic_affine = generate_synthetic_heatmaps(  # [HEATMAP:SKELETON_16]
            target_2d_arr,
            image_size=camera.image_size,
            heatmap_size=64,
            sigma=config.optimization.heatmap_sigma,
        )

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

    return metrics


def run_pipeline(config: RunConfig) -> None:
    """Run the full MediaPipe pipeline on all examples in config.

    Orchestrates [3D:SKELETON_16] data throughout: detection, synthetic
    heatmap generation, optimization, evaluation, and result output.

    Args:
        config: RunConfig with examples, optimization params, and output settings.
    """
    if config.output_dir:
        run_dir = config.output_dir
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        run_dir = os.path.join("output", f"mediapipe_{timestamp}")

    os.makedirs(run_dir, exist_ok=True)

    # Auto-discover examples if none specified
    examples = config.examples
    if not examples:
        logger.info("No examples specified — auto-discovering sequences in %s", config.data_root)
        examples = discover_examples(config.data_root)
        logger.info("Found %d sequences", len(examples))

    logger.info("=" * 60)
    logger.info("MediaPipe Pose Estimation Pipeline")
    logger.info("%d examples to process", len(examples))
    logger.info("Run: %s", run_dir)
    logger.info("=" * 60)

    logger.info("Loading MediaPipe model...")
    landmarker = load_landmarker()

    all_metrics: list[dict[str, Any]] = []
    for example in examples:
        try:
            metrics = process_example(
                config,
                example.sequence,
                example.camera,
                example.start_frame,
                example.num_frames,
                example.person_idx,
                run_dir,
                landmarker=landmarker,
            )
            if metrics:
                all_metrics.append(metrics)
        except Exception:
            logger.exception("ERROR processing %s_%d", example.sequence, example.start_frame)
            continue

    landmarker.close()

    if all_metrics:
        generate_aggregate_summary(all_metrics, run_dir)

    logger.info("=" * 60)
    logger.info("Done. %d/%d examples processed.", len(all_metrics), len(examples))
    logger.info("Results: %s", run_dir)
    logger.info("=" * 60)
