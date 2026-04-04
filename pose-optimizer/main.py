"""Unified entrypoint for pose estimation pipeline.

Usage:
    uv run main.py <config.json>

Runs MotionBERT and/or MediaPipe pipelines based on config,
with shared YOLO + StackedHourglass heatmap generation.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np

from camera import Camera
from cmu_data import (
    discover_examples,
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
    vw_si_mpjpe_per_joint,
    vw_si_mpjve_per_joint,
)
from graphs import (
    generate_aggregate_summary,
    generate_bone_lengths_graph,
    generate_cross_pipeline_metrics_comparison,
    generate_cross_pipeline_per_joint_position,
    generate_cross_pipeline_per_joint_velocity,
    generate_loss_curve,
    generate_per_frame_mpjpe,
    generate_per_frame_mpjve,
    generate_per_joint_error_bar,
    generate_summary,
    generate_trajectory_graphs,
)
from optimize import optimize
from overlay_video import generate_overlay_video
from skeleton import JOINT_NAMES, EVAL_JOINTS, NUM_JOINTS, PARENTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _example_name(seq: str, start: int) -> str:
    """Build a short example name from sequence name and start frame."""
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
            logger.warning("Camera %s not found, falling back to %s", camera_name, cname)
            return cal
    raise ValueError(f"Camera {camera_name} not found in calibration")


def _compute_bone_lengths(positions: np.ndarray) -> np.ndarray:
    """Compute bone lengths from joint positions."""
    bl = np.zeros(NUM_JOINTS)
    for j in range(1, NUM_JOINTS):
        p = int(PARENTS[j])
        bl[j] = float(np.linalg.norm(positions[j] - positions[p]))
    return bl


def _evaluate_and_collect_metrics(
    det_cam_positions: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_cam: list[np.ndarray | None],
    camera: Camera,
    name: str,
) -> dict[str, Any]:
    """Run evaluation, compute per-joint/per-frame breakdowns.

    Returns:
        Dict of evaluation metrics.
    """
    metrics: dict[str, Any] = {"name": name}

    gt_indices = [i for i, g in enumerate(gt_cam) if g is not None]
    if len(gt_indices) < 2:
        logger.info("    No ground truth available for evaluation.")
        return metrics

    gt_arr = np.array([gt_cam[i] for i in gt_indices])
    det_arr = np.array([det_cam_positions[i] for i in gt_indices])
    opt_arr = np.array([optimized_3d[i] for i in gt_indices])

    # Raw detector metrics
    det_metrics = evaluate(det_arr, gt_arr, camera)
    for k, v in det_metrics.items():
        metrics[f"det_{k}"] = v

    # Optimized metrics
    opt_metrics = evaluate(opt_arr, gt_arr, camera)
    for k, v in opt_metrics.items():
        metrics[f"opt_{k}"] = v

    # Per-joint breakdown (camera coordinates, not root-relative)
    det_eval = det_arr[:, EVAL_JOINTS, :]
    gt_eval = gt_arr[:, EVAL_JOINTS, :]
    opt_eval = opt_arr[:, EVAL_JOINTS, :]
    metrics["det_per_joint"] = mpjpe_per_joint(det_eval, gt_eval).tolist()
    metrics["opt_per_joint"] = mpjpe_per_joint(opt_eval, gt_eval).tolist()

    # Per-joint VW-SI-MPJPE and VW-SI-MPJVE (for cross-pipeline graphs)
    vis = compute_visibility_weights(gt_arr, camera)[:, EVAL_JOINTS]
    metrics["det_vw_si_mpjpe_per_joint"] = vw_si_mpjpe_per_joint(det_eval, gt_eval, vis).tolist()
    metrics["opt_vw_si_mpjpe_per_joint"] = vw_si_mpjpe_per_joint(opt_eval, gt_eval, vis).tolist()
    if len(gt_indices) >= 3:
        metrics["det_vw_si_mpjve_per_joint"] = vw_si_mpjve_per_joint(det_eval, gt_eval, vis).tolist()
        metrics["opt_vw_si_mpjve_per_joint"] = vw_si_mpjve_per_joint(opt_eval, gt_eval, vis).tolist()

    # Per-frame MPJPE
    metrics["det_per_frame_mpjpe"] = [
        float(np.mean(np.linalg.norm(det_eval[i] - gt_eval[i], axis=-1)))
        for i in range(len(gt_indices))
    ]
    metrics["opt_per_frame_mpjpe"] = [
        float(np.mean(np.linalg.norm(opt_eval[i] - gt_eval[i], axis=-1)))
        for i in range(len(gt_indices))
    ]

    # Per-frame MPJVE
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

    # Bone lengths
    gt_bl = _compute_bone_lengths(gt_arr[0])
    det_bl = _compute_bone_lengths(det_arr[0])
    metrics["gt_bone_lengths"] = gt_bl.tolist()
    metrics["det_bone_lengths"] = det_bl.tolist()

    # Log key metrics
    logger.info("    Det MPJPE:    %.2f cm", metrics['det_mpjpe'] * 100)
    logger.info("    Opt MPJPE:    %.2f cm", metrics['opt_mpjpe'] * 100)
    logger.info("    Det SI-MPJPE: %.2f cm", metrics['det_si_mpjpe'] * 100)
    logger.info("    Opt SI-MPJPE: %.2f cm", metrics['opt_si_mpjpe'] * 100)
    if "det_mpjve" in metrics:
        logger.info("    Det MPJVE:    %.2f cm/f", metrics['det_mpjve'] * 100)
        logger.info("    Opt MPJVE:    %.2f cm/f", metrics['opt_mpjve'] * 100)
    if "det_reprojected_mpjpe_2d" in metrics:
        logger.info("    Det 2D-MPJPE: %.2f px", metrics['det_reprojected_mpjpe_2d'])
        logger.info("    Opt 2D-MPJPE: %.2f px", metrics['opt_reprojected_mpjpe_2d'])

    return metrics


def _save_results(
    metrics: dict[str, Any],
    pipeline_name: str,
    name: str,
    num_frames: int,
    config: RunConfig,
    output_dir: str,
) -> None:
    """Save results.json to output directory."""
    _METRIC_KEYS = [
        "mpjpe", "p_mpjpe", "si_mpjpe", "vw_mpjpe", "vw_si_mpjpe",
        "mpjve", "si_mpjve", "vw_mpjve", "vw_si_mpjve",
        "reprojected_mpjpe_2d",
    ]
    results_data: dict[str, Any] = {
        "model": pipeline_name,
        "example": name,
        "num_frames": num_frames,
        "metrics": {k: metrics[f"opt_{k}"] for k in _METRIC_KEYS if f"opt_{k}" in metrics},
        "raw_metrics": {k: metrics[f"det_{k}"] for k in _METRIC_KEYS if f"det_{k}" in metrics},
        "per_joint": {
            "det_per_joint": metrics.get("det_per_joint", []),
            "opt_per_joint": metrics.get("opt_per_joint", []),
            "det_vw_si_mpjpe_per_joint": metrics.get("det_vw_si_mpjpe_per_joint", []),
            "opt_vw_si_mpjpe_per_joint": metrics.get("opt_vw_si_mpjpe_per_joint", []),
            "det_vw_si_mpjve_per_joint": metrics.get("det_vw_si_mpjve_per_joint", []),
            "opt_vw_si_mpjve_per_joint": metrics.get("opt_vw_si_mpjve_per_joint", []),
            "det_per_frame_mpjpe": metrics.get("det_per_frame_mpjpe", []),
            "opt_per_frame_mpjpe": metrics.get("opt_per_frame_mpjpe", []),
            "gt_bone_lengths": metrics.get("gt_bone_lengths", []),
            "det_bone_lengths": metrics.get("det_bone_lengths", []),
        },
        "config": config.model_dump(),
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    logger.info("    Saved results: %s", results_path)


def _save_trajectories(
    gt_cam: list[np.ndarray | None],
    det_cam_positions: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    camera: Camera,
    output_dir: str,
) -> None:
    """Save trajectories.json to output directory."""
    traj_data = {
        "joint_names": JOINT_NAMES,
        "ground_truth": [g.tolist() if g is not None else None for g in gt_cam],
        "raw_prediction": [d.tolist() for d in det_cam_positions],
        "optimized_prediction": [o.tolist() for o in optimized_3d],
        "camera": camera.to_dict(),
    }
    traj_path = os.path.join(output_dir, "trajectories.json")
    with open(traj_path, "w") as f:
        json.dump(traj_data, f, indent=2)
    logger.info("    Saved trajectories: %s", traj_path)


def _generate_graphs(
    det_cam_positions: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_cam: list[np.ndarray | None],
    loss_history: list[float],
    metrics: dict[str, Any],
    bone_lengths_final: np.ndarray,
    output_dir: str,
    name: str,
) -> None:
    """Generate all per-example graphs."""
    graphs_dir = os.path.join(output_dir, "graphs")
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

    # summary.png copy to output root
    summary_src = os.path.join(graphs_dir, "summary.png")
    summary_dst = os.path.join(output_dir, "summary.png")
    if os.path.exists(summary_src) and not os.path.exists(summary_dst):
        import shutil
        shutil.copy2(summary_src, summary_dst)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(config_path: str) -> None:
    """Run the unified pose estimation pipeline."""
    config = load_config(config_path)

    # Determine output directory
    if config.output_dir:
        run_dir = config.output_dir
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        run_dir = os.path.join("output", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Auto-discover examples if none specified
    examples = config.examples
    if not examples:
        logger.info("No examples specified -- auto-discovering sequences in %s", config.data_root)
        examples = discover_examples(config.data_root)
        logger.info("Found %d sequences", len(examples))

    run_motionbert = config.motionbert is not None
    run_mediapipe = config.mediapipe is not None

    if not run_motionbert and not run_mediapipe:
        logger.error("Config must have at least one of 'motionbert' or 'mediapipe' sections.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Unified Pose Estimation Pipeline")
    pipelines = []
    if run_motionbert:
        pipelines.append("MotionBERT")
    if run_mediapipe:
        pipelines.append("MediaPipe")
    logger.info("Pipelines: %s", " + ".join(pipelines))
    logger.info("%d examples to process", len(examples))
    logger.info("Run: %s", run_dir)
    logger.info("=" * 60)

    # Load models
    from run_motionbert.detect import (
        load_yolo_sh_models,
        load_motionbert_model,
        detect_2d_poses,
        run_motionbert as run_mb_3d,
        motionbert_to_camera_space,
    )

    logger.info("Loading YOLO + Stacked Hourglass models...")
    yolo_sh_models = load_yolo_sh_models()

    mb_model = None
    mb_device = yolo_sh_models.device
    if run_motionbert:
        logger.info("Loading MotionBERT model...")
        mb_model = load_motionbert_model()
        mb_model = mb_model.to(mb_device)

    mp_landmarker = None
    mp_detect_poses = None
    mediapipe_3d_to_camera = None
    if run_mediapipe:
        from run_mediapipe.detect import load_landmarker, detect_poses as _mp_detect_poses, mediapipe_3d_to_camera as _mp_3d_to_cam
        mp_detect_poses = _mp_detect_poses
        mediapipe_3d_to_camera = _mp_3d_to_cam
        logger.info("Loading MediaPipe model...")
        mp_landmarker = load_landmarker()

    all_mb_metrics: list[dict[str, Any]] = []
    all_mp_metrics: list[dict[str, Any]] = []

    for example in examples:
        seq_name = example.sequence
        camera_name = example.camera
        start_frame = example.start_frame
        num_frames = example.num_frames
        person_idx = example.person_idx
        name = _example_name(seq_name, start_frame)

        logger.info("=" * 60)
        logger.info("Processing: %s", name)
        logger.info("Sequence: %s, Camera: %s", seq_name, camera_name)
        logger.info("Frames: %d-%d, Person: %d", start_frame, start_frame + num_frames - 1, person_idx)
        logger.info("=" * 60)

        try:
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
                continue

            # --- 3. Shared YOLO + SH heatmaps ---
            logger.info("[3/7] Running YOLO + Stacked Hourglass on %d frames...", len(frames_rgb))
            (
                kp_2d,        # list of [2D:SKELETON_16] (16, 2)
                visibility,   # list of [VIS:SKELETON_16] (16,)
                heatmaps,     # list of [HEATMAP:MPII_16] (16, 64, 64)
                mpii_kp_2d,   # list of [2D:MPII_16] (16, 3)
                affine,       # (2, 3)
            ) = detect_2d_poses(
                frames_rgb,
                yolo_sh_models,
                sh_batch_size=config.sh_batch_size,
            )

            # --- Ground truth ---
            logger.info("    Loading ground truth...")
            gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
            gt_cam: list[np.ndarray | None] = []
            for gt in gt_world:
                if gt is not None:
                    gt_cam_pts = camera.world_to_camera(gt) * 0.01  # cm -> meters
                    gt_cam.append(gt_cam_pts)
                else:
                    gt_cam.append(None)
            n_gt = sum(1 for g in gt_cam if g is not None)
            logger.info("    Ground truth available for %d/%d frames", n_gt, len(frame_indices))

            example_dir = os.path.join(run_dir, name)
            os.makedirs(example_dir, exist_ok=True)

            # --- 4. MotionBERT pipeline ---
            if run_motionbert:
                logger.info("[4/7] Running MotionBERT pipeline...")
                # 3D lifting
                positions_3d_norm = run_mb_3d(
                    mpii_kp_2d,
                    model=mb_model,
                    device=mb_device,
                    conf_threshold=config.motionbert_conf_threshold,
                )

                # Convert to camera coordinates
                det_cam_positions_mb: list[np.ndarray] = []
                for i in range(len(frames_rgb)):
                    pos_cam = motionbert_to_camera_space(
                        positions_3d_norm[i], kp_2d[i], fx, fy, cx, cy,
                    )
                    det_cam_positions_mb.append(pos_cam)

                # Optimize
                logger.info("    Running FK optimization (MotionBERT)...")
                opt_mb, bl_mb, loss_mb = optimize(
                    raw_3d=det_cam_positions_mb,
                    camera=camera,
                    config=config.optimization,
                    heatmaps=heatmaps,
                    affine=affine,
                    visibility=visibility,
                    verbose=False,
                )

                # Evaluate
                logger.info("    Evaluating MotionBERT...")
                mb_metrics = _evaluate_and_collect_metrics(
                    det_cam_positions_mb, opt_mb, gt_cam, camera, name,
                )

                det_vw = mb_metrics.get("det_vw_si_mpjpe")
                opt_vw = mb_metrics.get("opt_vw_si_mpjpe")
                det_str = f"{det_vw * 100:.2f} cm" if det_vw is not None else "N/A"
                opt_str = f"{opt_vw * 100:.2f} cm" if opt_vw is not None else "N/A"
                print(f"[MB {name}] Det VW-SI-MPJPE: {det_str}  Opt VW-SI-MPJPE: {opt_str}")

                # Save results
                mb_output_dir = os.path.join(example_dir, "motionbert")
                os.makedirs(mb_output_dir, exist_ok=True)
                _save_results(mb_metrics, "motionbert", name, len(frames_rgb), config, mb_output_dir)
                _save_trajectories(gt_cam, det_cam_positions_mb, opt_mb, camera, mb_output_dir)

                if config.generate_graphs:
                    _generate_graphs(
                        det_cam_positions_mb, opt_mb, gt_cam,
                        loss_mb, mb_metrics, bl_mb, mb_output_dir, name,
                    )

                # Overlay video
                is_gen_video = config.generate_video
                if config.motionbert is not None and not config.motionbert.is_generate_heatmap_videos:
                    is_gen_video = False
                if is_gen_video:
                    overlay_path = os.path.join(mb_output_dir, "heatmap_overlay_video.mp4")
                    generate_overlay_video(
                        output_path=overlay_path,
                        frames_rgb=frames_rgb,
                        heatmaps=heatmaps,
                        detector_2d=mpii_kp_2d,
                        detector_3d=det_cam_positions_mb,
                        optimized_3d=opt_mb,
                        camera_fx=fx,
                        camera_fy=fy,
                        camera_cx=cx,
                        camera_cy=cy,
                        affine=affine,
                        frame_indices=frame_indices[:len(frames_rgb)],
                        gt_3d=gt_cam,
                        visibility=visibility,
                        pipeline_name="MotionBERT",
                    )

                all_mb_metrics.append(mb_metrics)

            # --- 5. MediaPipe pipeline ---
            if run_mediapipe:
                assert mp_detect_poses is not None and mediapipe_3d_to_camera is not None
                logger.info("[5/7] Running MediaPipe pipeline...")
                mp_kp_2d, mp_kp_3d, mp_visibility = mp_detect_poses(frames_rgb, landmarker=mp_landmarker)
                n_detected = sum(1 for v in mp_visibility if v.mean() > 0.3)
                logger.info("    Detected poses in %d/%d frames", n_detected, len(frames_rgb))

                # Convert to camera coordinates using solvePnP (with MP's own 2D landmarks)
                det_cam_positions_mp: list[np.ndarray] = []
                for i in range(len(frames_rgb)):
                    pos_cam = mediapipe_3d_to_camera(
                        mp_kp_3d[i], mp_kp_2d[i], fx, fy, cx, cy,
                    )
                    det_cam_positions_mp.append(pos_cam)

                # Optimize using SH heatmaps (same as MotionBERT!)
                logger.info("    Running FK optimization (MediaPipe)...")
                opt_mp, bl_mp, loss_mp = optimize(
                    raw_3d=det_cam_positions_mp,
                    camera=camera,
                    config=config.optimization,
                    heatmaps=heatmaps,
                    affine=affine,
                    visibility=visibility,
                    verbose=False,
                )

                # Evaluate
                logger.info("    Evaluating MediaPipe...")
                mp_metrics = _evaluate_and_collect_metrics(
                    det_cam_positions_mp, opt_mp, gt_cam, camera, name,
                )

                det_vw = mp_metrics.get("det_vw_si_mpjpe")
                opt_vw = mp_metrics.get("opt_vw_si_mpjpe")
                det_str = f"{det_vw * 100:.2f} cm" if det_vw is not None else "N/A"
                opt_str = f"{opt_vw * 100:.2f} cm" if opt_vw is not None else "N/A"
                print(f"[MP {name}] Det VW-SI-MPJPE: {det_str}  Opt VW-SI-MPJPE: {opt_str}")

                # Save results
                mp_output_dir = os.path.join(example_dir, "mediapipe")
                os.makedirs(mp_output_dir, exist_ok=True)
                _save_results(mp_metrics, "mediapipe", name, len(frames_rgb), config, mp_output_dir)
                _save_trajectories(gt_cam, det_cam_positions_mp, opt_mp, camera, mp_output_dir)

                if config.generate_graphs:
                    _generate_graphs(
                        det_cam_positions_mp, opt_mp, gt_cam,
                        loss_mp, mp_metrics, bl_mp, mp_output_dir, name,
                    )

                # Overlay video (with SH heatmaps)
                is_gen_video = config.generate_video
                if config.mediapipe is not None and not config.mediapipe.is_generate_heatmap_videos:
                    is_gen_video = False
                if is_gen_video:
                    overlay_path = os.path.join(mp_output_dir, "heatmap_overlay_video.mp4")
                    # Build per-frame 2D detection arrays with visibility as 3rd column
                    detector_2d_with_vis = []
                    for i in range(len(frames_rgb)):
                        kp_with_vis = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
                        kp_with_vis[:, :2] = mp_kp_2d[i]
                        kp_with_vis[:, 2] = mp_visibility[i]
                        detector_2d_with_vis.append(kp_with_vis)

                    generate_overlay_video(
                        output_path=overlay_path,
                        frames_rgb=frames_rgb,
                        heatmaps=heatmaps,
                        detector_2d=detector_2d_with_vis,
                        detector_3d=det_cam_positions_mp,
                        optimized_3d=opt_mp,
                        camera_fx=fx,
                        camera_fy=fy,
                        camera_cx=cx,
                        camera_cy=cy,
                        affine=affine,
                        frame_indices=frame_indices[:len(frames_rgb)],
                        gt_3d=gt_cam,
                        visibility=visibility,
                        pipeline_name="MediaPipe",
                    )

                all_mp_metrics.append(mp_metrics)

        except Exception:
            logger.exception("ERROR processing %s", name)
            continue

    # --- 6. Cross-pipeline graphs ---
    if config.graphs and run_motionbert and run_mediapipe:
        logger.info("Generating cross-pipeline comparison graphs...")
        cross_dir = os.path.join(run_dir, "cross_pipeline_graphs")
        os.makedirs(cross_dir, exist_ok=True)

        if config.graphs.position_error_per_joint_improvement:
            generate_cross_pipeline_per_joint_position(all_mb_metrics, all_mp_metrics, cross_dir)

        if config.graphs.velocity_error_per_joint_improvement:
            generate_cross_pipeline_per_joint_velocity(all_mb_metrics, all_mp_metrics, cross_dir)

        if config.graphs.metrics_comparison:
            generate_cross_pipeline_metrics_comparison(all_mb_metrics, all_mp_metrics, cross_dir)

        logger.info("Saved cross-pipeline graphs: %s", cross_dir)
    elif config.graphs:
        logger.warning("Cross-pipeline graphs require both motionbert and mediapipe sections in config; skipping")

    # --- 7. Aggregate summaries ---
    if all_mb_metrics:
        generate_aggregate_summary(all_mb_metrics, run_dir, prefix="motionbert")
    if all_mp_metrics:
        generate_aggregate_summary(all_mp_metrics, run_dir, prefix="mediapipe")

    # --- 8. Run-level results.json ---
    run_results: dict[str, Any] = {"config": config.model_dump()}
    if all_mb_metrics:
        run_results["motionbert"] = all_mb_metrics
    if all_mp_metrics:
        run_results["mediapipe"] = all_mp_metrics
    run_results_path = os.path.join(run_dir, "results.json")
    with open(run_results_path, "w") as f:
        json.dump(run_results, f, indent=2, default=str)
    logger.info("Saved run-level results: %s", run_results_path)

    if mp_landmarker is not None:
        mp_landmarker.close()

    logger.info("=" * 60)
    logger.info("Done. Examples processed: MB=%d, MP=%d", len(all_mb_metrics), len(all_mp_metrics))
    logger.info("Results: %s", run_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: uv run main.py <config.json>")
        sys.exit(1)

    main(sys.argv[1])
