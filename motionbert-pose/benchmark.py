"""Benchmark script for the MotionBERT pose optimization pipeline.

Runs all examples with per-stage timing and saves results to JSON.
Optionally profiles the optimization loop for per-step timing breakdown.
"""

import json
import os
import time
from typing import Any

import numpy as np

import config as cfg
from camera import Camera
from detect import detect_poses, motionbert_to_camera_space
from evaluate import compute_comparison_with_optimization
from optimize import run_optimization, run_optimization_batched
from panoptic import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
    world_to_camera,
)
from skeleton import JOINT_NAMES


def _find_camera_calibration(
    cameras: dict[str, dict[str, np.ndarray]],
    camera_name: str,
) -> dict[str, np.ndarray]:
    """Find camera calibration by name with fallbacks."""
    if camera_name in cameras:
        return cameras[camera_name]
    cam_name_full: str = f"00_{camera_name.split('_')[1]}" if "_" in camera_name else camera_name
    if cam_name_full in cameras:
        return cameras[cam_name_full]
    for cname, cal in cameras.items():
        if cname.startswith("00_00"):
            return cal
    raise ValueError(f"Camera {camera_name} not found in calibration")


def benchmark_example(
    seq_name: str,
    camera_name: str,
    start_frame: int,
    num_frames: int,
    person_idx: int,
    profile_optimization: bool = False,
    num_steps: int | None = None,
    learning_rate: float | None = None,
    bone_length_lr: float | None = None,
    heatmap_blur_sigma: float | None = None,
    skip_visualization: bool = True,
    use_batched: bool = False,
) -> dict[str, Any]:
    """Run one example with timing instrumentation.

    Args:
        seq_name: Sequence name.
        camera_name: Camera identifier.
        start_frame: Start frame index.
        num_frames: Number of frames.
        person_idx: Person index.
        profile_optimization: If True, profile the optimization loop.
        num_steps: Override NUM_STEPS.
        learning_rate: Override LEARNING_RATE.
        bone_length_lr: Override BONE_LENGTH_LR.
        heatmap_blur_sigma: Override HEATMAP_BLUR_SIGMA.
        skip_visualization: Skip graph/video generation for speed.

    Returns:
        Dict with timing and metrics.
    """
    name: str = f"{seq_name}_{start_frame}"
    result: dict[str, Any] = {"name": name, "num_frames": 0}

    # Temporarily override config if needed
    orig_num_steps = cfg.NUM_STEPS
    orig_lr = cfg.LEARNING_RATE
    orig_bl_lr = cfg.BONE_LENGTH_LR
    orig_blur = cfg.HEATMAP_BLUR_SIGMA
    if num_steps is not None:
        cfg.NUM_STEPS = num_steps
    if learning_rate is not None:
        cfg.LEARNING_RATE = learning_rate
    if bone_length_lr is not None:
        cfg.BONE_LENGTH_LR = bone_length_lr
    if heatmap_blur_sigma is not None:
        cfg.HEATMAP_BLUR_SIGMA = heatmap_blur_sigma

    try:
        total_start: float = time.perf_counter()

        # --- Detection phase ---
        det_start: float = time.perf_counter()

        seq_dir: str = get_sequence_dir(cfg.PANOPTIC_ROOT, seq_name)
        video_path: str = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)
        cameras = load_calibration(seq_dir)
        cam_calib = _find_camera_calibration(cameras, camera_name)

        K = cam_calib["K"]
        R = cam_calib["R"]
        t = cam_calib["t"]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        resolution = cam_calib["resolution"]

        camera: Camera = Camera.from_panoptic_calibration(K, resolution)

        video_fps: float = 30.0
        frame_step: int = max(1, int(round(video_fps / cfg.TARGET_FPS)))
        frame_indices: list[int] = list(range(start_frame, start_frame + num_frames, frame_step))

        frames_rgb: list[np.ndarray] = extract_video_frames(video_path, frame_indices)
        if len(frames_rgb) < len(frame_indices):
            frame_indices = frame_indices[:len(frames_rgb)]
        if len(frames_rgb) < 2:
            print(f"    ERROR: Need at least 2 frames for {name}. Skipping.")
            return result

        detection_result = detect_poses(frames_rgb, return_timing=True)
        kp_2d = detection_result[0]
        visibility = detection_result[1]
        heatmaps = detection_result[2]
        mpii_kp_2d = detection_result[3]
        affine = detection_result[4]
        positions_3d_norm = detection_result[5]
        detection_timing: dict[str, float] = detection_result[6]

        det_cam_positions: list[np.ndarray] = []
        for i in range(len(frames_rgb)):
            pos_cam = motionbert_to_camera_space(
                positions_3d_norm[i], kp_2d[i], fx, fy, cx, cy,
            )
            det_cam_positions.append(pos_cam)

        det_end: float = time.perf_counter()
        detection_time: float = det_end - det_start

        # --- Ground truth ---
        gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
        gt_cam: list[np.ndarray | None] = []
        for gt in gt_world:
            if gt is not None:
                gt_cam.append(world_to_camera(gt, R, t) * 0.01)
            else:
                gt_cam.append(None)

        # --- Optimization phase ---
        opt_start: float = time.perf_counter()
        opt_fn = run_optimization_batched if use_batched else run_optimization
        optimized_3d, bone_lengths_final, loss_history, profile_data = opt_fn(
            initial_positions_cam=det_cam_positions,
            target_2d=kp_2d,
            visibility=visibility,
            camera=camera,
            heatmaps=heatmaps,
            affine=affine,
            profile=profile_optimization,
        )
        opt_end: float = time.perf_counter()
        optimization_time: float = opt_end - opt_start

        # --- Evaluation phase ---
        eval_start: float = time.perf_counter()
        metrics = compute_comparison_with_optimization(
            detector_3d=det_cam_positions,
            optimized_3d=optimized_3d,
            gt_3d=gt_cam,
            camera=camera,
            detections_2d=kp_2d,
            visibility=visibility,
        )
        eval_end: float = time.perf_counter()
        evaluation_time: float = eval_end - eval_start

        total_end: float = time.perf_counter()
        total_time: float = total_end - total_start

        n_frames: int = len(frames_rgb)
        result.update({
            "num_frames": n_frames,
            "detection_time_s": round(detection_time, 3),
            "yolo_time_s": detection_timing.get("yolo_s", 0.0),
            "stacked_hourglass_time_s": detection_timing.get("stacked_hourglass_s", 0.0),
            "motionbert_time_s": detection_timing.get("motionbert_s", 0.0),
            "detection_postprocess_time_s": detection_timing.get("postprocess_s", 0.0),
            "optimization_time_s": round(optimization_time, 3),
            "evaluation_time_s": round(evaluation_time, 3),
            "total_time_s": round(total_time, 3),
            "latency_per_frame_s": round(total_time / n_frames, 3),
            "det_mpjpe": metrics.get("det_mpjpe"),
            "det_p_mpjpe": metrics.get("det_p_mpjpe"),
            "det_szi_mpjpe": metrics.get("det_szi_mpjpe"),
            "det_mpjve": metrics.get("det_mpjve"),
            "opt_mpjpe": metrics.get("opt_mpjpe"),
            "opt_p_mpjpe": metrics.get("opt_p_mpjpe"),
            "opt_szi_mpjpe": metrics.get("opt_szi_mpjpe"),
            "opt_mpjve": metrics.get("opt_mpjve"),
            "improvement": metrics.get("improvement"),
            "loss_history": loss_history,
        })

        if profile_data is not None:
            result["profile"] = profile_data

    finally:
        # Restore original config
        cfg.NUM_STEPS = orig_num_steps
        cfg.LEARNING_RATE = orig_lr
        cfg.BONE_LENGTH_LR = orig_bl_lr
        cfg.HEATMAP_BLUR_SIGMA = orig_blur

    return result


def run_benchmark(
    examples: list[tuple[str, str, int, int, int]] | None = None,
    profile_first: bool = False,
    num_steps: int | None = None,
    learning_rate: float | None = None,
    bone_length_lr: float | None = None,
    heatmap_blur_sigma: float | None = None,
    experiment_name: str = "baseline",
    use_batched: bool = False,
) -> dict[str, Any]:
    """Run benchmark on all examples and save results.

    Args:
        examples: List of examples to run (defaults to cfg.EXAMPLES).
        profile_first: Profile optimization on first example only.
        num_steps: Override NUM_STEPS for all examples.
        learning_rate: Override LEARNING_RATE.
        bone_length_lr: Override BONE_LENGTH_LR.
        heatmap_blur_sigma: Override HEATMAP_BLUR_SIGMA.
        experiment_name: Name for the experiment JSON file.
        use_batched: Use batched FK and scoring.

    Returns:
        Full results dict.
    """
    if examples is None:
        examples = cfg.EXAMPLES

    results: dict[str, Any] = {
        "experiment_name": experiment_name,
        "config": {
            "NUM_STEPS": num_steps if num_steps is not None else cfg.NUM_STEPS,
            "LEARNING_RATE": learning_rate if learning_rate is not None else cfg.LEARNING_RATE,
            "BONE_LENGTH_LR": bone_length_lr if bone_length_lr is not None else cfg.BONE_LENGTH_LR,
            "SIGMA": cfg.SIGMA,
            "HEATMAP_BLUR_SIGMA": heatmap_blur_sigma if heatmap_blur_sigma is not None else cfg.HEATMAP_BLUR_SIGMA,
            "POSITION_PENALTY_WEIGHT": cfg.POSITION_PENALTY_WEIGHT,
            "INIT_ANCHOR_WEIGHT": cfg.INIT_ANCHOR_WEIGHT,
            "CONFIDENCE_EPSILON": cfg.CONFIDENCE_EPSILON,
        },
        "examples": [],
    }

    for idx, example in enumerate(examples):
        seq_name, camera_name, start_frame, n_frames, person_idx = example
        name: str = f"{seq_name}_{start_frame}"
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(examples)}] {name}")
        print(f"{'='*60}")

        try:
            do_profile: bool = profile_first and idx == 0
            ex_result = benchmark_example(
                seq_name, camera_name, start_frame, n_frames, person_idx,
                profile_optimization=do_profile,
                num_steps=num_steps,
                learning_rate=learning_rate,
                bone_length_lr=bone_length_lr,
                heatmap_blur_sigma=heatmap_blur_sigma,
                use_batched=use_batched,
            )
            results["examples"].append(ex_result)

            if ex_result.get("opt_mpjpe") is not None:
                print(f"  Det MPJPE: {ex_result['det_mpjpe']*100:.2f} cm")
                print(f"  Opt MPJPE: {ex_result['opt_mpjpe']*100:.2f} cm")
                print(f"  Detection: {ex_result['detection_time_s']:.1f}s "
                      f"(YOLO={ex_result.get('yolo_time_s', 0):.1f}s, "
                      f"SH={ex_result.get('stacked_hourglass_time_s', 0):.1f}s, "
                      f"MB={ex_result.get('motionbert_time_s', 0):.1f}s), "
                      f"Optimization: {ex_result['optimization_time_s']:.1f}s, "
                      f"Eval: {ex_result['evaluation_time_s']:.1f}s")

            if do_profile and "profile" in ex_result:
                prof = ex_result["profile"]
                print(f"\n  Profile (per step averages):")
                print(f"    FK+Projection: {np.mean(prof['fk_projection_ms']):.1f} ms")
                print(f"    Scoring:       {np.mean(prof['scoring_ms']):.1f} ms")
                print(f"    Backward:      {np.mean(prof['backward_ms']):.1f} ms")
                print(f"    Step total:    {np.mean(prof['step_total_ms']):.1f} ms")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["examples"].append({"name": name, "error": str(e)})

    # Aggregate
    valid = [e for e in results["examples"] if e.get("opt_mpjpe") is not None]
    if valid:
        results["aggregate"] = {
            "n_examples": len(valid),
            "mean_det_mpjpe": float(np.mean([e["det_mpjpe"] for e in valid])),
            "mean_opt_mpjpe": float(np.mean([e["opt_mpjpe"] for e in valid])),
            "mean_det_p_mpjpe": float(np.mean([e["det_p_mpjpe"] for e in valid])),
            "mean_opt_p_mpjpe": float(np.mean([e["opt_p_mpjpe"] for e in valid])),
            "mean_det_mpjve": float(np.mean([e.get("det_mpjve", 0) for e in valid])),
            "mean_opt_mpjve": float(np.mean([e.get("opt_mpjve", 0) for e in valid])),
            "mean_detection_time_s": float(np.mean([e["detection_time_s"] for e in valid])),
            "mean_yolo_time_s": float(np.mean([e.get("yolo_time_s", 0) for e in valid])),
            "mean_stacked_hourglass_time_s": float(np.mean([e.get("stacked_hourglass_time_s", 0) for e in valid])),
            "mean_motionbert_time_s": float(np.mean([e.get("motionbert_time_s", 0) for e in valid])),
            "mean_optimization_time_s": float(np.mean([e["optimization_time_s"] for e in valid])),
            "mean_evaluation_time_s": float(np.mean([e["evaluation_time_s"] for e in valid])),
            "mean_total_time_s": float(np.mean([e["total_time_s"] for e in valid])),
            "mean_latency_per_frame_s": float(np.mean([e["latency_per_frame_s"] for e in valid])),
            "mean_improvement": float(np.mean([e.get("improvement", 0) for e in valid])),
        }

        print(f"\n{'='*60}")
        print(f"  AGGREGATE ({len(valid)} examples)")
        print(f"{'='*60}")
        agg = results["aggregate"]
        print(f"  Mean Det MPJPE:  {agg['mean_det_mpjpe']*100:.2f} cm")
        print(f"  Mean Opt MPJPE:  {agg['mean_opt_mpjpe']*100:.2f} cm")
        print(f"  Mean Det P-MPJPE: {agg['mean_det_p_mpjpe']*100:.2f} cm")
        print(f"  Mean Opt P-MPJPE: {agg['mean_opt_p_mpjpe']*100:.2f} cm")
        print(f"  Mean Improvement: {agg['mean_improvement']*100:+.2f} cm")
        print(f"  Mean Detection:   {agg['mean_detection_time_s']:.1f}s "
              f"(YOLO={agg['mean_yolo_time_s']:.1f}s, "
              f"SH={agg['mean_stacked_hourglass_time_s']:.1f}s, "
              f"MB={agg['mean_motionbert_time_s']:.1f}s)")
        print(f"  Mean Optimization:{agg['mean_optimization_time_s']:.1f}s")
        print(f"  Mean Total:       {agg['mean_total_time_s']:.1f}s")
        print(f"  Mean Latency/Frame: {agg['mean_latency_per_frame_s']:.3f}s")

    # Save
    experiments_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
    os.makedirs(experiments_dir, exist_ok=True)
    out_path: str = os.path.join(experiments_dir, f"{experiment_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark pose optimization pipeline")
    parser.add_argument("--profile", action="store_true", help="Profile first example")
    parser.add_argument("--name", type=str, default="baseline", help="Experiment name")
    parser.add_argument("--num-steps", type=int, default=None, help="Override NUM_STEPS")
    parser.add_argument("--lr", type=float, default=None, help="Override LEARNING_RATE")
    parser.add_argument("--bl-lr", type=float, default=None, help="Override BONE_LENGTH_LR")
    parser.add_argument("--blur", type=float, default=None, help="Override HEATMAP_BLUR_SIGMA")
    parser.add_argument("--examples", type=int, default=None, help="Limit to first N examples")
    parser.add_argument("--batched", action="store_true", help="Use batched FK and scoring")
    args = parser.parse_args()

    examples = cfg.EXAMPLES
    if args.examples is not None:
        examples = examples[:args.examples]

    run_benchmark(
        examples=examples,
        profile_first=args.profile,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        bone_length_lr=args.bl_lr,
        heatmap_blur_sigma=args.blur,
        experiment_name=args.name,
        use_batched=args.batched,
    )
