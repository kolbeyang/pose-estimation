"""MotionBERT hyperparameter sweep on single example.

Current: 50 steps, lr=0.001, bone_length_lr=0.005, anneal 16->4
         Raw 23.43 cm -> Opt 16.83 cm (28.1%)

Goal: push opt below 15 cm if possible; validate that config works.

Usage:
    cd pose-optimizer
    uv run python experiment/motionbert_sweep.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from run_motionbert.detect import (
    load_yolo_sh_models,
    load_motionbert_model,
    detect_2d_poses,
    run_motionbert as run_mb_3d,
    motionbert_to_camera_space,
)


def load_motionbert_example(
    seq: str = "171204_pose1_sample",
    camera_name: str = "00_00",
    start_frame: int = 0,
    num_frames: int = 100,
    person_idx: int = 0,
    target_fps: float = 10.0,
    data_root: str = "data/panoptic-toolbox",
):
    """Load and run MotionBERT detection for one example."""
    seq_dir = get_sequence_dir(data_root, seq)
    video_path = get_video_path(data_root, seq, camera_name)

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
    frame_step = max(1, int(round(video_fps / target_fps)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    frames_rgb = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[: len(frames_rgb)]

    print("  Loading YOLO + SH models...")
    yolo_sh_models = load_yolo_sh_models()
    print("  Running YOLO + SH detection...")
    kp_2d, visibility, heatmaps, mpii_kp_2d, affine = detect_2d_poses(
        frames_rgb, yolo_sh_models, sh_batch_size=32
    )

    print("  Loading MotionBERT model...")
    mb_model = load_motionbert_model()
    mb_model = mb_model.to(yolo_sh_models.device)
    print("  Running MotionBERT 3D lift...")
    positions_3d_norm = run_mb_3d(
        mpii_kp_2d, model=mb_model, device=yolo_sh_models.device, conf_threshold=0.0,
    )

    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        pos_cam = motionbert_to_camera_space(positions_3d_norm[i], kp_2d[i], fx, fy, cx, cy)
        det_cam_positions.append(pos_cam)

    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam: list[np.ndarray | None] = []
    for gt in gt_world:
        if gt is not None:
            gt_cam.append(camera.world_to_camera(gt) * 0.01)
        else:
            gt_cam.append(None)

    return {
        "heatmaps": heatmaps,
        "visibility": visibility,
        "affine": affine,
        "det_cam_positions": det_cam_positions,
        "gt_cam": gt_cam,
        "camera": camera,
    }


def run_optimization(data: dict, config: OptimizationConfig) -> dict:
    t0 = time.time()
    opt_3d, bl_final, loss_history = optimize(
        raw_3d=data["det_cam_positions"],
        camera=data["camera"],
        config=config,
        heatmaps=data["heatmaps"],
        affine=data["affine"],
        visibility=data["visibility"],
        verbose=False,
    )
    elapsed = time.time() - t0

    gt_indices = [i for i, g in enumerate(data["gt_cam"]) if g is not None]
    gt_arr = np.array([data["gt_cam"][i] for i in gt_indices])
    det_arr = np.array([data["det_cam_positions"][i] for i in gt_indices])
    opt_arr = np.array([opt_3d[i] for i in gt_indices])

    det_metrics = evaluate(det_arr, gt_arr, data["camera"])
    opt_metrics = evaluate(opt_arr, gt_arr, data["camera"])

    raw_vw = det_metrics["vw_si_mpjpe"] * 100
    opt_vw = opt_metrics["vw_si_mpjpe"] * 100
    improvement_pct = (raw_vw - opt_vw) / raw_vw * 100

    return {
        "raw_vw_si_mpjpe_cm": raw_vw,
        "opt_vw_si_mpjpe_cm": opt_vw,
        "improvement_pct": improvement_pct,
        "elapsed_s": elapsed,
    }


BASELINE = {
    "num_steps": 50,
    "learning_rate": 0.001,
    "bone_length_lr": 0.005,
    "heatmap_blur_sigma": 0.0,
    "heatmap_blur_sigma_start": 16.0,
    "heatmap_blur_sigma_end": 4.0,
    "anchor_weight": 0.0,
    "position_penalty_weight": 500.0,
    "rotation_penalty_scalar": 10.0,
    "confidence_epsilon": 1e-4,
}


def make_config(**overrides) -> OptimizationConfig:
    params = {**BASELINE, **overrides}
    return OptimizationConfig(**params)


def define_sweeps() -> list[tuple[str, OptimizationConfig]]:
    sweeps: list[tuple[str, OptimizationConfig]] = []

    # Baseline
    sweeps.append(("baseline(50,anneal=16->4)", make_config()))

    # More steps
    for s in [75, 100]:
        sweeps.append((f"steps={s}", make_config(num_steps=s)))

    # Anneal endpoints
    sweeps.append(("anneal=16->2", make_config(heatmap_blur_sigma_end=2.0)))
    sweeps.append(("anneal=16->1", make_config(heatmap_blur_sigma_end=1.0)))
    sweeps.append(("anneal=8->2", make_config(heatmap_blur_sigma_start=8.0, heatmap_blur_sigma_end=2.0)))

    # No anchor vs small anchor (MotionBERT is bad, anchor might hurt)
    sweeps.append(("anchor=100", make_config(anchor_weight=100.0)))
    sweeps.append(("anchor=500", make_config(anchor_weight=500.0)))

    # Higher bone_length_lr
    sweeps.append(("bone_lr=0.01", make_config(bone_length_lr=0.01)))

    # Higher LR
    sweeps.append(("lr=0.002", make_config(learning_rate=0.002)))
    sweeps.append(("lr=0.005", make_config(learning_rate=0.005)))

    # Combos
    sweeps.append(("steps=100+anneal=16->2", make_config(
        num_steps=100, heatmap_blur_sigma_end=2.0,
    )))
    sweeps.append(("steps=100+anneal=16->2+lr=0.002", make_config(
        num_steps=100, heatmap_blur_sigma_end=2.0, learning_rate=0.002,
    )))
    sweeps.append(("steps=75+anneal=16->2+bone=0.01", make_config(
        num_steps=75, heatmap_blur_sigma_end=2.0, bone_length_lr=0.01,
    )))

    return sweeps


def main():
    print("=" * 70)
    print("MotionBERT Hyperparameter Sweep (single example)")
    print("=" * 70)

    print("\nLoading example data (one-time cost)...")
    data = load_motionbert_example()
    print("  Done.\n")

    sweeps = define_sweeps()
    results: list[dict] = []

    for i, (name, cfg) in enumerate(sweeps):
        print(f"[{i+1}/{len(sweeps)}] {name}")
        res = run_optimization(data, cfg)
        res["name"] = name
        results.append(res)
        print(
            f"  Raw: {res['raw_vw_si_mpjpe_cm']:.2f} cm  "
            f"Opt: {res['opt_vw_si_mpjpe_cm']:.2f} cm  "
            f"Improvement: {res['improvement_pct']:.1f}%  "
            f"Time: {res['elapsed_s']:.1f}s"
        )

    results.sort(key=lambda r: r["improvement_pct"], reverse=True)

    print("\n" + "=" * 70)
    print("RESULTS (sorted by improvement)")
    print("=" * 70)
    print(f"{'Config':<50} {'Raw':>7} {'Opt':>7} {'Impr%':>7} {'Time':>6}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['name']:<50} "
            f"{r['raw_vw_si_mpjpe_cm']:>6.2f}  "
            f"{r['opt_vw_si_mpjpe_cm']:>6.2f}  "
            f"{r['improvement_pct']:>6.1f}%  "
            f"{r['elapsed_s']:>5.1f}s"
        )

    out_path = os.path.join(os.path.dirname(__file__), "motionbert_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
