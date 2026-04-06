"""MediaPipe hyperparameter sweep on single example.

Sweeps optimization parameters one-at-a-time (and small combos) to find
a config that pushes MediaPipe improvement past 2%.

Usage:
    cd pose-optimizer
    uv run python experiment/mediapipe_sweep.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Single-example data loader (shared across all sweep runs)
# ---------------------------------------------------------------------------

@dataclass
class ExampleData:
    """Pre-loaded data for one example."""
    frames_rgb: list
    heatmaps: list
    kp_2d: list
    visibility: list
    affine: np.ndarray
    mp_kp_2d: list
    mp_kp_3d: list
    mp_visibility: list
    det_cam_positions: list[np.ndarray]
    gt_cam: list[np.ndarray | None]
    camera: Camera


def load_example(
    seq: str = "171204_pose1_sample",
    camera_name: str = "00_00",
    start_frame: int = 0,
    num_frames: int = 100,
    person_idx: int = 0,
    target_fps: float = 10.0,
    data_root: str = "data/panoptic-toolbox",
) -> ExampleData:
    """Load and run shared detection for one example."""
    from run_motionbert.detect import load_yolo_sh_models, detect_2d_poses
    from run_mediapipe.detect import (
        load_landmarker,
        detect_poses as mp_detect_poses,
        mediapipe_3d_to_camera,
    )

    seq_dir = get_sequence_dir(data_root, seq)
    video_path = get_video_path(data_root, seq, camera_name)

    # Camera calibration
    cameras = load_calibration(seq_dir)
    cam_calib = cameras[camera_name]
    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    resolution = cam_calib["resolution"]
    camera = Camera.from_panoptic_calibration(K, R, t, resolution)

    # Extract frames
    video_fps = 30.0
    frame_step = max(1, int(round(video_fps / target_fps)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    frames_rgb = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[: len(frames_rgb)]

    # Shared YOLO + SH
    print("  Loading YOLO + SH models...")
    yolo_sh_models = load_yolo_sh_models()
    print("  Running YOLO + SH detection...")
    kp_2d, visibility, heatmaps, mpii_kp_2d, affine = detect_2d_poses(
        frames_rgb, yolo_sh_models, sh_batch_size=32
    )

    # MediaPipe detection
    print("  Loading MediaPipe model...")
    mp_landmarker = load_landmarker()
    print("  Running MediaPipe detection...")
    mp_kp_2d, mp_kp_3d, mp_visibility = mp_detect_poses(frames_rgb, landmarker=mp_landmarker)
    mp_landmarker.close()

    # Convert to camera space
    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        pos_cam = mediapipe_3d_to_camera(mp_kp_3d[i], mp_kp_2d[i], fx, fy, cx, cy)
        det_cam_positions.append(pos_cam)

    # Ground truth
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam: list[np.ndarray | None] = []
    for gt in gt_world:
        if gt is not None:
            gt_cam.append(camera.world_to_camera(gt) * 0.01)
        else:
            gt_cam.append(None)

    return ExampleData(
        frames_rgb=frames_rgb,
        heatmaps=heatmaps,
        kp_2d=kp_2d,
        visibility=visibility,
        affine=affine,
        mp_kp_2d=mp_kp_2d,
        mp_kp_3d=mp_kp_3d,
        mp_visibility=mp_visibility,
        det_cam_positions=det_cam_positions,
        gt_cam=gt_cam,
        camera=camera,
    )


def run_optimization(data: ExampleData, config: OptimizationConfig) -> dict:
    """Run optimization and return metrics dict."""
    t0 = time.time()
    opt_3d, bl_final, loss_history = optimize(
        raw_3d=data.det_cam_positions,
        camera=data.camera,
        config=config,
        heatmaps=data.heatmaps,
        affine=data.affine,
        visibility=data.visibility,
        verbose=False,
    )
    elapsed = time.time() - t0

    # Evaluate
    gt_indices = [i for i, g in enumerate(data.gt_cam) if g is not None]
    gt_arr = np.array([data.gt_cam[i] for i in gt_indices])
    det_arr = np.array([data.det_cam_positions[i] for i in gt_indices])
    opt_arr = np.array([opt_3d[i] for i in gt_indices])

    det_metrics = evaluate(det_arr, gt_arr, data.camera)
    opt_metrics = evaluate(opt_arr, gt_arr, data.camera)

    raw_vw = det_metrics["vw_si_mpjpe"] * 100
    opt_vw = opt_metrics["vw_si_mpjpe"] * 100
    improvement_pct = (raw_vw - opt_vw) / raw_vw * 100

    return {
        "raw_vw_si_mpjpe_cm": raw_vw,
        "opt_vw_si_mpjpe_cm": opt_vw,
        "improvement_pct": improvement_pct,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

# Baseline config (current best from Phase 3)
BASELINE = {
    "num_steps": 25,
    "learning_rate": 0.0005,
    "heatmap_blur_sigma": 16.0,
    "anchor_weight": 1000.0,
    "bone_length_lr": 0.001,
    "position_penalty_weight": 500.0,
    "rotation_penalty_scalar": 10.0,
    "confidence_epsilon": 1e-4,
}


def make_config(**overrides) -> OptimizationConfig:
    """Create config from baseline + overrides."""
    params = {**BASELINE, **overrides}
    return OptimizationConfig(**params)


def define_sweeps() -> list[tuple[str, OptimizationConfig]]:
    """Define all sweep configurations."""
    sweeps: list[tuple[str, OptimizationConfig]] = []

    # 0. Baseline
    sweeps.append(("baseline", make_config()))

    # 1. num_steps sweep
    for s in [50, 75, 100]:
        sweeps.append((f"steps={s}", make_config(num_steps=s)))

    # 2. learning_rate sweep
    for lr in [0.0001, 0.0003, 0.001]:
        sweeps.append((f"lr={lr}", make_config(learning_rate=lr)))

    # 3. anchor_weight sweep
    for aw in [500, 2000, 5000]:
        sweeps.append((f"anchor={aw}", make_config(anchor_weight=aw)))

    # 4. Fixed sigma sweep
    for sig in [4, 8, 12]:
        sweeps.append((f"sigma={sig}", make_config(heatmap_blur_sigma=sig)))

    # 5. Blur annealing (clear fixed sigma, set start/end)
    for s_start, s_end in [(16, 2), (16, 4), (12, 2), (8, 2)]:
        sweeps.append((
            f"anneal={s_start}->{s_end}",
            make_config(
                heatmap_blur_sigma=0.0,  # disable fixed
                heatmap_blur_sigma_start=float(s_start),
                heatmap_blur_sigma_end=float(s_end),
            ),
        ))

    # 6. bone_length_lr sweep
    for blr in [0.0005, 0.005]:
        sweeps.append((f"bone_lr={blr}", make_config(bone_length_lr=blr)))

    # 7. Promising combos: more steps + annealing
    for steps in [50, 75]:
        sweeps.append((
            f"steps={steps}+anneal=16->2",
            make_config(
                num_steps=steps,
                heatmap_blur_sigma=0.0,
                heatmap_blur_sigma_start=16.0,
                heatmap_blur_sigma_end=2.0,
            ),
        ))

    # 8. Combo: more steps + annealing + lower anchor
    sweeps.append((
        "steps=50+anneal=16->4+anchor=500",
        make_config(
            num_steps=50,
            heatmap_blur_sigma=0.0,
            heatmap_blur_sigma_start=16.0,
            heatmap_blur_sigma_end=4.0,
            anchor_weight=500.0,
        ),
    ))

    # 9. Combo: more steps + anneal + higher lr
    sweeps.append((
        "steps=50+anneal=16->2+lr=0.001",
        make_config(
            num_steps=50,
            learning_rate=0.001,
            heatmap_blur_sigma=0.0,
            heatmap_blur_sigma_start=16.0,
            heatmap_blur_sigma_end=2.0,
        ),
    ))

    # 10. Combo: higher anchor + more steps
    sweeps.append((
        "steps=75+anchor=2000",
        make_config(num_steps=75, anchor_weight=2000.0),
    ))

    return sweeps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MediaPipe Hyperparameter Sweep (single example)")
    print("=" * 70)

    print("\nLoading example data (one-time cost)...")
    data = load_example()
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

    # Sort by improvement
    results.sort(key=lambda r: r["improvement_pct"], reverse=True)

    print("\n" + "=" * 70)
    print("RESULTS (sorted by improvement)")
    print("=" * 70)
    print(f"{'Config':<42} {'Raw':>7} {'Opt':>7} {'Impr%':>7} {'Time':>6}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<42} "
            f"{r['raw_vw_si_mpjpe_cm']:>6.2f}  "
            f"{r['opt_vw_si_mpjpe_cm']:>6.2f}  "
            f"{r['improvement_pct']:>6.1f}%  "
            f"{r['elapsed_s']:>5.1f}s"
        )

    # Save to JSON
    out_path = os.path.join(os.path.dirname(__file__), "mediapipe_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    best = results[0]
    print(f"\nBest config: {best['name']}")
    print(f"  Raw: {best['raw_vw_si_mpjpe_cm']:.2f} cm -> Opt: {best['opt_vw_si_mpjpe_cm']:.2f} cm ({best['improvement_pct']:.1f}% improvement)")


if __name__ == "__main__":
    main()
