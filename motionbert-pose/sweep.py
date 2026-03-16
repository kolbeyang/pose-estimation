"""Parameter sweep for FK optimization hyperparameters.

Runs detection once, then re-optimizes with different parameter configs.
Reports MPJPE, P-MPJPE, MPJVE, and 2D-vs-detection reprojection error.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config as cfg
from camera import Camera
from detect import detect_poses, motionbert_to_camera_space
from evaluate import compute_comparison_with_optimization
from optimize import run_optimization
from panoptic import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
    world_to_camera,
)


@dataclass
class SweepConfig:
    """One parameter configuration to test."""

    name: str
    num_steps: int = 20
    position_penalty_weight: float = 50.0
    rotation_penalty_scalar: float = 10.0
    init_anchor_weight: float = 5.0
    all_joints_smooth_weight: float = 0.0
    sigma_schedule: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 80.0)]
    )
    heatmap_blur_sigma: float = 0.0  # Additional Gaussian blur on SH heatmaps


# Per-joint rotation multipliers (same as config.py)
_ROT_MULTIPLIERS: list[float] = [
    3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2,
    1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1,
]


def load_example(example_idx: int = 0) -> dict[str, Any]:
    """Load and detect poses for one example. Returns cached data dict."""
    seq_name, camera_name, start_frame, num_frames, person_idx = cfg.EXAMPLES[example_idx]
    name: str = f"{seq_name}_{start_frame}"

    seq_dir: str = get_sequence_dir(cfg.PANOPTIC_ROOT, seq_name)
    video_path: str = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)

    # 1. Camera calibration
    cameras = load_calibration(seq_dir)
    cam_calib = None
    for cname in cameras:
        if cname == camera_name or cname == f"00_{camera_name.split('_')[1]}":
            cam_calib = cameras[cname]
            break
    if cam_calib is None:
        for cname, cal in cameras.items():
            if cname.startswith("00_00"):
                cam_calib = cal
                break
    assert cam_calib is not None, f"Camera {camera_name} not found"

    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    resolution = cam_calib["resolution"]
    camera = Camera.from_panoptic_calibration(K, resolution)

    # 2. Extract frames
    video_fps: float = 30.0
    frame_step = max(1, int(round(video_fps / cfg.TARGET_FPS)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    frames_rgb = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[: len(frames_rgb)]

    # 3. Detect
    kp_2d, kp_3d, visibility, heatmaps, mpii_kp_2d, affine, positions_3d_norm, cs_params = (
        detect_poses(frames_rgb)
    )

    # 4. Camera-space conversion
    scale = cs_params["scale"]
    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        dist_coeffs = cam_calib.get("distCoef")
        pos_cam = motionbert_to_camera_space(
            positions_3d_norm[i],
            kp_2d[i],
            scale,
            fx,
            fy,
            cx,
            cy,
            dist_coeffs=dist_coeffs,
            visibility=visibility[i],
        )
        det_cam_positions.append(pos_cam)

    # 5. Build improved 2D targets
    improved_target_2d: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        target = kp_2d[i].copy()
        mb_projected = camera.world_to_image(det_cam_positions[i])
        for j in range(17):
            if visibility[i][j] < cfg.FK_TARGET_CONF_THRESHOLD:
                target[j] = mb_projected[j]
        improved_target_2d.append(target)

    # 6. Ground truth
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam: list[np.ndarray | None] = []
    for gt in gt_world:
        if gt is not None:
            gt_cam_cm = world_to_camera(gt, R, t)
            gt_cam.append(gt_cam_cm * 0.01)
        else:
            gt_cam.append(None)

    return {
        "camera": camera,
        "det_cam_positions": det_cam_positions,
        "kp_2d": kp_2d,
        "visibility": visibility,
        "heatmaps": heatmaps,
        "affine": affine,
        "improved_target_2d": improved_target_2d,
        "gt_cam": gt_cam,
        "frames_rgb": frames_rgb,
        "frame_indices": frame_indices,
        "name": name,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def run_sweep_config(data: dict[str, Any], config: SweepConfig) -> dict[str, Any]:
    """Run optimization with one parameter config, return metrics."""
    # Save originals
    orig_pos_weight = cfg.POSITION_PENALTY_WEIGHT
    orig_rot_scalar = cfg.ROTATION_PENALTY_SCALAR
    orig_rot_per_joint = cfg.ROTATION_PENALTY_PER_JOINT.copy()
    orig_anchor = cfg.INIT_ANCHOR_WEIGHT
    orig_smooth = cfg.ALL_JOINTS_SMOOTH_WEIGHT
    orig_sigma_schedule = cfg.SIGMA_SCHEDULE

    try:
        cfg.POSITION_PENALTY_WEIGHT = config.position_penalty_weight
        cfg.ROTATION_PENALTY_SCALAR = config.rotation_penalty_scalar
        cfg.INIT_ANCHOR_WEIGHT = config.init_anchor_weight
        cfg.ALL_JOINTS_SMOOTH_WEIGHT = config.all_joints_smooth_weight
        cfg.SIGMA_SCHEDULE = config.sigma_schedule

        # Rebuild per-joint rotation weights with new scalar
        cfg.ROTATION_PENALTY_PER_JOINT = np.array(
            [config.rotation_penalty_scalar * m for m in _ROT_MULTIPLIERS],
            dtype=np.float64,
        )

        # Optionally blur heatmaps
        heatmaps = data["heatmaps"]
        if config.heatmap_blur_sigma > 0:
            import scipy.ndimage

            heatmaps = [
                np.stack(
                    [
                        scipy.ndimage.gaussian_filter(hm[c], sigma=config.heatmap_blur_sigma)
                        for c in range(hm.shape[0])
                    ]
                )
                for hm in heatmaps
            ]

        optimized_3d, bone_lengths_final, loss_history = run_optimization(
            initial_positions_cam=data["det_cam_positions"],
            target_2d=data["improved_target_2d"],
            visibility=data["visibility"],
            camera=data["camera"],
            num_steps=config.num_steps,
            heatmaps=heatmaps,
            affine=data["affine"],
        )

        metrics = compute_comparison_with_optimization(
            detector_3d=data["det_cam_positions"],
            optimized_3d=optimized_3d,
            gt_3d=data["gt_cam"],
            camera=data["camera"],
            detections_2d=data["improved_target_2d"],
            visibility=data["visibility"],
        )
        metrics["loss_history"] = loss_history
        return metrics

    finally:
        # Restore original config
        cfg.POSITION_PENALTY_WEIGHT = orig_pos_weight
        cfg.ROTATION_PENALTY_SCALAR = orig_rot_scalar
        cfg.ROTATION_PENALTY_PER_JOINT = orig_rot_per_joint
        cfg.INIT_ANCHOR_WEIGHT = orig_anchor
        cfg.ALL_JOINTS_SMOOTH_WEIGHT = orig_smooth
        cfg.SIGMA_SCHEDULE = orig_sigma_schedule


def get_phase1_1_configs() -> list[SweepConfig]:
    """Return Phase 1.1 sweep configurations."""
    configs: list[SweepConfig] = []

    # --- Baseline ---
    configs.append(
        SweepConfig(
            name="baseline",
            position_penalty_weight=50.0,
            rotation_penalty_scalar=10.0,
            init_anchor_weight=5.0,
        )
    )

    # --- Position penalty weight sweep ---
    configs.append(SweepConfig(name="pos_w=0", position_penalty_weight=0.0, rotation_penalty_scalar=10.0, init_anchor_weight=5.0))
    configs.append(SweepConfig(name="pos_w=0.5", position_penalty_weight=0.5, rotation_penalty_scalar=10.0, init_anchor_weight=5.0))
    configs.append(SweepConfig(name="pos_w=5", position_penalty_weight=5.0, rotation_penalty_scalar=10.0, init_anchor_weight=5.0))
    configs.append(SweepConfig(name="pos_w=500", position_penalty_weight=500.0, rotation_penalty_scalar=10.0, init_anchor_weight=5.0))
    configs.append(SweepConfig(name="pos_w=5000", position_penalty_weight=5000.0, rotation_penalty_scalar=10.0, init_anchor_weight=5.0))

    # --- Rotation penalty scalar sweep ---
    configs.append(SweepConfig(name="rot_s=0", position_penalty_weight=50.0, rotation_penalty_scalar=0.0, init_anchor_weight=5.0))
    configs.append(SweepConfig(name="rot_s=0.1", position_penalty_weight=50.0, rotation_penalty_scalar=0.1, init_anchor_weight=5.0))
    configs.append(SweepConfig(name="rot_s=1", position_penalty_weight=50.0, rotation_penalty_scalar=1.0, init_anchor_weight=5.0))
    configs.append(SweepConfig(name="rot_s=100", position_penalty_weight=50.0, rotation_penalty_scalar=100.0, init_anchor_weight=5.0))
    configs.append(SweepConfig(name="rot_s=1000", position_penalty_weight=50.0, rotation_penalty_scalar=1000.0, init_anchor_weight=5.0))

    # --- Init anchor weight sweep ---
    configs.append(SweepConfig(name="anchor=0", position_penalty_weight=50.0, rotation_penalty_scalar=10.0, init_anchor_weight=0.0))
    configs.append(SweepConfig(name="anchor=0.05", position_penalty_weight=50.0, rotation_penalty_scalar=10.0, init_anchor_weight=0.05))
    configs.append(SweepConfig(name="anchor=0.5", position_penalty_weight=50.0, rotation_penalty_scalar=10.0, init_anchor_weight=0.5))
    configs.append(SweepConfig(name="anchor=50", position_penalty_weight=50.0, rotation_penalty_scalar=10.0, init_anchor_weight=50.0))
    configs.append(SweepConfig(name="anchor=500", position_penalty_weight=50.0, rotation_penalty_scalar=10.0, init_anchor_weight=500.0))

    # --- Heatmap only (all penalties off) ---
    configs.append(SweepConfig(name="heatmap_only", position_penalty_weight=0.0, rotation_penalty_scalar=0.0, init_anchor_weight=0.0))

    # --- All penalties very low ---
    configs.append(SweepConfig(name="all_low", position_penalty_weight=0.5, rotation_penalty_scalar=0.1, init_anchor_weight=0.05))

    # --- More steps ---
    configs.append(
        SweepConfig(
            name="baseline_100steps",
            num_steps=100,
            position_penalty_weight=50.0,
            rotation_penalty_scalar=10.0,
            init_anchor_weight=5.0,
        )
    )
    configs.append(
        SweepConfig(
            name="all_low_100steps",
            num_steps=100,
            position_penalty_weight=0.5,
            rotation_penalty_scalar=0.1,
            init_anchor_weight=0.05,
        )
    )

    return configs


def main() -> None:
    """Run the parameter sweep."""
    example_idx: int = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("Loading example data (detection + GT)...")
    data = load_example(example_idx)
    print(f"Example: {data['name']}, {len(data['det_cam_positions'])} frames\n")

    # Define sweep configurations -- Phase 1.1
    configs: list[SweepConfig] = get_phase1_1_configs()

    results: list[tuple[str, dict[str, Any]]] = []
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(
            f"  [{i+1}/{len(configs)}] {config.name}"
        )
        print(
            f"  pos_w={config.position_penalty_weight}, rot_s={config.rotation_penalty_scalar}, "
            f"anchor={config.init_anchor_weight}, smooth={config.all_joints_smooth_weight}"
        )
        print(f"{'='*60}")

        t0 = time.time()
        metrics = run_sweep_config(data, config)
        elapsed = time.time() - t0

        results.append((config.name, metrics))
        print(f"  Time: {elapsed:.1f}s")

    # Print summary table
    print(f"\n\n{'='*100}")
    print(f"  SWEEP RESULTS -- {data['name']}")
    print(f"{'='*100}")
    header = (
        f"{'Config':<40} {'Det MPJPE':>10} {'Opt MPJPE':>10} {'Improv':>8} "
        f"{'Opt P-MPJPE':>12} {'Det 2D-Det':>10} {'Opt 2D-Det':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results:
        det_mpjpe = m.get("det_mpjpe", 0) * 100
        opt_mpjpe = m.get("opt_mpjpe", 0) * 100
        improv = m.get("improvement", 0) * 100
        opt_p = m.get("opt_p_mpjpe", 0) * 100
        det_2d = m.get("det_2d_det_mpjpe_px", 0)
        opt_2d = m.get("opt_2d_det_mpjpe_px", 0)
        print(
            f"{name:<40} {det_mpjpe:>10.2f} {opt_mpjpe:>10.2f} {improv:>+8.2f} "
            f"{opt_p:>12.2f} {det_2d:>10.1f} {opt_2d:>10.1f}"
        )

    # Save results to JSON
    out_dir = os.path.join(cfg.TRAINING_RUNS_DIR, "sweep_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sweep_{data['name']}.json")
    save_data = []
    for name, m in results:
        save_data.append(
            {
                "config": name,
                "det_mpjpe_cm": m.get("det_mpjpe", 0) * 100,
                "opt_mpjpe_cm": m.get("opt_mpjpe", 0) * 100,
                "improvement_cm": m.get("improvement", 0) * 100,
                "opt_p_mpjpe_cm": m.get("opt_p_mpjpe", 0) * 100,
                "det_2d_det_mpjpe_px": m.get("det_2d_det_mpjpe_px", 0),
                "opt_2d_det_mpjpe_px": m.get("opt_2d_det_mpjpe_px", 0),
                "opt_mpjve_cm": m.get("opt_mpjve", 0) * 100
                if "opt_mpjve" in m
                else None,
            }
        )
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
