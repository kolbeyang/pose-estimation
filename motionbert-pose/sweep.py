"""Parameter sweep for FK optimization hyperparameters.

Runs detection once, then re-optimizes with different parameter configs.
Reports MPJPE, P-MPJPE, MPJVE, and 2D-vs-detection reprojection error.
"""

import json
import os
import time
from dataclasses import dataclass
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
    heatmap_blur_sigma: float = 0.0  # Fixed blur (applied inside run_optimization via cfg override)
    heatmap_blur_schedule: list[tuple[float, float]] | None = None  # Coarse-to-fine blur


# Per-joint rotation multipliers (same as config.py)
_ROT_MULTIPLIERS: list[float] = [
    3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2,  # Hip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle
    1.0, 1.0, 0.5,                         # Spine, Thorax, Neck
    0.5, 0.3, 0.1, 0.5, 0.3, 0.1,         # LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist
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
    kp_2d, visibility, heatmaps, mpii_kp_2d, affine, positions_3d_norm = detect_poses(frames_rgb)

    # 4. Camera-space conversion (per-frame pairwise depth estimation)
    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        pos_cam = motionbert_to_camera_space(
            positions_3d_norm[i], kp_2d[i], fx, fy, cx, cy,
        )
        det_cam_positions.append(pos_cam)

    # 5. Ground truth
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
        "target_2d": kp_2d,
        "visibility": visibility,
        "heatmaps": heatmaps,
        "affine": affine,
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
    orig_blur_sigma = cfg.HEATMAP_BLUR_SIGMA

    try:
        cfg.POSITION_PENALTY_WEIGHT = config.position_penalty_weight
        cfg.ROTATION_PENALTY_SCALAR = config.rotation_penalty_scalar
        cfg.INIT_ANCHOR_WEIGHT = config.init_anchor_weight
        cfg.HEATMAP_BLUR_SIGMA = config.heatmap_blur_sigma

        # Rebuild per-joint rotation weights with new scalar
        cfg.ROTATION_PENALTY_PER_JOINT = np.array(
            [config.rotation_penalty_scalar * m for m in _ROT_MULTIPLIERS],
            dtype=np.float64,
        )

        optimized_3d, bone_lengths_final, loss_history = run_optimization(
            initial_positions_cam=data["det_cam_positions"],
            target_2d=data["target_2d"],
            visibility=data["visibility"],
            camera=data["camera"],
            num_steps=config.num_steps,
            heatmaps=data["heatmaps"],
            affine=data["affine"],
            heatmap_blur_schedule=config.heatmap_blur_schedule,
        )

        metrics = compute_comparison_with_optimization(
            detector_3d=data["det_cam_positions"],
            optimized_3d=optimized_3d,
            gt_3d=data["gt_cam"],
            camera=data["camera"],
            detections_2d=data["target_2d"],
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
        cfg.HEATMAP_BLUR_SIGMA = orig_blur_sigma


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


def get_phase1_2_configs() -> list[SweepConfig]:
    """Return Phase 1.2 heatmap blur sweep configurations."""
    configs: list[SweepConfig] = []

    # Common penalty settings (baseline -- Phase 1.1 showed these don't matter much)
    pw = 50.0
    rs = 10.0
    aw = 5.0

    # --- Baseline (no blur) at multiple step counts ---
    for steps in [20, 50, 100]:
        configs.append(SweepConfig(
            name=f"no_blur_{steps}s",
            num_steps=steps,
            position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
        ))

    # --- Fixed blur sweep at 50 steps ---
    for sigma in [1.0, 2.0, 4.0, 8.0]:
        configs.append(SweepConfig(
            name=f"blur{sigma:.0f}_{50}s",
            num_steps=50,
            heatmap_blur_sigma=sigma,
            position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
        ))

    # --- Fixed blur sweep at 100 steps ---
    for sigma in [1.0, 2.0, 4.0, 8.0]:
        configs.append(SweepConfig(
            name=f"blur{sigma:.0f}_{100}s",
            num_steps=100,
            heatmap_blur_sigma=sigma,
            position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
        ))

    # --- Coarse-to-fine blur schedules at 100 steps ---
    configs.append(SweepConfig(
        name="c2f_8to0_100s",
        num_steps=100,
        heatmap_blur_schedule=[(0.3, 8.0), (0.7, 4.0), (1.0, 0.0)],
        position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
    ))
    configs.append(SweepConfig(
        name="c2f_4to0_100s",
        num_steps=100,
        heatmap_blur_schedule=[(0.3, 4.0), (0.7, 2.0), (1.0, 0.0)],
        position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
    ))
    configs.append(SweepConfig(
        name="c2f_4to1_100s",
        num_steps=100,
        heatmap_blur_schedule=[(0.3, 4.0), (0.7, 2.0), (1.0, 1.0)],
        position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
    ))
    configs.append(SweepConfig(
        name="c2f_8to2_100s",
        num_steps=100,
        heatmap_blur_schedule=[(0.3, 8.0), (0.7, 4.0), (1.0, 2.0)],
        position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
    ))

    # --- Coarse-to-fine blur at 50 steps ---
    configs.append(SweepConfig(
        name="c2f_4to0_50s",
        num_steps=50,
        heatmap_blur_schedule=[(0.4, 4.0), (0.8, 2.0), (1.0, 0.0)],
        position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
    ))
    configs.append(SweepConfig(
        name="c2f_8to0_50s",
        num_steps=50,
        heatmap_blur_schedule=[(0.4, 8.0), (0.8, 4.0), (1.0, 0.0)],
        position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw,
    ))

    return configs


def get_round6_configs() -> list[SweepConfig]:
    """Round 6: coarse sweep of rotation penalty scalar."""
    configs: list[SweepConfig] = []
    for rot_s in [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
        configs.append(SweepConfig(
            name=f"rot_s={int(rot_s)}",
            num_steps=50,
            position_penalty_weight=50.0,
            rotation_penalty_scalar=rot_s,
            init_anchor_weight=5.0,
            heatmap_blur_sigma=4.0,
        ))
    return configs


def get_round6_fine_configs() -> list[SweepConfig]:
    """Round 6 fine: sweep around best rotation scalar + position weight combos.

    The developer should update BEST_ROT_S based on coarse sweep results.
    """
    BEST_ROT_S: float = 100.0  # UPDATE after coarse sweep

    configs: list[SweepConfig] = []

    # Fine rotation sweep: 0.5x, 0.7x, 1.0x, 1.5x, 2.0x, 3.0x of best
    for mult in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        rot_s = BEST_ROT_S * mult
        configs.append(SweepConfig(
            name=f"rot_s={rot_s:.0f}_pos=50",
            num_steps=50,
            position_penalty_weight=50.0,
            rotation_penalty_scalar=rot_s,
            init_anchor_weight=5.0,
            heatmap_blur_sigma=4.0,
        ))

    # Position weight sweep at best rotation scalar
    for pos_w in [10.0, 25.0, 50.0, 100.0, 200.0, 500.0]:
        configs.append(SweepConfig(
            name=f"rot_s={BEST_ROT_S:.0f}_pos={int(pos_w)}",
            num_steps=50,
            position_penalty_weight=pos_w,
            rotation_penalty_scalar=BEST_ROT_S,
            init_anchor_weight=5.0,
            heatmap_blur_sigma=4.0,
        ))

    # Combo: best rotation * {0.5x, 2.0x} with best-ish position weights
    for rot_mult in [0.5, 2.0]:
        for pos_w in [25.0, 100.0, 200.0]:
            rot_s = BEST_ROT_S * rot_mult
            configs.append(SweepConfig(
                name=f"rot_s={rot_s:.0f}_pos={int(pos_w)}",
                num_steps=50,
                position_penalty_weight=pos_w,
                rotation_penalty_scalar=rot_s,
                init_anchor_weight=5.0,
                heatmap_blur_sigma=4.0,
            ))

    return configs


def main() -> None:
    """Run the parameter sweep."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=str, default="0",
                        help="Comma-separated example indices (e.g., '0,5')")
    parser.add_argument("--phase", choices=["1.1", "1.2", "round6", "round6-fine"],
                        default="round6",
                        help="Which config set to run")
    args = parser.parse_args()

    example_indices: list[int] = [int(x.strip()) for x in args.examples.split(",")]

    if args.phase == "1.1":
        configs: list[SweepConfig] = get_phase1_1_configs()
    elif args.phase == "1.2":
        configs = get_phase1_2_configs()
    elif args.phase == "round6":
        configs = get_round6_configs()
    else:
        configs = get_round6_fine_configs()

    for example_idx in example_indices:
        print(f"\n{'#'*70}")
        print(f"  Loading example {example_idx} (detection + GT)...")
        print(f"{'#'*70}")
        data = load_example(example_idx)
        print(f"Example: {data['name']}, {len(data['det_cam_positions'])} frames\n")

        results: list[tuple[str, dict[str, Any], SweepConfig]] = []
        for i, config in enumerate(configs):
            print(f"\n{'='*60}")
            print(
                f"  [{i+1}/{len(configs)}] {config.name}"
            )
            print(
                f"  pos_w={config.position_penalty_weight}, rot_s={config.rotation_penalty_scalar}, "
                f"anchor={config.init_anchor_weight}, "
                f"blur={config.heatmap_blur_sigma}, blur_sched={config.heatmap_blur_schedule}"
            )
            print(f"{'='*60}")

            t0 = time.time()
            metrics = run_sweep_config(data, config)
            elapsed = time.time() - t0

            results.append((config.name, metrics, config))
            print(f"  Time: {elapsed:.1f}s")

        # Print summary table
        print(f"\n\n{'='*125}")
        print(f"  SWEEP RESULTS -- {data['name']}")
        print(f"{'='*125}")
        header = (
            f"{'Config':<40} {'Det MPJPE':>10} {'Opt MPJPE':>10} {'Improv':>8} "
            f"{'Opt P-MPJPE':>12} {'Det MPJVE':>10} {'Opt MPJVE':>10} {'Det 2D-Det':>10} {'Opt 2D-Det':>10}"
        )
        print(header)
        print("-" * len(header))
        for name, m, config in results:
            det_mpjpe = m.get("det_mpjpe", 0) * 100
            opt_mpjpe = m.get("opt_mpjpe", 0) * 100
            improv = m.get("improvement", 0) * 100
            opt_p = m.get("opt_p_mpjpe", 0) * 100
            det_mpjve = m.get("det_mpjve", 0) * 100 if m.get("det_mpjve") is not None else 0.0
            opt_mpjve = m.get("opt_mpjve", 0) * 100 if m.get("opt_mpjve") is not None else 0.0
            det_2d = m.get("det_2d_det_mpjpe_px", 0)
            opt_2d = m.get("opt_2d_det_mpjpe_px", 0)
            print(
                f"{name:<40} {det_mpjpe:>10.2f} {opt_mpjpe:>10.2f} {improv:>+8.2f} "
                f"{opt_p:>12.2f} {det_mpjve:>10.2f} {opt_mpjve:>10.2f} {det_2d:>10.1f} {opt_2d:>10.1f}"
            )

        # Save results to JSON
        out_dir = os.path.join(cfg.TRAINING_RUNS_DIR, "sweep_results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"sweep_{data['name']}.json")
        save_data = []
        for name, m, config in results:
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
                    "det_mpjve_cm": m.get("det_mpjve", 0) * 100 if "det_mpjve" in m else None,
                    "num_steps": config.num_steps,
                    "heatmap_blur_sigma": config.heatmap_blur_sigma,
                    "heatmap_blur_schedule": config.heatmap_blur_schedule,
                }
            )
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
