"""
Headless evaluation script: generate synthetic video data and run optimization.
No vpython or interactive display needed.
"""

import json
import logging
import os
import sys
import tempfile

import numpy as np
import torch

# Configure logging to capture step-level data
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from model.arm import Arm
from model.camera import Camera
from utils import (
    EvaluationResult,
    OptimizationConfig,
    OptimizationResult,
    prepare_heatmaps,
    score,
    score_pose_against_heatmap,
)
from video import Video

# ──────────────────────────────────────────────
# Simulation constants (mirrors motion_simulation.py)
# ──────────────────────────────────────────────
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
FOCAL_LENGTH = (50.0, 50.0)
PRINCIPAL_POINT = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
A_B_LENGTH = 2.0
B_C_LENGTH = 2.0
NUM_FRAMES = 20

CAMERA_POSITION = np.array([8.0, 0.0, 0.0])
CAMERA_ROTATION = np.array([
    [0, 1, 0],
    [0, 0, -1],
    [-1, 0, 0],
])

# Velocity limits (from motion_simulation.py)
POS_VEL_STEP = 0.04
POS_VEL_MIN = -0.20; POS_VEL_MAX = 0.20
ANG_VEL_STEP = 0.04
ANG_VEL_MIN = -0.20; ANG_VEL_MAX = 0.20


def wrap(value, min_val, max_val):
    range_size = max_val - min_val
    return ((value - min_val) % range_size) + min_val


def generate_video(output_dir: str, num_frames: int = NUM_FRAMES, seed: int = 42) -> Video:
    """Generate synthetic video with random-walk arm motion."""
    np.random.seed(seed)

    video = Video(
        camera_position=CAMERA_POSITION,
        camera_rotation=CAMERA_ROTATION,
        focal_length=FOCAL_LENGTH,
        principal_point=PRINCIPAL_POINT,
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        a_b_length=A_B_LENGTH,
        b_c_length=B_C_LENGTH,
        output_dir=output_dir,
    )

    # Initial arm state
    a_pos = np.array([0.0, 0.0, 0.0])
    a_b_azimuth = np.random.uniform(-np.pi, np.pi)
    a_b_elev = np.random.uniform(-np.pi / 2, np.pi / 2)
    a_b_roll = np.random.uniform(-np.pi, np.pi)
    b_c_theta = np.random.uniform(-np.pi / 2, np.pi / 2)

    pos_vel = np.array([0.0, 0.0, 0.0])
    a_b_azimuth_vel = 0.0
    a_b_elev_vel = 0.0
    a_b_roll_vel = 0.0
    b_c_theta_vel = 0.0

    for _ in range(num_frames):
        # Update velocities with random walk
        for i in range(3):
            delta = np.random.choice([-1, 0, 1]) * POS_VEL_STEP
            pos_vel[i] = np.clip(pos_vel[i] + delta, POS_VEL_MIN, POS_VEL_MAX)

        a_b_azimuth_vel += np.random.choice([-1, 0, 1]) * ANG_VEL_STEP
        a_b_azimuth_vel = np.clip(a_b_azimuth_vel, ANG_VEL_MIN, ANG_VEL_MAX)
        a_b_elev_vel += np.random.choice([-1, 0, 1]) * ANG_VEL_STEP
        a_b_elev_vel = np.clip(a_b_elev_vel, ANG_VEL_MIN, ANG_VEL_MAX)
        a_b_roll_vel += np.random.choice([-1, 0, 1]) * ANG_VEL_STEP
        a_b_roll_vel = np.clip(a_b_roll_vel, ANG_VEL_MIN, ANG_VEL_MAX)
        b_c_theta_vel += np.random.choice([-1, 0, 1]) * ANG_VEL_STEP
        b_c_theta_vel = np.clip(b_c_theta_vel, ANG_VEL_MIN, ANG_VEL_MAX)

        a_pos = a_pos + pos_vel
        a_pos = np.clip(a_pos, -1.0, 1.0)
        a_b_azimuth = wrap(a_b_azimuth + a_b_azimuth_vel, -np.pi, np.pi)
        a_b_roll = wrap(a_b_roll + a_b_roll_vel, -np.pi, np.pi)

        a_b_elev += a_b_elev_vel
        if a_b_elev > np.pi / 2:
            a_b_elev = np.pi - a_b_elev
            a_b_azimuth = wrap(a_b_azimuth + np.pi, -np.pi, np.pi)
            a_b_elev_vel = -a_b_elev_vel
        elif a_b_elev < -np.pi / 2:
            a_b_elev = -np.pi - a_b_elev
            a_b_azimuth = wrap(a_b_azimuth + np.pi, -np.pi, np.pi)
            a_b_elev_vel = -a_b_elev_vel

        b_c_theta += b_c_theta_vel
        if b_c_theta > np.pi / 2:
            b_c_theta = np.pi - b_c_theta
            b_c_theta_vel = -b_c_theta_vel
        elif b_c_theta < -np.pi / 2:
            b_c_theta = -np.pi - b_c_theta
            b_c_theta_vel = -b_c_theta_vel

        arm = Arm(
            a_pos=a_pos.copy(),
            a_b_length=A_B_LENGTH,
            a_b_polar=(a_b_azimuth, a_b_elev, a_b_roll),
            b_c_length=B_C_LENGTH,
            b_c_theta=b_c_theta,
        )
        coords = arm.get_coordinates_numpy()
        video.update(coords, a_pos.copy(), (a_b_azimuth, a_b_elev, a_b_roll), b_c_theta)

    return video


def run_optimization_with_convergence(video: Video, config: OptimizationConfig):
    """Run optimization and capture per-step convergence data."""
    num_frames = video.get_frame_count()
    a_b_length = video.a_b_length
    b_c_length = video.b_c_length
    camera = video.camera

    all_a_pos = []
    all_a_b_polar = []
    all_b_c_theta = []
    all_gt_coords = []
    all_init_coords = []

    for frame_idx in range(num_frames):
        gt_arm = video.get_frame_arm(frame_idx)
        gt_coords = gt_arm.get_coordinates()
        all_gt_coords.append(gt_coords)

        torch.manual_seed(frame_idx)
        a_pos = (
            gt_arm.a_pos.detach().clone().float()
            + torch.randn(3) * config.position_init_noise
        )
        a_b_polar = (
            gt_arm.a_b_polar.detach().clone().float()
            + torch.randn(3) * config.angle_init_noise
        )
        b_c_theta = (
            gt_arm.b_c_theta.detach().clone().float()
            + torch.randn(1).item() * config.angle_init_noise
        )

        init_arm = Arm(a_pos, a_b_length, a_b_polar, b_c_length, b_c_theta)
        all_init_coords.append(init_arm.get_coordinates_numpy())

        a_pos.requires_grad_(True)
        a_b_polar.requires_grad_(True)
        b_c_theta = torch.tensor(b_c_theta, requires_grad=True)

        all_a_pos.append(a_pos)
        all_a_b_polar.append(a_b_polar)
        all_b_c_theta.append(b_c_theta)

    all_params = all_a_pos + all_a_b_polar + all_b_c_theta
    optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)

    all_log_heatmaps = [
        prepare_heatmaps(video.get_frame_heatmaps(i)) for i in range(num_frames)
    ]

    convergence = []  # (step, score, heatmap, position, ab_rotation, bc_rotation, weighted_motion)

    for step in range(config.num_steps):
        optimizer.zero_grad()

        arms = [
            Arm(all_a_pos[i], a_b_length, all_a_b_polar[i], b_c_length, all_b_c_theta[i])
            for i in range(num_frames)
        ]

        total_score, components = score(
            all_log_heatmaps, arms, camera, config, return_components=True
        )

        convergence.append({
            "step": step + 1,
            "score": total_score.item(),
            "heatmap": components["heatmap"],
            "position_penalty": components["position"],
            "ab_rotation_penalty": components["ab_rotation"],
            "bc_rotation_penalty": components["bc_rotation"],
            "weighted_motion": components["weighted_motion"],
        })

        loss = -total_score
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0:
            logger.info(
                "Step %d: score=%.4f (heatmap=%.4f, pos=%.4f, ab=%.4f, bc=%.4f, wt_motion=%.4f)",
                step + 1,
                total_score.item(),
                components["heatmap"],
                components["position"],
                components["ab_rotation"],
                components["bc_rotation"],
                components["weighted_motion"],
            )

    pred_arms = []
    pred_coords = []
    gt_arms = []

    with torch.no_grad():
        for i in range(num_frames):
            pred_arm = Arm(
                all_a_pos[i], a_b_length, all_a_b_polar[i], b_c_length, all_b_c_theta[i]
            )
            pred_arms.append(pred_arm)
            pred_coords.append(pred_arm.get_coordinates_numpy())
            gt_arms.append(video.get_frame_arm(i))

    opt_result = OptimizationResult(
        pred_arms=pred_arms,
        pred_coords=pred_coords,
        init_coords=all_init_coords,
        gt_arms=gt_arms,
        gt_coords=all_gt_coords,
    )

    return opt_result, convergence


def compute_mpjpe(coords_list_a, coords_list_b):
    """Compute mean per-joint position error between two lists of coordinate dicts."""
    per_frame = []
    for ca, cb in zip(coords_list_a, coords_list_b):
        frame_err = 0.0
        for joint in ["a", "b", "c"]:
            va = ca[joint] if isinstance(ca[joint], np.ndarray) else ca[joint].detach().numpy()
            vb = cb[joint] if isinstance(cb[joint], np.ndarray) else cb[joint].detach().numpy()
            frame_err += np.linalg.norm(va - vb)
        per_frame.append(frame_err / 3)
    return np.mean(per_frame), per_frame


def compute_velocity_error(coords_list_a, coords_list_b):
    """
    Compute mean per-joint velocity error.
    Velocity = frame-to-frame displacement; error = difference in velocities.
    """
    if len(coords_list_a) < 2:
        return None, []
    per_frame = []
    for i in range(len(coords_list_a) - 1):
        frame_err = 0.0
        for joint in ["a", "b", "c"]:
            # ground-truth velocity
            def to_np(x):
                return x if isinstance(x, np.ndarray) else x.detach().numpy()

            vel_a = to_np(coords_list_a[i+1][joint]) - to_np(coords_list_a[i][joint])
            vel_b = to_np(coords_list_b[i+1][joint]) - to_np(coords_list_b[i][joint])
            frame_err += np.linalg.norm(vel_a - vel_b)
        per_frame.append(frame_err / 3)
    return np.mean(per_frame), per_frame


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        print("=" * 60)
        print("TOY ARM POSE OPTIMIZATION - HEADLESS EVALUATION")
        print("=" * 60)

        # ── 1. Generate synthetic video ──────────────────────────────
        print(f"\n[1] Generating {NUM_FRAMES}-frame synthetic video (seed=42)...")
        video = generate_video(tmpdir, num_frames=NUM_FRAMES, seed=42)
        print(f"    Frames: {video.get_frame_count()}")
        print(f"    Segment lengths: A-B={video.a_b_length}, B-C={video.b_c_length}")
        print(f"    Heatmap noise std: {1} px (localization), Gaussian std={3.0} px")

        # ── 2. Run optimization ──────────────────────────────────────
        config = OptimizationConfig()
        print(f"\n[2] Running optimization ({config.num_steps} steps, lr={config.learning_rate})...")
        print(f"    Init noise: pos±{config.position_init_noise}, angle±{config.angle_init_noise} rad")
        print(f"    Penalty weights: pos={config.position_penalty_weight}, "
              f"ab={config.ab_rotation_penalty_weight}, bc={config.bc_rotation_penalty_weight}")

        opt_result, convergence = run_optimization_with_convergence(video, config)

        # ── 3. Compute MPJPE ─────────────────────────────────────────
        # init vs GT
        init_mpjpe, init_mpjpe_per_frame = compute_mpjpe(
            opt_result.init_coords,
            [{k: v.detach().numpy() if isinstance(v, torch.Tensor) else v
              for k, v in gc.items()} for gc in opt_result.gt_coords],
        )
        # pred vs GT
        pred_mpjpe, pred_mpjpe_per_frame = compute_mpjpe(
            opt_result.pred_coords,
            [{k: v.detach().numpy() if isinstance(v, torch.Tensor) else v
              for k, v in gc.items()} for gc in opt_result.gt_coords],
        )

        # ── 4. Compute velocity error ────────────────────────────────
        gt_coords_np = [
            {k: v.detach().numpy() if isinstance(v, torch.Tensor) else v
             for k, v in gc.items()}
            for gc in opt_result.gt_coords
        ]
        init_vel_err, init_vel_per = compute_velocity_error(opt_result.init_coords, gt_coords_np)
        pred_vel_err, pred_vel_per = compute_velocity_error(opt_result.pred_coords, gt_coords_np)

        # ── 5. Convergence summary ───────────────────────────────────
        step_1 = convergence[0]
        step_last = convergence[-1]
        step_20 = convergence[19]
        step_40 = convergence[39]
        step_60 = convergence[59]
        step_80 = convergence[79]

        # ── 6. Print results ─────────────────────────────────────────
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        print("\n--- POSITION ERROR (MPJPE, world-space units) ---")
        print(f"  Before optimization (init):  {init_mpjpe:.4f}")
        print(f"  After  optimization (pred):  {pred_mpjpe:.4f}")
        improvement_pos = (init_mpjpe - pred_mpjpe) / init_mpjpe * 100
        print(f"  Improvement:                 {improvement_pos:.1f}%")

        print("\n--- VELOCITY ERROR (mean |Δvel| per joint, world-space units) ---")
        print(f"  Before optimization (init):  {init_vel_err:.4f}")
        print(f"  After  optimization (pred):  {pred_vel_err:.4f}")
        improvement_vel = (init_vel_err - pred_vel_err) / init_vel_err * 100
        print(f"  Improvement:                 {improvement_vel:.1f}%")

        print("\n--- CONVERGENCE CURVE (score = heatmap_score - motion_penalty) ---")
        print(f"  {'Step':>5}  {'Total Score':>12}  {'Heatmap':>10}  {'Wt.Motion':>10}")
        for row in [step_1, step_20, step_40, step_60, step_80, step_last]:
            print(f"  {row['step']:>5}  {row['score']:>12.4f}  {row['heatmap']:>10.4f}  {row['weighted_motion']:>10.4f}")

        print("\n--- FULL CONVERGENCE DATA (every 10 steps) ---")
        print(f"  {'Step':>5}  {'Score':>10}  {'Heatmap':>10}  {'Pos Pen':>10}  {'AB Pen':>10}  {'BC Pen':>10}")
        for row in convergence[::10]:
            print(f"  {row['step']:>5}  {row['score']:>10.4f}  {row['heatmap']:>10.4f}  "
                  f"  {row['position_penalty']:>8.4f}  {row['ab_rotation_penalty']:>8.4f}  "
                  f"  {row['bc_rotation_penalty']:>8.4f}")

        print("\n--- PER-FRAME MPJPE ---")
        print(f"  {'Frame':>6}  {'Init MPJPE':>12}  {'Pred MPJPE':>12}")
        for i, (a, b) in enumerate(zip(init_mpjpe_per_frame, pred_mpjpe_per_frame)):
            print(f"  {i:>6}  {a:>12.4f}  {b:>12.4f}")

        print("\n--- WHAT THE TOY ARM VALIDATES ---")
        print("  Model:       3-joint planar arm (A, B, C) with fixed bone lengths")
        print("  Parameters:  A position (xyz), AB polar angles (azimuth/elev/roll), BC bend (theta)")
        print("  Heatmaps:    Synthetic 2D Gaussian blobs (sigma=3px) + localization noise (sigma=1px)")
        print("  Projection:  Pinhole camera (fx=fy=50, 200x200 image)")
        print("  Objective:   Sum of log-heatmap scores - weighted motion penalties")
        print("  Optimizer:   Adam, 100 steps, lr=0.1")
        print("  Bone lengths: FIXED (A-B=2.0, B-C=2.0; not optimized)")

        # ── 7. Save convergence to JSON for plotting ─────────────────
        def to_python(obj):
            """Recursively convert numpy scalars to Python native types for JSON."""
            if isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_python(v) for v in obj]
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj

        out_path = os.path.join(
            os.path.dirname(__file__), "convergence_data.json"
        )
        with open(out_path, "w") as f:
            json.dump(to_python({
                "convergence": convergence,
                "init_mpjpe": init_mpjpe,
                "pred_mpjpe": pred_mpjpe,
                "init_vel_err": init_vel_err,
                "pred_vel_err": pred_vel_err,
                "init_mpjpe_per_frame": init_mpjpe_per_frame,
                "pred_mpjpe_per_frame": pred_mpjpe_per_frame,
                "init_vel_per_frame": init_vel_per,
                "pred_vel_per_frame": pred_vel_per,
            }), f, indent=2)
        print(f"\n  Convergence data saved to: {out_path}")

        print("\n" + "=" * 60)
        print("DONE")
        print("=" * 60)


if __name__ == "__main__":
    main()
