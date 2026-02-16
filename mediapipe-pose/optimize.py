"""Optimization loop for arm pose estimation from heatmaps."""

from dataclasses import dataclass

import numpy as np
import torch

from model.arm import Arm
from model.camera import Camera
from scoring import OptimizationConfig, prepare_heatmaps, score


@dataclass
class OptimizationResult:
    mediapipe_arms: list[Arm]
    mediapipe_coords: list[dict[str, np.ndarray]]
    optimized_arms: list[Arm]
    optimized_coords: list[dict[str, np.ndarray]]


def run_optimization(
    initial_arms: list[Arm],
    heatmaps: list[dict[str, np.ndarray]],
    cameras: list[Camera],
    a_b_length: float,
    b_c_length: float,
    config: OptimizationConfig,
) -> OptimizationResult:
    """
    Optimize arm poses against heatmaps using Adam.

    Args:
        initial_arms: Initial arm poses from MediaPipe IK conversion
        heatmaps: Per-frame heatmaps for each joint (uint8)
        cameras: Per-frame fitted cameras
        a_b_length, b_c_length: Fixed segment lengths
        config: Optimization configuration

    Returns:
        OptimizationResult with MediaPipe and optimized arm data
    """
    n_frames = len(initial_arms)

    # Save MediaPipe coords before optimization
    mp_coords = [arm.get_coordinates_numpy() for arm in initial_arms]

    # Create optimizable parameters for each frame
    all_a_pos = []
    all_a_b_polar = []
    all_b_c_theta = []

    for arm in initial_arms:
        a_pos = arm.a_pos.clone().detach().requires_grad_(True)
        a_b_polar = arm.a_b_polar.clone().detach().requires_grad_(True)
        b_c_theta = arm.b_c_theta.clone().detach().requires_grad_(True)
        all_a_pos.append(a_pos)
        all_a_b_polar.append(a_b_polar)
        all_b_c_theta.append(b_c_theta)

    # Preload and log-normalize heatmaps
    all_log_heatmaps = [prepare_heatmaps(h) for h in heatmaps]

    # Set up optimizer
    params = (
        [{"params": all_a_pos, "lr": config.learning_rate * 5}]
        + [{"params": all_a_b_polar, "lr": config.learning_rate}]
        + [{"params": all_b_c_theta, "lr": config.learning_rate}]
    )
    optimizer = torch.optim.Adam(params)

    print(f"Optimizing {n_frames} frames for {config.num_steps} steps...")

    for step in range(config.num_steps):
        optimizer.zero_grad()

        # Build arms from current parameters
        arms = []
        for i in range(n_frames):
            arm = Arm(
                a_pos=all_a_pos[i],
                a_b_length=a_b_length,
                a_b_polar=all_a_b_polar[i],
                b_c_length=b_c_length,
                b_c_theta=all_b_c_theta[i],
            )
            arms.append(arm)

        # Compute score
        if step % 20 == 0:
            total_score, components = score(
                all_log_heatmaps, arms, cameras, config, return_components=True
            )
            print(
                f"  Step {step:3d}: score={total_score.item():.2f} "
                f"heatmap={components['heatmap']:.2f} "
                f"motion={components['weighted_motion']:.2f}"
            )
        else:
            total_score = score(all_log_heatmaps, arms, cameras, config)

        loss = -total_score
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            all_a_pos + all_a_b_polar + all_b_c_theta, max_norm=10.0
        )

        optimizer.step()

    # Build final optimized arms
    optimized_arms = []
    optimized_coords = []
    for i in range(n_frames):
        arm = Arm(
            a_pos=all_a_pos[i].detach(),
            a_b_length=a_b_length,
            a_b_polar=all_a_b_polar[i].detach(),
            b_c_length=b_c_length,
            b_c_theta=all_b_c_theta[i].detach(),
        )
        optimized_arms.append(arm)
        optimized_coords.append(arm.get_coordinates_numpy())

    # Rebuild mediapipe arms (non-grad) for comparison
    mp_arms = []
    for arm in initial_arms:
        mp_arm = Arm(
            a_pos=arm.a_pos.detach(),
            a_b_length=a_b_length,
            a_b_polar=arm.a_b_polar.detach(),
            b_c_length=b_c_length,
            b_c_theta=arm.b_c_theta.detach(),
        )
        mp_arms.append(mp_arm)

    return OptimizationResult(
        mediapipe_arms=mp_arms,
        mediapipe_coords=mp_coords,
        optimized_arms=optimized_arms,
        optimized_coords=optimized_coords,
    )
