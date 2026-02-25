"""Optimization loop for arm pose estimation from heatmaps.

Uses 3-phase coarse-to-fine: heavily blurred heatmaps → medium blur → sharp.
Each phase narrows the basin of attraction for better convergence.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from model.arm import Arm
from model.camera import Camera
from scoring import OptimizationConfig, prepare_heatmaps, score


# Coarse-to-fine blur schedule: (fraction_of_steps, blur_sigma)
# Phase 1: root position only with heavy blur (wide basin)
# Phase 2: all params with medium blur
# Phase 3: all params with no blur (precise matching)
BLUR_PHASES = [
    (0.33, 15.0),   # first 33%: sigma=15
    (0.66, 5.0),    # next 33%: sigma=5
    (1.00, 0.0),    # final 33%: no blur
]


@dataclass
class OptimizationResult:
    mediapipe_coords: list[dict[str, np.ndarray]]
    optimized_coords: list[dict[str, np.ndarray]]
    optimized_arms: list[Arm]
    bone_length_history: dict[str, list[float]] = field(default_factory=lambda: {"a_b": [], "b_c": []})
    mediapipe_bone_lengths: dict[str, float] = field(default_factory=dict)
    loss_history: list[float] = field(default_factory=list)


def run_optimization(
    initial_arms: list[Arm],
    heatmaps: list[dict[str, np.ndarray]],
    cameras: list[Camera],
    mp_a_b_length: float,
    mp_b_c_length: float,
    config: OptimizationConfig,
) -> OptimizationResult:
    """
    Optimize arm poses against heatmaps using Adam with coarse-to-fine.

    3 phases with decreasing heatmap blur (15 → 5 → 0 pixels sigma).
    Phase 1 optimizes only shoulder position; phases 2-3 optimize all params.
    """
    n_frames = len(initial_arms)

    # Save initial coords before optimization
    mp_coords = [arm.get_coordinates_numpy() for arm in initial_arms]

    # Shared learnable bone lengths
    a_b_length = torch.tensor(mp_a_b_length, dtype=torch.float32, requires_grad=True)
    b_c_length = torch.tensor(mp_b_c_length, dtype=torch.float32, requires_grad=True)

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

    # Precompute blurred heatmap versions for each phase
    phase_heatmaps = {}
    for _, sigma in BLUR_PHASES:
        if sigma not in phase_heatmaps:
            phase_heatmaps[sigma] = [prepare_heatmaps(h, blur_sigma=sigma) for h in heatmaps]
            print(f"  Prepared heatmaps with blur sigma={sigma:.0f}")

    bone_length_history = {"a_b": [], "b_c": []}
    loss_history = []

    # Set up optimizer — phase 1 freezes angles by using lr=0
    # We'll rebuild the optimizer at each phase transition
    current_phase = -1

    print(f"Optimizing {n_frames} frames for {config.num_steps} steps (3-phase coarse-to-fine)...")

    for step in range(config.num_steps):
        # Determine current phase
        frac = step / config.num_steps
        new_phase = 0
        current_sigma = BLUR_PHASES[0][1]
        for phase_idx, (end_frac, sigma) in enumerate(BLUR_PHASES):
            if frac < end_frac:
                new_phase = phase_idx
                current_sigma = sigma
                break

        # Rebuild optimizer on phase change
        if new_phase != current_phase:
            current_phase = new_phase
            if current_phase == 0:
                # Phase 1: only optimize position (freeze angles)
                params = [
                    {"params": all_a_pos, "lr": config.learning_rate * 5},
                    {"params": all_a_b_polar, "lr": 0.0},
                    {"params": all_b_c_theta, "lr": 0.0},
                    {"params": [a_b_length, b_c_length], "lr": 0.0},
                ]
            else:
                # Phases 2-3: optimize everything
                params = [
                    {"params": all_a_pos, "lr": config.learning_rate * 5},
                    {"params": all_a_b_polar, "lr": config.learning_rate},
                    {"params": all_b_c_theta, "lr": config.learning_rate},
                    {"params": [a_b_length, b_c_length], "lr": config.bone_length_lr},
                ]
            optimizer = torch.optim.Adam(params)
            print(f"  Phase {current_phase + 1} (step {step}): sigma={current_sigma:.0f}, "
                  f"params={'position only' if current_phase == 0 else 'all'}")

        all_log_heatmaps = phase_heatmaps[current_sigma]

        optimizer.zero_grad()

        # Record bone lengths
        bone_length_history["a_b"].append(a_b_length.item())
        bone_length_history["b_c"].append(b_c_length.item())

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
                f"motion={components['weighted_motion']:.2f} "
                f"a_b={a_b_length.item():.4f} b_c={b_c_length.item():.4f}"
            )
        else:
            total_score = score(all_log_heatmaps, arms, cameras, config)

        loss = -total_score
        loss_history.append(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            all_a_pos + all_a_b_polar + all_b_c_theta + [a_b_length, b_c_length],
            max_norm=10.0,
        )

        optimizer.step()

    # Build final optimized arms
    optimized_arms = []
    optimized_coords = []
    for i in range(n_frames):
        arm = Arm(
            a_pos=all_a_pos[i].detach(),
            a_b_length=a_b_length.detach(),
            a_b_polar=all_a_b_polar[i].detach(),
            b_c_length=b_c_length.detach(),
            b_c_theta=all_b_c_theta[i].detach(),
        )
        optimized_arms.append(arm)
        optimized_coords.append(arm.get_coordinates_numpy())

    return OptimizationResult(
        mediapipe_coords=mp_coords,
        optimized_coords=optimized_coords,
        optimized_arms=optimized_arms,
        bone_length_history=bone_length_history,
        mediapipe_bone_lengths={"a_b": mp_a_b_length, "b_c": mp_b_c_length},
        loss_history=loss_history,
    )
