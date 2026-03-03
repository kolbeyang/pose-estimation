"""Full-body FK optimization loop.

Uses 3-phase coarse-to-fine Gaussian sigma schedule.
Optimises FK parameters (root position, root rotation, local rotations,
and shared bone lengths) against 2D MediaPipe detections.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from fk import forward_kinematics, positions_to_fk_params
from scoring import compute_total_score
from skeleton import NUM_JOINTS, PARENTS
from model.camera import Camera
import config as cfg


@dataclass
class OptimizationResult:
    """Stores initial (MediaPipe) and optimised 3D predictions."""
    mediapipe_3d: list[np.ndarray]       # Per-frame (17, 3) initial camera-space
    optimized_3d: list[np.ndarray]       # Per-frame (17, 3) optimised camera-space
    bone_lengths_final: np.ndarray       # (17,) final bone lengths
    loss_history: list[float] = field(default_factory=list)
    score_details_history: list[dict] = field(default_factory=list)


def _get_sigma(step: int, num_steps: int) -> float:
    """Get Gaussian sigma for the current step based on blur schedule."""
    frac = step / max(num_steps - 1, 1)
    for phase_frac, sigma in cfg.BLUR_PHASES:
        if frac <= phase_frac:
            return sigma
    return cfg.BLUR_PHASES[-1][1]


def run_optimization(
    initial_positions_cam: list[np.ndarray],
    target_2d: list[np.ndarray],
    visibility: list[np.ndarray],
    camera: Camera,
    num_steps: int = None,
) -> OptimizationResult:
    """Run full-body FK optimisation.

    Args:
        initial_positions_cam: Per-frame (17, 3) camera-space positions from MediaPipe.
        target_2d: Per-frame (17, 2) pixel target positions.
        visibility: Per-frame (17,) visibility weights.
        camera: Camera for 3D→2D projection.
        num_steps: Override for cfg.NUM_STEPS.

    Returns:
        OptimizationResult with initial and optimised 3D predictions.
    """
    if num_steps is None:
        num_steps = cfg.NUM_STEPS

    n_frames = len(initial_positions_cam)

    # Save initial positions for comparison
    mediapipe_3d = [pos.copy() for pos in initial_positions_cam]

    # Convert initial positions to FK parameters
    all_root_pos = []
    all_root_rot = []
    all_local_rots = []
    all_bone_lengths = []

    for positions in initial_positions_cam:
        root_pos, root_rot, local_rots, bone_lengths = positions_to_fk_params(positions)
        all_root_pos.append(root_pos)
        all_root_rot.append(root_rot)
        all_local_rots.append(local_rots)
        all_bone_lengths.append(bone_lengths)

    # Create learnable parameters
    # Per-frame: root position, root rotation, local rotations
    param_root_pos = [
        torch.tensor(rp, dtype=torch.float32, requires_grad=True)
        for rp in all_root_pos
    ]
    param_root_rot = [
        torch.tensor(rr, dtype=torch.float32, requires_grad=True)
        for rr in all_root_rot
    ]
    param_local_rots = [
        torch.tensor(lr, dtype=torch.float32, requires_grad=True)
        for lr in all_local_rots
    ]

    # Shared bone lengths: median across frames
    median_bone_lengths = np.median(np.array(all_bone_lengths), axis=0)
    initial_bone_lengths = torch.tensor(
        median_bone_lengths, dtype=torch.float32,
    )
    param_bone_lengths = torch.tensor(
        median_bone_lengths, dtype=torch.float32, requires_grad=True,
    )

    # Target tensors (not learnable)
    target_2d_t = [
        torch.tensor(t, dtype=torch.float32) for t in target_2d
    ]
    visibility_t = [
        torch.tensor(v, dtype=torch.float32) for v in visibility
    ]

    # Optimiser
    all_params = (
        param_root_pos + param_root_rot + param_local_rots + [param_bone_lengths]
    )
    optimizer = torch.optim.Adam([
        {"params": param_root_pos, "lr": cfg.LEARNING_RATE},
        {"params": param_root_rot, "lr": cfg.LEARNING_RATE},
        {"params": param_local_rots, "lr": cfg.LEARNING_RATE},
        {"params": [param_bone_lengths], "lr": cfg.BONE_LENGTH_LR},
    ])

    loss_history = []
    score_details_history = []

    print(f"  Optimising {n_frames} frames for {num_steps} steps...")

    for step in range(num_steps):
        optimizer.zero_grad()

        sigma = _get_sigma(step, num_steps)

        # Forward pass: FK → 3D positions → project to 2D
        all_positions = []
        all_projected_2d = []
        all_local_rots_current = []

        for i in range(n_frames):
            positions_3d = forward_kinematics(
                param_root_pos[i],
                param_root_rot[i],
                param_local_rots[i],
                param_bone_lengths,
            )
            projected_2d = camera.world_to_image_torch(positions_3d)

            all_positions.append(positions_3d)
            all_projected_2d.append(projected_2d)
            all_local_rots_current.append(param_local_rots[i])

        # Compute score
        total_score, details = compute_total_score(
            all_positions,
            all_projected_2d,
            all_local_rots_current,
            target_2d_t,
            visibility_t,
            sigma,
            cfg.POSITION_PENALTY_WEIGHT,
            cfg.ROTATION_PENALTY_WEIGHT,
        )

        # Bone length regularisation: keep close to initial estimate
        bl_reg = ((param_bone_lengths - initial_bone_lengths) ** 2).sum()
        total_score = total_score - cfg.BONE_LENGTH_REG_WEIGHT * bl_reg
        details["bl_reg"] = float(bl_reg.item())

        loss = -total_score
        loss.backward()

        torch.nn.utils.clip_grad_norm_(all_params, cfg.GRAD_CLIP_NORM)
        optimizer.step()

        # Clamp bone lengths to positive
        with torch.no_grad():
            param_bone_lengths.clamp_(min=0.01)

        loss_history.append(float(loss.item()))
        score_details_history.append(details)

        if step % 50 == 0 or step == num_steps - 1:
            print(
                f"    Step {step:4d}/{num_steps}  "
                f"loss={loss.item():.1f}  "
                f"heatmap={details['heatmap']:.1f}  "
                f"sigma={sigma:.0f}  "
                f"pos_p={details['pos_penalty']:.4f}  "
                f"rot_p={details['rot_penalty']:.4f}"
            )

    # Extract final optimised 3D positions
    optimized_3d = []
    with torch.no_grad():
        for i in range(n_frames):
            positions = forward_kinematics(
                param_root_pos[i],
                param_root_rot[i],
                param_local_rots[i],
                param_bone_lengths,
            )
            optimized_3d.append(positions.numpy().copy())

    return OptimizationResult(
        mediapipe_3d=mediapipe_3d,
        optimized_3d=optimized_3d,
        bone_lengths_final=param_bone_lengths.detach().numpy().copy(),
        loss_history=loss_history,
        score_details_history=score_details_history,
    )
