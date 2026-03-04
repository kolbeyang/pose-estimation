"""Full-body FK optimization loop.

Uses 3-phase coarse-to-fine blur schedule on real Stacked Hourglass heatmaps.
Optimises FK parameters (root position, root rotation, local rotations,
and shared bone lengths) against real 2D heatmaps.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from fk import forward_kinematics, positions_to_fk_params
from scoring import compute_total_score, prepare_heatmaps
from model.camera import Camera
import config as cfg


@dataclass
class OptimizationResult:
    """Stores initial (detector) and optimised 3D predictions."""
    mediapipe_3d: list[np.ndarray]       # Per-frame (17, 3) initial camera-space
    optimized_3d: list[np.ndarray]       # Per-frame (17, 3) optimised camera-space
    bone_lengths_final: np.ndarray       # (17,) final bone lengths
    loss_history: list[float] = field(default_factory=list)
    score_details_history: list[dict] = field(default_factory=list)


def _get_blur_sigma(step: int, num_steps: int) -> float:
    """Get heatmap blur sigma for the current step based on blur schedule."""
    frac = step / max(num_steps - 1, 1)
    for phase_frac, sigma in cfg.BLUR_PHASES:
        if frac <= phase_frac:
            return sigma
    return cfg.BLUR_PHASES[-1][1]


def _compute_affine_pixel_to_hm(affine_256_to_pixel: np.ndarray) -> torch.Tensor:
    """Compute affine mapping: pixel coords → grid_sample [-1,1] coords.

    The coordinate chain is:
        pixel → 256-crop → 64-heatmap → grid_sample [-1,1]

    affine_256_to_pixel maps 256-crop → pixel:
        pixel_x = scale_x * crop_x + offset_x
        pixel_y = scale_y * crop_y + offset_y

    We need the inverse (pixel → crop_256):
        crop_x = (pixel_x - offset_x) / scale_x
        crop_y = (pixel_y - offset_y) / scale_y

    Then crop_256 → heatmap_64: divide by 4
    Then heatmap_64 → grid_sample [-1,1]: grid = (hm / 64) * 2 - 1 (align_corners=False)
                                          grid = hm / 32 - 1

    Combined (align_corners=False):
        grid_x = 2 * (hm_x + 0.5) / 64 - 1
        where hm_x = (pixel_x - offset_x) / (scale_x * 4)
        So: grid_x = (pixel_x - offset_x) / (scale_x * 128) + 1/64 - 1
    """
    scale_x = affine_256_to_pixel[0, 0]
    scale_y = affine_256_to_pixel[1, 1]
    offset_x = affine_256_to_pixel[0, 2]
    offset_y = affine_256_to_pixel[1, 2]

    # pixel → grid_sample [-1,1] (align_corners=False)
    # grid = (pixel - offset) / (scale * 128) + 1/64 - 1
    half_pixel = 1.0 / 64.0  # 0.015625
    a00 = 1.0 / (scale_x * 128.0)
    a02 = -offset_x / (scale_x * 128.0) + half_pixel - 1.0
    a11 = 1.0 / (scale_y * 128.0)
    a12 = -offset_y / (scale_y * 128.0) + half_pixel - 1.0

    affine = torch.tensor([
        [a00, 0.0, a02],
        [0.0, a11, a12],
    ], dtype=torch.float32)

    return affine


def run_optimization(
    initial_positions_cam: list[np.ndarray],
    heatmaps_raw: list[np.ndarray],
    affine_256_to_pixel: np.ndarray,
    visibility: list[np.ndarray],
    camera: Camera,
    num_steps: int | None = None,
) -> OptimizationResult:
    """Run full-body FK optimisation using real heatmaps.

    Args:
        initial_positions_cam: Per-frame (17, 3) camera-space positions.
        heatmaps_raw: Per-frame (16, 64, 64) raw Stacked Hourglass heatmaps.
        affine_256_to_pixel: (2, 3) affine from 256-crop coords to pixel coords.
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

    # Precompute affine: pixel → grid_sample coords
    affine_pixel_to_hm = _compute_affine_pixel_to_hm(affine_256_to_pixel)

    # Precompute blurred log-heatmaps for each phase
    unique_sigmas = sorted(set(sigma for _, sigma in cfg.BLUR_PHASES), reverse=True)
    print(f"  Precomputing {len(unique_sigmas)} blur levels: {unique_sigmas}")
    log_heatmaps_by_sigma = {}
    for sigma in unique_sigmas:
        log_heatmaps_by_sigma[sigma] = prepare_heatmaps(heatmaps_raw, blur_sigma=sigma)

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

    # Shared bone lengths: initialise and regularise from detector median
    median_bone_lengths = np.median(np.array(all_bone_lengths), axis=0)
    initial_bone_lengths = torch.tensor(
        median_bone_lengths, dtype=torch.float32,
    )
    param_bone_lengths = torch.tensor(
        median_bone_lengths, dtype=torch.float32, requires_grad=True,
    )

    # Initial positions as tensors (for regularisation, not learnable)
    init_positions_t = [
        torch.tensor(pos, dtype=torch.float32) for pos in initial_positions_cam
    ]

    # Visibility tensors (not learnable)
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

        sigma = _get_blur_sigma(step, num_steps)
        log_hm = log_heatmaps_by_sigma[sigma]

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
            log_hm,
            visibility_t,
            affine_pixel_to_hm,
            cfg.POSITION_PENALTY_WEIGHT,
            cfg.ROTATION_PENALTY_WEIGHT,
            init_positions=init_positions_t,
            init_position_reg_weight=cfg.INIT_POSITION_REG_WEIGHT,
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
                f"sigma={sigma:.1f}  "
                f"pos_p={details['pos_penalty']:.4f}  "
                f"rot_p={details['rot_penalty']:.4f}  "
                f"init_reg={details['init_reg']:.4f}"
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
