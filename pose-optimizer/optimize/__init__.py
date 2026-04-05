"""Pure-function optimizer: optimize(raw_3d, camera, ...) -> improved_3d.

Batched FK optimizer using real Stacked Hourglass heatmaps for scoring.
"""

from __future__ import annotations

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

from camera import Camera
from config import OptimizationConfig
from fk import forward_kinematics, forward_kinematics_batch, positions_to_fk_params
from scoring import (
    compute_total_score_batch,
    apply_blur,
)
from skeleton import NUM_JOINTS


def optimize(
    raw_3d: list[np.ndarray],
    camera: Camera,
    config: OptimizationConfig,
    heatmaps: list[np.ndarray],
    affine: np.ndarray,
    visibility: list[np.ndarray] | None = None,
    verbose: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    """Run FK optimization to improve 3D pose estimates.

    Args:
        raw_3d: Per-frame (16, 3) camera-space positions. [3D:SKELETON_16]
        camera: Camera for 3D->2D projection.
        config: Optimization hyperparameters.
        heatmaps: Per-frame (16, 64, 64) SH heatmaps. [HEATMAP:MPII_16]
        affine: (2, 3) affine from crop to pixel coords.
        visibility: Per-frame (16,) confidence scores. [VIS:SKELETON_16]
            Defaults to ones.
        verbose: Print progress.

    Returns:
        Tuple of:
            optimized_3d: List of (16, 3) improved camera-space positions. [3D:SKELETON_16]
            bone_lengths_final: (16,) final shared bone lengths.
            loss_history: List of loss values per step.
    """
    n_frames = len(raw_3d)

    # Default visibility to ones
    if visibility is None:
        visibility = [np.ones(NUM_JOINTS) for _ in range(n_frames)]

    heatmaps_np = np.array(heatmaps)  # [HEATMAP:MPII_16]
    affine_np = affine

    # --- Initialize FK parameters ---
    if verbose:
        logger.info("Initializing FK parameters for %d frames...", n_frames)

    all_root_pos: list[np.ndarray] = []
    all_root_rot: list[np.ndarray] = []
    all_local_rots: list[np.ndarray] = []
    all_bone_lengths: list[np.ndarray] = []
    roundtrip_errors: list[float] = []

    for positions in raw_3d:
        root_pos, root_rot, local_rots, bone_lengths = positions_to_fk_params(positions)

        with torch.no_grad():
            reconstructed = forward_kinematics(
                torch.tensor(root_pos, dtype=torch.float32),
                torch.tensor(root_rot, dtype=torch.float32),
                torch.tensor(local_rots, dtype=torch.float32),
                torch.tensor(bone_lengths, dtype=torch.float32),
            )
            rt_error = float(torch.mean(torch.norm(
                reconstructed - torch.tensor(positions, dtype=torch.float32),
                dim=-1,
            )).item())
            roundtrip_errors.append(rt_error)

        all_root_pos.append(root_pos)
        all_root_rot.append(root_rot)
        all_local_rots.append(local_rots)
        all_bone_lengths.append(bone_lengths)

    if verbose:
        mean_rt = float(np.mean(roundtrip_errors))
        max_rt = float(np.max(roundtrip_errors))
        logger.info("FK roundtrip error: mean=%.4f cm, max=%.4f cm", mean_rt * 100, max_rt * 100)

    # --- Create batched learnable parameters ---
    param_root_pos = torch.tensor(  # [FK_PARAMS] (F, 3)
        np.array(all_root_pos), dtype=torch.float32, requires_grad=True,
    )
    param_root_rot = torch.tensor(  # [FK_PARAMS] (F, 3)
        np.array(all_root_rot), dtype=torch.float32, requires_grad=True,
    )
    param_local_rots = torch.tensor(  # [FK_PARAMS] (F, 16, 3)
        np.array(all_local_rots), dtype=torch.float32, requires_grad=True,
    )

    # Shared bone lengths: median across frames
    median_bone_lengths = np.median(np.array(all_bone_lengths), axis=0)
    param_bone_lengths = torch.tensor(
        median_bone_lengths, dtype=torch.float32, requires_grad=True,
    )

    # --- Non-learnable tensors ---
    visibility_t = torch.tensor(np.array(visibility), dtype=torch.float32)  # (F, 16) [VIS:SKELETON_16]
    heatmaps_t = torch.tensor(heatmaps_np, dtype=torch.float32)  # (F, C, H, W) [HEATMAP:MPII_16] or [HEATMAP:SKELETON_16]
    affine_t = torch.tensor(affine_np, dtype=torch.float32)  # (2, 3)

    # Apply blur
    if config.heatmap_blur_sigma > 0:
        heatmaps_t = apply_blur(heatmaps_t, config.heatmap_blur_sigma)

    # Rotation penalty weights
    rot_per_joint_weights = torch.tensor(
        config.rotation_penalty_per_joint, dtype=torch.float32,
    )

    # --- Optimizer: 3 param groups ---
    optimizer = torch.optim.Adam([
        {"params": [param_root_pos], "lr": config.learning_rate * 1.5},
        {"params": [param_root_rot, param_local_rots], "lr": config.learning_rate},
        {"params": [param_bone_lengths], "lr": config.bone_length_lr},
    ])

    loss_history: list[float] = []
    num_steps = config.num_steps

    if verbose:
        logger.info("Optimising %d frames for %d steps (batched)...", n_frames, num_steps)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Batched FK
        all_positions = forward_kinematics_batch(
            param_root_pos, param_root_rot, param_local_rots, param_bone_lengths,
        )  # (F, 16, 3) [3D:SKELETON_16]

        # Batched projection
        pos_flat = all_positions.reshape(-1, 3)
        proj_flat = camera.camera_to_image_torch(pos_flat)
        all_projected_2d = proj_flat.reshape(n_frames, NUM_JOINTS, 2)  # (F, 16, 2) [2D:SKELETON_16]

        # Batched scoring
        total_score, details = compute_total_score_batch(
            all_positions, all_projected_2d, param_local_rots,
            visibility_t,
            config.position_penalty_weight, rot_per_joint_weights,
            heatmaps=heatmaps_t,
            affine=affine_t,
            confidence_epsilon=config.confidence_epsilon,
        )

        loss = -total_score
        loss.backward()
        optimizer.step()

        # Clamp bone lengths
        with torch.no_grad():
            param_bone_lengths.clamp_(min=0.01)

        loss_history.append(float(loss.item()))

        if verbose and (step % 20 == 0 or step == num_steps - 1):
            logger.info(
                "Step %4d/%d  loss=%.1f  heatmap=%.1f  pos_p=%.4f  rot_p=%.4f",
                step, num_steps, loss.item(),
                details['heatmap'], details['pos_penalty'], details['rot_penalty'],
            )

    # --- Extract final positions ---
    optimized_3d: list[np.ndarray] = []
    with torch.no_grad():
        final_positions = forward_kinematics_batch(
            param_root_pos, param_root_rot, param_local_rots, param_bone_lengths,
        )
        for i in range(n_frames):
            optimized_3d.append(final_positions[i].numpy().copy())

    bone_lengths_final = param_bone_lengths.detach().numpy().copy()

    return optimized_3d, bone_lengths_final, loss_history
