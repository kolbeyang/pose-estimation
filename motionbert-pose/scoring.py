"""Scoring functions for FK optimization.

Uses real Stacked Hourglass heatmap sampling: the optimizer samples directly
from the (16, 64, 64) heatmaps produced by Stacked Hourglass, preserving
the spatial uncertainty encoded in the heatmaps. Only the 14 joints with
a dedicated MPII heatmap are scored (Hip and Spine are synthetic midpoints
with no heatmap and are excluded from evaluation).
"""

import torch
import torch.nn.functional as F


# Mapping from H36M 16-joint index to MPII heatmap index.
# None means no single MPII heatmap exists (synthetic midpoint joints).
H36M_TO_MPII_HEATMAP: list[int | None] = [
    None,  # 0: Hip (midpoint of RHip + LHip, no single heatmap)
    2,     # 1: RHip
    1,     # 2: RKnee
    0,     # 3: RAnkle
    3,     # 4: LHip
    4,     # 5: LKnee
    5,     # 6: LAnkle
    None,  # 7: Spine (midpoint of Pelvis + Thorax, no single heatmap)
    7,     # 8: Thorax
    8,     # 9: Neck
    13,    # 10: LShoulder
    14,    # 11: LElbow
    15,    # 12: LWrist
    12,    # 13: RShoulder
    11,    # 14: RElbow
    10,    # 15: RWrist
]

# Precomputed indices for vectorized scoring (14 joints with real heatmaps)
_HM_H36M_INDICES: list[int] = [j for j, m in enumerate(H36M_TO_MPII_HEATMAP) if m is not None]
_HM_MPII_INDICES: list[int] = [m for m in H36M_TO_MPII_HEATMAP if m is not None]


def heatmap_score(
    projected_2d: torch.Tensor,
    heatmaps: torch.Tensor,
    affine: torch.Tensor,
    visibility: torch.Tensor,
    confidence_epsilon: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Score by sampling real Stacked Hourglass heatmaps at projected positions.

    Vectorized: samples all 14 heatmap joints in a single grid_sample call.
    Joints without a dedicated MPII heatmap (Hip=0, Spine=7) are skipped.

    Uses confidence-weighted scoring:
        log_val = log(value * conf + confidence_epsilon * (1 - conf))

    Args:
        projected_2d: (J, 2) projected positions in original image pixels.
        heatmaps: (16, 64, 64) Stacked Hourglass heatmaps (MPII joints).
        affine: (2, 3) affine transform from 256-crop coords to original pixels.
        visibility: (J,) confidence scores from Stacked Hourglass.
        confidence_epsilon: Floor for low-confidence joints (prevents log(0)).
        eps: Floor value to avoid log(0).

    Returns:
        Scalar score (higher = better alignment).
    """
    # Invert affine: original pixels -> 256-crop coords -> 64x64 heatmap -> [-1,1]
    sx: torch.Tensor = affine[0, 0]
    sy: torch.Tensor = affine[1, 1]
    tx: torch.Tensor = affine[0, 2]
    ty: torch.Tensor = affine[1, 2]

    # Vectorized heatmap joints (14 joints): single grid_sample call
    hm_proj: torch.Tensor = projected_2d[_HM_H36M_INDICES]  # (14, 2)
    hm_conf: torch.Tensor = visibility[_HM_H36M_INDICES]    # (14,)

    # Convert to 64x64 normalized coords
    x_256: torch.Tensor = (hm_proj[:, 0] - tx) / sx
    y_256: torch.Tensor = (hm_proj[:, 1] - ty) / sy
    grid_x: torch.Tensor = x_256 / 4.0 / 63.0 * 2.0 - 1.0  # (14,)
    grid_y: torch.Tensor = y_256 / 4.0 / 63.0 * 2.0 - 1.0  # (14,)

    # Use (14, 1, 64, 64) as batch dim, (14, 1, 1, 2) as grid
    # so each joint's heatmap is sampled at its own location
    hm_batch: torch.Tensor = heatmaps[_HM_MPII_INDICES].unsqueeze(1)  # (14, 1, 64, 64)
    grid_batch: torch.Tensor = torch.stack([grid_x, grid_y], dim=-1).reshape(14, 1, 1, 2)

    sampled: torch.Tensor = F.grid_sample(
        hm_batch, grid_batch, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (14, 1, 1, 1)
    values: torch.Tensor = sampled.reshape(14)  # (14,)

    # Confidence-weighted log-likelihood (vectorized)
    weighted: torch.Tensor = values * hm_conf + confidence_epsilon * (1.0 - hm_conf)
    log_vals: torch.Tensor = torch.log(torch.clamp(weighted, min=eps))
    return log_vals.sum()


def heatmap_score_batch(
    projected_2d_batch: torch.Tensor,
    heatmaps_batch: torch.Tensor,
    affine: torch.Tensor,
    visibility_batch: torch.Tensor,
    confidence_epsilon: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Batched heatmap scoring across all frames at once.

    Scores only the 14 joints with dedicated MPII heatmaps.

    Args:
        projected_2d_batch: (F, J, 2) projected positions.
        heatmaps_batch: (F, 16, 64, 64) heatmaps per frame.
        affine: (2, 3) shared affine transform.
        visibility_batch: (F, J) confidence scores.
        confidence_epsilon: Floor for low-confidence joints.
        eps: Floor to avoid log(0).

    Returns:
        Scalar total score across all frames.
    """
    n_frames: int = projected_2d_batch.shape[0]

    sx: torch.Tensor = affine[0, 0]
    sy: torch.Tensor = affine[1, 1]
    tx: torch.Tensor = affine[0, 2]
    ty: torch.Tensor = affine[1, 2]

    # Heatmap joints (14 joints, batched across frames)
    hm_proj: torch.Tensor = projected_2d_batch[:, _HM_H36M_INDICES, :]  # (F, 14, 2)
    hm_conf: torch.Tensor = visibility_batch[:, _HM_H36M_INDICES]       # (F, 14)

    # Convert to normalized grid coords
    x_256: torch.Tensor = (hm_proj[:, :, 0] - tx) / sx  # (F, 14)
    y_256: torch.Tensor = (hm_proj[:, :, 1] - ty) / sy
    grid_x: torch.Tensor = x_256 / 4.0 / 63.0 * 2.0 - 1.0
    grid_y: torch.Tensor = y_256 / 4.0 / 63.0 * 2.0 - 1.0

    # Reshape to (F*14, 1, 64, 64) batch and (F*14, 1, 1, 2) grid
    hm_selected: torch.Tensor = heatmaps_batch[:, _HM_MPII_INDICES, :, :]  # (F, 14, 64, 64)
    hm_flat: torch.Tensor = hm_selected.reshape(-1, 1, 64, 64)  # (F*14, 1, 64, 64)
    grid_flat: torch.Tensor = torch.stack([
        grid_x.reshape(-1), grid_y.reshape(-1)
    ], dim=-1).reshape(-1, 1, 1, 2)  # (F*14, 1, 1, 2)

    sampled: torch.Tensor = F.grid_sample(
        hm_flat, grid_flat, mode="bilinear", padding_mode="zeros", align_corners=True,
    )  # (F*14, 1, 1, 1)
    values: torch.Tensor = sampled.reshape(n_frames, 14)

    weighted: torch.Tensor = values * hm_conf + confidence_epsilon * (1.0 - hm_conf)
    log_vals: torch.Tensor = torch.log(torch.clamp(weighted, min=eps))
    return log_vals.sum()


def motion_penalty_position(
    positions_prev: torch.Tensor,
    positions_curr: torch.Tensor,
) -> torch.Tensor:
    """Penalize large root (hip) position jumps between consecutive frames.

    Args:
        positions_prev: (17, 3) previous frame joint positions.
        positions_curr: (17, 3) current frame joint positions.

    Returns:
        Scalar squared L2 distance of root joint.
    """
    diff: torch.Tensor = positions_curr[0] - positions_prev[0]
    return (diff ** 2).sum()


def motion_penalty_rotation(
    local_rots_prev: torch.Tensor,
    local_rots_curr: torch.Tensor,
    per_joint_weights: torch.Tensor,
) -> torch.Tensor:
    """Penalize large rotation jumps between consecutive frames.

    Uses chord distance which wraps correctly at +/-pi boundaries,
    unlike raw axis-angle diff.

    Args:
        local_rots_prev: (J, 3) axis-angle rotations for previous frame.
        local_rots_curr: (J, 3) axis-angle rotations for current frame.
        per_joint_weights: (J,) per-joint penalty weights (inner > outer).

    Returns:
        Scalar penalty.
    """
    cos_diff_sq: torch.Tensor = (torch.cos(local_rots_curr) - torch.cos(local_rots_prev)) ** 2
    sin_diff_sq: torch.Tensor = (torch.sin(local_rots_curr) - torch.sin(local_rots_prev)) ** 2
    per_joint: torch.Tensor = (cos_diff_sq + sin_diff_sq).sum(dim=-1)  # (J,)
    return (per_joint * per_joint_weights).sum()


def initialization_penalty(
    positions_3d: torch.Tensor,
    initial_positions_3d: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Penalize deviation from initial (MotionBERT) positions.

    Only penalizes joints that the detector is confident about.
    Low-visibility joints are free to move (the optimizer should fix them).

    Args:
        positions_3d: (17, 3) current optimized positions.
        initial_positions_3d: (17, 3) MotionBERT initial positions.
        visibility: (17,) confidence weights.

    Returns:
        Scalar penalty.
    """
    diff: torch.Tensor = positions_3d - initial_positions_3d
    sq_dist: torch.Tensor = (diff ** 2).sum(dim=-1)  # (17,)
    # Weight by visibility -- high-confidence joints are anchored more
    return (sq_dist * visibility).sum()


def compute_total_score(
    all_positions: list[torch.Tensor],
    all_projected_2d: list[torch.Tensor],
    all_local_rots: list[torch.Tensor],
    visibility_list: list[torch.Tensor],
    position_penalty_weight: float,
    rotation_per_joint_weights: torch.Tensor,
    initial_positions_list: list[torch.Tensor] | None = None,
    init_anchor_weight: float = 0.0,
    heatmaps_list: list[torch.Tensor] | None = None,
    affine: torch.Tensor | None = None,
    confidence_epsilon: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total score across all frames.

    total = sum(heatmap_scores) - pos_w * sum(pos_penalties) - sum(rot_penalties)
            - anchor_w * sum(init_penalties)

    Args:
        all_positions: Per-frame (17, 3) 3D positions.
        all_projected_2d: Per-frame (17, 2) projected 2D.
        all_local_rots: Per-frame (17, 3) local rotations.
        visibility_list: Per-frame (17,) visibility weights.
        position_penalty_weight: Weight for position penalty.
        rotation_per_joint_weights: (17,) per-joint rotation penalty weights.
        initial_positions_list: Per-frame (17, 3) initial MotionBERT positions (optional).
        init_anchor_weight: Weight for initialization anchor penalty.
        heatmaps_list: Per-frame (16, 64, 64) Stacked Hourglass heatmaps (optional).
        affine: (2, 3) affine transform from 256-crop to original pixels (optional).

    Returns:
        (total_score, details_dict)
        total_score is positive = good, to be maximised (loss = -score).
    """
    n_frames: int = len(all_positions)
    total_heatmap: torch.Tensor = torch.tensor(0.0)
    total_pos_penalty: torch.Tensor = torch.tensor(0.0)
    total_rot_penalty: torch.Tensor = torch.tensor(0.0)
    total_anchor_penalty: torch.Tensor = torch.tensor(0.0)

    if heatmaps_list is None or affine is None:
        raise ValueError("Heatmaps and affine are required for scoring.")

    for i in range(n_frames):
        total_heatmap = total_heatmap + heatmap_score(
            all_projected_2d[i],
            heatmaps_list[i],
            affine,
            visibility_list[i],
            confidence_epsilon=confidence_epsilon,
        )
        if i > 0:
            total_pos_penalty = total_pos_penalty + motion_penalty_position(
                all_positions[i - 1], all_positions[i],
            )
            total_rot_penalty = total_rot_penalty + motion_penalty_rotation(
                all_local_rots[i - 1], all_local_rots[i],
                rotation_per_joint_weights,
            )
        if initial_positions_list is not None and init_anchor_weight > 0.0:
            total_anchor_penalty = total_anchor_penalty + initialization_penalty(
                all_positions[i], initial_positions_list[i], visibility_list[i],
            )

    total_score: torch.Tensor = (
        total_heatmap
        - position_penalty_weight * total_pos_penalty
        - total_rot_penalty
        - init_anchor_weight * total_anchor_penalty
    )

    details: dict[str, float] = {
        "heatmap": float(total_heatmap.item()),
        "pos_penalty": float(total_pos_penalty.item()),
        "rot_penalty": float(total_rot_penalty.item()),
        "anchor_penalty": float(total_anchor_penalty.item()),
        "total": float(total_score.item()),
    }
    return total_score, details


def compute_total_score_batch(
    all_positions: torch.Tensor,
    all_projected_2d: torch.Tensor,
    all_local_rots: torch.Tensor,
    visibility: torch.Tensor,
    position_penalty_weight: float,
    rotation_per_joint_weights: torch.Tensor,
    heatmaps: torch.Tensor | None = None,
    affine: torch.Tensor | None = None,
    confidence_epsilon: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fully vectorized scoring across all frames.

    Args:
        all_positions: (F, J, 3) 3D positions.
        all_projected_2d: (F, J, 2) projected 2D.
        all_local_rots: (F, J, 3) local rotations.
        visibility: (F, J) visibility weights.
        position_penalty_weight: Weight for position penalty.
        rotation_per_joint_weights: (J,) per-joint rotation penalty weights.
        heatmaps: (F, 16, 64, 64) heatmaps.
        affine: (2, 3) affine transform.
        confidence_epsilon: Floor for low-confidence joints.

    Returns:
        (total_score, details_dict)
    """
    if heatmaps is None or affine is None:
        raise ValueError("Heatmaps and affine are required for scoring.")

    # Heatmap score (batched across all frames)
    total_heatmap: torch.Tensor = heatmap_score_batch(
        all_projected_2d, heatmaps, affine, visibility,
        confidence_epsilon=confidence_epsilon,
    )

    # Position penalty: root joint distance between consecutive frames
    if all_positions.shape[0] > 1:
        root_diff: torch.Tensor = all_positions[1:, 0, :] - all_positions[:-1, 0, :]  # (F-1, 3)
        total_pos_penalty: torch.Tensor = (root_diff ** 2).sum()
    else:
        total_pos_penalty = torch.tensor(0.0)

    # Rotation penalty: chord distance between consecutive frames
    if all_local_rots.shape[0] > 1:
        cos_diff_sq: torch.Tensor = (torch.cos(all_local_rots[1:]) - torch.cos(all_local_rots[:-1])) ** 2
        sin_diff_sq: torch.Tensor = (torch.sin(all_local_rots[1:]) - torch.sin(all_local_rots[:-1])) ** 2
        per_joint: torch.Tensor = (cos_diff_sq + sin_diff_sq).sum(dim=-1)  # (F-1, J)
        total_rot_penalty: torch.Tensor = (per_joint * rotation_per_joint_weights.unsqueeze(0)).sum()
    else:
        total_rot_penalty = torch.tensor(0.0)

    total_score: torch.Tensor = (
        total_heatmap
        - position_penalty_weight * total_pos_penalty
        - total_rot_penalty
    )

    details: dict[str, float] = {
        "heatmap": float(total_heatmap.item()),
        "pos_penalty": float(total_pos_penalty.item()),
        "rot_penalty": float(total_rot_penalty.item()),
        "anchor_penalty": 0.0,
        "total": float(total_score.item()),
    }
    return total_score, details
