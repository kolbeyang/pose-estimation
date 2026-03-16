"""Scoring functions for FK optimization.

Uses analytical Gaussian score (equivalent to sampling log-Gaussian heatmaps
but without materialising any heatmap images).
"""

import torch


def heatmap_score(
    projected_2d: torch.Tensor,
    target_2d: torch.Tensor,
    visibility: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Analytical Gaussian heatmap log-likelihood.

    Equivalent to generating a Gaussian blob at each target_2d position and
    sampling its log-value at the projected position, but computed in closed
    form -- O(J) instead of O(J * H * W).

    Args:
        projected_2d: (J, 2) differentiable projected 2D positions.
        target_2d: (J, 2) target 2D positions from detection.
        visibility: (J,) weights in [0, 1].
        sigma: Gaussian sigma in pixels (controls blur / gradient basin).

    Returns:
        Scalar score (higher = better alignment).
    """
    diff: torch.Tensor = projected_2d - target_2d
    sq_dist: torch.Tensor = (diff ** 2).sum(dim=-1)  # (J,)
    log_likelihood: torch.Tensor = -sq_dist / (2.0 * sigma ** 2)
    return (log_likelihood * visibility).sum()


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


def compute_total_score(
    all_positions: list[torch.Tensor],
    all_projected_2d: list[torch.Tensor],
    all_local_rots: list[torch.Tensor],
    target_2d_list: list[torch.Tensor],
    visibility_list: list[torch.Tensor],
    sigma: float,
    position_penalty_weight: float,
    rotation_per_joint_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total score across all frames.

    total = sum(heatmap_scores) - pos_w * sum(pos_penalties) - sum(rot_penalties)

    Args:
        all_positions: Per-frame (17, 3) 3D positions.
        all_projected_2d: Per-frame (17, 2) projected 2D.
        all_local_rots: Per-frame (17, 3) local rotations.
        target_2d_list: Per-frame (17, 2) target 2D positions.
        visibility_list: Per-frame (17,) visibility weights.
        sigma: Gaussian sigma in pixels.
        position_penalty_weight: Weight for position penalty.
        rotation_per_joint_weights: (17,) per-joint rotation penalty weights.

    Returns:
        (total_score, details_dict)
        total_score is positive = good, to be maximised (loss = -score).
    """
    n_frames: int = len(all_positions)
    total_heatmap: torch.Tensor = torch.tensor(0.0)
    total_pos_penalty: torch.Tensor = torch.tensor(0.0)
    total_rot_penalty: torch.Tensor = torch.tensor(0.0)

    for i in range(n_frames):
        total_heatmap = total_heatmap + heatmap_score(
            all_projected_2d[i], target_2d_list[i], visibility_list[i], sigma,
        )
        if i > 0:
            total_pos_penalty = total_pos_penalty + motion_penalty_position(
                all_positions[i - 1], all_positions[i],
            )
            total_rot_penalty = total_rot_penalty + motion_penalty_rotation(
                all_local_rots[i - 1], all_local_rots[i],
                rotation_per_joint_weights,
            )

    total_score: torch.Tensor = (
        total_heatmap
        - position_penalty_weight * total_pos_penalty
        - total_rot_penalty
    )

    details: dict[str, float] = {
        "heatmap": float(total_heatmap.item()),
        "pos_penalty": float(total_pos_penalty.item()),
        "rot_penalty": float(total_rot_penalty.item()),
        "total": float(total_score.item()),
    }
    return total_score, details
