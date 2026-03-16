"""Scoring functions for FK optimization.

Supports both analytical Gaussian scoring and real Stacked Hourglass heatmap
sampling. When USE_REAL_HEATMAPS is enabled, the optimizer samples directly
from the (16, 64, 64) heatmaps produced by Stacked Hourglass, preserving
the spatial uncertainty encoded in the heatmaps rather than collapsing to
point estimates.
"""

import torch
import torch.nn.functional as F


# Mapping from H36M joint index to MPII heatmap index.
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
    9,     # 10: Head
    13,    # 11: LShoulder
    14,    # 12: LElbow
    15,    # 13: LWrist
    12,    # 14: RShoulder
    11,    # 15: RElbow
    10,    # 16: RWrist
]


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


def real_heatmap_score(
    projected_2d: torch.Tensor,
    heatmaps: torch.Tensor,
    affine: torch.Tensor,
    visibility: torch.Tensor,
    target_2d: torch.Tensor,
    sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Score by sampling real Stacked Hourglass heatmaps at projected positions.

    For the 15 H36M joints that have a direct MPII heatmap, uses differentiable
    bilinear interpolation (grid_sample) to sample the heatmap value at the
    projected 2D location. For Hip (0) and Spine (7), which are synthetic
    midpoints with no dedicated heatmap, falls back to analytical Gaussian.

    Args:
        projected_2d: (J, 2) projected positions in original image pixels.
        heatmaps: (16, 64, 64) Stacked Hourglass heatmaps (MPII joints).
        affine: (2, 3) affine transform from 256-crop coords to original pixels.
        visibility: (J,) visibility weights in [0, 1].
        target_2d: (J, 2) target 2D positions (for fallback on joints without heatmaps).
        sigma: Gaussian sigma for fallback joints.
        eps: Floor value to avoid log(0).

    Returns:
        Scalar score (higher = better alignment).
    """
    n_joints: int = projected_2d.shape[0]
    total_score: torch.Tensor = torch.tensor(0.0)

    # Invert affine: original pixels -> 256-crop coords
    # affine maps 256-crop -> original: x_orig = sx * x_256 + tx
    # So: x_256 = (x_orig - tx) / sx
    sx: torch.Tensor = affine[0, 0]
    sy: torch.Tensor = affine[1, 1]
    tx: torch.Tensor = affine[0, 2]
    ty: torch.Tensor = affine[1, 2]

    for j in range(n_joints):
        if visibility[j] < 1e-6:
            continue

        mpii_idx: int | None = H36M_TO_MPII_HEATMAP[j]
        if mpii_idx is None:
            # Fallback: analytical Gaussian for Hip (0) and Spine (7)
            diff: torch.Tensor = projected_2d[j] - target_2d[j]
            sq_dist: torch.Tensor = (diff ** 2).sum()
            log_likelihood: torch.Tensor = -sq_dist / (2.0 * sigma ** 2)
            total_score = total_score + log_likelihood * visibility[j]
            continue

        # Convert projected position from original pixels to 256-crop coords
        x_256: torch.Tensor = (projected_2d[j, 0] - tx) / sx
        y_256: torch.Tensor = (projected_2d[j, 1] - ty) / sy

        # Convert 256-crop to 64x64 heatmap coords
        x_64: torch.Tensor = x_256 / 4.0
        y_64: torch.Tensor = y_256 / 4.0

        # Normalize to [-1, 1] for grid_sample
        grid_x: torch.Tensor = x_64 / 63.0 * 2.0 - 1.0
        grid_y: torch.Tensor = y_64 / 63.0 * 2.0 - 1.0

        # grid_sample expects (N, C, H, W) input and (N, H_out, W_out, 2) grid
        hm: torch.Tensor = heatmaps[mpii_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        grid: torch.Tensor = torch.stack([grid_x, grid_y]).reshape(1, 1, 1, 2)  # (1, 1, 1, 2)

        sampled: torch.Tensor = F.grid_sample(
            hm, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )  # (1, 1, 1, 1)
        value: torch.Tensor = sampled.squeeze()  # scalar

        # Log-likelihood weighted by visibility
        log_val: torch.Tensor = torch.log(torch.clamp(value, min=eps))
        total_score = total_score + log_val * visibility[j]

    return total_score


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


def motion_penalty_all_joints(
    positions_prev: torch.Tensor,
    positions_curr: torch.Tensor,
) -> torch.Tensor:
    """Penalize position jumps for ALL joints between consecutive frames.

    Args:
        positions_prev: (17, 3) previous frame.
        positions_curr: (17, 3) current frame.

    Returns:
        Scalar: sum of squared displacements.
    """
    diff: torch.Tensor = positions_curr - positions_prev
    return (diff ** 2).sum()


def compute_total_score(
    all_positions: list[torch.Tensor],
    all_projected_2d: list[torch.Tensor],
    all_local_rots: list[torch.Tensor],
    target_2d_list: list[torch.Tensor],
    visibility_list: list[torch.Tensor],
    sigma: float,
    position_penalty_weight: float,
    rotation_per_joint_weights: torch.Tensor,
    initial_positions_list: list[torch.Tensor] | None = None,
    init_anchor_weight: float = 0.0,
    all_joints_smooth_weight: float = 0.0,
    heatmaps_list: list[torch.Tensor] | None = None,
    affine: torch.Tensor | None = None,
    use_real_heatmaps: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total score across all frames.

    total = sum(heatmap_scores) - pos_w * sum(pos_penalties) - sum(rot_penalties)
            - anchor_w * sum(init_penalties) - smooth_w * sum(all_joint_penalties)

    Args:
        all_positions: Per-frame (17, 3) 3D positions.
        all_projected_2d: Per-frame (17, 2) projected 2D.
        all_local_rots: Per-frame (17, 3) local rotations.
        target_2d_list: Per-frame (17, 2) target 2D positions.
        visibility_list: Per-frame (17,) visibility weights.
        sigma: Gaussian sigma in pixels.
        position_penalty_weight: Weight for position penalty.
        rotation_per_joint_weights: (17,) per-joint rotation penalty weights.
        initial_positions_list: Per-frame (17, 3) initial MotionBERT positions (optional).
        init_anchor_weight: Weight for initialization anchor penalty.
        all_joints_smooth_weight: Weight for all-joint temporal smoothing penalty.
        heatmaps_list: Per-frame (16, 64, 64) Stacked Hourglass heatmaps (optional).
        affine: (2, 3) affine transform from 256-crop to original pixels (optional).
        use_real_heatmaps: If True and heatmaps are provided, sample from real heatmaps.

    Returns:
        (total_score, details_dict)
        total_score is positive = good, to be maximised (loss = -score).
    """
    n_frames: int = len(all_positions)
    total_heatmap: torch.Tensor = torch.tensor(0.0)
    total_pos_penalty: torch.Tensor = torch.tensor(0.0)
    total_rot_penalty: torch.Tensor = torch.tensor(0.0)
    total_anchor_penalty: torch.Tensor = torch.tensor(0.0)
    total_all_joints_smooth: torch.Tensor = torch.tensor(0.0)

    _use_real: bool = (
        use_real_heatmaps
        and heatmaps_list is not None
        and affine is not None
    )

    for i in range(n_frames):
        if _use_real:
            total_heatmap = total_heatmap + real_heatmap_score(
                all_projected_2d[i],
                heatmaps_list[i],  # type: ignore[index]
                affine,  # type: ignore[arg-type]
                visibility_list[i],
                target_2d_list[i],
                sigma,
            )
        else:
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
            if all_joints_smooth_weight > 0.0:
                total_all_joints_smooth = total_all_joints_smooth + motion_penalty_all_joints(
                    all_positions[i - 1], all_positions[i],
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
        - all_joints_smooth_weight * total_all_joints_smooth
    )

    details: dict[str, float] = {
        "heatmap": float(total_heatmap.item()),
        "pos_penalty": float(total_pos_penalty.item()),
        "rot_penalty": float(total_rot_penalty.item()),
        "anchor_penalty": float(total_anchor_penalty.item()),
        "all_joints_smooth": float(total_all_joints_smooth.item()),
        "total": float(total_score.item()),
    }
    return total_score, details
