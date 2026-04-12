"""Scoring functions for FK optimization.

Uses real Stacked Hourglass heatmaps with MPII joint mapping.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mapping from [3D:SKELETON_16] joint index to [HEATMAP:MPII_16] channel index.
# Used when scoring optimizer projections against real Stacked Hourglass heatmaps.
# All 16 skeleton joints have direct MPII heatmap channel mappings.
# ---------------------------------------------------------------------------
SKELETON_TO_MPII_HEATMAP: list[int] = [
    6,  # 0: Pelvis
    2,  # 1: RHip
    1,  # 2: RKnee
    0,  # 3: RAnkle
    3,  # 4: LHip
    4,  # 5: LKnee
    5,  # 6: LAnkle
    7,  # 7: Spine -- MPII Thorax (mapped to Spine in our skeleton)
    7,  # 8: Neck (Thorax / shoulder level) -- MPII Thorax
    9,  # 9: Nose -- MPII Head Top
    13,  # 10: LShoulder
    14,  # 11: LElbow
    15,  # 12: LWrist
    12,  # 13: RShoulder
    11,  # 14: RElbow
    10,  # 15: RWrist
]

_HM_SKEL_INDICES: list[int] = list(range(16))
_HM_MPII_INDICES: list[int] = SKELETON_TO_MPII_HEATMAP


# ---------------------------------------------------------------------------
# Heatmap scoring (batched)
# ---------------------------------------------------------------------------


# TODO: Investigate and simplify
def heatmap_score_batch(
    projected_2d_batch: torch.Tensor,
    heatmaps_batch: torch.Tensor,
    affine: torch.Tensor,
    visibility_batch: torch.Tensor,
    confidence_epsilon: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Batched heatmap scoring across all frames.

    Both pipelines use real SH heatmaps with MPII mapping to score all 16 joints.

    Args:
        projected_2d_batch: (F, 16, 2) projected positions in pixel coords.
            [2D:SKELETON_16]
        heatmaps_batch: (F, C, H, W) SH heatmaps per frame. [HEATMAP:MPII_16]
        affine: (2, 3) shared affine, or (F, 2, 3) per-frame affines.
        visibility_batch: (F, 16) confidence scores. [VIS:SKELETON_16]
        confidence_epsilon: Floor for low-confidence joints.
        eps: Floor to avoid log(0).

    Returns:
        Scalar total score across all frames.
    """
    n_frames = projected_2d_batch.shape[0]
    hm_h = heatmaps_batch.shape[2]
    hm_w = heatmaps_batch.shape[3]

    # Handle both shared (2, 3) and per-frame (F, 2, 3) affines
    if affine.dim() == 3:
        # Per-frame affines: (F, 2, 3)
        sx = affine[:, 0, 0].unsqueeze(1)  # (F, 1)
        sy = affine[:, 1, 1].unsqueeze(1)  # (F, 1)
        tx = affine[:, 0, 2].unsqueeze(1)  # (F, 1)
        ty = affine[:, 1, 2].unsqueeze(1)  # (F, 1)
    else:
        # Shared affine: (2, 3)
        sx = affine[0, 0]
        sy = affine[1, 1]
        tx = affine[0, 2]
        ty = affine[1, 2]

    # Map skeleton joints to MPII heatmap channels
    hm_proj = projected_2d_batch[:, _HM_SKEL_INDICES, :]  # (F, 16, 2)
    hm_conf = visibility_batch[:, _HM_SKEL_INDICES]  # (F, 16)
    hm_selected = heatmaps_batch[:, _HM_MPII_INDICES, :, :]  # (F, 16, H, W)
    n_joints = 16

    # Convert pixel coords to normalized grid coords via inverse affine
    x_crop = (hm_proj[:, :, 0] - tx) / sx  # (F, n_joints)
    y_crop = (hm_proj[:, :, 1] - ty) / sy

    crop_to_hm_ratio = 4.0  # SH: 256 -> 64

    hm_x = x_crop / crop_to_hm_ratio  # (F, n_joints)
    hm_y = y_crop / crop_to_hm_ratio
    grid_x = hm_x / (hm_w - 1) * 2.0 - 1.0
    grid_y = hm_y / (hm_h - 1) * 2.0 - 1.0

    # grid_sample: (N, C, H, W) input, (N, H_out, W_out, 2) grid
    # We want to sample each joint from its own channel, so reshape to
    # (F*n_joints, 1, H, W) and (F*n_joints, 1, 1, 2)
    hm_flat = hm_selected.reshape(-1, 1, hm_h, hm_w)  # (F*n_joints, 1, H, W)
    grid_flat = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1).reshape(
        -1, 1, 1, 2
    )  # (F*n_joints, 1, 1, 2)

    sampled = F.grid_sample(
        hm_flat,
        grid_flat,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (F*n_joints, 1, 1, 1)
    values = sampled.reshape(n_frames, n_joints)

    # Confidence-weighted log-likelihood
    weighted = values * hm_conf + confidence_epsilon * (1.0 - hm_conf)
    log_vals = torch.log(torch.clamp(weighted, min=eps))
    return log_vals.sum()


# ---------------------------------------------------------------------------
# Motion penalties (identical for both pipelines)
# ---------------------------------------------------------------------------


def motion_penalty_position_batch(
    all_positions: torch.Tensor,
) -> torch.Tensor:
    """Penalize large root position jumps between consecutive frames.

    Args:
        all_positions: (F, 16, 3) positions. [3D:SKELETON_16]

    Returns:
        Scalar squared L2 distance sum of root joint.
    """
    if all_positions.shape[0] <= 1:
        return torch.tensor(0.0)
    root_diff = all_positions[1:, 0, :] - all_positions[:-1, 0, :]  # (F-1, 3)
    return (root_diff**2).sum()


def motion_penalty_rotation_batch(
    all_local_rots: torch.Tensor,
    per_joint_weights: torch.Tensor,
) -> torch.Tensor:
    """Penalize large rotation jumps between consecutive frames.

    Uses chord distance.

    Args:
        all_local_rots: (F, 16, 3) axis-angle rotations. [FK_PARAMS]
        per_joint_weights: (16,) per-joint penalty weights indexed by
            [SKELETON_16] joint order.

    Returns:
        Scalar penalty.
    """
    if all_local_rots.shape[0] <= 1:
        return torch.tensor(0.0)
    cos_diff_sq = (torch.cos(all_local_rots[1:]) - torch.cos(all_local_rots[:-1])) ** 2
    sin_diff_sq = (torch.sin(all_local_rots[1:]) - torch.sin(all_local_rots[:-1])) ** 2
    per_joint = (cos_diff_sq + sin_diff_sq).sum(dim=-1)  # (F-1, J)
    return (per_joint * per_joint_weights.unsqueeze(0)).sum()


# ---------------------------------------------------------------------------
# Total score
# ---------------------------------------------------------------------------


def compute_total_score_batch(
    all_positions: torch.Tensor,
    all_projected_2d: torch.Tensor,
    all_local_rots: torch.Tensor,
    visibility: torch.Tensor,
    position_penalty_weight: float,
    rotation_per_joint_weights: torch.Tensor,
    heatmaps: torch.Tensor,
    affine: torch.Tensor,
    confidence_epsilon: float = 1e-4,
    anchor_positions: torch.Tensor | None = None,
    anchor_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fully vectorized scoring across all frames.

    Args:
        all_positions: (F, 16, 3) 3D positions. [3D:SKELETON_16]
        all_projected_2d: (F, 16, 2) projected 2D. [2D:SKELETON_16]
        all_local_rots: (F, 16, 3) local rotations. [FK_PARAMS]
        visibility: (F, 16) visibility weights. [VIS:SKELETON_16]
        position_penalty_weight: Weight for position penalty.
        rotation_per_joint_weights: (16,) per-joint rotation penalty weights.
        heatmaps: (F, C, H, W) SH heatmaps. [HEATMAP:MPII_16]
        affine: (2, 3) affine transform.
        confidence_epsilon: Floor for low-confidence joints.
        anchor_positions: (F, 16, 3) initial positions to anchor to. [3D:SKELETON_16]
            If None or anchor_weight=0, no anchor penalty is applied.
        anchor_weight: Weight for anchor penalty (higher = less movement from init).

    Returns:
        Tuple of (total_score tensor, details dict with component values).
    """
    total_heatmap = heatmap_score_batch(
        all_projected_2d,
        heatmaps,
        affine,
        visibility,
        confidence_epsilon=confidence_epsilon,
    )

    total_pos_penalty = motion_penalty_position_batch(all_positions)
    total_rot_penalty = motion_penalty_rotation_batch(
        all_local_rots,
        rotation_per_joint_weights,
    )

    total_score = (
        total_heatmap - position_penalty_weight * total_pos_penalty - total_rot_penalty
    )

    anchor_penalty_val = 0.0
    if anchor_positions is not None and anchor_weight > 0:
        anchor_penalty = ((all_positions - anchor_positions) ** 2).sum()
        total_score = total_score - anchor_weight * anchor_penalty
        anchor_penalty_val = float(anchor_penalty.item())

    details: dict[str, float] = {
        "heatmap": float(total_heatmap.item()),
        "pos_penalty": float(total_pos_penalty.item()),
        "rot_penalty": float(total_rot_penalty.item()),
        "anchor_penalty": anchor_penalty_val,
        "total": float(total_score.item()),
    }
    return total_score, details


# ---------------------------------------------------------------------------
# Heatmap blur utility
# ---------------------------------------------------------------------------


def apply_blur(
    heatmaps: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Apply Gaussian blur to heatmaps using separable torch conv2d.

    Args:
        heatmaps: (F, C, H, W) tensor.
        sigma: Gaussian sigma in heatmap pixel space.

    Returns:
        (F, C, H, W) blurred heatmaps (on same device as input).
    """
    if sigma <= 0:
        return heatmaps

    # Match scipy.ndimage.gaussian_filter default truncate=4.0,
    # but cap kernel to image size for large sigma
    radius = int(4.0 * sigma + 0.5)
    H, W = heatmaps.shape[2], heatmaps.shape[3]
    radius = min(radius, min(H, W) - 1)
    ks = 2 * radius + 1
    pad = radius

    device = heatmaps.device
    x = torch.arange(ks, dtype=torch.float32, device=device) - pad
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    F_count, C, H, W = heatmaps.shape
    flat = heatmaps.reshape(-1, 1, H, W)

    # Horizontal pass
    k_h = kernel_1d.reshape(1, 1, 1, ks)
    flat = torch.nn.functional.pad(flat, (pad, pad, 0, 0), mode='reflect')
    flat = torch.nn.functional.conv2d(flat, k_h)

    # Vertical pass
    k_v = kernel_1d.reshape(1, 1, ks, 1)
    flat = torch.nn.functional.pad(flat, (0, 0, pad, pad), mode='reflect')
    flat = torch.nn.functional.conv2d(flat, k_v)

    return flat.reshape(F_count, C, H, W)
