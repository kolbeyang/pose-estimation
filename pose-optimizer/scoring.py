"""Unified scoring functions for FK optimization.

Supports both real Stacked Hourglass heatmaps (MotionBert) and synthetic
Gaussian heatmaps (MediaPipe) through a single grid_sample-based pipeline.
"""

import numpy as np
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
    8,  # 8: Neck (Base of Neck) -- MPII Upper Neck
    9,  # 9: Head -- MPII Head Top
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
    use_mpii_mapping: bool = True,
) -> torch.Tensor:
    """Batched heatmap scoring across all frames.

    For MotionBERT (real SH heatmaps): uses MPII mapping to score all 16 joints.
    For MediaPipe (synthetic heatmaps): scores all 16 joints directly
    (heatmaps have 16 channels matching skeleton indices).

    Args:
        projected_2d_batch: (F, 16, 2) projected positions in pixel coords.
            [2D:SKELETON_16]
        heatmaps_batch: (F, C, H, W) heatmaps per frame.
            [HEATMAP:MPII_16] when use_mpii_mapping=True,
            [HEATMAP:SKELETON_16] when use_mpii_mapping=False.
        affine: (2, 3) affine transform from heatmap-crop to original pixels.
        visibility_batch: (F, 16) confidence scores. [VIS:SKELETON_16]
        confidence_epsilon: Floor for low-confidence joints.
        eps: Floor to avoid log(0).
        use_mpii_mapping: If True, use MPII->skeleton mapping.
            If False, assume heatmaps are (F, 16, H, W) in skeleton order.

    Returns:
        Scalar total score across all frames.
    """
    n_frames = projected_2d_batch.shape[0]
    hm_h = heatmaps_batch.shape[2]
    hm_w = heatmaps_batch.shape[3]

    sx = affine[0, 0]
    sy = affine[1, 1]
    tx = affine[0, 2]
    ty = affine[1, 2]

    if use_mpii_mapping:
        # MotionBert mode: all 16 joints with MPII heatmaps
        hm_proj = projected_2d_batch[:, _HM_SKEL_INDICES, :]  # (F, 16, 2)
        hm_conf = visibility_batch[:, _HM_SKEL_INDICES]  # (F, 16)
        hm_selected = heatmaps_batch[:, _HM_MPII_INDICES, :, :]  # (F, 16, H, W)
        n_joints = 16
    else:
        # MediaPipe mode: all 16 joints directly
        hm_proj = projected_2d_batch  # (F, 16, 2)
        hm_conf = visibility_batch  # (F, 16)
        hm_selected = heatmaps_batch  # (F, 16, H, W)
        n_joints = hm_proj.shape[1]

    # Convert pixel coords to normalized grid coords via inverse affine
    x_crop = (hm_proj[:, :, 0] - tx) / sx  # (F, n_joints)
    y_crop = (hm_proj[:, :, 1] - ty) / sy

    if use_mpii_mapping:
        crop_to_hm_ratio = 4.0  # SH: 256 -> 64
    else:
        crop_to_hm_ratio = 1.0  # synthetic: heatmap size = crop size

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
    use_mpii_mapping: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fully vectorized scoring across all frames.

    Args:
        all_positions: (F, 16, 3) 3D positions. [3D:SKELETON_16]
        all_projected_2d: (F, 16, 2) projected 2D. [2D:SKELETON_16]
        all_local_rots: (F, 16, 3) local rotations. [FK_PARAMS]
        visibility: (F, 16) visibility weights. [VIS:SKELETON_16]
        position_penalty_weight: Weight for position penalty.
        rotation_per_joint_weights: (16,) per-joint rotation penalty weights.
        heatmaps: (F, C, H, W) heatmaps.
            [HEATMAP:MPII_16] or [HEATMAP:SKELETON_16] depending on mode.
        affine: (2, 3) affine transform.
        confidence_epsilon: Floor for low-confidence joints.
        use_mpii_mapping: Whether to use MPII->skeleton joint mapping.

    Returns:
        Tuple of (total_score tensor, details dict with component values).
    """
    total_heatmap = heatmap_score_batch(
        all_projected_2d,
        heatmaps,
        affine,
        visibility,
        confidence_epsilon=confidence_epsilon,
        use_mpii_mapping=use_mpii_mapping,
    )

    total_pos_penalty = motion_penalty_position_batch(all_positions)
    total_rot_penalty = motion_penalty_rotation_batch(
        all_local_rots,
        rotation_per_joint_weights,
    )

    total_score = (
        total_heatmap - position_penalty_weight * total_pos_penalty - total_rot_penalty
    )

    details: dict[str, float] = {
        "heatmap": float(total_heatmap.item()),
        "pos_penalty": float(total_pos_penalty.item()),
        "rot_penalty": float(total_rot_penalty.item()),
        "total": float(total_score.item()),
    }
    return total_score, details


# ---------------------------------------------------------------------------
# Synthetic heatmap generation (for MediaPipe)
# ---------------------------------------------------------------------------


def generate_synthetic_heatmaps(
    target_2d: np.ndarray,
    image_size: tuple[int, int],
    heatmap_size: int = 64,
    sigma: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian heatmaps from 2D detections.

    Creates [HEATMAP:SKELETON_16] heatmaps with one Gaussian blob per joint,
    centered at the detected 2D position.

    Args:
        target_2d: (F, 16, 2) 2D detections in pixel coordinates.
            [2D:SKELETON_16]
        image_size: (height, width) of the original image.
        heatmap_size: Size of the output heatmaps (default 64).
        sigma: Gaussian sigma in pixel space.

    Returns:
        Tuple of:
            heatmaps: (F, 16, heatmap_size, heatmap_size) float32.
                [HEATMAP:SKELETON_16]
            affine: (2, 3) mapping from heatmap coords to pixel coords.
    """
    h, w = image_size
    n_frames, n_joints = target_2d.shape[0], target_2d.shape[1]

    # Affine: maps [0, heatmap_size-1] to [0, image_size-1]
    # So pixel = affine @ [hm_x, hm_y, 1]
    sx = w / heatmap_size
    sy = h / heatmap_size
    affine = np.array(
        [
            [sx, 0, 0],
            [0, sy, 0],
        ],
        dtype=np.float32,
    )

    # Convert sigma from pixel space to heatmap space (per-axis for non-square images)
    sigma_hm_x = sigma / sx
    sigma_hm_y = sigma / sy

    # Create coordinate grids
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size]  # (H, W) each
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    heatmaps = np.zeros(
        (n_frames, n_joints, heatmap_size, heatmap_size), dtype=np.float32
    )

    for f in range(n_frames):
        for j in range(n_joints):
            # Convert pixel coords to heatmap coords
            cx_hm = target_2d[f, j, 0] / sx
            cy_hm = target_2d[f, j, 1] / sy

            # Gaussian blob (circular in pixel space, elliptical in heatmap space)
            heatmaps[f, j] = np.exp(
                -(
                    (xx - cx_hm) ** 2 / (2 * sigma_hm_x**2)
                    + (yy - cy_hm) ** 2 / (2 * sigma_hm_y**2)
                )
            )

    return heatmaps, affine


# ---------------------------------------------------------------------------
# Heatmap blur utility
# ---------------------------------------------------------------------------


# TODO: I would prefer to use OpenCV (cv2) here instead of scipi
def apply_blur(
    heatmaps: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Apply Gaussian blur to heatmaps.

    Args:
        heatmaps: (F, C, H, W) tensor.
        sigma: Gaussian sigma in heatmap pixel space.

    Returns:
        (F, C, H, W) blurred heatmaps.
    """
    import scipy.ndimage

    result = torch.zeros_like(heatmaps)
    for f in range(heatmaps.shape[0]):
        for c in range(heatmaps.shape[1]):
            hm_np = heatmaps[f, c].numpy()
            blurred = scipy.ndimage.gaussian_filter(hm_np, sigma=sigma)
            result[f, c] = torch.tensor(blurred, dtype=torch.float32)
    return result
