"""Unified scoring functions for FK optimization.

Supports both real Stacked Hourglass heatmaps (MotionBert) and synthetic
Gaussian heatmaps (MediaPipe) through a single grid_sample-based pipeline.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mapping from skeleton 16-joint index to MPII heatmap index.
# All 16 joints now have direct MPII heatmap mappings.
# ---------------------------------------------------------------------------
SKELETON_TO_MPII_HEATMAP: list[int] = [
    6,     # 0: Pelvis
    2,     # 1: RHip
    1,     # 2: RKnee
    0,     # 3: RAnkle
    3,     # 4: LHip
    4,     # 5: LKnee
    5,     # 6: LAnkle
    7,     # 7: Spine -- MPII Thorax (mapped to Spine in our skeleton)
    8,     # 8: Neck (Base of Neck) -- MPII Upper Neck
    9,     # 9: Head -- MPII Head Top
    13,    # 10: LShoulder
    14,    # 11: LElbow
    15,    # 12: LWrist
    12,    # 13: RShoulder
    11,    # 14: RElbow
    10,    # 15: RWrist
]

_HM_SKEL_INDICES: list[int] = list(range(16))
_HM_MPII_INDICES: list[int] = SKELETON_TO_MPII_HEATMAP


# ---------------------------------------------------------------------------
# Heatmap scoring (batched)
# ---------------------------------------------------------------------------

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

    For MotionBert (real SH heatmaps): uses MPII mapping to score all 16 joints.
    For MediaPipe (synthetic heatmaps): scores all 16 joints directly
    (heatmaps have 16 channels matching skeleton indices).

    Args:
        projected_2d_batch: (F, J, 2) projected positions in pixel coords.
        heatmaps_batch: (F, C, H, W) heatmaps per frame.
        affine: (2, 3) affine transform from heatmap_size-crop to original pixels.
        visibility_batch: (F, J) confidence scores.
        confidence_epsilon: Floor for low-confidence joints.
        eps: Floor to avoid log(0).
        use_mpii_mapping: If True, use MPII->skeleton mapping (16 joints).
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
        hm_conf = visibility_batch[:, _HM_SKEL_INDICES]       # (F, 16)
        hm_selected = heatmaps_batch[:, _HM_MPII_INDICES, :, :]  # (F, 16, H, W)
        n_joints = 16
    else:
        # MediaPipe mode: all 16 joints directly
        hm_proj = projected_2d_batch  # (F, 16, 2)
        hm_conf = visibility_batch    # (F, 16)
        hm_selected = heatmaps_batch  # (F, 16, H, W)
        n_joints = hm_proj.shape[1]

    # Convert pixel coords to normalized grid coords via inverse affine
    x_crop = (hm_proj[:, :, 0] - tx) / sx  # (F, n_joints)
    y_crop = (hm_proj[:, :, 1] - ty) / sy

    # Crop coords -> heatmap coords -> [-1, 1] for grid_sample
    # crop_size = hm_size * (crop_px / hm_px), typically 256->64 so /4
    # But we want to be general: grid_sample expects [-1, 1] over the heatmap
    crop_to_hm_x = hm_w / (sx * hm_w)  # simplifies but keeping general
    crop_to_hm_y = hm_h / (sy * hm_h)

    # Actually: the affine maps heatmap-crop to pixels. The crop is typically
    # at crop_size (e.g. 256). The heatmap is at hm_size (e.g. 64).
    # So crop_coord -> hm_coord = crop_coord * hm_size / crop_size
    # For standard SH: crop_size=256, hm_size=64, ratio=0.25
    # grid_sample [-1,1] maps to [0, hm_size-1], so:
    #   grid = hm_coord / (hm_size - 1) * 2 - 1
    # The affine sx encodes crop_size / image_size, so:
    #   crop_coord = pixel_coord / sx (roughly)
    # We need: crop_coord -> normalized grid coord
    # crop_coord is in crop-pixel space (e.g., 0-255 for 256 crop)
    # For SH: crop_size = 256, hm = 64. hm_coord = crop_coord / 4
    # For synthetic: crop_size might equal hm_size

    # Simpler approach: the affine maps from crop_size space to pixel space.
    # The ratio from crop to heatmap is: heatmap_size / crop_size
    # But crop_size is implicit in the affine. sx = crop_size / image_width? No.
    # Actually the affine is: pixel = affine @ [crop_x, crop_y, 1]
    # So crop_x = (pixel_x - tx) / sx, crop_y = (pixel_y - ty) / sy
    # For SH: crop is 256x256, heatmap is 64x64, so hm_x = crop_x / 4

    # We use the crop_size that the affine implies and the actual heatmap size.
    # The crop_size can be inferred: it's the size of the input that produced
    # the heatmap. For SH, crop_size = 256 and hm = 64. For synthetic, both = 64.
    # The safest approach: assume hm_coord = crop_coord * hm_size / crop_size.
    # But we don't know crop_size explicitly. We can pass it, or assume
    # for SH: crop/hm = 4. For synthetic: crop/hm = 1.
    # Instead, let's just use: crop_coord / (crop_size-1) * 2 - 1 directly
    # where crop_size = 256 for SH. This is what the original code does:
    #   grid_x = x_256 / 4.0 / 63.0 * 2.0 - 1.0
    # = x_256 / 252.0 * 2.0 - 1.0 = x_256 / 126.0 - 1.0
    # Equivalently: hm_coord = x_256 / 4; grid = hm_coord / 63 * 2 - 1

    # General form: grid = (crop_coord / crop_to_hm_ratio) / (hm_size - 1) * 2 - 1
    # where crop_to_hm_ratio = crop_size / hm_size

    # For the affine, sx maps crop x to pixel x. The crop image size that
    # produced the heatmap can vary. Let's compute it from the affine scale
    # and image resolution, or just accept a simple assumption.

    # Actually, let's just use the approach from the original motionbert code:
    # x_256 = (pixel_x - tx) / sx; hm_x = x_256 / (crop_pixels / hm_pixels)
    # grid_x = hm_x / (hm_w - 1) * 2 - 1

    # For synthetic heatmaps where crop=hm, crop_to_hm_ratio=1, so:
    # hm_x = x_crop * 1 = x_crop, grid_x = x_crop / (hm_w-1) * 2 - 1

    # For SH: crop=256, hm=64, ratio=4, so:
    # hm_x = x_crop / 4, grid_x = hm_x / 63 * 2 - 1

    # The caller must set crop_to_hm_ratio in the affine or we need it as param.
    # The cleanest approach: derive from affine and heatmap size.
    # Actually, let's just accept it. The affine encodes the crop->pixel mapping.
    # The crop size is whatever it is. Let's compute:
    # The crop dimensions are the full range that maps to the image.
    # For SH: affine maps [0..255] to some pixel range. sx = pixel_range_x / 255.
    # So crop_size_x ~ 256. Then hm_ratio = 256 / 64 = 4.

    # In practice: sx encodes pixels_per_crop_unit. The crop "size" is determined
    # by how many units span the heatmap. Let's just do:
    # crop_size = crop_units * heatmap_size / heatmap_size = crop_units.
    # Hmm, this is getting circular. Let's be pragmatic:

    # We'll use the fact that grid_sample expects [-1,1] normalized coords.
    # A point at pixel (px, py) maps to crop coords via inverse affine,
    # then to heatmap coords via (crop_coord * hm_size / crop_size).
    # For SH: crop_size = 256, hm_size = 64, so factor = 0.25
    # For synthetic: crop_size = hm_size, factor = 1.0

    # Let's compute the effective crop size from the affine: the affine maps
    # crop [0, crop_size-1] to pixel [tx, tx + sx*(crop_size-1)].
    # We don't know crop_size from the affine alone. So we'll just compute
    # the grid coords by normalizing crop_coord to [-1, 1] where the edges
    # of the crop map to the edges of the heatmap.
    # For SH: crop is 256px, heatmap is 64px. crop_coord 0 -> hm 0 -> grid -1.
    # crop_coord 255 -> hm 63 -> grid 1.
    # So grid = crop_coord / 255 * 2 - 1  (NOT /4/63*2-1)
    # Wait, but: 255/4/63*2-1 = 255/252*2-1 = 2.0238-1 = 1.0238 != 1.0
    # Actually the original code has a slight edge effect. Let me re-derive:
    # hm_coord = crop_coord / 4 (for SH)
    # grid = hm_coord / (hm_w-1) * 2 - 1 (for align_corners=True)
    # At crop_coord=252: hm=63, grid=63/63*2-1=1.0 (correct)
    # At crop_coord=0: hm=0, grid=0/63*2-1=-1.0 (correct)
    # At crop_coord=255: hm=63.75, grid=63.75/63*2-1=1.0238 (slightly OOB)
    # This is fine -- grid_sample with padding_mode="zeros" handles it.

    # For generality, let's compute: the crop_size that the affine implies.
    # We'll assume crop maps [0, crop_size-1] uniformly. crop_size is such that
    # the affine scale = image_scale / crop_scale. For SH:
    # Actually, let's just have the caller tell us the crop_size to heatmap ratio.
    # For now, we'll auto-detect: if heatmaps_batch has 16 channels and
    # use_mpii_mapping is False, assume ratio=1 (synthetic). Otherwise ratio=4.

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
    grid_flat = torch.stack([
        grid_x.reshape(-1), grid_y.reshape(-1)
    ], dim=-1).reshape(-1, 1, 1, 2)  # (F*n_joints, 1, 1, 2)

    sampled = F.grid_sample(
        hm_flat, grid_flat, mode="bilinear", padding_mode="zeros", align_corners=True,
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
        all_positions: (F, J, 3) positions.

    Returns:
        Scalar squared L2 distance sum of root joint.
    """
    if all_positions.shape[0] <= 1:
        return torch.tensor(0.0)
    root_diff = all_positions[1:, 0, :] - all_positions[:-1, 0, :]  # (F-1, 3)
    return (root_diff ** 2).sum()


def motion_penalty_rotation_batch(
    all_local_rots: torch.Tensor,
    per_joint_weights: torch.Tensor,
) -> torch.Tensor:
    """Penalize large rotation jumps between consecutive frames.

    Uses chord distance.

    Args:
        all_local_rots: (F, J, 3) axis-angle rotations.
        per_joint_weights: (J,) per-joint penalty weights.

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
        all_positions: (F, J, 3) 3D positions.
        all_projected_2d: (F, J, 2) projected 2D.
        all_local_rots: (F, J, 3) local rotations.
        visibility: (F, J) visibility weights.
        position_penalty_weight: Weight for position penalty.
        rotation_per_joint_weights: (J,) per-joint rotation penalty weights.
        heatmaps: (F, C, H, W) heatmaps.
        affine: (2, 3) affine transform.
        confidence_epsilon: Floor for low-confidence joints.
        use_mpii_mapping: Whether to use MPII->skeleton joint mapping.

    Returns:
        (total_score, details_dict).
    """
    total_heatmap = heatmap_score_batch(
        all_projected_2d, heatmaps, affine, visibility,
        confidence_epsilon=confidence_epsilon,
        use_mpii_mapping=use_mpii_mapping,
    )

    total_pos_penalty = motion_penalty_position_batch(all_positions)
    total_rot_penalty = motion_penalty_rotation_batch(
        all_local_rots, rotation_per_joint_weights,
    )

    total_score = (
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

    Creates (F, K, heatmap_size, heatmap_size) Gaussian heatmaps and an
    affine transform mapping heatmap coords to pixel coords.

    Args:
        target_2d: (F, K, 2) 2D detections in pixel coordinates.
        image_size: (height, width) of the original image.
        heatmap_size: Size of the output heatmaps (default 64).
        sigma: Gaussian sigma in pixel space.

    Returns:
        Tuple of:
            heatmaps: (F, K, heatmap_size, heatmap_size) float32.
            affine: (2, 3) mapping from heatmap coords to pixel coords.
    """
    h, w = image_size
    n_frames, n_joints = target_2d.shape[0], target_2d.shape[1]

    # Affine: maps [0, heatmap_size-1] to [0, image_size-1]
    # So pixel = affine @ [hm_x, hm_y, 1]
    sx = w / heatmap_size
    sy = h / heatmap_size
    affine = np.array([
        [sx, 0, 0],
        [0, sy, 0],
    ], dtype=np.float32)

    # Convert sigma from pixel space to heatmap space (per-axis for non-square images)
    sigma_hm_x = sigma / sx
    sigma_hm_y = sigma / sy

    # Create coordinate grids
    yy, xx = np.mgrid[0:heatmap_size, 0:heatmap_size]  # (H, W) each
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    heatmaps = np.zeros((n_frames, n_joints, heatmap_size, heatmap_size), dtype=np.float32)

    for f in range(n_frames):
        for j in range(n_joints):
            # Convert pixel coords to heatmap coords
            cx_hm = target_2d[f, j, 0] / sx
            cy_hm = target_2d[f, j, 1] / sy

            # Gaussian blob (circular in pixel space, elliptical in heatmap space)
            heatmaps[f, j] = np.exp(
                -((xx - cx_hm) ** 2 / (2 * sigma_hm_x ** 2)
                  + (yy - cy_hm) ** 2 / (2 * sigma_hm_y ** 2))
            )

    return heatmaps, affine


# ---------------------------------------------------------------------------
# Heatmap blur utility
# ---------------------------------------------------------------------------

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
