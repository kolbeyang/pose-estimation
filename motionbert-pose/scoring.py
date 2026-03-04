"""Scoring functions for full-body FK optimization.

Uses real Stacked Hourglass heatmaps with bilinear sampling via F.grid_sample.
"""

import torch
import torch.nn.functional as F


# --- H36M ↔ MPII heatmap joint mapping ---
# Maps H36M joint index → MPII heatmap channel index (0-15).
# H36M joint 7 (Spine) has no direct MPII heatmap (it's a midpoint).
# H36M joints 9 (Head) and 10 (HeadTop) both map to MPII channel 9 (head).
H36M_TO_MPII_HEATMAP = {
    0: None,   # Hip = midpoint(rhip, lhip) — no single heatmap
    1: 2,      # RHip → MPII rhip
    2: 1,      # RKnee → MPII rkne
    3: 0,      # RAnkle → MPII rank
    4: 3,      # LHip → MPII lhip
    5: 4,      # LKnee → MPII lkne
    6: 5,      # LAnkle → MPII lank
    7: None,   # Spine = midpoint(pelv, thrx) — no single heatmap
    8: 8,      # Neck → MPII neck (note: H36M "Thorax" ≈ MPII "neck")
    9: 9,      # Head → MPII head
    10: None,  # HeadTop — shares MPII channel 9 with joint 9 (degenerate)
    11: 13,    # LShoulder → MPII lsho
    12: 14,    # LElbow → MPII lelb
    13: 15,    # LWrist → MPII lwri
    14: 12,    # RShoulder → MPII rsho
    15: 11,    # RElbow → MPII relb
    16: 10,    # RWrist → MPII rwri
}

# Precompute which H36M joints have heatmaps and their MPII channel
_JOINTS_WITH_HEATMAP = [(h36m_j, mpii_ch) for h36m_j, mpii_ch in H36M_TO_MPII_HEATMAP.items() if mpii_ch is not None]


def prepare_heatmaps(
    raw_heatmaps: list,
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    """Prepare raw heatmaps for scoring: normalize, optionally blur, take log.

    Args:
        raw_heatmaps: List of (16, 64, 64) numpy arrays, one per frame.
        blur_sigma: Gaussian blur sigma in heatmap space (64x64). 0 = no blur.

    Returns:
        (N, 16, 64, 64) log-heatmap tensor ready for grid_sample.
    """
    import numpy as np
    hm = np.stack(raw_heatmaps, axis=0)  # (N, 16, 64, 64)
    hm_t = torch.from_numpy(hm).float()

    # Normalize each joint's heatmap to [0, 1] per-frame
    # Reshape to (N*16, 64*64) for per-channel min/max
    N, C, H, W = hm_t.shape
    flat = hm_t.view(N * C, H * W)
    mins = flat.min(dim=1, keepdim=True).values
    maxs = flat.max(dim=1, keepdim=True).values
    denom = (maxs - mins).clamp(min=1e-8)
    flat = (flat - mins) / denom
    hm_t = flat.view(N, C, H, W)

    # Optional Gaussian blur
    if blur_sigma > 0.5:
        kernel_size = int(blur_sigma * 6) | 1  # ensure odd
        kernel_size = max(kernel_size, 3)
        # Separable 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / blur_sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Apply separable blur: horizontal then vertical
        # Reshape to (N*C, 1, H, W) for grouped conv
        hm_flat = hm_t.view(N * C, 1, H, W)
        pad_h = kernel_size // 2
        # Horizontal
        k_h = kernel_1d.view(1, 1, 1, kernel_size)
        hm_flat = F.pad(hm_flat, (pad_h, pad_h, 0, 0), mode='replicate')
        hm_flat = F.conv2d(hm_flat, k_h)
        # Vertical
        k_v = kernel_1d.view(1, 1, kernel_size, 1)
        hm_flat = F.pad(hm_flat, (0, 0, pad_h, pad_h), mode='replicate')
        hm_flat = F.conv2d(hm_flat, k_v)
        hm_t = hm_flat.view(N, C, H, W)

    # Clamp and take log (log-likelihood for scoring)
    hm_t = hm_t.clamp(min=1e-6)
    log_hm = torch.log(hm_t)

    return log_hm


def heatmap_score(
    projected_2d: torch.Tensor,
    log_heatmaps_frame: torch.Tensor,
    visibility: torch.Tensor,
    affine_pixel_to_hm: torch.Tensor,
) -> torch.Tensor:
    """Score projected 2D joints by sampling real log-heatmaps.

    Args:
        projected_2d: (17, 2) differentiable projected 2D positions in pixel coords.
        log_heatmaps_frame: (16, 64, 64) log-heatmaps for this frame.
        visibility: (17,) weights in [0, 1].
        affine_pixel_to_hm: (2, 3) affine mapping pixel coords → heatmap [-1,1] coords
                            for F.grid_sample.

    Returns:
        Scalar score (higher = better alignment).
    """
    total = torch.tensor(0.0)

    for h36m_j, mpii_ch in _JOINTS_WITH_HEATMAP:
        vis = visibility[h36m_j]
        if vis < 0.01:
            continue

        # Transform pixel coords to grid_sample coords [-1, 1]
        px = projected_2d[h36m_j]  # (2,)
        # affine: grid_x = a00*px_x + a01*px_y + a02
        #         grid_y = a10*px_x + a11*px_y + a12
        grid_x = affine_pixel_to_hm[0, 0] * px[0] + affine_pixel_to_hm[0, 1] * px[1] + affine_pixel_to_hm[0, 2]
        grid_y = affine_pixel_to_hm[1, 0] * px[0] + affine_pixel_to_hm[1, 1] * px[1] + affine_pixel_to_hm[1, 2]

        # grid_sample expects (N, C, H, W) input and (N, H_out, W_out, 2) grid
        # We sample a single point: (1, 1, 1, 1, 2)
        grid = torch.stack([grid_x, grid_y]).view(1, 1, 1, 2)  # (1, 1, 1, 2) x,y
        hm_single = log_heatmaps_frame[mpii_ch].view(1, 1, 64, 64)  # (1, 1, 64, 64)

        sampled = F.grid_sample(
            hm_single, grid, mode='bilinear', padding_mode='border', align_corners=False,
        )
        total = total + vis * sampled.squeeze()

    return total


def motion_penalty_position(
    positions_prev: torch.Tensor,
    positions_curr: torch.Tensor,
) -> torch.Tensor:
    """Penalise large position jumps between consecutive frames."""
    diff = positions_curr - positions_prev
    return (diff ** 2).sum()


def motion_penalty_rotation(
    local_rots_prev: torch.Tensor,
    local_rots_curr: torch.Tensor,
) -> torch.Tensor:
    """Penalise large rotation jumps between consecutive frames."""
    diff = local_rots_curr - local_rots_prev
    return (diff ** 2).sum()


def compute_total_score(
    all_positions: list[torch.Tensor],
    all_projected_2d: list[torch.Tensor],
    all_local_rots: list[torch.Tensor],
    log_heatmaps: torch.Tensor,
    visibility_list: list[torch.Tensor],
    affine_pixel_to_hm: torch.Tensor,
    position_penalty_weight: float,
    rotation_penalty_weight: float,
    init_positions: list[torch.Tensor] | None = None,
    init_position_reg_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total score across all frames using real heatmaps.

    Args:
        all_positions: Per-frame (17, 3) 3D positions.
        all_projected_2d: Per-frame (17, 2) projected 2D positions.
        all_local_rots: Per-frame (17, 3) local rotations.
        log_heatmaps: (N, 16, 64, 64) precomputed log-heatmaps.
        visibility_list: Per-frame (17,) visibility weights.
        affine_pixel_to_hm: (2, 3) pixel → grid_sample coord transform.
        position_penalty_weight: Weight for position smoothness.
        rotation_penalty_weight: Weight for rotation smoothness.
        init_positions: Per-frame (17, 3) initial detector 3D positions (for regularisation).
        init_position_reg_weight: Weight for initial position regularisation.

    Returns:
        (total_score, details_dict)
        total_score is positive = good, to be maximised (loss = -score).
    """
    n_frames = len(all_positions)
    total_heatmap = torch.tensor(0.0)
    total_pos_penalty = torch.tensor(0.0)
    total_rot_penalty = torch.tensor(0.0)
    total_init_reg = torch.tensor(0.0)

    for i in range(n_frames):
        total_heatmap = total_heatmap + heatmap_score(
            all_projected_2d[i], log_heatmaps[i], visibility_list[i],
            affine_pixel_to_hm,
        )
        if i > 0:
            total_pos_penalty = total_pos_penalty + motion_penalty_position(
                all_positions[i - 1], all_positions[i],
            )
            total_rot_penalty = total_rot_penalty + motion_penalty_rotation(
                all_local_rots[i - 1], all_local_rots[i],
            )
        if init_positions is not None and init_position_reg_weight > 0:
            diff = all_positions[i] - init_positions[i]
            total_init_reg = total_init_reg + (diff ** 2).sum()

    total_score = (
        total_heatmap
        - position_penalty_weight * total_pos_penalty
        - rotation_penalty_weight * total_rot_penalty
        - init_position_reg_weight * total_init_reg
    )

    details = {
        "heatmap": float(total_heatmap.item()),
        "pos_penalty": float(total_pos_penalty.item()),
        "rot_penalty": float(total_rot_penalty.item()),
        "init_reg": float(total_init_reg.item()),
        "total": float(total_score.item()),
    }
    return total_score, details
