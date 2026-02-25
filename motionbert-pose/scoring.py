"""Scoring functions for pose optimization (adapted from toy-arm/utils.py)."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from model.arm import Arm
from model.camera import Camera

EPSILON = 1e-10


@dataclass
class OptimizationConfig:
    num_steps: int = 100
    learning_rate: float = 0.0005  # Lower than toy-arm (meters vs arbitrary units)
    position_penalty_weight: float = 40
    ab_rotation_penalty_weight: float = 50
    bc_rotation_penalty_weight: float = 30
    bone_length_lr: float = 0.005


def prepare_heatmaps(
    heatmaps: dict[str, np.ndarray],
    blur_sigma: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Convert heatmaps to log-normalized torch tensors.

    Args:
        heatmaps: dict of uint8 heatmaps
        blur_sigma: Gaussian blur sigma in pixels.  0 = no blur.
    """
    result = {}
    for name, heatmap in heatmaps.items():
        normalized = heatmap.astype(np.float32) / 255.0
        if blur_sigma > 0:
            t = torch.tensor(normalized).unsqueeze(0).unsqueeze(0)
            ksize = int(6 * blur_sigma) | 1  # odd kernel size
            t = _gaussian_blur_2d(t, ksize, blur_sigma)
            normalized = t.squeeze().numpy()
        log_heatmap = np.log(normalized + EPSILON)
        result[name] = torch.tensor(log_heatmap).unsqueeze(0).unsqueeze(0)
    return result


def _gaussian_blur_2d(
    tensor: torch.Tensor, kernel_size: int, sigma: float,
) -> torch.Tensor:
    """Separable Gaussian blur on a (1, 1, H, W) tensor."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    # Horizontal then vertical
    kh = kernel_1d.view(1, 1, 1, -1)
    kv = kernel_1d.view(1, 1, -1, 1)
    pad_h = kernel_size // 2
    out = F.conv2d(F.pad(tensor, [pad_h, pad_h, 0, 0], mode="replicate"), kh)
    out = F.conv2d(F.pad(out, [0, 0, pad_h, pad_h], mode="replicate"), kv)
    return out


def sample_heatmap(
    heatmap: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Sample heatmap at (x, y) using differentiable bilinear interpolation."""
    H, W = heatmap.shape[2], heatmap.shape[3]

    x_norm = 2.0 * x / (W - 1) - 1.0
    y_norm = 2.0 * y / (H - 1) - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, 1, 1, 2)

    sampled = F.grid_sample(
        heatmap, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return sampled.squeeze()


def score_pose_against_heatmap(
    log_heatmaps: dict[str, torch.Tensor],
    arm: Arm,
    camera: Camera,
) -> torch.Tensor:
    """Score all joint positions against their respective heatmaps."""
    coords = arm.get_coordinates()
    total = torch.tensor(0.0)
    for name in ["a", "b", "c"]:
        image_point = camera.world_to_image_torch(coords[name])
        x, y = image_point[0], image_point[1]
        total = total + sample_heatmap(log_heatmaps[name], x, y)
    return total


def score_position_change(arm0: Arm, arm1: Arm) -> torch.Tensor:
    """Compute shoulder position change penalty between consecutive frames.

    Only penalizes global shoulder movement. Elbow/wrist smoothness is
    handled by the rotation change penalties (ab_rotation, bc_rotation).
    """
    coords0 = arm0.get_coordinates()
    coords1 = arm1.get_coordinates()

    diff_a = coords1["a"] - coords0["a"]
    return torch.sqrt(torch.sum(diff_a**2) + EPSILON)


def score_ab_rotation_change(arm0: Arm, arm1: Arm) -> torch.Tensor:
    """Compute upper arm rotation change penalty."""
    origin = torch.zeros(3)

    unit_arm0 = Arm(origin, 1.0, arm0.a_b_polar, 1.0, torch.tensor(0.0))
    b0 = unit_arm0.get_coordinates()["b"]

    unit_arm1 = Arm(origin, 1.0, arm1.a_b_polar, 1.0, torch.tensor(0.0))
    b1 = unit_arm1.get_coordinates()["b"]

    diff = b1 - b0
    return torch.sqrt(torch.sum(diff**2) + EPSILON)


def score_bc_rotation_change(arm0: Arm, arm1: Arm) -> torch.Tensor:
    """Compute forearm rotation change penalty."""
    origin = torch.zeros(3)

    polar0 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), arm0.a_b_polar[2]])
    unit_arm0 = Arm(origin, 1.0, polar0, 1.0, arm0.b_c_theta)
    c0 = unit_arm0.get_coordinates()["c"]

    polar1 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), arm1.a_b_polar[2]])
    unit_arm1 = Arm(origin, 1.0, polar1, 1.0, arm1.b_c_theta)
    c1 = unit_arm1.get_coordinates()["c"]

    diff = c1 - c0
    return torch.sqrt(torch.sum(diff**2) + EPSILON)


def score(
    all_log_heatmaps: list[dict[str, torch.Tensor]],
    arms: list[Arm],
    cameras: list[Camera],
    config: OptimizationConfig,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """
    Compute total score across all frames (higher is better).

    Uses per-frame cameras (one camera per frame).
    """
    assert len(all_log_heatmaps) == len(arms)
    assert len(cameras) == len(arms)

    heatmap_total = torch.tensor(0.0)
    for i, arm in enumerate(arms):
        heatmap_total = heatmap_total + score_pose_against_heatmap(
            all_log_heatmaps[i], arm, cameras[i]
        )

    position_penalty = torch.tensor(0.0)
    ab_rotation_penalty = torch.tensor(0.0)
    bc_rotation_penalty = torch.tensor(0.0)

    for i in range(len(arms) - 1):
        arm0 = arms[i]
        arm1 = arms[i + 1]

        position_penalty = position_penalty + score_position_change(arm0, arm1)
        ab_rotation_penalty = ab_rotation_penalty + score_ab_rotation_change(arm0, arm1)
        bc_rotation_penalty = bc_rotation_penalty + score_bc_rotation_change(arm0, arm1)

    motion_penalty = (
        config.position_penalty_weight * position_penalty
        + config.ab_rotation_penalty_weight * ab_rotation_penalty
        + config.bc_rotation_penalty_weight * bc_rotation_penalty
    )

    total_score = heatmap_total - motion_penalty

    if return_components:
        components = {
            "heatmap": heatmap_total.item(),
            "position": position_penalty.item(),
            "ab_rotation": ab_rotation_penalty.item(),
            "bc_rotation": bc_rotation_penalty.item(),
            "weighted_motion": motion_penalty.item(),
        }
        return total_score, components

    return total_score
