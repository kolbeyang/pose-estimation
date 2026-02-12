from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from model.arm import Arm
from model.camera import Camera

EPSILON = 1e-10  # For log stability


@dataclass
class OptimizationConfig:
    """Configuration for pose optimization.

    Default values optimized via high-noise SGLD experiment (2026-02-09):
    - MPJPE: 0.379 (single-run with noise_temp=0.1, noise_start_step=90)
    - noise_start_step=90 gives 90% deterministic SGD warmup, then SGLD exploration
    - Tested 8 noise_start_step values, 6 LR/step combos, 3 ensemble sizes
    """

    num_steps: int = 100
    learning_rate: float = 0.5
    lr_min: float = 1e-5  # Cosine annealing floor
    noise_temperature: float = 0.1  # SGLD noise (>= 0.1 for meaningful exploration)
    num_runs: int = 5  # Number of sampling runs
    position_init_noise: float = 0.5  # Std dev for jittering xyz coordinates
    angle_init_noise: float = 0.1  # Std dev for jittering angles (radians, ~6 degrees)
    length_init_noise: float = 0  # Std dev for jittering segment lengths
    position_penalty_weight: float = 0.5  # Weight for position changes
    ab_rotation_penalty_weight: float = 0.3  # Weight for upper arm rotation changes
    bc_rotation_penalty_weight: float = 0.4  # Weight for forearm rotation changes
    noise_start_step: int = (
        100  # Step at which SGLD noise injection begins (90% warmup)
    )


@dataclass
class OptimizationResult:
    """Results from pose optimization."""

    pred_arms: list[Arm]
    pred_coords: list[dict[str, np.ndarray]]
    init_coords: list[dict[str, np.ndarray]]
    gt_arms: list[Arm]
    gt_coords: list[dict[str, torch.Tensor]]
    a_b_length_history: list[float]
    b_c_length_history: list[float]
    mid_frame_idx: int
    mid_a_pos_history: list[list[float]]
    mid_a_b_polar_history: list[list[float]]
    mid_b_c_theta_history: list[float]


@dataclass
class EvaluationResult:
    """Evaluation metrics from pose optimization."""

    gt_total_score: float
    pred_total_score: float
    gt_per_frame_scores: list[float]
    pred_per_frame_scores: list[float]
    mpjpe: float  # Mean Per Joint Position Error
    mpjpe_per_frame: list[float]  # MPJPE for each frame


def prepare_heatmaps(heatmaps: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert heatmaps to log-normalized torch tensors."""
    result = {}
    for name, heatmap in heatmaps.items():
        # Normalize to [0, 1], add epsilon, take log
        normalized = heatmap.astype(np.float32) / 255.0
        log_heatmap = np.log(normalized + EPSILON)
        # Shape: (1, 1, H, W) for grid_sample
        result[name] = torch.tensor(log_heatmap).unsqueeze(0).unsqueeze(0)
    return result


def sample_heatmap(
    heatmap: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """
    Sample heatmap at (x, y) using differentiable bilinear interpolation.

    Args:
        heatmap: Shape (1, 1, H, W), log-normalized
        x, y: Pixel coordinates (can be fractional)

    Returns:
        Sampled value (scalar tensor)
    """
    H, W = heatmap.shape[2], heatmap.shape[3]

    # Normalize to [-1, 1] for grid_sample
    # x=0 -> -1, x=W-1 -> 1
    x_norm = 2.0 * x / (W - 1) - 1.0
    y_norm = 2.0 * y / (H - 1) - 1.0

    # grid_sample expects grid of shape (N, H_out, W_out, 2)
    # For single point: (1, 1, 1, 2)
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, 1, 1, 2)

    # Sample with bilinear interpolation, border padding for out-of-bounds
    sampled = F.grid_sample(
        heatmap, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    return sampled.squeeze()


def score_pose_against_heatmap(
    log_heatmaps: dict[str, torch.Tensor],
    arm: Arm,
    camera: Camera,
) -> torch.Tensor:
    """
    Compute differentiable score for a single pose against heatmaps (higher is better).

    Args:
        log_heatmaps: Log-normalized heatmaps for each joint
        arm: Arm object with current parameters
        camera: Camera object for projection
    """
    coords = arm.get_coordinates()

    total = torch.tensor(0.0)
    for name in ["a", "b", "c"]:
        # Project 3D -> 2D
        image_point = camera.world_to_image_torch(coords[name])
        x, y = image_point[0], image_point[1]

        # Sample from log-heatmap (already log-normalized)
        total = total + sample_heatmap(log_heatmaps[name], x, y)

    return total


def score_position_change(arm0: Arm, arm1: Arm) -> torch.Tensor:
    """Compute position change penalty between two consecutive arm poses."""
    coords0 = arm0.get_coordinates()
    coords1 = arm1.get_coordinates()

    total = torch.tensor(0.0)
    for name in ["a", "b", "c"]:
        diff = coords1[name] - coords0[name]
        total = total + torch.sqrt(torch.sum(diff**2) + EPSILON)
    return total


def score_ab_rotation_change(arm0: Arm, arm1: Arm) -> torch.Tensor:
    """
    Compute upper arm (AB segment) rotation change penalty.

    Creates unit arms at origin with same a_b_polar rotation and measures
    Euclidean distance between their B points.
    """
    origin = torch.zeros(3)

    # Unit arm with arm0's rotation
    unit_arm0 = Arm(origin, 1.0, arm0.a_b_polar, 1.0, torch.tensor(0.0))
    b0 = unit_arm0.get_coordinates()["b"]

    # Unit arm with arm1's rotation
    unit_arm1 = Arm(origin, 1.0, arm1.a_b_polar, 1.0, torch.tensor(0.0))
    b1 = unit_arm1.get_coordinates()["b"]

    # Euclidean distance between B points
    diff = b1 - b0
    return torch.sqrt(torch.sum(diff**2) + EPSILON)


def score_bc_rotation_change(arm0: Arm, arm1: Arm) -> torch.Tensor:
    """
    Compute forearm (BC segment) rotation change penalty.

    Creates unit arms at origin with zeroed azimuth/elevation but same roll and theta,
    then measures Euclidean distance between their C points.
    """
    origin = torch.zeros(3)

    # Unit arm with arm0's roll and theta (azimuth=0, elevation=0)
    polar0 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), arm0.a_b_polar[2]])
    unit_arm0 = Arm(origin, 1.0, polar0, 1.0, arm0.b_c_theta)
    c0 = unit_arm0.get_coordinates()["c"]

    # Unit arm with arm1's roll and theta (azimuth=0, elevation=0)
    polar1 = torch.stack([torch.tensor(0.0), torch.tensor(0.0), arm1.a_b_polar[2]])
    unit_arm1 = Arm(origin, 1.0, polar1, 1.0, arm1.b_c_theta)
    c1 = unit_arm1.get_coordinates()["c"]

    # Euclidean distance between C points
    diff = c1 - c0
    return torch.sqrt(torch.sum(diff**2) + EPSILON)


def score(
    all_log_heatmaps: list[dict[str, torch.Tensor]],
    arms: list[Arm],
    camera: Camera,
    config: OptimizationConfig,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """
    Compute total score across all frames (higher is better).

    Args:
        return_components: If True, also return individual score components for logging.

    Returns:
        If return_components is False: Combined score tensor
        If return_components is True: Tuple of (score tensor, components dict)
        (Use loss = -score for optimization)
    """
    assert len(all_log_heatmaps) == len(
        arms
    ), f"Number of heatmaps ({len(all_log_heatmaps)}) must match number of arms ({len(arms)})"

    # Sum heatmap scores across all frames
    heatmap_total = torch.tensor(0.0)
    for i, arm in enumerate(arms):
        heatmap_total = heatmap_total + score_pose_against_heatmap(
            all_log_heatmaps[i], arm, camera
        )

    # Motion penalties between consecutive frames
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
