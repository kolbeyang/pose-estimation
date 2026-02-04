import argparse
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from model.arm import Arm
from model.camera import Camera
from model.environment import Environment
from video import Video

logger = logging.getLogger(__name__)

EPSILON = 1e-10  # For log stability


@dataclass
class OptimizationConfig:
    """Configuration for pose optimization."""

    num_steps: int = 100
    learning_rate: float = 0.1
    position_init_noise: float = 0.5  # Std dev for jittering xyz coordinates
    angle_init_noise: float = 0.1  # Std dev for jittering angles (radians, ~6 degrees)
    position_penalty_weight: float = 1.0  # Weight for position changes
    ab_rotation_penalty_weight: float = 0.5  # Weight for upper arm rotation changes
    bc_rotation_penalty_weight: float = 0.3  # Weight for forearm rotation changes


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


@dataclass
class OptimizationResult:
    """Results from pose optimization."""

    pred_arms: list[Arm]
    pred_coords: list[dict[str, np.ndarray]]
    init_coords: list[dict[str, np.ndarray]]
    gt_arms: list[Arm]
    gt_coords: list[dict[str, torch.Tensor]]


def run_optimization(
    video: Video,
    config: OptimizationConfig,
) -> OptimizationResult:
    """
    Run pose optimization on all frames of a video.

    Args:
        video: Video object containing frames and ground truth
        config: Optimization hyperparameters

    Returns:
        OptimizationResult with predicted and ground truth data
    """
    num_frames = video.get_frame_count()
    a_b_length = video.a_b_length
    b_c_length = video.b_c_length
    camera = video.camera

    # Initialize parameters with jittered ground truth
    all_a_pos = []
    all_a_b_polar = []
    all_b_c_theta = []
    all_gt_coords = []
    all_init_coords = []

    logger.info("Initializing parameters from jittered ground truth...")
    for frame_idx in range(num_frames):
        gt_arm = video.get_frame_arm(frame_idx)
        gt_coords = gt_arm.get_coordinates()
        all_gt_coords.append(gt_coords)

        # Jitter ground truth for initialization
        a_pos = (
            gt_arm.a_pos.detach().clone().float()
            + torch.randn(3) * config.position_init_noise
        )
        a_b_polar = (
            gt_arm.a_b_polar.detach().clone().float()
            + torch.randn(3) * config.angle_init_noise
        )
        b_c_theta = (
            gt_arm.b_c_theta.detach().clone().float()
            + torch.randn(1).item() * config.angle_init_noise
        )

        # Store initial coordinates before enabling gradients
        init_arm = Arm(a_pos, a_b_length, a_b_polar, b_c_length, b_c_theta)
        all_init_coords.append(init_arm.get_coordinates_numpy())

        a_pos.requires_grad_(True)
        a_b_polar.requires_grad_(True)
        b_c_theta.requires_grad_(True)

        all_a_pos.append(a_pos)
        all_a_b_polar.append(a_b_polar)
        all_b_c_theta.append(b_c_theta)

    # Setup optimizer
    all_params = all_a_pos + all_a_b_polar + all_b_c_theta
    optimizer = torch.optim.Adam(all_params, lr=config.learning_rate)

    # Load heatmaps
    logger.info("Loading heatmaps...")
    all_log_heatmaps = [
        prepare_heatmaps(video.get_frame_heatmaps(i)) for i in range(num_frames)
    ]

    # Optimization loop
    logger.info("Starting batch optimization (%d steps)...", config.num_steps)
    logger.info(
        "Penalty weights: position=%.2f, ab_rotation=%.2f, bc_rotation=%.2f",
        config.position_penalty_weight,
        config.ab_rotation_penalty_weight,
        config.bc_rotation_penalty_weight,
    )

    for step in range(config.num_steps):
        optimizer.zero_grad()

        arms = [
            Arm(
                all_a_pos[i], a_b_length, all_a_b_polar[i], b_c_length, all_b_c_theta[i]
            )
            for i in range(num_frames)
        ]

        should_log = (step + 1) % 20 == 0
        result = score(all_log_heatmaps, arms, camera, config, return_components=should_log)

        if should_log:
            total_score, components = result
        else:
            total_score = result

        loss = -total_score
        loss.backward()
        optimizer.step()

        if should_log:
            logger.info(
                "Step %d: score=%.2f (heatmap=%.2f, pos=%.3f, ab=%.3f, bc=%.3f, weighted_motion=%.2f)",
                step + 1,
                total_score.item(),
                components["heatmap"],
                components["position"],
                components["ab_rotation"],
                components["bc_rotation"],
                components["weighted_motion"],
            )

    logger.info("Optimization complete.")

    # Build final predicted arms
    pred_arms = []
    pred_coords = []
    gt_arms = []

    with torch.no_grad():
        for i in range(num_frames):
            pred_arm = Arm(
                all_a_pos[i], a_b_length, all_a_b_polar[i], b_c_length, all_b_c_theta[i]
            )
            pred_arms.append(pred_arm)
            pred_coords.append(pred_arm.get_coordinates_numpy())
            gt_arms.append(video.get_frame_arm(i))

    return OptimizationResult(
        pred_arms=pred_arms,
        pred_coords=pred_coords,
        init_coords=all_init_coords,
        gt_arms=gt_arms,
        gt_coords=all_gt_coords,
    )


@dataclass
class EvaluationResult:
    """Evaluation metrics from pose optimization."""

    gt_total_score: float
    pred_total_score: float
    gt_per_frame_scores: list[float]
    pred_per_frame_scores: list[float]
    mpjpe: float  # Mean Per Joint Position Error
    mpjpe_per_frame: list[float]  # MPJPE for each frame


def evaluate_results(
    opt_result: OptimizationResult,
    video: Video,
    config: OptimizationConfig,
) -> EvaluationResult:
    """
    Evaluate optimization results against ground truth.

    Args:
        opt_result: Results from run_optimization
        video: Video object for heatmaps
        config: Config for score computation

    Returns:
        EvaluationResult with scores
    """
    num_frames = video.get_frame_count()
    camera = video.camera

    all_log_heatmaps = [
        prepare_heatmaps(video.get_frame_heatmaps(i)) for i in range(num_frames)
    ]

    gt_per_frame = []
    pred_per_frame = []
    mpjpe_per_frame = []

    with torch.no_grad():
        for i in range(num_frames):
            gt_score = score_pose_against_heatmap(
                all_log_heatmaps[i], opt_result.gt_arms[i], camera
            ).item()
            pred_score = score_pose_against_heatmap(
                all_log_heatmaps[i], opt_result.pred_arms[i], camera
            ).item()
            gt_per_frame.append(gt_score)
            pred_per_frame.append(pred_score)

            # Compute per-frame MPJPE
            gt_coords = opt_result.gt_arms[i].get_coordinates()
            pred_coords = opt_result.pred_arms[i].get_coordinates()
            frame_mpjpe = sum(
                torch.norm(pred_coords[j] - gt_coords[j]).item()
                for j in ["a", "b", "c"]
            ) / 3
            mpjpe_per_frame.append(frame_mpjpe)

        gt_total = score(all_log_heatmaps, opt_result.gt_arms, camera, config).item()
        pred_total = score(
            all_log_heatmaps, opt_result.pred_arms, camera, config
        ).item()

    mpjpe = sum(mpjpe_per_frame) / len(mpjpe_per_frame)

    return EvaluationResult(
        gt_total_score=gt_total,
        pred_total_score=pred_total,
        gt_per_frame_scores=gt_per_frame,
        pred_per_frame_scores=pred_per_frame,
        mpjpe=mpjpe,
        mpjpe_per_frame=mpjpe_per_frame,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Optimize arm parameters to fit heatmaps using PyTorch"
    )
    parser.add_argument("folder_path", help="Path to saved video folder")
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Visualize results after optimization",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Show matplotlib graph of scores over frames",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    # Load video
    video = Video.load(args.folder_path)
    logger.info("Loaded video from: %s", args.folder_path)
    logger.info("Frame count: %d", video.get_frame_count())
    logger.info(
        "Segment lengths: a_b=%.2f, b_c=%.2f", video.a_b_length, video.b_c_length
    )

    # Run optimization
    config = OptimizationConfig()
    result = run_optimization(video, config)

    # Evaluate
    eval_result = evaluate_results(result, video, config)

    # Log results
    logger.info("=" * 50)
    logger.info("Final results:")
    logger.info("Ground Truth score: %.4f", eval_result.gt_total_score)
    logger.info("Predicted score:    %.4f", eval_result.pred_total_score)
    logger.info("MPJPE:              %.4f", eval_result.mpjpe)

    # Graph with --graph flag
    if args.graph:
        import matplotlib.pyplot as plt

        frames = list(range(video.get_frame_count()))
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Heatmap scores
        ax1.plot(
            frames,
            eval_result.gt_per_frame_scores,
            color="green",
            label="Ground Truth",
            marker="o",
        )
        ax1.plot(
            frames,
            eval_result.pred_per_frame_scores,
            color="red",
            label="Predicted",
            marker="o",
        )
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Heatmap Score")
        ax1.set_title("Heatmap Scores per Frame")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: MPJPE
        ax2.plot(
            frames,
            eval_result.mpjpe_per_frame,
            color="red",
            label="MPJPE",
            marker="o",
        )
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("MPJPE (distance)")
        ax2.set_title("MPJPE per Frame")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Visualization with -v flag
    if args.visualize:
        import sys

        from vpython import rate

        from visualize import Visualizer

        env = Environment(cube_size=10.0)
        visualizer = Visualizer(
            env,
            [result.gt_coords[0], result.init_coords[0], result.pred_coords[0]],
            camera=video.camera,
            fps=2,
        )
        logger.info(
            "Visualization: green=ground truth, yellow=initialization, red=predicted"
        )
        logger.info("Animating through all frames...")

        try:
            while True:
                for i in range(video.get_frame_count()):
                    rate(visualizer.fps)
                    visualizer.update(
                        [
                            result.gt_coords[i],
                            result.init_coords[i],
                            result.pred_coords[i],
                        ]
                    )
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
