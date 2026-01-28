import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model.arm import Arm
from model.environment import Environment
from video import Video

# Optimization hyperparameters
NUM_STEPS = 100
LEARNING_RATE = 0.01
EPSILON = 1e-10  # For log stability


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


def project_point_torch(
    point: torch.Tensor,
    camera_position: torch.Tensor,
    camera_rotation: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
) -> torch.Tensor:
    """Project a 3D world point to 2D image coordinates (differentiable)."""
    # Transform to camera coordinates
    point_camera = camera_rotation @ (point - camera_position)

    # Project to image coordinates
    point_homogeneous = intrinsic_matrix @ point_camera
    image_point = point_homogeneous[:2] / point_homogeneous[2]

    return image_point


def score(
    log_heatmaps: dict[str, torch.Tensor],
    a_pos: torch.Tensor,
    a_b_polar: torch.Tensor,
    b_c_theta: torch.Tensor,
    a_b_length: float,
    b_c_length: float,
    camera_position: torch.Tensor,
    camera_rotation: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute differentiable score (higher is better)."""
    arm = Arm(a_pos, a_b_length, a_b_polar, b_c_length, b_c_theta)
    coords = arm.get_coordinates()

    total = torch.tensor(0.0)
    for name in ["a", "b", "c"]:
        # Project 3D -> 2D
        image_point = project_point_torch(
            coords[name], camera_position, camera_rotation, intrinsic_matrix
        )
        x, y = image_point[0], image_point[1]

        # Sample from log-heatmap (already log-normalized)
        total = total + sample_heatmap(log_heatmaps[name], x, y)

    return total


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

    # Load video data
    video = Video.load(args.folder_path)
    num_frames = video.get_frame_count()
    print(f"Loaded video from: {args.folder_path}")
    print(f"Frame count: {num_frames}")
    print(f"Segment lengths: a_b={video.a_b_length}, b_c={video.b_c_length}")

    # Fixed lengths from video
    a_b_length = video.a_b_length
    b_c_length = video.b_c_length

    # Camera parameters as tensors
    camera_position = torch.tensor(video.camera_position, dtype=torch.float32)
    camera_rotation = torch.tensor(video.camera_rotation, dtype=torch.float32)
    intrinsic_matrix = torch.tensor(video.K, dtype=torch.float32)

    # Storage for all frame results (for visualization and graphing)
    all_gt_coords = []
    all_pred_coords = []
    all_gt_scores = []
    all_pred_scores = []

    # Current optimized parameters (will be updated each frame)
    current_a_pos = None
    current_a_b_polar = None
    current_b_c_theta = None

    print("\nStarting multi-frame optimization...")

    for frame_idx in range(num_frames):
        print(f"\n{'='*50}")
        print(f"Frame {frame_idx}:")

        # Load heatmaps for this frame
        heatmaps = video.get_frame_heatmaps(frame_idx)
        log_heatmaps = prepare_heatmaps(heatmaps)

        # Load ground truth for this frame
        gt_arm = video.get_frame_arm(frame_idx)
        gt_coords = gt_arm.get_coordinates()
        all_gt_coords.append(gt_coords)

        # Initialize parameters
        if frame_idx == 0:
            # Frame 0: initialize from ground truth
            a_pos = torch.tensor(gt_arm.a_pos, dtype=torch.float32, requires_grad=True)
            a_b_polar = torch.tensor(
                list(gt_arm.a_b_polar), dtype=torch.float32, requires_grad=True
            )
            b_c_theta = torch.tensor(
                gt_arm.b_c_theta, dtype=torch.float32, requires_grad=True
            )
        else:
            # Frame 1+: initialize from previous frame's optimized params
            a_pos = torch.tensor(
                current_a_pos.detach().numpy(), dtype=torch.float32, requires_grad=True
            )
            a_b_polar = torch.tensor(
                current_a_b_polar.detach().numpy(),
                dtype=torch.float32,
                requires_grad=True,
            )
            b_c_theta = torch.tensor(
                current_b_c_theta.detach().item(),
                dtype=torch.float32,
                requires_grad=True,
            )

        # Optimizer for this frame
        optimizer = torch.optim.Adam([a_pos, a_b_polar, b_c_theta], lr=LEARNING_RATE)

        # Run optimization
        for step in range(NUM_STEPS):
            optimizer.zero_grad()

            log_likelihood = score(
                log_heatmaps,
                a_pos,
                a_b_polar,
                b_c_theta,
                a_b_length,
                b_c_length,
                camera_position,
                camera_rotation,
                intrinsic_matrix,
            )

            loss = -log_likelihood
            loss.backward()
            optimizer.step()

        # Store optimized params for next frame
        current_a_pos = a_pos
        current_a_b_polar = a_b_polar
        current_b_c_theta = b_c_theta

        # Get final predicted coordinates and compute scores
        with torch.no_grad():
            pred_arm = Arm(a_pos, a_b_length, a_b_polar, b_c_length, b_c_theta)
            pred_coords_np = pred_arm.get_coordinates_numpy()

            gt_a_pos = torch.tensor(gt_arm.a_pos, dtype=torch.float32)
            gt_a_b_polar = torch.tensor(list(gt_arm.a_b_polar), dtype=torch.float32)
            gt_b_c_theta = torch.tensor(gt_arm.b_c_theta, dtype=torch.float32)

            gt_score = score(
                log_heatmaps,
                gt_a_pos,
                gt_a_b_polar,
                gt_b_c_theta,
                a_b_length,
                b_c_length,
                camera_position,
                camera_rotation,
                intrinsic_matrix,
            ).item()

            pred_score = score(
                log_heatmaps,
                a_pos,
                a_b_polar,
                b_c_theta,
                a_b_length,
                b_c_length,
                camera_position,
                camera_rotation,
                intrinsic_matrix,
            ).item()

        all_pred_coords.append(pred_coords_np)
        all_gt_scores.append(gt_score)
        all_pred_scores.append(pred_score)

        # Print results
        print(
            f"  Ground truth coords:  A={gt_coords['a']} B={gt_coords['b']} C={gt_coords['c']}"
        )
        print(
            f"  Predicted coords:     A={pred_coords_np['a']} B={pred_coords_np['b']} C={pred_coords_np['c']}"
        )
        print(f"  Ground truth score:   {gt_score:.4f}")
        print(f"  Predicted score:      {pred_score:.4f}")

    print(f"\n{'='*50}")
    print("Optimization complete.")

    # Graph with --graph flag
    if args.graph:
        import matplotlib.pyplot as plt

        frames = list(range(num_frames))
        plt.figure(figsize=(10, 6))
        plt.plot(frames, all_gt_scores, color="green", label="Ground truth", marker="o")
        plt.plot(frames, all_pred_scores, color="red", label="Predicted", marker="o")
        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.title("Ground Truth vs Predicted Scores")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # Visualization with -v flag
    if args.visualize:
        from vpython import rate
        from visualize import Visualizer

        env = Environment(cube_size=10.0)
        visualizer = Visualizer(
            env, [all_gt_coords[0], all_pred_coords[0]], camera=video, fps=6
        )
        print("\nVisualization: green=ground truth, red=predicted")
        print("Animating through all frames...")

        # Animation loop
        try:
            while True:
                for i in range(num_frames):
                    rate(visualizer.fps)
                    visualizer.update([all_gt_coords[i], all_pred_coords[i]])
        except KeyboardInterrupt:
            import os

            os._exit(0)


if __name__ == "__main__":
    main()
