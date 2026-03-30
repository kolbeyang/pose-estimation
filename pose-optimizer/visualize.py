"""3D visualization from trajectories.json.

Loads trajectory data and renders an animated matplotlib 3D plot showing
ground truth (blue), raw predictions (green), and optimized predictions (red).

Usage:
    uv run python visualize.py path/to/trajectories.json
"""

import json
import sys

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation
import numpy as np

from skeleton import BONES, JOINT_NAMES, NUM_JOINTS


def load_trajectories(path: str) -> dict:
    """Load trajectory data from JSON file."""
    with open(path) as f:
        return json.load(f)


def _draw_skeleton_3d(
    ax: plt.Axes,
    positions: np.ndarray,
    color: str,
    label: str,
    alpha: float = 0.8,
) -> None:
    """Draw a single skeleton on a 3D axis."""
    for parent, child in BONES:
        ax.plot3D(
            [positions[parent, 0], positions[child, 0]],
            [positions[parent, 1], positions[child, 1]],
            [positions[parent, 2], positions[child, 2]],
            color=color, alpha=alpha, linewidth=2,
        )
    ax.scatter3D(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color=color, alpha=alpha, s=20, label=label,
    )


def animate_trajectories(data: dict) -> None:
    """Animate 3D skeletons from trajectory data.

    Args:
        data: Dict with keys 'joint_names', 'ground_truth', 'raw_prediction',
              'optimized_prediction'. Each trajectory is (N, K, 3).
    """
    gt = np.array(data.get("ground_truth", []))
    raw = np.array(data.get("raw_prediction", []))
    opt = np.array(data.get("optimized_prediction", []))

    n_frames = max(len(gt), len(raw), len(opt))
    if n_frames == 0:
        print("No trajectory data to visualize.")
        return

    # Determine axis limits from all available data
    all_pts = []
    if len(gt) > 0:
        all_pts.append(gt.reshape(-1, 3))
    if len(raw) > 0:
        all_pts.append(raw.reshape(-1, 3))
    if len(opt) > 0:
        all_pts.append(opt.reshape(-1, 3))
    all_pts = np.concatenate(all_pts, axis=0)

    center = all_pts.mean(axis=0)
    max_range = np.max(np.abs(all_pts - center)) * 1.2

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int) -> None:
        ax.cla()
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Frame {frame_idx}/{n_frames - 1}")

        if len(gt) > frame_idx:
            _draw_skeleton_3d(ax, gt[frame_idx], "blue", "Ground Truth")
        if len(raw) > frame_idx:
            _draw_skeleton_3d(ax, raw[frame_idx], "green", "Raw Prediction")
        if len(opt) > frame_idx:
            _draw_skeleton_3d(ax, opt[frame_idx], "red", "Optimized")

        ax.legend(loc="upper right", fontsize=8)

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=200, repeat=True,
    )
    plt.show()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python visualize.py <trajectories.json>")
        sys.exit(1)

    data = load_trajectories(sys.argv[1])
    animate_trajectories(data)


if __name__ == "__main__":
    main()
