"""Standalone 3D visualisation: load a predictions JSON and animate.

Usage:
    python visualize.py results/predictions/171204_pose1_sample_0.json

Shows MediaPipe (green) vs Optimised (red) skeletons side-by-side,
with optional Ground Truth (blue).
"""

import argparse
import json
import sys
import time

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from skeleton import JOINT_NAMES, BONES, NUM_JOINTS, BODY_GROUPS, GROUP_COLORS_RGB


def _draw_skeleton(ax, positions, color, label, alpha=0.8):
    """Draw joints + bones for one skeleton."""
    ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=[color], s=20, alpha=alpha, label=label,
    )
    for parent, child in BONES:
        ax.plot(
            [positions[parent, 0], positions[child, 0]],
            [positions[parent, 1], positions[child, 1]],
            [positions[parent, 2], positions[child, 2]],
            color=color, alpha=alpha, linewidth=1.5,
        )


def visualize_prediction_file(json_path: str, fps: float = 5.0):
    """Animate a predictions JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    frames = data["frames"]
    title = data.get("sequence", json_path)
    n_frames = len(frames)

    print(f"Loaded {n_frames} frames from {json_path}")
    print(f"Sequence: {title}")
    print("Controls: close window to exit")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Determine axis limits from all data
    all_positions = []
    for frame in frames:
        all_positions.append(np.array(frame["mediapipe_3d"]))
        all_positions.append(np.array(frame["optimized_3d"]))
        if frame.get("ground_truth_3d") is not None:
            all_positions.append(np.array(frame["ground_truth_3d"]))
    all_pts = np.concatenate(all_positions, axis=0)
    center = all_pts.mean(axis=0)
    span = max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2 * 1.2

    plt.ion()

    frame_idx = 0
    while plt.fignum_exists(fig.number):
        frame = frames[frame_idx]
        ax.cla()

        mp = np.array(frame["mediapipe_3d"])
        opt = np.array(frame["optimized_3d"])
        _draw_skeleton(ax, mp, "green", "MediaPipe")
        _draw_skeleton(ax, opt, "red", "Optimised")

        if frame.get("ground_truth_3d") is not None:
            gt = np.array(frame["ground_truth_3d"])
            _draw_skeleton(ax, gt, "blue", "Ground Truth", alpha=0.5)

        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title}  —  Frame {frame_idx}/{n_frames}")
        ax.legend(fontsize=8, loc="upper left")

        plt.draw()
        plt.pause(1.0 / fps)
        frame_idx = (frame_idx + 1) % n_frames

    plt.ioff()


def main():
    parser = argparse.ArgumentParser(description="Visualise 3D pose predictions")
    parser.add_argument("json_path", help="Path to predictions JSON file")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback FPS")
    args = parser.parse_args()

    visualize_prediction_file(args.json_path, fps=args.fps)


if __name__ == "__main__":
    main()
