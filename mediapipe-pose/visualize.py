"""Standalone 3D visualisation: load a predictions JSON and animate.

Usage:
    python visualize.py results/predictions/171204_pose1_sample_0.json

Shows MediaPipe (green) vs Optimised (red) skeletons side-by-side,
with optional Ground Truth (blue).
"""

import argparse
import json

import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from skeleton import BONES


def _draw_camera(ax, radius=0.15):
    """Draw the camera as a transparent sphere at the origin."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 15)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="gray", alpha=0.2)
    ax.scatter([0], [0], [0], c="black", s=30, marker="^", label="Camera")


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

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Determine axis limits from all data
    all_positions = [np.zeros((1, 3))]
    for frame in frames:
        all_positions.append(np.array(frame["mediapipe_3d"]))
        all_positions.append(np.array(frame["optimized_3d"]))
        if frame.get("ground_truth_3d") is not None:
            all_positions.append(np.array(frame["ground_truth_3d"]))
    all_pts = np.concatenate(all_positions, axis=0)
    center = all_pts.mean(axis=0)
    span = max(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2 * 1.2

    # Scroll to zoom
    zoom = {"span": span}

    def _on_scroll(event):
        factor = 0.8 if event.button == "up" else 1.25
        zoom["span"] *= factor

    fig.canvas.mpl_connect("scroll_event", _on_scroll)

    print(f"Loaded {n_frames} frames. Scroll to zoom, drag to rotate, close to exit.")
    plt.ion()

    frame_idx = 0
    while plt.fignum_exists(fig.number):
        frame = frames[frame_idx]
        ax.cla()

        mp = np.array(frame["mediapipe_3d"])
        opt = np.array(frame["optimized_3d"])
        _draw_camera(ax)
        _draw_skeleton(ax, mp, "green", "MediaPipe")
        _draw_skeleton(ax, opt, "red", "Optimised")

        if frame.get("ground_truth_3d") is not None:
            gt = np.array(frame["ground_truth_3d"])
            _draw_skeleton(ax, gt, "blue", "Ground Truth", alpha=0.5)

        s = zoom["span"]
        ax.set_xlim(center[0] - s, center[0] + s)
        ax.set_ylim(center[1] - s, center[1] + s)
        ax.set_zlim(center[2] - s, center[2] + s)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{title}  —  Frame {frame_idx}/{n_frames}")
        ax.legend(fontsize=8, loc="upper left")

        plt.draw()
        plt.pause(1.0 / fps)
        frame_idx = (frame_idx + 1) % n_frames

    plt.ioff()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise 3D pose predictions")
    parser.add_argument("json_path", help="Path to predictions JSON file")
    parser.add_argument("--fps", type=float, default=5.0, help="Playback FPS")
    args = parser.parse_args()
    visualize_prediction_file(args.json_path, fps=args.fps)
