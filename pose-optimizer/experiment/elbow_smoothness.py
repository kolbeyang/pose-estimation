"""Elbow Position Smoothness: plot X/Y/Z trajectories for a single joint.

Plots 15 lines: (GT, MB Raw, MB Opt, MP Raw, MP Opt) x (X, Y, Z).
5 trajectories differentiated by color, 3 dimensions by line style.

Usage:
    uv run experiment/elbow_smoothness.py <mb_trajectories.json> <mp_trajectories.json> [--joint JOINT] [--fps FPS]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Consistent color scheme
COLOR_GT = "#4285F4"          # Blue
COLOR_MB_RAW = "#FFCC80"      # Light Orange
COLOR_MB_OPT = "#FF9800"      # Orange
COLOR_MP_RAW = "#EF9A9A"      # Light Red
COLOR_MP_OPT = "#F44336"      # Red

# Line styles for dimensions: solid, wide dash, narrow dash
STYLE_X = {"linestyle": "-", "linewidth": 1.5}
STYLE_Y = {"linestyle": (0, (8, 3)), "linewidth": 1.5}  # wide dash
STYLE_Z = {"linestyle": (0, (3, 2)), "linewidth": 1.5}  # narrow dash
DIM_STYLES = [STYLE_X, STYLE_Y, STYLE_Z]
DIM_LABELS = ["X", "Y", "Z"]


def load_trajectories(path: str) -> dict:
    """Load a trajectories.json file."""
    with open(path) as f:
        return json.load(f)


def find_joint_index(joint_names: list[str], joint_name: str) -> int:
    """Find joint index by name (case-insensitive partial match)."""
    lower = joint_name.lower()
    for i, name in enumerate(joint_names):
        if name.lower() == lower:
            return i
    for i, name in enumerate(joint_names):
        if lower in name.lower():
            return i
    raise ValueError(f"Joint '{joint_name}' not found in {joint_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Elbow position smoothness graph")
    parser.add_argument("mb_trajectories", help="Path to MotionBERT trajectories.json")
    parser.add_argument("mp_trajectories", help="Path to MediaPipe trajectories.json")
    parser.add_argument("--joint", default="RElbow", help="Joint name to plot (default: RElbow)")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second (default: 10)")
    parser.add_argument("--output", default=None, help="Output directory (default: output/)")
    args = parser.parse_args()

    mb_data = load_trajectories(args.mb_trajectories)
    mp_data = load_trajectories(args.mp_trajectories)

    joint_names = mb_data["joint_names"]
    joint_idx = find_joint_index(joint_names, args.joint)
    joint_name = joint_names[joint_idx]
    print(f"Plotting joint: {joint_name} (index {joint_idx})")

    # Extract trajectories for this joint: (N, 3) in meters -> cm
    gt_frames = mb_data["ground_truth"]
    n_frames = len(gt_frames)
    time_axis = np.arange(n_frames) / args.fps

    def extract_joint(frames_list: list, idx: int) -> np.ndarray:
        """Extract a single joint's trajectory, handling None frames."""
        result = np.full((len(frames_list), 3), np.nan)
        for i, frame in enumerate(frames_list):
            if frame is not None:
                result[i] = np.array(frame[idx]) * 100  # meters -> cm
        return result

    gt = extract_joint(gt_frames, joint_idx)
    mb_raw = extract_joint(mb_data["raw_prediction"], joint_idx)
    mb_opt = extract_joint(mb_data["optimized_prediction"], joint_idx)
    mp_raw = extract_joint(mp_data["raw_prediction"], joint_idx)
    mp_opt = extract_joint(mp_data["optimized_prediction"], joint_idx)

    # Build figure
    fig, ax = plt.subplots(figsize=(14, 7))

    trajectories = [
        ("Ground Truth", COLOR_GT, gt),
        ("MotionBERT Raw", COLOR_MB_RAW, mb_raw),
        ("MotionBERT Opt", COLOR_MB_OPT, mb_opt),
        ("MediaPipe Raw", COLOR_MP_RAW, mp_raw),
        ("MediaPipe Opt", COLOR_MP_OPT, mp_opt),
    ]

    for traj_name, color, data in trajectories:
        for dim in range(3):
            style = DIM_STYLES[dim]
            label = f"{traj_name} {DIM_LABELS[dim]}"
            ax.plot(time_axis, data[:, dim], color=color, label=label,
                    alpha=0.85, **style)

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Position (cm)", fontsize=11)
    ax.set_title(f"{joint_name} Position Smoothness", fontsize=13)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)

    output_dir = args.output or "output"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"elbow_smoothness_{joint_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
