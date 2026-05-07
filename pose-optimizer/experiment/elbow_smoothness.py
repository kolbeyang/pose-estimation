"""Elbow Position Smoothness: plot X/Y/Z trajectories for a single joint.

Plots 15 lines: (GT, MB Raw, MB Opt, MP Raw, MP Opt) x (X, Y, Z).
5 trajectories differentiated by color, 3 dimensions by line style.

Also generates single-axis graphs (X only, Y only, Z only).

Usage:
    uv run experiment/elbow_smoothness.py <mb_trajectories.json> <mp_trajectories.json> [--joint JOINT] [--fps FPS]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "figure.titlesize": 28,
})


# Consistent color scheme
COLOR_GT = "#5871CA"          # Blue
COLOR_MB_RAW = "#FFA2DB"      # Light Pink
COLOR_MB_OPT = "#FF389C"      # Pink
COLOR_MP_RAW = "#FFB199"      # Light Orange
COLOR_MP_OPT = "#FF7A21"      # Orange

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
    parser.add_argument("--sample-name", default=None, help="Sample name to include in title")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Limit to first N frames (default: all)")
    parser.add_argument("--duration-seconds", type=float, default=None,
                        help="Limit to first D seconds (overrides --num-frames)")
    args = parser.parse_args()

    mb_data = load_trajectories(args.mb_trajectories)
    mp_data = load_trajectories(args.mp_trajectories)

    joint_names = mb_data["joint_names"]
    joint_idx = find_joint_index(joint_names, args.joint)
    joint_name = joint_names[joint_idx]
    print(f"Plotting joint: {joint_name} (index {joint_idx})")

    # Determine slice for first-N-frames limit.
    # Default: restrict to first 10 seconds (0-100 frames at 10 fps, inclusive).
    limit = None
    if args.duration_seconds is not None:
        limit = int(round(args.duration_seconds * args.fps))
    elif args.num_frames is not None:
        limit = args.num_frames
    else:
        limit = int(round(10.0 * args.fps)) + 1  # inclusive of frame 100

    # Extract trajectories for this joint: (N, 3) in meters -> cm
    gt_frames = mb_data["ground_truth"]
    if limit is not None:
        gt_frames = gt_frames[:limit]
    n_frames = len(gt_frames)
    time_axis = np.arange(n_frames) / args.fps

    def extract_joint(frames_list: list, idx: int) -> np.ndarray:
        """Extract a single joint's trajectory, handling None frames."""
        result = np.full((len(frames_list), 3), np.nan)
        for i, frame in enumerate(frames_list):
            if frame is not None:
                result[i] = np.array(frame[idx]) * 100  # meters -> cm
        return result

    def maybe_slice(frames_list):
        return frames_list[:limit] if limit is not None else frames_list

    gt = extract_joint(gt_frames, joint_idx)
    mb_raw = extract_joint(maybe_slice(mb_data["raw_prediction"]), joint_idx)
    mb_opt = extract_joint(maybe_slice(mb_data["optimized_prediction"]), joint_idx)
    mp_raw = extract_joint(maybe_slice(mp_data["raw_prediction"]), joint_idx)
    mp_opt = extract_joint(maybe_slice(mp_data["optimized_prediction"]), joint_idx)

    trajectories = [
        ("Ground Truth", COLOR_GT, gt),
        ("MotionBERT Raw", COLOR_MB_RAW, mb_raw),
        ("MotionBERT Opt", COLOR_MB_OPT, mb_opt),
        ("MediaPipe Raw", COLOR_MP_RAW, mp_raw),
        ("MediaPipe Opt", COLOR_MP_OPT, mp_opt),
    ]

    output_dir = args.output or "output"
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{args.sample_name}" if args.sample_name else ""

    # --- Combined graph (X/Y/Z together, with line styles for dims) ---
    fig, ax = plt.subplots(figsize=(14, 7))

    for traj_name, color, data in trajectories:
        for dim in range(3):
            style = DIM_STYLES[dim]
            label = f"{traj_name} {DIM_LABELS[dim]}"
            ax.plot(time_axis, data[:, dim], color=color, label=label,
                    alpha=0.85, **style)

    ax.set_xlabel("Time (seconds)", fontsize=22)
    ax.set_ylabel("Position (cm)", fontsize=22)
    title = f"{joint_name} Position Over Time"
    if args.sample_name:
        title = f"{title} — {args.sample_name}"
    ax.set_title(title, fontsize=26)
    ax.legend(fontsize=14, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)

    out_path = os.path.join(output_dir, f"position_{joint_name}{suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # --- Per-axis graphs (X only, Y only, Z only), solid lines ---
    for dim in range(3):
        dim_label = DIM_LABELS[dim]
        fig, ax = plt.subplots(figsize=(14, 7))
        for traj_name, color, data in trajectories:
            # Solid lines only for single-axis graphs
            ax.plot(time_axis, data[:, dim], color=color, label=traj_name,
                    alpha=0.9, linestyle="-", linewidth=2)

        ax.set_xlabel("Time (seconds)", fontsize=22)
        ax.set_ylabel(f"{dim_label} Position (cm)", fontsize=22)
        title = f"{joint_name} {dim_label} Position Over Time"
        if args.sample_name:
            title = f"{title} — {args.sample_name}"
        ax.set_title(title, fontsize=26)
        ax.legend(fontsize=18, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)

        out_path_axis = os.path.join(output_dir, f"position_{joint_name}_{dim_label}{suffix}.png")
        fig.savefig(out_path_axis, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path_axis}")


if __name__ == "__main__":
    main()
