"""Bone Length Variation: plot forearm (wrist-to-elbow) length over time.

Plots 5 lines: GT, MB Raw, MB Opt, MP Raw, MP Opt.
Optimized lines should be perfectly horizontal (shared bone length param).

Usage:
    uv run experiment/bone_length_variation.py <mb_trajectories.json> <mp_trajectories.json> [--fps FPS]
"""

import argparse
import json
import os

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


def load_trajectories(path: str) -> dict:
    """Load a trajectories.json file."""
    with open(path) as f:
        return json.load(f)


def compute_bone_lengths(
    frames: list,
    parent_idx: int,
    child_idx: int,
) -> np.ndarray:
    """Compute bone length between two joints across all frames.

    Args:
        frames: List of (J, 3) frames or None.
        parent_idx: Parent joint index.
        child_idx: Child joint index.

    Returns:
        (N,) array of bone lengths in cm.
    """
    result = np.full(len(frames), np.nan)
    for i, frame in enumerate(frames):
        if frame is not None:
            p = np.array(frame[parent_idx])
            c = np.array(frame[child_idx])
            result[i] = np.linalg.norm(c - p) * 100  # meters -> cm
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Bone length variation graph")
    parser.add_argument("mb_trajectories", help="Path to MotionBERT trajectories.json")
    parser.add_argument("mp_trajectories", help="Path to MediaPipe trajectories.json")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second (default: 10)")
    parser.add_argument("--output", default=None, help="Output directory (default: output/)")
    parser.add_argument("--side", default="R", choices=["L", "R"], help="Which arm (default: R)")
    args = parser.parse_args()

    mb_data = load_trajectories(args.mb_trajectories)
    mp_data = load_trajectories(args.mp_trajectories)

    joint_names = mb_data["joint_names"]

    # Find elbow and wrist indices
    side = args.side
    elbow_name = f"{side}Elbow"
    wrist_name = f"{side}Wrist"
    elbow_idx = joint_names.index(elbow_name)
    wrist_idx = joint_names.index(wrist_name)
    bone_label = f"{side} Forearm ({wrist_name} to {elbow_name})"
    print(f"Plotting bone: {bone_label}")
    print(f"  Elbow: index {elbow_idx}, Wrist: index {wrist_idx}")

    n_frames = len(mb_data["raw_prediction"])
    time_axis = np.arange(n_frames) / args.fps

    # Compute bone lengths for each trajectory
    gt_bl = compute_bone_lengths(mb_data["ground_truth"], elbow_idx, wrist_idx)
    mb_raw_bl = compute_bone_lengths(mb_data["raw_prediction"], elbow_idx, wrist_idx)
    mb_opt_bl = compute_bone_lengths(mb_data["optimized_prediction"], elbow_idx, wrist_idx)
    mp_raw_bl = compute_bone_lengths(mp_data["raw_prediction"], elbow_idx, wrist_idx)
    mp_opt_bl = compute_bone_lengths(mp_data["optimized_prediction"], elbow_idx, wrist_idx)

    # Build figure
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time_axis, gt_bl, color=COLOR_GT, linewidth=2, label="Ground Truth", alpha=0.9)
    ax.plot(time_axis, mb_raw_bl, color=COLOR_MB_RAW, linewidth=1.5, label="MotionBERT Raw", alpha=0.85)
    ax.plot(time_axis, mb_opt_bl, color=COLOR_MB_OPT, linewidth=2, label="MotionBERT Optimized", alpha=0.9)
    ax.plot(time_axis, mp_raw_bl, color=COLOR_MP_RAW, linewidth=1.5, label="MediaPipe Raw", alpha=0.85)
    ax.plot(time_axis, mp_opt_bl, color=COLOR_MP_OPT, linewidth=2, label="MediaPipe Optimized", alpha=0.9)

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Bone Length (cm)", fontsize=11)
    ax.set_title(f"Bone Length Variation: {bone_label}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    output_dir = args.output or "output"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"bone_length_variation_{side}_forearm.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print stats
    print(f"\nBone length statistics (cm):")
    for name, bl in [("GT", gt_bl), ("MB Raw", mb_raw_bl), ("MB Opt", mb_opt_bl),
                     ("MP Raw", mp_raw_bl), ("MP Opt", mp_opt_bl)]:
        valid = bl[~np.isnan(bl)]
        if len(valid) > 0:
            print(f"  {name:15s}: mean={np.mean(valid):.2f}, std={np.std(valid):.4f}, "
                  f"min={np.min(valid):.2f}, max={np.max(valid):.2f}")


if __name__ == "__main__":
    main()
