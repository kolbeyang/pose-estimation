"""Matplotlib visualizations for the MotionBERT pose estimation pipeline."""

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skeleton import EVAL_JOINT_NAMES, NUM_EVAL_JOINTS


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure and close."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def generate_per_joint_error_bar(
    per_joint: list[float],
    output_dir: str,
) -> None:
    """Bar chart of per-joint MPJPE for detector predictions (12 eval joints).

    Args:
        per_joint: List of 12 per-joint MPJPE values in meters.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    x: np.ndarray = np.arange(NUM_EVAL_JOINTS)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x, [v * 100 for v in per_joint], color="steelblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("Per-Joint Error vs Ground Truth (Detector, 12 eval joints)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "per_joint_error.png"))


def generate_per_frame_mpjpe(
    per_frame: list[float],
    output_dir: str,
) -> None:
    """Line plot of per-frame MPJPE over time.

    Args:
        per_frame: List of per-frame MPJPE values in meters.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([v * 100 for v in per_frame], "b-", alpha=0.8, label="Detector")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("Per-Frame MPJPE vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "per_frame_mpjpe.png"))


def generate_aggregate_summary(
    all_metrics: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Summary across all examples: bar chart + printed table.

    Args:
        all_metrics: List of metric dicts from compute_comparison.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter examples that have GT
    with_gt: list[dict[str, Any]] = [m for m in all_metrics if "det_mpjpe" in m]
    if not with_gt:
        print("  No examples with ground truth -- skipping aggregate summary.")
        return

    names: list[str] = [m.get("name", f"Ex{i}") for i, m in enumerate(with_gt)]
    det_mpjpe_cm: list[float] = [m["det_mpjpe"] * 100 for m in with_gt]
    det_p_mpjpe_cm: list[float] = [m["det_p_mpjpe"] * 100 for m in with_gt]

    # MPJPE bar chart
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    x: np.ndarray = np.arange(len(names))
    ax.bar(x, det_mpjpe_cm, color="steelblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("MPJPE Across Examples (Detector)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "aggregate_mpjpe.png"))

    # P-MPJPE bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax.bar(x, det_p_mpjpe_cm, color="darkorange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P-MPJPE (cm)")
    ax.set_title("Procrustes-Aligned MPJPE Across Examples (Detector)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "aggregate_p_mpjpe.png"))

    # Print table
    print("\n  === Aggregate Results ===")
    print(f"  {'Example':<35} {'MPJPE (cm)':>10} {'P-MPJPE (cm)':>12}")
    print(f"  {'-'*35} {'-'*10} {'-'*12}")
    for i, name in enumerate(names):
        print(f"  {name:<35} {det_mpjpe_cm[i]:>10.2f} {det_p_mpjpe_cm[i]:>12.2f}")
    mean_mpjpe: float = float(np.mean(det_mpjpe_cm))
    mean_p_mpjpe: float = float(np.mean(det_p_mpjpe_cm))
    print(f"  {'MEAN':<35} {mean_mpjpe:>10.2f} {mean_p_mpjpe:>12.2f}")
