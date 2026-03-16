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
    opt_per_joint: list[float] | None = None,
) -> None:
    """Bar chart of per-joint MPJPE for detector predictions (12 eval joints).

    Shows side-by-side bars if opt_per_joint is provided.

    Args:
        per_joint: List of 12 per-joint MPJPE values in meters (detector).
        output_dir: Directory to save the graph.
        opt_per_joint: Optional list of 12 per-joint MPJPE values (optimized).
    """
    os.makedirs(output_dir, exist_ok=True)
    x: np.ndarray = np.arange(NUM_EVAL_JOINTS)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(14, 5))

    if opt_per_joint is not None:
        width: float = 0.35
        ax.bar(x - width / 2, [v * 100 for v in per_joint],
               width, color="steelblue", alpha=0.7, label="Detector")
        ax.bar(x + width / 2, [v * 100 for v in opt_per_joint],
               width, color="forestgreen", alpha=0.7, label="Optimized")
        ax.legend()
        ax.set_title("Per-Joint Error vs Ground Truth (Detector vs Optimized, 12 eval joints)")
    else:
        ax.bar(x, [v * 100 for v in per_joint], color="steelblue", alpha=0.7)
        ax.set_title("Per-Joint Error vs Ground Truth (Detector, 12 eval joints)")

    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "per_joint_error.png"))


def generate_per_frame_mpjpe(
    per_frame: list[float],
    output_dir: str,
    opt_per_frame: list[float] | None = None,
) -> None:
    """Line plot of per-frame MPJPE over time.

    Shows both detector and optimized if opt_per_frame is provided.

    Args:
        per_frame: List of per-frame MPJPE values in meters (detector).
        output_dir: Directory to save the graph.
        opt_per_frame: Optional list of per-frame MPJPE values (optimized).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([v * 100 for v in per_frame], "b-", alpha=0.8, label="Detector")
    if opt_per_frame is not None:
        ax.plot([v * 100 for v in opt_per_frame], "g-", alpha=0.8, label="Optimized")
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

    Shows both detector and optimized if opt_mpjpe is present.

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

    has_opt: bool = all("opt_mpjpe" in m for m in with_gt)
    opt_mpjpe_cm: list[float] = []
    opt_p_mpjpe_cm: list[float] = []
    if has_opt:
        opt_mpjpe_cm = [m["opt_mpjpe"] * 100 for m in with_gt]
        opt_p_mpjpe_cm = [m["opt_p_mpjpe"] * 100 for m in with_gt]

    # MPJPE bar chart
    fig: plt.Figure
    ax: plt.Axes
    x: np.ndarray = np.arange(len(names))

    if has_opt:
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 5))
        width: float = 0.35
        ax.bar(x - width / 2, det_mpjpe_cm, width,
               color="steelblue", alpha=0.7, label="Detector")
        ax.bar(x + width / 2, opt_mpjpe_cm, width,
               color="forestgreen", alpha=0.7, label="Optimized")
        ax.legend()
        ax.set_title("MPJPE Across Examples (Detector vs Optimized)")
    else:
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        ax.bar(x, det_mpjpe_cm, color="steelblue", alpha=0.7)
        ax.set_title("MPJPE Across Examples (Detector)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "aggregate_mpjpe.png"))

    # P-MPJPE bar chart
    if has_opt:
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 5))
        ax.bar(x - width / 2, det_p_mpjpe_cm, width,
               color="darkorange", alpha=0.7, label="Detector")
        ax.bar(x + width / 2, opt_p_mpjpe_cm, width,
               color="forestgreen", alpha=0.7, label="Optimized")
        ax.legend()
        ax.set_title("Procrustes-Aligned MPJPE Across Examples (Detector vs Optimized)")
    else:
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        ax.bar(x, det_p_mpjpe_cm, color="darkorange", alpha=0.7)
        ax.set_title("Procrustes-Aligned MPJPE Across Examples (Detector)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P-MPJPE (cm)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "aggregate_p_mpjpe.png"))

    # Print table
    print("\n  === Aggregate Results ===")
    if has_opt:
        print(f"  {'Example':<35} {'Det MPJPE':>10} {'Opt MPJPE':>10} {'Improv':>8} {'Det P-MPJPE':>12} {'Opt P-MPJPE':>12}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8} {'-'*12} {'-'*12}")
        for i, name in enumerate(names):
            improv: float = det_mpjpe_cm[i] - opt_mpjpe_cm[i]
            print(
                f"  {name:<35} {det_mpjpe_cm[i]:>10.2f} {opt_mpjpe_cm[i]:>10.2f} "
                f"{improv:>+8.2f} {det_p_mpjpe_cm[i]:>12.2f} {opt_p_mpjpe_cm[i]:>12.2f}"
            )
        mean_det: float = float(np.mean(det_mpjpe_cm))
        mean_opt: float = float(np.mean(opt_mpjpe_cm))
        mean_improv: float = mean_det - mean_opt
        mean_det_p: float = float(np.mean(det_p_mpjpe_cm))
        mean_opt_p: float = float(np.mean(opt_p_mpjpe_cm))
        print(
            f"  {'MEAN':<35} {mean_det:>10.2f} {mean_opt:>10.2f} "
            f"{mean_improv:>+8.2f} {mean_det_p:>12.2f} {mean_opt_p:>12.2f}"
        )
    else:
        print(f"  {'Example':<35} {'MPJPE (cm)':>10} {'P-MPJPE (cm)':>12}")
        print(f"  {'-'*35} {'-'*10} {'-'*12}")
        for i, name in enumerate(names):
            print(f"  {name:<35} {det_mpjpe_cm[i]:>10.2f} {det_p_mpjpe_cm[i]:>12.2f}")
        mean_mpjpe: float = float(np.mean(det_mpjpe_cm))
        mean_p_mpjpe: float = float(np.mean(det_p_mpjpe_cm))
        print(f"  {'MEAN':<35} {mean_mpjpe:>10.2f} {mean_p_mpjpe:>12.2f}")
