"""Unified graph generation for both MotionBert and MediaPipe pipelines.

Generates trajectory graphs, per-joint error bars, per-frame MPJPE/MPJVE,
loss curves, bone lengths, and summary images.
"""

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skeleton import (
    JOINT_NAMES,
    NUM_JOINTS,
    EVAL_JOINTS,
    EVAL_JOINT_NAMES,
    NUM_EVAL_JOINTS,
    PARENTS,
)


COORD_NAMES: dict[int, str] = {0: "X", 1: "Y", 2: "Z"}


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure and close."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trajectory graphs
# ---------------------------------------------------------------------------

def generate_trajectory_graphs(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    output_dir: str,
) -> None:
    """One graph per joint per coordinate: Detector vs Optimized vs GT.

    Args:
        detector_3d: List of (16, 3) detector predictions.
        optimized_3d: List of (16, 3) optimized predictions.
        gt_3d: List of (16, 3) or None ground truth.
        output_dir: Directory to save the graphs.
    """
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    n = len(detector_3d)
    frames = list(range(n))
    det = np.array(detector_3d)
    opt = np.array(optimized_3d)
    has_gt = [g is not None for g in gt_3d]

    for j in range(NUM_JOINTS):
        for c, cname in COORD_NAMES.items():
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(frames, det[:, j, c], "g-", label="Detector", alpha=0.8)
            ax.plot(frames, opt[:, j, c], "r-", label="Optimized", alpha=0.8)
            if any(has_gt):
                gt_vals = [
                    gt_3d[i][j, c] if gt_3d[i] is not None else np.nan
                    for i in range(n)
                ]
                ax.plot(frames, gt_vals, "b--", label="Ground Truth", alpha=0.7)
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{cname} (m)")
            ax.set_title(f"{JOINT_NAMES[j]} {cname}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            _save(fig, os.path.join(traj_dir, f"{JOINT_NAMES[j]}_{cname}.png"))


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def generate_loss_curve(loss_history: list[float], output_dir: str) -> None:
    """Line plot of optimization loss over steps."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, "b-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization Loss")
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "loss_curve.png"))


# ---------------------------------------------------------------------------
# Per-joint error bar
# ---------------------------------------------------------------------------

def generate_per_joint_error_bar(
    det_per_joint: list[float],
    output_dir: str,
    opt_per_joint: list[float] | None = None,
) -> None:
    """Bar chart of per-joint MPJPE (12 eval joints)."""
    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(NUM_EVAL_JOINTS)

    fig, ax = plt.subplots(figsize=(14, 5))
    if opt_per_joint is not None:
        width = 0.35
        ax.bar(x - width / 2, [v * 100 for v in det_per_joint],
               width, color="steelblue", alpha=0.7, label="Detector")
        ax.bar(x + width / 2, [v * 100 for v in opt_per_joint],
               width, color="forestgreen", alpha=0.7, label="Optimized")
        ax.legend()
        ax.set_title("Per-Joint Error (Detector vs Optimized, 12 eval joints)")
    else:
        ax.bar(x, [v * 100 for v in det_per_joint], color="steelblue", alpha=0.7)
        ax.set_title("Per-Joint Error (Detector, 12 eval joints)")

    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "per_joint_error.png"))


# ---------------------------------------------------------------------------
# Per-frame MPJPE
# ---------------------------------------------------------------------------

def generate_per_frame_mpjpe(
    det_per_frame: list[float],
    output_dir: str,
    opt_per_frame: list[float] | None = None,
) -> None:
    """Line plot of per-frame MPJPE over time."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([v * 100 for v in det_per_frame], "b-", alpha=0.8, label="Detector")
    if opt_per_frame is not None:
        ax.plot([v * 100 for v in opt_per_frame], "g-", alpha=0.8, label="Optimized")
    ax.set_xlabel("Frame")
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("Per-Frame MPJPE vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "per_frame_mpjpe.png"))


# ---------------------------------------------------------------------------
# Per-frame MPJVE
# ---------------------------------------------------------------------------

def generate_per_frame_mpjve(
    det_per_frame: list[float],
    opt_per_frame: list[float] | None,
    output_dir: str,
) -> None:
    """Line plot of per-frame MPJVE over time."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([v * 100 for v in det_per_frame], "b-", alpha=0.8, label="Detector")
    if opt_per_frame is not None:
        ax.plot([v * 100 for v in opt_per_frame], "g-", alpha=0.8, label="Optimized")
    ax.set_xlabel("Frame Transition")
    ax.set_ylabel("MPJVE (cm/frame)")
    ax.set_title("Per-Frame Velocity Error vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "per_frame_mpjve.png"))


# ---------------------------------------------------------------------------
# Bone lengths
# ---------------------------------------------------------------------------

def generate_bone_lengths_graph(
    bone_lengths: np.ndarray,
    output_dir: str,
    gt_bone_lengths: np.ndarray | None = None,
    det_bone_lengths: np.ndarray | None = None,
) -> None:
    """Bar chart of bone lengths: GT / Detector / Optimized."""
    os.makedirs(output_dir, exist_ok=True)
    bone_lengths = np.asarray(bone_lengths)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(1, NUM_JOINTS)
    labels = [JOINT_NAMES[int(i)] for i in x]

    if gt_bone_lengths is not None and det_bone_lengths is not None:
        gt_bl = np.asarray(gt_bone_lengths)
        det_bl = np.asarray(det_bone_lengths)
        width = 0.25
        ax.bar(x - width, gt_bl[1:], width, label="Ground Truth", color="blue", alpha=0.7)
        ax.bar(x, det_bl[1:], width, label="Detector", color="green", alpha=0.7)
        ax.bar(x + width, bone_lengths[1:], width, label="Optimized", color="red", alpha=0.7)
        ax.legend()
        ax.set_title("Bone Lengths (m)")
    else:
        ax.bar(x, bone_lengths[1:], color="steelblue")
        ax.set_title("Optimized Bone Lengths")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Length (m)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "bone_lengths.png"))


# ---------------------------------------------------------------------------
# Summary image
# ---------------------------------------------------------------------------

def generate_summary(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    loss_history: list[float],
    metrics: dict[str, Any],
    bone_lengths: np.ndarray,
    output_dir: str,
    title: str = "",
) -> None:
    """Combined multi-panel summary image.

    Panels: loss curve, per-joint error, per-frame MPJPE, bone lengths.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Panel 1: Loss curve
    ax = axes[0, 0]
    ax.plot(loss_history, "b-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization Loss")
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-joint error bar
    ax = axes[0, 1]
    if "det_per_joint" in metrics:
        x = np.arange(NUM_EVAL_JOINTS)
        width = 0.35
        ax.bar(x - width / 2, [v * 100 for v in metrics["det_per_joint"]],
               width, color="steelblue", alpha=0.7, label="Detector")
        if "opt_per_joint" in metrics:
            ax.bar(x + width / 2, [v * 100 for v in metrics["opt_per_joint"]],
                   width, color="forestgreen", alpha=0.7, label="Optimized")
        ax.set_xticks(x)
        ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("MPJPE (cm)")
        ax.set_title("Per-Joint Error")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No GT data", ha="center", va="center", transform=ax.transAxes)

    # Panel 3: Per-frame MPJPE
    ax = axes[1, 0]
    if "det_per_frame_mpjpe" in metrics:
        ax.plot([v * 100 for v in metrics["det_per_frame_mpjpe"]], "b-", alpha=0.8, label="Detector")
        if "opt_per_frame_mpjpe" in metrics:
            ax.plot([v * 100 for v in metrics["opt_per_frame_mpjpe"]], "g-", alpha=0.8, label="Optimized")
        ax.set_xlabel("Frame")
        ax.set_ylabel("MPJPE (cm)")
        ax.set_title("Per-Frame MPJPE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No GT data", ha="center", va="center", transform=ax.transAxes)

    # Panel 4: Bone lengths
    ax = axes[1, 1]
    x = np.arange(1, NUM_JOINTS)
    labels = [JOINT_NAMES[int(i)] for i in x]
    bone_lengths = np.asarray(bone_lengths)
    if "gt_bone_lengths" in metrics and "det_bone_lengths" in metrics:
        gt_bl = np.asarray(metrics["gt_bone_lengths"])
        det_bl = np.asarray(metrics["det_bone_lengths"])
        bw = 0.25
        ax.bar(x - bw, gt_bl[1:], bw, label="GT", color="blue", alpha=0.7)
        ax.bar(x, det_bl[1:], bw, label="Detector", color="green", alpha=0.7)
        ax.bar(x + bw, bone_lengths[1:], bw, label="Optimized", color="red", alpha=0.7)
        ax.legend(fontsize=8)
    else:
        ax.bar(x, bone_lengths[1:], color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Length (m)")
    ax.set_title("Bone Lengths")
    ax.grid(True, alpha=0.3, axis="y")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "summary.png"))


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def generate_aggregate_summary(
    all_metrics: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Generate cross-example aggregate summary."""
    os.makedirs(output_dir, exist_ok=True)

    names = [m.get("name", f"ex_{i}") for i, m in enumerate(all_metrics)]

    # Collect available metrics
    metric_keys = ["det_mpjpe", "opt_mpjpe", "det_si_mpjpe", "opt_si_mpjpe",
                   "det_vw_si_mpjpe", "opt_vw_si_mpjpe"]
    available = {k: [] for k in metric_keys}
    for m in all_metrics:
        for k in metric_keys:
            if k in m:
                available[k].append(m[k])
            else:
                available[k].append(None)

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.8), 6))
    x = np.arange(len(names))
    bar_keys = [(k, l) for k, l in [
        ("det_mpjpe", "Det MPJPE"),
        ("opt_mpjpe", "Opt MPJPE"),
        ("det_vw_si_mpjpe", "Det VW-SI-MPJPE"),
        ("opt_vw_si_mpjpe", "Opt VW-SI-MPJPE"),
    ] if any(v is not None for v in available.get(k, []))]

    if bar_keys:
        width = 0.8 / len(bar_keys)
        colors = ["steelblue", "forestgreen", "orange", "red", "purple", "teal"]
        for idx, (key, label) in enumerate(bar_keys):
            vals = [v * 100 if v is not None else 0 for v in available[key]]
            offset = (idx - len(bar_keys) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=label, color=colors[idx % len(colors)], alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Error (cm)")
        ax.set_title("Aggregate Results")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    _save(fig, os.path.join(output_dir, "aggregate_summary.png"))
