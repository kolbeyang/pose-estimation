"""Matplotlib visualizations for the MotionBERT pose estimation pipeline."""

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from skeleton import (
    JOINT_NAMES,
    NUM_JOINTS,
    EVAL_JOINT_NAMES,
    NUM_EVAL_JOINTS,
    PARENTS,
)


COORD_NAMES: dict[int, str] = {0: "X", 1: "Y", 2: "Z"}


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure and close."""
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-example graphs
# ---------------------------------------------------------------------------


def generate_trajectory_graphs(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    output_dir: str,
) -> None:
    """One graph per joint per coordinate: Detector vs Optimized vs GT.

    Args:
        detector_3d: List of (17, 3) detector predictions.
        optimized_3d: List of (17, 3) optimized predictions.
        gt_3d: List of (17, 3) or None ground truth.
        output_dir: Directory to save the graphs.
    """
    os.makedirs(output_dir, exist_ok=True)
    n: int = len(detector_3d)
    frames: list[int] = list(range(n))
    det: np.ndarray = np.array(detector_3d)
    opt: np.ndarray = np.array(optimized_3d)
    has_gt: list[bool] = [g is not None for g in gt_3d]

    for j in range(NUM_JOINTS):
        for c, cname in COORD_NAMES.items():
            fig: plt.Figure
            ax: plt.Axes
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(frames, det[:, j, c], "g-", label="Detector", alpha=0.8)
            ax.plot(frames, opt[:, j, c], "r-", label="Optimized", alpha=0.8)
            if any(has_gt):
                gt_vals: list[float] = [
                    gt_3d[i][j, c] if gt_3d[i] is not None else np.nan
                    for i in range(n)
                ]
                ax.plot(frames, gt_vals, "b--", label="Ground Truth", alpha=0.7)
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{cname} (m)")
            ax.set_title(f"{JOINT_NAMES[j]} {cname}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            _save(fig, os.path.join(output_dir, f"{JOINT_NAMES[j]}_{cname}.png"))


def generate_loss_curve(loss_history: list[float], output_dir: str) -> None:
    """Line plot of optimization loss over steps.

    Args:
        loss_history: List of loss values per optimization step.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, "b-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization Loss")
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "loss_curve.png"))


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


def generate_per_joint_szi_error_bar(
    det_per_joint: list[float],
    det_szi_per_joint: list[float],
    output_dir: str,
    opt_per_joint: list[float] | None = None,
    opt_szi_per_joint: list[float] | None = None,
) -> None:
    """Combined 4-bar chart: MPJPE + SZI-MPJPE per joint (12 eval joints).

    Args:
        det_per_joint: 12 per-joint MPJPE values in meters (detector).
        det_szi_per_joint: 12 per-joint SZI-MPJPE values in meters (detector).
        output_dir: Directory to save the graph.
        opt_per_joint: Optional 12 per-joint MPJPE values (optimized).
        opt_szi_per_joint: Optional 12 per-joint SZI-MPJPE values (optimized).
    """
    os.makedirs(output_dir, exist_ok=True)
    x: np.ndarray = np.arange(NUM_EVAL_JOINTS)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(14, 6))

    if opt_per_joint is not None and opt_szi_per_joint is not None:
        bw: float = 0.2
        ax.bar(x - 1.5 * bw, [v * 100 for v in det_per_joint], bw,
               color="green", alpha=0.7, label="Det MPJPE")
        ax.bar(x - 0.5 * bw, [v * 100 for v in opt_per_joint], bw,
               color="red", alpha=0.7, label="Opt MPJPE")
        ax.bar(x + 0.5 * bw, [v * 100 for v in det_szi_per_joint], bw,
               color="lightgreen", alpha=0.7, label="Det SZI-MPJPE")
        ax.bar(x + 1.5 * bw, [v * 100 for v in opt_szi_per_joint], bw,
               color="lightcoral", alpha=0.7, label="Opt SZI-MPJPE")
        ax.legend()
        ax.set_title("Per-Joint MPJPE & SZI-MPJPE (12 eval joints)")
    else:
        ax.bar(x - 0.2, [v * 100 for v in det_per_joint], 0.4,
               color="green", alpha=0.7, label="Det MPJPE")
        ax.bar(x + 0.2, [v * 100 for v in det_szi_per_joint], 0.4,
               color="lightgreen", alpha=0.7, label="Det SZI-MPJPE")
        ax.legend()
        ax.set_title("Per-Joint MPJPE & SZI-MPJPE (Detector, 12 eval joints)")

    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "per_joint_szi_mpjpe.png"))


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


def generate_per_joint_mpjve_bar(
    det_per_joint: list[float],
    opt_per_joint: list[float] | None,
    output_dir: str,
) -> None:
    """Bar chart of per-joint MPJVE (12 eval joints).

    Args:
        det_per_joint: List of 12 per-joint velocity error values in meters/frame (detector).
        opt_per_joint: Optional list of 12 per-joint velocity error values (optimized).
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    x: np.ndarray = np.arange(NUM_EVAL_JOINTS)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(14, 5))

    if opt_per_joint is not None:
        width: float = 0.35
        ax.bar(x - width / 2, [v * 100 for v in det_per_joint],
               width, color="steelblue", alpha=0.7, label="Detector")
        ax.bar(x + width / 2, [v * 100 for v in opt_per_joint],
               width, color="forestgreen", alpha=0.7, label="Optimized")
        ax.legend()
        ax.set_title("Per-Joint Velocity Error (Detector vs Optimized, 12 eval joints)")
    else:
        ax.bar(x, [v * 100 for v in det_per_joint], color="steelblue", alpha=0.7)
        ax.set_title("Per-Joint Velocity Error (Detector, 12 eval joints)")

    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJVE (cm/frame)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "per_joint_mpjve.png"))


def generate_per_frame_mpjve(
    det_per_frame: list[float],
    opt_per_frame: list[float] | None,
    output_dir: str,
) -> None:
    """Line plot of per-frame MPJVE over time.

    Args:
        det_per_frame: List of per-frame velocity error values in meters/frame (detector).
        opt_per_frame: Optional list of per-frame velocity error values (optimized).
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig: plt.Figure
    ax: plt.Axes
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


def generate_bone_lengths_graph(
    bone_lengths: np.ndarray | list[float],
    output_dir: str,
    gt_bone_lengths: list[float] | None = None,
    det_bone_lengths: list[float] | None = None,
) -> None:
    """Bar chart of bone lengths: GT / Detector / Optimized.

    Args:
        bone_lengths: (17,) optimized bone lengths in meters.
        output_dir: Directory to save the graph.
        gt_bone_lengths: Optional (17,) GT bone lengths.
        det_bone_lengths: Optional (17,) detector bone lengths.
    """
    os.makedirs(output_dir, exist_ok=True)
    bone_lengths = np.asarray(bone_lengths)

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(14, 5))
    x: np.ndarray = np.arange(1, NUM_JOINTS)
    labels: list[str] = [JOINT_NAMES[int(i)] for i in x]

    if gt_bone_lengths is not None and det_bone_lengths is not None:
        gt_bl: np.ndarray = np.asarray(gt_bone_lengths)
        det_bl: np.ndarray = np.asarray(det_bone_lengths)
        width: float = 0.25
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


def generate_limb_length_graph(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    output_dir: str,
) -> None:
    """Line graph of 4 arm limb lengths over time for 3 sources (GT, Det, Opt).

    Plots R Shoulder-Elbow, R Elbow-Wrist, L Shoulder-Elbow, L Elbow-Wrist
    for ground truth (blue), detector (green), and optimized (red).

    Args:
        detector_3d: List of (16, 3) detector predictions.
        optimized_3d: List of (16, 3) optimized predictions.
        gt_3d: List of (16, 3) or None ground truth.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)

    LIMB_SEGMENTS: list[tuple[int, int, str]] = [
        (13, 14, "R Shoulder-Elbow"),
        (14, 15, "R Elbow-Wrist"),
        (10, 11, "L Shoulder-Elbow"),
        (11, 12, "L Elbow-Wrist"),
    ]
    LINESTYLES: list[str] = ["-", "--", ":", "-."]

    def _limb_length(positions: np.ndarray, j1: int, j2: int) -> float:
        return float(np.linalg.norm(positions[j2] - positions[j1]))

    n: int = len(detector_3d)
    frames: list[int] = list(range(n))

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(12, 6))

    for seg_idx, (j1, j2, limb_name) in enumerate(LIMB_SEGMENTS):
        ls: str = LINESTYLES[seg_idx]

        # Ground truth (blue)
        gt_lengths: list[float] = [
            _limb_length(gt_3d[i], j1, j2) if gt_3d[i] is not None else np.nan
            for i in range(n)
        ]
        ax.plot(frames, gt_lengths, color="blue", linestyle=ls,
                linewidth=1.2, alpha=0.8, label=f"GT: {limb_name}")

        # Detector (green)
        det_lengths: list[float] = [
            _limb_length(detector_3d[i], j1, j2) for i in range(n)
        ]
        ax.plot(frames, det_lengths, color="green", linestyle=ls,
                linewidth=1.2, alpha=0.8, label=f"Det: {limb_name}")

        # Optimized (red)
        opt_lengths: list[float] = [
            _limb_length(optimized_3d[i], j1, j2) for i in range(n)
        ]
        ax.plot(frames, opt_lengths, color="red", linestyle=ls,
                linewidth=1.2, alpha=0.8, label=f"Opt: {limb_name}")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Limb Length (m)")
    ax.set_title("Arm Limb Lengths Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "limb_lengths.png"))


def generate_confidence_score_graph(
    visibility: list[np.ndarray],
    output_dir: str,
) -> None:
    """Line graph of per-keypoint confidence scores across frames.

    Each line is one of the 16 H36M joints, showing how the Stacked Hourglass
    confidence score varies over time.

    Args:
        visibility: List of (16,) confidence score arrays, one per frame.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)
    vis_arr: np.ndarray = np.stack(visibility)  # (N, 16)
    n_frames: int = vis_arr.shape[0]
    frames: list[int] = list(range(n_frames))
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_JOINTS))

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(figsize=(14, 6))

    for j in range(NUM_JOINTS):
        ax.plot(frames, vis_arr[:, j], color=colors[j], linewidth=1.0,
                alpha=0.8, label=JOINT_NAMES[j])

    ax.set_xlabel("Frame")
    ax.set_ylabel("Confidence Score")
    ax.set_title("Per-Keypoint Confidence Scores")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "confidence_scores.png"))


# ---------------------------------------------------------------------------
# Summary grid (one per example)
# ---------------------------------------------------------------------------


def generate_summary(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    loss_history: list[float],
    metrics: dict[str, Any],
    bone_lengths: np.ndarray | list[float],
    output_dir: str,
    title: str = "Summary",
) -> None:
    """4x4 summary grid: trajectories + loss + per-joint error + bone lengths + MPJVE.

    Args:
        detector_3d: List of (17, 3) detector predictions.
        optimized_3d: List of (17, 3) optimized predictions.
        gt_3d: List of (17, 3) or None ground truth.
        loss_history: List of loss values per optimization step.
        metrics: Dict of computed metrics.
        bone_lengths: (17,) optimized bone lengths in meters.
        output_dir: Directory to save the graph.
        title: Title for the summary.
    """
    os.makedirs(output_dir, exist_ok=True)
    bone_lengths = np.asarray(bone_lengths)

    n: int = len(detector_3d)
    frames: list[int] = list(range(n))
    det: np.ndarray = np.array(detector_3d)
    opt: np.ndarray = np.array(optimized_3d)
    has_gt: bool = any(g is not None for g in gt_3d)

    fig: plt.Figure
    axes: np.ndarray
    fig, axes = plt.subplots(4, 4, figsize=(26, 18))

    # Row 0-1: 6 key joint trajectories (3 columns x 2 rows)
    key_joints: list[int] = [0, 8, 9, 12, 15, 3]  # Hip, Thorax, Neck, LWrist, RWrist, RAnkle
    coord_colors: dict[int, str] = {0: "tab:red", 1: "tab:green", 2: "tab:blue"}

    for idx, j in enumerate(key_joints):
        row: int
        col: int
        row, col = divmod(idx, 3)
        ax: plt.Axes = axes[row, col]
        for c, cname in COORD_NAMES.items():
            color: str = coord_colors[c]
            ax.plot(frames, det[:, j, c], linestyle="-", color=color,
                    alpha=0.4, linewidth=0.8)
            ax.plot(frames, opt[:, j, c], linestyle="--", color=color,
                    alpha=0.7, linewidth=0.8)
            if has_gt:
                gt_vals: list[float] = [
                    gt_3d[i][j, c] if gt_3d[i] is not None else np.nan
                    for i in range(n)
                ]
                ax.plot(frames, gt_vals, linestyle=":", color=color,
                        alpha=0.5, linewidth=0.8)
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.grid(True, alpha=0.2)

    # Legend panel in row 0, col 3
    ax_legend: plt.Axes = axes[0, 3]
    ax_legend.axis("off")
    style_handles: list[Line2D] = [
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5, label="Detector"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="Optimized"),
    ]
    if has_gt:
        style_handles.append(
            Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5, label="Ground Truth"))
    coord_handles: list[Line2D] = [
        Line2D([0], [0], color=coord_colors[0], linewidth=2, label="X"),
        Line2D([0], [0], color=coord_colors[1], linewidth=2, label="Y"),
        Line2D([0], [0], color=coord_colors[2], linewidth=2, label="Z"),
    ]
    ax_legend.legend(handles=style_handles + coord_handles, loc="center",
                     fontsize=10, frameon=True, title="Trajectories",
                     title_fontsize=11)

    # Row 1, Col 3: Per-frame MPJPE
    ax = axes[1, 3]
    if "det_per_frame_mpjpe" in metrics:
        ax.plot([v * 100 for v in metrics["det_per_frame_mpjpe"]], "g-",
                label="Detector", alpha=0.7, linewidth=0.8)
        if "opt_per_frame_mpjpe" in metrics:
            ax.plot([v * 100 for v in metrics["opt_per_frame_mpjpe"]], "r-",
                    label="Optimized", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("MPJPE (cm)", fontsize=7)
        ax.legend(fontsize=7)
    ax.set_title("Per-Frame MPJPE", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Row 2, Col 0: Loss curve
    ax = axes[2, 0]
    ax.plot(loss_history, "b-", linewidth=0.8, label="Loss")
    ax.set_title("Loss", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Row 2, Col 1: Per-joint MPJPE + SZI-MPJPE bar (12 eval joints, 4 bars)
    ax = axes[2, 1]
    if "det_per_joint" in metrics:
        x_ej: np.ndarray = np.arange(NUM_EVAL_JOINTS)
        has_szi_pj: bool = "det_szi_per_joint" in metrics and "opt_szi_per_joint" in metrics
        if has_szi_pj and "opt_per_joint" in metrics:
            bw: float = 0.2
            ax.bar(x_ej - 1.5 * bw, [v * 100 for v in metrics["det_per_joint"]], bw,
                   label="Det MPJPE", color="green", alpha=0.7)
            ax.bar(x_ej - 0.5 * bw, [v * 100 for v in metrics["opt_per_joint"]], bw,
                   label="Opt MPJPE", color="red", alpha=0.7)
            ax.bar(x_ej + 0.5 * bw, [v * 100 for v in metrics["det_szi_per_joint"]], bw,
                   label="Det SZI", color="lightgreen", alpha=0.7)
            ax.bar(x_ej + 1.5 * bw, [v * 100 for v in metrics["opt_szi_per_joint"]], bw,
                   label="Opt SZI", color="lightcoral", alpha=0.7)
        else:
            ax.bar(x_ej - 0.2, [v * 100 for v in metrics["det_per_joint"]], 0.4,
                   label="Detector", color="green", alpha=0.6)
            if "opt_per_joint" in metrics:
                ax.bar(x_ej + 0.2, [v * 100 for v in metrics["opt_per_joint"]], 0.4,
                       label="Optimized", color="red", alpha=0.6)
        ax.set_xticks(x_ej)
        ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=90, fontsize=5)
        ax.legend(fontsize=5)
    ax.set_title("Per-Joint MPJPE & SZI (cm)", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Row 2, Col 2: (empty panel)
    ax = axes[2, 2]
    ax.axis("off")

    # Row 2, Col 3: Bone lengths (GT / Detector / Optimized)
    ax = axes[2, 3]
    x_bones: np.ndarray = np.arange(1, NUM_JOINTS)
    gt_bl_data: list[float] | None = metrics.get("gt_bone_lengths")
    det_bl_data: list[float] | None = metrics.get("det_bone_lengths")
    if gt_bl_data is not None and det_bl_data is not None:
        width: float = 0.25
        ax.bar(x_bones - width, gt_bl_data[1:], width, label="GT", color="blue", alpha=0.6)
        ax.bar(x_bones, det_bl_data[1:], width, label="Detector", color="green", alpha=0.6)
        ax.bar(x_bones + width, bone_lengths[1:], width, label="Optimized", color="red", alpha=0.6)
    else:
        ax.bar(x_bones, bone_lengths[1:], color="steelblue", alpha=0.7, label="Optimized")
    ax.set_xticks(x_bones)
    ax.set_xticklabels([JOINT_NAMES[i] for i in range(1, NUM_JOINTS)], rotation=90, fontsize=5)
    ax.set_title("Bone Lengths (m)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")

    # Row 3, Col 0: Per-frame MPJVE
    ax = axes[3, 0]
    if "det_per_frame_mpjve" in metrics:
        ax.plot([v * 100 for v in metrics["det_per_frame_mpjve"]], "g-",
                label="Detector", alpha=0.7, linewidth=0.8)
        if "opt_per_frame_mpjve" in metrics:
            ax.plot([v * 100 for v in metrics["opt_per_frame_mpjve"]], "r-",
                    label="Optimized", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("MPJVE (cm/frame)", fontsize=7)
        ax.legend(fontsize=7)
    ax.set_title("Per-Frame MPJVE", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Row 3, Col 1: Per-joint MPJVE bar (12 eval joints)
    ax = axes[3, 1]
    if "det_mpjve_per_joint" in metrics:
        x_ej = np.arange(NUM_EVAL_JOINTS)
        ax.bar(x_ej - 0.2, [v * 100 for v in metrics["det_mpjve_per_joint"]], 0.4,
               label="Detector", color="green", alpha=0.6)
        if "opt_mpjve_per_joint" in metrics:
            ax.bar(x_ej + 0.2, [v * 100 for v in metrics["opt_mpjve_per_joint"]], 0.4,
                   label="Optimized", color="red", alpha=0.6)
        ax.set_xticks(x_ej)
        ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=90, fontsize=5)
        ax.legend(fontsize=7)
    ax.set_title("Per-Joint MPJVE (cm/frame)", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Row 3, Col 2: Per-frame 2D MPJPE (pixels)
    ax = axes[3, 2]
    if "det_per_frame_2d_mpjpe" in metrics:
        ax.plot(metrics["det_per_frame_2d_mpjpe"], "g-",
                label="Detector", alpha=0.7, linewidth=0.8)
        if "opt_per_frame_2d_mpjpe" in metrics:
            ax.plot(metrics["opt_per_frame_2d_mpjpe"], "r-",
                    label="Optimized", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("MPJPE (px)", fontsize=7)
        ax.legend(fontsize=7)
    ax.set_title("Per-Frame 2D MPJPE (px)", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Row 3, Col 3: Arm limb lengths over time
    ax = axes[3, 3]
    _LIMB_SEGMENTS: list[tuple[int, int, str]] = [
        (13, 14, "R Sh-El"),
        (14, 15, "R El-Wr"),
        (10, 11, "L Sh-El"),
        (11, 12, "L El-Wr"),
    ]
    _LINESTYLES: list[str] = ["-", "--", ":", "-."]
    for seg_idx, (j1, j2, lname) in enumerate(_LIMB_SEGMENTS):
        ls = _LINESTYLES[seg_idx]
        if has_gt:
            gt_ll: list[float] = [
                float(np.linalg.norm(gt_3d[i][j2] - gt_3d[i][j1]))
                if gt_3d[i] is not None else np.nan
                for i in range(n)
            ]
            ax.plot(frames, gt_ll, color="blue", linestyle=ls,
                    linewidth=0.8, alpha=0.7, label=f"GT: {lname}")
        det_ll: list[float] = [
            float(np.linalg.norm(det[i, j2] - det[i, j1])) for i in range(n)
        ]
        ax.plot(frames, det_ll, color="green", linestyle=ls,
                linewidth=0.8, alpha=0.7, label=f"Det: {lname}")
        opt_ll: list[float] = [
            float(np.linalg.norm(opt[i, j2] - opt[i, j1])) for i in range(n)
        ]
        ax.plot(frames, opt_ll, color="red", linestyle=ls,
                linewidth=0.8, alpha=0.7, label=f"Opt: {lname}")
    ax.set_title("Arm Limb Lengths (m)", fontsize=9)
    ax.legend(fontsize=5, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax.grid(True, alpha=0.2)

    # Overall title with metrics
    metric_str: str = ""
    if "det_mpjpe" in metrics:
        metric_str = (
            f"  |  Det MPJPE: {metrics['det_mpjpe']*100:.1f}cm"
            f"  Opt MPJPE: {metrics['opt_mpjpe']*100:.1f}cm"
        )
    if "det_szi_mpjpe" in metrics:
        metric_str += (
            f"  |  Det SZI: {metrics['det_szi_mpjpe']*100:.1f}cm"
            f"  Opt SZI: {metrics['opt_szi_mpjpe']*100:.1f}cm"
        )
    if "det_mpjve" in metrics:
        metric_str += (
            f"  |  Det MPJVE: {metrics['det_mpjve']*100:.2f}cm/f"
            f"  Opt MPJVE: {metrics['opt_mpjve']*100:.2f}cm/f"
        )
    fig.suptitle(f"{title}{metric_str}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, os.path.join(output_dir, "summary.png"))


# ---------------------------------------------------------------------------
# Aggregate summary across all examples
# ---------------------------------------------------------------------------


def generate_aggregate_summary(
    all_metrics: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Summary across all examples: bar charts + printed table.

    Shows both detector and optimized if opt_mpjpe is present.
    Includes MPJVE bar chart if velocity data is available.

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

    has_opt: bool = all("opt_mpjpe" in m for m in with_gt)
    opt_mpjpe_cm: list[float] = []
    if has_opt:
        opt_mpjpe_cm = [m["opt_mpjpe"] * 100 for m in with_gt]

    # MPJPE + SZI-MPJPE combined bar chart (4 bars per example)
    fig: plt.Figure
    ax: plt.Axes
    x: np.ndarray = np.arange(len(names))

    has_szi: bool = all("det_szi_mpjpe" in m for m in with_gt)
    has_opt_szi: bool = has_szi and all("opt_szi_mpjpe" in m for m in with_gt)

    if has_opt and has_opt_szi:
        det_szi_mpjpe_cm: list[float] = [m["det_szi_mpjpe"] * 100 for m in with_gt]
        opt_szi_mpjpe_cm: list[float] = [m["opt_szi_mpjpe"] * 100 for m in with_gt]
        width: float = 0.2
        fig, ax = plt.subplots(figsize=(max(12, len(names) * 2.0), 6))
        ax.bar(x - 1.5 * width, det_mpjpe_cm, width,
               color="green", alpha=0.7, label="Det MPJPE")
        ax.bar(x - 0.5 * width, opt_mpjpe_cm, width,
               color="red", alpha=0.7, label="Opt MPJPE")
        ax.bar(x + 0.5 * width, det_szi_mpjpe_cm, width,
               color="lightgreen", alpha=0.7, label="Det SZI-MPJPE")
        ax.bar(x + 1.5 * width, opt_szi_mpjpe_cm, width,
               color="lightcoral", alpha=0.7, label="Opt SZI-MPJPE")
        ax.legend()
        ax.set_title("MPJPE & SZI-MPJPE Across Examples")
    elif has_opt:
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 5))
        ax.bar(x - width / 2, det_mpjpe_cm, width,
               color="green", alpha=0.7, label="Detector")
        ax.bar(x + width / 2, opt_mpjpe_cm, width,
               color="red", alpha=0.7, label="Optimized")
        ax.legend()
        ax.set_title("MPJPE Across Examples (Detector vs Optimized)")
    else:
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        ax.bar(x, det_mpjpe_cm, color="green", alpha=0.7)
        ax.set_title("MPJPE Across Examples (Detector)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "aggregate_mpjpe.png"))

    # SZI-MPJPE standalone (kept for backwards compat)
    if has_szi:
        det_szi_cm: list[float] = [m["det_szi_mpjpe"] * 100 for m in with_gt]
        if has_opt_szi:
            opt_szi_cm: list[float] = [m["opt_szi_mpjpe"] * 100 for m in with_gt]
            fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 5))
            ax.bar(x - 0.175, det_szi_cm, 0.35,
                   color="mediumpurple", alpha=0.7, label="Detector")
            ax.bar(x + 0.175, opt_szi_cm, 0.35,
                   color="orchid", alpha=0.7, label="Optimized")
            ax.legend()
            ax.set_title("Scale-Z-Invariant MPJPE Across Examples (Detector vs Optimized)")
        else:
            fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
            ax.bar(x, det_szi_cm, color="mediumpurple", alpha=0.7)
            ax.set_title("Scale-Z-Invariant MPJPE Across Examples (Detector)")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("SZI-MPJPE (cm)")
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, os.path.join(output_dir, "aggregate_szi_mpjpe.png"))

    # MPJVE bar chart (only examples with velocity data)
    with_mpjve: list[dict[str, Any]] = [m for m in with_gt if "det_mpjve" in m]
    if with_mpjve:
        mpjve_names: list[str] = [m.get("name", f"Ex{i}") for i, m in enumerate(with_mpjve)]
        det_mpjve_cm: list[float] = [m["det_mpjve"] * 100 for m in with_mpjve]
        has_opt_mpjve: bool = all("opt_mpjve" in m for m in with_mpjve)
        xv: np.ndarray = np.arange(len(mpjve_names))

        if has_opt_mpjve:
            opt_mpjve_cm: list[float] = [m["opt_mpjve"] * 100 for m in with_mpjve]
            fig, ax = plt.subplots(figsize=(max(10, len(mpjve_names) * 1.5), 5))
            ax.bar(xv - width / 2, det_mpjve_cm, width,
                   color="steelblue", alpha=0.7, label="Detector")
            ax.bar(xv + width / 2, opt_mpjve_cm, width,
                   color="forestgreen", alpha=0.7, label="Optimized")
            ax.legend()
        else:
            fig, ax = plt.subplots(figsize=(max(8, len(mpjve_names) * 1.2), 5))
            ax.bar(xv, det_mpjve_cm, color="steelblue", alpha=0.7)

        ax.set_xticks(xv)
        ax.set_xticklabels(mpjve_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("MPJVE (cm/frame)")
        ax.set_title("Velocity Error Across Examples")
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, os.path.join(output_dir, "aggregate_mpjve.png"))

    # Print table
    print("\n  === Aggregate Results ===")
    if has_opt:
        print(
            f"  {'Example':<35} {'Det MPJPE':>10} {'Opt MPJPE':>10} {'Improv':>8} "
            f"{'Det MPJVE':>10} {'Opt MPJVE':>10}"
        )
        print(
            f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8} {'-'*10} {'-'*10}"
        )
        for i, name in enumerate(names):
            improv: float = det_mpjpe_cm[i] - opt_mpjpe_cm[i]
            det_v: float = with_gt[i].get("det_mpjve", 0) * 100
            opt_v: float = with_gt[i].get("opt_mpjve", 0) * 100
            print(
                f"  {name:<35} {det_mpjpe_cm[i]:>10.2f} {opt_mpjpe_cm[i]:>10.2f} "
                f"{improv:>+8.2f} "
                f"{det_v:>10.2f} {opt_v:>10.2f}"
            )
        mean_det: float = float(np.mean(det_mpjpe_cm))
        mean_opt: float = float(np.mean(opt_mpjpe_cm))
        mean_improv: float = mean_det - mean_opt
        mean_det_v: float = float(np.mean([m.get("det_mpjve", 0) * 100 for m in with_gt]))
        mean_opt_v: float = float(np.mean([m.get("opt_mpjve", 0) * 100 for m in with_gt]))
        print(
            f"  {'MEAN':<35} {mean_det:>10.2f} {mean_opt:>10.2f} "
            f"{mean_improv:>+8.2f} "
            f"{mean_det_v:>10.2f} {mean_opt_v:>10.2f}"
        )
    else:
        print(f"  {'Example':<35} {'MPJPE (cm)':>10}")
        print(f"  {'-'*35} {'-'*10}")
        for i, name in enumerate(names):
            print(f"  {name:<35} {det_mpjpe_cm[i]:>10.2f}")
        mean_mpjpe: float = float(np.mean(det_mpjpe_cm))
        print(f"  {'MEAN':<35} {mean_mpjpe:>10.2f}")
