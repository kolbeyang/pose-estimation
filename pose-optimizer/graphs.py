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
    EVAL_JOINT_NAMES,
    NUM_EVAL_JOINTS,
)


COORD_NAMES: dict[int, str] = {0: "X", 1: "Y", 2: "Z"}


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure to disk and close the matplotlib figure.

    Args:
        fig: Matplotlib figure to save.
        path: Output file path.
    """
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
        detector_3d: List of (16, 3) detector predictions. [3D:SKELETON_16]
        optimized_3d: List of (16, 3) optimized predictions. [3D:SKELETON_16]
        gt_3d: List of (16, 3) or None ground truth. [3D:SKELETON_16]
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
                    g[j, c] if (g := gt_3d[i]) is not None else np.nan
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
    """Line plot of optimization loss over steps.

    Args:
        loss_history: List of loss values per optimization step.
        output_dir: Directory to save the graph.
    """
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
    """Bar chart of per-joint MPJPE for [SKELETON_16_EVAL] joints (15 eval joints).

    Args:
        det_per_joint: List of per-joint errors for eval joints. [SKELETON_16_EVAL]
        output_dir: Directory to save the graph.
        opt_per_joint: Optional optimized per-joint errors. [SKELETON_16_EVAL]
    """
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
        ax.set_title("Per-Joint Error (Detector vs Optimized, 15 eval joints)")
    else:
        ax.bar(x, [v * 100 for v in det_per_joint], color="steelblue", alpha=0.7)
        ax.set_title("Per-Joint Error (Detector, 15 eval joints)")

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
    """Line plot of per-frame MPJPE over time.

    Args:
        det_per_frame: Per-frame MPJPE values for detector.
        output_dir: Directory to save the graph.
        opt_per_frame: Optional per-frame MPJPE values for optimized.
    """
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
    """Line plot of per-frame MPJVE (velocity error) over time.

    Args:
        det_per_frame: Per-frame MPJVE values for detector.
        opt_per_frame: Optional per-frame MPJVE values for optimized.
        output_dir: Directory to save the graph.
    """
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
    """Bar chart of bone lengths: GT / Detector / Optimized.

    Args:
        bone_lengths: (16,) optimized bone lengths for [3D:SKELETON_16].
        output_dir: Directory to save the graph.
        gt_bone_lengths: Optional (16,) GT bone lengths.
        det_bone_lengths: Optional (16,) detector bone lengths.
    """
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
    _detector_3d: list[np.ndarray],
    _optimized_3d: list[np.ndarray],
    _gt_3d: list[np.ndarray | None],
    loss_history: list[float],
    metrics: dict[str, Any],
    bone_lengths: np.ndarray,
    output_dir: str,
    title: str = "",
) -> None:
    """Combined multi-panel summary image.

    Panels: loss curve, per-joint error, per-frame MPJPE, bone lengths.

    Args:
        detector_3d: List of (16, 3) detector predictions. [3D:SKELETON_16]
        optimized_3d: List of (16, 3) optimized predictions. [3D:SKELETON_16]
        gt_3d: List of (16, 3) or None ground truth. [3D:SKELETON_16]
        loss_history: List of loss values per step.
        metrics: Dict with per-joint and per-frame metrics.
        bone_lengths: (16,) optimized bone lengths.
        output_dir: Directory to save the summary.
        title: Optional title for the figure.
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

# ---------------------------------------------------------------------------
# Cross-pipeline comparison graphs (Phase 4)
# ---------------------------------------------------------------------------

# Consistent color scheme for cross-pipeline graphs
COLOR_GT = "#5871CA"          # Blue
COLOR_MB_RAW = "#FFA2DB"      # Light Pink
COLOR_MB_OPT = "#FF389C"      # Pink
COLOR_MP_RAW = "#FFB199"      # Light Orange
COLOR_MP_OPT = "#FF7A21"      # Orange


def generate_cross_pipeline_per_joint_position(
    all_mb_metrics: list[dict[str, Any]],
    all_mp_metrics: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Bar chart of per-joint VW-SI-MPJPE averaged across examples, 4 bars per joint.

    Args:
        all_mb_metrics: List of per-example MotionBERT metrics.
        all_mp_metrics: List of per-example MediaPipe metrics.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect per-joint VW-SI-MPJPE across examples and average
    mb_det_all = [m["det_vw_si_mpjpe_per_joint"] for m in all_mb_metrics if "det_vw_si_mpjpe_per_joint" in m]
    mb_opt_all = [m["opt_vw_si_mpjpe_per_joint"] for m in all_mb_metrics if "opt_vw_si_mpjpe_per_joint" in m]
    mp_det_all = [m["det_vw_si_mpjpe_per_joint"] for m in all_mp_metrics if "det_vw_si_mpjpe_per_joint" in m]
    mp_opt_all = [m["opt_vw_si_mpjpe_per_joint"] for m in all_mp_metrics if "opt_vw_si_mpjpe_per_joint" in m]

    if not (mb_det_all and mb_opt_all and mp_det_all and mp_opt_all):
        return

    mb_det = np.mean(mb_det_all, axis=0) * 100  # to cm
    mb_opt = np.mean(mb_opt_all, axis=0) * 100
    mp_det = np.mean(mp_det_all, axis=0) * 100
    mp_opt = np.mean(mp_opt_all, axis=0) * 100

    # SEM across videos: std / sqrt(N) per joint (to cm)
    mb_det_err = np.std(mb_det_all, axis=0, ddof=1) / np.sqrt(len(mb_det_all)) * 100 if len(mb_det_all) > 1 else np.zeros_like(mb_det)
    mb_opt_err = np.std(mb_opt_all, axis=0, ddof=1) / np.sqrt(len(mb_opt_all)) * 100 if len(mb_opt_all) > 1 else np.zeros_like(mb_opt)
    mp_det_err = np.std(mp_det_all, axis=0, ddof=1) / np.sqrt(len(mp_det_all)) * 100 if len(mp_det_all) > 1 else np.zeros_like(mp_det)
    mp_opt_err = np.std(mp_opt_all, axis=0, ddof=1) / np.sqrt(len(mp_opt_all)) * 100 if len(mp_opt_all) > 1 else np.zeros_like(mp_opt)

    x = np.arange(NUM_EVAL_JOINTS)
    width = 0.2
    ekw = dict(ecolor="black", capsize=2, elinewidth=0.8)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 1.5 * width, mb_det, width, yerr=mb_det_err, error_kw=ekw, color=COLOR_MB_RAW, label="MotionBERT Raw")
    ax.bar(x - 0.5 * width, mb_opt, width, yerr=mb_opt_err, error_kw=ekw, color=COLOR_MB_OPT, label="MotionBERT Optimized")
    ax.bar(x + 0.5 * width, mp_det, width, yerr=mp_det_err, error_kw=ekw, color=COLOR_MP_RAW, label="MediaPipe Raw")
    ax.bar(x + 1.5 * width, mp_opt, width, yerr=mp_opt_err, error_kw=ekw, color=COLOR_MP_OPT, label="MediaPipe Optimized")

    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("VW-SI-MPJPE (cm)")
    ax.set_title(f"Position Error Per-Joint: MotionBERT vs MediaPipe (±SEM, N={len(mb_det_all)})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "cross_pipeline_per_joint_position.png"))


def generate_cross_pipeline_per_joint_velocity(
    all_mb_metrics: list[dict[str, Any]],
    all_mp_metrics: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Bar chart of per-joint VW-SI-MPJVE averaged across examples, 4 bars per joint.

    Args:
        all_mb_metrics: List of per-example MotionBERT metrics.
        all_mp_metrics: List of per-example MediaPipe metrics.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)

    mb_det_all = [m["det_vw_si_mpjve_per_joint"] for m in all_mb_metrics if "det_vw_si_mpjve_per_joint" in m]
    mb_opt_all = [m["opt_vw_si_mpjve_per_joint"] for m in all_mb_metrics if "opt_vw_si_mpjve_per_joint" in m]
    mp_det_all = [m["det_vw_si_mpjve_per_joint"] for m in all_mp_metrics if "det_vw_si_mpjve_per_joint" in m]
    mp_opt_all = [m["opt_vw_si_mpjve_per_joint"] for m in all_mp_metrics if "opt_vw_si_mpjve_per_joint" in m]

    if not (mb_det_all and mb_opt_all and mp_det_all and mp_opt_all):
        return

    mb_det = np.mean(mb_det_all, axis=0) * 100
    mb_opt = np.mean(mb_opt_all, axis=0) * 100
    mp_det = np.mean(mp_det_all, axis=0) * 100
    mp_opt = np.mean(mp_opt_all, axis=0) * 100

    mb_det_err = np.std(mb_det_all, axis=0, ddof=1) / np.sqrt(len(mb_det_all)) * 100 if len(mb_det_all) > 1 else np.zeros_like(mb_det)
    mb_opt_err = np.std(mb_opt_all, axis=0, ddof=1) / np.sqrt(len(mb_opt_all)) * 100 if len(mb_opt_all) > 1 else np.zeros_like(mb_opt)
    mp_det_err = np.std(mp_det_all, axis=0, ddof=1) / np.sqrt(len(mp_det_all)) * 100 if len(mp_det_all) > 1 else np.zeros_like(mp_det)
    mp_opt_err = np.std(mp_opt_all, axis=0, ddof=1) / np.sqrt(len(mp_opt_all)) * 100 if len(mp_opt_all) > 1 else np.zeros_like(mp_opt)

    x = np.arange(NUM_EVAL_JOINTS)
    width = 0.2
    ekw = dict(ecolor="black", capsize=2, elinewidth=0.8)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 1.5 * width, mb_det, width, yerr=mb_det_err, error_kw=ekw, color=COLOR_MB_RAW, label="MotionBERT Raw")
    ax.bar(x - 0.5 * width, mb_opt, width, yerr=mb_opt_err, error_kw=ekw, color=COLOR_MB_OPT, label="MotionBERT Optimized")
    ax.bar(x + 0.5 * width, mp_det, width, yerr=mp_det_err, error_kw=ekw, color=COLOR_MP_RAW, label="MediaPipe Raw")
    ax.bar(x + 1.5 * width, mp_opt, width, yerr=mp_opt_err, error_kw=ekw, color=COLOR_MP_OPT, label="MediaPipe Optimized")

    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("VW-SI-MPJVE (cm/frame)")
    ax.set_title(f"Velocity Error Per-Joint: MotionBERT vs MediaPipe (±SEM, N={len(mb_det_all)})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "cross_pipeline_per_joint_velocity.png"))


def generate_cross_pipeline_metrics_comparison(
    all_mb_metrics: list[dict[str, Any]],
    all_mp_metrics: list[dict[str, Any]],
    output_dir: str,
) -> None:
    """Bar chart comparing aggregate VW-SI-MPJPE and VW-SI-MPJVE across pipelines.

    4 bars per metric group: MB Raw, MB Opt, MP Raw, MP Opt.

    Args:
        all_mb_metrics: List of per-example MotionBERT metrics.
        all_mp_metrics: List of per-example MediaPipe metrics.
        output_dir: Directory to save the graph.
    """
    os.makedirs(output_dir, exist_ok=True)

    def _avg_sem(metrics_list: list[dict], key: str) -> tuple[float, float, int] | None:
        """Return (mean, SEM, N) of values for key, or None if empty.

        SEM = std / sqrt(N): standard error of the mean across videos.
        """
        vals = [m[key] for m in metrics_list if key in m]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, sem, len(arr)

    metric_groups = []  # list of (4 means * 100, 4 sems * 100)
    labels = []

    def _add_group(mb_det_key, mb_opt_key, mp_det_key, mp_opt_key, label):
        mb_d = _avg_sem(all_mb_metrics, mb_det_key)
        mb_o = _avg_sem(all_mb_metrics, mb_opt_key)
        mp_d = _avg_sem(all_mp_metrics, mp_det_key)
        mp_o = _avg_sem(all_mp_metrics, mp_opt_key)
        if mb_d is None or mb_o is None or mp_d is None or mp_o is None:
            return
        means = (mb_d[0] * 100, mb_o[0] * 100, mp_d[0] * 100, mp_o[0] * 100)
        sems = (mb_d[1] * 100, mb_o[1] * 100, mp_d[1] * 100, mp_o[1] * 100)
        metric_groups.append((means, sems))
        labels.append(label)

    _add_group("det_vw_si_mpjpe", "opt_vw_si_mpjpe", "det_vw_si_mpjpe", "opt_vw_si_mpjpe", "VW-SI-MPJPE (cm)")
    _add_group("det_vw_si_mpjve", "opt_vw_si_mpjve", "det_vw_si_mpjve", "opt_vw_si_mpjve", "VW-SI-MPJVE (cm/f)")

    if not metric_groups:
        return

    n_mb = len(all_mb_metrics)
    n_mp = len(all_mp_metrics)
    x = np.arange(len(metric_groups))
    width = 0.18
    ekw = dict(ecolor="black", capsize=3, elinewidth=1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (means, sems) in enumerate(metric_groups):
        mb_d, mb_o, mp_d, mp_o = means
        mb_d_e, mb_o_e, mp_d_e, mp_o_e = sems
        ax.bar(x[i] - 1.5 * width, mb_d, width, yerr=mb_d_e, error_kw=ekw, color=COLOR_MB_RAW,
               label="MotionBERT Raw" if i == 0 else "")
        ax.bar(x[i] - 0.5 * width, mb_o, width, yerr=mb_o_e, error_kw=ekw, color=COLOR_MB_OPT,
               label="MotionBERT Optimized" if i == 0 else "")
        ax.bar(x[i] + 0.5 * width, mp_d, width, yerr=mp_d_e, error_kw=ekw, color=COLOR_MP_RAW,
               label="MediaPipe Raw" if i == 0 else "")
        ax.bar(x[i] + 1.5 * width, mp_o, width, yerr=mp_o_e, error_kw=ekw, color=COLOR_MP_OPT,
               label="MediaPipe Optimized" if i == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Error (see x-axis for units)")
    ax.set_title(f"Metrics Comparison: MotionBERT vs MediaPipe (±SEM, N_MB={n_mb}, N_MP={n_mp})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "cross_pipeline_metrics_comparison.png"))


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def generate_aggregate_summary(
    all_metrics: list[dict[str, Any]],
    output_dir: str,
    prefix: str = "",
) -> None:
    """Generate cross-example aggregate summary bar chart.

    Args:
        all_metrics: List of per-example metrics dicts.
        output_dir: Directory to save the aggregate summary.
        prefix: Optional prefix for output filename (e.g. "motionbert").
    """
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
        ekw = dict(ecolor="black", capsize=2, elinewidth=0.8)
        for idx, (key, label) in enumerate(bar_keys):
            vals = [v * 100 if v is not None else 0 for v in available[key]]
            std_key = f"{key}_frame_std"
            errs = [
                (m[std_key] * 100 if std_key in m and m[std_key] is not None else 0)
                for m in all_metrics
            ]
            offset = (idx - len(bar_keys) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, yerr=errs, error_kw=ekw,
                   label=label, color=colors[idx % len(colors)], alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Error (cm)")
        ax.set_title("Aggregate Results (error bars = per-frame std within video)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    filename = f"{prefix}_aggregate_summary.png" if prefix else "aggregate_summary.png"
    _save(fig, os.path.join(output_dir, filename))
