"""Generate comparison graphs for the full-body FK optimisation pipeline."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from skeleton import JOINT_NAMES, NUM_JOINTS, JOINT_GROUP, GROUP_COLORS_RGB


COORD_NAMES = {0: "X", 1: "Y", 2: "Z"}


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-example graphs
# ---------------------------------------------------------------------------

def generate_trajectory_graphs(
    mediapipe_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    output_dir: str,
):
    """One graph per joint per coordinate: MediaPipe vs Optimised vs GT."""
    os.makedirs(output_dir, exist_ok=True)
    n = len(mediapipe_3d)
    frames = list(range(n))
    mp = np.array(mediapipe_3d)
    opt = np.array(optimized_3d)
    has_gt = [g is not None for g in gt_3d]

    for j in range(NUM_JOINTS):
        for c, cname in COORD_NAMES.items():
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(frames, mp[:, j, c], "g-", label="MediaPipe", alpha=0.8)
            ax.plot(frames, opt[:, j, c], "r-", label="Optimised", alpha=0.8)
            if any(has_gt):
                gt_vals = [gt_3d[i][j, c] if gt_3d[i] is not None else np.nan for i in range(n)]
                ax.plot(frames, gt_vals, "b--", label="Ground Truth", alpha=0.7)
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{cname} (m)")
            ax.set_title(f"{JOINT_NAMES[j]} {cname}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            _save(fig, os.path.join(output_dir, f"{JOINT_NAMES[j]}_{cname}.png"))


def generate_loss_curve(loss_history: list[float], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(loss_history, "b-", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Optimisation Loss")
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "loss_curve.png"))


def generate_per_joint_error_bar(
    mp_per_joint: list[float],
    opt_per_joint: list[float],
    output_dir: str,
):
    """Bar chart comparing per-joint MPJPE: MediaPipe vs Optimised."""
    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(NUM_JOINTS)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, mp_per_joint, width, label="MediaPipe", color="green", alpha=0.7)
    ax.bar(x + width / 2, opt_per_joint, width, label="Optimised", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (m)")
    ax.set_title("Per-Joint Error vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "per_joint_error.png"))


def generate_per_frame_mpjpe(
    mp_per_frame: list[float],
    opt_per_frame: list[float],
    output_dir: str,
):
    """Line plot of per-frame MPJPE over time."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mp_per_frame, "g-", label="MediaPipe", alpha=0.8)
    ax.plot(opt_per_frame, "r-", label="Optimised", alpha=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("MPJPE (m)")
    ax.set_title("Per-Frame MPJPE vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "per_frame_mpjpe.png"))


def generate_per_joint_mpjve_bar(
    mp_per_joint: list[float],
    opt_per_joint: list[float],
    output_dir: str,
):
    """Bar chart comparing per-joint MPJVE: MediaPipe vs Optimised."""
    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(NUM_JOINTS)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, mp_per_joint, width, label="MediaPipe", color="green", alpha=0.7)
    ax.bar(x + width / 2, opt_per_joint, width, label="Optimised", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJVE (m/frame)")
    ax.set_title("Per-Joint Velocity Error vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "per_joint_mpjve.png"))


def generate_per_frame_mpjve(
    mp_per_frame: list[float],
    opt_per_frame: list[float],
    output_dir: str,
):
    """Line plot of per-frame MPJVE over time."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mp_per_frame, "g-", label="MediaPipe", alpha=0.8)
    ax.plot(opt_per_frame, "r-", label="Optimised", alpha=0.8)
    ax.set_xlabel("Frame Transition")
    ax.set_ylabel("MPJVE (m/frame)")
    ax.set_title("Per-Frame Velocity Error vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(output_dir, "per_frame_mpjve.png"))


def generate_bone_lengths_graph(
    bone_lengths: np.ndarray,
    output_dir: str,
):
    """Bar chart of final optimised bone lengths."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(1, NUM_JOINTS)
    ax.bar(x, bone_lengths[1:], color="steelblue")
    ax.set_xticks(x)
    labels = [f"{JOINT_NAMES[int(i)]}" for i in x]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Length (m)")
    ax.set_title("Optimised Bone Lengths")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "bone_lengths.png"))


# ---------------------------------------------------------------------------
# Summary grid (one per example)
# ---------------------------------------------------------------------------

def generate_summary(
    mediapipe_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    loss_history: list[float],
    metrics: dict,
    bone_lengths: np.ndarray,
    output_dir: str,
    title: str = "Summary",
):
    """Summary grid: 3 trajectory panels + loss + per-joint error + bone lengths."""
    os.makedirs(output_dir, exist_ok=True)

    n = len(mediapipe_3d)
    frames = list(range(n))
    mp = np.array(mediapipe_3d)
    opt = np.array(optimized_3d)
    has_gt = any(g is not None for g in gt_3d)

    fig, axes = plt.subplots(4, 4, figsize=(26, 18))

    # Row 0-1: select 6 key joints trajectories (Y coord = most visible)
    # Use first 6 cells across rows 0-1 (3 columns each)
    key_joints = [0, 8, 9, 10, 12, 3]  # Hip, Thorax, Neck, LShoulder, LWrist, RAnkle
    coord_colors = {0: "tab:red", 1: "tab:green", 2: "tab:blue"}  # X, Y, Z
    for idx, j in enumerate(key_joints):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        for c, cname in COORD_NAMES.items():
            color = coord_colors[c]
            ax.plot(frames, mp[:, j, c], linestyle="-", color=color,
                    alpha=0.4, linewidth=0.8)
            ax.plot(frames, opt[:, j, c], linestyle="--", color=color,
                    alpha=0.7, linewidth=0.8)
            if has_gt:
                gt_vals = [gt_3d[i][j, c] if gt_3d[i] is not None else np.nan
                           for i in range(n)]
                ax.plot(frames, gt_vals, linestyle=":", color=color,
                        alpha=0.5, linewidth=0.8)
        ax.set_title(JOINT_NAMES[j], fontsize=9)
        ax.grid(True, alpha=0.2)

    # Legend panel in row 0, col 3
    ax_legend = axes[0, 3]
    ax_legend.axis("off")
    # Line style legend
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="-", linewidth=1.5, label="MediaPipe"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.5, label="Optimised"),
    ]
    if has_gt:
        style_handles.append(
            Line2D([0], [0], color="gray", linestyle=":", linewidth=1.5, label="Ground Truth"))
    coord_handles = [
        Line2D([0], [0], color=coord_colors[0], linewidth=2, label="X"),
        Line2D([0], [0], color=coord_colors[1], linewidth=2, label="Y"),
        Line2D([0], [0], color=coord_colors[2], linewidth=2, label="Z"),
    ]
    ax_legend.legend(handles=style_handles + coord_handles, loc="center",
                     fontsize=10, frameon=True, title="Trajectories",
                     title_fontsize=11)

    # Row 1, Col 3: Per-frame MPJPE (if GT available)
    ax = axes[1, 3]
    if "mp_per_frame_mpjpe" in metrics:
        ax.plot(metrics["mp_per_frame_mpjpe"], "g-", label="MediaPipe", alpha=0.7, linewidth=0.8)
        ax.plot(metrics["opt_per_frame_mpjpe"], "r-", label="Optimised", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("MPJPE (m)", fontsize=7)
        ax.legend(fontsize=7)
    ax.set_title("Per-Frame MPJPE", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Row 2, Col 0: Loss curve
    ax = axes[2, 0]
    ax.plot(loss_history, "b-", linewidth=0.8, label="Loss")
    ax.set_title("Loss", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Row 2, Col 1: Per-joint MPJPE (if GT available)
    ax = axes[2, 1]
    if "mp_per_joint" in metrics:
        x = np.arange(NUM_JOINTS)
        ax.bar(x - 0.2, metrics["mp_per_joint"], 0.4, label="MediaPipe", color="green", alpha=0.6)
        ax.bar(x + 0.2, metrics["opt_per_joint"], 0.4, label="Optimised", color="red", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=5)
        ax.legend(fontsize=7)
    ax.set_title("Per-Joint MPJPE (m)", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Row 2, Col 2: Per-joint P-MPJPE (if GT available)
    ax = axes[2, 2]
    if "mp_p_per_joint" in metrics:
        x = np.arange(NUM_JOINTS)
        ax.bar(x - 0.2, metrics["mp_p_per_joint"], 0.4, label="MediaPipe", color="green", alpha=0.6)
        ax.bar(x + 0.2, metrics["opt_p_per_joint"], 0.4, label="Optimised", color="red", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=5)
        ax.legend(fontsize=7)
    ax.set_title("Per-Joint P-MPJPE (m)", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Row 2, Col 3: Bone lengths
    ax = axes[2, 3]
    x = np.arange(1, NUM_JOINTS)
    ax.bar(x, bone_lengths[1:], color="steelblue", alpha=0.7, label="Bone Length")
    ax.set_xticks(x)
    ax.set_xticklabels([JOINT_NAMES[i] for i in range(1, NUM_JOINTS)], rotation=90, fontsize=5)
    ax.set_title("Bone Lengths (m)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis="y")

    # Row 3, Col 0: Per-frame MPJVE (if GT available, >=2 frames)
    ax = axes[3, 0]
    if "mp_per_frame_mpjve" in metrics:
        ax.plot(metrics["mp_per_frame_mpjve"], "g-", label="MediaPipe", alpha=0.7, linewidth=0.8)
        ax.plot(metrics["opt_per_frame_mpjve"], "r-", label="Optimised", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("MPJVE (m/frame)", fontsize=7)
        ax.legend(fontsize=7)
    ax.set_title("Per-Frame MPJVE", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Row 3, Col 1: Per-joint MPJVE (if GT available, >=2 frames)
    ax = axes[3, 1]
    if "mp_mpjve_per_joint" in metrics:
        x = np.arange(NUM_JOINTS)
        ax.bar(x - 0.2, metrics["mp_mpjve_per_joint"], 0.4, label="MediaPipe", color="green", alpha=0.6)
        ax.bar(x + 0.2, metrics["opt_mpjve_per_joint"], 0.4, label="Optimised", color="red", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=5)
        ax.legend(fontsize=7)
    ax.set_title("Per-Joint MPJVE (m/frame)", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Row 3, Col 2-3: empty
    axes[3, 2].axis("off")
    axes[3, 3].axis("off")

    # Overall title with metrics
    metric_str = ""
    if "mp_mpjpe" in metrics:
        metric_str = (
            f"  |  MP MPJPE: {metrics['mp_mpjpe']*100:.1f}cm"
            f"  Opt MPJPE: {metrics['opt_mpjpe']*100:.1f}cm"
            f"  |  MP P-MPJPE: {metrics['mp_p_mpjpe']*100:.1f}cm"
            f"  Opt P-MPJPE: {metrics['opt_p_mpjpe']*100:.1f}cm"
        )
    if "mp_mpjve" in metrics:
        metric_str += (
            f"  |  MP MPJVE: {metrics['mp_mpjve']*100:.2f}cm/f"
            f"  Opt MPJVE: {metrics['opt_mpjve']*100:.2f}cm/f"
        )
    fig.suptitle(f"{title}{metric_str}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, os.path.join(output_dir, "summary.png"))


# ---------------------------------------------------------------------------
# Aggregate summary across all examples
# ---------------------------------------------------------------------------

def generate_aggregate_summary(all_metrics: list[dict], output_dir: str):
    """Summary across all examples."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter examples that have GT
    with_gt = [m for m in all_metrics if "mp_mpjpe" in m]
    if not with_gt:
        print("  No examples with ground truth — skipping aggregate summary.")
        return

    names = [m.get("name", f"Ex{i}") for i, m in enumerate(with_gt)]
    mp_mpjpe = [m["mp_mpjpe"] * 100 for m in with_gt]
    opt_mpjpe = [m["opt_mpjpe"] * 100 for m in with_gt]
    mp_p_mpjpe = [m["mp_p_mpjpe"] * 100 for m in with_gt]
    opt_p_mpjpe = [m["opt_p_mpjpe"] * 100 for m in with_gt]

    # MPJPE bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, mp_mpjpe, width, label="MediaPipe", color="green", alpha=0.7)
    ax.bar(x + width / 2, opt_mpjpe, width, label="Optimised", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("MPJPE Comparison Across Examples")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "aggregate_mpjpe.png"))

    # P-MPJPE bar chart
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax.bar(x - width / 2, mp_p_mpjpe, width, label="MediaPipe", color="green", alpha=0.7)
    ax.bar(x + width / 2, opt_p_mpjpe, width, label="Optimised", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P-MPJPE (cm)")
    ax.set_title("Procrustes-Aligned MPJPE Comparison Across Examples")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "aggregate_p_mpjpe.png"))

    # MPJVE bar chart (only examples with velocity data)
    with_mpjve = [m for m in with_gt if "mp_mpjve" in m]
    if with_mpjve:
        mpjve_names = [m.get("name", f"Ex{i}") for i, m in enumerate(with_mpjve)]
        mp_mpjve = [m["mp_mpjve"] * 100 for m in with_mpjve]
        opt_mpjve = [m["opt_mpjve"] * 100 for m in with_mpjve]
        xv = np.arange(len(mpjve_names))

        fig, ax = plt.subplots(figsize=(max(8, len(mpjve_names) * 1.2), 5))
        ax.bar(xv - width / 2, mp_mpjve, width, label="MediaPipe", color="green", alpha=0.7)
        ax.bar(xv + width / 2, opt_mpjve, width, label="Optimised", color="red", alpha=0.7)
        ax.set_xticks(xv)
        ax.set_xticklabels(mpjve_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("MPJVE (cm/frame)")
        ax.set_title("Velocity Error Comparison Across Examples")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, os.path.join(output_dir, "aggregate_mpjve.png"))

    # Print table
    print("\n  === Aggregate Results ===")
    print(f"  {'Example':<30} {'MP MPJPE':>9} {'Opt MPJPE':>10} {'Improv':>7} {'MP P-MPJPE':>11} {'Opt P-MPJPE':>12} {'Improv':>7} {'MP MPJVE':>9} {'Opt MPJVE':>10} {'Improv':>7}")
    print(f"  {'-'*30} {'-'*9} {'-'*10} {'-'*7} {'-'*11} {'-'*12} {'-'*7} {'-'*9} {'-'*10} {'-'*7}")
    for i, name in enumerate(names):
        diff = mp_mpjpe[i] - opt_mpjpe[i]
        p_diff = mp_p_mpjpe[i] - opt_p_mpjpe[i]
        mp_v = with_gt[i].get("mp_mpjve", 0) * 100
        opt_v = with_gt[i].get("opt_mpjve", 0) * 100
        v_diff = mp_v - opt_v
        print(f"  {name:<30} {mp_mpjpe[i]:>8.2f} {opt_mpjpe[i]:>9.2f} {diff:>+6.2f} {mp_p_mpjpe[i]:>10.2f} {opt_p_mpjpe[i]:>11.2f} {p_diff:>+6.2f} {mp_v:>8.2f} {opt_v:>9.2f} {v_diff:>+6.2f}")
    mean_mp = np.mean(mp_mpjpe)
    mean_opt = np.mean(opt_mpjpe)
    mean_mp_p = np.mean(mp_p_mpjpe)
    mean_opt_p = np.mean(opt_p_mpjpe)
    mean_mp_v = np.mean([m.get("mp_mpjve", 0) * 100 for m in with_gt])
    mean_opt_v = np.mean([m.get("opt_mpjve", 0) * 100 for m in with_gt])
    print(f"  {'MEAN':<30} {mean_mp:>8.2f} {mean_opt:>9.2f} {mean_mp - mean_opt:>+6.2f} {mean_mp_p:>10.2f} {mean_opt_p:>11.2f} {mean_mp_p - mean_opt_p:>+6.2f} {mean_mp_v:>8.2f} {mean_opt_v:>9.2f} {mean_mp_v - mean_opt_v:>+6.2f}")
