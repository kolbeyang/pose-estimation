"""Analyze whether the optimizer improves MotionBert and MediaPipe.

Loads all results.json from both pipeline output directories and produces
graphs comparing detector-only (raw) vs optimized metrics.

Usage:
    uv run python experiment/optimization_benefit.py <mediapipe_dir> <motionbert_dir>

Example:
    uv run python experiment/optimization_benefit.py \
        output/mediapipe_2026_03_30_19_15 output/motionbert_2026_03_30_18_57
"""

import json
import os
import sys

# Add parent dir so we can import skeleton
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Data loading ──────────────────────────────────────────────────────────

def load_results(output_dir: str) -> dict[str, dict]:
    """Load all results.json from an output directory. Returns {example_name: data}."""
    results = {}
    for entry in sorted(os.listdir(output_dir)):
        rpath = os.path.join(output_dir, entry, "results.json")
        if os.path.isfile(rpath):
            with open(rpath) as f:
                results[entry] = json.load(f)
    return results


def to_cm(val):
    return val * 100


# ── Graphs ────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "vw_si_mpjpe": "VW-SI-MPJPE (cm)",
    "si_mpjpe": "SI-MPJPE (cm)",
    "mpjpe": "MPJPE (cm)",
    "p_mpjpe": "P-MPJPE (cm)",
    "vw_si_mpjve": "VW-SI-MPJVE (cm/frame)",
    "si_mpjve": "SI-MPJVE (cm/frame)",
    "mpjve": "MPJVE (cm/frame)",
}

POSITION_METRICS = ["vw_si_mpjpe", "si_mpjpe", "mpjpe", "p_mpjpe"]
VELOCITY_METRICS = ["vw_si_mpjve", "si_mpjve", "mpjve"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_before_after_bars(mp_results, mb_results, common, metric, out_dir):
    """Side-by-side before/after bars for each example, one pipeline at a time."""
    label = METRIC_LABELS.get(metric, metric)

    for pipeline, results, color_det, color_opt in [
        ("MediaPipe", mp_results, "#aec7e8", "#1f77b4"),
        ("MotionBert", mb_results, "#ffbb78", "#ff7f0e"),
    ]:
        det_vals = [to_cm(results[n]["raw_metrics"][metric]) for n in common]
        opt_vals = [to_cm(results[n]["metrics"][metric]) for n in common]

        # Sort by detector value (worst first)
        order = np.argsort(det_vals)[::-1]
        det_vals = [det_vals[i] for i in order]
        opt_vals = [opt_vals[i] for i in order]
        names = [common[i] for i in order]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.7), 6))
        ax.bar(x - width / 2, det_vals, width, label="Detector (raw)", color=color_det)
        ax.bar(x + width / 2, opt_vals, width, label="Optimized", color=color_opt)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(f"{pipeline}: Detector vs Optimized — {label}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fname = f"{pipeline.lower()}_{metric}_before_after.png"
        _save(fig, os.path.join(out_dir, fname))


def plot_aggregate_improvement(mp_results, mb_results, common, out_dir):
    """Bar chart: average % improvement per metric for each pipeline."""
    metrics = POSITION_METRICS + VELOCITY_METRICS
    mp_pct = []
    mb_pct = []

    for m in metrics:
        mp_det = np.mean([mp_results[n]["raw_metrics"][m] for n in common])
        mp_opt = np.mean([mp_results[n]["metrics"][m] for n in common])
        mb_det = np.mean([mb_results[n]["raw_metrics"][m] for n in common])
        mb_opt = np.mean([mb_results[n]["metrics"][m] for n in common])
        mp_pct.append((mp_det - mp_opt) / mp_det * 100 if mp_det > 0 else 0)
        mb_pct.append((mb_det - mb_opt) / mb_det * 100 if mb_det > 0 else 0)

    x = np.arange(len(metrics))
    width = 0.35
    labels = [METRIC_LABELS.get(m, m).split(" (")[0] for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, mb_pct, width, label="MotionBert", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width / 2, mp_pct, width, label="MediaPipe", color="#1f77b4", alpha=0.8)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Optimization Benefit: Average % Improvement by Metric")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(out_dir, "aggregate_pct_improvement.png"))


def plot_improvement_distribution(mp_results, mb_results, common, metric, out_dir):
    """Box plot of per-example improvement for both pipelines."""
    label = METRIC_LABELS.get(metric, metric)

    mp_impr = [to_cm(mp_results[n]["raw_metrics"][metric] - mp_results[n]["metrics"][metric]) for n in common]
    mb_impr = [to_cm(mb_results[n]["raw_metrics"][metric] - mb_results[n]["metrics"][metric]) for n in common]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot([mb_impr, mp_impr], tick_labels=["MotionBert", "MediaPipe"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#ff7f0e")
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor("#1f77b4")
    bp["boxes"][1].set_alpha(0.6)
    ax.axhline(y=0, color="red", linewidth=1, linestyle="--", label="No change")
    ax.set_ylabel(f"Improvement in {label} (positive = optimizer helped)")
    ax.set_title(f"Distribution of Optimization Benefit — {label}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(out_dir, f"improvement_distribution_{metric}.png"))


def plot_per_joint_improvement(mp_results, mb_results, common, out_dir):
    """Per-joint: how much does the optimizer help each joint?"""
    from skeleton import EVAL_JOINT_NAMES

    # det_per_joint and opt_per_joint are lists of 14 values (EVAL_JOINTS)
    mp_det_joints = np.zeros(len(EVAL_JOINT_NAMES))
    mp_opt_joints = np.zeros(len(EVAL_JOINT_NAMES))
    mb_det_joints = np.zeros(len(EVAL_JOINT_NAMES))
    mb_opt_joints = np.zeros(len(EVAL_JOINT_NAMES))
    count = 0

    for n in common:
        mp_d = mp_results[n]["per_joint"].get("det_per_joint")
        mp_o = mp_results[n]["per_joint"].get("opt_per_joint")
        mb_d = mb_results[n]["per_joint"].get("det_per_joint")
        mb_o = mb_results[n]["per_joint"].get("opt_per_joint")
        if mp_d and mp_o and mb_d and mb_o:
            mp_det_joints += np.array(mp_d)
            mp_opt_joints += np.array(mp_o)
            mb_det_joints += np.array(mb_d)
            mb_opt_joints += np.array(mb_o)
            count += 1

    if count == 0:
        return

    mp_det_joints = mp_det_joints / count * 100
    mp_opt_joints = mp_opt_joints / count * 100
    mb_det_joints = mb_det_joints / count * 100
    mb_opt_joints = mb_opt_joints / count * 100

    mp_impr = mp_det_joints - mp_opt_joints
    mb_impr = mb_det_joints - mb_opt_joints

    x = np.arange(len(EVAL_JOINT_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, mb_impr, width, label="MotionBert", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width / 2, mp_impr, width, label="MediaPipe", color="#1f77b4", alpha=0.8)
    ax.axhline(y=0, color="red", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Improvement (cm, positive = optimizer helped)")
    ax.set_title("Per-Joint Optimization Benefit (Raw MPJPE, averaged over 25 examples)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(out_dir, "per_joint_improvement.png"))


def plot_scatter_det_vs_opt(mp_results, mb_results, common, metric, out_dir):
    """Scatter: detector error (x) vs optimized error (y). Below y=x = improvement."""
    label = METRIC_LABELS.get(metric, metric)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pipeline, results, color in [
        (axes[0], "MediaPipe", mp_results, "#1f77b4"),
        (axes[1], "MotionBert", mb_results, "#ff7f0e"),
    ]:
        det = [to_cm(results[n]["raw_metrics"][metric]) for n in common]
        opt = [to_cm(results[n]["metrics"][metric]) for n in common]

        ax.scatter(det, opt, c=color, alpha=0.7, s=50, edgecolors="k", linewidths=0.5)
        lims = [0, max(max(det), max(opt)) * 1.1]
        ax.plot(lims, lims, "k--", alpha=0.5, label="No change (y=x)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Detector {label}")
        ax.set_ylabel(f"Optimized {label}")
        ax.set_title(f"{pipeline}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Count improved
        improved = sum(1 for d, o in zip(det, opt) if o < d)
        ax.text(0.05, 0.95, f"Improved: {improved}/{len(common)}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle(f"Detector vs Optimized — {label}", fontsize=13)
    _save(fig, os.path.join(out_dir, f"scatter_det_vs_opt_{metric}.png"))


def plot_summary_table(mp_results, mb_results, common, out_dir):
    """Render a summary table as a figure."""
    metrics = POSITION_METRICS + VELOCITY_METRICS

    rows = []
    for m in metrics:
        label = METRIC_LABELS.get(m, m).split(" (")[0]
        mp_det = np.mean([mp_results[n]["raw_metrics"][m] for n in common]) * 100
        mp_opt = np.mean([mp_results[n]["metrics"][m] for n in common]) * 100
        mb_det = np.mean([mb_results[n]["raw_metrics"][m] for n in common]) * 100
        mb_opt = np.mean([mb_results[n]["metrics"][m] for n in common]) * 100
        mp_pct = (mp_det - mp_opt) / mp_det * 100 if mp_det > 0 else 0
        mb_pct = (mb_det - mb_opt) / mb_det * 100 if mb_det > 0 else 0
        rows.append([label,
                     f"{mp_det:.1f}", f"{mp_opt:.1f}", f"{mp_pct:+.1f}%",
                     f"{mb_det:.1f}", f"{mb_opt:.1f}", f"{mb_pct:+.1f}%"])

    col_labels = ["Metric",
                  "MP Det", "MP Opt", "MP Δ%",
                  "MB Det", "MB Opt", "MB Δ%"]

    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.5 + 1.5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color the improvement columns
    for i, row in enumerate(rows):
        for col_idx in [3, 6]:  # MP Δ% and MB Δ%
            val = float(row[col_idx].replace("%", "").replace("+", ""))
            cell = table[i + 1, col_idx]
            if val > 0:
                cell.set_facecolor("#d4edda")  # green
            elif val < 0:
                cell.set_facecolor("#f8d7da")  # red

    ax.set_title("Optimization Benefit Summary (all values in cm, averaged over 25 examples)",
                 fontsize=12, pad=20)
    _save(fig, os.path.join(out_dir, "summary_table.png"))

    return rows, col_labels


def write_summary_json(mp_results, mb_results, common, out_dir):
    """Write machine-readable summary."""
    metrics = POSITION_METRICS + VELOCITY_METRICS
    summary = {"num_examples": len(common), "examples": common, "metrics": {}}

    for m in metrics:
        mp_det = [mp_results[n]["raw_metrics"][m] for n in common]
        mp_opt = [mp_results[n]["metrics"][m] for n in common]
        mb_det = [mb_results[n]["raw_metrics"][m] for n in common]
        mb_opt = [mb_results[n]["metrics"][m] for n in common]

        mp_improved = sum(1 for d, o in zip(mp_det, mp_opt) if o < d)
        mb_improved = sum(1 for d, o in zip(mb_det, mb_opt) if o < d)

        summary["metrics"][m] = {
            "mediapipe_det_mean_cm": float(np.mean(mp_det)) * 100,
            "mediapipe_opt_mean_cm": float(np.mean(mp_opt)) * 100,
            "mediapipe_pct_improvement": float((np.mean(mp_det) - np.mean(mp_opt)) / np.mean(mp_det) * 100) if np.mean(mp_det) > 0 else 0,
            "mediapipe_examples_improved": mp_improved,
            "motionbert_det_mean_cm": float(np.mean(mb_det)) * 100,
            "motionbert_opt_mean_cm": float(np.mean(mb_opt)) * 100,
            "motionbert_pct_improvement": float((np.mean(mb_det) - np.mean(mb_opt)) / np.mean(mb_det) * 100) if np.mean(mb_det) > 0 else 0,
            "motionbert_examples_improved": mb_improved,
        }

    # Build conclusion
    vw = summary["metrics"]["vw_si_mpjpe"]
    vel = summary["metrics"]["vw_si_mpjve"]
    summary["conclusion"] = (
        f"Optimizer improves MediaPipe position accuracy by {vw['mediapipe_pct_improvement']:.1f}% "
        f"({vw['mediapipe_examples_improved']}/{len(common)} examples improved) "
        f"and MotionBert by {vw['motionbert_pct_improvement']:.1f}% "
        f"({vw['motionbert_examples_improved']}/{len(common)} examples improved) on VW-SI-MPJPE. "
        f"Velocity: MediaPipe {vel['mediapipe_pct_improvement']:+.1f}%, "
        f"MotionBert {vel['motionbert_pct_improvement']:+.1f}%."
    )

    path = os.path.join(out_dir, "optimization_benefit_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {path}")
    print(f"\n  Conclusion: {summary['conclusion']}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <mediapipe_output_dir> <motionbert_output_dir>")
        sys.exit(1)

    mp_dir, mb_dir = sys.argv[1], sys.argv[2]
    mp_results = load_results(mp_dir)
    mb_results = load_results(mb_dir)
    common = sorted(set(mp_results) & set(mb_results))
    print(f"Loaded {len(common)} common examples")

    # Output directory
    from datetime import datetime
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = os.path.join("output", f"optimization_benefit_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # Generate all graphs
    print("\nGenerating graphs...")

    # 1. Before/after bars for primary metrics
    for metric in ["vw_si_mpjpe", "vw_si_mpjve"]:
        plot_before_after_bars(mp_results, mb_results, common, metric, out_dir)

    # 2. Aggregate % improvement across all metrics
    plot_aggregate_improvement(mp_results, mb_results, common, out_dir)

    # 3. Distribution of improvement (box plots)
    plot_improvement_distribution(mp_results, mb_results, common, "vw_si_mpjpe", out_dir)
    plot_improvement_distribution(mp_results, mb_results, common, "vw_si_mpjve", out_dir)

    # 4. Per-joint improvement
    plot_per_joint_improvement(mp_results, mb_results, common, out_dir)

    # 5. Scatter: detector vs optimized
    plot_scatter_det_vs_opt(mp_results, mb_results, common, "vw_si_mpjpe", out_dir)
    plot_scatter_det_vs_opt(mp_results, mb_results, common, "vw_si_mpjve", out_dir)

    # 6. Summary table
    plot_summary_table(mp_results, mb_results, common, out_dir)

    # 7. Summary JSON
    write_summary_json(mp_results, mb_results, common, out_dir)

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
