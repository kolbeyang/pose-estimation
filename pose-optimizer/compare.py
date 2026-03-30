"""Cross-pipeline comparison: MediaPipe vs MotionBert.

Usage:
    uv run python compare.py <mediapipe_output_dir> <motionbert_output_dir>

Produces a comparison/ directory with PNG graphs and summary.json.
"""

import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir for skeleton imports
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from skeleton import EVAL_JOINT_NAMES, NUM_EVAL_JOINTS


def load_results(output_dir: str) -> dict[str, dict]:
    """Load all results.json from an output directory, keyed by example name."""
    results = {}
    for entry in sorted(os.listdir(output_dir)):
        rpath = os.path.join(output_dir, entry, "results.json")
        if os.path.isfile(rpath):
            with open(rpath) as f:
                data = json.load(f)
            results[data["example"]] = data
    return results


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_comparison(mp_dir: str, mb_dir: str) -> None:
    mp_results = load_results(mp_dir)
    mb_results = load_results(mb_dir)

    # Match examples present in both
    common = sorted(set(mp_results) & set(mb_results))
    if not common:
        print("ERROR: No common examples found between the two directories.")
        sys.exit(1)

    print(f"Found {len(common)} common examples.")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    out_dir = os.path.join("output", f"comparison_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Extract metric arrays
    def get_metric(results, name, metric_key, raw=False):
        src = "raw_metrics" if raw else "metrics"
        return results[name].get(src, {}).get(metric_key, None)

    # --- Graph 1: Per-Example VW-SI-MPJPE Bar Chart ---
    _per_example_bar(common, mp_results, mb_results, "vw_si_mpjpe",
                     "VW-SI-MPJPE", out_dir)

    # --- Graph 2: Per-Example SI-MPJPE Bar Chart ---
    _per_example_bar(common, mp_results, mb_results, "si_mpjpe",
                     "SI-MPJPE", out_dir)

    # --- Graph 3: Aggregate Metrics Table ---
    _aggregate_metrics(common, mp_results, mb_results, out_dir)

    # --- Graph 4: Box Plot ---
    _box_plot(common, mp_results, mb_results, out_dir)

    # --- Graph 5: Scatter Plot ---
    _scatter_plot(common, mp_results, mb_results, out_dir)

    # --- Graph 6: Per-Joint Error Comparison ---
    _per_joint_comparison(common, mp_results, mb_results, out_dir)

    # --- Graph 7: Velocity Error Comparison ---
    _per_example_bar(common, mp_results, mb_results, "vw_si_mpjve",
                     "VW-SI-MPJVE (Velocity)", out_dir, filename="velocity_comparison.png")

    # --- Graph 8: Win/Loss Summary ---
    _win_loss(common, mp_results, mb_results, out_dir)

    # --- Graph 9: Improvement from optimization ---
    _optimization_improvement(common, mp_results, mb_results, out_dir)

    # --- Summary JSON ---
    summary = _build_summary(common, mp_results, mb_results, mp_dir, mb_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print text summary ---
    _print_summary(summary, out_dir)


def _per_example_bar(common, mp_results, mb_results, metric_key, label, out_dir,
                     filename=None):
    """Grouped bar chart, sorted by MotionBert value."""
    mp_vals = []
    mb_vals = []
    names = []
    for name in common:
        mp_v = mp_results[name].get("metrics", {}).get(metric_key)
        mb_v = mb_results[name].get("metrics", {}).get(metric_key)
        if mp_v is not None and mb_v is not None:
            mp_vals.append(mp_v * 100)
            mb_vals.append(mb_v * 100)
            names.append(name)

    # Sort by MotionBert value
    order = np.argsort(mb_vals)
    mp_vals = [mp_vals[i] for i in order]
    mb_vals = [mb_vals[i] for i in order]
    names = [names[i] for i in order]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.7), 6))
    ax.bar(x - width/2, mb_vals, width, label="MotionBert", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width/2, mp_vals, width, label="MediaPipe", color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"{label} (cm)")
    ax.set_title(f"Per-Example {label} (Optimized)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fname = filename or f"per_example_{metric_key}.png"
    _save(fig, os.path.join(out_dir, fname))


def _aggregate_metrics(common, mp_results, mb_results, out_dir):
    """Bar chart of mean metrics across all examples."""
    metric_keys = [
        "mpjpe", "p_mpjpe", "si_mpjpe", "vw_mpjpe", "vw_si_mpjpe",
        "mpjve", "si_mpjve", "vw_mpjve", "vw_si_mpjve",
    ]
    labels = []
    mp_means = []
    mb_means = []

    for key in metric_keys:
        mp_vals = [mp_results[n]["metrics"].get(key) for n in common
                   if mp_results[n]["metrics"].get(key) is not None]
        mb_vals = [mb_results[n]["metrics"].get(key) for n in common
                   if mb_results[n]["metrics"].get(key) is not None]
        if mp_vals and mb_vals:
            labels.append(key.upper())
            mp_means.append(np.mean(mp_vals) * 100)
            mb_means.append(np.mean(mb_vals) * 100)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_mb = ax.bar(x - width/2, mb_means, width, label="MotionBert", color="#ff7f0e", alpha=0.8)
    bars_mp = ax.bar(x + width/2, mp_means, width, label="MediaPipe", color="#1f77b4", alpha=0.8)

    # Add value labels on bars
    for bars in [bars_mb, bars_mp]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Error (cm)")
    ax.set_title("Aggregate Metrics (Mean over all examples, Optimized)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(out_dir, "aggregate_metrics.png"))


def _box_plot(common, mp_results, mb_results, out_dir):
    """Box plot of VW-SI-MPJPE distributions."""
    mp_raw = [mp_results[n]["raw_metrics"].get("vw_si_mpjpe", 0) * 100 for n in common]
    mp_opt = [mp_results[n]["metrics"].get("vw_si_mpjpe", 0) * 100 for n in common]
    mb_raw = [mb_results[n]["raw_metrics"].get("vw_si_mpjpe", 0) * 100 for n in common]
    mb_opt = [mb_results[n]["metrics"].get("vw_si_mpjpe", 0) * 100 for n in common]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        [mb_raw, mb_opt, mp_raw, mp_opt],
        tick_labels=["MB Raw", "MB Optimized", "MP Raw", "MP Optimized"],
        patch_artist=True,
        widths=0.5,
    )
    colors = ["#ffcc80", "#ff7f0e", "#90caf9", "#1f77b4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("VW-SI-MPJPE (cm)")
    ax.set_title("Error Distribution: Raw vs Optimized")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(out_dir, "error_distribution.png"))


def _scatter_plot(common, mp_results, mb_results, out_dir):
    """Scatter: x=MediaPipe, y=MotionBert, with y=x line."""
    mp_vals = []
    mb_vals = []
    names = []
    for n in common:
        mp_v = mp_results[n]["metrics"].get("vw_si_mpjpe")
        mb_v = mb_results[n]["metrics"].get("vw_si_mpjpe")
        if mp_v is not None and mb_v is not None:
            mp_vals.append(mp_v * 100)
            mb_vals.append(mb_v * 100)
            names.append(n)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mp_vals, mb_vals, s=40, alpha=0.7, zorder=5)

    # y=x line
    lo = min(min(mp_vals), min(mb_vals)) * 0.9
    hi = max(max(mp_vals), max(mb_vals)) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")

    # Annotate points
    for i, name in enumerate(names):
        short = name.split("_")[-1]  # just the frame number
        ax.annotate(short, (mp_vals[i], mb_vals[i]), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("MediaPipe VW-SI-MPJPE (cm)")
    ax.set_ylabel("MotionBert VW-SI-MPJPE (cm)")
    ax.set_title("Per-Example Comparison (below line = MotionBert wins)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "scatter_mpjpe.png"))


def _per_joint_comparison(common, mp_results, mb_results, out_dir):
    """Per-joint MPJPE averaged across all examples."""
    mp_joints = []
    mb_joints = []
    for n in common:
        mp_pj = mp_results[n].get("per_joint", {}).get("opt_per_joint", [])
        mb_pj = mb_results[n].get("per_joint", {}).get("opt_per_joint", [])
        if len(mp_pj) == NUM_EVAL_JOINTS and len(mb_pj) == NUM_EVAL_JOINTS:
            mp_joints.append(mp_pj)
            mb_joints.append(mb_pj)

    if not mp_joints:
        print("WARNING: No per-joint data available, skipping per-joint graph.")
        return

    mp_mean = np.mean(mp_joints, axis=0) * 100
    mb_mean = np.mean(mb_joints, axis=0) * 100

    x = np.arange(NUM_EVAL_JOINTS)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, mb_mean, width, label="MotionBert", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width/2, mp_mean, width, label="MediaPipe", color="#1f77b4", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(EVAL_JOINT_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("Per-Joint Error (Mean across all examples, Optimized)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(out_dir, "per_joint_comparison.png"))


def _win_loss(common, mp_results, mb_results, out_dir):
    """Win/loss bar chart across multiple metrics."""
    metrics_to_check = ["vw_si_mpjpe", "si_mpjpe", "mpjpe", "vw_si_mpjve"]
    metric_labels = ["VW-SI-MPJPE", "SI-MPJPE", "MPJPE", "VW-SI-MPJVE"]

    mp_wins = []
    mb_wins = []
    ties = []

    for key in metrics_to_check:
        mw, bw, t = 0, 0, 0
        for n in common:
            mp_v = mp_results[n]["metrics"].get(key)
            mb_v = mb_results[n]["metrics"].get(key)
            if mp_v is not None and mb_v is not None:
                if mp_v < mb_v:
                    mw += 1
                elif mb_v < mp_v:
                    bw += 1
                else:
                    t += 1
        mp_wins.append(mw)
        mb_wins.append(bw)
        ties.append(t)

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, mb_wins, width, label="MotionBert wins", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width/2, mp_wins, width, label="MediaPipe wins", color="#1f77b4", alpha=0.8)

    # Add count labels
    for i in range(len(metric_labels)):
        ax.text(x[i] - width/2, mb_wins[i] + 0.3, str(mb_wins[i]), ha="center", fontsize=10, fontweight="bold")
        ax.text(x[i] + width/2, mp_wins[i] + 0.3, str(mp_wins[i]), ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Number of examples won")
    ax.set_title(f"Win/Loss Summary ({len(common)} examples)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(out_dir, "win_loss.png"))


def _optimization_improvement(common, mp_results, mb_results, out_dir):
    """Show how much optimization improves each pipeline."""
    mp_raw = []
    mp_opt = []
    mb_raw = []
    mb_opt = []
    names = []

    for n in common:
        mr = mp_results[n]["raw_metrics"].get("vw_si_mpjpe")
        mo = mp_results[n]["metrics"].get("vw_si_mpjpe")
        br = mb_results[n]["raw_metrics"].get("vw_si_mpjpe")
        bo = mb_results[n]["metrics"].get("vw_si_mpjpe")
        if all(v is not None for v in [mr, mo, br, bo]):
            mp_raw.append(mr * 100)
            mp_opt.append(mo * 100)
            mb_raw.append(br * 100)
            mb_opt.append(bo * 100)
            names.append(n)

    mp_improvement = [r - o for r, o in zip(mp_raw, mp_opt)]
    mb_improvement = [r - o for r, o in zip(mb_raw, mb_opt)]

    # Sort by MP improvement
    order = np.argsort(mp_improvement)[::-1]
    mp_improvement = [mp_improvement[i] for i in order]
    mb_improvement = [mb_improvement[i] for i in order]
    names = [names[i] for i in order]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.7), 6))
    ax.bar(x - width/2, mb_improvement, width, label="MotionBert", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width/2, mp_improvement, width, label="MediaPipe", color="#1f77b4", alpha=0.8)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Improvement (cm, positive = optimization helped)")
    ax.set_title("Optimization Improvement (Raw - Optimized VW-SI-MPJPE)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(out_dir, "optimization_improvement.png"))


def _build_summary(common, mp_results, mb_results, mp_dir, mb_dir):
    """Build summary dict."""
    metric_key = "vw_si_mpjpe"
    mp_vals = [mp_results[n]["metrics"].get(metric_key, 0) for n in common]
    mb_vals = [mb_results[n]["metrics"].get(metric_key, 0) for n in common]

    mp_wins = sum(1 for m, b in zip(mp_vals, mb_vals) if m < b)
    mb_wins = sum(1 for m, b in zip(mp_vals, mb_vals) if b < m)

    per_example = {}
    for n in common:
        per_example[n] = {
            "mediapipe": mp_results[n].get("metrics", {}),
            "motionbert": mb_results[n].get("metrics", {}),
        }

    # Aggregate
    metric_keys = ["mpjpe", "p_mpjpe", "si_mpjpe", "vw_mpjpe", "vw_si_mpjpe",
                   "mpjve", "si_mpjve", "vw_mpjve", "vw_si_mpjve"]
    aggregate = {}
    for key in metric_keys:
        mp_v = [mp_results[n]["metrics"].get(key) for n in common
                if mp_results[n]["metrics"].get(key) is not None]
        mb_v = [mb_results[n]["metrics"].get(key) for n in common
                if mb_results[n]["metrics"].get(key) is not None]
        if mp_v and mb_v:
            aggregate[key] = {
                "mediapipe_mean": float(np.mean(mp_v)),
                "mediapipe_median": float(np.median(mp_v)),
                "mediapipe_std": float(np.std(mp_v)),
                "motionbert_mean": float(np.mean(mb_v)),
                "motionbert_median": float(np.median(mb_v)),
                "motionbert_std": float(np.std(mb_v)),
            }

    return {
        "mediapipe_dir": mp_dir,
        "motionbert_dir": mb_dir,
        "num_examples": len(common),
        "mediapipe_mean_vw_si_mpjpe": float(np.mean(mp_vals)),
        "motionbert_mean_vw_si_mpjpe": float(np.mean(mb_vals)),
        "mediapipe_wins": mp_wins,
        "motionbert_wins": mb_wins,
        "per_example": per_example,
        "aggregate": aggregate,
    }


def _print_summary(summary, out_dir):
    """Print text summary to stdout."""
    print(f"\n{'='*60}")
    print("  Pipeline Comparison: MediaPipe vs MotionBert")
    print(f"{'='*60}")
    print(f"  Examples compared: {summary['num_examples']}")

    agg = summary["aggregate"]
    if "vw_si_mpjpe" in agg:
        d = agg["vw_si_mpjpe"]
        print(f"\n  Primary Metric (VW-SI-MPJPE, cm):")
        print(f"    MediaPipe:  mean={d['mediapipe_mean']*100:.1f}  median={d['mediapipe_median']*100:.1f}  std={d['mediapipe_std']*100:.1f}")
        print(f"    MotionBert: mean={d['motionbert_mean']*100:.1f}  median={d['motionbert_median']*100:.1f}  std={d['motionbert_std']*100:.1f}")

    if "si_mpjpe" in agg:
        d = agg["si_mpjpe"]
        print(f"\n  Secondary Metric (SI-MPJPE, cm):")
        print(f"    MediaPipe:  mean={d['mediapipe_mean']*100:.1f}  median={d['mediapipe_median']*100:.1f}  std={d['mediapipe_std']*100:.1f}")
        print(f"    MotionBert: mean={d['motionbert_mean']*100:.1f}  median={d['motionbert_median']*100:.1f}  std={d['motionbert_std']*100:.1f}")

    if "vw_si_mpjve" in agg:
        d = agg["vw_si_mpjve"]
        print(f"\n  Velocity Metric (VW-SI-MPJVE, cm/frame):")
        print(f"    MediaPipe:  mean={d['mediapipe_mean']*100:.1f}  median={d['mediapipe_median']*100:.1f}  std={d['mediapipe_std']*100:.1f}")
        print(f"    MotionBert: mean={d['motionbert_mean']*100:.1f}  median={d['motionbert_median']*100:.1f}  std={d['motionbert_std']*100:.1f}")

    print(f"\n  Winner by VW-SI-MPJPE: MediaPipe wins {summary['mediapipe_wins']}/{summary['num_examples']}, "
          f"MotionBert wins {summary['motionbert_wins']}/{summary['num_examples']}")
    print(f"\n  Graphs saved to: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <mediapipe_output_dir> <motionbert_output_dir>")
        sys.exit(1)
    make_comparison(sys.argv[1], sys.argv[2])
