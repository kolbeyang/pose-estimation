"""Cross-pipeline comparison: MediaPipe lite vs heavy (motionbert-pose).

Reads prediction JSONs from both pipelines, computes per-axis errors,
and generates comparison graphs.

Usage:
    cd comparison && uv run python compare_pipelines.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

LITE_PREDICTIONS = os.path.join(_REPO_ROOT, "mediapipe-pose", "results", "predictions")
HEAVY_PREDICTIONS = os.path.join(_REPO_ROOT, "motionbert-pose", "results", "predictions")
OUTPUT_DIR = os.path.join(_THIS_DIR, "results")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_predictions(predictions_dir: str) -> dict[str, dict]:
    """Load all prediction JSONs from a directory, keyed by example name."""
    results = {}
    if not os.path.isdir(predictions_dir):
        return results
    for fname in sorted(os.listdir(predictions_dir)):
        if not fname.endswith(".json"):
            continue
        name = fname.replace(".json", "")
        with open(os.path.join(predictions_dir, fname)) as f:
            results[name] = json.load(f)
    return results


def compute_per_axis_mpjpe(frames: list[dict]) -> dict[str, np.ndarray]:
    """Compute per-axis (X, Y, Z) MPJPE from frame data.

    Returns dict with keys: mp_per_axis (3,), opt_per_axis (3,)
    """
    mp_errors_x, mp_errors_y, mp_errors_z = [], [], []
    opt_errors_x, opt_errors_y, opt_errors_z = [], [], []

    for frame in frames:
        gt = frame.get("ground_truth_3d")
        if gt is None:
            continue

        gt = np.array(gt)
        mp = np.array(frame["mediapipe_3d"])
        opt = np.array(frame["optimized_3d"])

        # Root-relative
        gt_rr = gt - gt[0:1]
        mp_rr = mp - mp[0:1]
        opt_rr = opt - opt[0:1]

        mp_diff = np.abs(mp_rr - gt_rr)
        opt_diff = np.abs(opt_rr - gt_rr)

        mp_errors_x.append(np.mean(mp_diff[:, 0]))
        mp_errors_y.append(np.mean(mp_diff[:, 1]))
        mp_errors_z.append(np.mean(mp_diff[:, 2]))
        opt_errors_x.append(np.mean(opt_diff[:, 0]))
        opt_errors_y.append(np.mean(opt_diff[:, 1]))
        opt_errors_z.append(np.mean(opt_diff[:, 2]))

    if not mp_errors_x:
        return {}

    return {
        "mp_per_axis": np.array([
            np.mean(mp_errors_x), np.mean(mp_errors_y), np.mean(mp_errors_z),
        ]),
        "opt_per_axis": np.array([
            np.mean(opt_errors_x), np.mean(opt_errors_y), np.mean(opt_errors_z),
        ]),
    }


def main():
    print("=" * 60)
    print("  Cross-Pipeline Comparison: MP-Lite vs MP-Heavy")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lite_preds = load_predictions(LITE_PREDICTIONS)
    heavy_preds = load_predictions(HEAVY_PREDICTIONS)

    if not lite_preds:
        print(f"  ERROR: No lite predictions found in {LITE_PREDICTIONS}")
        sys.exit(1)
    if not heavy_preds:
        print(f"  ERROR: No heavy predictions found in {HEAVY_PREDICTIONS}")
        sys.exit(1)

    # Find matching examples
    common = sorted(set(lite_preds.keys()) & set(heavy_preds.keys()))
    print(f"\n  Found {len(common)} matching examples")

    if not common:
        print("  No matching examples. Exiting.")
        sys.exit(1)

    # Collect metrics
    names = []
    lite_init_mpjpe = []
    lite_opt_mpjpe = []
    heavy_init_mpjpe = []
    heavy_opt_mpjpe = []
    lite_init_p_mpjpe = []
    lite_opt_p_mpjpe = []
    heavy_init_p_mpjpe = []
    heavy_opt_p_mpjpe = []

    # Per-axis aggregators
    lite_mp_axes_all = []
    lite_opt_axes_all = []
    heavy_mp_axes_all = []
    heavy_opt_axes_all = []

    for name in common:
        lite_m = lite_preds[name].get("metrics", {})
        heavy_m = heavy_preds[name].get("metrics", {})

        if "mp_mpjpe" not in lite_m or "mp_mpjpe" not in heavy_m:
            continue

        names.append(name)
        lite_init_mpjpe.append(lite_m["mp_mpjpe"] * 100)
        lite_opt_mpjpe.append(lite_m["opt_mpjpe"] * 100)
        heavy_init_mpjpe.append(heavy_m["mp_mpjpe"] * 100)
        heavy_opt_mpjpe.append(heavy_m["opt_mpjpe"] * 100)

        lite_init_p_mpjpe.append(lite_m["mp_p_mpjpe"] * 100)
        lite_opt_p_mpjpe.append(lite_m["opt_p_mpjpe"] * 100)
        heavy_init_p_mpjpe.append(heavy_m["mp_p_mpjpe"] * 100)
        heavy_opt_p_mpjpe.append(heavy_m["opt_p_mpjpe"] * 100)

        # Per-axis errors
        lite_axes = compute_per_axis_mpjpe(lite_preds[name]["frames"])
        heavy_axes = compute_per_axis_mpjpe(heavy_preds[name]["frames"])
        if lite_axes and heavy_axes:
            lite_mp_axes_all.append(lite_axes["mp_per_axis"] * 100)
            lite_opt_axes_all.append(lite_axes["opt_per_axis"] * 100)
            heavy_mp_axes_all.append(heavy_axes["mp_per_axis"] * 100)
            heavy_opt_axes_all.append(heavy_axes["opt_per_axis"] * 100)

    if not names:
        print("  No examples with ground truth. Exiting.")
        sys.exit(1)

    # --- Graph 1: Aggregate MPJPE comparison ---
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    x = np.arange(len(names))
    width = 0.2
    ax.bar(x - 1.5 * width, lite_init_mpjpe, width, label="Lite init", color="#90ee90", alpha=0.8)
    ax.bar(x - 0.5 * width, lite_opt_mpjpe, width, label="Lite opt", color="#006400", alpha=0.8)
    ax.bar(x + 0.5 * width, heavy_init_mpjpe, width, label="Heavy init", color="#ffb3b3", alpha=0.8)
    ax.bar(x + 1.5 * width, heavy_opt_mpjpe, width, label="Heavy opt", color="#8b0000", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MPJPE (cm)")
    ax.set_title("MPJPE Comparison: MediaPipe Lite vs Heavy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(OUTPUT_DIR, "aggregate_comparison.png"))

    # --- Graph 2: Per-axis error ---
    if lite_mp_axes_all:
        lite_mp_mean = np.mean(lite_mp_axes_all, axis=0)
        lite_opt_mean = np.mean(lite_opt_axes_all, axis=0)
        heavy_mp_mean = np.mean(heavy_mp_axes_all, axis=0)
        heavy_opt_mean = np.mean(heavy_opt_axes_all, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        axis_labels = ["X (horizontal)", "Y (vertical)", "Z (depth)"]
        x = np.arange(3)
        width = 0.2
        ax.bar(x - 1.5 * width, lite_mp_mean, width, label="Lite init", color="#90ee90", alpha=0.8)
        ax.bar(x - 0.5 * width, lite_opt_mean, width, label="Lite opt", color="#006400", alpha=0.8)
        ax.bar(x + 0.5 * width, heavy_mp_mean, width, label="Heavy init", color="#ffb3b3", alpha=0.8)
        ax.bar(x + 1.5 * width, heavy_opt_mean, width, label="Heavy opt", color="#8b0000", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(axis_labels)
        ax.set_ylabel("Mean Absolute Error (cm)")
        ax.set_title("Per-Axis Error: X vs Y vs Z (tests depth hypothesis)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        _save(fig, os.path.join(OUTPUT_DIR, "per_axis_error.png"))

    # --- Graph 3: P-MPJPE comparison ---
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    x = np.arange(len(names))
    width = 0.2
    ax.bar(x - 1.5 * width, lite_init_p_mpjpe, width, label="Lite init", color="#90ee90", alpha=0.8)
    ax.bar(x - 0.5 * width, lite_opt_p_mpjpe, width, label="Lite opt", color="#006400", alpha=0.8)
    ax.bar(x + 0.5 * width, heavy_init_p_mpjpe, width, label="Heavy init", color="#ffb3b3", alpha=0.8)
    ax.bar(x + 1.5 * width, heavy_opt_p_mpjpe, width, label="Heavy opt", color="#8b0000", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("P-MPJPE (cm)")
    ax.set_title("P-MPJPE Comparison: MediaPipe Lite vs Heavy (structure quality)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(OUTPUT_DIR, "p_mpjpe_comparison.png"))

    # --- Graph 4: Improvement from optimization ---
    lite_improvement = [init - opt for init, opt in zip(lite_init_mpjpe, lite_opt_mpjpe)]
    heavy_improvement = [init - opt for init, opt in zip(heavy_init_mpjpe, heavy_opt_mpjpe)]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, lite_improvement, width, label="Lite improvement", color="green", alpha=0.7)
    ax.bar(x + width / 2, heavy_improvement, width, label="Heavy improvement", color="red", alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MPJPE Improvement (cm)")
    ax.set_title("Optimisation Improvement: Lite vs Heavy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(OUTPUT_DIR, "improvement_comparison.png"))

    # --- Summary table ---
    print(f"\n  {'='*90}")
    print(f"  {'Example':<30} {'Lite Init':>10} {'Lite Opt':>10} {'Heavy Init':>11} {'Heavy Opt':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*11} {'-'*10}")
    for i, name in enumerate(names):
        print(
            f"  {name:<30} "
            f"{lite_init_mpjpe[i]:>10.2f} "
            f"{lite_opt_mpjpe[i]:>10.2f} "
            f"{heavy_init_mpjpe[i]:>11.2f} "
            f"{heavy_opt_mpjpe[i]:>10.2f}"
        )
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*11} {'-'*10}")
    print(
        f"  {'MEAN':<30} "
        f"{np.mean(lite_init_mpjpe):>10.2f} "
        f"{np.mean(lite_opt_mpjpe):>10.2f} "
        f"{np.mean(heavy_init_mpjpe):>11.2f} "
        f"{np.mean(heavy_opt_mpjpe):>10.2f}"
    )
    print(f"  {'='*90}")

    # Depth hypothesis summary
    if lite_mp_axes_all:
        print(f"\n  Per-Axis Mean Error (cm):")
        print(f"  {'':>20} {'X':>8} {'Y':>8} {'Z (depth)':>10}")
        print(f"  {'Lite init':<20} {lite_mp_mean[0]:>8.2f} {lite_mp_mean[1]:>8.2f} {lite_mp_mean[2]:>10.2f}")
        print(f"  {'Lite opt':<20} {lite_opt_mean[0]:>8.2f} {lite_opt_mean[1]:>8.2f} {lite_opt_mean[2]:>10.2f}")
        print(f"  {'Heavy init':<20} {heavy_mp_mean[0]:>8.2f} {heavy_mp_mean[1]:>8.2f} {heavy_mp_mean[2]:>10.2f}")
        print(f"  {'Heavy opt':<20} {heavy_opt_mean[0]:>8.2f} {heavy_opt_mean[1]:>8.2f} {heavy_opt_mean[2]:>10.2f}")

    print(f"\n  Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
