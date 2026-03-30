#!/usr/bin/env python3
"""Cross-pipeline comparison: MotionBert vs MediaPipe.

Usage:
    uv run python compare.py output/motionbert_DIR/ output/mediapipe_DIR/ --output output/comparison/

Generates three levels of comparison visualizations:
  Level 1: Single-video X/Y/Z trajectory comparison per joint
  Level 2: Per-video metrics heatmap across 4 variants
  Level 3: Aggregate bar chart across all videos
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Joint definitions (duplicated to keep script standalone)
JOINT_NAMES = [
    "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck", "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist",
]
EVAL_JOINTS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]
EVAL_JOINT_NAMES = [JOINT_NAMES[j] for j in EVAL_JOINTS]

ALL_METRICS = [
    "mpjpe", "p_mpjpe", "si_mpjpe", "vw_mpjpe", "vw_si_mpjpe",
    "mpjve", "si_mpjve", "vw_mpjve", "vw_si_mpjve",
]

LEVEL2_METRICS = ["vw_si_mpjpe", "si_mpjpe", "vw_si_mpjve", "mpjve"]


def load_pipeline_data(pipeline_dir: str) -> dict:
    """Load all results.json and trajectories.json from a pipeline output dir."""
    data = {}
    if not os.path.isdir(pipeline_dir):
        print(f"ERROR: Directory not found: {pipeline_dir}")
        sys.exit(1)

    for entry in sorted(os.listdir(pipeline_dir)):
        example_dir = os.path.join(pipeline_dir, entry)
        results_path = os.path.join(example_dir, "results.json")
        traj_path = os.path.join(example_dir, "trajectories.json")
        if not os.path.isfile(results_path):
            continue

        with open(results_path) as f:
            results = json.load(f)

        traj = None
        if os.path.isfile(traj_path):
            with open(traj_path) as f:
                traj = json.load(f)

        # Skip examples with empty metrics
        if not results.get("metrics"):
            continue

        data[entry] = {"results": results, "trajectories": traj}

    return data


def find_common_examples(data_a: dict, data_b: dict) -> list[str]:
    """Find example names present in both pipeline outputs."""
    return sorted(set(data_a.keys()) & set(data_b.keys()))


# ---------------------------------------------------------------------------
# Level 1: Single-video trajectory comparison
# ---------------------------------------------------------------------------

def generate_level1(
    mb_data: dict, mp_data: dict, example: str, output_dir: str
) -> None:
    """Generate per-joint X/Y/Z trajectory plots for one example."""
    single_dir = os.path.join(output_dir, "single_video")
    os.makedirs(single_dir, exist_ok=True)

    mb_traj = mb_data[example]["trajectories"]
    mp_traj = mp_data[example]["trajectories"]

    gt = np.array(mb_traj["ground_truth"])          # (N, 16, 3)
    mb_raw = np.array(mb_traj["raw_prediction"])     # (N, 16, 3)
    mb_opt = np.array(mb_traj["optimized_prediction"])
    mp_raw = np.array(mp_traj["raw_prediction"])
    mp_opt = np.array(mp_traj["optimized_prediction"])

    n_frames = gt.shape[0]
    frames = np.arange(n_frames)
    axis_labels = ["X", "Y", "Z"]

    # Individual plots per joint per axis
    for ji, j in enumerate(EVAL_JOINTS):
        jname = JOINT_NAMES[j]
        for c, cname in enumerate(axis_labels):
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.plot(frames, gt[:, j, c], "b-", linewidth=1.5, label="Ground Truth")
            ax.plot(frames, mp_raw[:, j, c], "g--", linewidth=1.0, alpha=0.7, label="MediaPipe Raw")
            ax.plot(frames, mp_opt[:, j, c], "g-", linewidth=1.2, label="MediaPipe Opt")
            ax.plot(frames, mb_raw[:, j, c], "r--", linewidth=1.0, alpha=0.7, label="MotionBert Raw")
            ax.plot(frames, mb_opt[:, j, c], "r-", linewidth=1.2, label="MotionBert Opt")
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{cname} (m)")
            ax.set_title(f"{jname} - {cname} axis  [{example}]")
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(single_dir, f"{jname}_{cname}.png"), dpi=150)
            plt.close(fig)

    # Summary grid: 12 joints x 3 axes
    fig = plt.figure(figsize=(24, 36))
    gs = gridspec.GridSpec(len(EVAL_JOINTS), 3, hspace=0.35, wspace=0.25)

    for ji, j in enumerate(EVAL_JOINTS):
        jname = JOINT_NAMES[j]
        for c, cname in enumerate(axis_labels):
            ax = fig.add_subplot(gs[ji, c])
            ax.plot(frames, gt[:, j, c], "b-", linewidth=1.2, label="GT")
            ax.plot(frames, mp_raw[:, j, c], "g--", linewidth=0.8, alpha=0.6, label="MP Raw")
            ax.plot(frames, mp_opt[:, j, c], "g-", linewidth=1.0, label="MP Opt")
            ax.plot(frames, mb_raw[:, j, c], "r--", linewidth=0.8, alpha=0.6, label="MB Raw")
            ax.plot(frames, mb_opt[:, j, c], "r-", linewidth=1.0, label="MB Opt")
            ax.set_title(f"{jname} {cname}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)
            if ji == 0 and c == 0:
                ax.legend(fontsize=6, loc="best")

    fig.suptitle(f"Trajectory Comparison: {example}", fontsize=14, y=0.998)
    fig.savefig(os.path.join(single_dir, "summary_grid.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Level 1: Saved {len(EVAL_JOINTS) * 3} individual plots + summary grid to {single_dir}/")


# ---------------------------------------------------------------------------
# Level 2: Per-video metrics comparison
# ---------------------------------------------------------------------------

def generate_level2(
    mb_data: dict, mp_data: dict, examples: list[str], output_dir: str
) -> None:
    """Generate per-video metrics heatmap and JSON."""
    os.makedirs(output_dir, exist_ok=True)

    variants = ["MediaPipe Raw", "MediaPipe Opt", "MotionBert Raw", "MotionBert Opt"]
    n_examples = len(examples)
    n_metrics = len(LEVEL2_METRICS)
    n_variants = len(variants)

    # Build data table: examples x metrics x variants
    table = np.full((n_examples, n_metrics, n_variants), np.nan)
    json_data = {}

    for ei, ex in enumerate(examples):
        mb_res = mb_data[ex]["results"]
        mp_res = mp_data[ex]["results"]

        row = {}
        for mi, metric in enumerate(LEVEL2_METRICS):
            mp_raw_val = mp_res["raw_metrics"].get(metric, np.nan)
            mp_opt_val = mp_res["metrics"].get(metric, np.nan)
            mb_raw_val = mb_res["raw_metrics"].get(metric, np.nan)
            mb_opt_val = mb_res["metrics"].get(metric, np.nan)

            table[ei, mi, 0] = mp_raw_val
            table[ei, mi, 1] = mp_opt_val
            table[ei, mi, 2] = mb_raw_val
            table[ei, mi, 3] = mb_opt_val

            row[metric] = {
                "mediapipe_raw": mp_raw_val,
                "mediapipe_opt": mp_opt_val,
                "motionbert_raw": mb_raw_val,
                "motionbert_opt": mb_opt_val,
            }
        json_data[ex] = row

    # Save JSON
    json_path = os.path.join(output_dir, "per_video_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # Generate heatmap figure: one subplot per metric
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(6, 0.4 * n_examples + 2)))

    for mi, metric in enumerate(LEVEL2_METRICS):
        ax = axes[mi]
        data_slice = table[:, mi, :] * 100  # Convert to cm

        im = ax.imshow(data_slice, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(n_variants))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_examples))
        ax.set_yticklabels(examples, fontsize=6)
        ax.set_title(f"{metric} (cm)", fontsize=10)

        # Annotate cells
        for ei in range(n_examples):
            for vi in range(n_variants):
                val = data_slice[ei, vi]
                if not np.isnan(val):
                    color = "white" if val > np.nanmean(data_slice) else "black"
                    ax.text(vi, ei, f"{val:.1f}", ha="center", va="center",
                            fontsize=5, color=color)

        plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Per-Video Metrics Comparison (cm)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(output_dir, "per_video_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Level 2: Saved per_video_metrics.png and .json to {output_dir}/")


# ---------------------------------------------------------------------------
# Level 3: Aggregate comparison
# ---------------------------------------------------------------------------

def generate_level3(
    mb_data: dict, mp_data: dict, examples: list[str], output_dir: str
) -> None:
    """Generate aggregate bar chart and JSON across all videos."""
    os.makedirs(output_dir, exist_ok=True)

    variants = ["MediaPipe Raw", "MediaPipe Opt", "MotionBert Raw", "MotionBert Opt"]
    colors = ["#66bb6a", "#2e7d32", "#ef5350", "#b71c1c"]

    # Collect per-metric averages
    agg = {v: {m: [] for m in ALL_METRICS} for v in variants}

    for ex in examples:
        mb_res = mb_data[ex]["results"]
        mp_res = mp_data[ex]["results"]

        for metric in ALL_METRICS:
            agg["MediaPipe Raw"][metric].append(mp_res["raw_metrics"].get(metric, np.nan))
            agg["MediaPipe Opt"][metric].append(mp_res["metrics"].get(metric, np.nan))
            agg["MotionBert Raw"][metric].append(mb_res["raw_metrics"].get(metric, np.nan))
            agg["MotionBert Opt"][metric].append(mb_res["metrics"].get(metric, np.nan))

    # Compute means
    avg = {}
    for v in variants:
        avg[v] = {}
        for m in ALL_METRICS:
            vals = [x for x in agg[v][m] if not np.isnan(x)]
            avg[v][m] = float(np.mean(vals)) if vals else None

    # Save JSON
    json_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(json_path, "w") as f:
        json.dump(avg, f, indent=2)

    # Bar chart
    n_metrics = len(ALL_METRICS)
    x = np.arange(n_metrics)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    for vi, v in enumerate(variants):
        vals = [avg[v][m] * 100 if avg[v][m] is not None else 0 for m in ALL_METRICS]
        ax.bar(x + vi * width - 1.5 * width, vals, width,
               label=v, color=colors[vi], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_METRICS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Value (cm)")
    ax.set_title(f"Aggregate Metrics Comparison ({len(examples)} videos)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=5, padding=1)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregate_metrics.png"), dpi=150)
    plt.close(fig)
    print(f"  Level 3: Saved aggregate_metrics.png and .json to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare MotionBert and MediaPipe pipeline outputs."
    )
    parser.add_argument("motionbert_dir", help="Path to MotionBert output directory")
    parser.add_argument("mediapipe_dir", help="Path to MediaPipe output directory")
    parser.add_argument("--output", default="output/comparison/",
                        help="Output directory for comparison visualizations")
    args = parser.parse_args()

    print("Loading pipeline data...")
    mb_data = load_pipeline_data(args.motionbert_dir)
    mp_data = load_pipeline_data(args.mediapipe_dir)
    print(f"  MotionBert: {len(mb_data)} examples with metrics")
    print(f"  MediaPipe:  {len(mp_data)} examples with metrics")

    examples = find_common_examples(mb_data, mp_data)
    print(f"  Common examples: {len(examples)}")
    if not examples:
        print("ERROR: No common examples found between the two pipelines.")
        sys.exit(1)

    # Find first example that has trajectories in both
    traj_example = None
    for ex in examples:
        if mb_data[ex]["trajectories"] and mp_data[ex]["trajectories"]:
            traj_example = ex
            break

    os.makedirs(args.output, exist_ok=True)

    # Level 1: Single-video trajectories
    if traj_example:
        print(f"\nLevel 1: Single-video trajectory comparison [{traj_example}]")
        generate_level1(mb_data, mp_data, traj_example, args.output)
    else:
        print("\nLevel 1: SKIPPED (no example has trajectories in both pipelines)")

    # Level 2: Per-video metrics
    print("\nLevel 2: Per-video metrics comparison")
    generate_level2(mb_data, mp_data, examples, args.output)

    # Level 3: Aggregate comparison
    print("\nLevel 3: Aggregate metrics comparison")
    generate_level3(mb_data, mp_data, examples, args.output)

    print(f"\nDone. All outputs saved to {args.output}")


if __name__ == "__main__":
    main()
