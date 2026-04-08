"""Generate aggregate cross-pipeline graphs from saved results.json.

Reads the run-level results.json (which contains per-example metrics for both
MotionBERT and MediaPipe) and generates:
  1. Cross-pipeline per-joint position error (VW-SI-MPJPE)
  2. Cross-pipeline per-joint velocity error (VW-SI-MPJVE)
  3. Cross-pipeline aggregate metrics comparison
  4. Per-pipeline aggregate summary (bar chart per example)
  5. Per-example improvement waterfall (delta from det to opt)

Usage:
    uv run python experiment/generate_aggregate_graphs.py <results.json> [output_dir]

    output_dir defaults to graphs/ next to results.json.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphs import (
    generate_cross_pipeline_per_joint_position,
    generate_cross_pipeline_per_joint_velocity,
    generate_cross_pipeline_metrics_comparison,
    generate_aggregate_summary,
    _save,
    COLOR_MB_RAW, COLOR_MB_OPT, COLOR_MP_RAW, COLOR_MP_OPT,
)


def generate_improvement_waterfall(
    mb_metrics: list[dict],
    mp_metrics: list[dict],
    output_dir: str,
) -> None:
    """Bar chart showing per-example delta (opt - det) VW-SI-MPJPE."""
    os.makedirs(output_dir, exist_ok=True)

    names = [m["name"] for m in mb_metrics]
    mb_delta = [(m.get("opt_vw_si_mpjpe", 0) - m.get("det_vw_si_mpjpe", 0)) * 100 for m in mb_metrics]
    mp_delta = [(m.get("opt_vw_si_mpjpe", 0) - m.get("det_vw_si_mpjpe", 0)) * 100 for m in mp_metrics]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.9), 6))
    ax.bar(x - width / 2, mb_delta, width, color=COLOR_MB_OPT, alpha=0.8, label="MotionBERT")
    ax.bar(x + width / 2, mp_delta, width, color=COLOR_MP_OPT, alpha=0.8, label="MediaPipe")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("VW-SI-MPJPE Change (cm)")
    ax.set_title("Optimization Effect Per Example (negative = improvement)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "improvement_waterfall.png"))


def generate_per_example_comparison(
    mb_metrics: list[dict],
    mp_metrics: list[dict],
    output_dir: str,
) -> None:
    """Grouped bar chart: det vs opt VW-SI-MPJPE per example, both pipelines."""
    os.makedirs(output_dir, exist_ok=True)

    names = [m["name"] for m in mb_metrics]
    x = np.arange(len(names))
    width = 0.2

    mb_det = [m.get("det_vw_si_mpjpe", 0) * 100 for m in mb_metrics]
    mb_opt = [m.get("opt_vw_si_mpjpe", 0) * 100 for m in mb_metrics]
    mp_det = [m.get("det_vw_si_mpjpe", 0) * 100 for m in mp_metrics]
    mp_opt = [m.get("opt_vw_si_mpjpe", 0) * 100 for m in mp_metrics]

    fig, ax = plt.subplots(figsize=(max(16, len(names) * 1.0), 7))
    ax.bar(x - 1.5 * width, mb_det, width, color=COLOR_MB_RAW, label="MB Raw")
    ax.bar(x - 0.5 * width, mb_opt, width, color=COLOR_MB_OPT, label="MB Optimized")
    ax.bar(x + 0.5 * width, mp_det, width, color=COLOR_MP_RAW, label="MP Raw")
    ax.bar(x + 1.5 * width, mp_opt, width, color=COLOR_MP_OPT, label="MP Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("VW-SI-MPJPE (cm)")
    ax.set_title("Per-Example VW-SI-MPJPE: All Pipelines")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, os.path.join(output_dir, "per_example_comparison.png"))


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python experiment/generate_aggregate_graphs.py <results.json> [output_dir]")
        sys.exit(1)

    results_path = sys.argv[1]
    with open(results_path) as f:
        data = json.load(f)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(results_path), "graphs")
    os.makedirs(output_dir, exist_ok=True)

    mb_metrics = data.get("motionbert", [])
    mp_metrics = data.get("mediapipe", [])

    print(f"Loaded {len(mb_metrics)} MotionBERT examples, {len(mp_metrics)} MediaPipe examples")
    print(f"Output: {output_dir}")

    # The cross-pipeline functions expect metrics dicts with specific keys.
    # The run-level results.json stores them flat — map to what graphs.py expects.
    # Check if per-joint data exists in the run-level results
    has_per_joint = any("det_vw_si_mpjpe_per_joint" in m for m in mb_metrics)

    if has_per_joint:
        print("Generating cross-pipeline per-joint position graph...")
        generate_cross_pipeline_per_joint_position(mb_metrics, mp_metrics, output_dir)

        print("Generating cross-pipeline per-joint velocity graph...")
        generate_cross_pipeline_per_joint_velocity(mb_metrics, mp_metrics, output_dir)
    else:
        print("No per-joint data in run-level results — reading per-example results.json files...")
        # Try reading per-example results.json to get per-joint data
        results_dir = os.path.dirname(results_path)
        mb_detailed, mp_detailed = [], []
        for m in mb_metrics:
            name = m["name"]
            per_ex_path = os.path.join(results_dir, name, "motionbert", "results.json")
            if os.path.exists(per_ex_path):
                with open(per_ex_path) as f:
                    per_ex = json.load(f)
                # Merge per-joint data from per_ex into m
                metrics = per_ex.get("metrics", {})
                raw = per_ex.get("raw_metrics", {})
                merged = dict(m)
                if "vw_si_mpjpe_per_joint" in metrics:
                    merged["opt_vw_si_mpjpe_per_joint"] = metrics["vw_si_mpjpe_per_joint"]
                if "vw_si_mpjpe_per_joint" in raw:
                    merged["det_vw_si_mpjpe_per_joint"] = raw["vw_si_mpjpe_per_joint"]
                if "vw_si_mpjve_per_joint" in metrics:
                    merged["opt_vw_si_mpjve_per_joint"] = metrics["vw_si_mpjve_per_joint"]
                if "vw_si_mpjve_per_joint" in raw:
                    merged["det_vw_si_mpjve_per_joint"] = raw["vw_si_mpjve_per_joint"]
                mb_detailed.append(merged)
        for m in mp_metrics:
            name = m["name"]
            per_ex_path = os.path.join(results_dir, name, "mediapipe", "results.json")
            if os.path.exists(per_ex_path):
                with open(per_ex_path) as f:
                    per_ex = json.load(f)
                metrics = per_ex.get("metrics", {})
                raw = per_ex.get("raw_metrics", {})
                merged = dict(m)
                if "vw_si_mpjpe_per_joint" in metrics:
                    merged["opt_vw_si_mpjpe_per_joint"] = metrics["vw_si_mpjpe_per_joint"]
                if "vw_si_mpjpe_per_joint" in raw:
                    merged["det_vw_si_mpjpe_per_joint"] = raw["vw_si_mpjpe_per_joint"]
                if "vw_si_mpjve_per_joint" in metrics:
                    merged["opt_vw_si_mpjve_per_joint"] = metrics["vw_si_mpjve_per_joint"]
                if "vw_si_mpjve_per_joint" in raw:
                    merged["det_vw_si_mpjve_per_joint"] = raw["vw_si_mpjve_per_joint"]
                mp_detailed.append(merged)

        if mb_detailed and mp_detailed:
            print(f"  Found per-joint data for {len(mb_detailed)} MB, {len(mp_detailed)} MP examples")
            print("Generating cross-pipeline per-joint position graph...")
            generate_cross_pipeline_per_joint_position(mb_detailed, mp_detailed, output_dir)
            print("Generating cross-pipeline per-joint velocity graph...")
            generate_cross_pipeline_per_joint_velocity(mb_detailed, mp_detailed, output_dir)
        else:
            print("  Could not find per-example results.json files, skipping per-joint graphs")

    print("Generating cross-pipeline metrics comparison...")
    generate_cross_pipeline_metrics_comparison(mb_metrics, mp_metrics, output_dir)

    print("Generating MotionBERT aggregate summary...")
    generate_aggregate_summary(mb_metrics, output_dir, prefix="motionbert")

    print("Generating MediaPipe aggregate summary...")
    generate_aggregate_summary(mp_metrics, output_dir, prefix="mediapipe")

    print("Generating improvement waterfall...")
    generate_improvement_waterfall(mb_metrics, mp_metrics, output_dir)

    print("Generating per-example comparison...")
    generate_per_example_comparison(mb_metrics, mp_metrics, output_dir)

    graphs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    print(f"\nGenerated {len(graphs)} graphs in {output_dir}:")
    for g in sorted(graphs):
        print(f"  {g}")


if __name__ == "__main__":
    main()
