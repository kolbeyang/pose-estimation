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
from camera import Camera
from evaluate import (
    compute_visibility_weights,
    vw_si_mpjpe_per_frame,
    vw_si_mpjve_per_frame,
)
from skeleton import EVAL_JOINTS


def _compute_per_frame_stds(traj_path: str) -> dict[str, float]:
    """Compute per-frame std (within-video spread) for VW-SI-MPJPE/MPJVE.

    Returns std of per-frame VW-SI-MPJPE/MPJVE arrays (in meters) for det
    and opt predictions. Keys are aligned with the metric names used in
    per-example metric dicts so error bars can attach generically.
    """
    with open(traj_path) as f:
        traj = json.load(f)

    # Ground truth may contain None for frames without GT annotation.
    # Filter to only frames where all three arrays exist.
    gt_raw = traj["ground_truth"]
    det_raw = traj["raw_prediction"]
    opt_raw = traj["optimized_prediction"]
    valid = [i for i in range(len(gt_raw)) if gt_raw[i] is not None]
    if len(valid) < 2:
        return {k: 0.0 for k in [
            "det_vw_si_mpjpe_frame_std", "opt_vw_si_mpjpe_frame_std",
            "det_vw_si_mpjve_frame_std", "opt_vw_si_mpjve_frame_std",
            "det_mpjpe_frame_std", "opt_mpjpe_frame_std",
            "det_si_mpjpe_frame_std", "opt_si_mpjpe_frame_std",
        ]}

    gt = np.asarray([gt_raw[i] for i in valid], dtype=np.float64)
    det = np.asarray([det_raw[i] for i in valid], dtype=np.float64)
    opt = np.asarray([opt_raw[i] for i in valid], dtype=np.float64)
    cam = Camera.from_dict(traj["camera"])

    vis = compute_visibility_weights(gt, cam)[:, EVAL_JOINTS]
    gt_e = gt[:, EVAL_JOINTS, :]
    det_e = det[:, EVAL_JOINTS, :]
    opt_e = opt[:, EVAL_JOINTS, :]

    F = gt.shape[0]
    det_pos_std = float(np.std(vw_si_mpjpe_per_frame(det_e, gt_e, vis), ddof=1)) if F > 1 else 0.0
    opt_pos_std = float(np.std(vw_si_mpjpe_per_frame(opt_e, gt_e, vis), ddof=1)) if F > 1 else 0.0
    if F >= 3:
        det_vel_std = float(np.std(vw_si_mpjve_per_frame(det_e, gt_e, vis), ddof=1))
        opt_vel_std = float(np.std(vw_si_mpjve_per_frame(opt_e, gt_e, vis), ddof=1))
    else:
        det_vel_std = 0.0
        opt_vel_std = 0.0

    return {
        "det_vw_si_mpjpe_frame_std": det_pos_std,
        "opt_vw_si_mpjpe_frame_std": opt_pos_std,
        "det_vw_si_mpjve_frame_std": det_vel_std,
        "opt_vw_si_mpjve_frame_std": opt_vel_std,
        # Also surface as std for related position metrics used in aggregate summary
        "det_mpjpe_frame_std": det_pos_std,
        "opt_mpjpe_frame_std": opt_pos_std,
        "det_si_mpjpe_frame_std": det_pos_std,
        "opt_si_mpjpe_frame_std": opt_pos_std,
    }


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

    # Error bars for delta: std of (opt - det) per frame. We don't have
    # per-frame delta readily, so use the larger of the two per-frame stds
    # as a conservative proxy.
    def _delta_err(m):
        a = m.get("opt_vw_si_mpjpe_frame_std", 0) or 0
        b = m.get("det_vw_si_mpjpe_frame_std", 0) or 0
        return max(a, b) * 100
    mb_err = [_delta_err(m) for m in mb_metrics]
    mp_err = [_delta_err(m) for m in mp_metrics]

    x = np.arange(len(names))
    width = 0.35
    ekw = dict(ecolor="black", capsize=2, elinewidth=0.8)

    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.9), 6))
    ax.bar(x - width / 2, mb_delta, width, yerr=mb_err, error_kw=ekw, color=COLOR_MB_OPT, alpha=0.8, label="MotionBERT")
    ax.bar(x + width / 2, mp_delta, width, yerr=mp_err, error_kw=ekw, color=COLOR_MP_OPT, alpha=0.8, label="MediaPipe")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("VW-SI-MPJPE Change (cm)")
    ax.set_title("Optimization Effect Per Example (error bars = per-frame std within video)")
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

    # Per-frame stds for within-video error bars
    mb_det_err = [(m.get("det_vw_si_mpjpe_frame_std", 0) or 0) * 100 for m in mb_metrics]
    mb_opt_err = [(m.get("opt_vw_si_mpjpe_frame_std", 0) or 0) * 100 for m in mb_metrics]
    mp_det_err = [(m.get("det_vw_si_mpjpe_frame_std", 0) or 0) * 100 for m in mp_metrics]
    mp_opt_err = [(m.get("opt_vw_si_mpjpe_frame_std", 0) or 0) * 100 for m in mp_metrics]

    ekw = dict(ecolor="black", capsize=2, elinewidth=0.8)
    fig, ax = plt.subplots(figsize=(max(16, len(names) * 1.0), 7))
    ax.bar(x - 1.5 * width, mb_det, width, yerr=mb_det_err, error_kw=ekw, color=COLOR_MB_RAW, label="MB Raw")
    ax.bar(x - 0.5 * width, mb_opt, width, yerr=mb_opt_err, error_kw=ekw, color=COLOR_MB_OPT, label="MB Optimized")
    ax.bar(x + 0.5 * width, mp_det, width, yerr=mp_det_err, error_kw=ekw, color=COLOR_MP_RAW, label="MP Raw")
    ax.bar(x + 1.5 * width, mp_opt, width, yerr=mp_opt_err, error_kw=ekw, color=COLOR_MP_OPT, label="MP Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("VW-SI-MPJPE (cm)")
    ax.set_title("Per-Example VW-SI-MPJPE: All Pipelines (error bars = per-frame std within video)")
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

    results_dir = os.path.dirname(results_path)

    def _augment(metrics_list, pipeline):
        """Merge per-joint metrics and per-frame stds from per-example results."""
        out = []
        for m in metrics_list:
            name = m["name"]
            merged = dict(m)
            per_ex_path = os.path.join(results_dir, name, pipeline, "results.json")
            if os.path.exists(per_ex_path):
                with open(per_ex_path) as f:
                    per_ex = json.load(f)
                pm = per_ex.get("metrics", {})
                pr = per_ex.get("raw_metrics", {})
                if "vw_si_mpjpe_per_joint" in pm:
                    merged["opt_vw_si_mpjpe_per_joint"] = pm["vw_si_mpjpe_per_joint"]
                if "vw_si_mpjpe_per_joint" in pr:
                    merged["det_vw_si_mpjpe_per_joint"] = pr["vw_si_mpjpe_per_joint"]
                if "vw_si_mpjve_per_joint" in pm:
                    merged["opt_vw_si_mpjve_per_joint"] = pm["vw_si_mpjve_per_joint"]
                if "vw_si_mpjve_per_joint" in pr:
                    merged["det_vw_si_mpjve_per_joint"] = pr["vw_si_mpjve_per_joint"]
            traj_path = os.path.join(results_dir, name, pipeline, "trajectories.json")
            if os.path.exists(traj_path):
                try:
                    merged.update(_compute_per_frame_stds(traj_path))
                except Exception as e:
                    print(f"  warn: failed to compute per-frame stds for {name}/{pipeline}: {e}")
            out.append(merged)
        return out

    print("Augmenting metrics with per-joint data and per-frame stds...")
    mb_metrics = _augment(mb_metrics, "motionbert")
    mp_metrics = _augment(mp_metrics, "mediapipe")

    has_per_joint = any("det_vw_si_mpjpe_per_joint" in m for m in mb_metrics) and \
                    any("det_vw_si_mpjpe_per_joint" in m for m in mp_metrics)

    if has_per_joint:
        print("Generating cross-pipeline per-joint position graph...")
        generate_cross_pipeline_per_joint_position(mb_metrics, mp_metrics, output_dir)
        print("Generating cross-pipeline per-joint velocity graph...")
        generate_cross_pipeline_per_joint_velocity(mb_metrics, mp_metrics, output_dir)
    else:
        print("  No per-joint data available — skipping per-joint graphs")

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
