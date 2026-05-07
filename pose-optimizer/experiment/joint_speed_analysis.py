"""Per-joint speed comparison: GT vs detector vs optimized.

Addresses SPEC Hypothesis 1: verifies whether the optimizer over-smooths
specific joints (elbows, wrists, knees) by comparing average joint speeds.

Compares:
- Ground truth speeds
- Raw detector (MotionBert) speeds
- Old optimized (num_steps=50) speeds
- New optimized (num_steps=5) speeds

Usage:
    uv run python experiment/joint_speed_analysis.py
"""

import json
import os
import sys

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from camera import Camera
from evaluate import root_relative, _optimal_scale_weighted, compute_visibility_weights
from skeleton import EVAL_JOINTS, EVAL_JOINT_NAMES


# Joints of interest from the SPEC
FOCUS_JOINTS = {
    "LElbow": 11,
    "RElbow": 14,
    "LWrist": 12,
    "RWrist": 15,
    "LKnee": 5,
    "RKnee": 2,
}

# Map skeleton index to eval-joint index
SKEL_TO_EVAL = {skel_idx: eval_idx for eval_idx, skel_idx in enumerate(EVAL_JOINTS)}


def load_example(example_name, output_dir):
    """Load trajectories and camera from a saved example."""
    traj_path = os.path.join(output_dir, example_name, "trajectories.json")
    if not os.path.exists(traj_path):
        return None
    with open(traj_path) as f:
        data = json.load(f)

    camera = Camera.from_dict(data["camera"])
    gt_cam = []
    for g in data["ground_truth"]:
        gt_cam.append(np.array(g) if g is not None else None)

    raw_pred = [np.array(p) for p in data["raw_prediction"]]
    opt_pred = [np.array(p) for p in data["optimized_prediction"]]

    gt_indices = [i for i, g in enumerate(gt_cam) if g is not None]
    if len(gt_indices) < 3:
        return None

    gt_arr = np.array([gt_cam[i] for i in gt_indices])
    raw_arr = np.array([raw_pred[i] for i in gt_indices])
    opt_arr = np.array([opt_pred[i] for i in gt_indices])
    return gt_arr, raw_arr, opt_arr, camera


def compute_joint_speeds(positions_rr):
    """Compute per-joint average speed (L2 norm of frame-to-frame displacement).

    Args:
        positions_rr: (F, J, 3) root-relative positions.

    Returns:
        (J,) average speed per joint.
    """
    velocities = np.diff(positions_rr, axis=0)  # (F-1, J, 3)
    speeds = np.linalg.norm(velocities, axis=-1)  # (F-1, J)
    return speeds.mean(axis=0)  # (J,)


def scale_to_gt(pred_rr, gt_rr, camera, gt_3d):
    """Apply optimal scale to make prediction scale-comparable to GT."""
    vis = compute_visibility_weights(gt_3d, camera)[:, EVAL_JOINTS]
    s = _optimal_scale_weighted(pred_rr, gt_rr, vis)
    return s * pred_rr


def main():
    new_dir = "output/motionbert_2026_03_30_21_52"   # steps=5
    old_dir = "output/motionbert_2026_03_30_18_57"    # steps=50

    examples = sorted([
        d for d in os.listdir(new_dir)
        if os.path.isdir(os.path.join(new_dir, d))
    ])

    print("=" * 70)
    print("  Per-Joint Speed Analysis (SPEC Hypothesis 1)")
    print("=" * 70)

    # Accumulate per-joint speeds across all examples
    focus_names = list(FOCUS_JOINTS.keys())
    # columns: GT, Detector, OldOpt(50), NewOpt(5)
    all_speeds = {name: {"gt": [], "det": [], "old_opt": [], "new_opt": []} for name in focus_names}

    n_valid = 0
    for ex in examples:
        new_result = load_example(ex, new_dir)
        old_result = load_example(ex, old_dir)
        if new_result is None or old_result is None:
            continue

        gt_arr, raw_arr, new_opt_arr, camera = new_result
        _, _, old_opt_arr, _ = old_result

        # Root-relative, eval joints
        gt_rr = root_relative(gt_arr)[:, EVAL_JOINTS, :]
        raw_rr = root_relative(raw_arr)[:, EVAL_JOINTS, :]
        new_opt_rr = root_relative(new_opt_arr)[:, EVAL_JOINTS, :]
        old_opt_rr = root_relative(old_opt_arr)[:, EVAL_JOINTS, :]

        # Scale predictions to GT scale for fair speed comparison
        raw_rr_scaled = scale_to_gt(raw_rr, gt_rr, camera, gt_arr)
        new_opt_rr_scaled = scale_to_gt(new_opt_rr, gt_rr, camera, gt_arr)
        old_opt_rr_scaled = scale_to_gt(old_opt_rr, gt_rr, camera, gt_arr)

        gt_speeds = compute_joint_speeds(gt_rr)
        raw_speeds = compute_joint_speeds(raw_rr_scaled)
        old_opt_speeds = compute_joint_speeds(old_opt_rr_scaled)
        new_opt_speeds = compute_joint_speeds(new_opt_rr_scaled)

        for name, skel_idx in FOCUS_JOINTS.items():
            eval_idx = SKEL_TO_EVAL[skel_idx]
            all_speeds[name]["gt"].append(gt_speeds[eval_idx] * 100)       # cm
            all_speeds[name]["det"].append(raw_speeds[eval_idx] * 100)
            all_speeds[name]["old_opt"].append(old_opt_speeds[eval_idx] * 100)
            all_speeds[name]["new_opt"].append(new_opt_speeds[eval_idx] * 100)

        n_valid += 1

    print(f"\n  Analyzed {n_valid} examples\n")

    # Print table
    header = f"{'Joint':<12} {'GT':>8} {'Detector':>10} {'Opt50':>8} {'Opt5':>8} {'Det/GT':>8} {'Opt50/GT':>9} {'Opt5/GT':>8}"
    print(header)
    print("-" * len(header))

    for name in focus_names:
        gt_avg = np.mean(all_speeds[name]["gt"])
        det_avg = np.mean(all_speeds[name]["det"])
        old_avg = np.mean(all_speeds[name]["old_opt"])
        new_avg = np.mean(all_speeds[name]["new_opt"])

        print(f"  {name:<12} {gt_avg:>7.2f} {det_avg:>10.2f} {old_avg:>8.2f} {new_avg:>8.2f} "
              f"{det_avg/gt_avg:>7.2f}x {old_avg/gt_avg:>8.2f}x {new_avg/gt_avg:>7.2f}x")

    # Aggregated
    all_gt = np.mean([np.mean(all_speeds[n]["gt"]) for n in focus_names])
    all_det = np.mean([np.mean(all_speeds[n]["det"]) for n in focus_names])
    all_old = np.mean([np.mean(all_speeds[n]["old_opt"]) for n in focus_names])
    all_new = np.mean([np.mean(all_speeds[n]["new_opt"]) for n in focus_names])
    print("-" * len(header))
    print(f"  {'AVERAGE':<12} {all_gt:>7.2f} {all_det:>10.2f} {all_old:>8.2f} {all_new:>8.2f} "
          f"{all_det/all_gt:>7.2f}x {all_old/all_gt:>8.2f}x {all_new/all_gt:>7.2f}x")

    # Speed ratio analysis
    print(f"\n  Speed ratio to GT (closer to 1.0 = better):")
    print(f"    Detector:     {all_det/all_gt:.3f}x")
    print(f"    Opt steps=50: {all_old/all_gt:.3f}x")
    print(f"    Opt steps=5:  {all_new/all_gt:.3f}x")

    if all_old / all_gt < all_det / all_gt:
        print(f"\n  -> steps=50 REDUCES speed below detector level (over-smoothing)")
    else:
        print(f"\n  -> steps=50 does NOT reduce speed below detector (no over-smoothing)")

    if abs(all_new / all_gt - 1.0) < abs(all_old / all_gt - 1.0):
        print(f"  -> steps=5 keeps speeds closer to GT than steps=50")

    # ------------------------------------------------------------------
    # Generate graph
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(focus_names))
    width = 0.2

    gt_vals = [np.mean(all_speeds[n]["gt"]) for n in focus_names]
    det_vals = [np.mean(all_speeds[n]["det"]) for n in focus_names]
    old_vals = [np.mean(all_speeds[n]["old_opt"]) for n in focus_names]
    new_vals = [np.mean(all_speeds[n]["new_opt"]) for n in focus_names]

    ax.bar(x - 1.5*width, gt_vals, width, label="Ground Truth", color="#2ecc71")
    ax.bar(x - 0.5*width, det_vals, width, label="Detector (raw)", color="#3498db")
    ax.bar(x + 0.5*width, old_vals, width, label="Optimized (steps=50)", color="#e74c3c")
    ax.bar(x + 1.5*width, new_vals, width, label="Optimized (steps=5)", color="#f39c12")

    ax.set_ylabel("Average Speed (cm/frame)")
    ax.set_title("Per-Joint Average Speed: GT vs Detector vs Optimized\n(MotionBert, scale-corrected)")
    ax.set_xticks(x)
    ax.set_xticklabels(focus_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = "experiment/joint_speed_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n  Graph saved to {out_path}")


if __name__ == "__main__":
    main()
