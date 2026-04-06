"""Investigate whether the SI scale is pathological in camera space.

Compares:
1. Global scale on full camera-space positions (current behavior)
2. Scale on root-relative positions + keep GT root (what we did before)
3. Per-frame depth alignment + skeleton scale

Usage:
    cd pose-optimizer
    uv run experiment/audit_scale_effect.py <trajectories.json>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from camera import Camera
from evaluate import (
    compute_visibility_weights,
    optimal_scale,
    _optimal_scale_weighted,
    mpjpe,
    vw_si_mpjpe,
    vw_mpjpe,
)
from skeleton import EVAL_JOINTS, EVAL_JOINT_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", help="Path to trajectories.json")
    args = parser.parse_args()

    with open(args.trajectories) as f:
        traj = json.load(f)

    camera = Camera.from_dict(traj["camera"])
    gt_raw = traj["ground_truth"]
    pred_raw = traj["raw_prediction"]
    opt_raw = traj["optimized_prediction"]

    gt_indices = [i for i, g in enumerate(gt_raw) if g is not None]
    gt_arr = np.array([gt_raw[i] for i in gt_indices])
    pred_arr = np.array([pred_raw[i] for i in gt_indices])
    opt_arr = np.array([opt_raw[i] for i in gt_indices])

    gt_eval = gt_arr[:, EVAL_JOINTS, :]
    pred_eval = pred_arr[:, EVAL_JOINTS, :]
    opt_eval = opt_arr[:, EVAL_JOINTS, :]

    vis = compute_visibility_weights(gt_arr, camera)[:, EVAL_JOINTS]

    print("=" * 60)
    print("APPROACH 1: Current — global scale on camera-space positions")
    print("=" * 60)
    s1 = _optimal_scale_weighted(pred_eval, gt_eval, vis)
    vw_si_1 = vw_si_mpjpe(pred_eval, gt_eval, vis)
    print(f"Scale: {s1:.6f}")
    print(f"VW-SI-MPJPE: {vw_si_1*100:.2f} cm")

    # Show what the scale does to depth
    print(f"Pred mean Z: {np.mean(pred_eval[:,:,2]):.4f} m")
    print(f"GT mean Z:   {np.mean(gt_eval[:,:,2]):.4f} m")
    print(f"Scaled pred mean Z: {s1 * np.mean(pred_eval[:,:,2]):.4f} m")
    print()

    print("=" * 60)
    print("APPROACH 2: Root-relative scale + GT root (old behavior)")
    print("=" * 60)
    # Make root-relative
    pred_rr = pred_eval - pred_eval[:, 0:1, :]  # (F, 15, 3) relative to pelvis
    gt_rr = gt_eval - gt_eval[:, 0:1, :]
    s2 = _optimal_scale_weighted(pred_rr, gt_rr, vis)
    # Apply scale to root-relative, then add back GT root
    scaled_rr = s2 * pred_rr
    # Compute error in root-relative space
    errors_rr = np.linalg.norm(scaled_rr - gt_rr, axis=-1)
    w_sum = float(vis.sum())
    vw_si_2 = float(np.sum(errors_rr * vis) / w_sum) if w_sum > 0 else 0
    print(f"Scale: {s2:.6f}")
    print(f"VW-SI-MPJPE (root-relative): {vw_si_2*100:.2f} cm")
    print()

    print("=" * 60)
    print("APPROACH 3: Per-frame depth correction + skeleton scale")
    print("=" * 60)
    # Shift pred to match GT pelvis depth per frame, then scale skeleton
    pred_depth_corrected = pred_eval.copy()
    for f in range(pred_eval.shape[0]):
        depth_offset = gt_eval[f, 0, 2] - pred_eval[f, 0, 2]  # align pelvis Z
        pred_depth_corrected[f, :, 2] += depth_offset
        # Also align pelvis XY
        xy_offset = gt_eval[f, 0, :2] - pred_eval[f, 0, :2]
        pred_depth_corrected[f, :, :2] += xy_offset

    s3 = _optimal_scale_weighted(
        pred_depth_corrected - pred_depth_corrected[:, 0:1, :],
        gt_eval - gt_eval[:, 0:1, :],
        vis,
    )
    pred_rr3 = pred_depth_corrected - pred_depth_corrected[:, 0:1, :]
    gt_rr3 = gt_eval - gt_eval[:, 0:1, :]
    scaled_rr3 = s3 * pred_rr3 + gt_eval[:, 0:1, :]  # scale skeleton, keep GT root
    errors_3 = np.linalg.norm(scaled_rr3 - gt_eval, axis=-1)
    vw_si_3 = float(np.sum(errors_3 * vis) / w_sum) if w_sum > 0 else 0
    print(f"Scale: {s3:.6f}")
    print(f"VW-SI-MPJPE (depth-corrected + root-relative scale): {vw_si_3*100:.2f} cm")
    print()

    print("=" * 60)
    print("APPROACH 4: Just VW-MPJPE (no scale at all)")
    print("=" * 60)
    vw_4 = vw_mpjpe(pred_eval, gt_eval, vis)
    print(f"VW-MPJPE: {vw_4*100:.2f} cm")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Current (camera-space SI):     {vw_si_1*100:.2f} cm")
    print(f"Root-relative SI:              {vw_si_2*100:.2f} cm")
    print(f"Depth-corrected + RR SI:       {vw_si_3*100:.2f} cm")
    print(f"No scale (VW-MPJPE):           {vw_4*100:.2f} cm")
    print()

    # ---- What are the biggest error contributors? ----
    print("=" * 60)
    print("PER-JOINT CONTRIBUTION TO VW-SI-MPJPE (current method)")
    print("=" * 60)
    scaled_pred = s1 * pred_eval
    errors = np.linalg.norm(scaled_pred - gt_eval, axis=-1)
    total_weighted = float(np.sum(errors * vis))
    for j_idx, j_name in enumerate(EVAL_JOINT_NAMES):
        j_weighted = float(np.sum(errors[:, j_idx] * vis[:, j_idx]))
        pct = j_weighted / total_weighted * 100 if total_weighted > 0 else 0
        j_vis = vis[:, j_idx].mean()
        print(f"  {j_name:15s}: {j_weighted/w_sum*100:6.2f} cm ({pct:5.1f}% of total)  vis={j_vis:.0%}")


if __name__ == "__main__":
    main()
