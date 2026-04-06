"""Audit VW-SI-MPJPE computation step by step.

Loads a pipeline run's trajectories.json, recomputes metrics from scratch,
and prints every intermediate value to verify correctness.

Usage:
    cd pose-optimizer
    uv run experiment/audit_vw_si_mpjpe.py <trajectories.json> <results.json>
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
    si_mpjpe,
    vw_mpjpe,
    vw_si_mpjpe,
)
from skeleton import EVAL_JOINTS, EVAL_JOINT_NAMES, JOINT_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", help="Path to trajectories.json")
    parser.add_argument("results", nargs="?", help="Path to results.json (optional, for comparison)")
    args = parser.parse_args()

    with open(args.trajectories) as f:
        traj = json.load(f)

    camera = Camera.from_dict(traj["camera"])
    gt_raw = traj["ground_truth"]
    pred_raw = traj["raw_prediction"]
    opt_raw = traj["optimized_prediction"]

    # Find frames with GT
    gt_indices = [i for i, g in enumerate(gt_raw) if g is not None]
    print(f"Total frames: {len(pred_raw)}")
    print(f"Frames with GT: {len(gt_indices)}")
    print()

    gt_arr = np.array([gt_raw[i] for i in gt_indices])    # (F, 16, 3)
    pred_arr = np.array([pred_raw[i] for i in gt_indices]) # (F, 16, 3)
    opt_arr = np.array([opt_raw[i] for i in gt_indices])   # (F, 16, 3)

    # Slice to eval joints
    gt_eval = gt_arr[:, EVAL_JOINTS, :]     # (F, 15, 3)
    pred_eval = pred_arr[:, EVAL_JOINTS, :] # (F, 15, 3)
    opt_eval = opt_arr[:, EVAL_JOINTS, :]   # (F, 15, 3)

    print(f"Camera image_size: {camera.image_size} (h, w)")
    print(f"Camera fx={camera.fx:.1f} fy={camera.fy:.1f} cx={camera.cx:.1f} cy={camera.cy:.1f}")
    print()

    # ---- 1. Visibility weights ----
    print("=" * 60)
    print("1. VISIBILITY WEIGHTS")
    print("=" * 60)
    vis_all = compute_visibility_weights(gt_arr, camera)  # (F, 16)
    vis_eval = vis_all[:, EVAL_JOINTS]                    # (F, 15)

    print(f"Shape: {vis_eval.shape}")
    print(f"Total visible joint-frames: {int(vis_eval.sum())} / {vis_eval.size}")
    print(f"Visibility rate: {vis_eval.mean():.2%}")
    print()

    # Per-joint visibility
    print("Per-joint visibility rate:")
    for j_idx, j_name in enumerate(EVAL_JOINT_NAMES):
        rate = vis_eval[:, j_idx].mean()
        print(f"  {j_name:15s}: {rate:.2%} ({int(vis_eval[:, j_idx].sum())}/{vis_eval.shape[0]} frames)")
    print()

    # Check a sample frame: project GT to 2D and show coordinates
    mid = len(gt_indices) // 2
    gt_2d = camera.camera_to_image(gt_arr[mid])  # (16, 2)
    print(f"Sample frame {mid}: GT 2D projections (all 16 joints):")
    h, w = camera.image_size
    for j in range(16):
        u, v = gt_2d[j]
        in_frame = 0 <= u < w and 0 <= v < h
        print(f"  {JOINT_NAMES[j]:15s}: ({u:7.1f}, {v:7.1f})  {'IN' if in_frame else 'OUT'}")
    print()

    # ---- 2. Scale factors ----
    print("=" * 60)
    print("2. OPTIMAL SCALE FACTORS")
    print("=" * 60)

    s_det_unweighted = optimal_scale(pred_eval, gt_eval)
    s_det_weighted = _optimal_scale_weighted(pred_eval, gt_eval, vis_eval)
    s_opt_unweighted = optimal_scale(opt_eval, gt_eval)
    s_opt_weighted = _optimal_scale_weighted(opt_eval, gt_eval, vis_eval)

    print(f"Det unweighted scale: {s_det_unweighted:.6f}")
    print(f"Det weighted scale:   {s_det_weighted:.6f}")
    print(f"Opt unweighted scale: {s_opt_unweighted:.6f}")
    print(f"Opt weighted scale:   {s_opt_weighted:.6f}")
    print()

    # Check if scale is hitting bounds
    print("Scale bounds check (bounds are 0.5 to 2.0):")
    for name, s in [("det_weighted", s_det_weighted), ("opt_weighted", s_opt_weighted)]:
        if abs(s - 0.5) < 0.01:
            print(f"  WARNING: {name} scale {s:.4f} is at LOWER bound!")
        elif abs(s - 2.0) < 0.01:
            print(f"  WARNING: {name} scale {s:.4f} is at UPPER bound!")
        else:
            print(f"  {name}: {s:.4f} OK (not at bounds)")
    print()

    # ---- 3. Position magnitudes ----
    print("=" * 60)
    print("3. POSITION MAGNITUDES (camera space)")
    print("=" * 60)

    gt_mean_dist = np.mean(np.linalg.norm(gt_eval, axis=-1))
    pred_mean_dist = np.mean(np.linalg.norm(pred_eval, axis=-1))
    opt_mean_dist = np.mean(np.linalg.norm(opt_eval, axis=-1))

    print(f"Mean distance from origin:")
    print(f"  GT:   {gt_mean_dist:.4f} m")
    print(f"  Pred: {pred_mean_dist:.4f} m")
    print(f"  Opt:  {opt_mean_dist:.4f} m")
    print(f"  Pred/GT ratio: {pred_mean_dist/gt_mean_dist:.4f}")
    print(f"  Opt/GT ratio:  {opt_mean_dist/gt_mean_dist:.4f}")
    print()

    # Mean Z (depth)
    gt_mean_z = np.mean(gt_eval[:, :, 2])
    pred_mean_z = np.mean(pred_eval[:, :, 2])
    opt_mean_z = np.mean(opt_eval[:, :, 2])
    print(f"Mean Z (depth):")
    print(f"  GT:   {gt_mean_z:.4f} m")
    print(f"  Pred: {pred_mean_z:.4f} m")
    print(f"  Opt:  {opt_mean_z:.4f} m")
    print()

    # ---- 4. Metric breakdown ----
    print("=" * 60)
    print("4. METRIC BREAKDOWN (det)")
    print("=" * 60)

    mpjpe_val = mpjpe(pred_eval, gt_eval)
    si_mpjpe_val = si_mpjpe(pred_eval, gt_eval)
    vw_mpjpe_val = vw_mpjpe(pred_eval, gt_eval, vis_eval)
    vw_si_mpjpe_val = vw_si_mpjpe(pred_eval, gt_eval, vis_eval)

    print(f"MPJPE:       {mpjpe_val*100:.2f} cm")
    print(f"SI-MPJPE:    {si_mpjpe_val*100:.2f} cm")
    print(f"VW-MPJPE:    {vw_mpjpe_val*100:.2f} cm")
    print(f"VW-SI-MPJPE: {vw_si_mpjpe_val*100:.2f} cm")
    print()

    # ---- 5. Per-joint error with visibility ----
    print("=" * 60)
    print("5. PER-JOINT ERROR (det, camera-space, before scale)")
    print("=" * 60)

    errors = np.linalg.norm(pred_eval - gt_eval, axis=-1)  # (F, 15)
    for j_idx, j_name in enumerate(EVAL_JOINT_NAMES):
        vis_rate = vis_eval[:, j_idx].mean()
        mean_err = np.mean(errors[:, j_idx])
        visible_frames = vis_eval[:, j_idx] > 0
        if visible_frames.any():
            visible_err = np.mean(errors[:, j_idx][visible_frames])
        else:
            visible_err = float('nan')
        print(f"  {j_name:15s}: all={mean_err*100:7.2f}cm  visible_only={visible_err*100:7.2f}cm  vis_rate={vis_rate:.0%}")
    print()

    # ---- 6. After scale ----
    print("=" * 60)
    print("6. PER-JOINT ERROR (det, after optimal weighted scale)")
    print("=" * 60)

    scaled_pred = s_det_weighted * pred_eval
    errors_scaled = np.linalg.norm(scaled_pred - gt_eval, axis=-1)
    for j_idx, j_name in enumerate(EVAL_JOINT_NAMES):
        vis_rate = vis_eval[:, j_idx].mean()
        visible_frames = vis_eval[:, j_idx] > 0
        if visible_frames.any():
            visible_err = np.mean(errors_scaled[:, j_idx][visible_frames])
        else:
            visible_err = float('nan')
        weighted_contrib = float(np.sum(errors_scaled[:, j_idx] * vis_eval[:, j_idx]))
        print(f"  {j_name:15s}: vis_err={visible_err*100:7.2f}cm  vis_rate={vis_rate:.0%}  weighted_contrib={weighted_contrib:.4f}")
    print()

    # ---- 7. Sanity: recompute VW-SI-MPJPE manually ----
    print("=" * 60)
    print("7. MANUAL VW-SI-MPJPE RECOMPUTATION")
    print("=" * 60)
    w_sum = float(vis_eval.sum())
    manual_vw_si = float(np.sum(errors_scaled * vis_eval) / w_sum) if w_sum > 0 else 0
    print(f"Library VW-SI-MPJPE:  {vw_si_mpjpe_val*100:.4f} cm")
    print(f"Manual VW-SI-MPJPE:   {manual_vw_si*100:.4f} cm")
    print(f"Match: {'YES' if abs(manual_vw_si - vw_si_mpjpe_val) < 1e-8 else 'NO'}")
    print()

    # Compare with stored results if provided
    if args.results:
        with open(args.results) as f:
            stored = json.load(f)
        # Try to find det_vw_si_mpjpe in the results
        if "motionbert" in stored:
            for m in stored["motionbert"]:
                print(f"Stored det_vw_si_mpjpe: {m.get('det_vw_si_mpjpe', 'N/A')}")
                print(f"Stored opt_vw_si_mpjpe: {m.get('opt_vw_si_mpjpe', 'N/A')}")


if __name__ == "__main__":
    main()
