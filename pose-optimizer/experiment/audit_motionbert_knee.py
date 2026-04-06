"""Audit MotionBERT knee issue: bone length distortion in lower body.

Analyzes the detected bone lengths from MotionBERT and how they compare
to ground truth. Checks if the issue is systematic or per-frame.

Usage:
    cd pose-optimizer
    uv run python experiment/audit_motionbert_knee.py <mb_trajectories.json>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from skeleton import JOINT_NAMES, NUM_JOINTS, PARENTS, DEFAULT_BONE_LENGTHS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", help="Path to MotionBERT trajectories.json")
    args = parser.parse_args()

    with open(args.trajectories) as f:
        traj = json.load(f)

    gt_raw = traj["ground_truth"]
    pred_raw = traj["raw_prediction"]

    gt_indices = [i for i, g in enumerate(gt_raw) if g is not None]
    gt_arr = np.array([gt_raw[i] for i in gt_indices])
    pred_arr = np.array([pred_raw[i] for i in gt_indices])
    n_frames = len(gt_indices)

    # ---- 1. Per-frame bone lengths ----
    print("=" * 70)
    print("1. PER-FRAME BONE LENGTHS: MotionBERT vs GT")
    print("=" * 70)

    leg_joints = [2, 3, 5, 6]  # RKnee, RAnkle, LKnee, LAnkle
    arm_joints = [11, 12, 14, 15]  # LElbow, LWrist, RElbow, RWrist

    for j in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        p = int(PARENTS[j])
        gt_bls = []
        pred_bls = []
        for f_idx in range(n_frames):
            gt_bl = np.linalg.norm(gt_arr[f_idx, j] - gt_arr[f_idx, p])
            pred_bl = np.linalg.norm(pred_arr[f_idx, j] - pred_arr[f_idx, p])
            gt_bls.append(gt_bl)
            pred_bls.append(pred_bl)
        gt_bls = np.array(gt_bls)
        pred_bls = np.array(pred_bls)

        flag = " ***" if j in leg_joints else ""
        print(f"  {JOINT_NAMES[j]:15s} ({JOINT_NAMES[p]:10s} -> {JOINT_NAMES[j]:10s}): "
              f"GT={np.mean(gt_bls)*100:.1f}cm  "
              f"Pred: mean={np.mean(pred_bls)*100:.1f}cm "
              f"std={np.std(pred_bls)*100:.1f}cm "
              f"min={np.min(pred_bls)*100:.1f}cm "
              f"max={np.max(pred_bls)*100:.1f}cm "
              f"ratio={np.mean(pred_bls)/np.mean(gt_bls):.2f}x{flag}")
    print()

    # ---- 2. Check direction of knee bones ----
    print("=" * 70)
    print("2. KNEE BONE DIRECTION ANALYSIS (sample frames)")
    print("=" * 70)

    for f_idx in [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1]:
        print(f"\n  Frame {f_idx}:")
        for j in [2, 5]:  # RKnee, LKnee
            p = int(PARENTS[j])
            gt_vec = gt_arr[f_idx, j] - gt_arr[f_idx, p]
            pred_vec = pred_arr[f_idx, j] - pred_arr[f_idx, p]
            gt_dir = gt_vec / (np.linalg.norm(gt_vec) + 1e-10)
            pred_dir = pred_vec / (np.linalg.norm(pred_vec) + 1e-10)
            cos_sim = np.dot(gt_dir, pred_dir)
            print(f"    {JOINT_NAMES[j]:10s}: "
                  f"GT dir=({gt_dir[0]:+.2f}, {gt_dir[1]:+.2f}, {gt_dir[2]:+.2f}) "
                  f"Pred dir=({pred_dir[0]:+.2f}, {pred_dir[1]:+.2f}, {pred_dir[2]:+.2f}) "
                  f"cos_sim={cos_sim:.3f} "
                  f"GT_len={np.linalg.norm(gt_vec)*100:.1f}cm "
                  f"Pred_len={np.linalg.norm(pred_vec)*100:.1f}cm")
    print()

    # ---- 3. Check _RELIABLE_BONES_FOR_SCALE effect ----
    print("=" * 70)
    print("3. RELIABLE BONES FOR SCALE: ratio analysis")
    print("=" * 70)

    # Simulate what motionbert_to_camera_space does
    reliable_set = {10, 11, 12, 13, 14, 15}

    for f_idx in [0, n_frames // 2, n_frames - 1]:
        print(f"\n  Frame {f_idx}:")
        # Root-relative positions
        rr_pred = pred_arr[f_idx] - pred_arr[f_idx, 0:1]
        rr_gt = gt_arr[f_idx] - gt_arr[f_idx, 0:1]

        reliable_ratios = []
        all_ratios = []
        for j in range(1, NUM_JOINTS):
            p = int(PARENTS[j])
            det_bl = np.linalg.norm(rr_pred[j] - rr_pred[p])
            ref_bl = float(DEFAULT_BONE_LENGTHS[j])
            if det_bl > 1e-4 and ref_bl > 1e-4:
                ratio = ref_bl / det_bl
                all_ratios.append((j, ratio, det_bl, ref_bl))
                if j in reliable_set:
                    reliable_ratios.append((j, ratio, det_bl, ref_bl))

        print(f"    All bone scale ratios (DEFAULT / detected):")
        for j, ratio, det_bl, ref_bl in all_ratios:
            marker = " [RELIABLE]" if j in reliable_set else ""
            print(f"      {JOINT_NAMES[j]:15s}: ratio={ratio:.3f} "
                  f"(det={det_bl*100:.1f}cm / ref={ref_bl*100:.1f}cm){marker}")

        reliable_vals = [r[1] for r in reliable_ratios]
        all_vals = [r[1] for r in all_ratios]
        print(f"    Reliable median: {np.median(reliable_vals):.4f}")
        print(f"    All median:      {np.median(all_vals):.4f}")


if __name__ == "__main__":
    main()
