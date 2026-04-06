"""Audit FK roundtrip error and optimization regression for MediaPipe.

Loads trajectories.json, converts raw 3D -> FK params -> 3D, measures error.
Also compares raw vs optimized per-joint to find which joints regress.

Usage:
    cd pose-optimizer
    uv run python experiment/audit_fk_roundtrip.py <trajectories.json>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from camera import Camera
from evaluate import (
    compute_visibility_weights,
    _optimal_scale_weighted,
    vw_si_mpjpe,
    vw_si_mpjpe_per_joint,
)
from fk import forward_kinematics, positions_to_fk_params
from skeleton import EVAL_JOINTS, EVAL_JOINT_NAMES, JOINT_NAMES, NUM_JOINTS, PARENTS


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

    n_frames = len(gt_indices)

    # ---- 1. FK Roundtrip Error ----
    print("=" * 60)
    print("1. FK ROUNDTRIP ERROR (raw prediction -> FK params -> reconstruction)")
    print("=" * 60)

    roundtrip_errors_per_joint = np.zeros((n_frames, NUM_JOINTS))
    roundtrip_errors_per_frame = np.zeros(n_frames)

    for f_idx in range(n_frames):
        positions = pred_arr[f_idx]  # (16, 3)
        root_pos, root_rot, local_rots, bone_lengths = positions_to_fk_params(positions)

        with torch.no_grad():
            reconstructed = forward_kinematics(
                torch.tensor(root_pos, dtype=torch.float32),
                torch.tensor(root_rot, dtype=torch.float32),
                torch.tensor(local_rots, dtype=torch.float32),
                torch.tensor(bone_lengths, dtype=torch.float32),
            ).numpy()

        per_joint_err = np.linalg.norm(reconstructed - positions, axis=-1)
        roundtrip_errors_per_joint[f_idx] = per_joint_err
        roundtrip_errors_per_frame[f_idx] = np.mean(per_joint_err)

    print(f"Mean roundtrip error: {np.mean(roundtrip_errors_per_frame)*100:.4f} cm")
    print(f"Max roundtrip error:  {np.max(roundtrip_errors_per_frame)*100:.4f} cm")
    print()
    print("Per-joint mean roundtrip error:")
    for j in range(NUM_JOINTS):
        mean_err = np.mean(roundtrip_errors_per_joint[:, j])
        max_err = np.max(roundtrip_errors_per_joint[:, j])
        print(f"  {JOINT_NAMES[j]:15s}: mean={mean_err*100:.4f} cm  max={max_err*100:.4f} cm")
    print()

    # ---- 2. Bone length comparison ----
    print("=" * 60)
    print("2. DETECTED BONE LENGTHS vs GT")
    print("=" * 60)

    gt_bone_lengths = np.zeros(NUM_JOINTS)
    pred_bone_lengths_all = np.zeros((n_frames, NUM_JOINTS))

    for f_idx in range(n_frames):
        for j in range(1, NUM_JOINTS):
            p = int(PARENTS[j])
            gt_bone_lengths[j] += np.linalg.norm(gt_arr[f_idx, j] - gt_arr[f_idx, p])
            pred_bone_lengths_all[f_idx, j] = np.linalg.norm(
                pred_arr[f_idx, j] - pred_arr[f_idx, p]
            )

    gt_bone_lengths /= n_frames
    pred_bone_lengths_median = np.median(pred_bone_lengths_all, axis=0)

    print(f"{'Joint':15s} {'GT':>8s} {'Pred Med':>10s} {'Ratio':>8s} {'Std':>8s}")
    for j in range(1, NUM_JOINTS):
        ratio = pred_bone_lengths_median[j] / gt_bone_lengths[j] if gt_bone_lengths[j] > 0.001 else 0
        std = np.std(pred_bone_lengths_all[:, j])
        print(
            f"  {JOINT_NAMES[j]:15s} {gt_bone_lengths[j]*100:7.2f}cm "
            f"{pred_bone_lengths_median[j]*100:9.2f}cm "
            f"{ratio:7.2f}x "
            f"{std*100:7.3f}cm"
        )
    print()

    # ---- 3. Raw vs Optimized comparison ----
    print("=" * 60)
    print("3. RAW vs OPTIMIZED: per-joint VW-SI-MPJPE")
    print("=" * 60)

    gt_eval = gt_arr[:, EVAL_JOINTS, :]
    pred_eval = pred_arr[:, EVAL_JOINTS, :]
    opt_eval = opt_arr[:, EVAL_JOINTS, :]
    vis = compute_visibility_weights(gt_arr, camera)[:, EVAL_JOINTS]

    raw_vw_si = vw_si_mpjpe(pred_eval, gt_eval, vis)
    opt_vw_si = vw_si_mpjpe(opt_eval, gt_eval, vis)

    raw_per_joint = vw_si_mpjpe_per_joint(pred_eval, gt_eval, vis)
    opt_per_joint = vw_si_mpjpe_per_joint(opt_eval, gt_eval, vis)

    print(f"Overall VW-SI-MPJPE: raw={raw_vw_si*100:.2f} cm  opt={opt_vw_si*100:.2f} cm  "
          f"delta={((opt_vw_si - raw_vw_si) / raw_vw_si * 100):+.1f}%")
    print()
    print(f"{'Joint':15s} {'Raw':>8s} {'Opt':>8s} {'Delta':>8s} {'%Change':>8s}")
    for j_idx, j_name in enumerate(EVAL_JOINT_NAMES):
        raw_val = raw_per_joint[j_idx] * 100
        opt_val = opt_per_joint[j_idx] * 100
        delta = opt_val - raw_val
        pct = delta / raw_val * 100 if raw_val > 0.01 else 0
        marker = "  <-- WORSE" if delta > 0.1 else ""
        print(f"  {j_name:15s} {raw_val:7.2f}cm {opt_val:7.2f}cm {delta:+7.2f}cm {pct:+7.1f}%{marker}")
    print()

    # ---- 4. 2D reprojection check ----
    print("=" * 60)
    print("4. 2D REPROJECTION: how well do raw predictions project to image?")
    print("=" * 60)

    # Check if raw 3D -> 2D is accurate (low reprojection error means good 2D, depth is off)
    raw_2d = camera.camera_to_image(pred_arr)  # (F, 16, 2)
    gt_2d = camera.camera_to_image(gt_arr)     # (F, 16, 2)
    opt_2d = camera.camera_to_image(opt_arr)   # (F, 16, 2)

    for stage, proj_2d in [("raw", raw_2d), ("opt", opt_2d)]:
        vis_all = compute_visibility_weights(gt_arr, camera)
        errors_2d = np.linalg.norm(proj_2d - gt_2d, axis=-1)  # (F, 16)
        for j in range(NUM_JOINTS):
            vis_frames = vis_all[:, j] > 0
            if vis_frames.any():
                mean_err = np.mean(errors_2d[:, j][vis_frames])
            else:
                mean_err = float('nan')
            if j != 7:  # skip Spine
                print(f"  {stage:3s} {JOINT_NAMES[j]:15s}: {mean_err:7.1f} px")
        print()


if __name__ == "__main__":
    main()
