"""Test SI-MPJPE metric consistency.

Verifies:
1. SI-MPJPE scale factor actually minimizes MPJPE
2. VW-SI-MPJPE is consistent with VW-MPJPE after scaling
3. Scale=1.0 gives result >= optimal (optimizer finds true minimum)

Usage:
    uv run python experiment/test_si_consistency.py
"""

import json
import os
import sys

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np

from camera import Camera
from evaluate import (
    evaluate,
    mpjpe,
    si_mpjpe,
    vw_mpjpe,
    vw_si_mpjpe,
    optimal_scale,
    _optimal_scale_weighted,
    root_relative,
    compute_visibility_weights,
)
from skeleton import EVAL_JOINTS


def load_example(example_name, output_dir="output/motionbert_2026_03_30_18_57"):
    """Load trajectories and camera from a saved example."""
    traj_path = os.path.join(output_dir, example_name, "trajectories.json")
    with open(traj_path) as f:
        data = json.load(f)

    camera = Camera.from_dict(data["camera"])

    gt_cam = []
    for g in data["ground_truth"]:
        if g is not None:
            gt_cam.append(np.array(g))
        else:
            gt_cam.append(None)

    raw_pred = [np.array(p) for p in data["raw_prediction"]]
    opt_pred = [np.array(p) for p in data["optimized_prediction"]]

    # Filter frames with GT
    gt_indices = [i for i, g in enumerate(gt_cam) if g is not None]
    gt_arr = np.array([gt_cam[i] for i in gt_indices])
    raw_arr = np.array([raw_pred[i] for i in gt_indices])
    opt_arr = np.array([opt_pred[i] for i in gt_indices])

    return gt_arr, raw_arr, opt_arr, camera


def test_si_scale_minimizes_mpjpe(pred_rr, gt_rr):
    """Test 1: SI-MPJPE scale factor actually minimizes MPJPE."""
    s = optimal_scale(pred_rr, gt_rr)
    si_val = mpjpe(s * pred_rr, gt_rr)

    # Test nearby scale values -- they should give worse MPJPE
    for delta in [-0.05, -0.01, +0.01, +0.05]:
        nearby_val = mpjpe((s + delta) * pred_rr, gt_rr)
        if nearby_val < si_val - 1e-8:
            return False, f"scale={s:.4f} gives MPJPE={si_val:.6f}, but scale={s+delta:.4f} gives {nearby_val:.6f}"

    # Also verify si_mpjpe matches manual computation
    si_result = si_mpjpe(pred_rr, gt_rr)
    if abs(si_result - si_val) > 1e-8:
        return False, f"si_mpjpe()={si_result:.8f} != mpjpe(s*pred, gt)={si_val:.8f}"

    return True, f"OK: scale={s:.4f}, SI-MPJPE={si_val*100:.4f} cm"


def test_vw_si_consistent_with_vw_mpjpe(pred_rr, gt_rr, weights):
    """Test 2: VW-SI-MPJPE is consistent with VW-MPJPE after scaling."""
    s = _optimal_scale_weighted(pred_rr, gt_rr, weights)
    scaled_pred = s * pred_rr

    vw_si_val = vw_si_mpjpe(pred_rr, gt_rr, weights)
    vw_val_at_scale = vw_mpjpe(scaled_pred, gt_rr, weights)

    if abs(vw_si_val - vw_val_at_scale) > 1e-8:
        return False, f"VW-SI-MPJPE={vw_si_val:.8f} != VW-MPJPE(s*pred)={vw_val_at_scale:.8f}"

    # Also check nearby scales give worse results
    for delta in [-0.05, -0.01, +0.01, +0.05]:
        nearby_val = vw_mpjpe((s + delta) * pred_rr, gt_rr, weights)
        if nearby_val < vw_si_val - 1e-8:
            return False, f"scale={s:.4f} gives VW-MPJPE={vw_si_val:.6f}, but scale={s+delta:.4f} gives {nearby_val:.6f}"

    return True, f"OK: scale={s:.4f}, VW-SI-MPJPE={vw_si_val*100:.4f} cm"


def test_scale_1_worse_than_optimal(pred_rr, gt_rr, weights):
    """Test 3: Scale=1.0 gives result >= optimal."""
    s = _optimal_scale_weighted(pred_rr, gt_rr, weights)
    optimal_val = vw_si_mpjpe(pred_rr, gt_rr, weights)
    scale1_val = vw_mpjpe(pred_rr, gt_rr, weights)  # scale=1.0

    if scale1_val < optimal_val - 1e-8:
        return False, f"scale=1.0 ({scale1_val:.6f}) < optimal ({optimal_val:.6f}). Scale factor={s:.4f}"

    return True, f"OK: scale=1.0 gives {scale1_val*100:.4f} cm >= optimal {optimal_val*100:.4f} cm (scale={s:.4f})"


def test_optimization_doesnt_hurt_scale_invariant(gt_arr, raw_arr, opt_arr, camera):
    """Test 4: If optimization improves raw MPJPE, does SI-MPJPE also improve?"""
    pred_rr = root_relative(raw_arr)[:, EVAL_JOINTS, :]
    opt_rr = root_relative(opt_arr)[:, EVAL_JOINTS, :]
    gt_rr = root_relative(gt_arr)[:, EVAL_JOINTS, :]

    vis = compute_visibility_weights(gt_arr, camera)[:, EVAL_JOINTS]

    raw_mpjpe = vw_mpjpe(pred_rr, gt_rr, vis)
    opt_mpjpe_val = vw_mpjpe(opt_rr, gt_rr, vis)
    raw_si = vw_si_mpjpe(pred_rr, gt_rr, vis)
    opt_si = vw_si_mpjpe(opt_rr, gt_rr, vis)

    raw_s = _optimal_scale_weighted(pred_rr, gt_rr, vis)
    opt_s = _optimal_scale_weighted(opt_rr, gt_rr, vis)

    mpjpe_improved = opt_mpjpe_val < raw_mpjpe
    si_improved = opt_si < raw_si

    msg = (
        f"VW-MPJPE: raw={raw_mpjpe*100:.2f}, opt={opt_mpjpe_val*100:.2f} "
        f"({'improved' if mpjpe_improved else 'WORSE'})\n"
        f"         VW-SI-MPJPE: raw={raw_si*100:.2f}, opt={opt_si*100:.2f} "
        f"({'improved' if si_improved else 'WORSE'})\n"
        f"         Scales: raw={raw_s:.4f}, opt={opt_s:.4f}"
    )

    if mpjpe_improved and not si_improved:
        return False, f"INCONSISTENT: MPJPE improved but SI-MPJPE got worse!\n         {msg}"

    return True, msg


def main():
    print("=" * 70)
    print("  SI-MPJPE Metric Consistency Tests")
    print("=" * 70)

    # Load a few examples
    test_examples = [
        "171204_pose3_2000",   # worst regression
        "171204_pose1_5000",   # good improvement
        "171204_pose2_10000",  # best improvement
    ]

    all_passed = True

    for example_name in test_examples:
        print(f"\n--- {example_name} ---")
        gt_arr, raw_arr, opt_arr, camera = load_example(example_name)

        # Root-relative, eval joints
        pred_rr = root_relative(raw_arr)[:, EVAL_JOINTS, :]
        gt_rr = root_relative(gt_arr)[:, EVAL_JOINTS, :]
        vis = compute_visibility_weights(gt_arr, camera)[:, EVAL_JOINTS]

        # Test 1
        ok, msg = test_si_scale_minimizes_mpjpe(pred_rr, gt_rr)
        status = "PASS" if ok else "FAIL"
        print(f"  Test 1 (SI scale minimizes MPJPE): {status} - {msg}")
        if not ok:
            all_passed = False

        # Test 2
        ok, msg = test_vw_si_consistent_with_vw_mpjpe(pred_rr, gt_rr, vis)
        status = "PASS" if ok else "FAIL"
        print(f"  Test 2 (VW-SI consistent with VW-MPJPE): {status} - {msg}")
        if not ok:
            all_passed = False

        # Test 3
        ok, msg = test_scale_1_worse_than_optimal(pred_rr, gt_rr, vis)
        status = "PASS" if ok else "FAIL"
        print(f"  Test 3 (scale=1.0 >= optimal): {status} - {msg}")
        if not ok:
            all_passed = False

        # Test 4
        ok, msg = test_optimization_doesnt_hurt_scale_invariant(gt_arr, raw_arr, opt_arr, camera)
        status = "PASS" if ok else "FAIL"
        print(f"  Test 4 (optimization SI consistency): {status}")
        for line in msg.split("\n"):
            print(f"         {line}")
        if not ok:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
