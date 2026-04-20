"""Test hypothesis: old run's low VW-SI-MPJPE was artificially deflated
because many joints projected off-screen, making them "invisible" in the
visibility-weighted metric. New run places joints correctly on-screen,
so visibility weighting no longer masks the error.

Evidence we're looking for:
1. Old run has many joints projecting outside image bounds (negative pixels, etc.)
2. Old run's unweighted SI-MPJPE is similar to or worse than new run
3. Old run's "visible" joint count is much lower than new run
4. Per-joint: joints that were off-screen in old run are the ones with huge
   apparent improvement in VW metrics

Usage:
    uv run python experiment/visibility_masking_hypothesis.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import compute_visibility_weights
from camera import Camera
from skeleton import EVAL_JOINTS, JOINT_NAMES

OLD_RESULTS = "/home/kky2806/pose-estimation-output/run_2026_04_14_14_45/results.json"
NEW_RESULTS = "/home/kky2806/pose-estimation-output/run_2026_04_19_12_16/results.json"
OLD_RUN_DIR = "/home/kky2806/pose-estimation-output/run_2026_04_14_14_45"
NEW_RUN_DIR = "/home/kky2806/pose-estimation-output/run_2026_04_19_12_16"

IMAGE_W, IMAGE_H = 1920, 1080


def load_trajectories(run_dir: str, example_name: str, pipeline: str = "motionbert"):
    path = os.path.join(run_dir, example_name, pipeline, "trajectories.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def count_offscreen_joints(traj: dict, margin: float = 0.0) -> dict:
    """Count how many joint-frames project outside image bounds."""
    cam_data = traj["camera"]
    fx, fy = cam_data["fx"], cam_data["fy"]
    cx, cy = cam_data["cx"], cam_data["cy"]

    raw = np.array(traj["raw_prediction"])  # (F, 16, 3)
    F = raw.shape[0]

    offscreen_counts = np.zeros(16, dtype=int)
    total_frames = F

    for f in range(F):
        for j in range(16):
            xyz = raw[f, j]
            if abs(xyz[2]) < 1e-6:
                offscreen_counts[j] += 1
                continue
            u = fx * xyz[0] / xyz[2] + cx
            v = fy * xyz[1] / xyz[2] + cy
            if u < -margin or u > IMAGE_W + margin or v < -margin or v > IMAGE_H + margin:
                offscreen_counts[j] += 1

    return {
        "offscreen_per_joint": offscreen_counts.tolist(),
        "total_offscreen": int(offscreen_counts.sum()),
        "total_joint_frames": total_frames * 16,
        "pct_offscreen": float(offscreen_counts.sum()) / (total_frames * 16) * 100,
        "joints_always_offscreen": [
            JOINT_NAMES[j] for j in range(16) if offscreen_counts[j] == total_frames
        ],
    }


def compute_visibility_from_traj(traj: dict) -> np.ndarray:
    """Recompute visibility weights from trajectories using GT."""
    if traj.get("ground_truth") is None:
        return None
    gt_list = traj["ground_truth"]
    cam_data = traj["camera"]
    camera = Camera(
        fx=cam_data["fx"], fy=cam_data["fy"],
        cx=cam_data["cx"], cy=cam_data["cy"],
        image_size=(IMAGE_W, IMAGE_H),
    )
    gt_valid = [g for g in gt_list if g is not None]
    if not gt_valid:
        return None
    gt_arr = np.array(gt_valid)
    return compute_visibility_weights(gt_arr, camera)  # (F, 16)


def main():
    old_results = json.load(open(OLD_RESULTS))
    new_results = json.load(open(NEW_RESULTS))

    old_mb = {r["name"]: r for r in old_results.get("motionbert", [])}
    new_mb = {r["name"]: r for r in new_results.get("motionbert", [])}

    # Focus on sequences where old was "good" but new is "bad"
    regressions = []
    for name in sorted(set(old_mb) & set(new_mb)):
        ov = old_mb[name].get("opt_vw_si_mpjpe")
        nv = new_mb[name].get("opt_vw_si_mpjpe")
        if ov and nv:
            regressions.append((name, ov * 100, nv * 100, (nv - ov) * 100))

    regressions.sort(key=lambda x: x[3], reverse=True)

    print("=" * 80)
    print("HYPOTHESIS: Old run's VW-SI-MPJPE was artificially low due to off-screen")
    print("joints being masked by visibility weighting.")
    print("=" * 80)

    print(f"\n{'Example':<35} {'Old VW-SI':>9} {'New VW-SI':>9} {'Old SI':>9} {'New SI':>9} {'Delta VW':>9}")
    print("-" * 85)
    for name, ov, nv, delta in regressions[:20]:
        old_si = old_mb[name].get("opt_si_mpjpe", 0) * 100
        new_si = new_mb[name].get("opt_si_mpjpe", 0) * 100
        print(f"{name:<35} {ov:>9.2f} {nv:>9.2f} {old_si:>9.2f} {new_si:>9.2f} {delta:>+9.2f}")

    print("\n" + "=" * 80)
    print("OFF-SCREEN JOINT ANALYSIS (top 10 regressing sequences)")
    print("=" * 80)

    for name, ov, nv, delta in regressions[:10]:
        old_traj = load_trajectories(OLD_RUN_DIR, name)
        new_traj = load_trajectories(NEW_RUN_DIR, name)

        if old_traj is None or new_traj is None:
            print(f"\n{name}: trajectories not found")
            continue

        old_off = count_offscreen_joints(old_traj)
        new_off = count_offscreen_joints(new_traj)

        print(f"\n{name}  (VW-SI: {ov:.1f} -> {nv:.1f} cm, delta={delta:+.1f})")
        print(f"  OLD off-screen: {old_off['pct_offscreen']:.1f}% of joint-frames "
              f"({old_off['total_offscreen']}/{old_off['total_joint_frames']})")
        print(f"  NEW off-screen: {new_off['pct_offscreen']:.1f}% of joint-frames "
              f"({new_off['total_offscreen']}/{new_off['total_joint_frames']})")
        if old_off["joints_always_offscreen"]:
            print(f"  OLD joints always off-screen: {old_off['joints_always_offscreen']}")
        if new_off["joints_always_offscreen"]:
            print(f"  NEW joints always off-screen: {new_off['joints_always_offscreen']}")

        # Per-joint off-screen counts
        old_per = np.array(old_off["offscreen_per_joint"])
        new_per = np.array(new_off["offscreen_per_joint"])
        F_old = len(old_traj["raw_prediction"])
        F_new = len(new_traj["raw_prediction"])
        print(f"  Per-joint off-screen % (OLD | NEW):")
        for j in EVAL_JOINTS:
            op = old_per[j] / F_old * 100
            np_ = new_per[j] / F_new * 100
            if op > 10 or np_ > 10:
                print(f"    {JOINT_NAMES[j]:<12}: OLD {op:5.1f}%  NEW {np_:5.1f}%")

    print("\n" + "=" * 80)
    print("AGGREGATE: SI-MPJPE (unweighted, scale-invariant) comparison")
    print("Shows true error without visibility masking")
    print("=" * 80)
    old_si_vals = [r["opt_si_mpjpe"] * 100 for r in old_mb.values() if r.get("opt_si_mpjpe")]
    new_si_vals = [r["opt_si_mpjpe"] * 100 for r in new_mb.values() if r.get("opt_si_mpjpe")]
    old_vw_vals = [r["opt_vw_si_mpjpe"] * 100 for r in old_mb.values() if r.get("opt_vw_si_mpjpe")]
    new_vw_vals = [r["opt_vw_si_mpjpe"] * 100 for r in new_mb.values() if r.get("opt_vw_si_mpjpe")]

    print(f"\n  OLD: SI-MPJPE={np.mean(old_si_vals):.2f}  VW-SI-MPJPE={np.mean(old_vw_vals):.2f}")
    print(f"  NEW: SI-MPJPE={np.mean(new_si_vals):.2f}  VW-SI-MPJPE={np.mean(new_vw_vals):.2f}")
    print(f"\n  Gap (SI vs VW-SI):")
    print(f"    OLD gap: {np.mean(old_si_vals) - np.mean(old_vw_vals):+.2f} cm "
          f"(VW-SI was {np.mean(old_si_vals) - np.mean(old_vw_vals):.2f} cm lower)")
    print(f"    NEW gap: {np.mean(new_si_vals) - np.mean(new_vw_vals):+.2f} cm")
    print()
    print("  If hypothesis is TRUE: OLD gap should be much larger than NEW gap,")
    print("  meaning old VW-SI was deflated by masking off-screen joints.")


if __name__ == "__main__":
    main()
