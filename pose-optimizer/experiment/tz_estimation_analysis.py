"""Experiment: Trace why Tz (depth estimate) changed between old and new run.

Hypothesis: Changing joint 9 from Nose->HeadTop shifts the distribution of
Tz candidates in motionbert_to_camera_space, changing the IQR-filtered median
and therefore the entire skeleton's depth placement.

Tests:
1. Simulate old vs new Tz estimation on a synthetic skeleton
2. Show which joint pairs dominate the Tz estimate
3. Identify whether joint 9 pairs are driving the difference

Run locally:
    uv run python experiment/tz_estimation_analysis.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from skeleton import JOINT_NAMES, PARENTS, DEFAULT_BONE_LENGTHS, NUM_JOINTS
from run_motionbert.detect import _iqr_filtered_median, _RELIABLE_BONES_FOR_SCALE


def simulate_tz_estimation(
    positions_3d_norm: np.ndarray,  # (16, 3) MB output, skeleton-space
    kp_2d: np.ndarray,              # (16, 2) SH detected pixel coords
    fx: float, fy: float, cx: float, cy: float,
    label: str,
) -> tuple[float, float, list]:
    """Replicate motionbert_to_camera_space logic, return (tz, bone_scale, candidates)."""
    # Step 1: bone scale
    root_relative = positions_3d_norm - positions_3d_norm[0:1]
    reliable_ratios = []
    all_ratios = []
    for j in range(1, NUM_JOINTS):
        p = int(PARENTS[j])
        det_bl = float(np.linalg.norm(root_relative[j] - root_relative[p]))
        ref_bl = float(DEFAULT_BONE_LENGTHS[j])
        if det_bl > 1e-4 and ref_bl > 1e-4:
            ratio = ref_bl / det_bl
            all_ratios.append(ratio)
            if j in _RELIABLE_BONES_FOR_SCALE:
                reliable_ratios.append(ratio)

    if len(reliable_ratios) >= 4:
        bone_scale = _iqr_filtered_median(np.array(reliable_ratios))
    elif all_ratios:
        bone_scale = _iqr_filtered_median(np.array(all_ratios))
    else:
        bone_scale = 1.0

    root_relative_m = root_relative * bone_scale

    # Step 2: Tz candidates
    tz_candidates = []
    tz_by_pair = []
    for i in range(NUM_JOINTS):
        for j in range(NUM_JOINTS):
            if i == j:
                continue
            if np.linalg.norm(kp_2d[i]) <= 1.0 or np.linalg.norm(kp_2d[j]) <= 1.0:
                continue
            dy_3d = abs(float(root_relative_m[i, 1]) - float(root_relative_m[j, 1]))
            if dy_3d < 0.001:
                continue
            dv_2d = abs(float(kp_2d[i, 1]) - float(kp_2d[j, 1]))
            if dv_2d > 5.0:
                tz_est = fy * dy_3d / dv_2d
                tz_candidates.append(tz_est)
                tz_by_pair.append((i, j, tz_est, dy_3d, dv_2d))

    if len(tz_candidates) >= 2:
        tz = _iqr_filtered_median(np.array(tz_candidates))
        tz = float(np.clip(tz, 1.0, 8.0))
    else:
        tz = 3.0

    print(f"\n{label}:")
    print(f"  bone_scale = {bone_scale:.4f}")
    print(f"  tz_candidates: n={len(tz_candidates)}, "
          f"min={min(tz_candidates):.2f}, max={max(tz_candidates):.2f}, "
          f"median={np.median(tz_candidates):.2f}, IQR_median={tz:.2f}")

    # Which pairs involve joint 9?
    j9_pairs = [(i, j, tz_e) for i, j, tz_e, _, _ in tz_by_pair if i == 9 or j == 9]
    non_j9 = [tz_e for i, j, tz_e, _, _ in tz_by_pair if i != 9 and j != 9]
    print(f"  Joint-9 pairs: n={len(j9_pairs)}, "
          f"mean_tz={np.mean([t for _,_,t in j9_pairs]):.2f} (if any)")
    if non_j9:
        print(f"  Non-joint-9 pairs: n={len(non_j9)}, mean_tz={np.mean(non_j9):.2f}")
        non_j9_iqr = _iqr_filtered_median(np.array(non_j9))
        print(f"  Tz from non-joint-9 pairs only: {non_j9_iqr:.2f}m")

    return tz, bone_scale, tz_candidates


def build_realistic_skeleton(tz_true: float, person_height_m: float = 1.7) -> tuple:
    """Build a synthetic skeleton at a given true depth, return 3D positions and 2D projections."""
    # Rough skeleton in camera space (Y-down, person facing camera)
    # Positions relative to realistic human proportions
    scale = person_height_m / 1.7  # scale to actual height
    joints = np.array([
        [0.0,    0.0,   tz_true],   # 0  Pelvis (root)
        [-0.12,  0.1,   tz_true],   # 1  RHip
        [-0.12,  0.52,  tz_true],   # 2  RKnee
        [-0.12,  0.92,  tz_true],   # 3  RAnkle
        [ 0.12,  0.1,   tz_true],   # 4  LHip
        [ 0.12,  0.52,  tz_true],   # 5  LKnee
        [ 0.12,  0.92,  tz_true],   # 6  LAnkle
        [0.0,   -0.22,  tz_true],   # 7  Spine
        [0.0,   -0.44,  tz_true],   # 8  Neck
        [0.0,   -0.57,  tz_true],   # 9  NOSE (old) -- between Neck and HeadTop
        [-0.18, -0.44,  tz_true],   # 10 LShoulder
        [-0.46, -0.44,  tz_true],   # 11 LElbow
        [-0.71, -0.44,  tz_true],   # 12 LWrist
        [ 0.18, -0.44,  tz_true],   # 13 RShoulder
        [ 0.46, -0.44,  tz_true],   # 14 RElbow
        [ 0.71, -0.44,  tz_true],   # 15 RWrist
    ]) * scale
    # Offset so person is centered
    joints[:, 0] += 0.0
    joints[:, 1] += 0.2
    return joints


def project(cam_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Project (16,3) camera coords to (16,2) pixel coords."""
    px = fx * cam_3d[:, 0] / cam_3d[:, 2] + cx
    py = fy * cam_3d[:, 1] / cam_3d[:, 2] + cy
    return np.stack([px, py], axis=-1)


def normalize_like_motionbert(cam_3d: np.ndarray) -> np.ndarray:
    """Simulate MotionBERT output: root-relative, normalized by hip width."""
    rr = cam_3d - cam_3d[0:1]  # root-relative
    # MotionBERT normalizes by some scale; we'll use hip width
    hip_width = float(np.linalg.norm(cam_3d[1] - cam_3d[4]))
    return rr / (hip_width + 1e-8)


def main():
    fx, fy = 1633.34, 1628.84
    cx, cy = 942.256, 557.344

    print("=" * 70)
    print("Tz ESTIMATION ANALYSIS: How joint 9 change affects depth estimate")
    print("=" * 70)

    for true_tz in [1.5, 2.0, 3.0]:
        print(f"\n{'='*70}")
        print(f"TRUE Tz = {true_tz}m (ground truth depth)")
        print(f"{'='*70}")

        # Build skeleton at true depth
        cam_3d = build_realistic_skeleton(true_tz)

        # HeadTop is ~0.15m above Nose in Y (Y-down, so HeadTop has smaller Y)
        headtop_y_offset = -0.15  # relative to Nose position
        cam_3d_headtop = cam_3d.copy()
        cam_3d_headtop[9, 1] += headtop_y_offset  # move joint 9 up to HeadTop

        # 2D projections (ground truth 2D - what SH would detect)
        # OLD: 2D[9] = HeadTop pixel (SH always detected HeadTop at MPII[9])
        kp_2d_headtop = project(cam_3d_headtop, fx, fy, cx, cy)  # SH always sees HeadTop

        # MotionBERT normalized output
        mb_norm_nose    = normalize_like_motionbert(cam_3d)          # OLD: joint 9 = Nose
        mb_norm_headtop = normalize_like_motionbert(cam_3d_headtop)  # NEW: joint 9 = HeadTop

        # OLD: 3D joint 9 = Nose position, 2D joint 9 = HeadTop pixel
        tz_old, scale_old, _ = simulate_tz_estimation(
            mb_norm_nose, kp_2d_headtop, fx, fy, cx, cy,
            f"OLD (3D joint9=Nose, 2D joint9=HeadTop pixel)"
        )

        # NEW: 3D joint 9 = HeadTop position, 2D joint 9 = HeadTop pixel (consistent)
        tz_new, scale_new, _ = simulate_tz_estimation(
            mb_norm_headtop, kp_2d_headtop, fx, fy, cx, cy,
            f"NEW (3D joint9=HeadTop, 2D joint9=HeadTop pixel)"
        )

        print(f"\n  True Tz: {true_tz:.2f}m  |  Old estimate: {tz_old:.2f}m  "
              f"|  New estimate: {tz_new:.2f}m")
        print(f"  Old error: {abs(tz_old - true_tz):.2f}m  |  "
              f"New error: {abs(tz_new - true_tz):.2f}m")
        better = "NEW" if abs(tz_new - true_tz) < abs(tz_old - true_tz) else "OLD"
        print(f"  => {better} estimate is closer to ground truth")


if __name__ == "__main__":
    main()
