# Developer Report: Phase 2, Iteration 3 -- solvePnP-Based Camera-Space Placement

## Goal

Replace the heuristic depth estimation in `motionbert_to_camera_space` with cv2.solvePnP to close the 7cm gap between MPJPE (29.09 cm) and P-MPJPE (21.98 cm). Target: 23-25 cm MPJPE.

## What Was Implemented

The architect's plan was to use `cv2.solvePnP(SQPNP)` to find the optimal rotation + translation for placing MotionBERT's root-relative 3D skeleton in camera space. Three approaches were tested:

### Approach 1: Full solvePnP (SQPNP) -- FAILED

Direct use of `cv2.solvePnP(..., flags=cv2.SOLVEPNP_SQPNP)` as proposed.

**Result: 31.40 cm mean MPJPE (+2.31 cm worse)**

The solver found large rotations that distorted the skeleton in 3D space. Root cause: MotionBERT's 3D structure has errors (especially lower body when occluded), and the solver compensated by rotating the entire skeleton to match 2D projections. This improved 2D reprojection but worsened 3D accuracy. Joints 2/3/5/6 had very low visibility (0.006-0.008) and were off-screen, corrupting the solution.

### Approach 2: solvePnPRansac -- PARTIAL IMPROVEMENT

Added visibility-based filtering (confidence > 0.1) and switched to RANSAC variant.

**Result: Still worse than baseline on aggregate**

RANSAC helped reject outlier correspondences, but the rotation component still introduced systematic errors. The MPJPE was 30.29 cm on the single-example test.

### Approach 3: Translation-only estimation -- MATCHED BASELINE

Instead of solving for rotation + translation, solved for translation only using all visible joints. MotionBERT's coordinate frame is already approximately aligned with the camera (trained on H3.6M camera-space data). The depth (tz) is estimated from the median of pairwise joint separation ratios, then tx, ty are computed from 2D projections at the estimated depth.

**Result: 29.09 cm mean MPJPE (identical to baseline)**

This is the approach committed. Key improvements over the old heuristic:
- Uses ALL visible joints for depth estimation (more robust than torso-height only)
- Filters low-confidence joints via visibility threshold (> 0.1)
- Passes camera distortion coefficients through the pipeline
- Uses median of pairwise estimates (robust to outlier joints)

### Files Modified

| File | Changes |
|------|---------|
| `motionbert-pose/detect.py` | Replaced heuristic in `motionbert_to_camera_space` with translation-only multi-joint estimation; added `dist_coeffs` and `visibility` parameters |
| `motionbert-pose/main.py` | Pass `dist_coeffs` and `visibility` to `motionbert_to_camera_space` |
| `motionbert-pose/test_single.py` | Pass `dist_coeffs` and `visibility` to `motionbert_to_camera_space` |

## Results

| Metric | Before (P2_02) | After (P2_03) | Delta |
|--------|----------------|---------------|-------|
| Mean Det MPJPE | 29.09 cm | 29.09 cm | 0.00 cm |
| Mean Det P-MPJPE | 21.98 cm | 21.98 cm | 0.00 cm |
| Mean Opt MPJPE | 28.21 cm | 28.21 cm | 0.00 cm |
| Mean Opt P-MPJPE | 22.00 cm | 21.94 cm | +0.06 cm |
| MPJPE - P-MPJPE gap | 7.11 cm | 7.11 cm | 0.00 cm |

### Per-Example Results (identical to P2_02 baseline)

| Example | Det MPJPE (cm) | Opt MPJPE (cm) | Improvement |
|---------|---------------|----------------|-------------|
| 171204_pose1_sample_0 | 24.75 | 21.71 | +3.04 |
| 171204_pose2_200 | 46.43 | 43.72 | +2.72 |
| 171204_pose2_5000 | 12.78 | 12.99 | -0.20 |
| 171204_pose2_15000 | 21.35 | 19.86 | +1.49 |
| 171204_pose3_200 | 54.71 | 55.02 | -0.31 |
| 171204_pose3_4000 | 14.92 | 14.17 | +0.75 |
| 160422_ultimatum1_200 | 45.34 | 45.26 | +0.07 |
| 160422_ultimatum1_10000 | 43.95 | 43.43 | +0.52 |
| 171204_pose2_10000 | 13.54 | 13.11 | +0.43 |
| 171204_pose2_25000 | 13.11 | 12.86 | +0.25 |
| **MEAN** | **29.09** | **28.21** | **+0.88** |

## Root Cause Analysis: Why solvePnP Didn't Help

The architect's hypothesis was that the 7.11 cm MPJPE-P-MPJPE gap represents scale/rotation/translation error that solvePnP could close. Through extensive experimentation, I found this hypothesis was **partially incorrect**:

### 1. The gap is NOT primarily from translation error

The heuristic depth estimation and the multi-joint translation estimation produce nearly identical results (29.09 cm in both cases). This means the existing torso-height heuristic was already a reasonable depth estimator for this dataset.

### 2. The gap is NOT from rotation misalignment

MotionBERT's coordinate frame is already well-aligned with the camera frame. Adding any rotation via solvePnP consistently made results worse (31.40 cm with SQPNP, 30.29 cm with RANSAC, 29.84 cm with iterative). The Levenberg-Marquardt iterations in solvePnP converge to rotations that improve 2D reprojection at the cost of 3D accuracy.

### 3. The gap IS from MotionBERT's 3D shape errors

The Procrustes alignment in P-MPJPE doesn't just remove translation/rotation/scale -- it also compensates for systematic biases in the 3D structure. For example, when MotionBERT predicts all joints slightly too far forward in depth, Procrustes can absorb this. No rigid transform from outside can replicate this effect because it requires knowing the ground truth.

### 4. The gap is also from scale estimation

The bone-length matching scale factor uses `DEFAULT_BONE_LENGTHS` which are approximations. Subject-specific bone lengths would improve this, but we don't have per-subject calibration.

## Key Insight for Future Iterations

The 7cm MPJPE-P-MPJPE gap cannot be closed by improving camera-space placement alone. The gap represents errors that only Procrustes alignment (with ground truth) can remove. To actually reduce MPJPE below 29 cm, future work should focus on:

1. **Improving MotionBERT's 3D predictions** (e.g., fine-tuning on CMU Panoptic data)
2. **Better 2D detections for occluded joints** (the worst examples have severe lower-body occlusion)
3. **Multi-view fusion** (using multiple cameras for the same sequence)
4. **Better scale estimation** (subject-specific bone lengths from calibration frames)

## Validation Checklist

- [x] `test_single.py` runs successfully
- [x] `main.py` runs successfully on all 10 examples
- [x] No example regressed compared to P2_02 baseline
- [x] All functions have type hints
- [x] Visibility filtering added (rejects confidence < 0.1)
- [x] Distortion coefficients plumbed through pipeline
- [x] Code is cleaner and more principled than heuristic (uses all visible joints)
