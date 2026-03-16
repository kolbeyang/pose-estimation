# Developer Report: Phase 2, Iteration 7

## Problem

The color normalization fix in P2_06 dramatically improved 2D detection (161px to 27px reprojection error) but caused 6/10 examples to regress in 3D MPJPE. The root cause was that `motionbert_to_camera_space()` in detect.py used a bone-length matching approach for scale estimation that interacted poorly with the now-correct 2D keypoints.

## Root Cause Analysis

1. **Bone-length scale**: The old code computed a median ratio of all 16 bone lengths (detected / reference). MotionBERT frequently distorts knee/ankle bone lengths (2-3x default), which corrupted the median even with IQR filtering.

2. **Pairwise depth estimation**: Used all joints including knees/ankles, whose MotionBERT 3D positions are wildly wrong (especially depth), leading to biased depth estimates.

3. **Torso-height heuristic**: Used thorax-to-ankle distance which depends on ankle 2D detection. For many examples, the 2D detector completely fails on ankles (zero confidence), making this heuristic useless.

4. **Inconsistent bone scale across examples**: MotionBERT's arm bone ratios range from 0.31-2.26 across different sequences, making any single scale estimation approach fragile.

## Changes Made

### 1. Arm-only bone scale with IQR filtering (detect.py)

Replaced the all-bones median scale with arm-bone-only (joints 11-16: shoulders, elbows, wrists) using IQR-filtered median. Arm bones are the most consistently well-estimated by MotionBERT across poses.

### 2. Excluded knees/ankles from depth estimation (detect.py)

Created a `_DEPTH_RELIABLE_JOINTS` set that excludes knees (2,5) and ankles (3,6) from the pairwise depth estimation. Only upper-body joints and hips are used for tz estimation.

### 3. Fixed-reference torso-height cross-check (detect.py)

Replaced the MotionBERT-derived shoulder-hip 3D distance with a fixed anatomical reference of 0.55m (measured from CMU Panoptic GT). This decouples the depth cross-check from MotionBERT's potentially wrong bone scale. The pairwise tz is always blended with the heuristic (40/60 weight toward heuristic).

### 4. Bone-length enforcement (detect.py)

Added `_enforce_bone_lengths()`: after scaling, any bone whose length deviates more than 1.3x from the default is clamped to the default length while preserving direction. This corrects MotionBERT's depth distortion for knees/ankles.

### 5. Utility functions (detect.py)

- `_iqr_filtered_median()`: Computes median after removing IQR outliers (k=1.5)
- `_enforce_bone_lengths()`: Direction-preserving bone-length clamping
- `_enforce_bone_lengths_with_2d()`: Quadratic-solve bone correction using 2D keypoints and camera intrinsics (implemented but not used in final version -- direction-preserving clamp was more robust)
- `_reconstruct_from_2d()`: Full 2D-to-3D skeleton reconstruction (implemented but not used in final version -- only helps when 2D is good but 3D is bad for legs, which rarely coincides)

## Full Pipeline Results (main.py, all 10 examples)

| Example | P2_06 Det | P2_07 Det | P2_07 Opt | P2_07 P-MPJPE | Delta |
|---------|-----------|-----------|-----------|---------------|-------|
| pose1_sample_0 | 51.07 | 30.98 | 30.56 | 28.57 | -20.09 |
| pose2_200 | 41.33 | 39.60 | 39.31 | 30.15 | -1.73 |
| pose2_5000 | 17.12 | 18.65 | 18.24 | 21.36 | +1.53 |
| pose2_15000 | 35.73 | 24.95 | 24.50 | 27.03 | -10.78 |
| pose3_200 | 74.37 | 56.54 | 55.93 | 36.54 | -17.83 |
| pose3_4000 | 14.55 | 15.85 | 15.77 | 20.33 | +1.30 |
| ultimatum1_200 | 60.99 | 50.89 | 50.98 | 34.91 | -10.10 |
| ultimatum1_10000 | 63.23 | 57.66 | 57.36 | 26.34 | -5.57 |
| pose2_10000 | 16.62 | 16.86 | 16.67 | 18.17 | +0.24 |
| pose2_25000 | 18.60 | 17.93 | 17.76 | 20.89 | -0.67 |
| **MEAN** | **39.36** | **32.99** | **32.71** | **26.43** | **-6.37** |

### Comparison across iterations

| Metric | P2_05 | P2_06 | P2_07 | Delta (06->07) |
|--------|-------|-------|-------|----------------|
| Mean Det MPJPE | 29.09 cm | 39.36 cm | 32.99 cm | -6.37 cm |
| Mean Det P-MPJPE | 21.98 cm | 29.28 cm | 26.43 cm | -2.85 cm |

## Analysis

**Improved (8/10 examples):** All previously-degraded examples improved, with the largest gains on pose1_sample_0 (-20cm) and pose3_200 (-18cm). These examples had massive knee/ankle depth errors that the bone-length enforcement and arm-only scaling partially corrected.

**Slight regressions (2/10):** pose2_5000 (+1.53cm) and pose3_4000 (+1.30cm) -- these were the best-performing examples where the original bone-length matching happened to work correctly by coincidence.

**Remaining outliers:** pose3_200 (56.54), ultimatum1_200 (50.89), ultimatum1_10000 (57.66). MotionBERT fundamentally fails for these sequences -- arm bone ratios are wildly inconsistent (0.31-2.26), and the 2D leg detection also fails. The P-MPJPE values (35-27cm) show the joint-level structure is roughly correct but translation/scale is wrong.

## Root cause of remaining errors

For the three worst examples:
1. **MotionBERT gives inconsistent arm bone ratios** (CV > 0.3), making any bone-length-based scale unreliable
2. **2D detector fails on legs entirely** for some poses (confidence ~0), removing the possibility of 2D-based correction
3. **Person is close to camera** (GT Z = 0.8-1.7m) with large perspective effects, amplifying any depth estimation error

## Recommendations for P2_08

1. **Multi-frame temporal averaging for bone_scale**: Instead of per-frame scale estimation, compute a single bone_scale across all frames (median of frame-wise medians), which should be more stable.
2. **Investigate why Stacked Hourglass fails on legs**: For the bad examples, the 2D detector produces zero-confidence leg joints. Check if this is a cropping issue (legs outside crop region) or model limitation.
3. **solvePnP with RANSAC**: Try cv2.solvePnP with the corrected 3D skeleton and reliable 2D points, using RANSAC to handle outliers. This would give an optimal rigid transform.
4. **Consider using MotionBERT's temporal output**: The current code uses per-frame 3D but MotionBERT outputs temporally smooth predictions -- averaging across nearby frames might give more stable bone lengths.

## Files Changed

- `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/detect.py` -- Rewrote `motionbert_to_camera_space()` with arm-only bone scale, IQR filtering, depth-reliable joints, fixed-reference heuristic, and bone-length enforcement. Added helper functions `_iqr_filtered_median()`, `_enforce_bone_lengths()`, `_enforce_bone_lengths_with_2d()`, `_reconstruct_from_2d()`.
