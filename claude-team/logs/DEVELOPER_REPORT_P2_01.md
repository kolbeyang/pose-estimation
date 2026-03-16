# Developer Report: Phase 2, Iteration 1 -- MPJPE Bug Fix

## Goal

Fix the systematic scale error in the MotionBERT pipeline's back-projection from normalized model output to camera-space meters. The old `pixel_aligned_to_camera_space` function produced bone lengths 2-5x too large, inflating MPJPE to ~44.89 cm.

## Root Cause Analysis

The architect identified three compounding bugs. During implementation, I confirmed the primary issue through detailed investigation:

**The old `pixel_aligned_to_camera_space` function** used MotionBERT's X,Y output as pixel coordinates for pinhole back-projection. This amplified errors because:
1. The MotionBERT X,Y predictions are imperfect approximations of pixel positions.
2. The torso-height depth heuristic (`root_depth = fx * 1.38 / pixel_height`) used `fx` instead of `fy`, and more critically, was highly sensitive to 2D detection quality -- bad ankle detections (e.g., off-screen in example 0) produced wildly wrong depth estimates (5.9m vs true 2.4m).
3. The back-projection formula `x_cam = (u - cx) * z_cam / fx` then magnified these errors by the overestimated depth.

**Key finding during implementation:** The architect's recommended "Option A" (pixel displacement approach) still relied on the torso-height depth heuristic for scale, which was the dominant error source. When 2D ankle detections were wrong, the depth estimate was 1.3-2.4x too large, directly scaling up all bone lengths.

## What Was Implemented

### Approach: Bone Length Matching (Option B from plan)

Instead of relying on the error-prone torso-height depth heuristic for scale, I implemented median bone length matching:

1. Compute root-relative structure from MotionBERT's normalized output
2. Compute bone lengths in normalized space
3. Compute scale factor as `median(reference_bone_lengths / detected_bone_lengths)`
4. Scale root-relative positions by this factor to get meters
5. Place root in camera space using 2D detection + depth estimation (depth heuristic still used for root placement only, not for scale)

This is more robust because:
- The median operation is insensitive to individual bad joint predictions
- It doesn't depend on 2D ankle detection quality for scale
- It directly maps the model's output structure to the correct physical scale

### Files Modified

| File | Changes |
|------|---------|
| `motionbert-pose/detect.py` | (1) Modified `run_motionbert` to return normalized positions + crop_scale params alongside pixel-aligned output. (2) Added new `motionbert_to_camera_space` function using bone length matching. (3) Updated `detect_poses` to return normalized positions and crop_scale params. (4) Marked old `pixel_aligned_to_camera_space` as deprecated. |
| `motionbert-pose/test_single.py` | Updated to use new `motionbert_to_camera_space` and receive extended return values from `detect_poses`. |
| `motionbert-pose/main.py` | Same updates as test_single.py. |

No new files were created.

## Results

### Before vs After Comparison (10 examples)

| Example | Old MPJPE (cm) | New MPJPE (cm) | Change |
|---------|---------------|----------------|--------|
| 171204_pose1_sample_0 | 58.04 | 24.75 | -33.29 |
| 171204_pose2_200 | 62.90 | 46.43 | -16.47 |
| 171204_pose2_5000 | 18.08 | 12.78 | -5.30 |
| 171204_pose2_15000 | 50.97 | 21.35 | -29.62 |
| 171204_pose3_200 | 54.38 | 54.71 | +0.33 |
| 171204_pose3_4000 | 39.57 | 14.92 | -24.65 |
| 160422_ultimatum1_200 | 59.62 | 45.34 | -14.28 |
| 160422_ultimatum1_10000 | 65.10 | 43.95 | -21.15 |
| 171204_pose2_10000 | 20.36 | 13.54 | -6.82 |
| 171204_pose2_25000 | 19.92 | 13.11 | -6.81 |
| **MEAN** | **44.89** | **29.09** | **-15.80** |

### P-MPJPE (unchanged, as expected)

| Metric | Old | New |
|--------|-----|-----|
| Mean P-MPJPE | 21.29 cm | 21.98 cm |
| MPJPE/P-MPJPE ratio | 2.11x | 1.32x |

### Bone Lengths (example 2, frame 0)

| Bone | Old (m) | New (m) | Expected (m) | New Ratio |
|------|---------|---------|-------------|-----------|
| RKnee | 0.606 | 0.406 | 0.420 | 0.97 |
| LElbow | 0.461 | 0.309 | 0.280 | 1.10 |
| LWrist | 0.372 | 0.249 | 0.250 | 1.00 |
| RElbow | 0.445 | 0.298 | 0.280 | 1.06 |
| RWrist | 0.346 | 0.232 | 0.250 | 0.93 |

### With Optimization (10 steps)

| Metric | New Det | New Opt | Improvement |
|--------|---------|---------|-------------|
| Mean MPJPE | 29.09 cm | 28.21 cm | +0.88 cm |
| Mean P-MPJPE | 21.98 cm | 21.95 cm | +0.03 cm |

Note: The optimization provides less improvement than before (0.88 cm vs 11.61 cm) because the bone length matching already fixes most of the scale error that the optimizer was previously correcting. The remaining errors are from bad 2D detections and MotionBERT prediction quality, which the FK optimizer has limited ability to fix in 10 steps.

## Validation Checklist

- [x] `test_single.py` runs successfully on example 0
- [x] `main.py` runs successfully on all 10 examples
- [x] MPJPE dropped significantly: 44.89 -> 29.09 cm (-35%)
- [x] P-MPJPE is roughly unchanged: 21.29 -> 21.98 cm
- [x] MPJPE/P-MPJPE ratio improved: 2.11 -> 1.32
- [x] Bone lengths are now within 20% of anatomical values (for examples with good 2D detections)
- [x] All functions have type hints
- [x] No new files created (all changes in existing files)

## Key Observations

1. **The fix is most impactful for examples with extreme depth errors.** Example 0 improved by 33 cm because the old depth estimate was 5.9m vs true 2.4m -- a 2.5x error that inflated all bone lengths.

2. **Some examples still have high MPJPE (40-55 cm).** These are cases where the 2D detector (Stacked Hourglass) fails badly, causing both poor MotionBERT predictions AND poor root placement. Bone length matching fixes the scale but can't fix fundamentally wrong 3D pose structure.

3. **One example slightly regressed** (171204_pose3_200: +0.33 cm). The bone length matching scale was slightly worse than the old heuristic for this particular case, but the difference is negligible.

4. **The optimization delta shrank from 11.6 cm to 0.9 cm** because the optimizer was previously compensating for scale errors. With correct scale, the remaining error is structural (bad pose predictions from bad 2D input), which requires more optimization steps or a different approach.

## Deviations from Plan

1. **Used bone length matching (Option B) instead of pixel displacement + depth (Option A).** During implementation, I discovered that Option A still suffered from the same torso-height depth estimation errors because the 2D ankle detections were frequently wrong (off-screen in some examples). Bone length matching is independent of 2D detection quality for scale estimation.

2. **Kept old `pixel_aligned_to_camera_space` as deprecated** rather than removing it, for reference.
