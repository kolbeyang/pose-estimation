# Developer Report: Phase 2, Iteration 6

## Changes Made

### 1. Bug Fix: Color Normalization in Stacked Hourglass (detect.py)
Added RGB channel mean subtraction after `/255.0` normalization in `run_hourglass()`, matching the official `HumanPosePredictor` preprocessing:
- `img[0] -= 0.4404` (R), `img[1] -= 0.4440` (G), `img[2] -= 0.4327` (B)
- Applied to both the original and flipped image paths.

### 2. Bug Fix: LayerNorm Epsilon in MotionBERT (detect.py)
Added `norm_layer=partial(nn.LayerNorm, eps=1e-6)` to the DSTformer constructor, matching the official MotionBERT config. Added `from functools import partial` and `import torch.nn as nn`.

### 3. Config Change: TARGET_FPS = 30.0 (config.py)
Changed from 10.0 to 30.0, using native video frame rate. This gives MotionBERT 100-150 frames of temporal context instead of 34-50.

## Per-Change Impact (test_single.py, example 0)

| Config | Det MPJPE | Det MPJPE (no ankles) | 2D Reproj Error |
|--------|-----------|----------------------|-----------------|
| Baseline (no changes, 10fps) | 24.75 cm | 14.90 cm | 161.3 px |
| + Color norm only (10fps) | 51.89 cm | 19.87 cm | 46.3 px |
| + Color norm + epsilon (10fps) | 51.89 cm | 19.87 cm | 46.3 px |
| + All fixes (30fps) | 51.07 cm | 20.98 cm | 27.0 px |

**Key observation:** The color normalization fix dramatically improved 2D keypoint quality (reprojection error dropped from 161 px to 46 px), but the overall MPJPE got worse because the knee/ankle 3D predictions degraded. The epsilon fix had zero measurable impact (as expected).

## Full Pipeline Results (main.py, all 10 examples)

| Example | Det MPJPE | Opt MPJPE | Det P-MPJPE |
|---------|-----------|-----------|-------------|
| pose1_sample_0 | 51.07 | 44.59 | 41.88 |
| pose2_200 | 41.33 | 37.31 | 29.51 |
| pose2_5000 | 17.12 | 16.79 | 20.52 |
| pose2_15000 | 35.73 | 33.45 | 38.09 |
| pose3_200 | 74.37 | 72.23 | 36.27 |
| pose3_4000 | 14.55 | 14.18 | 18.59 |
| ultimatum1_200 | 60.99 | 59.81 | 41.81 |
| ultimatum1_10000 | 63.23 | 64.45 | 25.79 |
| pose2_10000 | 16.62 | 16.44 | 17.89 |
| pose2_25000 | 18.60 | 17.87 | 22.41 |
| **MEAN** | **39.36** | **37.71** | **29.28** |

### Comparison with Previous (P2_05 baseline)

| Metric | P2_05 | P2_06 | Delta |
|--------|-------|-------|-------|
| Mean Det MPJPE | 29.09 cm | 39.36 cm | +10.27 cm (worse) |
| Mean Det P-MPJPE | 21.98 cm | 29.28 cm | +7.30 cm (worse) |
| Mean Opt MPJPE | 28.20 cm | 37.71 cm | +9.51 cm (worse) |

## Analysis

The overall MPJPE regressed significantly (+10 cm). However, this masks a bimodal distribution:

**Improved examples** (MPJPE < 20 cm): pose2_5000 (17.12), pose3_4000 (14.55), pose2_10000 (16.62), pose2_25000 (18.60) -- these are all strong results.

**Degraded examples** (MPJPE > 40 cm): pose1_sample_0 (51.07), pose3_200 (74.37), ultimatum1_200 (60.99), ultimatum1_10000 (63.23) -- massive knee/ankle errors dominate.

The root cause is clear from per-joint analysis: the color normalization fix produces more accurate 2D keypoints (confirmed by 3-4x reduction in reprojection error), but the `motionbert_to_camera_space()` bone-length-based depth estimation is being thrown off. With the old (incorrect) normalization, the 2D keypoints had systematic biases that happened to partially cancel out during 3D lifting for some examples. With correct normalization, MotionBERT's 3D output has different characteristics that interact poorly with the pairwise depth estimation heuristic.

The knee bone lengths are wildly off (2-3x default), suggesting MotionBERT's normalized 3D output places knees incorrectly in depth, and the bone-length scaling amplifies this error.

## Recommendation for Next Iteration

The detection pipeline fixes are correct (confirmed by reprojection improvement). The bottleneck is now in `motionbert_to_camera_space()` -- specifically the bone-length scaling and depth estimation. Consider:
1. Using a different depth estimation approach (e.g., direct regression of root depth)
2. Investigating why MotionBERT's leg predictions are unreliable for certain poses
3. Applying bone-length constraints during or after 3D lifting
4. Potentially clamping bone-length ratios to prevent extreme scaling

## Files Changed
- `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/detect.py` -- Color normalization, LayerNorm epsilon
- `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/config.py` -- TARGET_FPS 10.0 -> 30.0
