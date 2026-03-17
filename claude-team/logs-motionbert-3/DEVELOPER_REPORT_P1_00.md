# Developer Report: Phase 1, Iteration 0 -- solvePnP + Remove ALL_JOINTS_SMOOTH_WEIGHT

**Date:** 2026-03-17
**Spec:** `claude-team/specs/motion-bert-round-5.md` (Phase 1)
**Architect Plan:** `claude-team/logs/ARCHITECT_PLAN_P1_00.md`

## What Was Implemented

1. **Removed `ALL_JOINTS_SMOOTH_WEIGHT`** from `config.py` and all references in `scoring.py` and `optimize.py`. This was dead code (weight was already 0.0) but the spec explicitly asked for removal.

2. **Replaced pairwise depth estimation with solvePnP** in `motionbert_to_camera_space()` in `detect.py`. The new approach:
   - Takes MotionBERT's root-relative 3D output (already scaled to meters via bone-length matching)
   - Uses 2D Stacked Hourglass keypoints as image points
   - Calls `cv2.solvePnP` with SQPNP to solve for the rigid transform (rotation + translation)
   - Applies the transform to get camera-space positions
   - Falls back to a person-height depth heuristic if solvePnP fails or root Z is outside 0.5-15m

3. **Removed unused helper functions** `_enforce_bone_lengths_with_2d()` and `_reconstruct_from_2d()` from `detect.py`.

## Files Changed

- `motionbert-pose/config.py` -- Removed `ALL_JOINTS_SMOOTH_WEIGHT` constant
- `motionbert-pose/scoring.py` -- Removed `motion_penalty_all_joints()` function and all `all_joints_smooth_weight` parameter usage from `compute_total_score()`
- `motionbert-pose/optimize.py` -- Removed `all_joints_smooth_weight=cfg.ALL_JOINTS_SMOOTH_WEIGHT` argument from `compute_total_score()` call
- `motionbert-pose/detect.py` -- Replaced depth estimation with solvePnP, removed `_enforce_bone_lengths_with_2d()` and `_reconstruct_from_2d()`

## Commands Run

### Example 0: `171204_pose1_sample_0` (100 frames)
```bash
cd motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-solvepnp-test')
os.makedirs(run_dir, exist_ok=True)
seq, cam, start, nf, pidx = cfg.EXAMPLES[0]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```
**Result:** Completed successfully. No errors.

### Example 5: `171204_pose3_4000` (150 frames)
```bash
cd motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-solvepnp-test')
seq, cam, start, nf, pidx = cfg.EXAMPLES[5]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```
**Result:** Completed successfully. No errors.

### solvePnP Diagnostics (Example 5, frame 0)
```
Bone scale: 0.6008
Valid joints: 14
solvePnP success: True
rvec magnitude: 0.3452 rad (19.8 deg)
tvec: [0.136, 0.390, 2.350]
Root Z: 2.3496
```

solvePnP succeeded on all frames for both examples (no fallbacks to the depth heuristic).

## Results Comparison

### Example 0: `171204_pose1_sample_0`

| Metric | Round 4 (old) | Round 5 (solvePnP) | Change |
|---|---|---|---|
| Det MPJPE (cm) | 30.98 | 34.72 | +3.74 (worse) |
| Opt MPJPE (cm) | 30.44 | 34.12 | +3.68 (worse) |
| Det P-MPJPE (cm) | 28.57 | 28.57 | 0.00 |
| Opt P-MPJPE (cm) | 28.42 | 28.30 | -0.12 (better) |
| Det MPJVE (cm/f) | 0.94 | 3.69 | +2.75 (worse) |
| Opt MPJVE (cm/f) | 1.05 | 3.38 | +2.33 (worse) |
| Root Z range | -- | 1.44 - 2.45 m | -- |
| GT root Z range | -- | 2.57 - 2.63 m | -- |

### Example 5: `171204_pose3_4000`

| Metric | Round 4 (old) | Round 5 (solvePnP) | Change |
|---|---|---|---|
| Det MPJPE (cm) | 15.85 | 18.26 | +2.41 (worse) |
| Opt MPJPE (cm) | 15.44 | 18.65 | +3.21 (worse) |
| Det P-MPJPE (cm) | 20.33 | 20.33 | 0.00 |
| Opt P-MPJPE (cm) | 20.22 | 20.52 | +0.30 (worse) |
| Det MPJVE (cm/f) | 0.51 | 0.93 | +0.42 (worse) |
| Opt MPJVE (cm/f) | 0.48 | 0.54 | +0.06 (worse) |
| Root Z range | -- | 2.27 - 2.34 m | -- |
| GT root Z range | -- | 2.40 - 2.41 m | -- |

### Key Observations

1. **P-MPJPE is essentially unchanged** for both examples. This means the skeleton *shape* is the same -- the MPJPE regression is purely a global alignment/depth issue. solvePnP is placing the skeleton at a slightly different (worse) global position.

2. **Root Z is systematically lower than GT** by 0.1-1.2m. solvePnP is placing the skeleton slightly too close to the camera.

3. **solvePnP rotation is non-trivial** (~20 degrees for example 5). MotionBERT's coordinate frame is approximately camera-aligned but not exactly. The rotation correction is legitimate but may be introducing depth estimation noise.

4. **MPJVE is significantly worse for example 0** (0.94 -> 3.69 cm/f pre-optimization). solvePnP runs independently per frame, producing noisy depth estimates. The wide root Z range (1.44-2.45m vs GT 2.57-2.63m) confirms frame-to-frame jitter. The optimizer reduces MPJVE from 3.69 to 3.38, showing smoothing works, but it is still much worse than baseline.

5. **Example 5 is less affected** because the person is at a more stable distance and the poses are less dynamic.

## Decisions Made

1. **Kept bone-length enforcement after solvePnP** as specified in the architect's plan. After solvePnP transforms the skeleton, bone-length constraints (max_ratio=1.3) are enforced on root-relative positions before re-adding the root position.

2. **Fallback uses person-height heuristic** rather than the simple 3.0m default. When solvePnP fails or produces unreasonable root Z, the fallback estimates depth from the ratio of 3D torso-to-ankle height vs 2D pixel height, matching the mediapipe-pose fallback approach.

3. **Distortion coefficients are passed through** from camera calibration to solvePnP, rather than always using zero distortion.

## Concerns

1. **MPJPE regression is significant.** Both tested examples show 2-4cm worse MPJPE with solvePnP vs the old pairwise method. The P-MPJPE being unchanged proves this is purely a depth/translation issue, not a shape issue.

2. **MPJVE regression is very significant for example 0.** The per-frame solvePnP produces much noisier depth estimates (root Z range 1.44-2.45m) than the old pairwise method. This contradicts the spec's requirement that "MPJVE should be improved for ALL test videos."

3. **The ~20-degree rotation from solvePnP suggests a coordinate frame mismatch.** MotionBERT's output coordinate frame may not perfectly match what solvePnP expects. The old pairwise method avoided this by only estimating translation (no rotation), which may have been a better match for MotionBERT's already-camera-aligned output.

4. **Potential mitigation ideas (for next iteration):**
   - Restrict solvePnP to translation-only by setting rotation to identity (use solvePnP only for tvec)
   - Use RANSAC variant of solvePnP to handle outlier 2D detections
   - Filter to only upper-body joints (excluding ankles) for solvePnP input
   - Apply temporal smoothing to solvePnP results before feeding to the optimizer
   - Consider reverting to pairwise method if solvePnP cannot be made to work better

## Deviations from Plan

None. All steps in the architect's plan were followed exactly as specified. The regression in metrics is an expected risk acknowledged in the plan (Risk 1: solvePnP rotation estimate may be wrong, Risk 3: per-frame solvePnP produces jittery z values).

## Output Artifacts

- Predictions: `motionbert-pose/training_runs/p1-solvepnp-test/predictions/`
  - `171204_pose1_sample_0.json`
  - `171204_pose3_4000.json`
- Graphs + overlay videos: `motionbert-pose/training_runs/p1-solvepnp-test/graphs/`
  - `171204_pose1_sample_0/` (including overlay video)
  - `171204_pose3_4000/` (including overlay video)
