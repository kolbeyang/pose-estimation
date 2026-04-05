# Testing Report: Phase 3 - 2D Reprojected MPJPE

**Date**: 2026-04-04
**Evaluator**: Eve
**Branch**: refactor-2026-04-04

## Verdict: PASS

Phase 3 is correctly implemented. The new `reprojected_mpjpe_2d` metric is properly integrated across all three entrypoints and produces reasonable values.

## Code Review

### `evaluate.py` - `reprojected_mpjpe_2d()` (lines 390-422)

**Correct behavior:**
- Projects both predicted and GT to 2D via `camera.camera_to_image()`.
- Falls back to computing visibility from GT frame-boundary check (`camera.is_in_frame(gt_2d)`) when `visibility=None`.
- Returns visibility-weighted mean pixel distance.
- Handles edge case where all joints are invisible (falls back to unweighted mean).

**Integration in `evaluate()` (lines 482-485):**
- Passes `pred_eval` and `gt_eval` (already sliced to 14 eval joints).
- Passes `vis` which is the same GT-projection-based visibility used by all other VW metrics.
- Correctly computes as the 10th metric in the results dict.

### `main.py`

- `_METRIC_KEYS` (line 176-180): includes `"reprojected_mpjpe_2d"` -- saved to results.json.
- `_evaluate_and_collect_metrics` (lines 160-162): logs `Det 2D-MPJPE` and `Opt 2D-MPJPE` in pixels.

### `run_motionbert/__init__.py`

- `_METRIC_KEYS` (lines 314-318): includes `"reprojected_mpjpe_2d"`.
- Logging (lines 294-296): prints Det/Opt 2D-MPJPE.

### `run_mediapipe/__init__.py`

- `_METRIC_KEYS` (lines 296-300): includes `"reprojected_mpjpe_2d"`.
- Logging (lines 277-278): prints Det/Opt 2D-MPJPE.

## Spec Compliance

**Spec requirement**: "Only include points that are within the frame boundaries (should be equivalent to the keypoint visibility scores with the way those are calculated)."

**Finding**: The function accepts an optional `visibility` parameter. When called from `evaluate()`, it receives the same `compute_visibility_weights()` output used by all other VW metrics -- which projects GT 3D points to 2D via `camera.camera_to_image()` then checks `camera.is_in_frame()`. When `visibility=None`, the function computes the same check internally from the GT 2D projection. This is equivalent to keypoint visibility scores as required.

## Smoke Test

**Command**: `cd pose-optimizer && uv run python main.py configs/motionbert-single.json`

**Result**: SUCCESS

**Output (relevant lines)**:
```
Det 2D-MPJPE: 112.73 px
Opt 2D-MPJPE: 84.58 px
```

**results.json confirmed**: Both `raw_metrics.reprojected_mpjpe_2d` (112.73) and `metrics.reprojected_mpjpe_2d` (84.58) are present and serialized correctly.

**Value reasonableness**: The 112/84 px values exceed the anticipated 5-50 px range. However, this is consistent with the large 3D errors in this example (Det MPJPE = 62 cm, Opt MPJPE = 60 cm). At a focal length of ~1630 px with subjects a few meters from camera, a 50 cm 3D offset reprojects to ~100 px easily. The optimization reduces the metric by 25% (112 -> 84), which tracks the 3D improvement direction. This is not a bug; better-performing examples should yield lower pixel errors.

## Issues Found

No blocking issues. One informational item added to TODO.md:
- The 2D-MPJPE values are higher than the 5-50 px range mentioned in the task brief, but this is explained by the large 3D error on this particular example. Team should verify on better examples.
