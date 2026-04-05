# Testing Report: Heatmap Overlay Fix Verification

**Date:** 2026-04-04
**Tester:** Evaluator Eve
**Branch:** refactor-2026-04-04
**Fix under test:** Removal of `sx > 5.0` heuristic in `overlay_video.py::_resize_heatmap_to_frame()`

## 1. Code Review

### `pose-optimizer/overlay_video.py` -- `_resize_heatmap_to_frame()` (lines 22-74)

**PASS.** The old `sx > 5.0` heuristic and the synthetic branch have been completely removed. The function now unconditionally uses the SH path:

```python
crop_w = max(1, int(round(256 * sx)))
crop_h = max(1, int(round(256 * sy)))
```

This is correct: `affine` maps [0..255] crop coords to original pixel coords, and the 64x64 heatmap covers the full 256x256 crop. The resize + paste logic with boundary clipping is clean and handles edge cases (off-screen regions).

No trace of `sx > 5.0`, `synthetic`, or the old branching logic remains in the function.

**Nit:** The module docstring (line 1) still says "real SH or synthetic Gaussian" but the synthetic path no longer exists.

## 2. Visual Test -- MotionBERT

**Config:** `configs/motionbert-single.json`
**Command:** `uv run python main.py configs/motionbert-single.json`
**Frame inspected:** Frame 15 (labeled "Frame 45" in video)

**PASS.** Observations:
- Heatmaps (HOT colormap glow) are centered on the person's joints -- head, shoulders, elbows, wrists, hips, knees, ankles all show visible heat
- Green skeleton (MotionBERT raw 3D projected) aligns with the person
- Red skeleton (optimized 3D projected) aligns with the person
- Blue skeleton (GT) is visible and overlaps the person
- Yellow 2D detection dots are on the person's joints
- No heatmaps stuck in a corner or off-screen

## 3. Visual Test -- MediaPipe

**Config:** `configs/mediapipe-single.json`
**Command:** `uv run python main.py configs/mediapipe-single.json`
**Frame inspected:** Frame 15 (labeled "Frame 45" in video)

**PASS.** Observations:
- Heatmaps correctly overlay the person's body (same SH heatmaps, correctly placed)
- Green skeleton (MediaPipe raw) aligns well with the person
- Red skeleton (optimized) aligns with the person
- Blue skeleton (GT) visible
- Yellow dots on joints

## 4. Summary

| Check | Result |
|---|---|
| `sx > 5.0` heuristic removed | PASS |
| Synthetic branch removed | PASS |
| Only SH path remains | PASS |
| MotionBERT heatmaps on person | PASS |
| MotionBERT skeletons aligned | PASS |
| MediaPipe heatmaps on person | PASS |
| MediaPipe skeletons aligned | PASS |
| GT skeleton visible (both) | PASS |

**Verdict: PASS.** The fix is correct and both pipelines produce properly aligned heatmap overlay videos.

## Issues Found

- **(low) Stale docstring:** Module docstring in `overlay_video.py` line 1 still references "synthetic Gaussian" heatmaps, but that code path no longer exists. Should be updated to just say "SH heatmaps." Added to TODO.md.
