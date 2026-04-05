# Final Report — Heatmap Overlay Fix

**Branch**: `refactor-2026-04-04`  
**Spec**: `agent-team/specs/2026-04-04-21-39.md`

## Root Cause

`overlay_video.py:_resize_heatmap_to_frame()` had an `sx > 5.0` heuristic to distinguish SH heatmaps (256px crop space) from synthetic Gaussian heatmaps (64px heatmap space). For high-resolution CMU Panoptic videos (1920x1080) with large bounding boxes, `sx = crop_size / 256` exceeded 5.0, incorrectly triggering the synthetic path. This computed `crop_w = 64 * sx` (~360px) instead of `256 * sx` (~1440px), making heatmaps 4x too small.

## Fix

Removed the heuristic and synthetic branch entirely (commit `7e39fb8`). Since the pipeline unification (Phase 1), all heatmaps are Stacked Hourglass — the synthetic path is dead code.

## Verification

Both MotionBERT and MediaPipe overlay videos visually confirmed: heatmaps align with the person, all skeleton overlays (detector, optimized, GT) are correct.

## Known Limitation

The standalone `python -m run_mediapipe` entrypoint still uses synthetic heatmaps for overlays. This only affects the legacy entrypoint, not the unified `main.py`.
