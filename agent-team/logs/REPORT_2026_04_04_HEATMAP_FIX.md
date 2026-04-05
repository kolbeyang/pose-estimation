# Heatmap Overlay Alignment Fix Report

**Date:** 2026-04-04
**Branch:** refactor-2026-04-04
**Commit:** 7e39fb8

## Iteration 1

### Observe

Ran `main.py configs/motionbert-single.json` and extracted frame 15 from the overlay video. Findings:
- The person was centered in frame with arms spread
- Skeletons (green, red, blue) were roughly aligned with the person
- Yellow 2D detection dots were on the person
- Heatmaps were barely visible as a faint glow in the upper-left corner, not on the person

### Hypothesize

Root cause: `_resize_heatmap_to_frame()` in `overlay_video.py` line 91 used an `sx > 5.0` heuristic to distinguish SH heatmaps from synthetic heatmaps.

For this video (1920x1080, large bounding box):
- Union bbox: (413, 34) to (1610, 1072)
- Crop side with 20% padding: 1436 pixels
- `sx = 1436 / 256 = 5.609`, which is > 5.0

This triggered the "synthetic" path, which computed `crop_w = 64 * 5.609 = 359` instead of the correct `256 * 5.609 = 1436`. The heatmap was resized to ~359x359 (4x too small) and placed at offset (293, -165), putting it mostly off-screen in the upper-left corner.

Since the refactor unified both pipelines to use Stacked Hourglass, the synthetic path is never needed.

### Implement

Removed the `sx > 5.0` heuristic and the synthetic branch entirely. The function now always uses the SH path: `crop_w = 256 * sx`, `crop_h = 256 * sy`. Updated docstring accordingly.

**File changed:** `pose-optimizer/overlay_video.py` -- `_resize_heatmap_to_frame()` (7 insertions, 57 deletions)

### Verify

Re-ran both pipelines:
- `main.py configs/motionbert-single.json` -- heatmaps now correctly aligned with the person (head, elbows, wrists, knees all glowing)
- `main.py configs/mediapipe-single.json` -- heatmaps also correctly aligned

### Result: PASS

## Final State

The heatmap overlay alignment issue is fixed for both MotionBert and MediaPipe pipelines. The root cause was exactly as predicted in the plan (Hypothesis A). No remaining issues.
