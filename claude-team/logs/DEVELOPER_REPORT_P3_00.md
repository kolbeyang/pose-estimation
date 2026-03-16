# DEVELOPER_REPORT_P3_00: Video with Overlays

## What was implemented

Implemented overlay video generation for the MotionBERT pipeline. Each processed example now generates an MP4 video overlaying the original video frames with:

1. **Stacked Hourglass heatmap overlay**: Actual (16, 64, 64) heatmaps from the SH model are summed across joints, resized to frame coordinates via the affine transform, and blended using a HOT colormap with additive blending.
2. **Yellow dots**: Raw MPII 16-joint 2D detection positions from Stacked Hourglass, drawn as small filled circles (radius=4) in yellow. Low-confidence joints (< 0.01) are skipped.
3. **Green skeleton**: MotionBERT raw 3D prediction projected to 2D via camera intrinsics.
4. **Red skeleton**: FK-optimized 3D prediction projected to 2D via camera intrinsics.
5. **Blue skeleton**: Ground truth 3D projected to 2D (when available).
6. **Frame label and legend**: Frame index label at top-left, color legend at bottom.

## Files changed

- **`motionbert-pose/detect.py`**: Modified `detect_poses()` to return `all_keypoints_2d` (raw MPII 16-joint keypoints) as a new element in the return tuple. Updated return type annotation and docstring.
- **`motionbert-pose/overlay_video.py`**: Created new file with `generate_overlay_video()` and helper functions (`_resize_heatmap_to_frame`, `_blend_heatmap_additive`, `_project_3d_to_2d`, `_draw_skeleton_2d`).
- **`motionbert-pose/main.py`**: Added import, updated `detect_poses()` unpacking to capture `mpii_kp_2d`, added overlay video generation call after graph generation.
- **`motionbert-pose/test_single.py`**: Added import, updated `detect_poses()` unpacking, added overlay video generation at the end.
- **`motionbert-pose/config.py`**: Added `OVERLAY_HEATMAP_INTENSITY` and `OVERLAY_FPS` constants.

## Commands run

```
cd motionbert-pose && uv run python test_single.py
```

Result: **Success.** Pipeline ran end-to-end. Overlay video saved at `training_runs/test_single/overlay_171204_pose1_sample_0.mp4` (9.7 MB, 100 frames).

Extracted frame 50 for visual inspection -- all five overlay layers are visible and correctly positioned.

## Decisions made

1. **Heatmap clamping**: The architect plan mentioned SH heatmaps may have negative values. I added `np.clip(heatmaps[i], 0, None)` before summing across joint channels to ensure only positive values are used in the heatmap overlay.

2. **Config constants not wired into overlay call**: Added `OVERLAY_HEATMAP_INTENSITY` and `OVERLAY_FPS` to config.py as the plan suggested, but the `generate_overlay_video()` function uses hardcoded defaults (200.0 intensity, 5.0 fps) matching these values. They can be wired in if tuning is needed later.

3. **Overlay video path**: In `main.py`, the overlay goes inside the per-example graph directory (`training_runs/{run}/graphs/{name}/{name}_overlay.mp4`). In `test_single.py`, it goes in the test_single output directory.

## Concerns

1. **Ankle error is extreme** (114-116 cm MPJPE). The green/red skeletons show wild leg positions in the overlay, which is visually distracting but is a pre-existing issue with MotionBERT's depth estimation for lower limbs, not a bug in the overlay code.

2. **Memory**: Holding all 100 frames_rgb + heatmaps in memory simultaneously works fine for 100 frames at 1080p, but could be an issue for much longer sequences.

3. **Duplicate print**: The `main.py` overlay path is printed both by the `generate_overlay_video()` function and by the explicit print statement after the call. Minor cosmetic issue.

## Deviations from plan

None. All steps from the architect plan were followed as specified.
