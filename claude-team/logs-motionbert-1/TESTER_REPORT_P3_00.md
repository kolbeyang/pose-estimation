# TESTER_REPORT_P3_00: Video with Overlays

## Tests Run

### T1: Smoke test -- `test_single.py`
- **Command:** `cd motionbert-pose && uv run python test_single.py`
- **Result:** PASS. Exited 0. Full pipeline ran (SH detection, MotionBERT lifting, optimization, evaluation, overlay video generation).

### T2: Overlay video file exists and is non-trivial
- **Check:** `training_runs/test_single/overlay_171204_pose1_sample_0.mp4`
- **Result:** PASS. File exists, 9.7 MB, 100 frames, 1920x1080 resolution.

### T3: Extract middle frame from overlay video
- **Command:** Python script using cv2 to read frame 50, save to `artifacts/overlay_frame_50.png`
- **Result:** PASS. Frame 50 extracted (1080x1920x3).

### T4: Visual inspection of frame 50
Inspected `artifacts/overlay_frame_50.png`. Findings per overlay layer:

1. **Heatmap overlay (orange/yellow glow):** PASS. Orange/yellow heatmap glow is clearly visible, concentrated around upper body (arms, shoulders, head). The pattern shows distributed heat rather than single-point Gaussians, confirming actual Stacked Hourglass heatmaps are being used. The heatmap is summed across all 16 MPII joint channels and rendered via HOT colormap with additive blending.

2. **Yellow dots (raw SH 2D detections):** PASS. Small bright yellow dots are visible at joint positions on the arms and shoulders. These represent the 16 MPII raw 2D detections from Stacked Hourglass.

3. **Green skeleton (MotionBERT raw 3D projected to 2D):** PASS. Green lines and joint circles are visible. Upper body is reasonably aligned with the person. Legs are wildly misaligned (extending far below and to the side), which is a known pre-existing issue (ankle MPJPE = 114-116 cm, not a bug in the overlay code).

4. **Red skeleton (Optimized 3D projected to 2D):** PASS. Red lines and joint circles are visible, overlapping but distinct from the green skeleton. The optimization has slightly adjusted positions.

5. **Blue skeleton (Ground truth):** PASS. Blue/teal lines visible and aligned much better with the actual person, particularly for torso and arms.

6. **Frame label:** PASS. "Frame 50" displayed at top-left.

7. **Legend:** PASS. Color legend text present at the bottom of the frame, though quite small.

### T5: Code review -- `detect_poses()` return signature
- **Check:** Return tuple has 8 elements: `kp_2d_list, kp_3d_list, visibility_list, all_heatmaps, all_keypoints_2d, affine, positions_3d_norm, cs_params`
- **Callers in main.py (line 165) and test_single.py (line 99) both unpack 8 elements.**
- **Result:** PASS. No tuple unpacking mismatch. The concern flagged in the task instructions about a tuple mismatch does not apply -- both callers correctly unpack all 8 elements.

### T6: Code review -- overlay_video.py correctness
- **Heatmap source:** PASS. Uses `heatmaps[i]` (actual SH heatmaps), not synthetic Gaussians. Lines 223-226 sum across joint channels and resize using affine.
- **Heatmap clamping:** PASS. `np.clip(heatmaps[i], 0, None)` on line 224 handles potential negative values.
- **Heatmap resize:** PASS. `_resize_heatmap_to_frame()` correctly computes crop region size from affine (`256 * sx`, `256 * sy`) and pastes at offset `(tx, ty)` with bounds clipping.
- **3D projection:** PASS. `_project_3d_to_2d()` uses standard pinhole model `u = fx * X / Z + cx`.
- **Color assignments:** PASS. Yellow=(0,255,255) dots, Green=(0,255,0) MotionBERT, Red=(0,0,255) optimized, Blue=(255,100,0) GT. All in BGR as expected for OpenCV.
- **Confidence filter:** PASS. Yellow dots skip joints with confidence < 0.01 (line 232).

### T7: Code review -- consistency with mediapipe-pose reference
- **Result:** PASS. Same HOT colormap additive blend pattern, same skeleton drawing approach, same `mp4v` codec, same 5 fps. Key difference (by design): motionbert overlay uses actual SH heatmaps instead of synthetic Gaussians.

## Bugs Found

None blocking.

## Code Review Findings

### Finding 1: Duplicate overlay save message (Minor)
- **Description:** Both `generate_overlay_video()` (line 277 in overlay_video.py) and the caller in `test_single.py` (line 254) print a "Saved overlay video" message, resulting in duplicate output.
- **Severity:** Minor cosmetic. Does not affect functionality.
- **Suggested fix:** Remove the explicit print in test_single.py since the function already prints.

### Finding 2: Config constants not wired through (Minor)
- **Description:** `OVERLAY_HEATMAP_INTENSITY` and `OVERLAY_FPS` are defined in config.py but not actually used by callers. The overlay function uses hardcoded defaults (200.0, 5.0) that happen to match.
- **Severity:** Minor. Values match, so behavior is correct. But if someone changes config.py expecting it to take effect, it won't.
- **Suggested fix:** Wire config values into the `generate_overlay_video()` calls in main.py and test_single.py.

### Finding 3: Leg skeleton severely misaligned in overlay (Pre-existing, not a bug)
- **Description:** The green and red skeletons show wildly incorrect leg positions (extending off to the lower-left). This is due to MotionBERT's extremely high ankle MPJPE (114-116 cm) and high 2D reprojection error (mean 178 px), not an overlay rendering bug.
- **Severity:** Not a Phase 3 bug. Pre-existing depth estimation issue from Phase 2.

## Verdict

**YES** -- the current state meets the Phase 3 spec's definition of done.

All required overlay layers are present and correctly rendered:
1. Actual Stacked Hourglass heatmaps (not synthetic Gaussians) -- confirmed
2. Raw SH 2D estimations as yellow dots -- confirmed
3. MotionBERT raw prediction projected in green -- confirmed
4. Optimized prediction projected in red -- confirmed

The overlay video generates without errors and visually confirms the pipeline is working. The leg misalignment is a pre-existing 3D estimation issue, not an overlay bug.
