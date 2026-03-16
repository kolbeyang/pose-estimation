# TESTER_PLAN_P3_00: Video with Overlays

## What to Test

### T1: Smoke test -- `test_single.py` runs without errors
- **Command:** `cd motionbert-pose && uv run python test_single.py`
- **Expected:** Exits 0, produces overlay video and diagnostics JSON

### T2: Overlay video file exists and is non-trivial
- **Check:** `training_runs/test_single/overlay_171204_pose1_sample_0.mp4` exists, >100KB

### T3: Extract middle frame from overlay video
- **Command:** Use ffmpeg or cv2 to extract frame 50 from the overlay video, save to `artifacts/overlay_frame_50.png`
- **Expected:** PNG image saved

### T4: Visual inspection of frame 50
- **Check all five overlay layers:**
  1. Orange/yellow heatmap glow from actual SH (16, 64, 64) heatmaps (NOT synthetic Gaussian)
  2. Small yellow dots at raw SH 2D detection positions (16 MPII joints)
  3. Green skeleton from MotionBERT 3D projected to 2D
  4. Red skeleton from optimized 3D projected to 2D
  5. (Optional) Blue skeleton from ground truth if available
  6. Frame label and legend text

### T5: Code review -- `detect_poses()` return signature
- **Check:** All callers (main.py, test_single.py) correctly unpack 8-element tuple
- **Expected:** No tuple unpacking mismatch

### T6: Code review -- overlay_video.py correctness
- **Check:**
  - Heatmaps are actual SH heatmaps (not synthetic Gaussians)
  - Heatmap resize uses affine correctly (64x64 -> 256x256 scale -> crop region)
  - 3D-to-2D projection uses camera intrinsics (perspective divide)
  - Color assignments match spec (yellow dots, green=MB, red=optimized)
  - Negative heatmap values are clamped

### T7: Code review -- consistency with mediapipe-pose reference
- **Check:** Same general patterns (HOT colormap, additive blend, skeleton drawing, video codec)

## Regression Checks

### R1: main.py entry point unchanged in behavior
- Check `detect_poses()` call is correctly updated in main.py

### R2: test_single.py updated consistently
- Check `detect_poses()` call is correctly updated in test_single.py
