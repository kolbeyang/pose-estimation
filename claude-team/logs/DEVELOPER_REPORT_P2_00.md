# Developer Report: Phase 2, Iteration 0 -- Occlusion Handling

## What Was Implemented

1. **Confidence-weighted scoring in `scoring.py`**: Removed the old analytical `heatmap_score` function. Renamed `real_heatmap_score` to `heatmap_score`. Replaced the log-likelihood formula with the spec's confidence-weighted version: `log(value * conf + confidence_epsilon * (1 - conf))`. This applies to both heatmap-sampled joints and fallback analytical Gaussian joints (Hip, Spine). Removed the `visibility[j] < 1e-6: continue` skip logic so all joints now contribute.

2. **Removed hard visibility thresholding in `optimize.py`**: Deleted the block that zeroed out visibility below `VISIBILITY_THRESHOLD` (0.5). Confidence is now handled inside the scoring formula.

3. **Updated `config.py`**: Removed `VISIBILITY_THRESHOLD`. Added `CONFIDENCE_EPSILON = 1e-4` and `OVERLAY_VISIBILITY_THRESHOLD = 0.3`.

4. **Added confidence score graph in `graphs.py`**: New `generate_confidence_score_graph()` function plots per-keypoint confidence scores across frames with distinct colors per joint.

5. **Wired up in `main.py`**: Added import and call to `generate_confidence_score_graph`. Passed `visibility` and `visibility_threshold` to `generate_overlay_video`.

6. **Visibility-based filtering in `overlay_video.py`**: Updated `_draw_skeleton_2d` to accept an optional `visible_mask` parameter. Joints and bones below the visibility threshold are not drawn for detector and optimized skeletons. Ground truth skeleton always drawn fully. Yellow SH dots now filtered using the same `visibility_threshold`.

7. **Updated `test_single.py`**: Passed visibility to overlay video. Fixed pre-existing off-by-one bug in bone length diagnostic loop (`range(1, 17)` -> `range(1, 16)` since NUM_JOINTS=16).

8. **Created `create_occlusion_videos.py`**: Generates 8 augmented test videos (2 examples x 4 patterns) with frames replaced by black images. Saves manifest JSON.

## Files Changed

- `motionbert-pose/scoring.py` -- Removed old `heatmap_score`, renamed `real_heatmap_score`, updated scoring formula, updated `compute_total_score`
- `motionbert-pose/optimize.py` -- Removed hard visibility thresholding, pass `confidence_epsilon` to `compute_total_score`
- `motionbert-pose/config.py` -- Removed `VISIBILITY_THRESHOLD`, added `CONFIDENCE_EPSILON` and `OVERLAY_VISIBILITY_THRESHOLD`
- `motionbert-pose/graphs.py` -- Added `generate_confidence_score_graph`
- `motionbert-pose/main.py` -- Added confidence graph call, passed visibility to overlay video
- `motionbert-pose/overlay_video.py` -- Added `visible_mask` to `_draw_skeleton_2d`, added `visibility`/`visibility_threshold` to `generate_overlay_video`
- `motionbert-pose/test_single.py` -- Passed visibility to overlay video, fixed pre-existing bone length loop bug
- `motionbert-pose/create_occlusion_videos.py` -- New file, creates augmented occlusion test videos

## Commands Run

1. `uv run python -c "import config as cfg; ..."` -- Verified config imports cleanly, new constants present, old `VISIBILITY_THRESHOLD` removed.
2. `uv run python -c "..."` (confidence analysis) -- Ran detection on EXAMPLES[0] and printed per-joint confidence statistics to determine overlay threshold.
3. `uv run python test_single.py` -- Full pipeline (detect + optimize + evaluate + overlay video). Completed successfully.
4. `uv run python create_occlusion_videos.py` -- Created 8 occlusion test videos. All generated successfully.

## Decisions Made

1. **Overlay visibility threshold = 0.3**: Analyzed confidence scores from EXAMPLES[0]. RAnkle and LAnkle had max confidence of ~0.04 (clearly off-screen). All other visible joints had median confidence > 0.36. Threshold of 0.3 cleanly separates visible from occluded joints in this example.

2. **`compute_total_score` now raises ValueError if heatmaps are not provided**: The old fallback to the analytical `heatmap_score` was dead code in the MotionBERT pipeline. Since the old function was deleted, I added a guard that raises an error if heatmaps/affine are missing, per the architect's plan.

3. **`evaluate.py` untouched**: The `reprojection_error_vs_detections` function has its own `visibility_threshold` parameter with a default of 0.5. It does NOT import `cfg.VISIBILITY_THRESHOLD`, so removing it from config causes no breakage.

4. **Occlusion video examples**: Used indices 0 (171204_pose1_sample) and 4 (171204_pose3) as specified in the architect plan.

## Concerns

1. **Confidence values can exceed 1.0**: The architect noted this. SH heatmap peak values are raw network outputs and can be > 1.0 (max observed: 0.91 in this example, but could be higher in others). With `conf > 1.0`, the formula `value * conf + eps * (1 - conf)` has `(1 - conf) < 0`, which could make the inner expression slightly smaller than `value * conf` alone. This is mathematically fine but worth monitoring.

2. **Large errors on ankles**: RAnkle MPJPE is 195 cm, LAnkle 206 cm. These joints have near-zero confidence (0.01 median), so the new scoring formula correctly gives them no gradient. However, the optimization still has to place them somewhere (via FK), and the motion/anchor penalties can still pull them to bad positions. This is expected behavior -- the system correctly identifies these as unreliable but can't magically fix them without visibility data.

3. **MPJPE may shift slightly**: The new scoring formula changes the loss landscape. The previous approach hard-zeroed low-confidence joints and used `log(value) * visibility` for the rest. The new formula uses `log(value * conf + eps * (1-conf))` without the external visibility multiplier. This could shift optimization behavior slightly. The test run showed reasonable results (5.80 cm improvement).

## Deviations from Plan

None. All steps followed as specified.
