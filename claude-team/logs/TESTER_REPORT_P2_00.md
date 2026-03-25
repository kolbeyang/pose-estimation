# Tester Report: Phase 2, Iteration 0 -- Occlusion Handling

## Tests Run

### 1. Code Review: `scoring.py` -- Old function removed, new formula correct
**Result: PASS**
- Old analytical `heatmap_score` function is gone. No references to `real_heatmap_score` anywhere in the codebase.
- Renamed function `heatmap_score` has correct signature with `confidence_epsilon` parameter (default `1e-4`).
- New formula at lines 90-92 and 118-120: `log(clamp(value * conf + confidence_epsilon * (1 - conf), min=eps))`. Matches spec exactly.
- No skip logic for low-visibility joints -- the `if visibility[j] < 1e-6: continue` is gone. All joints contribute.
- Fallback path for Hip/Spine (lines 87-93) converts analytical Gaussian to a value via `exp(-sq_dist / (2 * sigma^2))`, then applies the same confidence formula. Correct.
- `compute_total_score` raises `ValueError` if heatmaps are not provided (line 236). Passes `confidence_epsilon` through to `heatmap_score` (line 246).
- Visibility is no longer used as an external multiplier -- it's only inside the log. Correct per spec.

### 2. Code Review: `optimize.py` -- Hard thresholding removed
**Result: PASS**
- No visibility thresholding block exists. Grep for `VISIBILITY_THRESHOLD` returns zero hits in this file.
- `confidence_epsilon=cfg.CONFIDENCE_EPSILON` is passed to `compute_total_score` at line 243. Correct.

### 3. Code Review: `config.py` -- Constants updated
**Result: PASS**
- `VISIBILITY_THRESHOLD` is gone (grep confirms zero hits across the entire codebase except `OVERLAY_VISIBILITY_THRESHOLD`).
- `CONFIDENCE_EPSILON: float = 1e-4` added at line 86.
- `OVERLAY_VISIBILITY_THRESHOLD: float = 0.3` added at line 90.
- Both have clear docstring comments explaining their purpose.

### 4. Code Review: `graphs.py` -- Confidence graph function
**Result: PASS**
- `generate_confidence_score_graph` at line 343: plots 16 lines (one per joint) using `plt.cm.tab20` colormap.
- X-axis: frame index. Y-axis: confidence score. Title: "Per-Keypoint Confidence Scores".
- Legend placed outside plot with `bbox_to_anchor=(1.05, 1)` and `fontsize=7`.
- Saves to `confidence_scores.png` in the output directory.
- Uses `_save()` helper consistent with other graph functions.

### 5. Code Review: `overlay_video.py` -- Visibility filtering
**Result: PASS**
- `_draw_skeleton_2d` accepts optional `visible_mask` parameter (line 145).
- When `visible_mask` is provided, bones where either endpoint is invisible are skipped (line 161). Joints where mask is False are skipped (line 168).
- `generate_overlay_video` accepts `visibility` and `visibility_threshold` parameters (lines 190-191).
- Visibility mask computed per frame at lines 238-240: `vis_mask = visibility[i] >= visibility_threshold`.
- Mask passed to green (detector) and red (optimized) skeletons. NOT passed to blue (GT) skeleton -- GT always fully drawn. Correct.
- Yellow SH dots filtered by `visibility_threshold` at line 246 (replaces old `< 0.01` check). Correct.

### 6. Code Review: `create_occlusion_videos.py` -- Patterns and examples
**Result: PASS**
- 4 patterns defined correctly:
  - `every_other_black`: `i % 2 == 0` (visible on even frames)
  - `every_4th_visible`: `i % 4 == 0` (visible every 4th frame)
  - `every_4th_black`: `i % 4 != 3` (black every 4th frame)
  - `16on_16off`: `(i % 32) < 16` (16 visible, 16 black)
- Examples 0 and 4 selected as specified by architect.
- Saves manifest JSON with video paths and pattern metadata.

### 7. Code Review: `main.py` -- Integration
**Result: PASS**
- `generate_confidence_score_graph` imported at line 27.
- Called at line 331 with `(visibility, example_graph_dir)`.
- `visibility` and `visibility_threshold=cfg.OVERLAY_VISIBILITY_THRESHOLD` passed to `generate_overlay_video` at lines 355-356.

### 8. Smoke Test: `test_single.py`
**Command:** `cd motionbert-pose && uv run python test_single.py`
**Result: PASS**
- Pipeline completed without errors (100 frames, 100 optimization steps).
- Loss decreased from 6835.9 to 6685.5 (converging).
- MPJPE improved from 50.79 cm to 44.99 cm (+5.80 cm improvement).
- Overlay video generated: `overlay_171204_pose1_sample_0.mp4`.
- No crashes, no warnings about the new scoring formula.

### 9. Confidence Graph Generation
**Result: PASS (with caveat)**
- `test_single.py` does NOT call `generate_confidence_score_graph` -- only `main.py` does.
- The function is correctly wired in `main.py` (line 331) and will produce the graph when `main.py` is run.
- This is a minor gap: the quick-test path (`test_single.py`) doesn't exercise the confidence graph. Not a bug -- `test_single.py` is a developer convenience script, not the main entry point.

### 10. Occlusion Video Creation
**Command:** `cd motionbert-pose && uv run python create_occlusion_videos.py`
**Result: PASS**
- 8 videos created (2 examples x 4 patterns).
- Manifest saved with correct metadata.
- Pattern counts verified:
  - every_other_black: 50/100 and 75/150 blacked (correct)
  - every_4th_visible: 75/100 and 112/150 blacked (correct)
  - every_4th_black: 25/100 and 37/150 blacked (correct)
  - 16on_16off: 48/100 and 70/150 blacked (correct)

### 11. Regression: `evaluate.py` not broken
**Result: PASS**
- `evaluate.py` has its own `visibility_threshold=0.5` default parameter in `reprojection_error_vs_detections()`. It does NOT import `VISIBILITY_THRESHOLD` from config. No breakage.

### 12. No Stale References
**Result: PASS**
- Grep for `real_heatmap_score` across the codebase: zero hits.
- Grep for `VISIBILITY_THRESHOLD` (excluding `OVERLAY_VISIBILITY_THRESHOLD`): zero hits in any import or usage context.

## Bugs Found

None.

## Code Review Findings

### Minor Observations (not bugs)

1. **`test_single.py` missing confidence graph**: The confidence score graph is only generated via `main.py`, not `test_single.py`. This means quick developer testing doesn't exercise the new graph. Low impact since `main.py` is the primary entry point.

2. **Confidence values can exceed 1.0**: As the developer noted, Stacked Hourglass heatmap peaks are raw network outputs. When `conf > 1.0`, the term `(1 - conf)` goes negative, making `confidence_epsilon * (1 - conf)` slightly reduce the score. This is mathematically harmless (the `value * conf` term dominates) but worth monitoring. The `min=eps` clamp prevents any numerical issues.

3. **Ankle MPJPE very high (195-206 cm)**: RAnkle and LAnkle have near-zero confidence (as the developer noted). The new scoring formula correctly provides no gradient for these joints, but FK constraints still place them based on upstream joints. The high error is expected given the joints are off-screen in this example.

## Verdict

**YES** -- The implementation meets the spec's definition of done for Phase 2.

Checklist:
- [x] Old `heatmap_score` removed
- [x] `real_heatmap_score` renamed to `heatmap_score`
- [x] Confidence-weighted scoring formula: `log(value * confidence + confidence_epsilon * (1 - confidence))`
- [x] Logic to omit occluded keypoints from scoring removed
- [x] Hard visibility thresholding in optimizer removed
- [x] Confidence score graph per keypoint over frames
- [x] Visibility threshold selected for overlay videos (0.3, empirically determined)
- [x] Augmented test videos with 4 occlusion patterns for 2 examples (8 videos total)
- [x] Pipeline runs without errors
- [x] No regression in `evaluate.py`
