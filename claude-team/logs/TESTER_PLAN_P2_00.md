# Tester Plan: Phase 2, Iteration 0 -- Occlusion Handling

## What to Test

### Code Review
1. **scoring.py**: Old `heatmap_score` (analytical Gaussian) removed. `real_heatmap_score` renamed to `heatmap_score`. New confidence-weighted formula: `log(value * conf + confidence_epsilon * (1 - conf))`. No skip logic for low-visibility joints. Fallback joints (Hip, Spine) also use new formula. `compute_total_score` raises ValueError if heatmaps missing.
2. **optimize.py**: Hard visibility thresholding block removed (the block that zeroed out joints below `VISIBILITY_THRESHOLD`).
3. **config.py**: `VISIBILITY_THRESHOLD` removed. `CONFIDENCE_EPSILON` added. `OVERLAY_VISIBILITY_THRESHOLD` added.
4. **graphs.py**: `generate_confidence_score_graph` function exists, plots per-keypoint confidence over frames with 16 lines.
5. **overlay_video.py**: `_draw_skeleton_2d` accepts `visible_mask`. `generate_overlay_video` accepts `visibility` and `visibility_threshold`. GT skeleton always drawn fully. Yellow SH dots filtered by threshold.
6. **create_occlusion_videos.py**: 4 patterns correct, 2 examples selected.
7. **main.py**: Calls `generate_confidence_score_graph`. Passes visibility to overlay video.

### Smoke Tests
8. **Pipeline smoke test**: `cd motionbert-pose && uv run python test_single.py` completes without errors.
9. **Confidence graph generated**: Verify `confidence_scores.png` exists after smoke test.
10. **Occlusion video creation**: `uv run python create_occlusion_videos.py` produces 8 videos + manifest.

### Regression Checks
11. **evaluate.py not broken**: Confirm it doesn't import removed `VISIBILITY_THRESHOLD` from config.
12. **No stale references**: Grep for `real_heatmap_score` and old `VISIBILITY_THRESHOLD` imports across codebase.

## How to Test Each Item
- Items 1-7: Read source files and verify against architect plan.
- Item 8: Run `test_single.py` and check exit code + output.
- Item 9: Check for `confidence_scores.png` in output directory.
- Item 10: Run `create_occlusion_videos.py` and verify manifest.json lists 8 videos.
- Items 11-12: Grep for stale references.
