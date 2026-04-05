# Phase 1 Testing Report: Unify Mediapipe and Motionbert Pipelines

**Date:** 2026-04-04
**Evaluator:** Evaluator Eve
**Branch:** `refactor-2026-04-04`

## Summary

Phase 1 is **PASS**. All core deliverables are implemented and working correctly. Found only cosmetic issues (stale comments) and one minor spec deviation (missing run-level results.json).

## Test Results

### 1. Smoke Test: MotionBERT Single (`motionbert-single.json`)

**Result: PASS**

- Completed without errors
- VW-SI-MPJPE: Det 17.20 cm, Opt 13.00 cm
- Matches report exactly (0.00 cm difference from old pipeline)

### 2. Smoke Test: MediaPipe Single (`mediapipe-single.json`)

**Result: PASS**

- Completed without errors
- VW-SI-MPJPE: Det 11.06 cm, Opt 10.24 cm
- Matches report exactly. Slight improvement over old pipeline (10.45 -> 10.24 cm) due to real SH heatmaps.

### 3. Smoke Test: Both Pipelines (`both-local-single.json`)

**Result: PASS**

- Both pipelines ran sequentially on the same example
- MotionBERT: VW-SI-MPJPE 17.20 / 13.00 cm (identical to solo run)
- MediaPipe: VW-SI-MPJPE 11.06 / 10.66 cm (slightly worse than solo run due to `num_steps: 10` in this config vs `num_steps: 50` in mediapipe-single.json -- not a bug)
- YOLO + SH heatmaps computed once and shared between both pipelines (verified via log output)

### 4. Config Validation

**Result: PASS**

7 config files present (6 required + 1 extra):
- `mediapipe-single.json` -- valid, MP only
- `motionbert-single.json` -- valid, MB only
- `mediapipe-local-25-examples.json` -- valid, 25 examples, MP only
- `motionbert-local-25-examples.json` -- valid, 25 examples, MB only
- `server-one-each.json` -- valid, both pipelines, server data_root
- `server-full-single.json` -- valid, both pipelines, server data_root
- `both-local-single.json` -- valid, both pipelines, local data_root

All parse correctly as valid JSON. All contain proper `mediapipe` and/or `motionbert` sections. Pydantic validation works correctly.

### 5. Code Review: skeleton.py

**Result: PASS**

- Joint 9 renamed from "Head" to "Nose"
- `EVAL_JOINTS` has 15 entries (indices 0-6, 8-15; Spine(7) excluded)
- `NUM_EVAL_JOINTS = 15`
- `DEFAULT_BONE_LENGTHS[9] = 0.08` (Neck->Nose, shorter than old Neck->Head)
- `REST_DIRECTIONS[9] = [0, -0.5, -0.866]` (forward and slightly up)
- Nose synthesis in `mpii_to_skeleton()`: `skel[9] = skel[8] + 0.3 * (keypoints_mpii[9] - skel[8])` -- correct per spec

### 6. Code Review: config.py

**Result: PASS**

- `PipelineGraphConfig`, `PipelineConfig`, `CrossPipelineGraphConfig` properly defined
- `RunConfig` has optional `mediapipe`, `motionbert`, and `graphs` sections
- None/absent = pipeline skipped (verified in main.py)
- Minor: comment says "# 9: Head" in `rotation_penalty_multipliers` -- should be "# 9: Nose" (cosmetic only)

### 7. Code Review: main.py

**Result: PASS**

- Unified entrypoint works: `uv run main.py <config.json>`
- YOLO + SH runs once per example (shared between pipelines)
- MotionBERT path: SH heatmaps -> MotionBERT 3D -> optimizer (with SH heatmaps)
- MediaPipe path: MP 2D landmarks for solvePnP, SH heatmaps for optimizer -- correct per spec
- Output structure: `run_timestamp/example_name/motionbert/` and `.../mediapipe/`
- Cross-pipeline graph placeholder for Phase 4
- Proper error handling with try/except per example

### 8. Code Review: detect.py

**Result: PASS**

- `Detection2DModels` dataclass for YOLO + SH only
- `load_yolo_sh_models()` loads just 2D models
- `detect_2d_poses()` runs YOLO + SH and returns heatmaps, 2D keypoints, visibility, affine
- `load_all_models()` preserved for backward compatibility
- `detect_poses()` refactored to use `detect_2d_poses()` internally

### 9. Output Directory Structure

**Result: PASS** (with minor deviation)

Verified structure:
```
run_timestamp/
    example_name/
        motionbert/
            results.json, trajectories.json, graphs/, heatmap_overlay_video.mp4, summary.png
        mediapipe/
            results.json, trajectories.json, graphs/, heatmap_overlay_video.mp4, summary.png
    motionbert_aggregate_summary.png
    mediapipe_aggregate_summary.png
```

Minor deviation: Spec shows `results.json` at the run root level, but implementation puts it per-pipeline. Not a functional issue.

### 10. Heatmap Overlay Visual Inspection

**Result: PASS**

Extracted frame 17 (mid-video) from both MotionBERT and MediaPipe overlay videos:
- Human subject is clearly visible and correctly centered
- Three skeleton overlays visible: blue (GT), green (raw detector), red (optimized)
- Skeletons are properly aligned with the person's body
- No obvious coordinate system mismatches
- Heatmaps are blended at low alpha (subtle but present)

### 11. Performance

**Result: PASS**

| Pipeline | Metric | Old | New | Delta |
|----------|--------|-----|-----|-------|
| MotionBERT | VW-SI-MPJPE (det) | 17.20 cm | 17.20 cm | 0.00 cm |
| MotionBERT | VW-SI-MPJPE (opt) | 13.00 cm | 13.00 cm | 0.00 cm |
| MediaPipe | VW-SI-MPJPE (det) | 11.06 cm | 11.06 cm | 0.00 cm |
| MediaPipe | VW-SI-MPJPE (opt) | 10.45 cm | 10.24 cm | -0.21 cm (improved) |

MotionBERT: identical. MediaPipe: improved by 0.21 cm (2.0% improvement). Well within the 1% degradation tolerance.

## Issues Found

All issues are cosmetic or minor. See TODO.md.

1. **Stale comments in graphs.py**: Lines 113, 131, 134 reference "14 eval joints" -- should be 15. No logic bug (code uses `NUM_EVAL_JOINTS` dynamically).

2. **Stale comment in experiment/optimization_benefit.py**: Line 157 says "14 values" -- should be 15. No logic bug.

3. **Stale comment in config.py**: Line 43 comment says "# 9: Head" -- should be "# 9: Nose". No logic bug.

4. **Missing run-level results.json**: Spec shows `results.json` at the run root but implementation only creates per-pipeline `results.json` files. Minor spec deviation.

## Notes to Developer

- Phase 1 is clean and well-implemented. The nose synthesis, shared heatmaps, config restructure, and unified entrypoint all work correctly.
- The 15-eval-joint change propagates cleanly through `EVAL_JOINTS` and `NUM_EVAL_JOINTS` -- no hardcoded 14s in logic code.
- The stale "14 eval joints" comments are low-priority but should be cleaned up to avoid confusion.
- The `both-local-single.json` config uses only 10 optimization steps, which makes MediaPipe results look slightly worse than the 50-step `mediapipe-single.json`. This is by design (quick test config) but worth knowing.
- Old entrypoints (`uv run python -m run_motionbert`, `uv run python -m run_mediapipe`) are preserved and appear functional.
