# Phase 1 Report: Unify Mediapipe and Motionbert Pipelines

**Date:** 2026-04-04
**Branch:** `refactor-2026-04-04`

## Summary

Phase 1 is complete. Both MotionBERT and MediaPipe pipelines now run through a unified `main.py` entrypoint with shared YOLO + StackedHourglass 2D heatmap generation.

## Changes Made

### 1. Skeleton Update (`skeleton.py`)
- Renamed `Head(9)` to `Nose(9)` across JOINT_NAMES, docstrings, and comments
- Updated `EVAL_JOINTS` from 14 joints to 15 joints (added index 9, only Spine(7) excluded)
- Updated `DEFAULT_BONE_LENGTHS[9]` from 0.12 to 0.08 (Neck->Nose shorter than Neck->HeadTop)
- Updated `REST_DIRECTIONS[9]` from `[0, -1, 0]` to `[0, -0.5, -0.866]` (nose forward+up)
- Updated `mpii_to_skeleton()`: Nose is now synthesized as `skel[8] + 0.3 * (head_top - skel[8])` instead of direct Head Top mapping

### 2. Config System (`config.py`)
- Added `PipelineGraphConfig`, `PipelineConfig`, `CrossPipelineGraphConfig` models
- Added `mediapipe`, `motionbert`, and `graphs` optional sections to `RunConfig`
- If `mediapipe` is `None`/absent, MediaPipe pipeline is skipped; same for `motionbert`

### 3. YOLO + SH Extraction (`run_motionbert/detect.py`)
- Added `Detection2DModels` dataclass (YOLO + SH only, no MotionBERT model)
- Added `load_yolo_sh_models()` function for loading just 2D detection models
- Added `detect_2d_poses()` function that runs YOLO + SH pipeline only
- Refactored `detect_poses()` to internally use `detect_2d_poses()` + MotionBERT 3D lifting
- `load_all_models()` preserved for backward compatibility

### 4. Unified Main Entrypoint (`main.py`)
- New unified entrypoint: `uv run main.py <config.json>`
- Runs YOLO + SH once per example (shared between both pipelines)
- MotionBERT uses SH heatmaps -> MotionBERT 3D -> optimizer (with SH heatmaps)
- MediaPipe uses MP 2D landmarks for solvePnP camera estimation, but SH heatmaps for optimizer
- Output structure: `run_timestamp/example_name/motionbert/` and `.../mediapipe/`

### 5. Config Files (`configs/`)
Created 7 config files:
- `configs/mediapipe-single.json` - single example, MP only
- `configs/motionbert-single.json` - single example, MB only
- `configs/mediapipe-local-25-examples.json` - 25 examples, MP only
- `configs/motionbert-local-25-examples.json` - 25 examples, MB only
- `configs/server-one-each.json` - single example, both pipelines + cross-pipeline graphs, server data_root
- `configs/server-full-single.json` - single example, both pipelines, server data_root, full optimization
- `configs/both-local-single.json` - single example, both pipelines, local data_root

### 6. Graphs (`graphs.py`)
- Added optional `prefix` parameter to `generate_aggregate_summary()` so both pipelines can produce separate aggregate summaries

## Backward Compatibility
- Old entrypoints (`uv run python -m run_motionbert`, `uv run python -m run_mediapipe`) still work with their existing configs
- Old flat config format (no `mediapipe`/`motionbert` sections) still loads correctly via Pydantic defaults

## Critical Decisions

1. **MediaPipe optimizer now uses SH heatmaps** instead of synthetic Gaussian heatmaps. This is the biggest behavioral change. The solvePnP step still uses MediaPipe's own 2D landmarks (as specified).

2. **Nose synthesis for MotionBERT**: `nose = neck + 0.3 * (head_top - neck)`. This places the nose 30% of the way from the base of neck toward the head top, which is anatomically reasonable.

3. **Shared heatmap generation**: YOLO + SH runs once per example and the resulting heatmaps/affine are shared between both pipelines. This avoids redundant computation.

## Test Results

### MotionBERT Pipeline (VW-SI-MPJPE on pose1_sample)
| Pipeline | Det | Opt |
|----------|-----|-----|
| Old (`run_motionbert`) | 17.20 cm | 13.00 cm |
| New (`main.py`) | 17.20 cm | 13.00 cm |
| **Difference** | 0.00 cm | 0.00 cm |

### MediaPipe Pipeline (VW-SI-MPJPE on pose1_sample)
| Pipeline | Det | Opt |
|----------|-----|-----|
| Old (`run_mediapipe`, synthetic heatmaps) | 11.06 cm | 10.45 cm |
| New (`main.py`, SH heatmaps) | 11.06 cm | 10.24 cm |
| **Difference** | 0.00 cm | -0.21 cm (improvement) |

The MediaPipe pipeline actually improved slightly with real SH heatmaps. No degradation.

### Both Pipelines Together
Successfully ran both pipelines on the same example. Output directory structure verified:
```
run_timestamp/
  example_name/
    motionbert/results.json, trajectories.json, graphs/, heatmap_overlay_video.mp4
    mediapipe/results.json, trajectories.json, graphs/, heatmap_overlay_video.mp4
  motionbert_aggregate_summary.png
  mediapipe_aggregate_summary.png
```

### Heatmap Overlay Videos
Visual inspection of mid-video frames from both pipelines confirms:
- SH heatmaps align correctly with the human subject
- All three skeletons (GT=blue, raw=green, optimized=red) are positioned correctly
- No coordinate system mismatches

### Old Entrypoints
Both `uv run python -m run_motionbert` and `uv run python -m run_mediapipe` produce identical results to before.

## Issues / Red Flags
- None. All tests pass with expected or better results.
- The 15-eval-joint change (adding Nose) means per-joint arrays are now 15-long instead of 14-long. This propagates automatically through `EVAL_JOINTS` but any code that hard-codes 14 joints would break. No such code was found.
