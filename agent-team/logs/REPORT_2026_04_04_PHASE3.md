# Phase 3 Report: Add 2D Reprojected MPJPE

**Date**: 2026-04-04
**Developer**: Dan
**Branch**: refactor-2026-04-04

## Summary

Added `reprojected_mpjpe_2d` as the 10th evaluation metric. This metric projects both predicted and ground truth 3D trajectories to 2D image coordinates and computes the mean pixel distance, filtered to only visible (in-frame) joints.

## Changes

### `pose-optimizer/evaluate.py`
- Added `reprojected_mpjpe_2d()` standalone function
  - Takes predicted (F, J, 3), target (F, J, 3), Camera, and optional visibility (F, J)
  - Projects both to 2D via `camera.camera_to_image()`
  - Filters to visible joints using the provided visibility weights (or computes them from GT frame-boundary check if not provided)
  - Returns mean pixel distance as a scalar float
- Integrated into `evaluate()` so it is computed automatically alongside the other 9 metrics
- Updated module docstring to list all 10 metrics

### `pose-optimizer/main.py`
- Added `"reprojected_mpjpe_2d"` to `_METRIC_KEYS` in `_save_results()` so it appears in `results.json`
- Added logging lines in `_evaluate_and_collect_metrics()` for the new metric (Det/Opt 2D-MPJPE in px)

### `pose-optimizer/run_motionbert/__init__.py`
- Added `"reprojected_mpjpe_2d"` to `_METRIC_KEYS`
- Added logging lines for Det/Opt 2D-MPJPE

### `pose-optimizer/run_mediapipe/__init__.py`
- Added `"reprojected_mpjpe_2d"` to `_METRIC_KEYS`
- Added logging lines for Det/Opt 2D-MPJPE

## Testing

Ran a unit test with synthetic data (5 frames, 16 joints, Camera with fx=fy=500):
- With small perturbations (~0.05m), the metric returned ~11 px, which is in the expected reasonable range
- Visibility filtering works correctly: setting some joints to invisible changes the result
- The metric appears in the `evaluate()` return dict alongside all 9 existing metrics
- All 10 metric keys confirmed present: `mpjpe`, `p_mpjpe`, `si_mpjpe`, `vw_mpjpe`, `vw_si_mpjpe`, `mpjve`, `si_mpjve`, `vw_mpjve`, `vw_si_mpjve`, `reprojected_mpjpe_2d`

## Commits

- `ab3926a` - evaluate.py changes (auto-committed by linter with docstring cleanup)
- `bdb6497` - Pipeline integration (main.py, both __init__.py files)
