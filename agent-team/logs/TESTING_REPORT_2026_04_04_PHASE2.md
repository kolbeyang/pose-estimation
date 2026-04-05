# Testing Report: Phase 2 -- Camera-Space Evaluation Metrics

**Date**: 2026-04-04
**Tester**: Evaluator Eve
**Branch**: refactor-2026-04-04

## Summary

Phase 2 is **PASSED**. Evaluation metrics now operate in camera coordinates instead of root-relative coordinates. Pelvis error is non-zero, P-MPJPE is unchanged, and no `root_relative()` calls remain in the main evaluation pipeline.

## Code Review

### Files Checked

| File | Status | Notes |
|------|--------|-------|
| `evaluate.py` | PASS | `evaluate()` no longer calls `root_relative()`. Slices to eval joints and computes all 9 metrics in camera space. Comment on line 413 explicitly notes "camera coordinates, not root-relative". |
| `main.py` | PASS | `_evaluate_and_collect_metrics()` passes camera-space arrays directly to `evaluate()`. Per-joint breakdowns also use camera coordinates (comment on line 115). |
| `run_motionbert/__init__.py` | PASS | `evaluate()` called with camera-space `det_arr`, `gt_arr`, `opt_arr`. Per-joint breakdown comment (line 249) confirms camera coordinates. |
| `run_mediapipe/__init__.py` | PASS | Same pattern as motionbert. Comment on line 233 confirms camera coordinates. |

### root_relative() Usage

- **Definition**: Still exists in `evaluate.py` line 28 (correct -- needed by experiment scripts).
- **Callers in main eval pipeline**: None found. Confirmed zero calls from `evaluate()`, `main.py`, `run_motionbert/__init__.py`, or `run_mediapipe/__init__.py`.
- **Callers in experiment scripts**: `test_phase2.py`, `test_si_consistency.py`, `visibility_weight_investigation.py`, `joint_speed_analysis.py` -- all correct backward-compat usage.

## Smoke Test

**Command**: `cd pose-optimizer && uv run python main.py configs/motionbert-single.json`
**Result**: Completed successfully in ~12 seconds.

### Key Metrics (motionbert-single, 34 frames)

| Metric | Value | Matches Dev Report |
|--------|-------|--------------------|
| MPJPE | 59.62 cm | Yes (59.62) |
| P-MPJPE | 42.47 cm | Yes (42.47) |
| SI-MPJPE | 48.22 cm | Yes (48.22) |
| VW-SI-MPJPE | 18.86 cm | Yes (18.86) |
| MPJVE | 4.83 cm/f | Yes (4.83) |

### Pelvis Error Verification

- `det_per_joint[0]` (Pelvis, detector): **33.3 cm** -- non-zero, confirming root-relative cheating is fixed.
- `opt_per_joint[0]` (Pelvis, optimized): **33.2 cm** -- non-zero.

### P-MPJPE Consistency

P-MPJPE = 42.47 cm, matching the dev report's "before" and "after" values exactly. This is expected because Procrustes alignment already handles translation independently, so switching from root-relative to camera-space has zero effect on P-MPJPE.

## Issues Found

### Minor: Stale Docstrings in evaluate.py

Several function docstrings still describe their `predicted`/`target` parameters as "root-relative":
- `optimal_scale()` (lines 68-69)
- `_optimal_scale_weighted()` (lines 92-93)
- `si_mpjpe()` (lines 214-215)
- `vw_si_mpjpe()` (lines 256-257)
- `si_mpjve()` (lines 297-298)
- `vw_si_mpjve()` (lines 343-344)

These functions are coordinate-system agnostic, but the docs are misleading now that they are called with camera-space data. Added to TODO.

## Verdict

**PASSED** -- No blocking issues. One minor docstring cleanup item added to TODO.
