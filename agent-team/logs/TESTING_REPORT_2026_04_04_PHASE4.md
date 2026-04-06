# Phase 4 Testing Report

**Date**: 2026-04-04
**Tester**: Evaluator Eve
**Branch**: `refactor-2026-04-04`
**Commit tested**: `f3db7e3`

## Checklist

### Config-controlled cross-pipeline graphs

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Only run when BOTH `motionbert` and `mediapipe` present | PASS | `main.py` line 621: `if config.graphs and run_motionbert and run_mediapipe` |
| 2 | `logger.warning` if `graphs` set but condition not met | PASS | `main.py` line 637: `logger.warning("Cross-pipeline graphs require both...")` |
| 3a | `generate_cross_pipeline_per_joint_position` exists with correct colors | PASS | MB Raw=#FFCC80, MB Opt=#FF9800, MP Raw=#EF9A9A, MP Opt=#F44336 |
| 3b | `generate_cross_pipeline_per_joint_velocity` exists with correct colors | PASS | Same color constants used |
| 3c | `generate_cross_pipeline_metrics_comparison` exists with correct colors | PASS | Same color constants used |
| 3d | GT color Blue (#4285F4) defined | PASS | Constant defined at line 355 of `graphs.py`, though not used in cross-pipeline graphs (spec only requires 4 bars, no GT bar) |
| 4a | Position graph Y-axis: VW-SI-MPJPE | PASS | Label reads "VW-SI-MPJPE (cm)" |
| 4b | Velocity graph Y-axis: VW-SI-MPJVE | PASS | Label reads "VW-SI-MPJVE (cm/frame)" |
| 4c | Metrics comparison X-axis groups | PASS | "VW-SI-MPJPE (cm)" and "VW-SI-MPJVE (cm/f)" |

### Experiment scripts

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 5a | `experiment/elbow_smoothness.py` exists | PASS | |
| 5b | Takes trajectories.json as input | PASS | Two positional args: mb_trajectories, mp_trajectories |
| 5c | 15 lines: 5 colors x 3 line styles | PASS | Verified visually: solid/wide-dash/narrow-dash for X/Y/Z |
| 5d | Y: cm, X: seconds | PASS | Y="Position (cm)", X="Time (seconds)" |
| 6a | `experiment/bone_length_variation.py` exists | PASS | |
| 6b | Takes trajectories.json as input | PASS | Two positional args: mb_trajectories, mp_trajectories |
| 6c | 5 lines | PASS | GT, MB Raw, MB Opt, MP Raw, MP Opt |
| 6d | Y: cm, X: seconds | PASS | Y="Bone Length (cm)", X="Time (seconds)" |
| 6e | Optimized lines are horizontal | PASS | std=0.0000 for both MB Opt and MP Opt |

### Smoke test

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 7 | `both-local-single.json` runs end-to-end | PASS | Completed in ~17 seconds |
| 8 | Cross-pipeline graph PNGs generated | PASS | 3 files in `cross_pipeline_graphs/` |
| 9a | Elbow smoothness script runs | PASS | Output: `elbow_smoothness_RElbow.png` |
| 9b | Bone length variation script runs | PASS | Output: `bone_length_variation_R_forearm.png` |

## Visual Inspection

All three cross-pipeline graphs were visually inspected:
- Colors are correct and distinguishable (light orange / orange / light red / red)
- Bar groupings are correct (4 bars per joint or metric group)
- Axes labels match the spec

Both experiment graphs were visually inspected:
- Elbow smoothness shows 15 lines with correct color/style differentiation
- Bone length variation shows 5 lines; optimized lines are perfectly flat

## Issues Found

### Minor (non-blocking)
1. **Metrics comparison Y-axis is generic**: The `generate_cross_pipeline_metrics_comparison` graph uses "Error" as the Y-axis label. Since the two groups have different units (cm vs cm/frame), a single label is inherently imprecise. Suggestion: annotate bar groups or use dual Y-axes.
2. **GT Blue bar not in cross-pipeline graphs**: The spec defines GT=Blue in the color scheme but only specifies 4 bars per group (no GT). The implementation correctly follows the spec with 4 bars. The GT color constant is defined but unused in cross-pipeline functions.

### No blocking issues found.

## Metrics from smoke test

| Pipeline | Det VW-SI-MPJPE | Opt VW-SI-MPJPE |
|----------|-----------------|-----------------|
| MotionBERT | 23.43 cm | 18.86 cm |
| MediaPipe | 15.66 cm | 15.86 cm |

## Verdict

**PASS** - Phase 4 is complete and correct. All deliverables are implemented per spec.
