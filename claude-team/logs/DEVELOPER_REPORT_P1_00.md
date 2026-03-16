# Developer Report P1-00: Parameter Sweep Implementation

## What Was Implemented

1. **New 2D-vs-detection reprojection metric** in `evaluate.py`: Added `reprojection_error_vs_detections()` function that projects 3D predictions to 2D via the camera model and measures pixel distance to the 2D Stacked Hourglass keypoints (not GT projections). Integrated into `compute_comparison_with_optimization()` via optional `detections_2d` and `visibility` parameters.

2. **Wired new metric into `main.py` and `test_single.py`**: Both now pass `detections_2d=improved_target_2d` and `visibility=visibility` to the evaluation function and print the new `Det 2D-vs-Det` and `Opt 2D-vs-Det` metrics.

3. **Created `sweep.py`**: Parameter sweep script with 20 configurations that:
   - Loads detection data once (expensive) and caches it
   - Re-runs optimization with different parameter configs
   - Reports MPJPE, P-MPJPE, 2D-vs-detection reprojection error in a table
   - Saves results to JSON

## Files Changed

- **Modified:** `motionbert-pose/evaluate.py` -- Added `reprojection_error_vs_detections()`, extended `compute_comparison_with_optimization()` signature
- **Modified:** `motionbert-pose/main.py` -- Pass new params to evaluation, print new metrics
- **Modified:** `motionbert-pose/test_single.py` -- Pass new params to evaluation, print new metrics
- **Created:** `motionbert-pose/sweep.py` -- Parameter sweep script

## Commands Run

```bash
cd motionbert-pose && uv run python -c "import sweep; print('Import OK')"
# Result: Import OK, 20 configs

cd motionbert-pose && uv run python sweep.py
# Result: All 20 configs completed successfully (~2.5 minutes total)
```

## Sweep Results (Example 0: 171204_pose1_sample_0)

| Config | Det MPJPE (cm) | Opt MPJPE (cm) | Improv (cm) | Opt P-MPJPE (cm) | Det 2D-Det (px) | Opt 2D-Det (px) |
|--------|---------------|----------------|-------------|-------------------|-----------------|-----------------|
| baseline | 30.98 | 30.44 | +0.54 | 28.42 | 38.1 | 33.8 |
| pos_w=0 | 30.98 | 30.45 | +0.53 | 28.42 | 38.1 | 33.7 |
| pos_w=5000 | 30.98 | 30.39 | +0.59 | 28.43 | 38.1 | 34.1 |
| rot_s=0 | 30.98 | 30.45 | +0.53 | 28.43 | 38.1 | 33.8 |
| rot_s=1000 | 30.98 | 30.39 | +0.59 | 28.47 | 38.1 | 34.3 |
| anchor=0 | 30.98 | 30.43 | +0.55 | 28.42 | 38.1 | 33.8 |
| anchor=500 | 30.98 | 30.51 | +0.47 | 28.46 | 38.1 | 34.9 |
| heatmap_only | 30.98 | 30.44 | +0.54 | 28.43 | 38.1 | 33.7 |
| all_low | 30.98 | 30.44 | +0.54 | 28.42 | 38.1 | 33.7 |
| baseline_100steps | 30.98 | 31.10 | -0.12 | 28.30 | 38.1 | 28.2 |
| all_low_100steps | 30.98 | 31.42 | -0.44 | 28.29 | 38.1 | 27.2 |

## Key Findings

1. **Penalty weights barely matter at 20 steps**: All 20-step configs produce nearly identical results (30.39--30.55 cm opt MPJPE). The parameter space is extremely insensitive at this step count -- even `heatmap_only` (all penalties = 0) matches baseline.

2. **More steps improve 2D alignment but hurt 3D accuracy**: At 100 steps, 2D-vs-Det error drops significantly (28.2 px vs 33.8 px) but 3D MPJPE gets worse (31.10 cm vs 30.44 cm). This suggests the optimizer is overfitting to noisy 2D detections, sacrificing depth accuracy.

3. **P-MPJPE improves with more steps**: Both 100-step configs show better P-MPJPE (28.29-28.30 cm vs 28.42 cm), indicating the shape is better aligned even as absolute positioning gets worse.

4. **The 2D-vs-Det baseline is 38.1 px**: MotionBERT's 3D predictions, when reprojected, are ~38 pixels off from the SH 2D detections on average. Optimization reduces this to ~28 px at 100 steps.

## Decisions Made

- The architect's plan specified ~22 configs but listed only 20 unique ones (the baseline appears as both a standalone config and as `pos_w=50 (baseline)`). I included 20 distinct configs to avoid redundancy.

## Concerns

1. The sweep results show that at 20 steps, penalty weights have negligible effect. This could mean: (a) 20 steps is too few for any meaningful optimization, or (b) the heatmap signal itself is the dominant factor. The 100-step results showing worse MPJPE suggest the fundamental issue may be that heatmap-based optimization pushes the skeleton toward noisy 2D targets at the cost of depth accuracy.

2. The Phase 1.2 heatmap blur approach may help the gradient basin issue, but the tradeoff between 2D alignment and 3D accuracy will likely persist.

## Deviations from Plan

None. All steps implemented as specified.
