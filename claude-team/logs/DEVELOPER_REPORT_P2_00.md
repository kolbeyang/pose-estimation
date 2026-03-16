# Developer Report: Phase 2, Iteration 0

## Goal
Layer differentiable FK optimization on top of MotionBERT's Phase 1 detector predictions to reduce MPJPE.

## What Was Implemented

All files created/modified following the architect's plan step by step.

### Files Created
| File | Purpose |
|------|---------|
| `motionbert-pose/fk.py` | Forward and inverse kinematics with axis-angle rotations, 17 joints |
| `motionbert-pose/scoring.py` | Heatmap scoring (analytical Gaussian) + temporal motion penalties |
| `motionbert-pose/optimize.py` | FK optimization loop with Adam optimizer |

### Files Modified
| File | Changes |
|------|---------|
| `motionbert-pose/skeleton.py` | Added `REST_DIRECTIONS` (17 joints, Y-down camera convention) |
| `motionbert-pose/config.py` | Added optimization hyperparameters (NUM_STEPS=10, SIGMA=50, LR=0.001, etc.) |
| `motionbert-pose/models.py` | Added `OptimizationConfig`, `ComparisonResult` Pydantic models; extended `ExampleResult` with opt_* fields |
| `motionbert-pose/evaluate.py` | Added `compute_comparison_with_optimization()` for detector vs optimized vs GT |
| `motionbert-pose/graphs.py` | Updated all 3 graph functions for side-by-side detector vs optimized display |
| `motionbert-pose/main.py` | Integrated optimization into pipeline (steps 5-7), saves both det and opt metrics |
| `motionbert-pose/test_single.py` | Added optimization step and comparison output |

## Commands Run

1. `uv run python test_single.py` -- verified single example (FK roundtrip, optimization, evaluation)
2. `uv run python main.py` -- processed all 10 examples with optimization

## Results

### FK Roundtrip Verification
FK roundtrip error: **0.0000 cm** on all examples. The inverse FK -> forward FK roundtrip is perfect.

### Optimization Results (10 steps, all 10 examples)

| Example | Det MPJPE (cm) | Opt MPJPE (cm) | Improvement (cm) |
|---------|---------------|----------------|-------------------|
| 171204_pose1_sample_0 | 58.04 | 21.15 | +36.89 |
| 171204_pose2_200 | 62.90 | 54.48 | +8.42 |
| 171204_pose2_5000 | 18.08 | 12.59 | +5.49 |
| 171204_pose2_15000 | 50.97 | 35.40 | +15.57 |
| 171204_pose3_200 | 54.38 | 49.02 | +5.36 |
| 171204_pose3_4000 | 39.57 | 24.36 | +15.21 |
| 160422_ultimatum1_200 | 59.62 | 53.41 | +6.21 |
| 160422_ultimatum1_10000 | 65.10 | 51.92 | +13.17 |
| 171204_pose2_10000 | 20.36 | 15.44 | +4.92 |
| 171204_pose2_25000 | 19.92 | 15.02 | +4.90 |
| **MEAN** | **44.89** | **33.28** | **+11.61** |

### P-MPJPE (Procrustes-aligned)
Mean Det P-MPJPE: 21.29 cm -> Mean Opt P-MPJPE: 21.62 cm (slight regression of 0.33 cm, expected since FK optimization primarily improves absolute positioning/depth, not relative shape).

## Validation Checklist

- [x] FK roundtrip error < 0.001 cm on all 10 examples (actual: 0.0000 cm)
- [x] Optimization loss decreases monotonically over 10 steps (confirmed on all examples)
- [x] Optimized MPJPE improves on majority of examples (10/10 improved)
- [x] No example regresses by more than 2 cm (all improved)
- [x] Mean MPJPE is lower after optimization (44.89 -> 33.28 cm, -11.61 cm)
- [x] Results JSON includes both `det_*` and `opt_*` metrics
- [x] Graphs show both detector and optimized results
- [x] All new functions have type hints
- [x] Pydantic models validate optimization config and results
- [x] `uv run python main.py` completes without errors on all 10 examples

## Key Observations

1. **All 10/10 examples improved** with only 10 optimization steps. The improvement ranges from +4.90 cm to +36.89 cm.

2. **Example 0 had the largest improvement** (+36.89 cm). This example had severely noisy depth estimates from the detector (Z range 2.26-8.35m vs GT 2.07-2.73m). The FK optimization's shared bone lengths constraint dramatically corrected this.

3. **P-MPJPE is roughly stable** (~21 cm). This confirms that the optimizer is primarily fixing depth/scale issues (which MPJPE captures) rather than changing the pose shape (which P-MPJPE measures).

4. **Loss decreases monotonically** on all examples, confirming the optimization is well-behaved with these hyperparameters.

## Deviations from Plan

None. All files were created/modified in the specified order. NUM_STEPS=10 as specified for initial testing.

## Next Steps

Per the architect's plan, after confirming 10-step correctness:
1. Increase NUM_STEPS to 50, then 100, then 300
2. Verify MPJPE continues to improve
3. Adjust sigma, penalty weights if needed
