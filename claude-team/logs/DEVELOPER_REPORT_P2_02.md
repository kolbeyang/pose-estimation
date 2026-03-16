# Developer Report: Phase 2, Iteration 2 -- FK Optimization Tuning

## Goal

Increase optimization steps, add coarse-to-fine sigma schedule, visibility thresholding, and tune hyperparameters. Target: ~24-26cm optimized MPJPE.

## What Was Implemented

All five changes from the architect plan were implemented:

1. **Sigma schedule infrastructure** (`optimize.py`): Added `_get_sigma()` helper that reads `SIGMA_SCHEDULE` from config to vary sigma across optimization steps.
2. **Visibility thresholding** (`optimize.py`): Joints with confidence < `VISIBILITY_THRESHOLD` (0.5) are zeroed out before optimization. This prevents low-confidence detections (e.g., ankles at 0.003-0.006) from corrupting the optimizer.
3. **Separate root_pos learning rate** (`optimize.py`): Root position gets 1.5x the base learning rate in a separate Adam param group.
4. **Sigma schedule config** (`config.py`): Added `SIGMA_SCHEDULE` parameter.
5. **Print every 20 steps** (`optimize.py`): Reduced log verbosity, added sigma to output.

### Files Modified

| File | Changes |
|------|---------|
| `motionbert-pose/config.py` | Added `SIGMA_SCHEDULE`, changed `NUM_STEPS` from 10 to 20 |
| `motionbert-pose/optimize.py` | Added `_get_sigma()`, visibility thresholding, separate root_pos LR, sigma in print, print every 20 steps |

## Extensive Hyperparameter Search

The architect's recommended settings (200 steps, sigma schedule 80->40->15, rotation penalty scalar 5, root_pos 3x LR) were tested first but caused **universal regression** across all 10 examples:

| Configuration | NUM_STEPS | Sigma Schedule | Rot Penalty | Root LR | Mean Opt MPJPE | Improvement |
|---------------|-----------|----------------|-------------|---------|----------------|-------------|
| P2_01 baseline | 10 | fixed 50 | 10.0 | 1x | 28.21 cm | +0.88 cm |
| Architect plan | 200 | 80->40->15 | 5.0 | 3x | 32.86 cm | **-3.77 cm** |
| Conservative 1 | 200 | 80->50->30 | 10.0 | 1.5x | 32.47 cm | **-3.38 cm** |
| Lower LR | 200 | 80->50->30 | 10.0 | 1.5x (LR=0.0003) | 29.52 cm | -0.43 cm |
| Constant sigma | 100 | constant 80 | 10.0 | 1.5x | 29.52 cm | -0.43 cm |
| Constant sigma | 50 | constant 80 | 10.0 | 1.5x | 28.59 cm | +0.50 cm |
| Constant sigma | 30 | constant 80 | 10.0 | 1.5x | 28.34 cm | +0.75 cm |
| **Final** | **20** | **constant 80** | **10.0** | **1.5x** | **28.21 cm** | **+0.88 cm** |

## Root Cause: Why More Steps Hurts

The architect's analysis correctly identified that the loss was still decreasing at step 10. However, **decreasing loss does not imply improving 3D accuracy**. The optimizer minimizes the 2D reprojection loss against Stacked Hourglass detections, but:

1. **Large initial reprojection error**: The MotionBERT 3D positions, when projected to 2D, are 87-334 pixels away from the Stacked Hourglass 2D detections (depending on the example). This mismatch is because (a) root depth estimation is inaccurate, and (b) MotionBERT's 3D structure doesn't perfectly align with Stacked Hourglass's 2D keypoints.

2. **2D accuracy != 3D accuracy**: The optimizer moves 3D joints to match 2D targets, but this can worsen 3D positions. For example, pushing a joint's depth to make its projection match a 2D target moves it further from its true 3D location.

3. **No initialization penalty**: The motion penalties only penalize frame-to-frame changes, not deviation from initialization. With enough steps, the entire sequence drifts gradually away from the (good) MotionBERT initialization.

4. **Smaller sigma amplifies the problem**: Smaller sigma creates stronger gradients near 2D targets, causing more aggressive 3D movement toward potentially wrong 2D positions.

## Final Results

| Example | Det MPJPE (cm) | Opt MPJPE (cm) | Improvement |
|---------|---------------|----------------|-------------|
| 171204_pose1_sample_0 | 24.75 | 21.88 | +2.88 |
| 171204_pose2_200 | 46.43 | 43.82 | +2.61 |
| 171204_pose2_5000 | 12.78 | 12.32 | +0.46 |
| 171204_pose2_15000 | 21.35 | 19.87 | +1.48 |
| 171204_pose3_200 | 54.71 | 55.02 | -0.31 |
| 171204_pose3_4000 | 14.92 | 14.56 | +0.36 |
| 160422_ultimatum1_200 | 45.34 | 45.24 | +0.09 |
| 160422_ultimatum1_10000 | 43.95 | 43.49 | +0.46 |
| 171204_pose2_10000 | 13.54 | 12.90 | +0.65 |
| 171204_pose2_25000 | 13.11 | 12.97 | +0.14 |
| **MEAN** | **29.09** | **28.21** | **+0.88** |

- **9/10 examples improve** (vs 8/10 in P2_01 baseline)
- **Max regression: 0.31 cm** (vs 0.37 cm in P2_01 baseline)
- **Mean improvement: +0.88 cm** (same as P2_01 baseline)
- **P-MPJPE: 21.98 -> 22.00 cm** (essentially unchanged)

## Deviations from Plan

1. **NUM_STEPS = 20 instead of 200**: Extensive testing showed that more steps consistently worsens MPJPE. The optimizer converges on the wrong objective (2D reprojection to noisy targets) rather than improving 3D accuracy.

2. **Constant sigma=80 instead of coarse-to-fine schedule**: The sigma schedule infrastructure is in place and works correctly, but the fine phases (sigma=40, sigma=15) cause overfitting to noisy 2D detections. Constant sigma=80 provides the best results.

3. **ROTATION_PENALTY_SCALAR kept at 10.0 instead of reduced to 5.0**: Reducing the rotation penalty allows more deviation from initialization, which worsens results given the 2D-3D mismatch.

4. **Root_pos LR multiplier 1.5x instead of 3x**: The 3x multiplier caused excessive depth drift.

## Key Insight for Future Iterations

The FK optimization is fundamentally limited by the **2D-3D alignment gap**. The ~7cm gap between MPJPE (29.09) and P-MPJPE (21.98) represents depth/position/scale errors that the 2D-based optimizer cannot reliably fix because:

- It has no direct depth supervision
- The 2D targets come from a different model (Stacked Hourglass) than the 3D predictions (MotionBERT)
- Moving 3D positions to match 2D targets often worsens depth accuracy

To achieve the target of 24-26cm, future iterations should consider:
- **Initialization penalty**: Add a term that penalizes deviation from the MotionBERT initialization (not just frame-to-frame smoothness)
- **Depth regularization**: Use MotionBERT's relative depth ordering as a soft constraint
- **Better 2D-3D alignment**: Use MotionBERT's own 2D predictions as targets instead of Stacked Hourglass

## Validation Checklist

- [x] `test_single.py` runs successfully
- [x] `main.py` runs successfully on all 10 examples
- [x] Sigma schedule infrastructure works (visible in step output)
- [x] Visibility thresholding zeroes low-confidence joints (ankles at 0.003-0.006 -> 0)
- [x] No example regressed more than 0.5 cm
- [x] 9/10 examples improve (up from 8/10)
- [x] All functions have type hints
- [x] No new files created
