# Developer Report: Phase 2, Iteration 4 -- Input Quality & Ankle Error Fix

## Goal

Address the dominant error source: garbage ankle detections from Stacked Hourglass (confidence < 0.1 in 97% of frames) that corrupt MotionBERT's input and cause 64 cm average ankle error. The architect's hypothesis was that zeroing out low-confidence joints before MotionBERT would let the model infer missing positions from temporal/structural context. Target: reduce overall MPJPE from 29.09 cm to 22-25 cm.

## What Was Implemented

### Change 1: Confidence threshold infrastructure (config.py, detect.py)

Added `MOTIONBERT_CONF_THRESHOLD` to `config.py` (default 0.0). In `detect.py:run_motionbert`, added logic to zero out all three channels (x, y, confidence) for joints below the threshold before `crop_scale` normalization. This tells MotionBERT the joints are missing.

### Change 2: No-ankle MPJPE diagnostic (evaluate.py)

Added `EVAL_JOINTS_NO_ANKLES`, `det_mpjpe_no_ankles`, `det_p_mpjpe_no_ankles`, `opt_mpjpe_no_ankles`, `opt_p_mpjpe_no_ankles` to both `compute_comparison` and `compute_comparison_with_optimization`. These metrics exclude joints 3 (RAnkle) and 6 (LAnkle).

### Change 3: Improved FK optimization targets (main.py, test_single.py)

For joints where Stacked Hourglass confidence is below `MOTIONBERT_CONF_THRESHOLD`, the FK optimizer now uses MotionBERT's projected 2D positions as targets instead of the garbage Stacked Hourglass detections. With threshold=0.0, this has no effect (no joints are replaced).

## Test Results

### Threshold=0.1: FAILED (significantly worse)

| Example | Baseline (0.0) | Threshold=0.1 | Change |
|---------|---------------|---------------|--------|
| example 0 (pose1_sample_0) | 24.75 cm | 44.85 cm | +20.10 cm (worse) |
| example 2 (pose2_5000) | 12.78 cm | 34.63 cm | +21.85 cm (worse) |

Per-joint analysis for example 2 (pose2_5000):

| Joint | Baseline | Threshold=0.1 |
|-------|----------|---------------|
| RAnkle | 23.86 cm | 164.38 cm |
| LAnkle | 67.88 cm | 172.14 cm |
| RKnee | 3.77 cm | 6.94 cm |
| LKnee | 4.57 cm | 8.08 cm |
| Upper body avg | ~8 cm | ~10 cm |

### Threshold=0.3: Same as 0.1

For example 2, threshold=0.3 zeroed the same joints as 0.1 (100/850 = 11.8%). No joints had confidence between 0.1-0.3, so results were identical: 34.63 cm MPJPE.

### Root Cause: MotionBERT-Lite Does NOT Handle Missing Joints

The architect's hypothesis that "MotionBERT handles missing joints" was incorrect for this model variant. Key evidence:

1. **crop_scale normalization changes**: Zeroing ankle positions shrinks the bounding box. For example 2: scale went from 988.7 (baseline) to 852.3 (threshold=0.1), shifting the entire normalization and corrupting ALL joint positions.

2. **MotionBERT produces distorted 3D**: With zeroed ankles, the model placed ankles at 3x normal bone length from knees (1.17m vs 0.40m default). The model was not trained on inputs with zero-valued missing joints.

3. **The problem is architectural**: MotionBERT-Lite's `no_conf: False` flag means it takes confidence as input but was likely trained with all joints present. Zeroing joints produces out-of-distribution inputs.

### Baseline (Threshold=0.0): Unchanged

With threshold=0.0, the pipeline matches the previous iteration exactly:

| Example | Det MPJPE | Opt MPJPE | Improvement |
|---------|-----------|-----------|-------------|
| pose1_sample_0 | 24.75 | 21.71 | +3.04 |
| pose2_200 | 46.43 | 43.72 | +2.72 |
| pose2_5000 | 12.78 | 12.99 | -0.20 |
| pose2_15000 | 21.35 | 19.86 | +1.49 |
| pose3_200 | 54.71 | 55.02 | -0.31 |
| pose3_4000 | 14.92 | 14.17 | +0.75 |
| ultimatum1_200 | 45.34 | 45.26 | +0.07 |
| ultimatum1_10000 | 43.95 | 43.43 | +0.52 |
| pose2_10000 | 13.54 | 13.11 | +0.43 |
| pose2_25000 | 13.11 | 12.86 | +0.25 |
| **MEAN** | **29.09** | **28.21** | **+0.88** |

### No-Ankle MPJPE (Key Diagnostic Finding)

| Example | Det MPJPE | Det (no ankles) | Ankle contribution |
|---------|-----------|-----------------|-------------------|
| pose1_sample_0 | 24.75 | 14.90 | 9.85 cm |
| pose2_200 | 46.43 | 35.66 | 10.77 cm |
| pose2_5000 | 12.78 | 6.17 | 6.61 cm |
| pose2_15000 | 21.35 | 8.29 | 13.06 cm |
| pose3_200 | 54.71 | 56.18 | -1.47 cm (ankles better than avg) |
| pose3_4000 | 14.92 | 4.42 | 10.50 cm |
| ultimatum1_200 | 45.34 | 44.42 | 0.92 cm |
| ultimatum1_10000 | 43.95 | 35.46 | 8.49 cm |
| pose2_10000 | 13.54 | 8.00 | 5.54 cm |
| pose2_25000 | 13.11 | 7.40 | 5.71 cm |

Ankles account for approximately 7 cm of the 29.09 cm mean MPJPE. However, the best examples (pose2_5000, pose3_4000, pose2_10000, pose2_25000) already achieve 4-8 cm no-ankle MPJPE, showing excellent upper body tracking.

## Files Changed

- `/motionbert-pose/config.py` -- Added `MOTIONBERT_CONF_THRESHOLD` (set to 0.0)
- `/motionbert-pose/detect.py` -- Added confidence thresholding before crop_scale (inactive at 0.0)
- `/motionbert-pose/evaluate.py` -- Added `EVAL_JOINTS_NO_ANKLES` and no-ankle metrics
- `/motionbert-pose/main.py` -- Added improved 2D targets for FK optimization + no-ankle metric printing
- `/motionbert-pose/test_single.py` -- Same FK target improvement + no-ankle metric printing

## Conclusion

The architect's primary hypothesis (zeroing low-confidence joints helps MotionBERT) was **disproven**. MotionBERT-Lite does not gracefully handle missing joints. Zeroing ankles caused 2-3x worse ankle error due to:
1. Distorted crop_scale normalization (smaller bounding box)
2. Out-of-distribution input (model never saw zero-valued joints during training)

The threshold is left at 0.0 (no change to MotionBERT input). The infrastructure for thresholding remains in place for future experiments.

The no-ankle diagnostic reveals that ankle error contributes ~7 cm to the 29.09 cm mean. The pipeline achieves 4-8 cm MPJPE on upper body joints for good examples. Fixing ankles requires a different approach than input masking -- likely a different 2D detector (e.g., ViTPose, HRNet) that can reliably detect ankles, or post-processing that uses kinematic constraints to correct ankle positions from knee + bone length priors.
