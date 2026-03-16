# Architect Plan: Phase 2, Iteration 4 -- Input Quality & Ankle Error Fix

## Current State

- Detector baseline: 29.09 cm MPJPE, 21.98 cm P-MPJPE
- FK optimization: 28.21 cm MPJPE (+0.88 cm improvement, marginal)
- solvePnP for root placement: no help (Iteration 3 finding)
- More optimization steps HURT (overfits to noisy 2D targets)

## Deep Dive Root Cause Analysis

After reading every file in the pipeline and analyzing per-joint errors across all 10 examples, I identified the **dominant source of error**: low-quality Stacked Hourglass 2D detections for ankles being fed directly to MotionBERT.

### Per-Joint Error Breakdown (averaged across all 10 examples)

| Joint | Det MPJPE (cm) | Contribution |
|-------|---------------|-------------|
| RAnkle | **64.46** | Catastrophic |
| LAnkle | **63.71** | Catastrophic |
| RKnee | **30.26** | High |
| LKnee | **24.78** | High |
| LElbow | 27.76 | Moderate |
| LWrist | 26.39 | Moderate |
| RWrist | 25.54 | Moderate |
| RElbow | 25.50 | Moderate |
| RShoulder | 21.99 | Moderate |
| LShoulder | 21.56 | Moderate |
| RHip | 8.67 | Low |
| LHip | 8.43 | Low |

**Ankles alone contribute 64 cm average error** -- more than double any other joint. If we could fix just the ankles to match knee-level accuracy (25 cm), the mean MPJPE would drop from 29.09 to ~22.5 cm.

### Why Ankles Are So Bad: The Full Chain of Failure

1. **Stacked Hourglass almost never detects ankles** -- confidence is below 0.1 in **97% of all frames** across all 10 examples. The feet are often partially occluded or at image edges in the CMU Panoptic HD camera.

2. **Low-confidence detections have garbage 2D positions** -- Example: LAnkle detected at y=737 (hip height) when it should be at y=1100+ (below knees). The detector literally places the ankle at the wrong body part.

3. **These garbage positions are fed to MotionBERT as "real" detections** -- because the confidence values are ~0.003 (not exactly zero), they pass through `crop_scale`'s filter (`motion[..., 2] != 0`) and MotionBERT processes them as valid joint positions.

4. **MotionBERT trusts the bad input** and produces 3D output where ankles are at hip height -- For frame 0 of pose2_5000: LAnkle Y = 9.7 cm below hip (detected), vs 81.3 cm below hip (ground truth). A **72 cm error** in the vertical direction alone.

5. **This error is baked into the root-relative 3D structure** and cannot be fixed by camera-space placement (solvePnP, depth estimation, etc.) or FK optimization.

### The Critical Insight: MotionBERT Handles Missing Joints

MotionBERT was trained with `no_conf: False`, meaning it takes (x, y, confidence) as input. The official `crop_scale` function explicitly filters joints with `confidence == 0` from the bounding box computation. **Setting confidence to 0 for unreliable joints tells MotionBERT they are missing**, allowing the model to infer their positions from:
- Temporal context (adjacent frames)
- Structural context (other visible joints in the same frame)
- Learned human pose priors

This is exactly what the model was designed to handle. We're currently defeating this capability by feeding it garbage positions with nonzero confidence.

## Implementation Plan

### Change 1 (P0, CRITICAL): Threshold low-confidence 2D keypoints before MotionBERT

**File: `motionbert-pose/detect.py`**, function `run_motionbert`

After converting MPII to H36M format and before calling `crop_scale`, zero out joints with confidence below a threshold. This is the single highest-impact change.

```python
# In run_motionbert, after building keypoints_h36m and before crop_scale:
MOTIONBERT_CONF_THRESHOLD = 0.1  # or define in config.py

# Zero out low-confidence joints so MotionBERT treats them as missing
for i in range(n_frames):
    for j in range(17):
        if keypoints_h36m[i, j, 2] < MOTIONBERT_CONF_THRESHOLD:
            keypoints_h36m[i, j, :] = 0.0  # Zero x, y, and confidence
```

**Why this threshold**: 0.1 is conservative -- ankles (the main problem) have confidence ~0.003, so they'll be zeroed. Joints with moderate confidence (0.1-0.5) are kept because their 2D positions are usually reasonable even if imprecise.

**Define the threshold in `config.py`:**
```python
# Confidence threshold for 2D keypoints fed to MotionBERT.
# Joints below this threshold have their coordinates zeroed out,
# telling MotionBERT to treat them as missing and infer from context.
MOTIONBERT_CONF_THRESHOLD: float = 0.1
```

### Change 2 (P0): Also threshold at 0.3 as an experiment

The developer should implement the thresholding with a configurable parameter and test at least two values:
- `MOTIONBERT_CONF_THRESHOLD = 0.1` (conservative: only zeros out clearly bad detections)
- `MOTIONBERT_CONF_THRESHOLD = 0.3` (aggressive: zeros out more uncertain detections)

Compare both against the current baseline (no thresholding, equivalent to threshold=0.0).

### Change 3 (P1): Improve the 2D target quality for FK optimization

Currently, the FK optimization uses 2D targets from Stacked Hourglass. These are the same garbage positions that caused the ankle errors in the first place. After MotionBERT produces better 3D with missing-joint inference, we should use MotionBERT's own projected 2D positions as targets for joints where the original 2D detection was unreliable.

**File: `motionbert-pose/main.py`**, in the section that calls `run_optimization`

For low-confidence joints, replace the Stacked Hourglass 2D target with the 2D projection of MotionBERT's 3D prediction:

```python
# Before calling run_optimization:
# For joints where Stacked Hourglass is unreliable, use MotionBERT's
# predicted 2D projection as the target instead
improved_target_2d: list[np.ndarray] = []
for i in range(len(frames_rgb)):
    target = kp_2d[i].copy()
    # Project MotionBERT 3D prediction to 2D
    mb_projected = camera.world_to_image(det_cam_positions[i])
    for j in range(17):
        if visibility[i][j] < cfg.MOTIONBERT_CONF_THRESHOLD:
            target[j] = mb_projected[j]
    improved_target_2d.append(target)

# Use improved_target_2d instead of kp_2d for optimization
optimized_3d, bone_lengths_final, loss_history = run_optimization(
    initial_positions_cam=det_cam_positions,
    target_2d=improved_target_2d,  # <-- changed
    visibility=visibility,
    camera=camera,
)
```

This ensures FK optimization doesn't try to move the 3D skeleton toward garbage 2D ankle positions, which was the root cause of the "more steps hurts" problem.

### Change 4 (P2): Consider excluding ankles from EVAL_JOINTS

This is NOT a code change but a diagnostic: compute MPJPE both with and without ankles. Report both numbers. This tells us:
1. What our "achievable" MPJPE would be if we had perfect ankles
2. How much of the overall error is due to ankles vs other joints

**File: `motionbert-pose/evaluate.py`**

Add a second eval metric that excludes ankles (joints 3 and 6):

```python
EVAL_JOINTS_NO_ANKLES: list[int] = [1, 2, 4, 5, 11, 12, 13, 14, 15, 16]
```

And compute `det_mpjpe_no_ankles` alongside `det_mpjpe` in `compute_comparison`.

## Expected Results

### Optimistic Scenario (threshold helps MotionBERT significantly)
| Metric | Current | Expected |
|--------|---------|----------|
| Mean MPJPE (detector) | 29.09 cm | ~22-25 cm |
| Ankle MPJPE | ~64 cm | ~30-40 cm |
| Knee MPJPE | ~27 cm | ~20-25 cm |
| Mean MPJPE (optimized) | 28.21 cm | ~21-24 cm |

### Conservative Scenario (threshold helps but MotionBERT can't fully hallucinate ankles)
| Metric | Current | Expected |
|--------|---------|----------|
| Mean MPJPE (detector) | 29.09 cm | ~25-28 cm |
| Ankle MPJPE | ~64 cm | ~40-55 cm |
| Mean MPJPE (optimized) | 28.21 cm | ~24-27 cm |

The key metric to watch is ankle MPJPE. Even a 50% reduction (64 -> 32 cm) would drop overall MPJPE by ~5 cm.

## Validation Plan

1. Run `test_single.py` with threshold=0.0 (current), 0.1, and 0.3 on example 0
   - Compare per-joint MPJPE, especially ankles
   - Verify no regression in upper body joints
   - Check that MotionBERT's 3D ankle predictions are more reasonable

2. Run `main.py` with the best threshold on all 10 examples
   - Compare mean MPJPE, P-MPJPE across all examples
   - Check per-joint MPJPE averaged across examples
   - Verify the P-MPJPE anomaly (P > MPJPE for some examples) resolves
     (it was caused by Procrustes scaling up predictions with broken ankle bone lengths)

3. Report both "with ankles" and "without ankles" MPJPE for context

## What NOT to Change

- **FK code (`fk.py`)**: Working correctly (0.0000 cm roundtrip error)
- **Skeleton definition (`skeleton.py`)**: Joint mappings are correct (verified against official MotionBERT)
- **Evaluation code (`evaluate.py`)**: Working correctly (P-MPJPE > MPJPE anomaly is caused by the data, not the code)
- **Camera model (`camera.py`)**: Correct
- **MPII->H36M mapping**: Verified correct against official MotionBERT halpe2h36m
- **COCO19->H36M mapping**: Verified correct for ground truth
- **Stacked Hourglass model/inference**: Don't change the 2D detector itself
- **MotionBERT model/inference**: Don't change the model architecture or weights

## Risk Assessment

- **Low risk**: Zeroing out clearly bad detections (confidence < 0.1) should only help. These joints have garbage 2D positions that actively corrupt the pipeline.
- **Medium risk**: Threshold=0.3 is more aggressive and may zero out joints that have reasonable positions but low confidence. This needs testing.
- **Low risk**: Using MotionBERT projections as 2D targets for FK optimization is conceptually sound -- it prevents optimizing toward known-bad targets.

## Summary

The single biggest opportunity is telling MotionBERT that low-confidence joints are missing rather than feeding it garbage positions. This is a 2-line code change that addresses the dominant source of error (64 cm average ankle error, present in 97% of frames). Everything else we've tried (FK optimization tuning, solvePnP, more steps) was treating symptoms while the root cause -- bad input to MotionBERT -- went unaddressed.
