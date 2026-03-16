# Architect Plan: Phase 2, Iteration 5 (FINAL) -- Temporal Smoothing + Improved FK Targets

## Current State

| Metric | Value |
|--------|-------|
| Detector MPJPE | 29.09 cm |
| Detector P-MPJPE | 21.98 cm |
| Optimized MPJPE | 28.21 cm (+0.88 cm improvement) |
| Optimized (no ankles) | ~22 cm |
| Best single example | pose2_5000: 12.78 cm |
| Worst single example | pose3_200: 54.71 cm |

## What Has Been Tried and Why It Failed

1. **More FK optimization steps**: Overfits to noisy 2D targets (Iter 2)
2. **Coarse-to-fine sigma schedule**: Fine sigma causes overfitting (Iter 2)
3. **solvePnP root placement**: Gap is shape errors not position (Iter 3)
4. **Zeroing low-confidence joints before MotionBERT**: MotionBERT-Lite can't handle missing joints -- produces worse output (Iter 4)
5. **Improved 2D targets**: Added MotionBERT projections for low-conf joints, but still only running 20 steps with threshold=0.0, so this code path is never activated (Iter 4)

## Root Cause Analysis

After reading every file, here is where the errors come from:

### Error decomposition (from Iter 4 no-ankle diagnostics)

| Source | Estimated Contribution |
|--------|----------------------|
| Ankle errors (64 cm avg ankle MPJPE) | ~7 cm of overall 29 cm |
| Depth/scale errors (MPJPE - P-MPJPE gap) | ~7 cm |
| Shape errors (P-MPJPE floor) | ~22 cm |

The optimization can only target the first source (fixing 3D positions via 2D targets). But with noisy 2D targets, more optimization makes things worse.

### The key untried opportunity

The improved 2D target code from Iter 4 IS in the codebase (`main.py` lines 203-210, `test_single.py` lines 127-134). However, `MOTIONBERT_CONF_THRESHOLD` is 0.0, so the condition `visibility[i][j] < cfg.MOTIONBERT_CONF_THRESHOLD` is never true. **The improved targets are never used.**

To activate the improved targets WITHOUT changing MotionBERT's input (which we proved doesn't work), we need a SEPARATE threshold that only controls which 2D targets the FK optimizer uses. Currently, the same `MOTIONBERT_CONF_THRESHOLD` is used for both (a) zeroing MotionBERT input and (b) replacing FK targets. We need to decouple these.

## Implementation Plan

### Change 1 (P0, CRITICAL): Decouple FK target threshold from MotionBERT input threshold

**File: `config.py`**

Add a new parameter:
```python
# Confidence threshold for replacing FK optimization 2D targets.
# For joints below this threshold, use MotionBERT's projected 2D
# instead of (potentially garbage) Stacked Hourglass detections.
# This does NOT affect MotionBERT's input (MOTIONBERT_CONF_THRESHOLD controls that).
FK_TARGET_CONF_THRESHOLD: float = 0.1
```

Keep `MOTIONBERT_CONF_THRESHOLD` at 0.0 (proven: changing MotionBERT input makes things worse).

**File: `main.py`** -- Change the improved target logic to use the new threshold:
```python
# Replace line:
#   if visibility[i][j] < cfg.MOTIONBERT_CONF_THRESHOLD:
# With:
    if visibility[i][j] < cfg.FK_TARGET_CONF_THRESHOLD:
```

**File: `test_single.py`** -- Same change.

### Change 2 (P0): Increase optimization steps to test with improved targets

**File: `config.py`**

The "more steps hurts" finding from Iter 2 was with the OLD noisy targets. With improved targets (MotionBERT projections replacing garbage ankle positions), more steps should no longer overfit to bad targets.

Test progression:
1. First test: `NUM_STEPS=20` with `FK_TARGET_CONF_THRESHOLD=0.1` (just activate targets, keep steps same)
2. If (1) shows improvement: test `NUM_STEPS=50`
3. If (2) shows more improvement: test `NUM_STEPS=100`
4. Stop at the step count that maximizes improvement

The developer should test at least the first two configurations and pick the best.

### Change 3 (P1): Add an initialization anchor penalty

The optimizer currently has no penalty for drifting away from the MotionBERT initialization. The motion penalties only penalize frame-to-frame changes, so the entire sequence can gradually drift. Adding a soft anchor to the initial positions prevents this.

**File: `scoring.py`** -- Add a new penalty function:

```python
def initialization_penalty(
    positions_3d: torch.Tensor,
    initial_positions_3d: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Penalize deviation from initial (MotionBERT) positions.

    Only penalizes joints that the detector is confident about.
    Low-visibility joints are free to move (the optimizer should fix them).

    Args:
        positions_3d: (17, 3) current optimized positions.
        initial_positions_3d: (17, 3) MotionBERT initial positions.
        visibility: (17,) confidence weights.

    Returns:
        Scalar penalty.
    """
    diff = positions_3d - initial_positions_3d
    sq_dist = (diff ** 2).sum(dim=-1)  # (17,)
    # Weight by visibility -- high-confidence joints are anchored more
    return (sq_dist * visibility).sum()
```

**File: `config.py`** -- Add weight:
```python
# Weight for initialization anchor penalty.
# Prevents optimizer from drifting away from MotionBERT predictions.
INIT_ANCHOR_WEIGHT: float = 5.0
```

**File: `optimize.py`** -- Pass initial positions to scoring, include penalty in loss.

**File: `scoring.py`** -- Update `compute_total_score` to accept and use initial positions.

The key insight is that this penalty is visibility-weighted: high-confidence joints (which MotionBERT got from good 2D input) are anchored strongly, while low-confidence joints (ankles etc.) are free to move toward whatever the optimizer finds. This prevents the "drift from good initialization" problem identified in Iter 2.

### Change 4 (P2): Add temporal smoothing penalty on ALL joint positions (not just root)

Currently, `motion_penalty_position` only penalizes root (hip) position jumps. But frame-to-frame jitter in other joints also contributes to error. Add a penalty on ALL joint position velocities, weighted by visibility.

**File: `scoring.py`** -- Add:

```python
def motion_penalty_all_joints(
    positions_prev: torch.Tensor,
    positions_curr: torch.Tensor,
) -> torch.Tensor:
    """Penalize position jumps for ALL joints between consecutive frames.

    Args:
        positions_prev: (17, 3) previous frame.
        positions_curr: (17, 3) current frame.

    Returns:
        Scalar: sum of squared displacements.
    """
    diff = positions_curr - positions_prev
    return (diff ** 2).sum()
```

**File: `config.py`**:
```python
ALL_JOINTS_SMOOTH_WEIGHT: float = 2.0
```

**File: `scoring.py`** -- Include in `compute_total_score`.

## Priority and Testing Order

1. **First** (Quick win): Implement Change 1 + Change 2 (decouple threshold + test step counts). This is the highest-signal change because it activates code that already exists but was never tested properly.

2. **Second** (Moderate complexity): Implement Change 3 (init anchor). This directly addresses the "drift from initialization" problem identified in Iter 2.

3. **Third** (If time): Implement Change 4 (all-joint smoothing). This is lower priority because the motion penalties already exist for root position and rotations.

The developer should test after each change and report results. If Change 1 alone gives significant improvement, Changes 3 and 4 may not be needed.

## Expected Results

### With Change 1 alone (FK target threshold = 0.1)
- Ankles: The optimizer will no longer try to match garbage 2D ankle positions. Instead it matches MotionBERT's projected ankles, which are already the best available estimate. Expected ankle MPJPE: 50-60 cm -> 40-50 cm.
- Other joints: No change (they already have good 2D targets).
- Overall: MPJPE 28.21 -> ~26-27 cm.

### With Change 1 + more steps (NUM_STEPS=50)
- With clean targets, more steps should now help rather than hurt.
- Expected: MPJPE ~25-26 cm.

### With all changes
- Best case: MPJPE ~24-25 cm.
- Conservative: MPJPE ~26-27 cm.

## Validation Plan

1. Run `test_single.py` on example 0 with:
   - Baseline (FK_TARGET_CONF_THRESHOLD=0.0, NUM_STEPS=20) -- should match 21.71 cm opt
   - FK_TARGET_CONF_THRESHOLD=0.1, NUM_STEPS=20
   - FK_TARGET_CONF_THRESHOLD=0.1, NUM_STEPS=50
   - Compare per-joint MPJPE, especially ankles

2. Run `main.py` with best config on all 10 examples.
   - Verify mean MPJPE improves
   - Check no example regresses by more than 2 cm
   - Compare per-joint averaged across examples

3. Report complete table: baseline vs optimized for each example, plus mean.

## What NOT to Change

- **MotionBERT input processing** (`detect.py`, `MOTIONBERT_CONF_THRESHOLD`): Keep at 0.0. Proven harmful to change.
- **Joint mappings** (`skeleton.py`): Verified correct against official MotionBERT.
- **FK code** (`fk.py`): Working correctly (0.0000 cm roundtrip error).
- **Camera model** (`camera.py`): Correct.
- **Evaluation code** (`evaluate.py`): Correct.

## Risk Assessment

- **Change 1**: Very low risk. The code already exists; we're just activating it with a properly decoupled threshold. Worst case: no improvement (optimizer ignores the better targets at sigma=80).
- **Change 2**: Low risk. Easy to revert step count if it regresses. But with better targets, overfitting risk is lower.
- **Change 3**: Medium risk. The anchor weight needs tuning -- too high locks everything to MotionBERT, too low has no effect. Start at 5.0 and test.
- **Change 4**: Low risk. Smoothing is conservative and unlikely to hurt.

## Summary

The single biggest opportunity is **activating the improved 2D targets that already exist in the code but are gated behind a threshold of 0.0**. This requires decoupling the FK target threshold from the MotionBERT input threshold (a ~5-line change across 3 files). Combined with testing higher step counts (which may now help instead of hurt), and an initialization anchor to prevent drift, this should yield measurable MPJPE improvement.
