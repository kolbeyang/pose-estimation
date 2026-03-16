# Architect Plan: Phase 2, Iteration 2 -- Maximize FK Optimization Impact

## Current State

- Detector baseline MPJPE: 29.09 cm, P-MPJPE: 21.98 cm
- After 10-step FK optimization: 28.21 cm MPJPE (+0.88 cm improvement)
- 8/10 examples improve, 2 regress slightly (< 0.4 cm)
- The loss is decreasing over 10 steps, but the step budget is far too small
- The spec's prior implementation achieved ~19% improvement with 300 steps

## Root Cause Analysis: Why Optimization Barely Helps

I identified **five issues**, ranked by expected impact:

### Issue 1 (HIGH): Only 10 optimization steps -- way too few

The current `config.py` line 33: `NUM_STEPS: int = 10`. The mediapipe-pose reference uses 300 steps (`mediapipe-pose/config.py` line 38). The loss is clearly still decreasing at step 10 (e.g., 12319 -> 12125 for ultimatum1_10000). With only 10 steps the optimizer has barely moved from initialization.

**Evidence**: Loss drops ~1-2% in 10 steps across all examples. At 300 steps with Adam, we'd expect convergence to a much better local minimum.

### Issue 2 (HIGH): No coarse-to-fine sigma schedule

The current code uses a fixed `SIGMA = 50.0` throughout all steps (`config.py` line 38, `optimize.py` line 160 just reads `cfg.SIGMA`). The mediapipe-pose reference also uses fixed sigma=50, but DOCS.md section "Phase 2: Full Skeleton with BVH Data" explicitly describes the benefit: "3-phase coarse-to-fine: blur=20 -> blur=8 -> blur=0".

A large sigma creates a wide gradient basin (good for coarse alignment) but poor precision. A small sigma gives sharp gradients near the target (good for fine-tuning) but no gradient far away. A schedule gives both.

For the analytical Gaussian scoring function, sigma controls the gradient basin directly:
- `score = -dist^2 / (2 * sigma^2)`
- Gradient w.r.t. projected position = `dist / sigma^2`
- Large sigma = weak gradient = wider basin
- Small sigma = strong gradient = sharp convergence near target

**Proposed schedule for 200 steps**:
- Phase 1 (steps 0-79): sigma=80 -- coarse alignment, moves root position and large angles
- Phase 2 (steps 80-159): sigma=40 -- medium refinement
- Phase 3 (steps 160-199): sigma=15 -- fine-tuning, precise joint placement

### Issue 3 (MEDIUM): Heatmap-based scoring uses detected 2D keypoints, not actual heatmaps

The current scoring (`scoring.py` lines 10-34) computes distance from the projected 3D skeleton to the extracted 2D keypoint peaks. This is a point-to-point distance, not a true heatmap score. The actual Stacked Hourglass heatmaps (16 channels, 64x64) are available but unused.

The 2D keypoint extraction (`detect.py` `_parse_heatmaps`, lines 169-190) takes the argmax of each 64x64 heatmap. This loses all information about detection uncertainty. For example, if the heatmap has a broad peak (uncertain detection), the argmax gives no indication -- the optimizer treats it the same as a sharp peak (confident detection).

However, **this is a deliberate design choice** and the analytical Gaussian approach is well-validated. The visibility weight already down-weights uncertain joints. Using actual heatmaps would be more expensive and complex. I recommend keeping the analytical approach but exploring better use of the heatmap confidence for visibility weighting (see Issue 4).

### Issue 4 (MEDIUM): Visibility weights are raw Stacked Hourglass confidence, not thresholded

Looking at the data, the visibility values are the raw heatmap peak values from Stacked Hourglass. For example, joint 3 (RAnkle) has vis=0.004 and joint 6 (LAnkle) has vis=0.003 -- these are effectively invisible but still contribute to the score (weight 0.004). Meanwhile, well-detected joints like Thorax have vis=0.801.

The config has `VISIBILITY_THRESHOLD: float = 0.5` (`config.py` line 68) but **it's never used**. Looking at `optimize.py`, the raw visibility arrays are passed directly to `compute_total_score` without any thresholding.

**Impact**: Low-confidence joints with bad 2D detections pull the optimizer in wrong directions. For pose3_200 (worst example), the shoulder errors are 80-94 cm -- likely caused by bad 2D detections that the optimizer tries to fit.

### Issue 5 (LOW-MEDIUM): Per-joint rotation penalties may be too high for extremities

The current penalty weights have `ROTATION_PENALTY_SCALAR = 10.0` applied to all joints. Wrists get `10.0 * 0.1 = 1.0`, elbows get `10.0 * 0.3 = 3.0`. These penalties fight against the heatmap score. With sigma=50, the heatmap score gradient is `dist / sigma^2 = dist / 2500`. For a 50-pixel error, that's 0.02 per joint. The rotation penalty gradient can easily dominate, preventing the optimizer from correcting large errors.

This is less critical once we increase steps (the optimizer will converge even with high penalties, just more slowly), but reducing rotation penalties for extremities could help.

## Recommended Changes (Priority Order)

### Change 1: Increase NUM_STEPS to 200, add coarse-to-fine sigma schedule

**File: `motionbert-pose/config.py`**

Replace:
```python
NUM_STEPS: int = 10  # Start small for testing, increase after confirming correctness
```
With:
```python
NUM_STEPS: int = 200
```

Add new config constants:
```python
# Coarse-to-fine sigma schedule: (fraction_of_steps, sigma)
# Phase 1: wide basin for coarse alignment
# Phase 2: medium for refinement
# Phase 3: narrow for precision
SIGMA_SCHEDULE: list[tuple[float, float]] = [
    (0.4, 80.0),   # Steps 0-39%: coarse
    (0.8, 40.0),   # Steps 40-79%: medium
    (1.0, 15.0),   # Steps 80-100%: fine
]
```

**File: `motionbert-pose/optimize.py`**

In the optimization loop (line 130), replace the fixed sigma with a schedule lookup:

Current (`optimize.py` line 154-163):
```python
total_score, details = compute_total_score(
    ...
    cfg.SIGMA,
    ...
)
```

Change to compute sigma from schedule at each step. Add a helper function:

```python
def _get_sigma(step: int, num_steps: int) -> float:
    """Get sigma for current step from coarse-to-fine schedule."""
    progress: float = step / max(num_steps - 1, 1)
    for frac, sigma in cfg.SIGMA_SCHEDULE:
        if progress <= frac:
            return sigma
    return cfg.SIGMA_SCHEDULE[-1][1]
```

Replace `cfg.SIGMA` in the `compute_total_score` call (line 160) with `_get_sigma(step, num_steps)`.

Also update the print statement (line 177-182) to print sigma and only print every 20 steps instead of every step:

```python
if step % 20 == 0 or step == num_steps - 1:
    print(
        f"    Step {step:4d}/{num_steps}  "
        f"loss={loss.item():.1f}  "
        f"heatmap={details['heatmap']:.1f}  "
        f"sigma={sigma:.0f}  "
        f"pos_p={details['pos_penalty']:.4f}  "
        f"rot_p={details['rot_penalty']:.4f}"
    )
```

### Change 2: Apply visibility threshold before optimization

**File: `motionbert-pose/optimize.py`**

After building `visibility_t` (lines 109-111), add thresholding:

```python
# Apply visibility threshold -- zero out low-confidence joints
for i in range(len(visibility_t)):
    visibility_t[i] = torch.where(
        visibility_t[i] >= cfg.VISIBILITY_THRESHOLD,
        visibility_t[i],
        torch.zeros_like(visibility_t[i]),
    )
```

This ensures joints with confidence < 0.5 (like the ankles at 0.003-0.004) don't corrupt the optimization.

### Change 3: Reduce rotation penalty scalar for better convergence

**File: `motionbert-pose/config.py`**

Change:
```python
ROTATION_PENALTY_SCALAR: float = 10.0
```
To:
```python
ROTATION_PENALTY_SCALAR: float = 5.0
```

This halves all rotation penalties. The rationale: with 200 steps and coarse-to-fine sigma, we have enough steps for temporal smoothing to emerge naturally. The current penalties are too strong relative to the heatmap signal, especially for extremity joints. The mediapipe-pose reference uses the same `ROTATION_PENALTY_SCALAR = 10.0`, but mediapipe has better 2D detections (33 landmarks vs 16), so the heatmap signal is stronger there.

### Change 4: Increase position penalty weight to stabilize root

**File: `motionbert-pose/config.py`**

Keep `POSITION_PENALTY_WEIGHT: float = 50.0` unchanged. This is already well-tuned in the reference.

However, with the sigma schedule, the heatmap score magnitude changes across phases. At sigma=80, the score is `exp(-dist^2/12800)` which is smaller gradient. At sigma=15, it's `exp(-dist^2/450)` which has a much stronger gradient. The position penalty is constant. This means:
- In coarse phase: position penalty dominates (good, keeps root stable)
- In fine phase: heatmap signal dominates (good, allows precise joint placement)

This is actually the desired behavior, so no change needed here.

### Change 5 (Optional): Add root_pos learning rate multiplier

The mediapipe-pose reference doesn't do this, but DOCS.md mentions "root_pos gets 5x LR" was helpful for the BVH skeleton. For MotionBERT, the root position may need more aggressive correction since depth estimates are noisy.

**File: `motionbert-pose/optimize.py`**

Change the optimizer setup (lines 120-124) to give root position a separate param group:

Current:
```python
pose_params: list[torch.Tensor] = param_root_pos + param_root_rot + param_local_rots
optimizer: torch.optim.Adam = torch.optim.Adam([
    {"params": pose_params, "lr": cfg.LEARNING_RATE},
    {"params": [param_bone_lengths], "lr": cfg.BONE_LENGTH_LR},
])
```

New:
```python
angle_params: list[torch.Tensor] = param_root_rot + param_local_rots
optimizer: torch.optim.Adam = torch.optim.Adam([
    {"params": param_root_pos, "lr": cfg.LEARNING_RATE * 3.0},
    {"params": angle_params, "lr": cfg.LEARNING_RATE},
    {"params": [param_bone_lengths], "lr": cfg.BONE_LENGTH_LR},
])
```

The 3x multiplier for root_pos helps it converge faster to the correct depth, which is the primary source of MPJPE error (the gap between MPJPE and P-MPJPE is ~7 cm, meaning depth/position is responsible for ~7 cm of error).

## Summary of Changes

| Priority | File | Change | Lines |
|----------|------|--------|-------|
| **P0** | `config.py` | `NUM_STEPS = 200`, add `SIGMA_SCHEDULE` | Lines 33, new ~lines 39-44 |
| **P0** | `optimize.py` | Add `_get_sigma()` helper, use schedule in loop | New function, line ~160 |
| **P0** | `optimize.py` | Print every 20 steps instead of every step | Line ~177 |
| **P1** | `optimize.py` | Apply visibility threshold to `visibility_t` | After line ~111 |
| **P1** | `config.py` | `ROTATION_PENALTY_SCALAR = 5.0` | Line 46 |
| **P2** | `optimize.py` | Separate root_pos param group with 3x LR | Lines ~120-124 |

## Expected Impact

| Metric | Current | Expected After Changes |
|--------|---------|----------------------|
| Mean MPJPE | 28.21 cm | ~24-26 cm |
| Mean improvement | 0.88 cm | ~3-5 cm |
| Examples improving | 8/10 | 9-10/10 |
| Max regression | 0.37 cm | < 0.5 cm |

The biggest gain will come from Change 1 (more steps + sigma schedule). Changes 2-3 are supporting improvements that prevent the optimizer from being pulled by bad detections.

The target of ~24-26 cm is based on: P-MPJPE is ~22 cm, so the remaining 7 cm gap between MPJPE and P-MPJPE represents depth/position/scale errors. The optimizer should be able to close ~3-5 cm of that gap with 200 steps and coarse-to-fine sigma.

## Validation Plan

1. Run `test_single.py` first to verify the changes work on one example
2. Confirm loss decreases through all three sigma phases
3. Confirm the sigma transitions are visible in the printed output
4. Run `main.py` on all 10 examples
5. Verify mean MPJPE improved over current 28.21 cm baseline
6. Verify no example regressed by more than 2 cm
7. Check that worst examples (pose3_200, pose2_200, ultimatum1 examples) show improvement -- these have the most room for the optimizer to help

## What NOT to Change

- **FK code (`fk.py`)**: Roundtrip error is 0.0000 cm. The FK is correct.
- **Scoring function type**: Keep analytical Gaussian. Don't switch to actual heatmap sampling.
- **Evaluation code (`evaluate.py`)**: Working correctly.
- **Detection pipeline (`detect.py`)**: The 2D/3D detection is the upstream bottleneck for some examples, but changing it is out of scope for this iteration.
- **Bone length matching (`motionbert_to_camera_space`)**: Working well, produces correct scale.
