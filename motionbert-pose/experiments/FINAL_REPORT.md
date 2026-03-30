# Phase 3: Performance and Latency Reduction -- Final Report

## Baseline Performance (Non-Batched, 100 steps, LR=0.001)

Profiled on 1 example (100 frames):

| Stage | Time | % Total |
|-------|------|---------|
| Detection (YOLO + SH + MotionBERT) | 73.8s | 54% |
| Optimization (100 steps) | 61.5s | 45% |
| Evaluation | <0.1s | <1% |
| **Total** | **135.4s** | **100%** |

Per optimization step breakdown:
- FK + Projection: 128ms (21%)
- Scoring: 100ms (16%)
- Backward pass: 363ms (60%)
- Step total: 608ms

**Bottleneck:** The backward pass dominated due to a large autograd graph built from 100 sequential Python loops (one per frame), each containing 16 sequential Rodrigues calls (one per joint) and 14 individual grid_sample calls (one per heatmap joint).

## Experiments

### Experiment 1+2: Vectorize FK and Scoring (WINNER)

**Changes:**
- `fk.py`: Added `forward_kinematics_batch()` -- vectorizes Rodrigues formula and kinematic tree walk across frames using batch matrix operations
- `scoring.py`: Added `heatmap_score_batch()` -- single `grid_sample` call for all F*14 joints/frames; vectorized motion and rotation penalties
- `optimize.py`: Added `run_optimization_batched()` -- stacks all per-frame parameters into batch tensors, uses batch FK and batch scoring

**Results (all 17 examples):**

| Metric | Unbatched (1ex) | Batched (17ex) |
|--------|---------|---------|
| Opt MPJPE | 44.99 cm | 38.69 cm |
| Optimization time | 61.5s | 0.96s |
| Per-step time | 608ms | ~10ms |
| Total latency/frame | 1.35s | 0.52s |

**Speedup: 64x for optimization, identical accuracy.**

Loss curves are identical between batched and unbatched (verified on same example -- same starting loss 6835.9, same ending loss 6685.5).

### Experiment 3: Reduce Steps + Increase LR

Smoke tested on 2 examples, then full run on 17 examples for the best config.

| Config | Opt MPJPE (2ex) | Opt MPJPE (17ex) | Opt Time |
|--------|----------------|-----------------|----------|
| 100 steps, LR=0.001 | 41.21 cm | 38.69 cm | 0.96s |
| 50 steps, LR=0.002 | 41.16 cm | 38.69 cm | 0.49s |
| 50 steps, LR=0.003 | 41.35 cm | -- | -- |
| 30 steps, LR=0.003 | 41.13 cm | -- | -- |
| 30 steps, LR=0.005 | 41.30 cm | -- | -- |

**50 steps at LR=0.002** achieves identical MPJPE to 100 steps at LR=0.001. The optimizer converges well before 100 steps. Since optimization is now <2% of total time, this is a minor absolute gain.

### Experiment 4: Remove Heatmap Blur

| Config | Opt MPJPE (2ex) |
|--------|----------------|
| blur=4.0 | 41.21 cm |
| blur=0.0 | 41.65 cm |

Removing blur degrades accuracy by 0.44 cm. **Keep blur at 4.0.**

## Unequivocal Improvements

1. **Batch FK and scoring (Experiments 1+2):** 64x optimization speedup with zero accuracy loss. This is a pure refactoring win -- identical mathematical operations, just vectorized.

2. **50 steps, LR=0.002 (Experiment 3):** Same accuracy, half the optimization steps. Negligible absolute time saving since optimization is already sub-second.

## Tradeoffs

None of the experiments involved accuracy tradeoffs. All improvements were either pure wins or rejected (blur removal).

## Final Recommended Configuration

```python
NUM_STEPS = 50
LEARNING_RATE = 0.002
BONE_LENGTH_LR = 0.0001
HEATMAP_BLUR_SIGMA = 4.0
# Use run_optimization_batched (default in main.py)
```

**End-to-end performance:**
- 17 examples, mean 148 frames each
- Detection: ~41s (98% of total)
- Optimization: ~0.5s (1% of total)
- Total latency/frame: 0.28s

## Future Optimization Opportunities

Since detection now accounts for 98% of total time, further latency reduction requires:
1. **Batch Stacked Hourglass inference** -- currently runs 1 frame at a time
2. **Skip YOLO** if person bbox is known/cached from previous frame
3. **Use lighter 2D detector** (e.g., RTMPose instead of Stacked Hourglass)
4. **GPU acceleration** for SH/MotionBERT (currently CPU-only on this machine)
