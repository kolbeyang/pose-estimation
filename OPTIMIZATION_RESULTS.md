# Pose Optimization Results

## Summary

**Target:** Score ≥ -16.00

**Achieved:** -13.55 in 100 steps ✅ (beats target by 2.45!)

## Winning Configuration

```python
num_steps: 100
learning_rate: 0.5
lr_min: 1e-5  # Cosine annealing
noise_temperature: 0.0  # No SGLD noise
position_penalty_weight: 0.2
ab_rotation_penalty_weight: 0.5
bc_rotation_penalty_weight: 0.3
```

## Hypothesis Testing Results

### ✅ Hypothesis 1: Noise scale too high - CONFIRMED
**Problem:** Original `noise_temperature=0.01` was causing poor performance

**Evidence:**
- noise=0.01: -158.05 (50 steps) - terrible!
- noise=1e-3: -66.69 (50 steps) - still bad
- noise=1e-5: -63.92 (50 steps) - better but not great
- noise=0.0: Works best with proper LR

**Solution:** Remove SGLD noise or use extremely tiny values (1e-6)

### ✅ Hypothesis 2: Position penalty too high - CONFIRMED
**Problem:** High position penalties prevent optimization from fitting the heatmaps

**Evidence (50 steps, noise=0.0, lr=0.1):**
- pos_penalty=0.6: -56.53
- pos_penalty=0.4: -186.83 (default was terrible!)
- pos_penalty=0.3: -50.96
- pos_penalty=0.2: -33.16 ⭐
- pos_penalty=0.1: -39.46

**Solution:** Use `position_penalty_weight=0.2`

### ✅ Hypothesis 3: Learning rate too low for fast convergence - CONFIRMED
**Problem:** LR=0.1 required 1000 steps to converge

**Evidence (100 steps, noise=0.0, pos=0.2):**
- lr=0.1: -26.41 (with slow annealing)
- lr=0.2: -15.06 (with tiny noise)
- lr=0.3: -14.83 ⭐
- lr=0.5: -13.55 ⭐⭐ BEST

**Solution:** Use `learning_rate=0.5` for 100-step convergence

### ✅ Hypothesis 4: Cosine annealing helps - CONFIRMED
**Evidence (100 steps, lr variants, pos=0.2):**
- Constant LR=0.1: -15.44
- Constant LR=0.2: -16.13
- Cosine (lr=0.1→1e-5): -14.42
- Cosine (lr=0.5→1e-5): -13.55 ⭐ BEST

**Solution:** Keep cosine annealing with `lr_min=1e-5`

## Performance Comparison

| Configuration | Steps | Score | Status |
|--------------|-------|-------|--------|
| **Optimal (fast)** | 100 | **-13.55** | ✅ Beats target by 2.45 |
| High LR variant | 100 | -14.42 | ✅ Beats target by 1.58 |
| Original (SGLD) | 1000 | -18.79 | ❌ Miss by 2.79 |
| Baseline (no SGLD) | 100 | -31.13 | ❌ Miss by 15.13 |

## Key Insights

1. **SGLD noise at 0.01 was the main culprit** - removing it immediately improved performance
2. **Position penalty was set too high** - reducing from 0.4 to 0.2 allows better heatmap fitting
3. **Higher learning rate enables fast convergence** - lr=0.5 converges in 100 steps vs 1000
4. **Cosine annealing still helps** - provides better final convergence than constant LR
5. **The optimization needs freedom to fit heatmaps** - lower penalties = better scores

## Recommendations

For **fast iteration** (100 steps):
- Use the winning config above
- Experiment with `position_penalty` in range [0.1, 0.3]
- Keep `noise_temperature=0.0` for deterministic results
- Use `lr=0.5` for aggressive optimization

For **production/best quality** (if compute budget allows):
- Can try 200-300 steps with same config for potential further improvement
- Consider `num_runs=5` for ensemble (currently set for single run during experiments)
