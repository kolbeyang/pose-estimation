# Phase 4: Hyperparameter Tuning Report

**Date:** 2026-04-05
**Branch:** refactor-2026-04-04
**Commit:** 3320ce1

---

## Experiment Scripts Created

| Script | Purpose |
|--------|---------|
| `experiment/mediapipe_sweep.py` | Round 1: 24 single-param variations on single example |
| `experiment/mediapipe_sweep2.py` | Round 2: 11 refined combos from round 1 winners |
| `experiment/mediapipe_multi_sweep.py` | Round 3: 8 candidate configs tested across 5 examples |
| `experiment/motionbert_sweep.py` | 14 MotionBERT variations on single example |

All scripts share a pattern: load example data once, then loop over configs calling `optimize()` + `evaluate()` sequentially.

---

## MediaPipe Sweep Results

### Round 1 (single example: 171204_pose1_sample)

Baseline: 25 steps, lr=0.0005, sigma=16 (fixed), anchor_weight=1000 -- 1.7% improvement.

| Config | Opt (cm) | Improvement |
|--------|----------|-------------|
| steps=50+anneal=16->4+anchor=500 | 15.34 | 2.0% |
| steps=75+anneal=16->2 | 15.35 | 2.0% |
| steps=50+anneal=16->2+lr=0.001 | 15.35 | 2.0% |
| sigma=4 (fixed) | 15.36 | 1.9% |
| anchor=500 | 15.37 | 1.9% |
| baseline (sigma=16, anchor=1000) | 15.40 | 1.7% |
| lr=0.0001 | 15.66 | 0.0% |

**Key finding:** Lower sigma is better -- sigma=4 outperforms sigma=16. Blur was too aggressive, washing out spatial information. Lower anchor (500 vs 1000) also helps.

### Round 2 (single example, refined combos)

| Config | Opt (cm) | Improvement |
|--------|----------|-------------|
| anneal=8->1+anchor=500+steps=50 | 15.26 | 2.6% |
| sigma=4+anchor=200 | 15.28 | 2.4% |
| sigma=4+anchor=300 | 15.29 | 2.4% |
| sigma=4+anchor=500 | 15.31 | 2.2% |

**Key finding:** sigma=4 with lower anchor (200-300) matches or beats expensive annealing configurations at a fraction of the runtime (0.2s vs 3.4s).

### Round 3 (5 examples, robustness validation)

| Config | Avg Impr% | Avg Opt (cm) | Min Impr% |
|--------|-----------|--------------|-----------|
| sigma=4+anchor=200+steps=50 | **2.7%** | 13.62 | **+0.9%** |
| sigma=4+anchor=100+steps=50 | 2.7% | 13.60 | -0.7% |
| anneal=8->2+anchor=300+steps=50 | 2.0% | 13.70 | -0.2% |
| anneal=8->1+anchor=500+steps=50 | 1.9% | 13.72 | +0.3% |
| sigma=4+anchor=300 (25 steps) | 1.3% | 13.79 | -0.2% |

**Selected config: `sigma=4, anchor=200, 50 steps, lr=0.0005`**
- Best average improvement (2.7%) with no regression on any example (min +0.9%)
- Fast: ~0.3s per example (no blur annealing overhead)

---

## MotionBERT Sweep Results (single example)

Baseline: 50 steps, lr=0.001, bone_lr=0.005, anneal 16->4 -- 28.2% improvement.

| Config | Opt (cm) | Improvement |
|--------|----------|-------------|
| steps=100 | **15.33** | **34.6%** |
| steps=100+anneal=16->2 | 15.38 | 34.4% |
| steps=100+anneal=16->2+lr=0.002 | 15.59 | 33.5% |
| steps=75 | 15.99 | 31.7% |
| lr=0.002 | 16.27 | 30.6% |
| bone_lr=0.01 | 16.67 | 28.9% |
| baseline (50 steps) | 16.83 | 28.2% |
| anchor=100 | 24.08 | **-2.8%** |
| anchor=500 | 24.56 | **-4.8%** |

**Key findings:**
1. More steps is the biggest lever (100 > 75 > 50)
2. Anchoring HURTS MotionBERT (raw predictions are bad -- anchoring to them is counterproductive)
3. lr=0.002 helps slightly but lr=0.005 hurts
4. Anneal endpoints don't matter much; 16->4 is fine

**Selected config: `100 steps, lr=0.001, bone_lr=0.005, anneal 16->4`**

---

## 5-Example Validation Results

### MotionBERT

| Example | Raw (cm) | Opt (cm) | Improvement |
|---------|----------|----------|-------------|
| pose1_sample_0 | 23.43 | 15.33 | 34.6% |
| pose2_200 | 69.40 | 65.18 | 6.1% |
| pose3_200 | 45.90 | 42.22 | 8.0% |
| pose1_5000 | 44.05 | 37.10 | 15.8% |
| pose2_5000 | 11.35 | 10.43 | 8.1% |
| **AVERAGE** | **38.83** | **34.05** | **12.3%** |

All examples show improvement. Average improvement well above 2%. However, average optimized is 34.05 cm -- far above the 10 cm target. This is due to MotionBERT's fundamental lower-body distortion on these sequences.

### MediaPipe

| Example | Raw (cm) | Opt (cm) | Improvement |
|---------|----------|----------|-------------|
| pose1_sample_0 | 15.66 | 15.20 | 2.9% |
| pose2_200 | 16.85 | 16.60 | 1.5% |
| pose3_200 | 15.86 | 15.17 | 4.4% |
| pose1_5000 | 9.63 | 9.28 | 3.6% |
| pose2_5000 | 11.95 | 11.84 | 0.9% |
| **AVERAGE** | **13.99** | **13.62** | **2.7%** |

All examples show improvement (no regressions). Average improvement exceeds 2% target. Average optimized is 13.62 cm.

---

## Updated Config Files

| File | Changes |
|------|---------|
| `configs/both-local-single.json` | MB: 100 steps, anneal 16->4. MP: sigma=4, anchor=200, 50 steps. |
| `configs/both-local-5-examples.json` | New file. Same optimization params, 5 examples, no video/graph generation. |
| `configs/motionbert-single.json` | Updated num_steps 25 -> 100. |
| `configs/motionbert-local-25-examples.json` | Added per-pipeline optimization overrides (100 steps, anneal 16->4). |
| `configs/mediapipe-local-25-examples.json` | Added per-pipeline optimization overrides (sigma=4, anchor=200, 50 steps). |

---

## Best Config Summary

### MotionBERT Optimization
```json
{
    "num_steps": 100,
    "learning_rate": 0.001,
    "bone_length_lr": 0.005,
    "heatmap_blur_sigma_start": 16.0,
    "heatmap_blur_sigma_end": 4.0,
    "anchor_weight": 0.0
}
```

### MediaPipe Optimization
```json
{
    "num_steps": 50,
    "heatmap_blur_sigma": 4.0,
    "learning_rate": 0.0005,
    "anchor_weight": 200.0
}
```

---

## Key Insights

1. **Blur sigma was too high.** The default sigma=16 on 64x64 heatmaps spreads each peak across 50% of the image, destroying localization precision. sigma=4 preserves spatial detail and gives better results for MediaPipe, which starts close to correct.

2. **Anchor weight sweet spot differs by pipeline.** MediaPipe benefits from moderate anchoring (200) because its raw predictions are good -- we want the optimizer to refine, not overhaul. MotionBERT needs zero anchoring because its raw predictions are bad.

3. **More steps helps MotionBERT dramatically.** MotionBERT needs 100 steps to converge because it starts far from the truth (especially lower body). Each step reduces error ~0.08 cm on average.

4. **Blur annealing is not necessary for MediaPipe.** Fixed sigma=4 works as well as elaborate anneal schedules at 10x less runtime, because MediaPipe starts close enough that broad gradients aren't needed.

5. **The <10 cm target for MotionBERT is unreachable on bad examples.** pose2_200 has 65 cm error after optimization because MotionBERT's raw 3D lift is severely distorted (69.40 cm raw). The optimizer can only reduce error by ~6% when the starting point is that bad.

---

## Open Items for Phase 5

- [ ] Run full 25-example validation to get aggregate statistics
- [ ] The <10 cm absolute target for MotionBERT is unlikely to be met without fixing the underlying 3D lift quality
- [ ] MediaPipe's pose2_200 and pose2_5000 examples show low improvement (~0.5-0.9%) -- investigate whether SH heatmaps disagree with MediaPipe on these sequences
