# Developer Report P1-01: Heatmap Blur Sweep (Phase 1.2)

## What Was Implemented

1. **Added `HEATMAP_BLUR_SIGMA` config parameter** in `config.py` (default 0.0, no blur).

2. **Added coarse-to-fine heatmap blur schedule support** in `optimize.py`:
   - `_get_blur_sigma()`: returns blur sigma for a given step from a schedule (same `(fraction, sigma)` format as `SIGMA_SCHEDULE`).
   - `_apply_blur_torch()`: applies scipy Gaussian blur to torch heatmap tensors (converts to numpy, blurs, converts back). Called only at phase transitions, not per-step.
   - `run_optimization()` gains optional `heatmap_blur_schedule` parameter. When set, stores original unblurred heatmaps and re-blurs at phase transition points.
   - Blur sigma printed in step logging output.

3. **Updated `sweep.py`**:
   - `SweepConfig` gains `heatmap_blur_schedule` field (default None).
   - Fixed pre-blur in `run_sweep_config()` is skipped when a dynamic schedule is set (prevents double-blur).
   - `heatmap_blur_schedule` passed through to `run_optimization()`.
   - Added `get_phase1_2_configs()` with 17 configurations: 3 baselines + 4 fixed-blur-50s + 4 fixed-blur-100s + 4 c2f-100s + 2 c2f-50s.
   - Added `--phase` CLI argument (choices: "1.1", "1.2", default "1.2") via argparse.
   - MPJVE column added to console summary table (tester recommendation CR-2).
   - Blur info printed in per-config header.
   - JSON output includes `num_steps`, `heatmap_blur_sigma`, `heatmap_blur_schedule`.
   - Results tuple now stores `SweepConfig` alongside metrics for JSON serialization.

## Files Changed

- **Modified:** `motionbert-pose/config.py` -- Added `HEATMAP_BLUR_SIGMA` parameter
- **Modified:** `motionbert-pose/optimize.py` -- Added `_get_blur_sigma()`, `_apply_blur_torch()`, blur schedule support in `run_optimization()`
- **Modified:** `motionbert-pose/sweep.py` -- Added `get_phase1_2_configs()`, argparse, MPJVE in table, blur schedule passthrough, JSON enrichment

## Commands Run

```bash
cd motionbert-pose && uv run python -c "from sweep import get_phase1_2_configs; configs = get_phase1_2_configs(); print(f'{len(configs)} configs')"
# Result: 17 configs -- correct

cd motionbert-pose && uv run python sweep.py --phase 1.2
# Result: All 17 configs completed successfully in ~6 minutes total
# Saved: motionbert-pose/training_runs/sweep_results/sweep_171204_pose1_sample_0.json
```

## Sweep Results (Example 0: 171204_pose1_sample_0, 100 frames)

| Config | Opt MPJPE (cm) | Improv (cm) | Opt P-MPJPE (cm) | Opt MPJVE (cm/f) | Opt 2D-Det (px) |
|--------|---------------|-------------|-------------------|------------------|-----------------|
| no_blur_20s | 30.44 | +0.54 | 28.42 | 1.05 | 33.8 |
| no_blur_50s | 30.74 | +0.24 | 28.37 | 1.17 | 30.9 |
| no_blur_100s | 31.10 | -0.12 | 28.30 | 1.36 | 28.2 |
| blur1_50s | 30.71 | +0.27 | 28.36 | 1.17 | 30.8 |
| blur2_50s | 30.56 | +0.42 | 28.33 | 1.26 | 31.5 |
| blur4_50s | 30.34 | +0.64 | 28.29 | 1.46 | 32.2 |
| blur8_50s | 30.22 | +0.76 | 28.29 | 1.56 | 33.0 |
| blur1_100s | 30.82 | +0.16 | 28.24 | 1.37 | 28.5 |
| blur2_100s | 30.43 | +0.55 | 28.19 | 1.48 | 29.4 |
| blur4_100s | 30.06 | +0.92 | 28.14 | 1.62 | 30.2 |
| blur8_100s | 30.00 | +0.98 | 28.13 | 1.66 | 31.4 |
| c2f_8to0_100s | 30.40 | +0.58 | 28.13 | 1.69 | 26.3 |
| c2f_4to0_100s | 30.58 | +0.40 | 28.16 | 1.61 | 26.0 |
| c2f_4to1_100s | 30.44 | +0.54 | 28.14 | 1.57 | 26.6 |
| c2f_8to2_100s | 30.02 | +0.96 | 28.07 | 1.59 | 27.5 |
| c2f_4to0_50s | 30.51 | +0.47 | 28.28 | 1.39 | 30.1 |
| c2f_8to0_50s | 30.26 | +0.72 | 28.28 | 1.55 | 31.2 |

### Key Findings

1. **Blur substantially improves 3D MPJPE**. Best fixed blur: blur8_100s at 30.00 cm (+0.98 cm improvement over detector). Without blur, 100 steps makes things *worse* (31.10 cm, -0.12 cm). With sigma=8 blur, 100 steps achieves the best MPJPE in the sweep.

2. **Higher blur = better MPJPE but worse MPJVE**. There is a clear monotonic trend: more blur -> lower MPJPE but higher velocity error. blur8_100s achieves 30.00 cm MPJPE but 1.66 cm/f MPJVE vs baseline 1.05 cm/f. The optimizer preserves depth better but introduces temporal jitter.

3. **Coarse-to-fine schedules get the best 2D alignment AND good 3D**. c2f_8to2_100s achieves nearly the best MPJPE (30.02 cm) while also getting good 2D fit (27.5 px) and reasonable MPJVE (1.59). c2f_4to0_100s gets the tightest 2D alignment (26.0 px) with moderate MPJPE improvement (30.58 cm).

4. **c2f_8to2_100s is the overall winner**: 30.02 cm MPJPE (+0.96 cm improvement), 28.07 cm P-MPJPE (best), 27.5 px 2D-Det, 1.59 cm/f MPJVE. It balances all metrics well.

5. **P-MPJPE consistently improves with blur/steps**. Every blur and step-count configuration improves P-MPJPE over the 20-step baseline (28.42 cm). Best: c2f_8to2_100s at 28.07 cm. This suggests blur helps find better pose shapes.

6. **50-step blur is a good middle ground**. blur8_50s achieves 30.22 cm MPJPE in half the time of 100-step configs, with lower MPJVE (1.56 vs 1.66 for blur8_100s).

## Decisions Made

- The architect's plan showed `blur=0.0` in the step logging for fixed-blur configs. This is correct -- the `current_blur_sigma` variable tracks the *dynamic schedule* blur, not the pre-applied fixed blur. The fixed blur is applied once before optimization in `run_sweep_config()`, so `current_blur_sigma` stays at 0.0. This is consistent with the plan's intent.

## Concerns

1. **MPJVE degrades with all blur approaches**. Even the best configs have MPJVE 1.5-1.7 cm/f vs baseline 1.05 cm/f. The `ALL_JOINTS_SMOOTH_WEIGHT` parameter (currently 0.0) was not swept here and could potentially address this.

2. **Single example only**. These results are from example 0 only. The winning configs should be validated on multiple examples before drawing final conclusions.

## Deviations from Plan

None. All steps implemented as specified.
