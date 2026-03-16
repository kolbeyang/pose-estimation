# DEVELOPER_REPORT_P1_00: Switch to Real Stacked Hourglass Heatmaps

## What Was Implemented

Replaced the analytical Gaussian heatmap scoring in the FK optimization with real Stacked Hourglass heatmap sampling. The optimizer now uses `torch.nn.functional.grid_sample` to perform differentiable bilinear interpolation on the actual (16, 64, 64) heatmaps produced by Stacked Hourglass, rather than constructing synthetic Gaussian blobs centered on extracted keypoint positions.

For the 2 H36M joints that have no direct MPII heatmap (Hip=0 as midpoint of RHip+LHip, Spine=7 as midpoint of Pelvis+Thorax), the code falls back to the existing analytical Gaussian scoring.

A `USE_REAL_HEATMAPS` config flag enables A/B testing. The old analytical Gaussian path is fully preserved as fallback.

## Files Changed

1. **`motionbert-pose/scoring.py`** -- Added `H36M_TO_MPII_HEATMAP` constant (maps each H36M joint index to its MPII heatmap index or None), added `real_heatmap_score()` function, updated `compute_total_score()` with 3 new optional parameters (`heatmaps_list`, `affine`, `use_real_heatmaps`).

2. **`motionbert-pose/optimize.py`** -- Added `heatmaps` and `affine` parameters to `run_optimization()`. Converts heatmaps to torch tensors once at initialization (not per step). Passes them through to `compute_total_score`.

3. **`motionbert-pose/main.py`** -- Threads heatmaps and affine from `detect_poses()` through to `run_optimization()`.

4. **`motionbert-pose/test_single.py`** -- Same threading of heatmaps/affine to `run_optimization()`.

5. **`motionbert-pose/config.py`** -- Added `USE_REAL_HEATMAPS: bool = True`.

## Commands Run

```
cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python test_single.py
```

**Result:** Ran to completion successfully. Key output:
- Loss decreased from 1054.9 to 813.5 over 20 steps (optimizer is converging)
- Log confirms: "Using real Stacked Hourglass heatmaps for scoring"
- Det MPJPE: 30.98 cm, Opt MPJPE: 30.44 cm (improvement: +0.54 cm)
- Det MPJPE (no ankles): 14.14 cm, Opt MPJPE (no ankles): 13.85 cm
- No errors or warnings

## Decisions Made

1. **Affine inversion approach:** The affine transform maps from 256-crop coords to original pixel coords (x_orig = sx * x_256 + tx). To go the other direction (original -> 256-crop), I invert it as x_256 = (x_orig - tx) / sx. This is a simple division since the affine is axis-aligned (no rotation component in the crop transform).

2. **grid_sample coordinate normalization:** Used `align_corners=True` with normalization formula `grid_x = x_64 / 63.0 * 2.0 - 1.0` so that pixel 0 maps to -1 and pixel 63 maps to +1, which is the standard convention when align_corners=True.

3. **Per-joint loop vs batched:** Implemented the heatmap sampling as a per-joint loop rather than batching across joints. This is because each joint samples from a different heatmap channel, and the fallback logic for Hip/Spine joints differs. With only 17 joints, the per-joint overhead is negligible compared to the FK computation.

4. **Epsilon for log:** Used `eps=1e-8` as the floor for `torch.log(torch.clamp(value, min=eps))` to avoid log(0) = -inf.

## Concerns

1. **Small improvement magnitude:** The optimization only improved MPJPE by 0.54 cm in 20 steps. This could improve with more steps or hyperparameter tuning. The heatmap scores are in a very different range than the analytical Gaussian scores (log of heatmap values vs scaled squared distance), so the balance between heatmap score, position penalty, rotation penalty, and anchor penalty may need retuning.

2. **Ankle errors dominate:** Ankles have ~115 cm MPJPE, dwarfing all other joints. The Stacked Hourglass heatmaps for ankles often have very low confidence. The real heatmaps should be more informative here (the optimizer can see that the heatmap is diffuse/flat rather than peaked), but the depth ambiguity for ankles remains fundamental.

3. **Heatmap value range:** Stacked Hourglass heatmaps are not normalized to [0, 1] -- their peak values depend on the network's output activation. If peak values are much less than 1.0, the log scores will be very negative, potentially overwhelming other loss terms. If peak values are much greater than 1.0, log scores will be positive. The current implementation handles both cases correctly but the relative weighting of heatmap score vs penalties may need adjustment.

## Deviations from Plan

None. All 5 steps of the architect's plan were implemented as specified.
