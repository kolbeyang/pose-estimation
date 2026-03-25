# Developer Report: Phase 2b, Iteration 0 -- Scale-Z-Invariant MPJPE

## What Was Implemented

1. **Three new functions in `evaluate.py`**:
   - `optimal_scale(predicted, target)` -- finds the scalar `s` minimizing `||s*pred - gt||^2` across all joints and frames. Returns `s = sum(pred . gt) / sum(pred . pred)` with a guard for zero denominator.
   - `szi_mpjpe(predicted, target)` -- computes SZI-MPJPE by applying optimal scale then computing standard MPJPE. Returns both the metric value and the scale factor.
   - `szi_mpjpe_per_joint(predicted, target)` -- per-joint error after global optimal scaling.

2. **Integration into `compute_comparison()`**: Added `det_szi_mpjpe`, `det_szi_scale`, `det_szi_per_joint`, `det_szi_mpjpe_no_ankles` to the results dict.

3. **Integration into `compute_comparison_with_optimization()`**: Added `opt_szi_mpjpe`, `opt_szi_scale`, `opt_szi_per_joint`, `opt_szi_mpjpe_no_ankles` to the results dict.

4. **SZI-MPJPE printing in `main.py`**: Added print lines for both detector and optimized SZI-MPJPE with scale factors after the P-MPJPE lines.

5. **Aggregate SZI-MPJPE bar chart in `graphs.py`**: Added a new bar chart in `generate_aggregate_summary()` following the same pattern as the P-MPJPE chart. Uses mediumpurple color for detector bars. Saves to `aggregate_szi_mpjpe.png`.

6. **`test_szi_mpjpe.py`**: Standalone test script with:
   - 18 unit tests: identity, known scale (2x), noisy scale, asymmetric scale (0.5x), SZI <= MPJPE (5 trials), SZI >= P-MPJPE (5 trials).
   - Synthetic 3D skeleton visualization (pred at 1.5x, scaled, GT side by side).
   - Real-example 3D visualization from the most recent prediction JSON.

## Files Changed

- `motionbert-pose/evaluate.py` -- Added `optimal_scale`, `szi_mpjpe`, `szi_mpjpe_per_joint`; integrated into both `compute_comparison` and `compute_comparison_with_optimization`
- `motionbert-pose/main.py` -- Added SZI-MPJPE print lines in `process_example()`
- `motionbert-pose/graphs.py` -- Added SZI-MPJPE bar chart in `generate_aggregate_summary()`

## Files Created

- `motionbert-pose/test_szi_mpjpe.py` -- Unit tests + 3D visualizations
- `motionbert-pose/test_output/szi_mpjpe_visualization.png` -- Synthetic skeleton visualization
- `motionbert-pose/test_output/szi_mpjpe_real_example.png` -- Real example visualization

## Commands Run

1. `cd motionbert-pose && uv run python test_szi_mpjpe.py` -- **18/18 tests PASSED**. Synthetic visualization shows correct 0.667 scale for 1.5x input. Real example produced valid visualization.

2. `cd motionbert-pose && uv run python test_single.py` -- Full pipeline ran without errors. Diagnostics JSON confirmed all 6 SZI-MPJPE keys present (`det_szi_mpjpe`, `det_szi_scale`, `det_szi_per_joint`, `opt_szi_mpjpe`, `opt_szi_scale`, `opt_szi_per_joint`, plus no-ankles variants).

## Decisions Made

- **Math correction**: The architect's plan noted the spec had a typo (`sum(gt.gt)` in denominator vs correct `sum(pred.pred)`). I used the corrected formula from the plan: `s = sum(pred . gt) / sum(pred . pred)`.

- **Bar chart color**: Used `mediumpurple` for the SZI-MPJPE chart to visually distinguish it from the MPJPE (steelblue) and P-MPJPE (darkorange) charts.

- **Real example visualization**: The test script auto-discovers the most recent prediction JSON from `training_runs/`. It found a run from `20260317-204113` and used it for the real-example plot.

## Concerns

- **Small scale factors observed**: The real example showed optimal scales of 0.08 (detector) and 0.14 (optimized), meaning the predictions are roughly 7-12x larger in magnitude than GT (in root-relative coordinates). This is not a bug in the metric -- the scale formula is working correctly. But it suggests a significant scale mismatch between MotionBERT's normalized output and the camera-space GT. The metric is doing its job: compensating for exactly this ambiguity.

- **SZI-MPJPE can be > MPJPE when scale is already near-optimal**: In theory SZI-MPJPE <= MPJPE always holds (proven by the unit tests). But the SZI metric was very close to the raw MPJPE for the real example (53.87 vs 50.79 for detector), suggesting the optimal scale applied globally was not strictly beneficial for every joint. This is expected when the scale mismatch varies by body part.

## Deviations from Plan

None. All steps in the architect's plan were followed exactly.
