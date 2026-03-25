# Tester Report: Phase 2b, Iteration 0 -- Scale-Z-Invariant MPJPE

## Tests Run

### 1. Code Review: evaluate.py math correctness -- PASS

- `optimal_scale()` at line 194: formula `s = sum(pred * gt) / sum(pred * pred)` is correct per the derivation in the architect plan.
- Zero-denominator guard at `< 1e-12` returning 1.0 is correct.
- `szi_mpjpe()` correctly applies the global scale then delegates to `mpjpe()`.
- `szi_mpjpe_per_joint()` correctly computes scale globally, then per-joint error. Does not use per-joint scales.
- Integration in `compute_comparison()` (line 289-293) and `compute_comparison_with_optimization()` (line 375-379) is correct. Both include no-ankles variants.
- Type hints and docstrings are present and consistent with the existing codebase style.

### 2. Unit Tests: `uv run python test_szi_mpjpe.py` -- PASS (18/18)

```
PASS: Identity: scale == 1.0
PASS: Identity: SZI-MPJPE == 0.0
PASS: Known scale 2x: s == 0.5
PASS: Known scale 2x: SZI-MPJPE == 0.0
PASS: Noisy 2x: s ~= 0.5
PASS: Noisy 2x: SZI-MPJPE small but nonzero
PASS: Asymmetric 0.5x: s == 2.0
PASS: Asymmetric 0.5x: SZI-MPJPE == 0.0
PASS: SZI-MPJPE <= MPJPE (trials 0-4)
PASS: SZI-MPJPE >= P-MPJPE (trials 0-4)
Results: 18 passed, 0 failed out of 18
```

Synthetic visualization: scale 0.667 for 1.5x input (correct).
Real example visualization: scale 0.081 (loaded from `motionbert-run-20260317-204113`).

### 3. 3D Visualization -- PASS

Both PNG files exist and are visually correct:
- `motionbert-pose/test_output/szi_mpjpe_visualization.png` -- synthetic, three skeletons side by side. Red (scaled) and blue (GT) overlap well.
- `motionbert-pose/test_output/szi_mpjpe_real_example.png` -- real example, three skeletons from non-camera viewpoint.

### 4. Smoke Test: `uv run python test_single.py` -- PASS (with caveat)

Pipeline completed without errors. All SZI-MPJPE keys are present in `diagnostics.json`:
- `det_szi_mpjpe`: 0.5387 (53.87 cm)
- `det_szi_scale`: 0.0812
- `opt_szi_mpjpe`: 0.5271 (52.71 cm)
- `opt_szi_scale`: 0.1402
- `det_szi_mpjpe_no_ankles`: 0.3020
- `opt_szi_mpjpe_no_ankles`: 0.1405

**Caveat**: SZI-MPJPE is NOT printed to stdout during the test_single run. See Bug #2.

### 5. Metric Ordering -- FAIL on real data

- **Unit tests (random data)**: SZI-MPJPE <= MPJPE holds for all 5 trials. PASS.
- **Real pipeline data**: SZI-MPJPE > MPJPE. FAIL.
  - Det: SZI-MPJPE 53.87 cm > MPJPE 50.79 cm
  - Opt: SZI-MPJPE 52.71 cm > MPJPE 44.99 cm
  - However, P-MPJPE < SZI-MPJPE holds correctly (41.80 < 53.87 and 40.85 < 52.71).

See Bug #1 for analysis.

### 6. Regression -- PASS

All existing metrics (MPJPE, P-MPJPE, MPJVE, per-joint, no-ankles) continue to appear correctly.
`graphs.py` contains the new SZI-MPJPE bar chart code at lines 705-729 following the same pattern as P-MPJPE.

## Bugs Found

### Bug #1: SZI-MPJPE can exceed MPJPE on real data (Severity: MEDIUM)

**Description**: The `optimal_scale` formula minimizes sum-of-squared-errors (SSE), not MPJPE (mean of L2 norms). These are different objectives. On the real test_single example, the optimal scale (0.08) reduces SSE from 1040 to 449 but INCREASES MPJPE from 50.79 to 53.87 cm. This happens because the scale factor heavily helps outlier joints (legs at 195+ cm error) in squared terms while hurting the majority of joints in L2-norm terms.

**How observed**: Checked metric ordering from diagnostics.json. Also confirmed manually: SSE improves (correct) but MPJPE does not (unexpected).

**Impact**: The SZI-MPJPE metric is not strictly a lower bound on MPJPE as the spec and unit tests assume. The unit test for `SZI-MPJPE <= MPJPE` passes only on random data where error variance is low, making it a weak test.

**Suggested fix direction**: Two options:
1. Accept this behavior and document it. The metric still has value -- it optimizes a well-defined objective (MSE) and the scale is still meaningful.
2. If the property `SZI-MPJPE <= MPJPE` is important, change the scale computation to directly minimize MPJPE (mean of L2 norms) via line search or iterative optimization, rather than the closed-form MSE solution.

### Bug #2: SZI-MPJPE not printed in test_single.py output (Severity: LOW)

**Description**: `test_single.py` has its own print block (lines 142-161) that was not updated to include SZI-MPJPE print lines. The developer only updated `main.py` (lines 232-240). The metrics ARE computed and stored in diagnostics.json, just not printed to stdout.

**How observed**: Ran `test_single.py` and searched output for "SZI" -- not found.

**Suggested fix**: Add SZI-MPJPE print lines to `test_single.py` matching the pattern in `main.py`.

### Bug #3: Per-joint SZI-MPJPE not saved to diagnostics.json (Severity: LOW)

**Description**: `det_szi_per_joint` and `opt_szi_per_joint` are computed (they are lists) but filtered out by `test_single.py` line 207 which only saves scalar values: `{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}`.

**How observed**: Checked diagnostics.json keys. Lists like `det_szi_per_joint` are absent.

**Note**: This is pre-existing behavior that also affects other per-joint metrics. Not introduced by this phase.

## Code Review Findings

1. **Code quality is good.** Functions have type hints, docstrings, and consistent naming. The new functions follow existing patterns exactly.
2. **Edge case handling**: The zero-denominator guard in `optimal_scale` is appropriate. Negative scale is allowed as documented in the architect plan.
3. **No off-by-one or array shape issues** found. The eval-joint slicing is consistent with existing metrics.

## Verdict

**NO** -- the spec's definition of done is not fully met.

The core implementation is correct and complete: the math is right, unit tests pass, visualizations work, and the metric is integrated into the pipeline. However, Bug #1 (SZI-MPJPE > MPJPE on real data) indicates that the metric's theoretical property (SZI-MPJPE <= MPJPE) does not hold in practice for this data. The spec says "Make sure this algorithm works and makes sense" -- the algorithm works mathematically, but the MSE-vs-MPJPE discrepancy should be acknowledged.

Bug #2 (not printed in test_single.py) is a minor omission that is easy to fix.

Recommended next steps:
1. Fix Bug #2 (add SZI-MPJPE print lines to test_single.py).
2. Discuss Bug #1 with the architect/spec owner. If the MSE-optimal scale is acceptable, document it. If MPJPE-optimal scale is needed, implement a simple 1D line search.
