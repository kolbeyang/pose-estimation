# Tester Plan: Phase 2b, Iteration 0 -- Scale-Z-Invariant MPJPE

## What to Test

### 1. Code Review: Math correctness in evaluate.py
- Verify `optimal_scale` formula: `s = sum(pred * gt) / sum(pred * pred)`
- Check zero-denominator guard
- Check `szi_mpjpe` applies scale then delegates to existing `mpjpe`
- Check `szi_mpjpe_per_joint` uses global scale (not per-joint scale)
- Check integration in `compute_comparison()` and `compute_comparison_with_optimization()`

### 2. Unit Tests
- Run `test_szi_mpjpe.py` and verify all 18 tests pass
- Review test coverage: identity, known scale, noisy scale, asymmetric scale, ordering invariants

### 3. 3D Visualization
- Verify synthetic and real-example PNGs exist in `test_output/`
- Visually inspect the synthetic visualization

### 4. Smoke Test: Pipeline Integration
- Run `test_single.py` and verify SZI-MPJPE appears in output
- Verify both detector and optimized SZI-MPJPE are printed with scale factors

### 5. Metric Ordering
- Verify from unit tests or pipeline output: SZI-MPJPE >= P-MPJPE and SZI-MPJPE <= MPJPE (generally)

### 6. Regression Checks
- Existing metrics (MPJPE, P-MPJPE, MPJVE) should still be present and unchanged
- graphs.py should still generate all existing charts plus the new SZI-MPJPE chart

## How to Test

| Test | Command/Check |
|------|---------------|
| Code review | Read evaluate.py, verify math |
| Unit tests | `cd motionbert-pose && uv run python test_szi_mpjpe.py` |
| Visualization | Check test_output/ for PNGs |
| Smoke test | `cd motionbert-pose && uv run python test_single.py` |
| Ordering | Check unit test output + pipeline output |
