# Architect Plan: Phase 2b, Iteration 1 -- SZI-MPJPE Bug Fixes

## Goal Summary

The core SZI-MPJPE implementation from iteration 0 is mathematically correct per the spec. The tester found that SZI-MPJPE can exceed MPJPE on real data (Bug #1). This is not actually a bug -- the spec defines the objective as minimizing sum of squared magnitudes, and the closed-form `s = sum(pred . gt) / sum(pred . pred)` correctly minimizes that. However, MPJPE (mean of L2 norms) is a different objective than MSE, so the optimal-MSE scale does not guarantee MPJPE improvement. This iteration addresses the tester's bugs and adds documentation.

## Analysis of Bug #1: SZI-MPJPE > MPJPE

The spec says: "MPJPE = arg min of s ... sum(squared_magnitude(scale * pred - gt))". The squared_magnitude formulation means we minimize MSE, not mean-L2. The closed-form solution IS optimal for that objective.

Why SZI-MPJPE can exceed MPJPE: when error variance across joints is very high (e.g., legs at 195cm, arms at 10cm), the MSE-optimal scale is pulled toward reducing the large outlier errors (since they're squared). This can increase L2 errors for the majority of joints. The MPJPE we report after scaling is the mean of L2 norms, which is a different quantity than what was minimized.

This is mathematically correct and expected behavior. No code fix needed. We will:
1. Add a clarifying comment in the code.
2. Update the unit test that asserts SZI-MPJPE <= MPJPE (this property does NOT hold in general; remove or weaken the test).

## Changes Required

### Fix Bug #2: SZI-MPJPE not printed in test_single.py

**File**: `motionbert-pose/test_single.py`

**What to change**: In the `=== RESULTS ===` print block (around lines 142-161), add SZI-MPJPE print lines after the existing P-MPJPE lines, matching the pattern from `main.py`.

Add after line 148 (after the `Det P-MPJPE (no ankles)` line):
```python
if "det_szi_mpjpe" in metrics:
    print(f"  Det SZI-MPJPE: {metrics['det_szi_mpjpe']*100:.2f} cm (scale={metrics['det_szi_scale']:.4f})")
```

Add after line 154 (after the `Opt P-MPJPE (no ankles)` line):
```python
if "opt_szi_mpjpe" in metrics:
    print(f"  Opt SZI-MPJPE: {metrics['opt_szi_mpjpe']*100:.2f} cm (scale={metrics['opt_szi_scale']:.4f})")
```

Also add no-ankles variants:
```python
if "det_szi_mpjpe_no_ankles" in metrics:
    print(f"  Det SZI-MPJPE (no ankles): {metrics['det_szi_mpjpe_no_ankles']*100:.2f} cm")
if "opt_szi_mpjpe_no_ankles" in metrics:
    print(f"  Opt SZI-MPJPE (no ankles): {metrics['opt_szi_mpjpe_no_ankles']*100:.2f} cm")
```

### Fix unit test: Remove incorrect SZI-MPJPE <= MPJPE assertion

**File**: `motionbert-pose/test_szi_mpjpe.py`

**What to change**: The test "SZI-MPJPE <= MPJPE" (tests 9-13, the 5 random trials) asserts a property that does not hold in general. The MSE-optimal scale minimizes sum of squared errors, not mean of L2 norms. These are different objectives, and the MSE-optimal scale can increase MPJPE when error variance across joints is high.

Replace these 5 tests with a single test that verifies the correct property: **SZI-SSE <= SSE** (the sum of squared errors after optimal scaling is always less than or equal to the SSE without scaling, i.e., at scale=1.0). This is the property that the math actually guarantees.

The replacement test:
```python
# Test: Optimal scale minimizes SSE (this IS guaranteed by the math)
for trial in range(5):
    pred = np.random.randn(10, 16, 3)
    gt = np.random.randn(10, 16, 3)
    s = optimal_scale(pred, gt)
    sse_unscaled = np.sum((pred - gt) ** 2)
    sse_scaled = np.sum((s * pred - gt) ** 2)
    assert sse_scaled <= sse_unscaled + 1e-10, f"SSE after scaling should be <= SSE before scaling"
    # PASS
```

Keep the "SZI-MPJPE >= P-MPJPE" tests (tests 14-18) -- that property does hold because Procrustes has strictly more degrees of freedom.

### Add clarifying comment to evaluate.py

**File**: `motionbert-pose/evaluate.py`

**What to change**: Add a docstring note to `szi_mpjpe()` explaining the MSE vs MPJPE distinction:

In the `szi_mpjpe()` function docstring, add:
```
Note: The optimal scale minimizes sum of squared errors (MSE), not mean of
L2 norms (MPJPE). As a result, SZI-MPJPE is NOT guaranteed to be <= MPJPE.
When error variance across joints is high, the MSE-optimal scale can increase
MPJPE. This is mathematically correct per the spec definition.
```

## Files to Modify

1. **`motionbert-pose/test_single.py`** -- Add SZI-MPJPE print lines (Bug #2 fix)
2. **`motionbert-pose/test_szi_mpjpe.py`** -- Replace incorrect SZI <= MPJPE test with correct SSE property test
3. **`motionbert-pose/evaluate.py`** -- Add clarifying comment to `szi_mpjpe()` docstring

## Files to Create

None.

## Step-by-Step Instructions

1. Add SZI-MPJPE print lines to `test_single.py` as described above. Place them logically after the corresponding P-MPJPE lines for det and opt sections respectively.

2. In `test_szi_mpjpe.py`, find the loop that tests "SZI-MPJPE <= MPJPE" (should be 5 trials with random data). Replace it with a loop testing "SSE after scaling <= SSE before scaling" using the formula above. Update the test count if needed.

3. In `evaluate.py`, add the clarifying note to the `szi_mpjpe()` docstring.

4. Run `uv run python test_szi_mpjpe.py` to confirm all tests still pass.

5. Run `uv run python test_single.py` to confirm SZI-MPJPE is now printed to stdout.

## Integration Points

No new integration points. All changes are to existing files -- print statements, comments, and test corrections.

## Risks and Edge Cases

None. These are minor fixes with no risk of regression. The core metric implementation is unchanged.
