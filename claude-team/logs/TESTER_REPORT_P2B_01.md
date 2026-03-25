# Tester Report: Phase 2b, Iteration 1 -- SZI-MPJPE Bug Fixes

## Tests Run

### 1. Unit Tests: `uv run python test_szi_mpjpe.py` -- PASS (18/18)

```
PASS: Identity: scale == 1.0
PASS: Identity: SZI-MPJPE == 0.0
PASS: Known scale 2x: s == 0.5
PASS: Known scale 2x: SZI-MPJPE == 0.0
PASS: Noisy 2x: s ~= 0.5
PASS: Noisy 2x: SZI-MPJPE small but nonzero
PASS: Asymmetric 0.5x: s == 2.0
PASS: Asymmetric 0.5x: SZI-MPJPE == 0.0
PASS: SSE after scaling <= SSE before scaling (trial 0)
PASS: SSE after scaling <= SSE before scaling (trial 1)
PASS: SSE after scaling <= SSE before scaling (trial 2)
PASS: SSE after scaling <= SSE before scaling (trial 3)
PASS: SSE after scaling <= SSE before scaling (trial 4)
PASS: SZI-MPJPE >= P-MPJPE (trial 0)
PASS: SZI-MPJPE >= P-MPJPE (trial 1)
PASS: SZI-MPJPE >= P-MPJPE (trial 2)
PASS: SZI-MPJPE >= P-MPJPE (trial 3)
PASS: SZI-MPJPE >= P-MPJPE (trial 4)
Results: 18 passed, 0 failed out of 18
```

The new SSE property tests (trials 0-4) correctly verify the mathematical guarantee: SSE after optimal scaling is always <= SSE before scaling. This replaces the old incorrect "SZI-MPJPE <= MPJPE" assertion.

### 2. Smoke Test: `uv run python test_single.py` -- PASS

Pipeline completed without errors. SZI-MPJPE values now appear in stdout (Bug #2 from iteration 0 is fixed):

```
Det SZI-MPJPE: 53.87 cm (scale=0.0812)
Det SZI-MPJPE (no ankles): 30.20 cm
Opt SZI-MPJPE: 52.71 cm (scale=0.1402)
Opt SZI-MPJPE (no ankles): 14.05 cm
```

All four SZI-MPJPE print lines (det, det no-ankles, opt, opt no-ankles) are present.

### 3. Code Review: 3 Fixes Verified -- PASS

**Fix 1: Clarifying docstring in evaluate.py (line 218-221)**
Confirmed. The `szi_mpjpe()` function now has a clear note:
> "Note: The optimal scale minimizes sum of squared errors (MSE), not mean of L2 norms (MPJPE). As a result, SZI-MPJPE is NOT guaranteed to be <= MPJPE."

This accurately documents the behavior observed in iteration 0's Bug #1.

**Fix 2: Unit test corrected in test_szi_mpjpe.py (lines 124-135)**
Confirmed. The old "SZI-MPJPE <= MPJPE" loop (which tested an incorrect property) has been replaced with "SSE after scaling <= SSE before scaling" (which tests the correct mathematical guarantee). Test count remains at 18.

**Fix 3: SZI-MPJPE prints added to test_single.py (lines 146-162)**
Confirmed. Four conditional print blocks added for `det_szi_mpjpe`, `det_szi_mpjpe_no_ankles`, `opt_szi_mpjpe`, and `opt_szi_mpjpe_no_ankles`, following the same pattern as existing metric prints.

### 4. Regression Check -- PASS

All existing metrics continue to appear correctly in the test_single.py output:
- MPJPE: Det 50.79 cm, Opt 44.99 cm
- P-MPJPE: Det 41.80 cm, Opt 40.85 cm
- No-ankles variants present
- MPJVE, per-joint, bone lengths, coordinate ranges all present
- Overlay video generated successfully

## Bugs Found

None.

## Code Review Findings

No issues. All three changes are minimal, correct, and consistent with the existing codebase style.

## Verdict

**YES** -- the iteration 1 fixes are complete and correct.

All three bugs/issues from the iteration 0 tester report have been addressed:
- Bug #1 (SZI-MPJPE > MPJPE): Correctly identified as not-a-bug by the architect; documented in code.
- Bug #2 (SZI-MPJPE not printed): Fixed; all four SZI-MPJPE values now appear in stdout.
- Unit test: Corrected to test the actual mathematical property (SSE reduction) rather than an incorrect property (MPJPE reduction).

The Phase 2b SZI-MPJPE implementation is complete and meets the spec's definition of done.
