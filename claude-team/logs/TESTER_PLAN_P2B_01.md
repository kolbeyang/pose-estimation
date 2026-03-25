# Tester Plan: Phase 2b, Iteration 1 -- SZI-MPJPE Bug Fixes

## Context

Iteration 0 found three issues:
- Bug #1: SZI-MPJPE > MPJPE on real data (architect ruled this NOT a bug; MSE != MPJPE objectives)
- Bug #2: SZI-MPJPE not printed in test_single.py
- Bug #3: Per-joint SZI-MPJPE not saved to diagnostics.json (pre-existing, not addressed)

Developer applied three fixes:
1. Clarifying docstring on `szi_mpjpe()` in evaluate.py
2. Unit test corrected: "SZI-MPJPE <= MPJPE" replaced with "SSE after <= SSE before"
3. SZI-MPJPE print lines added to test_single.py

## What to Test

### 1. Unit tests pass
- **Command**: `cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python test_szi_mpjpe.py`
- **Expected**: 18/18 pass, including the new SSE property tests

### 2. Smoke test: test_single.py
- **Command**: `cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python test_single.py`
- **Expected**: Completes without error; SZI-MPJPE values appear in stdout output

### 3. Code review: Verify 3 fixes in place
- **evaluate.py**: docstring on `szi_mpjpe()` explains MSE vs MPJPE distinction
- **test_szi_mpjpe.py**: old "SZI-MPJPE <= MPJPE" assertion removed, replaced with SSE property
- **test_single.py**: SZI-MPJPE print lines present for det, opt, and no-ankles variants

### 4. Regression check
- Existing metrics (MPJPE, P-MPJPE, MPJVE) still appear in test_single.py output
- Unit test count still 18 (no tests lost)
