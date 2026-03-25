# Tester Plan: Phase 4, Iteration 1 -- Anchor Penalty Removal + Final Report

## What to Test

1. **Code review: scoring.py** -- Verify `compute_total_score_batch()` no longer accepts or uses `initial_positions` or `init_anchor_weight` parameters. Verify anchor penalty computation block is removed. Verify `"anchor_penalty": 0.0` kept in details dict. Verify `compute_total_score()` (non-batched) still has full anchor support.
2. **Code review: optimize.py** -- Verify `run_optimization_batched()` no longer creates `initial_positions_t` or passes anchor-related params. Verify `run_optimization()` (non-batched) still has anchor support.
3. **Code review: config.py** -- Verify `INIT_ANCHOR_WEIGHT = 0.0`.
4. **Smoke test** -- `cd motionbert-pose && uv run python test_single.py` completes without errors.
5. **PHASE4_REPORT.md** -- Exists and contains timing data, accuracy data, A/B test results, comparison to Phase 3, and full benchmark table.
6. **A/B test results** -- `anchor_ab_with.json` and `anchor_ab_without.json` exist with plausible MPJPE values.
7. **Full benchmark** -- `phase4_final.json` exists with 17 examples.
8. **Refactoring review** -- Check ALL modified files for dead code, unused imports, stale comments, and remaining refactoring opportunities.

## How to Test Each Item

1-3. Read the source files and verify the changes match the architect's plan.
4. Run `uv run python test_single.py` and check exit code + output.
5. Read `PHASE4_REPORT.md` and verify all required sections present.
6-7. Parse JSON files and check key metrics.
8. Run grep for TODOs, unused imports, dead code patterns across all .py files in motionbert-pose/.

## Regression Checks

- Smoke test verifies the main entry point (`test_single.py`) still runs.
- Non-batched path (`compute_total_score`, `run_optimization`) must be unchanged.
- `sweep.py` must still work with anchor weight overrides.
