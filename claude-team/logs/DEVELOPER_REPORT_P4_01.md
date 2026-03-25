# Developer Report: Phase 4, Iteration 1 -- Anchor Penalty Removal + Final Report

## What was implemented

### 1. Anchor Penalty Removal from Batched Scoring

Removed the initialization anchor penalty from `compute_total_score_batch()` in scoring.py:
- Removed `initial_positions` and `init_anchor_weight` parameters from the function signature.
- Removed the anchor penalty computation block (the `if initial_positions is not None` block).
- Removed `- init_anchor_weight * total_anchor_penalty` from the total score computation.
- Kept `"anchor_penalty": 0.0` in the details dict for backwards compatibility.

Updated `run_optimization_batched()` in optimize.py:
- Removed `initial_positions_t` tensor creation (no longer needed).
- Removed `initial_positions=initial_positions_t` and `init_anchor_weight=cfg.INIT_ANCHOR_WEIGHT` from the `compute_total_score_batch()` call.

Set `INIT_ANCHOR_WEIGHT = 0.0` in config.py (was 5.0). The constant is kept for the non-batched path and sweep scripts.

### 2. Tester Refactoring Items R1-R4

All items were already resolved in Phase 4 iteration 0 or were already correct in the current code:
- **R1**: `_SH_RGB_MEAN` constant already extracted in detect.py (done in P4 iter 0).
- **R2**: `_normalize_for_sh()` helper already created in detect.py (done in P4 iter 0).
- **R3**: `detect_poses()` already returns timing as 7th element (None when not requested). No change needed.
- **R4/R5**: `benchmark.py` already uses tuple unpacking for `detect_poses()` result. No change needed.

### 3. A/B Test and Full Benchmark

Ran A/B test and full 18-example benchmark (17 completed, 1 skipped due to missing video data).

### 4. Phase 4 Final Report

Wrote comprehensive report to `motionbert-pose/experiments/PHASE4_REPORT.md`.

## Files changed

- **Modified**: `motionbert-pose/scoring.py` -- Removed anchor penalty params and computation from `compute_total_score_batch()`.
- **Modified**: `motionbert-pose/optimize.py` -- Removed `initial_positions_t` creation and anchor args from scoring call in `run_optimization_batched()`.
- **Modified**: `motionbert-pose/config.py` -- Set `INIT_ANCHOR_WEIGHT = 0.0`.
- **Created**: `motionbert-pose/experiments/PHASE4_REPORT.md` -- Final Phase 4 report.
- **Created**: `motionbert-pose/experiments/anchor_ab_with.json` -- A/B test (anchor=5.0).
- **Created**: `motionbert-pose/experiments/anchor_ab_without.json` -- A/B test (anchor=0.0).
- **Created**: `motionbert-pose/experiments/phase4_final.json` -- Full benchmark results.

## Commands run

1. `cd motionbert-pose && uv run python benchmark.py --examples 1 --batched --name anchor_ab_with` -- A/B baseline with anchor=5.0 (SUCCESS). Opt MPJPE = 44.87 cm.
2. *(Made code changes)*
3. `cd motionbert-pose && uv run python benchmark.py --examples 1 --batched --name anchor_ab_without` -- A/B test without anchor (SUCCESS). Opt MPJPE = 48.70 cm.
4. `cd motionbert-pose && uv run python benchmark.py --batched --name phase4_final` -- Full 18-example benchmark (SUCCESS). 17 completed, 1 skipped. Mean Opt MPJPE = 39.11 cm.

## Decisions made

1. **Refactoring items R1-R4**: The architect's plan noted these might already be resolved. After verifying the current code, confirmed all four items were already addressed. No additional changes were made.

2. **Anchor A/B test order**: Ran the "with anchor" test first (before code changes), then made changes and ran "without anchor" test, as specified in the plan.

## Concerns

1. **Anchor removal degrades accuracy**: The A/B test shows removing the anchor penalty increases mean Opt MPJPE by 0.42 cm (38.69 -> 39.11 cm). On the single-example test, the regression was much larger: 3.83 cm (44.87 -> 48.70 cm). The anchor was providing useful regularization by keeping the optimizer close to MotionBERT's predictions. On 6 of 17 examples, optimization now makes MPJPE worse (vs 4 of 17 with anchor). The architect's plan specified this change, so it was implemented as directed.

2. **Non-batched path still has anchor support**: The non-batched `compute_total_score()` and `run_optimization()` still reference `initial_positions` and `INIT_ANCHOR_WEIGHT`. With the config default now 0.0, these are effectively disabled but can be re-enabled for experiments. `test_single.py` uses the non-batched path, so it will also use anchor=0.0 by default.

## Deviations from plan

None. All steps were implemented as specified.
