# Tester Report: Phase 4, Iteration 1 -- Anchor Penalty Removal + Final Report

## Tests Run

### 1. Code Review: scoring.py -- Anchor removal from batched function
**PASS**

- `compute_total_score_batch()` (line 351) no longer has `initial_positions` or `init_anchor_weight` parameters. Confirmed.
- The anchor penalty computation block (the `if initial_positions is not None` block) is fully removed. The total score computation (line 407-411) only includes `total_heatmap`, `position_penalty`, and `rotation_penalty`. Confirmed.
- `"anchor_penalty": 0.0` is kept in the details dict (line 417) for backwards compatibility. Confirmed.

### 2. Code Review: scoring.py -- Non-batched function still has anchor support
**PASS**

- `compute_total_score()` (line 264) still accepts `initial_positions_list` and `init_anchor_weight` parameters. Confirmed.
- The `initialization_penalty()` function (line 240) is still defined and used by the non-batched path. Confirmed.
- Lines 329-332 still compute anchor penalty when `initial_positions_list is not None and init_anchor_weight > 0.0`. Confirmed.
- Line 338 still subtracts `init_anchor_weight * total_anchor_penalty` from total score. Confirmed.

### 3. Code Review: optimize.py -- Batched path anchor removal
**PASS**

- `run_optimization_batched()` (line 324) no longer creates `initial_positions_t`. Confirmed -- no reference to `initial_positions_t` exists in this function.
- The `compute_total_score_batch()` call (line 465-472) does not pass `initial_positions` or `init_anchor_weight`. Confirmed.

### 4. Code Review: optimize.py -- Non-batched path still has anchor support
**PASS**

- `run_optimization()` (line 42) still creates `initial_positions_t` (line 140-142) with the comment "Store initial positions (for anchor penalty)".
- The `compute_total_score()` call (line 249-263) still passes `initial_positions_list=initial_positions_t` and `init_anchor_weight=cfg.INIT_ANCHOR_WEIGHT`. Confirmed.

### 5. Code Review: config.py -- INIT_ANCHOR_WEIGHT
**PASS**

- Line 99: `INIT_ANCHOR_WEIGHT: float = 0.0`. Confirmed (was 5.0).
- The constant is preserved (not deleted), allowing sweep.py and non-batched path to override it. Confirmed.

### 6. Smoke Test: test_single.py
**PASS**

Command: `cd motionbert-pose && uv run python test_single.py`

Output confirms:
- SH loaded on MPS, batched inference (4 batches of 32 for 100 frames).
- MotionBERT loaded on MPS.
- Optimization ran 50 steps (batched not used by test_single -- uses non-batched path).
- Step 0 loss: 6827.8 (matches expected value for anchor=0.0).
- Step 49 loss: 6634.2.
- Det MPJPE: 50.79 cm, Opt MPJPE: 48.70 cm (matches anchor=0.0 A/B test).
- Overlay video generated successfully.
- No errors.

### 7. PHASE4_REPORT.md Exists and Has Required Sections
**PASS**

File: `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/experiments/PHASE4_REPORT.md`

Contains all required sections:
- Summary (1 paragraph)
- Phase 4a: Batched SH Inference (description + impact)
- Phase 4b: GPU/MPS Acceleration (description + impact)
- Anchor Penalty A/B Test (single-example and full-benchmark comparison tables)
- Refactoring (R1-R4 status table)
- Full Benchmark Results (17-example table with detection/optimization/total timing)
- Comparison to Phase 3 Baseline (table with deltas)

### 8. A/B Test Results Documented
**PASS**

- `anchor_ab_with.json`: opt_mpjpe = 0.4487 (44.87 cm). Matches developer report.
- `anchor_ab_without.json`: opt_mpjpe = 0.4870 (48.70 cm). Matches developer report.
- Delta: +3.83 cm. Documented in PHASE4_REPORT.md. Confirmed.

### 9. Full Benchmark Results
**PASS**

- `phase4_final.json`: 17 examples completed (1 skipped -- female_example_01, likely missing video data).
- `mean_opt_mpjpe` = 0.3911 (39.11 cm). Matches developer report and PHASE4_REPORT.md.
- Timing data present (detection, optimization, sub-stage breakdowns).

### 10. Refactoring Review -- Dead Code, Unused Imports, Stale Comments
**PASS (with findings)**

Checks performed:
- **No TODO/FIXME/HACK/XXX comments** in any .py file in motionbert-pose/. Clean. (The TODO on line 693 from the P4_00 report is no longer present -- line numbers shifted or it was removed.)
- **scoring.py imports**: Only `torch` and `torch.nn.functional as F`. Both are used. No unused imports.
- **optimize.py imports**: `numpy`, `torch`, `Camera`, `forward_kinematics`, `forward_kinematics_batch`, `positions_to_fk_params`, `compute_total_score`, `compute_total_score_batch`, `NUM_JOINTS`, `config`. All used. No unused imports.
- **config.py imports**: `os`, `numpy`. Both used. No unused imports.
- **`initialization_penalty()` in scoring.py** (line 240-261): This standalone function is only called by `compute_total_score()` (non-batched). It is NOT dead code -- the non-batched path and sweep.py still use it. Correct to keep.
- **sweep.py**: Still references `init_anchor_weight=5.0` in hardcoded sweep configs. This is correct -- sweep configs define specific experiments and should keep their original values.
- **benchmark.py line 126**: Uses tuple unpacking for `detect_poses()` result. R5 from previous report is resolved. Confirmed.

## Bugs Found

None.

## Code Review Findings

### Finding 1: Non-batched path creates anchor tensors even when weight is 0.0 (Minor, pre-existing)
- **Description**: In `run_optimization()` (optimize.py, lines 139-142), `initial_positions_t` is always created from `initial_positions_cam`, even when `cfg.INIT_ANCHOR_WEIGHT = 0.0`. This wastes memory creating tensors that will never contribute to the score. The guard `init_anchor_weight > 0.0` exists in `compute_total_score()`, so no incorrect behavior occurs -- just unnecessary allocation.
- **Severity**: Minor (micro-optimization, no functional impact)
- **Location**: `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/optimize.py` lines 139-142

### Finding 2: Anchor A/B test shows meaningful accuracy regression (Informational)
- **Description**: Removing the anchor penalty increased mean Opt MPJPE by 0.42 cm across 17 examples (38.69 -> 39.11 cm). On the single-example test, the regression was 3.83 cm (44.87 -> 48.70 cm). The number of examples where optimization makes things worse increased from 4/17 to 6/17. This is documented in the report and was an expected trade-off, but it suggests the anchor penalty was providing useful regularization.
- **Severity**: Informational (documented design decision)

### Finding 3: Model reloading on every call (Pre-existing, carried from P4_00 R7)
- **Description**: Both Stacked Hourglass and MotionBERT models are loaded from disk on every call to `run_hourglass()` and `run_motionbert()`. For benchmark.py running 17+ examples, this means reloading ~100MB models repeatedly. This was flagged in TESTER_REPORT_P4_00 as R7 and was not in scope for this iteration.
- **Severity**: Minor (performance, pre-existing)

### Finding 4: Config mutation pattern (Pre-existing, carried from P4_00 R6)
- **Description**: Both `benchmark.py` and `sweep.py` temporarily mutate global `cfg` module attributes. This is fragile if benchmarking were ever parallelized. Was flagged in previous reports and remains.
- **Severity**: Minor (tech debt, pre-existing)

## REFACTORING TODO LIST

| # | File | Description | Severity | Status |
|---|------|-------------|----------|--------|
| R1 | detect.py | Extract `_SH_RGB_MEAN` constant | Minor | DONE (P4 iter 0) |
| R2 | detect.py | Extract `_normalize_for_sh()` helper | Minor | DONE (P4 iter 0) |
| R3 | detect.py | Simplify `detect_poses()` return type | Minor | RESOLVED (already returns 7-tuple) |
| R4 | detect.py | TODO about camera params for back-projection | Design question | RESOLVED (TODO removed) |
| R5 | benchmark.py | Use tuple unpacking for detect_poses result | Minor | RESOLVED (already done) |
| R6 | benchmark.py / sweep.py | Config mutation pattern -- use function params instead of mutating globals | Minor/Tech debt | OPEN |
| R7 | detect.py | Cache loaded models (SH, MotionBERT) to avoid reloading per call | Minor/Performance | OPEN |
| R8 | optimize.py | Non-batched path creates `initial_positions_t` even when anchor weight is 0.0 -- guard with `if cfg.INIT_ANCHOR_WEIGHT > 0.0` | Minor | NEW |

Items R1-R5 are resolved. Items R6-R8 remain open but are minor/optional.

## Verdict

**YES** -- The implementation meets the Phase 4 iteration 1 definition of done.

All required deliverables verified:
1. `compute_total_score_batch()` no longer accepts or uses `initial_positions` or `init_anchor_weight`. Confirmed by code review.
2. `run_optimization_batched()` no longer creates `initial_positions_t` or passes anchor-related params. Confirmed by code review.
3. `INIT_ANCHOR_WEIGHT = 0.0` in config.py. Confirmed.
4. A/B test results documented: MPJPE with anchor=5.0 (44.87 cm) vs anchor=0.0 (48.70 cm) on 1 example, and 38.69 vs 39.11 cm on 17 examples. Confirmed in PHASE4_REPORT.md and JSON files.
5. Full benchmark on 17 examples saved to `phase4_final.json`. Confirmed.
6. `PHASE4_REPORT.md` written with all 7 required sections. Confirmed.
7. Smoke test (`test_single.py`) passes. Confirmed.

No bugs found. Three open refactoring items (R6, R7, R8) identified for optional future cleanup.
