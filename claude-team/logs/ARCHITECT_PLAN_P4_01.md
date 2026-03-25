# Architect Plan: Phase 4, Iteration 1 -- Anchor Penalty Removal + Tester Refactoring + Final Report

## Goal Summary

This iteration has three jobs:

1. **Remove anchor penalty from batched scoring.** The `compute_total_score_batch()` function in `scoring.py` has an initialization anchor penalty (weighted by `INIT_ANCHOR_WEIGHT=5.0`) that penalizes deviation from MotionBERT's initial 3D predictions. We want to remove this from the batched path and set the config default to 0.0. The non-batched path (`compute_total_score()`) keeps anchor support for sweep/experimentation. Before committing, run a quick A/B test on 1 example to measure MPJPE impact.

2. **Address tester refactoring TODOs R1-R4** from TESTER_REPORT_P4_00.md. Skip R5, R6, R7.

3. **Run full benchmark and write Phase 4 Final Report** to `motionbert-pose/experiments/PHASE4_REPORT.md`.

## Files to Modify

### 1. `motionbert-pose/scoring.py`
- Remove `initial_positions` and `init_anchor_weight` parameters from `compute_total_score_batch()`.
- Remove the anchor penalty computation block and its contribution to `total_score`.
- Remove `"anchor_penalty"` from the details dict (or set it to 0.0 for compatibility).

### 2. `motionbert-pose/optimize.py`
- In `run_optimization_batched()`: stop passing `initial_positions` and `init_anchor_weight` to `compute_total_score_batch()`.
- Remove the `initial_positions_t` tensor creation (lines 401-403) since it is no longer needed by the batched scoring function.

### 3. `motionbert-pose/config.py`
- Set `INIT_ANCHOR_WEIGHT: float = 0.0` (was 5.0). Keep the constant for the non-batched path and sweep scripts.

### 4. `motionbert-pose/detect.py`
- R1 already done (developer already extracted `_SH_RGB_MEAN` in Phase 4 iter 0 -- confirmed by reading current detect.py).
- R2 already done (developer already created `_normalize_for_sh()` helper -- confirmed by reading current detect.py).
- R3: Simplify `detect_poses()` return type. Always return timing as the 7th element (None when `return_timing=False`). The current code already does this (line 803 returns the 7-element tuple with `timing` which is None when not requested). The type annotation on lines 731-739 already reflects this. **No change needed -- R3 is already resolved.**

### 5. `motionbert-pose/benchmark.py`
- R4 (R5 in tester numbering): Fix index-based tuple unpacking of `detect_poses()` result. Use proper tuple unpacking instead of `detection_result[0]` through `detection_result[6]`.

Looking at the current code (line 126), it already uses tuple unpacking:
```python
kp_2d, visibility, heatmaps, mpii_kp_2d, affine, positions_3d_norm, detection_timing = detect_poses(frames_rgb, return_timing=True)
```
**R4/R5 is already resolved.** The developer already fixed this in the P4_00 implementation.

## Files to Create

### 1. `motionbert-pose/experiments/PHASE4_REPORT.md`
Final report summarizing all Phase 4 work (batching, GPU acceleration, anchor removal, refactoring, final benchmark numbers).

## Step-by-Step Instructions

### Step 1: Remove anchor penalty from `compute_total_score_batch()` in scoring.py

In `compute_total_score_batch()`:

1. Remove parameters `initial_positions` and `init_anchor_weight` from the function signature (lines 360-361).
2. Remove the anchor penalty computation block (lines 412-416):
   ```python
   # DELETE these lines:
   total_anchor_penalty: torch.Tensor = torch.tensor(0.0)
   if initial_positions is not None and init_anchor_weight > 0.0:
       diff: torch.Tensor = all_positions - initial_positions
       sq_dist: torch.Tensor = (diff ** 2).sum(dim=-1)
       total_anchor_penalty = (sq_dist * visibility).sum()
   ```
3. Remove `- init_anchor_weight * total_anchor_penalty` from the total_score computation (line 421).
4. Keep `"anchor_penalty": 0.0` in the details dict for backwards compatibility with any code that reads the details dict.

### Step 2: Update `run_optimization_batched()` in optimize.py

1. Remove the `initial_positions_t` tensor creation (lines 401-403):
   ```python
   # DELETE these lines:
   initial_positions_t: torch.Tensor = torch.tensor(
       np.array(initial_positions_cam), dtype=torch.float32,
   )
   ```
2. Remove `initial_positions=initial_positions_t` and `init_anchor_weight=cfg.INIT_ANCHOR_WEIGHT` from the `compute_total_score_batch()` call (lines 472-473).

### Step 3: Set INIT_ANCHOR_WEIGHT=0.0 in config.py

Change line 99:
```python
INIT_ANCHOR_WEIGHT: float = 0.0
```

Keep the constant (don't delete it) so the non-batched path and any sweep scripts still work.

### Step 4: A/B test anchor penalty removal

Run a quick 1-example benchmark comparing anchor=5.0 vs anchor=0.0. Since the code change removes anchor from the batched path, we need to test before and after:

**Before the code change** (anchor=5.0):
```bash
cd motionbert-pose && uv run python benchmark.py --examples 1 --batched --name anchor_ab_with
```

**After the code change** (anchor removed):
```bash
cd motionbert-pose && uv run python benchmark.py --examples 1 --batched --name anchor_ab_without
```

Compare the `opt_mpjpe` values. Record both in the final report. The expectation is that removing anchor may slightly increase or decrease MPJPE -- the anchor penalty prevents the optimizer from drifting too far from MotionBERT predictions, but it can also prevent the optimizer from finding better solutions.

**Important**: Run the "before" test FIRST, before making any code changes. Then make the changes and run the "after" test.

### Step 5: Run full benchmark

After all changes are made, run the full benchmark on all 18 examples:

```bash
cd motionbert-pose && uv run python benchmark.py --batched --name phase4_final
```

This will produce `motionbert-pose/experiments/phase4_final.json` with all timing and MPJPE data.

### Step 6: Write Phase 4 Final Report

Create `motionbert-pose/experiments/PHASE4_REPORT.md` with the following sections:

1. **Summary**: One paragraph covering Phase 4 work (batching, GPU, anchor removal, refactoring).
2. **Phase 4a: Batched SH Inference**: Description and timing improvement.
3. **Phase 4b: GPU/MPS Acceleration**: Description and timing improvement.
4. **Anchor Penalty A/B Test**: Table showing MPJPE with/without anchor penalty.
5. **Refactoring**: List of tester items addressed (R1-R4).
6. **Full Benchmark Results**: Table with all 18 examples showing det_mpjpe, opt_mpjpe, improvement, and timing. Include aggregate row.
7. **Comparison to Phase 3 Baseline**: How do the numbers compare to the Phase 3 final numbers?

Use data from `phase4_final.json` and the A/B test JSONs.

## Integration Points

- `compute_total_score_batch()` signature changes -- the only caller is `run_optimization_batched()` in optimize.py. That's the only callsite to update.
- `compute_total_score()` (non-batched) is unchanged. It still supports anchor penalty for the non-batched optimization path.
- `INIT_ANCHOR_WEIGHT=0.0` in config means the non-batched path also gets anchor=0.0 by default, but it can still be overridden via benchmark.py's config mutation pattern.

## Risks and Edge Cases

1. **Anchor removal may increase MPJPE.** The anchor penalty prevents the optimizer from drifting away from MotionBERT's initial predictions, which are generally good. Without it, the optimizer is free to explore further but may overshoot. The A/B test in Step 4 will quantify this risk.

2. **Details dict compatibility.** Any code that reads `details["anchor_penalty"]` (e.g., logging) will break if we remove the key. Solution: keep the key but set it to 0.0 in the batched path.

3. **Non-batched path still uses anchor.** With `INIT_ANCHOR_WEIGHT=0.0`, the non-batched `compute_total_score()` effectively has anchor disabled too. This is intentional -- the constant is kept so sweeps can override it.

## Definition of Done

1. `compute_total_score_batch()` no longer accepts or uses `initial_positions` or `init_anchor_weight` parameters.
2. `run_optimization_batched()` no longer creates `initial_positions_t` or passes anchor-related params to scoring.
3. `INIT_ANCHOR_WEIGHT=0.0` in config.py.
4. A/B test results documented: MPJPE with anchor=5.0 vs anchor=0.0 on 1 example.
5. Full benchmark on all 18 examples saved to `phase4_final.json`.
6. `PHASE4_REPORT.md` written with all required sections.
7. All code passes a smoke test (`test_single.py` or `benchmark.py --examples 1`).
