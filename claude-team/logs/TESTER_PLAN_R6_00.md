# Tester Plan - Round 6, Iteration 0

## Goal

Find `rotation_penalty_scalar` (and optionally `position_penalty_weight`) such that the FK optimizer's MPJVE is strictly less than the raw MotionBERT detector's MPJVE on both Example 0 and Example 5.

Baseline targets:
- Example 0: Det MPJVE = 0.94 cm/f → need Opt MPJVE < 0.94 cm/f
- Example 5: Det MPJVE = 0.51 cm/f → need Opt MPJVE < 0.51 cm/f

Secondary constraint: Opt MPJPE must not regress vs Det MPJPE.

## Context

The developer (Round 6) implemented the following changes to `sweep.py`:
1. Added `det_mpjve_cm` to saved JSON output and printed summary table.
2. Added `--examples` CLI arg for multi-example sweeps.
3. Added `get_round6_configs()`: coarse sweep of rotation_penalty_scalar over {1, 5, 10, 50, 100, 200, 500, 1000}, fixed pos_w=50, blur=4, 100 steps.
4. Added `get_round6_fine_configs()`: fine sweep around `BEST_ROT_S=100.0` (placeholder).
5. Wired `--phase round6` and `--phase round6-fine` into the CLI.

**Step 1 (this tester iteration):** Change `num_steps=100` to `num_steps=50` in both `get_round6_configs()` and `get_round6_fine_configs()`. Rationale: order-of-magnitude sweeps show their structural behavior at 20-30 steps; 100 steps doubles runtime with no benefit for coarse signal.

## What To Test

### Pre-run code edit
- Change `num_steps=100` → `num_steps=50` in `get_round6_configs()` and `get_round6_fine_configs()`.

### Smoke test
- Verify `sweep.py --help` runs correctly and shows `round6` and `round6-fine` in phase choices.
- Verify `sweep.py --phase round6 --examples 0,5` runs to completion without errors.

### Phase A: Coarse sweep (primary)
- Run 8 configs x 2 examples = 16 optimization runs at 50 steps each.
- Metric criterion:
  - PRIMARY: `opt_mpjve_cm < det_mpjve_cm` for BOTH examples 0 and 5
  - SECONDARY: `opt_mpjpe_cm <= det_mpjpe_cm` (no MPJPE regression)
- Check that `det_mpjve_cm` is present in the JSON output.
- Check the printed table shows the Det MPJVE column.

### Phase B: Fine sweep (if Phase A finds a promising region)
- Update `BEST_ROT_S` in `get_round6_fine_configs()` to the best value from Phase A.
- Run ~18 configs x 2 examples at 50 steps each.
- Same metric criterion.

### Phase C: Default update verification
- Update `config.py` ROTATION_PENALTY_SCALAR (and POSITION_PENALTY_WEIGHT if changed).
- Run `main.py 0` and `main.py 5` and verify final MPJVE < det MPJVE.

## How to Test Each Item

| Test | Command |
|------|---------|
| Smoke test (help) | `uv run python sweep.py --help` |
| Phase A coarse sweep | `uv run python sweep.py --phase round6 --examples 0,5` |
| Check JSON output | Inspect `training_runs/sweep_results/` for `det_mpjve_cm` field |
| Phase B fine sweep | `uv run python sweep.py --phase round6-fine --examples 0,5` (after updating BEST_ROT_S) |
| Verification | `uv run python main.py 0` and `uv run python main.py 5` |

## Regression Checks

- `det_mpjpe_cm` should remain constant across all configs for the same example (detection does not change).
- `det_mpjve_cm` should remain constant across all configs for the same example.
- The main entry point `main.py` should still complete without errors after config.py update.

## Expected Runtime

- Phase A: ~10-20 min (8 configs x 2 examples x ~50 steps)
- Phase B: ~15-30 min (18 configs x 2 examples x ~50 steps)
- Phase C: ~5-10 min
