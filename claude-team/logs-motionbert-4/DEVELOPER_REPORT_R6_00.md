# Developer Report - Round 6, Iteration 0

## What Was Implemented

All five steps from the architect's plan were implemented in `motionbert-pose/sweep.py`:

1. **`det_mpjve_cm` added to JSON output and summary table.** The printed table now has a `Det MPJVE` column between `Opt P-MPJPE` and `Opt MPJVE`. The JSON save_data loop now includes `"det_mpjve_cm"` using the same pattern as `opt_mpjve_cm` (None-safe guard via `"det_mpjve" in m`).

2. **Multi-example support via `--examples` CLI arg.** Removed the old `example_idx` positional argument. Added `--examples` (default `"0"`) that accepts a comma-separated string parsed into `list[int]`. The main loop now iterates over each example index: loads data once per example, runs all configs, prints the summary table, and saves a separate JSON file per example.

3. **`get_round6_configs()` added.** Coarse sweep of `rotation_penalty_scalar` across {1, 5, 10, 50, 100, 200, 500, 1000} with fixed `position_penalty_weight=50.0`, `num_steps=100`, `heatmap_blur_sigma=4.0`, `init_anchor_weight=5.0`.

4. **`get_round6_fine_configs()` added.** Fine sweep using `BEST_ROT_S=100.0` (placeholder, to be updated after coarse sweep results). Runs:
   - Fine rotation sweep at 6 multipliers (0.5x, 0.7x, 1.0x, 1.5x, 2.0x, 3.0x) of BEST_ROT_S with pos=50
   - Position weight sweep {10, 25, 50, 100, 200, 500} at BEST_ROT_S
   - Combo grid: rot_mult {0.5x, 2.0x} x pos_w {25, 100, 200} = 6 additional configs
   - Total: 18 configs

5. **`--phase round6` and `--phase round6-fine` wired up.** Default phase changed to `"round6"`. Phase selection uses explicit `elif` branches.

## Files Changed

- `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/sweep.py` — all changes in this file only

## Commands Run

```
cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose
uv run pyright sweep.py
```

Result:
```
/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/sweep.py
  sweep.py:81:50 - error: Argument of type "ndarray[_AnyShape, dtype[Any]]" cannot be assigned to parameter "resolution" of type "tuple[int, int]" in function "from_panoptic_calibration"
    "ndarray[_AnyShape, dtype[Any]]" is incompatible with "tuple[int, int]" (reportArgumentType)
1 error, 0 warnings, 0 informations
```

The 1 remaining error is **pre-existing** (line 81 in `load_example()`, unchanged). The original file had 16 pyright errors; after my changes it has 1. No new type errors were introduced.

```
uv run python sweep.py --help
```

Result (verified CLI is correct):
```
usage: sweep.py [-h] [--examples EXAMPLES]
                [--phase {1.1,1.2,round6,round6-fine}]

options:
  -h, --help            show this help message and exit
  --examples EXAMPLES   Comma-separated example indices (e.g., '0,5')
  --phase {1.1,1.2,round6,round6-fine}
                        Which config set to run
```

## Decisions Made

1. **No `**base` dict unpacking.** The architect plan's code examples used `**base` dict unpacking, but the task specification explicitly forbids this due to Pyright type errors. All new config functions use explicit keyword arguments for each field, following the pattern already established in `get_phase1_2_configs()`.

2. **Summary table width expanded to 125 chars.** Adding `Det MPJVE` column required widening the separator lines and the `=` header bar from 110 to 125 to accommodate the new column without line wrapping.

3. **`get_round6_fine_configs()` signature.** The architect plan's Step 4 header showed `get_round6_fine_configs(best_rot: float)` but the body text specified a hardcoded `BEST_ROT_S` constant (with a comment "UPDATE after coarse sweep") and no parameter. The body text also shows no argument passed at the call site. I followed the body text (no parameter, hardcoded constant) since that matches the Step 5 call site `configs = get_round6_fine_configs()`.

4. **Multi-example loop structure.** `results` list is re-initialized inside the per-example loop so each example gets its own independent results, table print, and JSON file. Configs are built once outside the loop (no need to rebuild per example).

5. **`det_mpjve_cm` fallback in JSON.** Used `if "det_mpjve" in m else None` (matching the existing pattern for `opt_mpjve_cm`) rather than `m.get("det_mpjve", 0) * 100` to avoid silently outputting `0.0` when the metric is missing.

## Concerns

1. **`BEST_ROT_S` in `get_round6_fine_configs()` is hardcoded to 100.0.** The developer running Phase B must manually update this value after analyzing Phase A results. This is by design per the architect's plan, but it is easy to forget and run the fine sweep with the wrong baseline.

2. **JSON filename collision between examples.** Each example saves to `sweep_{data['name']}.json`. Since `data['name']` is `{seq_name}_{start_frame}`, different examples with different sequences will have different names. But if two examples share the same sequence and start frame (unlikely), they would overwrite each other. This is not a concern with the current `cfg.EXAMPLES` entries.

3. **Pre-existing Pyright error on line 81.** The `resolution` type mismatch in `Camera.from_panoptic_calibration()` was present before this change and is out of scope. It does not affect runtime behavior.

## Deviations from Plan

None. All 5 steps were implemented as specified. The only interpretation call was the `get_round6_fine_configs()` signature (see Decision 3 above), which follows the plan body text rather than the conflicting header signature.
