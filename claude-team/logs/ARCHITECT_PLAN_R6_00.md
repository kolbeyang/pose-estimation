# Architect Plan - Round 6, Iteration 0

## Goal Summary

The FK optimizer currently makes temporal jitter (MPJVE) **worse** than the raw MotionBERT detector on some examples -- notably Example 0 (Det MPJVE=0.94 cm/f, Opt MPJVE=1.56-1.67 cm/f). The goal is to find rotation penalty weights that ensure Opt MPJVE < Det MPJVE on **all** examples, without regressing MPJPE. The approach is a systematic parameter sweep of `ROTATION_PENALTY_SCALAR` and `POSITION_PENALTY_WEIGHT`, using the existing `sweep.py` infrastructure.

## Current State

- `ROTATION_PENALTY_SCALAR = 10.0` (multiplied by per-joint factors: Hip 3.0x, Spine/Thorax 1.0x, Shoulder 0.5x, Elbow 0.3x, Wrist/Ankle 0.1-0.2x)
- `POSITION_PENALTY_WEIGHT = 50.0`
- `NUM_STEPS = 100`, `HEATMAP_BLUR_SIGMA = 4.0`
- `INIT_ANCHOR_WEIGHT = 5.0`
- The existing sweep results (Phase 1.2) only swept blur/steps, not penalty weights at the current defaults
- Phase 1.1 swept rot_s in {0, 0.1, 1, 100, 1000} but only at 20 steps, and the sweep results JSON does **not** save `det_mpjve` -- only `opt_mpjve`
- The key prior data point: from the Phase 1.2 sweep at 100 steps with blur=4.0 (current defaults), Example 0 shows `opt_mpjve_cm = 1.62` while the user reports `det_mpjve = 0.94`. That's a 72% regression.

## Files to Modify

### 1. `motionbert-pose/sweep.py`

**Changes:**
- Add `get_round6_configs()` function returning the coarse sweep configs
- Add `get_round6_fine_configs(best_rot: float)` function returning fine sweep configs around a given rotation scalar
- Extend the `--phase` argument to accept `"round6"` and `"round6-fine"`
- Add `det_mpjve_cm` to the saved JSON output (it's computed but not saved)
- Add `det_mpjve` column to the printed summary table
- Support `--examples` argument to run on multiple examples in one invocation (e.g., `--examples 0,5`)

### No New Files

## Step-by-Step Instructions

### Step 1: Add `det_mpjve_cm` to sweep output

In `sweep.py`, the `save_data` loop (lines ~399-416) currently omits `det_mpjve`. Add it:

```python
"det_mpjve_cm": m.get("det_mpjve", 0) * 100 if "det_mpjve" in m else None,
```

Also add a `Det MPJVE` column to the printed summary table header (line ~377) and the corresponding print line (line ~392). This is critical so we can compare Det vs Opt MPJVE at a glance.

### Step 2: Add multi-example support

Currently `sweep.py` takes a single `example_idx` positional arg. Change `main()` to support `--examples` as a comma-separated list:

```python
parser.add_argument("--examples", type=str, default="0",
                    help="Comma-separated example indices (e.g., '0,5')")
```

Remove the `example_idx` positional arg. The main loop should iterate over each example index, load data once per example, then run all configs on that example. Save separate JSON files per example (this already happens via the `data['name']` key in the filename).

### Step 3: Add `get_round6_configs()`

Add to `sweep.py`:

```python
def get_round6_configs() -> list[SweepConfig]:
    """Round 6: coarse sweep of rotation penalty scalar."""
    configs: list[SweepConfig] = []
    base = dict(
        num_steps=100,
        position_penalty_weight=50.0,
        init_anchor_weight=5.0,
        heatmap_blur_sigma=4.0,
    )
    for rot_s in [1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
        configs.append(SweepConfig(
            name=f"rot_s={int(rot_s)}",
            rotation_penalty_scalar=rot_s,
            **base,
        ))
    return configs
```

Notes:
- Keep `position_penalty_weight=50.0` fixed for the first sweep
- Keep `heatmap_blur_sigma=4.0` (current default)
- The values {1, 5, 10, 50, 100, 200, 500, 1000} span 3 orders of magnitude
- `rot_s=10` is the current default, serving as a baseline reference point

### Step 4: Add `get_round6_fine_configs()`

This function takes the best rotation scalar from the coarse sweep and does a finer sweep around it, also sweeping position penalty weight:

```python
def get_round6_fine_configs() -> list[SweepConfig]:
    """Round 6 fine: sweep around best rotation scalar + position weight combos.

    The developer should update BEST_ROT_S based on coarse sweep results.
    """
    BEST_ROT_S: float = 100.0  # UPDATE after coarse sweep

    configs: list[SweepConfig] = []
    base = dict(
        num_steps=100,
        init_anchor_weight=5.0,
        heatmap_blur_sigma=4.0,
    )

    # Fine rotation sweep: 0.5x, 0.7x, 1.0x, 1.5x, 2.0x, 3.0x of best
    for mult in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        rot_s = BEST_ROT_S * mult
        configs.append(SweepConfig(
            name=f"rot_s={rot_s:.0f}_pos=50",
            rotation_penalty_scalar=rot_s,
            position_penalty_weight=50.0,
            **base,
        ))

    # Position weight sweep at best rotation scalar
    for pos_w in [10.0, 25.0, 50.0, 100.0, 200.0, 500.0]:
        configs.append(SweepConfig(
            name=f"rot_s={BEST_ROT_S:.0f}_pos={int(pos_w)}",
            rotation_penalty_scalar=BEST_ROT_S,
            position_penalty_weight=pos_w,
            **base,
        ))

    # Combo: best rotation * {0.5x, 2.0x} with best-ish position weights
    for rot_mult in [0.5, 2.0]:
        for pos_w in [25.0, 100.0, 200.0]:
            rot_s = BEST_ROT_S * rot_mult
            configs.append(SweepConfig(
                name=f"rot_s={rot_s:.0f}_pos={int(pos_w)}",
                rotation_penalty_scalar=rot_s,
                position_penalty_weight=pos_w,
                **base,
            ))

    return configs
```

### Step 5: Wire up the `--phase` argument

Extend the `--phase` choices in `main()`:

```python
parser.add_argument("--phase", choices=["1.1", "1.2", "round6", "round6-fine"],
                    default="round6",
                    help="Which config set to run")
```

Add the corresponding branches:

```python
elif args.phase == "round6":
    configs = get_round6_configs()
elif args.phase == "round6-fine":
    configs = get_round6_fine_configs()
```

### Step 6: Execution Plan for the Tester/Developer

**Phase A -- Coarse sweep (8 configs x 2 examples = 16 runs):**

```bash
cd motionbert-pose
uv run python sweep.py --phase round6 --examples 0,5
```

This runs all 8 rotation scalar values on Examples 0 and 5. Expected runtime: ~15-30 min depending on hardware (100 steps per config, ~10 sec per config per example).

**Analyze results:** Look at the JSON output. The key criterion is:
- `opt_mpjve_cm < det_mpjve_cm` for BOTH examples
- Secondary: `opt_mpjpe_cm <= det_mpjpe_cm` (don't regress MPJPE)

Identify the rotation scalar value(s) where opt_mpjve < det_mpjve for both examples. If multiple values work, prefer the one that also gives the best MPJPE improvement.

**Phase B -- Fine sweep:**

Update `BEST_ROT_S` in `get_round6_fine_configs()` based on Phase A results, then:

```bash
uv run python sweep.py --phase round6-fine --examples 0,5
```

This runs ~24 configs x 2 examples = 48 runs. Expected runtime: ~40-80 min.

**Phase C -- Validation:**

Once the best `(rotation_penalty_scalar, position_penalty_weight)` pair is found, update `config.py` defaults and run `main.py` on all examples to verify no regressions.

## Integration Points

- `sweep.py` already has the full infrastructure for loading examples, running optimization with config overrides, and saving results. We are only adding new config generators and extending the CLI.
- The `SweepConfig` dataclass already has all needed fields (`rotation_penalty_scalar`, `position_penalty_weight`, etc.).
- `run_sweep_config()` already rebuilds `ROTATION_PENALTY_PER_JOINT` from the scalar and the per-joint multipliers. No changes needed there.
- The per-joint multiplier ratios (Hip 3.0x, Spine 1.0x, etc.) are NOT being changed -- only the overall scalar.

## Risks and Edge Cases

1. **High rotation penalty may hurt MPJPE.** Very high rotation penalties will freeze the skeleton, preventing it from fitting the heatmaps. The sweet spot is where MPJVE improves without MPJPE degrading. This is why we sweep both and check both metrics.

2. **Example 0 may be an outlier.** Example 0 uses the small sample sequence (`171204_pose1_sample`, 100 frames). It may have different characteristics than the full sequences. Running on Example 5 as a regression check mitigates this.

3. **Position penalty and rotation penalty interact.** Higher rotation penalty means the optimizer relies more on root position to track motion, which means position penalty becomes more important. This is why Phase B sweeps both together.

4. **Runtime.** Phase A: 16 runs. Phase B: ~48 runs. At ~10-15 sec per run, total is ~15-90 min. This is manageable.

## Critical Decisions and Assumptions for Review

1. **Per-joint multiplier ratios are held fixed.** The user suggested scaling everything uniformly rather than tuning individual joints. The plan only varies `ROTATION_PENALTY_SCALAR` (the global multiplier), keeping the per-joint ratios (Hip 3.0x, Spine 1.0x, Shoulder 0.5x, etc.) unchanged. If the coarse sweep fails to find a good global scalar, we may need to revisit the per-joint ratios, but that is out of scope for this iteration.

2. **INIT_ANCHOR_WEIGHT is held fixed at 5.0.** This parameter anchors the optimizer to MotionBERT's initial predictions. Changing it could interact with the penalty weights, but the user's spec focuses on rotation/position penalties. If needed, this could be swept in a future round.

3. **HEATMAP_BLUR_SIGMA is held fixed at 4.0.** The Phase 1.2 sweep already established this as the best blur value. Changing it would confound the rotation penalty sweep.

4. **The sweep JSON currently does not save `det_mpjve`.** Step 1 fixes this, but it means we cannot retroactively compare against prior sweep results for det_mpjve. The developer should note the det_mpjve value from the first run -- it should be constant across all configs for a given example (since detection doesn't change, only optimization does).

5. **`BEST_ROT_S` is hardcoded in `get_round6_fine_configs()`.** This is intentionally not dynamic -- the developer must manually update it after reviewing Phase A results. This keeps the code simple and forces a human review checkpoint between phases. An alternative would be to accept it as a CLI argument, but that adds complexity for minimal benefit.

6. **The summary table should print Det MPJVE alongside Opt MPJVE.** Currently the table omits Det MPJVE, making it hard to tell at a glance whether optimization helped or hurt smoothness. This is a display-only change.
