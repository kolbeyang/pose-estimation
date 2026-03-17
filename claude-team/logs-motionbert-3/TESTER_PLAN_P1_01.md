# Tester Plan P1-01: Heatmap Blur (Phase 1.2) Verification

## Test Items

### T1: Verify Phase 1.2 configs load correctly
- **What**: `get_phase1_2_configs()` returns 17 configs as specified in architect plan.
- **How**: `uv run python -c "from sweep import get_phase1_2_configs; configs = get_phase1_2_configs(); print(f'{len(configs)} configs')"`
- **Pass criteria**: 17 configs, correct names, correct blur/schedule/step combinations.

### T2: Code review -- `_get_blur_sigma()` boundary behavior
- **What**: Verify the schedule lookup produces correct sigma at phase boundaries.
- **How**: Call with step values at each boundary of the `c2f_8to2_100s` schedule `[(0.3, 8.0), (0.7, 4.0), (1.0, 2.0)]`. Check steps 0, 29, 30, 69, 70, 99.
- **Pass criteria**: Transitions happen at the correct step indices (29->30 = 8->4, 69->70 = 4->2).

### T3: Code review -- `_apply_blur_torch()` correctness
- **What**: Verify blur function produces correct shapes, dtype, and actually blurs.
- **How**: Create dummy (16, 64, 64) tensors, apply blur, check output.
- **Pass criteria**: Same shape, same dtype (float32), reduced variance.

### T4: Code review -- double-blur guard in `run_sweep_config()`
- **What**: Verify that when `heatmap_blur_schedule` is set, the fixed pre-blur in `run_sweep_config()` is skipped.
- **How**: Read code; also verify in JSON that all schedule configs have `heatmap_blur_sigma=0.0`.
- **Pass criteria**: Guard condition `if config.heatmap_blur_sigma > 0 and config.heatmap_blur_schedule is None` is correct.

### T5: Verify sweep results JSON completeness
- **What**: All 17 configs present, all fields populated, no nulls in MPJVE.
- **How**: Parse JSON, check field presence and non-null.
- **Pass criteria**: 17 entries, all required fields present and non-null.

### T6: Verify detector baseline is constant across configs
- **What**: `det_mpjpe_cm` and `det_2d_det_mpjpe_px` should be identical for all configs.
- **How**: Check unique values in JSON.
- **Pass criteria**: Exactly 1 unique value for each.

### T7: Code review -- blur applied to heatmaps in optimize.py
- **What**: Verify blur is applied to the correct data path (heatmaps, not analytical Gaussian), re-blurs from originals at phase transitions, and does not break gradient flow.
- **How**: Read `optimize.py` lines 170-239.
- **Pass criteria**: (a) `heatmaps_t_orig` stores unblurred copies; (b) re-blur uses originals; (c) blur only happens at transitions (not per-step); (d) tensors not requiring grad (no gradient breakage).

### T8: Code review -- MPJVE in console output
- **What**: Tester report P1-00 CR-2 recommended adding MPJVE to the console table.
- **How**: Read sweep.py summary table code.
- **Pass criteria**: MPJVE column present in header and per-row output.

### T9: Analyze full sweep results for Phase 1.2 goals
- **What**: Determine if blur improves the 2D-3D tradeoff identified in Phase 1.1.
- **How**: Compare MPJPE, P-MPJPE, MPJVE, 2D-vs-Det across blur configurations.
- **Pass criteria**: Clear analysis with recommendations.

### T10: Edge case -- `_get_blur_sigma` with 1 or 0 steps
- **What**: Verify the `max(num_steps - 1, 1)` guard prevents division by zero.
- **How**: Call with num_steps=1 and num_steps=0.
- **Pass criteria**: No crash, returns valid sigma.

## Regression Checks

### R1: main.py import regression
- **How**: `uv run python -c "from main import process_example; print('OK')"`
- **Pass criteria**: No import errors.
