# Tester Plan P1-00: Parameter Sweep Verification

## Test Items

### T1: Verify test_single.py displays new 2D-vs-Det metric
- **What**: Run `test_single.py` and confirm it prints `Det 2D-vs-Det` and `Opt 2D-vs-Det` lines
- **How**: `cd motionbert-pose && uv run python test_single.py 0` -- check stdout for the new metric lines
- **Pass criteria**: Both `Det 2D-vs-Det: XX.X px` and `Opt 2D-vs-Det: XX.X px` appear in output

### T2: Verify sweep results JSON exists and is well-formed
- **What**: Check `training_runs/sweep_results/sweep_171204_pose1_sample_0.json` for completeness
- **How**: Read the file, verify all 20 configs present, all required fields non-null
- **Pass criteria**: 20 entries, each with `config`, `det_mpjpe_cm`, `opt_mpjpe_cm`, `improvement_cm`, `opt_p_mpjpe_cm`, `det_2d_det_mpjpe_px`, `opt_2d_det_mpjpe_px`, `opt_mpjve_cm`

### T3: Verify det_2d_det_mpjpe_px is constant across configs
- **What**: The detector baseline 2D-vs-Det metric should be identical for all configs (detector doesn't change)
- **How**: Check that all `det_2d_det_mpjpe_px` values in the JSON are the same
- **Pass criteria**: All values equal

### T4: Code review -- evaluate.py new metric correctness
- **What**: Review `reprojection_error_vs_detections()` for correctness
- **How**: Read code, verify it projects 3D->2D via camera and compares to SH detections (not GT)
- **Pass criteria**: Uses `camera.world_to_image()` on `positions_3d`, compares to `detections_2d`, visibility masking works

### T5: Code review -- sweep.py config restoration
- **What**: Verify sweep.py properly restores cfg values after each run
- **How**: Read try/finally block in `run_sweep_config()`
- **Pass criteria**: All overridden config values are saved and restored in finally block

### T6: Verify sweep.py can import without errors
- **How**: `cd motionbert-pose && uv run python -c "from sweep import get_phase1_1_configs; configs = get_phase1_1_configs(); print(f'{len(configs)} configs')"`
- **Pass criteria**: Prints "20 configs" with no errors

### T7: Analyze sweep results for Phase 1.1 goals
- **What**: Evaluate whether the coarse-to-fine parameter tuning produced actionable insights
- **How**: Analyze the JSON data for trends across parameter sweeps
- **Pass criteria**: Results are documented with clear interpretation

## Regression Checks

### R1: main.py still imports and has the new metric wired
- **How**: `cd motionbert-pose && uv run python -c "from main import process_example; print('OK')"`
- **Pass criteria**: No import errors
