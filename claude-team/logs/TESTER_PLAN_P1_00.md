# Test Plan: Phase 1, Iteration 0

## Scope
Verify the MotionBERT pose estimation pipeline meets Phase 1 Definition of Done.

## Tests

### T1: Smoke Test -- `uv run python main.py` Runs to Completion
- Run `main.py` from `motionbert-pose/`
- Verify all 10 examples process without errors
- Verify exit code 0

### T2: Results JSON Validation
- Check that 10 prediction JSON files exist in `training_runs/{run}/predictions/`
- Verify each JSON has required fields: sequence, camera, joint_names, camera_intrinsics, metrics (with det_mpjpe, det_p_mpjpe), frames
- Verify camera_intrinsics matches CameraParams schema (fx, fy, cx, cy, image_width, image_height)
- Verify metrics.det_mpjpe and metrics.det_p_mpjpe are present and numeric

### T3: Graphs Validation
- Check that per-joint and per-frame graphs exist for each of 10 examples
- Check that aggregate MPJPE and P-MPJPE bar charts exist

### T4: MPJPE Range Check
- Verify all 10 examples have MPJPE in 20-60 cm range (spec: "roughly 20-60 cm")
- Flag any values outside range as borderline or failing
- Check that mean MPJPE is reasonable

### T5: Code Review -- Type Hints
- Check EVERY function in EVERY .py file for type annotations on all parameters and return values
- Check for un-annotated functions
- Spec says "No exceptions"

### T6: Code Review -- Pydantic Usage
- Verify Pydantic models defined for: CameraParams, ExampleConfig, DetectionResult, EvaluationResult, ExampleResult
- Verify Pydantic is actually used at pipeline data boundaries (config, results)

### T7: Code Review -- No Sibling Imports
- Grep for imports from mediapipe-pose/, rtmw-pose/, or any sibling directory
- Verify all imports are local to motionbert-pose/ or standard packages

### T8: Code Review -- Extra Files
- Check extra files (overlay_heatmaps.py, test_norm_comparison.py, run_all.py) for type hints compliance
- These are part of the codebase and subject to the same constraints
