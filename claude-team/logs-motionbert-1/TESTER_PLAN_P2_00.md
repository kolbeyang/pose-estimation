# TESTER_PLAN_P2_00: More In-Depth Outputs

## What to Test

Phase 2 spec requirements:
1. Same 3D visualizations as mediapipe-pose
2. All the graphs that mediapipe-pose generates
3. Summary graph per video per run (like mediapipe-pose)

### Test 1: Smoke test -- run test_single.py for one example
- **Command**: `cd motionbert-pose && uv run python test_single.py`
- **Expected**: Completes without errors, prints metrics including MPJVE and 2D MPJPE

### Test 2: Check all expected graph files are generated
- **Command**: List contents of the test_single output directory (or run main.py for one example and check graphs dir)
- **Expected files**:
  - `summary.png` (4x4 grid)
  - `loss_curve.png`
  - `bone_lengths.png`
  - `per_joint_error.png`
  - `per_frame_mpjpe.png`
  - `per_joint_mpjve.png`
  - `per_frame_mpjve.png`
  - 51 trajectory PNGs (17 joints x 3 coords: `{JointName}_{X|Y|Z}.png`)

### Test 3: Compare graph types against mediapipe-pose
- **Command**: List mediapipe-pose graph files from a training run and compare against motionbert-pose output
- **Expected**: Feature parity -- same graph types present in both

### Test 4: Visualize.py runs without errors
- **Command**: `cd motionbert-pose && uv run python visualize.py <path-to-prediction-json>`
- **Expected**: Starts without import errors or crash (will kill after brief run since no display)

### Test 5: Code review -- verify consistency
- Check that motionbert graphs.py has all functions from mediapipe graphs.py
- Check that evaluate.py computes all needed metrics (MPJVE, bone lengths, 2D reprojection)
- Check that main.py calls all graph functions
- Check that visualize.py uses correct JSON keys (`detector_3d` not `mediapipe_3d`)

## Regression Checks
- test_single.py should still produce metrics (det_mpjpe, opt_mpjpe, etc.)
- main.py imports should all resolve correctly
