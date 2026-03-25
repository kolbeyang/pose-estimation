# Tester Plan: Phase 4, Iteration 0 -- Batch SH Inference + GPU Acceleration

## What to Test

### Code Review
1. **detect.py -- `_get_device()`**: Verify CUDA > MPS > CPU priority is correct.
2. **detect.py -- `run_hourglass()` batched rewrite**: Verify preprocessing, batched forward passes, flip averaging, keypoint parsing are correct. Check MPS fallback logic.
3. **detect.py -- `run_motionbert()` device update**: Verify uses `_get_device()` and has MPS fallback.
4. **detect.py -- `detect_poses()` timing**: Verify timing instrumentation is correct, optional `return_timing` parameter works.
5. **config.py -- `SH_BATCH_SIZE`**: Verify added with correct default.
6. **benchmark.py -- timing capture**: Verify sub-stage timing is captured and printed.

### Refactoring Review (ALL modified files)
7. **detect.py**: Dead code, unused imports, duplicated logic, confusing naming, TODO/FIXME/HACK comments.
8. **config.py**: Dead code, unused variables, organization issues.
9. **benchmark.py**: Dead code, duplicated logic, config mutation pattern.

### Smoke Test
10. `cd motionbert-pose && uv run python test_single.py` -- Verify full pipeline runs without errors.

### Benchmark Comparison
11. Verify detection timing breakdown is printed and sub-stage times are plausible.

### Accuracy Check
12. Compare MPJPE output against Phase 3 baseline (Det=50.79cm, Opt=44.87cm for example 0).

## How to Test Each Item

- Items 1-9: Code review by reading source files.
- Item 10: Run command and check for errors.
- Items 11-12: Inspect smoke test output and compare with Phase 3 baseline.

## Regression Checks
- `test_single.py` still runs without errors (existing callers unaffected by `return_timing=False` default).
- MPJPE values are unchanged from Phase 3.
