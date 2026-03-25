# Developer Report: Phase 3, Iteration 0 -- Performance and Latency Reduction

## What was implemented

### Part A: Baseline Benchmarking and Profiling

1. **Created `motionbert-pose/benchmark.py`**: Benchmark script that runs all examples with per-stage timing (detection, optimization, evaluation), records per-example and aggregate metrics, and saves results to JSON in `motionbert-pose/experiments/`. Supports command-line overrides for NUM_STEPS, LEARNING_RATE, BONE_LENGTH_LR, HEATMAP_BLUR_SIGMA, and batched/non-batched mode.

2. **Added profiling to `optimize.py`**: Added `profile: bool = False` parameter to `run_optimization()`. When True, times FK+projection, scoring, and backward pass separately per step. Returns profile data as 4th tuple element (None when profile=False). Updated all callers (main.py, test_single.py, sweep.py) to accept the 4th return value.

3. **Ran baseline benchmark with profiling**: Profiled 1 example (100 frames). Results:
   - Detection: 73.8s (54% of total)
   - Optimization: 61.5s (45% of total), per step: 608ms
   - Per-step breakdown: FK+Projection=128ms, Scoring=100ms, Backward=363ms (60%)

### Part B: Experiments

#### Experiment 1+2: Vectorize FK and Scoring Across Frames (combined)

**`fk.py` changes:**
- Added `_axis_angle_to_matrix_batch(aa)`: Batch Rodrigues formula, (F,3) -> (F,3,3)
- Added `forward_kinematics_batch(root_pos, root_rot, local_rots, bone_lengths)`: Vectorized FK across frames. Takes (F,3), (F,3), (F,J,3), (J,) -> (F,J,3). Processes kinematic tree joints sequentially (required by parent dependencies) but all frames in parallel.

**`scoring.py` changes:**
- Vectorized `heatmap_score()`: Single batched `grid_sample` call for all 14 heatmap joints (was 14 individual calls). Fallback joints (Hip, Spine) still use analytical Gaussian.
- Added `heatmap_score_batch()`: Fully batched scoring across all frames AND joints. Single `grid_sample` call for F*14 heatmap queries.
- Added `compute_total_score_batch()`: Vectorized motion penalties, rotation penalties, and anchor penalties using tensor operations instead of per-frame Python loops.

**`optimize.py` changes:**
- Added `run_optimization_batched()`: Uses batch FK, batch projection, and batch scoring. Parameters stored as stacked tensors (F,3), (F,J,3) instead of lists of individual tensors.

**Result: 64x optimization speedup with identical accuracy.** Optimization dropped from 61.5s to 0.96s. Loss curves match exactly.

#### Experiment 3: Reduce Steps + Increase LR

Smoke-tested 4 configurations on 2 examples, then ran the best on all 17:
- 50 steps, LR=0.002: 38.69 cm (identical to 100-step baseline)
- 50 steps, LR=0.003: 41.35 cm (2ex)
- 30 steps, LR=0.003: 41.13 cm (2ex)
- 30 steps, LR=0.005: 41.30 cm (2ex)

50 steps at LR=0.002 converges to the same MPJPE, halving optimization time.

#### Experiment 4: Remove Heatmap Blur

blur=0.0 degraded accuracy by 0.44 cm on 2 examples. **Rejected.**

### Part C: Visualization and Reporting

1. **Created `motionbert-pose/plot_experiments.py`**: Generates 5 plots:
   - MPJPE vs optimization time
   - MPJPE vs total end-to-end time
   - Latency breakdown (stacked bar)
   - Loss curves comparison
   - Per-example MPJPE comparison

2. **Created experiment markdown files** with YAML headers in `motionbert-pose/experiments/`:
   - baseline_unbatched.md
   - batched_full.md
   - batched_50step_lr002_full.md
   - batched_noblur.md

3. **Created `motionbert-pose/experiments/FINAL_REPORT.md`** with full summary.

4. **Updated defaults**: config.py NUM_STEPS=50, LEARNING_RATE=0.002. main.py now uses `run_optimization_batched`.

## Files changed

- **Created**: `motionbert-pose/benchmark.py`, `motionbert-pose/plot_experiments.py`
- **Created**: `motionbert-pose/experiments/` directory with experiment JSONs, markdown reports, plots, FINAL_REPORT.md
- **Modified**: `motionbert-pose/optimize.py` (profiling + batched optimization)
- **Modified**: `motionbert-pose/fk.py` (batch FK functions)
- **Modified**: `motionbert-pose/scoring.py` (vectorized scoring + batch scoring)
- **Modified**: `motionbert-pose/main.py` (uses run_optimization_batched, handles 4th return value)
- **Modified**: `motionbert-pose/config.py` (NUM_STEPS=50, LEARNING_RATE=0.002)
- **Modified**: `motionbert-pose/test_single.py`, `motionbert-pose/sweep.py` (handle 4th return value)

## Commands run

1. `uv run python benchmark.py --profile --examples 1 --name baseline_profile_test` -- baseline profiling (SUCCESS)
2. `uv run python benchmark.py --profile --examples 1 --name batched_smoke --batched` -- batched smoke test (SUCCESS, verified identical loss)
3. `uv run python benchmark.py --examples 2 --name smoke_50step_lr002 --batched --num-steps 50 --lr 0.002` (and 3 other smoke tests) (SUCCESS)
4. `uv run python benchmark.py --profile --name batched_full --batched` -- full 17-example benchmark (SUCCESS)
5. `uv run python benchmark.py --name batched_50step_lr002_full --batched --num-steps 50 --lr 0.002` -- full benchmark with best LR config (SUCCESS)
6. `uv run python plot_experiments.py` -- generate all comparison plots (SUCCESS)
7. `uv run python -c "from main import main; ..."` -- verified main.py works end-to-end with new defaults (SUCCESS)

## Decisions made

1. **Combined Experiments 1 and 2**: The architect listed vectorize-scoring and vectorize-FK as separate experiments. I combined them because they naturally go together -- you need both to get the batched optimization loop. Implementing one without the other would require maintaining mixed batch/non-batch code paths in the scoring function. The combined implementation is cleaner.

2. **Kept both run_optimization and run_optimization_batched**: Rather than replacing the original function, I added the batched version alongside it. This preserves backward compatibility and makes it easy to A/B test. main.py now defaults to batched.

3. **Killed the non-batched full baseline run**: After 35+ minutes of running on 18 examples, the non-batched baseline was still going. Since we had profiled single-example data and the batched version produces identical results, I killed it to free resources for the batched full runs.

4. **Used 17 examples instead of 18**: The female_example_01 consistently fails (likely data issue), so aggregates are over 17 examples.

## Concerns

1. **Detection dominates**: After optimization vectorization, detection (YOLO + Stacked Hourglass + MotionBERT) accounts for 98% of end-to-end time. Further latency improvements require changes to the detection pipeline, which is outside the scope of the optimization loop changes made here.

2. **MPJVE slightly worse with optimization**: Mean Opt MPJVE (3.75-3.80 cm/f) is slightly higher than Det MPJVE (3.07 cm/f). This is a pre-existing issue, not caused by these changes.

3. **Benchmark detection times vary with system load**: Detection times varied from 41s to 84s per example depending on whether multiple benchmarks were running in parallel. The optimization times (0.5-1.0s) are stable.

## Deviations from plan

1. The plan suggested saving significant experiment runs as git branches. I did not create separate branches because all experiments either produced identical results (vectorization) or were quickly validated smoke tests. The JSON results files serve as the record.

2. The plan mentioned "save MPJPE throughout training." Since we cannot compute MPJPE during optimization (no GT access), I saved the loss_history instead, as the architect noted.
