# Tester Report: Phase 3, Iteration 0 -- Performance and Latency Reduction

## Tests Run

### 1. Code Review: fk.py -- Batch FK
**PASS**

- `_axis_angle_to_matrix_batch`: Correctly vectorizes Rodrigues formula for (F,3) input. Small-angle fallback uses float masking to blend with identity. Correct.
- `forward_kinematics_batch`: Iterates joints sequentially (required by kinematic tree dependencies) but vectorizes across frames (independent). Shapes are correct: (F,3) root_pos, (F,3,3) rotations, bone_lengths broadcast correctly. Uses `@` for batch matmul (F,3,3)@(F,3,3). Returns (F,J,3) via `torch.stack(positions, dim=1)`. Correct.
- Original single-frame `forward_kinematics` and `positions_to_fk_params` preserved for backward compatibility.

### 2. Code Review: scoring.py -- Vectorized Scoring
**PASS with minor finding**

- `heatmap_score`: Correctly uses batch `grid_sample` with (14,1,64,64) batch dim and (14,1,1,2) grid to sample each joint's heatmap at its own location. Confidence-weighted log-likelihood formula matches spec: `log(value * conf + eps * (1 - conf))`.
- `heatmap_score_batch`: Correctly reshapes to (F*14,1,64,64) for a single `grid_sample` call. Fallback joints (Hip, Spine) handled with vectorized analytical Gaussian. Correct.
- `compute_total_score_batch`: Vectorized position penalty uses `all_positions[1:,0,:] - all_positions[:-1,0,:]`, correct. Rotation penalty uses chord distance with broadcasting of per_joint_weights. Anchor penalty vectorized correctly.
- **Minor finding**: Lines 101-105 in `heatmap_score` build an `hm_input` tensor `(1, 14, 64, 64)` that is never used. The actual computation uses `hm_batch` (lines 110-111). This is dead code.

### 3. Code Review: optimize.py -- Batched Optimization
**PASS**

- `run_optimization_batched` correctly stacks parameters as batch tensors: (F,3) for root_pos/root_rot, (F,J,3) for local_rots.
- Uses `forward_kinematics_batch` for batch FK, reshapes to (F*J,3) for batch projection, then (F,J,2) for scoring.
- Uses `compute_total_score_batch` for vectorized scoring.
- Profiling instrumentation mirrors the original function with `time.perf_counter()`.
- Blur application converts stacked tensor to list, applies blur, re-stacks. Correct but slightly inefficient (could apply blur in-place on the stacked tensor). Not a bug.
- Original `run_optimization` (non-batched) preserved for comparison.

### 4. Config Defaults Updated
**PASS**

- `config.py` line 42: `NUM_STEPS: int = 50` (was 100)
- `config.py` line 43: `LEARNING_RATE: float = 0.002` (was 0.001)
- All other hyperparameters unchanged and consistent with experiment results.

### 5. benchmark.py Smoke Test
**PASS**

Command: `cd motionbert-pose && uv run python benchmark.py --examples 1 --name tester_smoke --batched`
- Completed successfully in ~22.6s total.
- Detection: 22.1s, Optimization: 0.4s, Eval: 0.0s
- Det MPJPE: 50.79 cm, Opt MPJPE: 44.87 cm (improvement: +5.92 cm)
- Results saved to `experiments/tester_smoke.json`
- Loss monotonically decreasing (6835.9 -> 6702.4 over 50 steps)

### 6. Experiment JSON Files
**PASS**

All expected JSON files present in `motionbert-pose/experiments/`:
- `baseline_profile_test.json` -- profiling data
- `batched_full.json` -- 17 examples, full metrics, loss_history per example
- `batched_50step_lr002_full.json` -- 17 examples with optimized config
- `batched_noblur.json`, smoke test JSONs
- All contain: config block, examples array with per-example timing/metrics/loss_history, aggregate block

### 7. MPJPE-vs-Time Graph
**PASS**

File exists: `motionbert-pose/experiments/mpjpe_vs_optimization_time.png`
Additional graphs also present: `mpjpe_vs_total_time.png`, `latency_breakdown.png`, `loss_curves.png`, `per_example_comparison.png` (5 total)

### 8. Per-Experiment Markdown Files with YAML Headers
**PASS with minor finding**

All 4 experiment markdown files present with YAML frontmatter:
- `baseline_unbatched.md` -- YAML header present, `mean_opt_mpjve_cm: ~` (null, because only 1 example profiled without MPJVE)
- `batched_full.md` -- YAML header present, all fields populated
- `batched_50step_lr002_full.md` -- YAML header present, all fields populated
- `batched_noblur.md` -- YAML header present, but **missing `mean_opt_mpjve_cm` field entirely**

### 9. FINAL_REPORT.md
**PASS**

File exists at `motionbert-pose/experiments/FINAL_REPORT.md`. Contains:
- Baseline performance table with timing and per-step breakdown
- Per-experiment results with comparison tables
- Clear identification of unequivocal improvements vs tradeoffs
- Final recommended configuration matching config.py defaults
- Future optimization opportunities section

### 10. main.py Uses Batched Optimization by Default
**PASS**

Line 208: `optimized_3d, bone_lengths_final, loss_history, _ = run_optimization_batched(...)`. Correct.

### 11. Loss History Saved in JSON
**PASS**

Verified `batched_full.json` example 0: `loss_history` array contains 100 float values (one per step), monotonically decreasing. Present for all examples.

### 12. Regression: Default Config Works End-to-End
**PASS**

Benchmark smoke test (item 5) confirms the full pipeline runs with the new default config (50 steps, LR=0.002, batched). Optimization converges, MPJPE improves over detection baseline.

## Bugs Found

### Bug 1: Dead code in heatmap_score (Minor)
- **Description**: `scoring.py` lines 101-105 build `hm_input` tensor (1, 14, 64, 64) and `grid` tensor (1, 1, 14, 2) that are never used. The actual `grid_sample` call uses `hm_batch` and `grid_batch` (lines 110-116).
- **Observed**: Code review
- **Severity**: Minor (dead code, no functional impact)
- **Suggested fix**: Delete lines 91-105 (from `# Gather projected positions` through the unused `grid` construction). The comment on line 107-108 explains the issue well.

### Bug 2: Missing YAML field in batched_noblur.md (Minor)
- **Description**: `batched_noblur.md` YAML header is missing the `mean_opt_mpjve_cm` field that other experiment markdown files include.
- **Observed**: Comparing YAML headers across experiment files
- **Severity**: Minor (incomplete metadata, no functional impact)
- **Suggested fix**: Add `mean_opt_mpjve_cm: ~` to the YAML header (or the actual value if available from the JSON).

## Code Review Findings

1. **Dead code in heatmap_score** (Bug 1 above).
2. **Blur application in batched optimizer**: `_apply_blur_torch` takes a list of tensors and returns a list. The batched optimizer converts its stacked (F,16,64,64) tensor to a list, blurs, then re-stacks. This works but could be slightly more efficient by operating on the stacked tensor directly. Not a bug, just a minor optimization opportunity.
3. **Config mutation in benchmark.py**: `benchmark_example` temporarily mutates global `cfg` module attributes (lines 82-93) and restores them in a `finally` block. This is functional but fragile if the benchmark were ever parallelized. Not a bug for current single-threaded usage.

## Verdict

**YES** -- The implementation meets the Phase 3 spec's definition of done.

All required deliverables are present:
- Benchmarks with MPJPE, P-MPJPE, MPJVE, latency/frame for all test videos
- Profiling of each step's duration
- MPJPE-vs-time graph and additional visualizations
- Per-experiment markdown files with YAML headers
- FINAL_REPORT.md with top configuration as default
- Loss history (proxy for MPJPE throughout training) saved
- config.py updated with optimized defaults (50 steps, LR=0.002)
- main.py uses `run_optimization_batched` by default

The 64x optimization speedup is a substantial improvement. Two minor issues found (dead code, missing YAML field) do not block acceptance.
