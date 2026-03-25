# Tester Plan: Phase 3, Iteration 0 -- Performance and Latency Reduction

## What to Test

### Code Review
1. **fk.py**: Verify batch FK implementation correctness (axis-angle batch, kinematic tree walk across frames)
2. **scoring.py**: Verify vectorized heatmap_score, heatmap_score_batch, compute_total_score_batch
3. **optimize.py**: Verify run_optimization_batched uses batch FK/scoring, profiling instrumentation correct
4. **config.py**: Verify defaults updated to NUM_STEPS=50, LEARNING_RATE=0.002

### Functional Tests
5. **benchmark.py exists and runs**: `uv run python benchmark.py --examples 1 --name tester_smoke --batched`
6. **Experiment JSON files exist** in experiments/ with expected fields (config, examples, aggregate, loss_history)
7. **MPJPE-vs-time graph exists**: mpjpe_vs_optimization_time.png
8. **Per-experiment markdown files** with YAML headers: baseline_unbatched.md, batched_full.md, batched_50step_lr002_full.md, batched_noblur.md
9. **FINAL_REPORT.md exists** with summary, recommendations, config
10. **main.py uses run_optimization_batched by default**
11. **Loss history (MPJPE throughout training) saved** in JSON experiment files

### Regression Checks
12. **main.py smoke test**: Confirm default config works end-to-end (verified via benchmark.py --examples 1)

## How to Test Each Item

- Items 1-4: Read and review code
- Item 5: Run benchmark.py with --examples 1
- Items 6-11: Check file existence and contents
- Item 12: Run benchmark with default config (batched mode)
