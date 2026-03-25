# Architect Plan: Phase 3, Iteration 0 -- Performance and Latency Reduction

## Goal Summary

Establish benchmarks and profiling for the full MotionBERT pose optimization pipeline, then run a series of experiments to reduce end-to-end latency per frame while maintaining or improving MPJPE/P-MPJPE/MPJVE accuracy. Produce visualizations (MPJPE vs training-time) and a final report with the best configuration left as the default.

## Pipeline Performance Analysis

The end-to-end pipeline for one example (~150 frames) does:

1. **Load calibration** -- disk I/O, negligible
2. **Extract video frames** -- OpenCV decode, moderate
3. **YOLO person detection** -- 1 forward pass per frame, moderate
4. **Stacked Hourglass 2D** -- 2 forward passes per frame (original + flip), significant
5. **MotionBERT 3D lifting** -- 1 forward pass on full sequence, moderate
6. **motionbert_to_camera_space** -- numpy, negligible
7. **FK initialization (positions_to_fk_params)** -- per-frame inverse FK, likely negligible
8. **Optimization loop (100 steps)** -- per step: FK for all frames + projection + scoring + backward. This is the PRIMARY target.
9. **Evaluation** -- numpy, negligible
10. **Graph generation + overlay video** -- matplotlib + cv2, moderate but not on the critical optimization path

**Key bottleneck in optimization (step 8):** The inner loop iterates over frames sequentially in Python (lines 202-213 of optimize.py). Each frame does `forward_kinematics` (16 sequential Rodrigues formula calls in Python), `world_to_image_torch` (simple division), and then `heatmap_score` (Python loop over 16 joints with individual `grid_sample` calls). The scoring function in `compute_total_score` also loops over frames. None of this is batched/vectorized.

## Files to Create

1. **`motionbert-pose/benchmark.py`** -- Benchmark script that runs all examples, records per-stage timing + metrics, saves results to JSON.
2. **`motionbert-pose/experiments/`** -- Directory for experiment markdown reports.
3. **`motionbert-pose/plot_experiments.py`** -- Script to read experiment results and produce MPJPE-vs-time graphs.

## Files to Modify

1. **`motionbert-pose/optimize.py`** -- Add timing instrumentation; later, vectorize the inner loop.
2. **`motionbert-pose/scoring.py`** -- Vectorize `heatmap_score` to batch all joints; vectorize `compute_total_score` to batch all frames.
3. **`motionbert-pose/fk.py`** -- Batch `forward_kinematics` across frames; vectorize `_axis_angle_to_matrix`.
4. **`motionbert-pose/config.py`** -- Update default hyperparameters based on experiment results.

## Step-by-Step Instructions

### Part A: Baseline Benchmarking and Profiling (do this first)

#### Step 1: Create `benchmark.py`

Create `motionbert-pose/benchmark.py` that:

- Imports and reuses `process_example` from `main.py` (or its substeps directly).
- For each example in `cfg.EXAMPLES`, times the following stages separately using `time.perf_counter()`:
  - `detection_time`: Steps 1-4 (calibration load through `motionbert_to_camera_space`)
  - `optimization_time`: Step 5 (`run_optimization`)
  - `evaluation_time`: Step 6 (`compute_comparison_with_optimization`)
  - `visualization_time`: Steps 7 (graphs + overlay video)
  - `total_time`: wall-clock for the entire example
- Records the frame count per example.
- Computes `latency_per_frame = total_time / num_frames`.
- Also extracts from the returned metrics: `det_mpjpe`, `opt_mpjpe`, `det_p_mpjpe`, `opt_p_mpjpe`, `det_mpjve`, `opt_mpjve`.
- Saves a JSON file at `motionbert-pose/experiments/baseline.json` containing:
  ```json
  {
    "config": { "NUM_STEPS": 100, "LEARNING_RATE": 0.001, ... },
    "examples": [
      {
        "name": "...",
        "num_frames": 50,
        "detection_time_s": ...,
        "optimization_time_s": ...,
        "evaluation_time_s": ...,
        "visualization_time_s": ...,
        "total_time_s": ...,
        "latency_per_frame_s": ...,
        "opt_mpjpe": ...,
        "opt_p_mpjpe": ...,
        "opt_mpjve": ...,
        "det_mpjpe": ...,
        "loss_history": [...]
      }, ...
    ],
    "aggregate": {
      "mean_opt_mpjpe": ...,
      "mean_latency_per_frame_s": ...,
      ...
    }
  }
  ```
- Also saves the **loss_history** (MPJPE throughout training is not directly available during optimization since we don't have GT there, but we DO have the loss curve). The spec says "save out a data file with the MPJPE throughout training" -- since we can't compute MPJPE during training without GT, save the loss_history which is already returned by `run_optimization`. This is the optimization objective value over steps.
- Prints a summary table at the end.

**Implementation note:** To time the substeps, the simplest approach is to refactor `process_example` in `main.py` to accept optional timing hooks, OR (simpler) just duplicate the high-level flow in `benchmark.py` with timing wrappers around each phase. Prefer the simpler duplication approach -- benchmark.py will call the same underlying functions but wrap them in timers.

#### Step 2: Add profiling to the optimization loop

Modify `run_optimization` in `optimize.py` to optionally collect per-step timing. Add a parameter `profile: bool = False`. When True:
- Time each step's forward pass (FK + projection), scoring, and backward pass separately using `time.perf_counter()`.
- Return an additional dict with timing breakdown: `{"fk_projection_ms": [...], "scoring_ms": [...], "backward_ms": [...], "step_total_ms": [...]}`.
- Keep the default `profile=False` so normal runs aren't affected.

Change the return type to:
```python
def run_optimization(..., profile: bool = False) -> tuple[list[np.ndarray], np.ndarray, list[float], dict | None]:
```

The fourth element is `None` when `profile=False`, or the timing dict when `profile=True`.

Update callers (`main.py`, `benchmark.py`) to accept the 4th return value (can just `_ = ` it in main.py).

#### Step 3: Run baseline benchmark

Run `benchmark.py` on ALL examples. This establishes the baseline. Save the output JSON.

Also run with `profile=True` on ONE example (e.g., the first one) to get the per-step timing breakdown. Print the mean per-step timing for FK+projection, scoring, and backward. This tells us where time is spent within the optimization loop.

### Part B: Experiments (start after baseline is established)

The developer should adapt based on profiling results. Here are the first 4 experiments to try, in priority order:

#### Experiment 1: Vectorize scoring across joints

**Hypothesis:** The `heatmap_score` function loops over 16 joints individually, calling `grid_sample` 14 times (one per joint with a heatmap). Batching all joints into a single `grid_sample` call should be significantly faster.

**Changes to `scoring.py`:**
- In `heatmap_score`, separate joints into two groups: those with MPII heatmaps (14 joints) and fallback joints (2 joints: Hip, Spine).
- For the 14 heatmap joints, build the grid tensor once as `(1, 1, 14, 2)` and the heatmap input as `(1, 14, 64, 64)`, then call `grid_sample` once.
- Apply the confidence-weighted log-likelihood formula vectorized across all 14 joints.
- Sum with the 2 fallback joint scores.

**Validation:** Run on one example, compare loss curve and final MPJPE to baseline. They should be identical (this is a pure refactor).

#### Experiment 2: Vectorize FK and scoring across frames

**Hypothesis:** The optimization loop iterates over frames in Python. Batching FK across frames would reduce Python overhead.

**Changes to `fk.py`:**
- Add `forward_kinematics_batch(root_pos_batch, root_rot_batch, local_rots_batch, bone_lengths)` that takes `(F, 3)`, `(F, 3)`, `(F, J, 3)`, `(J,)` and returns `(F, J, 3)`.
- Vectorize `_axis_angle_to_matrix` to handle `(F, 3)` -> `(F, 3, 3)` using batch Rodrigues.

**Changes to `optimize.py`:**
- Stack all per-frame parameters into batch tensors before the loop.
- Call `forward_kinematics_batch` once instead of N times.
- Call `camera.world_to_image_torch` on `(F*J, 3)` reshaped input.

**Changes to `scoring.py`:**
- Add batched `compute_total_score_batch` that processes all frames at once.

**Validation:** Same as Experiment 1 -- loss curve and MPJPE should match baseline exactly.

#### Experiment 3: Reduce steps + increase LR

**Hypothesis:** 100 steps may be more than needed. With higher LR, we may converge in fewer steps.

**Configurations to try (quick smoke tests first -- run ~20 steps each, check if loss is decreasing reasonably):**
- 50 steps, LR=0.002
- 50 steps, LR=0.003
- 30 steps, LR=0.005
- 30 steps, LR=0.003

For each that looks promising in the smoke test, run the full example set and compare MPJPE + latency.

**Note:** When changing LR, the penalty weights may need adjustment. The developer should check if the loss components (heatmap vs penalties) remain balanced.

#### Experiment 4: Reduce heatmap blur overhead

**Hypothesis:** The config has `HEATMAP_BLUR_SIGMA = 4.0`, which applies scipy Gaussian blur to all heatmaps before optimization. If the blur schedule is not used (it's None by default), this is a one-time cost, but it may still be significant for 150 frames x 16 channels.

**Try:** Set `HEATMAP_BLUR_SIGMA = 0.0` and see if accuracy degrades. If not, it's a free speedup.

### Part C: Visualization and Reporting

#### Step 4: Create `plot_experiments.py`

Create `motionbert-pose/plot_experiments.py` that:
- Reads all JSON files in `motionbert-pose/experiments/`.
- For each experiment, plots MPJPE (y-axis) vs cumulative optimization time in seconds (x-axis). Use the loss_history length * per-step time to reconstruct the time axis, and the final opt_mpjpe as the endpoint.
- Each experiment is a labeled point or line on the graph.
- Saves the graph to `motionbert-pose/experiments/mpjpe_vs_time.png`.
- Also generates any other useful comparison graphs the developer sees fit (e.g., latency breakdown bar chart).

#### Step 5: Write experiment markdown files

For each experiment that was run to completion (not just smoke tests), save a markdown file at `motionbert-pose/experiments/<experiment_name>.md` with the format:

```yaml
---
name: <descriptive name matching graph label>
num_steps: <N>
learning_rate: <LR>
bone_length_lr: <BL_LR>
sigma: <sigma>
heatmap_blur_sigma: <blur>
position_penalty_weight: <ppw>
init_anchor_weight: <iaw>
mean_opt_mpjpe_cm: <val>
mean_opt_p_mpjpe_cm: <val>
mean_opt_mpjve_cm: <val>
mean_latency_per_frame_s: <val>
mean_optimization_time_s: <val>
---

## Description
<1-2 sentences describing what was changed and why>

## Results
<Key numbers: MPJPE, latency, comparison to baseline>

## Conclusion
<Was this an improvement? What tradeoff was made?>
```

#### Step 6: Final report and config update

- Identify the best configuration (best tradeoff of MPJPE vs latency).
- Update `config.py` with the winning hyperparameters as defaults.
- Write `motionbert-pose/experiments/FINAL_REPORT.md` summarizing:
  - Baseline performance
  - Each experiment tried, results, whether it was an improvement
  - Which changes were unequivocal wins vs tradeoffs
  - The final recommended configuration and why

## Integration Points

- `benchmark.py` reuses the same detection/optimization/evaluation functions as `main.py`.
- Vectorization changes to `fk.py` and `scoring.py` must be backward-compatible: existing single-frame functions should still work, or callers in `main.py` must be updated.
- The 4th return value from `run_optimization` must not break `main.py` -- use `*rest` unpacking or update all callers.

## Risks and Edge Cases

1. **Vectorized FK correctness:** The kinematic tree walk is inherently sequential (each child depends on its parent). Batching across frames is safe since frames are independent. Batching across joints within a frame is NOT possible for the tree walk -- joints must still be processed in order. Only the per-frame dimension can be batched.

2. **Numerical differences from vectorization:** Batched `grid_sample` may produce slightly different results than per-joint calls due to floating-point ordering. Verify with `torch.allclose(atol=1e-5)`.

3. **LR experiments may interact with penalty weights:** If LR is doubled, the effective step size for penalties also doubles. The developer should monitor the loss component breakdown (heatmap vs pos_penalty vs rot_penalty) to ensure balance is maintained.

4. **Detection time dominates:** If profiling shows that YOLO + Stacked Hourglass + MotionBERT collectively take much more time than optimization, then optimizing the optimization loop has limited impact on end-to-end latency. In that case, focus on reducing detection overhead (e.g., skip YOLO if bbox is known, batch SH inference, etc.).

5. **Graph generation and overlay video:** These are post-processing and don't affect the core pipeline latency, but they can be slow. The benchmark should report visualization time separately so it's clear how much of the wall-clock is just I/O.

## Developer Notes

- Use `time.perf_counter()` not `time.time()` for timing (higher resolution).
- For smoke tests, run on just 1-2 examples, not all 18. The spec says to scale investment proportional to evidence.
- When a configuration looks promising, run on all examples and save the full JSON.
- The spec says significant runs should be saved in git history on branches. Create a branch like `experiment/<name>` for each significant result.
- Keep experiment JSONs small -- don't dump full per-frame 3D positions, just the metrics and loss history.
