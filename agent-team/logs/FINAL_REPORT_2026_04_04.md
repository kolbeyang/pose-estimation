# Final Report — 2026-04-04 Refactor

**Branch**: `refactor-2026-04-04`  
**Commits**: 9 (from `a88cf36` to `3e89aa9`)  
**Spec**: `agent-team/specs/2026-04-04-11-44.md`

---

## Summary

All 4 phases completed and verified. Each phase went through Developer → Evaluator → Nit-pick cycles.

## Phase 1: Unify Mediapipe and MotionBERT Pipelines

**Commit**: `a88cf36` (feat), `85ff384` + `9a2c88d` (fixes)

### What changed
- **`main.py`** (new): Unified entrypoint — `uv run main.py config.json` runs both pipelines based on config sections.
- **`config.py`**: Added `PipelineConfig`, `PipelineGraphConfig`, `CrossPipelineGraphConfig`. Config has optional `mediapipe`/`motionbert`/`graphs` sections.
- **`skeleton.py`**: Joint 9 renamed Head→Nose. 15 eval joints (excludes only Spine). Nose synthesis: `skel[8] + 0.3 * (keypoints_mpii[9] - skel[8])`.
- **`run_motionbert/detect.py`**: Extracted `detect_2d_poses()` + `load_yolo_sh_models()` for shared YOLO+SH usage.
- **`configs/`**: 7 config files (6 required + `both-local-single.json`).

### Key decision
MediaPipe still uses its own 2D landmarks for solvePnP camera estimation, but uses StackedHourglass heatmaps for optimization — matching the spec's intent that solvePnP uses MP landmarks while the optimizer uses real heatmaps.

### Performance
- MotionBERT: identical (17.20→13.00 cm VW-SI-MPJPE raw→optimized)
- MediaPipe: slightly improved (10.45→10.24 cm) with real SH heatmaps

## Phase 2: Camera-Coordinate Evaluation

**Commit**: `a05fe0b` (feat), `ab3926a` (docstring fix)

### What changed
Removed `root_relative()` calls from all evaluation paths in `evaluate.py`, `main.py`, `run_motionbert/__init__.py`, `run_mediapipe/__init__.py`.

### Impact
- Pelvis per-joint error: **0.00 cm → 33.3 cm** (no longer always zero)
- VW-SI-MPJPE: 13.00 → 18.86 cm (optimized), 17.20 → 23.43 cm (raw)
- P-MPJPE unchanged at 42.47 cm (Procrustes handles translation independently, confirming correctness)

The `root_relative()` function is preserved for experiment scripts.

## Phase 3: 2D Reprojected MPJPE

**Commit**: `bdb6497`

### What changed
Added `reprojected_mpjpe_2d()` to `evaluate.py` — projects both predicted and GT to 2D pixels, computes mean distance over visible joints. Integrated as the 10th metric in `evaluate()`.

### Key detail
Uses the same `compute_visibility_weights()` as all other VW metrics (GT projection + frame-boundary check), satisfying the spec's requirement that visibility filtering be "equivalent to keypoint visibility scores."

Values: Det 112.73 px, Opt 84.58 px on the test example (consistent with large 3D error).

## Phase 4: Cross-Pipeline Graphs and Experiments

**Commit**: `f3db7e3` (feat), `40dace4` (fixes)

### Config-controlled graphs (`graphs.py`)
Colors: GT=#4285F4 (Blue), MB Raw=#FFCC80, MB Opt=#FF9800, MP Raw=#EF9A9A, MP Opt=#F44336.

1. `generate_cross_pipeline_per_joint_position` — VW-SI-MPJPE per joint, 4 bars
2. `generate_cross_pipeline_per_joint_velocity` — VW-SI-MPJVE per joint, 4 bars
3. `generate_cross_pipeline_metrics_comparison` — aggregate metrics, 4 bars per group

Guard: only runs when both pipelines present; `logger.warning` otherwise.

### Experiment scripts
- `experiment/elbow_smoothness.py` — 15 lines (5 sources × 3 dimensions), line styles differentiate X/Y/Z
- `experiment/bone_length_variation.py` — 5 lines, confirmed optimized lines have std=0.0000 (perfectly horizontal)

### Per-joint VW-SI metrics
Added `vw_si_mpjpe_per_joint()` and `vw_si_mpjve_per_joint()` to `evaluate.py` to power the cross-pipeline graphs.

---

## Open Items (non-blocking)

From TODO.md, remaining items are all cosmetic/informational:
- `scoring.py:27` comment could be clearer about Nose vs MPII Head Top mapping
- Metrics comparison Y-axis label "Error" is generic (two groups have different units)
- GT bars not in cross-pipeline graphs (spec only calls for 4 bars, not 5)
- 2D-MPJPE values are high on this test example but consistent with 3D error

## Flow / Architecture

```
uv run main.py <config.json>
  ↓
load_config() → RunConfig with optional mediapipe/motionbert/graphs sections
  ↓
For each example:
  1. Load camera calibration (CMU Panoptic)
  2. Extract video frames
  3. Shared YOLO + StackedHourglass → heatmaps, 2D keypoints, visibility
  4. [if motionbert] MotionBERT 3D lifting → camera space → FK optimize → evaluate
  5. [if mediapipe]  MediaPipe 3D → solvePnP → camera space → FK optimize → evaluate
  6. Save results/trajectories/graphs per pipeline per example
  ↓
Cross-pipeline graphs (if both pipelines ran)
Aggregate summaries per pipeline
Run-level results.json
```

Key files:
- `main.py:276` — `main()` function orchestrating everything
- `main.py:83` — `_evaluate_and_collect_metrics()` computing all 10 metrics + per-joint breakdowns
- `evaluate.py:496` — `evaluate()` computing 10 metrics
- `graphs.py:362-532` — cross-pipeline graph functions
- `skeleton.py:1-50` — joint definitions, 15 eval joints

## Spec Issues

None. The spec was clear and internally consistent. The only ambiguity was the phrase "should be equivalent to the keypoint visibility scores" in Phase 3, which we resolved by using the same visibility computation already used by all VW metrics.

## Agent Process Notes

- Phase 1 was the riskiest and largest. Single dev agent handled it well.
- Phases 2-4 were smaller and could run dev + QA in parallel.
- Nathan's nit-picks were almost entirely stale comments (14→15 joints, Head→Nose). These were consistently the most common issue type across all phases — a natural consequence of the joint mapping changes.
- Eve caught the missing run-level `results.json` in Phase 1 which was a real spec violation.
- Total: 4 dev iterations, 4 QA passes, 2 nit-pick passes, 0 rollbacks needed.
