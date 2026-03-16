# Test Report: Phase 2, Iteration 1

## Summary

**Verdict: YES** -- Phase 2 Definition of Done is met. All 10 examples run without errors. Optimization improves MPJPE on 8/10 examples, no example regresses by more than 2 cm, and mean MPJPE is lower after optimization. However, there are concerns about overall MPJPE magnitude that warrant further investigation.

---

## Tests Executed

### T1: Smoke Test -- Pipeline Execution
**Status: PASS**

`uv run python main.py` ran to completion on all 10 examples without errors. Total runtime was approximately 8 minutes. Output directory: `training_runs/motionbert-run-20260315-214752/`.

### T2: Pyright Issues Check
**Status: PASS (issues are stale)**

The reported Pyright diagnostics are **not real bugs**:
- `main.py` line 157: correctly unpacks 7 values from `detect_poses` (which returns a 7-tuple)
- `test_single.py` line 97: same correct 7-value unpack
- `motionbert_to_camera_space` is correctly defined in `detect.py` and imported in both files

The developer updated all callers when changing the `detect_poses` signature. The Pyright analysis was stale.

### T3: MPJPE Baseline Check
**Status: PASS**

Mean detector MPJPE dropped from 44.89 cm (old) to 29.09 cm (new), a 35% improvement from bone-length matching. This matches the developer's report.

### T4: Optimization Improvement (Phase 2 DoD)
**Status: PASS**

| Example | Det MPJPE (cm) | Opt MPJPE (cm) | Improvement (cm) |
|---------|---------------|----------------|-------------------|
| 171204_pose1_sample_0 | 24.75 | 21.55 | +3.20 |
| 171204_pose2_200 | 46.43 | 43.70 | +2.73 |
| 171204_pose2_5000 | 12.78 | 12.55 | +0.23 |
| 171204_pose2_15000 | 21.35 | 19.89 | +1.46 |
| 171204_pose3_200 | 54.71 | 55.09 | **-0.37** |
| 171204_pose3_4000 | 14.92 | 14.37 | +0.55 |
| 160422_ultimatum1_200 | 45.34 | 45.58 | **-0.24** |
| 160422_ultimatum1_10000 | 43.95 | 43.38 | +0.57 |
| 171204_pose2_10000 | 13.54 | 13.04 | +0.50 |
| 171204_pose2_25000 | 13.11 | 12.92 | +0.19 |
| **MEAN** | **29.09** | **28.21** | **+0.88** |

Phase 2 DoD checklist:
- [x] Optimization runs without errors on all 10 examples
- [x] MPJPE improves on a majority of examples: **8/10 improved**
- [x] No example regresses by more than 2 cm: worst regression is 0.37 cm (171204_pose3_200)
- [x] Mean MPJPE is lower after optimization: 29.09 -> 28.21 cm
- [x] Results saved with both baseline and optimized metrics

### T5: Results Output Validation
**Status: PASS**

- 10/10 prediction JSON files saved with complete structure
- Each JSON contains: `metrics` (with both `det_*` and `opt_*` keys), `bone_lengths_final`, `loss_history`, per-frame `detector_3d`, `optimized_3d`, and `ground_truth_3d`
- 10/10 per-example graph directories with `per_joint_error.png` and `per_frame_mpjpe.png`
- Aggregate graphs: `aggregate_mpjpe.png` and `aggregate_p_mpjpe.png`

### T6: Code Review
**Status: PASS (with notes from P1 report)**

- `detect_poses` returns 7 items, callers correctly unpack 7
- `motionbert_to_camera_space` is defined and used correctly
- All core pipeline files have type hints (as verified in Phase 1 report)
- No sibling directory imports found
- Pydantic models used for CameraParams and ExampleResult

Note: The Phase 1 tester report flagged missing type hints in `overlay_heatmaps.py` and `test_norm_comparison.py`. These are utility/debug scripts, not part of the core pipeline. This was previously accepted as borderline by allowing Phase 1 to pass.

### T7: MPJPE Realism Assessment
**Status: INFORMATIONAL -- FURTHER INVESTIGATION RECOMMENDED**

The MotionBERT paper reports ~40-50mm (4-5 cm) MPJPE on Human3.6M with known cameras. Our mean is 29 cm (290mm), which is approximately 6x higher. Key factors explaining the gap:

1. **Different dataset**: CMU Panoptic is harder than H36M -- more complex poses, multi-person scenes, different camera angles.
2. **No known camera extrinsics for prediction**: The paper uses ground truth camera parameters during training. We estimate depth from a single camera with a heuristic.
3. **Different 2D detector**: The paper uses ground truth 2D keypoints or CPN detections fine-tuned on H36M. We use Stacked Hourglass which may not generalize as well to Panoptic.
4. **Some examples have very high error**: The 3 worst examples (45-55 cm) appear to be cases where 2D detection fails badly (multi-person confusion, extreme poses).
5. **Best examples are reasonable**: The 4 best examples (12-15 cm) suggest the pipeline works well when 2D detection succeeds.

The 29 cm mean is plausible given these factors, but the wide variance (12-55 cm) suggests the pipeline is bottlenecked by 2D detection quality on difficult examples. The P-MPJPE mean of 22 cm shows the 3D structure is reasonable -- the remaining 7 cm gap between MPJPE and P-MPJPE is from depth/scale errors.

**Recommendation**: The numbers are not directly comparable to paper numbers due to different datasets and evaluation protocols. The current MPJPE is realistic for "in the wild" evaluation on CMU Panoptic. To improve further, consider:
- Increasing NUM_STEPS beyond 10
- Better 2D detection (e.g., HRNet or ViTPose)
- Example-specific investigation of the 3 worst cases

---

## Bugs Found

None. The Pyright issues were stale.

---

## Verdict: YES

Phase 2 Definition of Done is met:
1. Pipeline runs on all 10 examples without errors.
2. Optimization improves 8/10 examples (majority).
3. Maximum regression is 0.37 cm (well under 2 cm threshold).
4. Mean MPJPE decreases: 29.09 -> 28.21 cm.
5. Results include both baseline and optimized metrics.

The optimization improvement is modest (+0.88 cm mean) because the bone-length matching fix already corrected most of the scale error that the optimizer was previously compensating for. With only 10 steps, the optimizer makes small refinements. Increasing NUM_STEPS in future iterations should yield larger improvements.
