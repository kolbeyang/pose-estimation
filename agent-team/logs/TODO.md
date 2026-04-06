# TODO

## Phase 0 Findings (2026-04-05)

### Critical
- [x] **MotionBERT lower body distortion**: Fixed in Phase 2 via bone length clamping [0.5x, 1.5x] of defaults + increased bone_length_lr. VW-SI-MPJPE improved from 18.86 to 16.83 cm.
- [x] **MediaPipe wrist/ankle regression**: Fixed in Phase 3 via anchor penalty (anchor_weight=1000) anchoring to raw detector positions. VW-SI-MPJPE improved from 15.86 (regression) to 15.40 cm (improvement).

### High Priority
- [x] **Implement blur sigma annealing**: Implemented in Phase 3. Linear interpolation from sigma_start to sigma_end over steps. Used by MotionBERT (16->4).
- [x] **Revert motionbert-single.json**: Reverted sigma=32 to use blur annealing 16->4.
- [x] **Per-pipeline optimization configs**: Implemented in Phase 3. PipelineConfig.optimization dict merged on top of shared config.
- [x] **MediaPipe improvement now 2.7% avg on 5 examples**: Tuned to sigma=4, anchor=200, 50 steps. Validated on 5 examples. All examples positive except none negative with this config.
- [ ] **Blur annealing performance**: Re-blurs all heatmaps via scipy every step during annealing. For 50 steps this adds several seconds. Consider pre-computing a few sigma levels or caching. Verify by timing optimize() with and without annealing.

### Medium Priority
- [ ] **Add root-relative SI-MPJPE as supplementary metric**: Camera-space SI is 26-29% worse than root-relative SI because it must simultaneously fix depth and skeleton scale. Root-relative helps diagnose which is the bottleneck.
- [ ] **Investigate SH heatmap quality for lower body**: MotionBERT knees have 467px 2D reprojection error. Need to visually inspect SH heatmaps for knee channels when legs are partially out of frame.
- [x] **25-example configs updated**: Per-pipeline optimization settings applied to mediapipe-local-25-examples.json and motionbert-local-25-examples.json.
- [x] **Run full 25-example validation**: Completed in Phase 5. 15/25 examples ran (9 failed due to missing/short videos, 1 N/A). Results: MB avg +9.4% improvement (11/15 improved), MP avg +3.0% improvement excluding outlier (14/15 improved). Neither pipeline meets <10 cm target. See REPORT_2026_04_05_PHASE5.md.

### Low Priority
- [ ] **Consider early stopping**: Monitor VW-SI-MPJPE during optimization and stop if it plateaus or regresses.
- [ ] **Audit visibility weights for fairness**: Currently ankles get 0% visibility weight, meaning their massive errors are invisible to the primary metric. This is by design but should be documented.

## Nit-pick Nathan Findings (2026-04-05)

### Critical
- [ ] **Clarify evaluate.py root-relative removal with user**: The `evaluate()` function was changed from root-relative to camera-space in commit `a05fe0b`, which increased all metric values by 19-144%. This changes what "VW-SI-MPJPE < 10 cm" means relative to the spec. The user should be informed and the <10 cm target should be re-calibrated. Verify by comparing old root-relative VW-SI-MPJPE (e.g., MB det was 17.20 cm) to new camera-space (23.43 cm) and confirming the user accepts the new baseline.
- [x] **Run full 25-example validation**: Completed in Phase 5. 15/25 examples produced metrics. MB: +9.4% avg improvement. MP: +3.0% avg improvement (excl 113cm outlier). See REPORT_2026_04_05_PHASE5.md.

### High Priority
- [ ] **Missing SH heatmap quality audit for knees**: The PLAN called for `experiment/audit_heatmap_quality.py` to inspect MPII knee heatmap channels, but this script was never created. The knee investigation answered "why are bone lengths wrong?" but not "are there good heatmaps for the knees?" Verify by creating the script and inspecting knee heatmap channels (MPII 1=RKnee, 4=LKnee) for peak presence and location accuracy.
- [ ] **Testing report contains false claim**: TESTING_REPORT_2026_04_05_PHASE1_3.md line 49 says "Evaluation code was not modified, raw metrics are identical" but evaluate.py WAS modified (root-relative removal). The raw VW-SI-MPJPE changed from 17.20 to 23.43 cm for MotionBERT. Correct the report. Verify by reading the corrected report.

### Medium Priority
- [ ] **Experiment reports do not follow spec format**: The spec requires each experiment to have: (1) Motivating observation, (2) Hypothesis, (3) Verification method, (4) Verification results, (5) Solution, (6) Solution results. Current reports use ad-hoc formats. Restructure reports or create a FINAL_REPORT that follows the format. Verify by checking each experiment section has all 6 elements.
- [ ] **Stale experiment scripts use root-relative metrics**: `experiment/joint_speed_analysis.py`, `experiment/test_si_consistency.py`, and `experiment/visibility_weight_investigation.py` call `root_relative()` before computing metrics, while `evaluate()` no longer does. These scripts compute different metric values than the main pipeline. Update to use camera-space or add a flag. Verify by running a script and comparing its output to results.json for the same data.
- [ ] **vw_si_mpjpe_per_joint fallback for invisible joints is misleading**: When a joint has zero visibility, the per-joint function returns unweighted mean error, but the aggregate VW-SI-MPJPE excludes that joint entirely. Consider returning NaN or 0.0 for invisible joints. Verify by checking per_joint values in results.json for joints with 0 visible frames.
- [ ] **Clarify spec typo VW-SW-MPJPE vs VW-SI-MPJPE**: Spec line 62 says "VW-SW-MPJPE" which is likely a typo for "VW-SI-MPJPE". Confirm with user.

### Low Priority
- [ ] **Shared optimization config in 25-example JSONs is misleading**: `motionbert-local-25-examples.json` has a shared optimization section with stale defaults (num_steps=10, sigma=16). The per-pipeline override takes precedence, but removing the override would silently degrade. Clean up or add a comment. Verify by inspecting the JSON and confirming shared section matches intended defaults.
- [ ] **No MotionBERT multi-example sweep script**: `motionbert_sweep.py` only tests on 1 example. The 5-example validation for MotionBERT was done manually, not via a reusable sweep script like `mediapipe_multi_sweep.py`. Consider adding `motionbert_multi_sweep.py`. Verify by checking the script exists and runs on multiple examples.
