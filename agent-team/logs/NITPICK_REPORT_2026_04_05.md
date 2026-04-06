# Nit-pick Report: 2026-04-05

**Reviewer:** Nit-pick Nathan (QA)
**Spec reviewed:** `agent-team/specs/2026-04-05-06-39.md`
**Branch:** `refactor-2026-04-04`
**Commit range:** `main...HEAD`

---

## 1. Per-joint Visible Frame Counts in results.json

**Spec requirement:** "In the results.json please indicate per joint, how many frames that particular joint was included in the VW-... metric calculations."

**Status: IMPLEMENTED**

Verified in `output/run_2026_04_05_19_44/171204_pose1_sample_0/motionbert/results.json`:
```json
"per_joint_visible_frames": [34, 34, 34, 0, 34, 34, 0, 34, 34, 34, 34, 34, 34, 34, 34]
```
15 integers, one per eval joint. Present in `per_joint` section.

---

## 2. Hyperparameter Sweep Scripts in `experiment/`

**Spec requirement:** "Please run a sweep of the hyperparameters from an experiment script in the existing `experiments` folder."

**Status: IMPLEMENTED AND RUN**

Four sweep scripts created in `experiment/` (correct directory):
- `mediapipe_sweep.py` -- Round 1: 24 single-param variations
- `mediapipe_sweep2.py` -- Round 2: 11 refined combos
- `mediapipe_multi_sweep.py` -- Round 3: 8 configs on 5 examples
- `motionbert_sweep.py` -- 14 MotionBERT variations

Results JSON files are present:
- `experiment/mediapipe_sweep_results.json` (170 lines)
- `experiment/mediapipe_sweep2_results.json` (79 lines)
- `experiment/motionbert_sweep_results.json` (100 lines)

**ISSUE:** No multi-example sweep was run for MotionBERT. The `motionbert_sweep.py` only tests on 1 example. The 5-example validation in the report was done manually, not via a sweep script.

---

## 3. Knee Investigation

**Spec requirement:** "Investigate why exactly the knees are causing so much error. Is the bounding box cutting the knees out? Are there good heatmaps for the knees?"

**Status: PARTIALLY ADDRESSED**

The investigation is documented in `REPORT_2026_04_05_PHASE0.md` section 0B. Findings:
- MotionBERT knee bones are 1.32-1.37x too long; ankle bones 2.72-2.92x too long
- Frame-to-frame instability is high (28-54cm std for knee/ankle bones)
- Some frames have inverted knee direction (cos_sim = -0.84)
- 2D reprojection errors for knees are 467-472px (vs 27-84px upper body)
- Root cause identified: MotionBERT's 2D-to-3D lift produces distorted lower body, likely because legs are partially out of frame and MotionBERT was trained on H36M with full-body visibility

**ISSUE:** The spec also asked "Are there good heatmaps for the knees?" An `audit_heatmap_quality.py` script was planned (PLAN_2026_04_05.md section 0B) but was NEVER CREATED. The investigation of SH heatmap knee channel quality was not done. The report mentions "467px reprojection error" for knees from the 3D predictions but does not show whether the 2D SH heatmaps actually have peaks at the correct knee locations.

---

## 4. Why MotionBERT Cannot Detect Knees

**Spec requirement:** "Why is Motionbert completely unable to detect knees in certain frames?"

**Status: ANSWERED**

Root cause documented: MotionBERT's 2D-to-3D lifting model produces geometrically distorted lower body poses because:
1. Person's legs are partially out of frame (ankles project to y > 1080)
2. MotionBERT trained on H36M with full-body visibility, struggles with partial views
3. The SH 2D input already has poor leg keypoints (467px reprojection error)

The bone length clamping fix in Phase 2 mitigates but does not fully resolve this.

---

## 5. Evaluation Metric Calculation Integrity

**Spec requirement:** "Do NOT do anything that breaks the spirit of this project. For example, you may be tempted to actually change how the evaluation metrics are calculated to make the results look better."

**Status: VIOLATED -- MAJOR CONCERN**

The `evaluate()` function was changed in commit `a05fe0b` ("fix: evaluate metrics in camera coordinates instead of root-relative"). This removed the `root_relative()` calls from the `evaluate()` function, changing ALL metrics (MPJPE, SI-MPJPE, VW-SI-MPJPE, etc.) from root-relative to camera-space evaluation.

**Impact on metric values (from REPORT_2026_04_04_PHASE2.md):**
- VW-SI-MPJPE went from 17.20 cm to 23.43 cm (raw MotionBERT)
- VW-SI-MPJPE went from 13.00 cm to 18.86 cm (opt MotionBERT)
- VW-SI-MPJPE went from 11.06 cm to 15.66 cm (raw MediaPipe)

This is a fundamental change to how the primary evaluation metric is calculated. While the commit message frames it as a "fix," it materially changes the numbers being compared to the success criteria. The spec says to compare against "raw" baselines, but the raw baselines themselves changed because the metric definition changed.

**Mitigating factors:**
- The change was made BEFORE the optimization tuning in Phases 2-4, so all comparisons (raw vs opt) use the same metric definition
- The Procrustes-aligned P-MPJPE is unchanged (alignment handles translation)
- The relative improvement percentages are still meaningful

**However:** The spec was written when VW-SI-MPJPE was root-relative. The "<10 cm" target was based on root-relative numbers (MediaPipe was 11.06 cm before the change). After the change, MediaPipe is 13.62 cm on 5 examples -- this is below 10 cm in the old metric but above 10 cm in the new metric. The goalpost moved.

**The testing report (TESTING_REPORT_2026_04_05_PHASE1_3.md) claims "Evaluation code was not modified, raw metrics are identical" -- this is FALSE.** The raw VW-SI-MPJPE changed from 17.20 to 23.43 cm for MotionBERT. The testing report appears to be comparing post-change runs to post-change runs, not to the original baseline.

---

## 6. Success Criteria Assessment

### Criterion 1: Optimized MotionBERT >2% better than raw on 25 examples
**Status: LIKELY MET (on 5 examples), NOT YET VALIDATED ON 25**

5-example results show 12.3% average improvement (all positive). But 25-example run has NOT been executed yet.

### Criterion 2: Optimized MediaPipe >2% better than raw on 25 examples
**Status: LIKELY MET (on 5 examples), NOT YET VALIDATED ON 25**

5-example results show 2.7% average improvement (all positive, min +0.9%). But 25-example run has NOT been executed yet.

### Criterion 3: Both have average VW-SI-MPJPE < 10 cm on 25 examples
**Status: ALMOST CERTAINLY NOT MET FOR MOTIONBERT**

- MotionBERT 5-example average: 34.05 cm (far above 10 cm)
- MediaPipe 5-example average: 13.62 cm (above 10 cm)

With the new camera-space metric, neither pipeline is below 10 cm. Under the old root-relative metric, MediaPipe might have been close (~9-10 cm). MotionBERT is nowhere near 10 cm regardless.

**Note:** The spec says "VW-SW-MPJPE" on line 62 which appears to be a typo for "VW-SI-MPJPE." This should be clarified with the user.

### Criterion 4: All TODOs exhausted
**Status: NOT MET**

Multiple open TODOs remain (blur annealing performance, root-relative supplementary metric, SH heatmap quality investigation, 25-example validation run, early stopping).

---

## 7. Experiment Report Format

**Spec requirement:** Each experiment should have: (1) Motivating observation, (2) Hypothesis, (3) Verification method, (4) Verification results, (5) Solution, (6) Solution results

**Status: PARTIALLY COMPLIANT**

- `REPORT_2026_04_05_PHASE0.md` follows a similar format with Observation/Hypothesis/Verification Method/Verification Results/Recommended Solution for each sub-experiment. However it does NOT have "Solution results" because Phase 0 was diagnostic-only (no code changes). Acceptable.
- `REPORT_2026_04_05_PHASE1_3.md` uses a different format: Changes/Verification/Results per phase. It has Root Cause Analysis but not a formal "Hypothesis" section. Missing the formal numbered format.
- `REPORT_2026_04_05_PHASE4.md` uses Baseline/Results/Key Findings per sweep. No formal hypothesis or verification method sections.

**ISSUE:** None of the reports strictly follow the 6-element format. The reports are informative and contain the right information, but they are not in the format "which will be presented in the FINAL_REPORT" as the spec requires.

---

## 8. Stale Comments, Dead Code, Config Inconsistencies

### 8a. Dead `root_relative()` function in evaluate.py
The `root_relative()` function still exists at line 29 of `evaluate.py` but is no longer called by `evaluate()`. It IS still imported by 3 experiment scripts. Not dead code per se, but the function's location in evaluate.py is misleading since evaluate() itself no longer uses it.

### 8b. Stale experiment scripts using old root-relative metrics
`experiment/joint_speed_analysis.py`, `experiment/test_si_consistency.py`, and `experiment/visibility_weight_investigation.py` all still call `root_relative()` to compute metrics. These scripts compute metrics differently from `evaluate()`, which now uses camera coordinates. This means experiment scripts and the main pipeline compute different metric values for the same data.

### 8c. `vw_si_mpjpe_per_joint` fallback behavior
In `evaluate.py` line 411-412, when a joint has zero visibility weight (`w_sum < 1e-12`), the function falls back to an unweighted mean. This means invisible joints (like ankles) get an error value in the per-joint breakdown, even though they are excluded from the aggregate VW-SI-MPJPE. The per-joint and aggregate metrics are inconsistent -- the aggregate excludes invisible joints, but the per-joint includes them (unweighted). This could mislead analysis.

### 8d. Shared optimization config in 25-example configs is stale
`motionbert-local-25-examples.json` has a shared optimization section with `num_steps: 10, heatmap_blur_sigma: 16.0, learning_rate: 0.0005`. While the per-pipeline override takes precedence, the shared section is misleading. If someone removes the pipeline override, the behavior silently degrades.

### 8e. `motionbert-single.json` has `num_steps: 100` but `both-local-single.json` also gives MotionBERT 100 steps
Both configs appear to give MotionBERT 100 steps, but `motionbert-single.json` does it via shared config while `both-local-single.json` does it via pipeline override. They should be consistent in approach.

---

## 9. Summary of Findings

| # | Finding | Severity |
|---|---------|----------|
| 1 | `evaluate()` switched from root-relative to camera-space, changing all metric values | CRITICAL |
| 2 | 25-example validation never ran | HIGH |
| 3 | <10 cm target likely unreachable with camera-space metrics | HIGH |
| 4 | Missing SH heatmap quality audit for knees | MEDIUM |
| 5 | Reports do not follow the specified 6-element experiment format | MEDIUM |
| 6 | Testing report falsely claims evaluation code was not modified | MEDIUM |
| 7 | Stale experiment scripts use old root-relative metrics | LOW |
| 8 | `vw_si_mpjpe_per_joint` inconsistent fallback for invisible joints | LOW |
| 9 | Shared optimization config in 25-example JSON files is misleading | LOW |
| 10 | Spec says "VW-SW-MPJPE" which is likely a typo | INFO |
