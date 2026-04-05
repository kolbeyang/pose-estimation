# Nit-Pick Report: SPEC Compliance Review

**Date**: 2026-03-30 23:00
**Reviewer**: Nit-pick Nathan (Claude Opus 4.6)
**Reviewing**: Dan's REPORT_2026_03_30_22_30.md against SPEC 2026-03-30-08-35.md

---

## 1. Were all 3 hypotheses investigated?

| Hypothesis | Investigated? | Notes |
|---|---|---|
| H1: Over-smoothing (rotation penalty too strong) | PARTIAL | Dan tested rotation_penalty_scalar sweep but did NOT perform the SPEC's specific verification: "compare average joint speeds (GT vs raw vs optimized)" for wrists/elbows. The SPEC explicitly asked for per-joint speed comparison to diagnose over-smoothing. Dan only checked aggregate VW-SI-MPJPE, which is a different signal. |
| H2: SI-MPJPE metric mismatch | YES | Tests written in `experiment/test_si_consistency.py`. All 4 test types passed on 3 examples. |
| H3: Visibility weighting logic incorrect | NO | Not mentioned anywhere in the report. No investigation, no tests, no discussion. Completely skipped. |

**Verdict**: 1 of 3 fully investigated, 1 partially, 1 entirely skipped.

---

## 2. Did the report follow the required experiment format?

The SPEC requires each experiment to have:
1. (optional) Motivating observation
2. Hypothesis
3. Verification method
4. Verification results
5. Solution
6. Solution results

| Experiment | Format compliant? | Missing |
|---|---|---|
| Exp 1: Rotation penalty sweep | PARTIAL | Has observation, hypothesis, verification method, verification results. Missing explicit "Solution" and "Solution results" sections -- just a conclusion that it didn't work. Acceptable since negative result. |
| Exp 2: Num steps sweep | YES | All sections present. |
| Exp 3: SI-MPJPE consistency | PARTIAL | Has hypothesis, verification method, results. Missing "Solution" and "Solution results" -- acceptable since tests passed and no fix was needed. |
| Visibility weighting | MISSING | No experiment at all. |

---

## 3. Was the 2% target achieved?

YES. Dan reports +2.3% improvement on VW-SI-MPJPE with 15/25 examples improved for MotionBert at num_steps=5. This exceeds the 2% target.

---

## 4. SPEC requirement: "doing explicitly what I say is insufficient. You should also systematically investigate..."

Dan did go beyond the 3 hypotheses by discovering `num_steps` as the dominant factor. This is good systematic investigation -- he tested rotation penalty first, found it insufficient, then hypothesized that step count itself was the problem, and confirmed it with a grid sweep.

However, the investigation was narrow in scope:
- Only two variables were explored (rotation_penalty_scalar, num_steps).
- No per-joint analysis was done (which joints are regressing, which are improving).
- No analysis of what the optimizer is actually doing wrong at high step counts (is it the heatmap loss? the smoothing loss? bone length loss?).
- The root cause explanation ("too many steps = too much damage") is somewhat surface-level. Why does the optimizer damage poses? Which loss term is responsible?

---

## 5. SPEC-specific verification requirements

### "verify by comparing average joint speeds (GT vs raw vs optimized)"

NOT DONE. The SPEC explicitly asked Dan to calculate average speeds of wrists and elbows across all frames for GT, raw MotionBert, and optimized output. This was never performed. The report contains no per-joint velocity analysis at all.

### "write tests" for SI-MPJPE consistency

DONE. `experiment/test_si_consistency.py` contains 4 test functions run on 3 examples (12 total checks). The tests verify:
1. Scale factor minimizes MPJPE (checked with delta perturbations)
2. VW-SI-MPJPE equals VW-MPJPE after scaling
3. Scale=1.0 >= optimal
4. If MPJPE improves, SI-MPJPE shouldn't get worse

These tests are reasonable and match the SPEC's intent. One gap: The SPEC said "pass values through both to the standard MPJPE calculation and the SI-MPJPE calculation" and "make sure the function being optimized inside the scale factor finding function is reused." Test 1 does verify the scale factor is optimal, and Test 2 verifies consistency. However, there is no test that explicitly checks code reuse between the optimization and evaluation paths.

### "VW visibility projection logic"

NOT INVESTIGATED. The SPEC listed Hypothesis 3 about visibility weighting logic being incorrect. Dan did not:
- Inspect the `compute_visibility_weights` function
- Test whether visibility weights are correct (e.g., do visible joints match what's actually in-frame?)
- Check if the projection logic handles edge cases properly
- Investigate why VW metrics perform worse than non-VW metrics

---

## 6. Test quality review: `experiment/test_si_consistency.py`

Concerns:
- **Hardcoded output directory**: Line 37 hardcodes `output/motionbert_2026_03_30_18_57`. This will break if someone runs the tests later without that specific output directory. Should accept a CLI arg or auto-detect the latest run.
- **Not a proper test framework**: Uses ad-hoc pass/fail rather than pytest assertions. The SPEC said "write tests" which implies something runnable and reusable.
- **Only tests 3 examples**: Should ideally test all 25 for confidence, or at minimum test edge cases (e.g., examples where many joints are out of frame).
- **No negative tests**: All tests check the "happy path." No tests for degenerate inputs (e.g., single frame, all joints invisible, prediction far from GT).

---

## 7. Summary of issues found

### Critical (SPEC requirements not met)
1. Hypothesis 3 (visibility weighting) was completely skipped.
2. The SPEC's specific joint speed comparison (GT vs raw vs optimized for wrists/elbows) was never performed.

### Moderate
3. No per-joint breakdown of which joints are regressing and which are improving -- the SPEC's Observation 1 specifically called out elbows, wrists, and knees.
4. Test file hardcodes an output directory path, making it non-portable.

### Minor
5. The solution (num_steps=5) is effective but the root cause is still unclear. Why does the optimizer damage MotionBert poses after 5 steps? Which loss component is responsible? This should be understood to prevent recurrence.
6. No analysis of whether the num_steps=5 fix also helps MediaPipe or whether MediaPipe was tested with lower step counts.

---

## Final Assessment

Dan achieved the 2% target, which is the bottom-line success criterion. The `num_steps` discovery was a good systematic finding. However, two of the three SPEC hypotheses were not properly investigated (H1 was only partially addressed, H3 was skipped entirely), and the SPEC's most specific verification request (joint speed comparison) was not performed. The work prioritized finding a quick fix over thorough investigation.
