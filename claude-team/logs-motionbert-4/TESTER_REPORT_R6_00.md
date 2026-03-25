# Tester Report - Round 6, Iteration 0

## Summary

**VERDICT: SPEC NOT MET.** No value of `rotation_penalty_scalar` from 1 to 1000 achieves Opt MPJVE < Det MPJVE on either example. The rotation penalty does not address the root cause of the MPJVE regression. Phase B (fine sweep) was skipped — there is no promising coarse region to refine. Phase C (config update) was not performed — no improvement was found.

---

## Step 1: Code Edit

Changed `num_steps=100` → `num_steps=50` in both `get_round6_configs()` and `get_round6_fine_configs()` in `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/sweep.py`. Verified by reading both functions after edit.

---

## Smoke Test

```
uv run python sweep.py --help
```

Output confirmed `round6` and `round6-fine` are valid phase choices. CLI is correct.

---

## Phase A: Coarse Sweep Results

Command run:
```
uv run python sweep.py --phase round6 --examples 0,5 2>&1
```

Completed without errors. Both JSON files written to `training_runs/sweep_results/`.

### Example 0 (171204_pose1_sample_0) — target: Opt MPJVE < 0.94 cm/f

Det MPJPE = 30.98 cm, Det MPJVE = **0.94 cm/f**

| Config    | Opt MPJPE | Improv | Opt MPJVE | PASS MPJVE? | PASS MPJPE? |
|-----------|-----------|--------|-----------|-------------|-------------|
| rot_s=1   | 31.10     | -0.12  | 1.70      | NO          | NO          |
| rot_s=5   | 31.08     | -0.10  | 1.57      | NO          | NO          |
| rot_s=10  | 31.08     | -0.10  | 1.52      | NO          | NO          |
| rot_s=50  | 31.08     | -0.10  | 1.46      | NO          | NO          |
| rot_s=100 | 31.08     | -0.10  | 1.46      | NO          | NO          |
| rot_s=200 | 31.06     | -0.08  | 1.47      | NO          | NO          |
| rot_s=500 | 31.02     | -0.04  | 1.46      | NO          | NO          |
| rot_s=1000| 31.02     | -0.04  | 1.47      | NO          | NO          |

Best MPJVE: 1.46 cm/f at rot_s=50, 100, 500. Target: < 0.94 cm/f. Miss by 0.52 cm/f (55%).

### Example 5 (171204_pose3_4000) — target: Opt MPJVE < 0.51 cm/f

Det MPJPE = 15.85 cm, Det MPJVE = **0.51 cm/f**

| Config    | Opt MPJPE | Improv | Opt MPJVE | PASS MPJVE? | PASS MPJPE? |
|-----------|-----------|--------|-----------|-------------|-------------|
| rot_s=1   | 15.71     | +0.14  | 0.60      | NO          | YES         |
| rot_s=5   | 15.67     | +0.19  | 0.62      | NO          | YES         |
| rot_s=10  | 15.66     | +0.19  | 0.61      | NO          | YES         |
| rot_s=50  | 15.64     | +0.21  | 0.60      | NO          | YES         |
| rot_s=100 | 15.64     | +0.22  | 0.59      | NO          | YES         |
| rot_s=200 | 15.64     | +0.22  | 0.59      | NO          | YES         |
| rot_s=500 | 15.65     | +0.21  | 0.60      | NO          | YES         |
| rot_s=1000| 15.67     | +0.19  | 0.62      | NO          | YES         |

Best MPJVE: 0.59 cm/f at rot_s=100, 200. Target: < 0.51 cm/f. Miss by 0.08 cm/f (16%).

---

## Phase B Decision

**Phase B was not run.** There is no "promising region" to fine-tune. The coarse sweep over 3 orders of magnitude (rot_s=1 to rot_s=1000) shows:
- Example 0 MPJVE is stuck in the range 1.46–1.70 cm/f regardless of rot_s. The penalty reduces MPJVE slightly (1.70 → 1.46) but is nowhere near the 0.94 target.
- Example 5 MPJVE is stuck in the range 0.59–0.62 cm/f. The minimum is 0.59 at rot_s=100-200, but the target is 0.51.

A fine sweep around any single rot_s value cannot bridge these gaps.

---

## Phase C Decision

**Phase C was not run.** No improved configuration was found to update `config.py` defaults with.

---

## Root Cause Analysis

### Why does MPJVE not improve?

1. **The MPJVE problem is structural, not a penalty-tuning issue.** The optimization improves heatmap fit (Opt 2D-Det error drops from 34.3px to ~29.9-32.2px for Example 0, showing the optimizer IS doing something), but it introduces frame-to-frame jitter as a side effect of per-frame independent optimization. No amount of rotation penalty can suppress this because:
   - The rotation penalty penalizes large angle changes between consecutive frames
   - But the optimizer still finds local minima that fit the heatmaps with jittery pose sequences
   - The MPJVE floor at ~1.46 cm/f on Example 0 represents this local-minima jitter

2. **On Example 0, both MPJPE and MPJVE are worse than the detector.** This is the worst case: the optimizer is actively harming both metrics. Example 0 is a 100-frame sequence from `171204_pose1_sample` — it may have different characteristics (shorter, different motion type).

3. **The rotation penalty approaches a plateau.** On Example 0, MPJVE drops from 1.70 (rot_s=1) to 1.46 (rot_s=50) and then is flat from rot_s=50 to rot_s=1000. This plateau means there is no benefit to higher penalties. Similarly on Example 5, MPJVE plateaus at ~0.59 from rot_s=50 onward.

4. **The INIT_ANCHOR_WEIGHT (=5.0) may be insufficient to prevent temporal jitter.** The anchor term penalizes deviation from MotionBERT's per-frame predictions but does not penalize inter-frame differences. Consider: if MotionBERT's per-frame predictions have low MPJVE (0.94, 0.51 cm/f), but the optimizer deviates independently per frame while still staying near the initialization, it can still produce higher MPJVE.

---

## Bugs Found

### Bug 1: MPJVE cannot be fixed by rotation penalty alone (architecture-level)

- **Description:** The per-joint rotation penalty (penalizing angle changes between consecutive frames) is insufficient to suppress temporal jitter below the detector baseline. The root cause appears to be that the optimization problem has many per-frame local minima that all fit the heatmaps approximately equally well, but result in jittery trajectories.
- **How observed:** Phase A coarse sweep, 8 configs x 2 examples, all failing the Opt MPJVE < Det MPJVE criterion.
- **Severity:** Blocks the spec's definition of done (primary criterion not met).
- **Suggested fix direction:** Consider one or more of:
  a. **Stronger temporal smoothness via direct velocity penalty on joint positions** (not rotation angles). The current rotation penalty operates in angle space; a position-space velocity penalty would directly target MPJVE.
  b. **Increase INIT_ANCHOR_WEIGHT substantially** to prevent the optimizer from deviating far from MotionBERT's temporally smooth predictions.
  c. **Joint trajectory smoothing post-optimization** (e.g., Gaussian smoothing of the final 3D trajectory). This is a post-hoc fix but would directly reduce MPJVE.
  d. **Multi-frame joint optimization** (batch all frames together and add explicit position velocity penalty `||x[t+1] - x[t]||^2`). The current rotation penalty in angle space is an indirect proxy.

### Bug 2: Example 0 MPJPE regresses (secondary)

- **Description:** For Example 0, all rotation scalars produce Opt MPJPE slightly worse than Det MPJPE (improvement_cm is negative, ranging from -0.04 to -0.12 cm). Example 5 is fine (positive improvements).
- **How observed:** Phase A coarse sweep, Example 0 column `Improv` is negative for all 8 configs.
- **Severity:** Secondary criterion (do not regress MPJPE) is failing for Example 0.
- **Suggested fix direction:** This may be inherent to Example 0's data characteristics. The sequence is shorter (100 frames vs 150) and uses a different source (`171204_pose1_sample`). Could try lower INIT_ANCHOR_WEIGHT to give the optimizer more freedom, but that risks worsening MPJVE further.

---

## Code Review Findings

### Finding 1: No structural issues introduced by the developer

The Round 6 developer changes to `sweep.py` are correct:
- `det_mpjve_cm` is present in both the JSON output and the printed table (verified in actual output).
- `det_mpjve_cm` is constant across all configs for the same example (0.9361 for Example 0, 0.5100 for Example 5) — confirms detection is not affected by sweep configs.
- Multi-example `--examples` arg works correctly.
- `num_steps=50` edit was applied cleanly to both `get_round6_configs()` and `get_round6_fine_configs()`.

### Finding 2: Position penalty has near-zero contribution in Example 5

Looking at the loss breakdown for Example 5 (from console output): `pos_p` drops from ~0.025 at step 0 to ~0.0005 at step 49. The position penalty converges to near zero quickly — meaning it is not actually constraining anything meaningfully. This is consistent with the MPJVE not being affected by it.

### Finding 3: Rotation penalty is the dominant loss term but does not translate to MPJVE improvement

For rot_s=1000, Example 5: total loss is ~5199, heatmap term is ~-2888, rotation penalty ~2310. The rotation penalty is massive (nearly as large as the heatmap term) yet MPJVE is still worse than the detector. This confirms that penalizing rotation changes does not directly suppress MPJVE in this formulation.

---

## Critical Decisions for User Review

1. **Phase B skipped.** With no config achieving Opt MPJVE < Det MPJVE at any coarse rotation scalar, fine-tuning around any particular value would not cross the threshold. Proceeding with Phase B would waste runtime with no expected benefit.

2. **config.py NOT updated.** Since no improvement was found, the defaults remain at `ROTATION_PENALTY_SCALAR=10.0` and `POSITION_PENALTY_WEIGHT=50.0`. Updating them to any value from this sweep would make things worse or identical.

3. **The spec's goal may require a different approach.** The rotation penalty targets angle-space smoothness, not position-space velocity. Since MPJVE is defined as mean per-joint velocity error in 3D position space (cm/frame), the most direct fix would be a penalty on `||positions[t+1] - positions[t]||` computed after FK, not on `||rotations[t+1] - rotations[t]||`. These are related but not equivalent, especially when some motion comes from root translation rather than joint rotation.

4. **Example 0 is particularly problematic.** It is the only example where MPJPE also regresses. This may be a data quality issue (the `_sample` sequence), a shorter-sequence effect, or something specific to this person/motion. If Example 0 is not representative of the production use case, it may be acceptable to deprioritize it in favor of the other 9 examples.

---

## Tests Run Summary

| Test | Command | Result |
|------|---------|--------|
| Step 1 code edit | Manual edit, verified by Read | PASS |
| Smoke test | `uv run python sweep.py --help` | PASS |
| Phase A coarse sweep | `uv run python sweep.py --phase round6 --examples 0,5` | COMPLETE — no configs meet target |
| JSON output check | Read sweep_results/*.json | PASS — `det_mpjve_cm` present, correct values |
| Regression: det metrics constant | Verified in JSON | PASS — det_mpjve/det_mpjpe identical across all configs |
| Phase B | Skipped | N/A (no promising region found) |
| Phase C | Skipped | N/A (no best config to update) |
