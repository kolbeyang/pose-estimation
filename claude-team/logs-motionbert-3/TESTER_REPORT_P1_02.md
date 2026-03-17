# Tester Report: Phase 1, Iteration 2

**Date:** 2026-03-17
**Plan:** `claude-team/logs/TESTER_PLAN_P1_02.md`
**Spec:** `claude-team/specs/motion-bert-round-5.md` (Phase 1)
**Developer Report:** `claude-team/logs/DEVELOPER_REPORT_P1_02.md`

---

## Tests Run

### T1: Config constants removed -- PASS

**Command:**
```python
import config as cfg
for attr in ['SIGMA_SCHEDULE', 'HEATMAP_BLUR_SCHEDULE', 'FK_TARGET_CONF_THRESHOLD', 'USE_REAL_HEATMAPS']:
    assert not hasattr(cfg, attr)
```
**Result:** All 4 constants absent. `SIGMA=50.0`, `HEATMAP_BLUR_SIGMA=0.0`, `NUM_STEPS=100` remain.

---

### T2: `motionbert_to_camera_space_batch` removed -- PASS

**Command:**
```python
import detect
assert not hasattr(detect, 'motionbert_to_camera_space_batch')
```
**Result:** No batch-related names found in detect module.

---

### T3: `_get_sigma` removed, `_get_blur_sigma` kept -- PASS

**Command:**
```python
import optimize
assert '_get_sigma' not in dir(optimize)
assert '_get_blur_sigma' in dir(optimize)
```
**Result:** `_get_sigma` absent, `_get_blur_sigma` present (correct: sweep still uses it).

---

### T4: `use_real_heatmaps` removed from `compute_total_score` -- PASS

**Command:**
```python
import inspect
from scoring import compute_total_score
sig = inspect.signature(compute_total_score)
assert 'use_real_heatmaps' not in sig.parameters
```
**Result:** Parameters are `['all_positions', 'all_projected_2d', 'all_local_rots', 'target_2d_list', 'visibility_list', 'sigma', 'position_penalty_weight', 'rotation_per_joint_weights', 'initial_positions_list', 'init_anchor_weight', 'heatmaps_list', 'affine']`. `use_real_heatmaps` absent.

---

### T5: sweep.py loads and SweepConfig fields correct -- PASS

**Command:**
```python
from sweep import SweepConfig, get_phase1_2_configs
configs = get_phase1_2_configs()
assert len(configs) == 17
assert not hasattr(configs[0], 'sigma_schedule')
assert not hasattr(configs[0], 'all_joints_smooth_weight')
```
**Result:** 17 configs, fields are `['name', 'num_steps', 'position_penalty_weight', 'rotation_penalty_scalar', 'init_anchor_weight', 'heatmap_blur_sigma', 'heatmap_blur_schedule']`. Both stale fields absent.

---

### T6: main.py regression -- PASS

Verified via code review:
- Import changed to `motionbert_to_camera_space` (not batch)
- `improved_target_2d` variable and loop absent
- `FK_TARGET_CONF_THRESHOLD` not referenced
- `NUM_JOINTS` import not present (removed)
- Per-frame conversion loop present, correct signature

---

### T7: Z-value analysis -- PASS with observations

**Example 0 (171204_pose1_sample_0, 100 frames):**
```
Det root Z: min=1.916  max=2.383  mean=2.296  std=0.068 m
Opt root Z: min=2.029  max=2.388  mean=2.317  std=0.050 m
GT root Z:  min=2.569  max=2.635  mean=2.588  std=0.016 m
```

All Z values positive, in range [1, 8] m. No Z <= 0, no Z < 1.0.

Observation: Det root Z mean is 2.296 m vs GT mean 2.588 m -- a systematic underestimate of ~0.29 m (11%). This is the known pairwise depth bias discussed in prior tester reports. Z std is 0.068 m (much better than solvePnP's 0.147 m from P1-01). The low end of 1.916 m (frames 42-43) is a localized jitter event but within the clipping range.

Per-frame Z values for all 100 frames: smooth trajectory with the dip at frames 41-46 (min 1.92 m), matching the visual frame 42 where the person is partially occluded.

**Example 5 (171204_pose3_4000, 150 frames):**
```
Det root Z: min=2.343  max=2.404  mean=2.381  std=0.015 m
Opt root Z: min=2.371  max=2.395  mean=2.385  std=0.007 m
GT root Z:  min=2.401  max=2.412  mean=2.405  std=0.003 m
```

Excellent Z quality. Det mean 2.381 vs GT mean 2.405 -- only 0.024 m difference (1%). Very stable across frames (std=0.015 m). No anomalies.

---

### T8: Overlay video frame analysis -- PASS with observations

Extracted frames from `p1-pairwise-revert` overlay video.

**Example 0, Frame 0 (body at rest, standing still):**
Heatmaps visible as orange/yellow hotspots at joints. Red skeleton (detector) and green skeleton (optimized) overlay the person correctly. Hip joint placement is centered on the torso. Arms and legs are plausible. No obviously wrong joint placement.

**Example 0, Frame 42 (person extending arms sideways):**
Arms extended wide. Heatmaps correctly placed at wrist/elbow/shoulder positions. Red and green skeletons largely agree. The depth drop at this frame (Z=1.92 vs neighboring ~2.3 m) is visible as a very slight scale change in the projected skeleton but not catastrophic.

**Example 5, Frame 4000:**
Person standing with hands on hips. Skeleton placement accurate. Heatmaps match joint locations well. Detector (red) and optimized (green) are very close to GT (blue).

---

### T9: MPJVE spec requirement -- PARTIAL FAIL

Spec: "MPJVE should be improved for ALL test videos vs round-4 baseline."

**Actual results:**

| Example | Metric | Round-4 | P1-02 | Delta | Result |
|---------|--------|---------|-------|-------|--------|
| Ex0 | Det MPJVE | 0.94 cm/f | 0.94 cm/f | 0.00 | TIED (not improved) |
| Ex0 | Opt MPJVE | 1.05 cm/f | 1.56 cm/f | +0.51 | REGRESSION |
| Ex5 | Det MPJVE | 0.51 cm/f | 0.51 cm/f | 0.00 | TIED (not improved) |
| Ex5 | Opt MPJVE | 0.48 cm/f | 0.38 cm/f | -0.10 | IMPROVED |

The spec requires MPJVE to be **improved** for all test videos. Results:
- Det MPJVE: ties for both, not improvements. 0.00 delta = "SAME", not "IMPROVED."
- Opt MPJVE: Ex5 improves (+0.10 better), Ex0 regresses (0.51 worse).

The architect's P1-02 plan acknowledged this: "recovers to round-4 level -- no rotation jitter from solvePnP." But the spec says "improved", not "recovered to baseline." A tie is not an improvement. And Opt MPJVE for Example 0 is significantly worse (1.56 vs 1.05).

---

### T10: Other metrics vs round-4 -- PASS with observations

**Example 0 full comparison:**

| Metric | Round-4 | P1-02 | Delta |
|--------|---------|-------|-------|
| Det MPJPE | 30.98 cm | 30.98 cm | 0.00 |
| Opt MPJPE | 30.44 cm | 31.67 cm | +1.23 (regression) |
| Det P-MPJPE | 28.57 cm | 28.57 cm | 0.00 |
| Opt P-MPJPE | 28.42 cm | 28.21 cm | -0.21 (slight improvement) |
| Det MPJVE | 0.94 cm/f | 0.94 cm/f | 0.00 |
| Opt MPJVE | 1.05 cm/f | 1.56 cm/f | +0.51 (regression) |

**Example 5 full comparison:**

| Metric | Round-4 | P1-02 | Delta |
|--------|---------|-------|-------|
| Det MPJPE | 15.85 cm | 15.85 cm | 0.00 |
| Opt MPJPE | 15.44 cm | 15.52 cm | +0.08 (slight regression) |
| Det P-MPJPE | 20.33 cm | 20.33 cm | 0.00 |
| Opt P-MPJPE | 20.22 cm | 20.15 cm | -0.07 (slight improvement) |
| Det MPJVE | 0.51 cm/f | 0.51 cm/f | 0.00 |
| Opt MPJVE | 0.48 cm/f | 0.38 cm/f | -0.10 (improvement) |

The detector output (det_*) is identical to round-4 for both examples (same detection pipeline, same pairwise depth estimation), which is expected and correct.

The optimization output (opt_*) regresses on MPJPE and MPJVE for Example 0 but shows slight improvements on P-MPJPE for both examples. The Ex0 Opt MPJVE regression (1.05 -> 1.56 cm/f) is particularly notable -- it was also present in round-4 (det 0.94, opt 1.05, so even round-4 had the optimizer introducing jitter).

---

## Code Review Findings

### CR-1: `optimize.py` docstring references `(17, 3)` instead of `(16, 3)` -- Minor

Lines 55, 64, and the `scoring.py` docstring at lines 232-235 still reference "17" joints. The skeleton has 16 joints (HEAD removed in Phase 0). These are stale comments that could confuse maintainers.

- `optimize.py` line 55: `"Per-frame (17, 3) camera-space positions from detector."`
- `optimize.py` line 64: `"optimized_3d: list of (17, 3) optimized camera-space positions per frame"`
- `scoring.py` lines 232-235 in `compute_total_score` docstring: references `"(17, 3)"` multiple times

**Severity:** Minor (cosmetic, no runtime impact). The actual tensors are correctly (16, 3) -- `NUM_JOINTS=16` -- so this is purely a documentation error.

### CR-2: `current_blur_sigma` variable is dead when `heatmap_blur_schedule=None` -- Minor (acknowledged by developer)

In `optimize.py` lines 171-174, `current_blur_sigma` is initialized to 0.0 and only updated inside the `if heatmaps_t_orig is not None and heatmap_blur_schedule is not None` block. When `heatmap_blur_schedule=None` (which is the `main.py` code path and the most common case), this variable is set but never changes. The developer acknowledged this in their report. It causes no bugs and is technically dead code.

### CR-3: `sweep.py` `load_example()` returns both `"kp_2d"` and `"target_2d"` keys pointing to the same list object -- Minor

Per the developer's decision note #5, both keys exist in the returned dict. This is harmless but slightly untidy. The `"target_2d"` key is what `run_sweep_config()` uses; `"kp_2d"` was already there for other downstream uses. No bug.

### CR-4: Opt MPJVE regression from optimization for Example 0 is a known structural issue

The optimizer introduces velocity jitter (Det MPJVE 0.94 -> Opt MPJVE 1.56 cm/f for Ex0). This was also a concern in P1-01 and appears to be a fundamental characteristic of the heatmap-only loss with no position smoothing enabled. The `POSITION_PENALTY_WEIGHT=50.0` currently only penalizes large root jumps, not gradual velocity. This is not a bug introduced by P1-02 -- it was also present in round-4 (det 0.94, opt 1.05). But the regression from 1.05 -> 1.56 suggests the current default (100 steps, no blur) is somewhat worse than the round-4 config (which was 20 steps).

---

## Bugs Found

### BUG-1 (BLOCKS SPEC): Opt MPJVE regresses for Example 0 (1.05 -> 1.56 cm/f)

**Description:** The optimizer increases velocity error for Example 0 by +0.51 cm/f. The spec requires MPJVE to be "improved for ALL test videos."

**How observed:** T9 metric comparison.

**Severity:** Blocks spec definition of done. The spec's Phase 1 goal is "MPJVE should be improved for ALL test videos." Opt MPJVE is what matters for the final output (main.py returns `optimized_3d`). Example 0 shows clear regression.

**Context:** This was already a known failure mode from the P1-01 blur sweep (TESTER_REPORT_P1_01.md, CR-3): "Every blur configuration increases MPJVE relative to the no-blur baseline." The round-4 MPJVE (1.05) was achieved with 20 steps (no blur). The current default is 100 steps (no blur). The combination of more steps and no blur is causing the optimizer to overfit to noisy 2D targets frame-by-frame.

**Suggested fix direction:** The Tester REPORT_P1_01 sweep showed that the 20-step no-blur baseline (the round-4 config) was the only configuration that achieved Opt MPJVE <= 1.05 for Ex0. Achieving both better MPJPE AND better MPJVE requires either: (a) fewer steps + lower anchor weight (to reduce jitter while improving 3D), or (b) stronger temporal smoothing (position or rotation penalties). The P1-01 sweep found that blur configurations improved MPJPE but at the cost of MPJVE.

### BUG-2 (MEDIUM): Det MPJVE ties round-4 rather than improving

**Description:** Det MPJVE matches round-4 exactly for both examples (0.94 = 0.94, 0.51 = 0.51). The spec says "improved" not "unchanged." While this is a successful recovery from the solvePnP regression (P1-00 had 3.69 cm/f), it is not an improvement.

**How observed:** T9 metric comparison.

**Severity:** Medium. The detector output is a clean re-run of pairwise depth estimation, which by definition produces the same result as round-4. Improving Det MPJVE would require a fundamentally different depth estimation strategy or temporal smoothing of the detector output.

**Suggested fix direction:** Temporal smoothing of the detected positions (e.g., a Kalman filter or running median on root Z) applied before FK optimization could reduce Det MPJVE. Alternatively, better depth estimation could produce a smoother trajectory. This is a longer-term fix.

---

## Summary Table

| Test | Result |
|------|--------|
| T1: Config constants removed | PASS |
| T2: `motionbert_to_camera_space_batch` removed | PASS |
| T3: `_get_sigma` removed, `_get_blur_sigma` kept | PASS |
| T4: `use_real_heatmaps` removed from `compute_total_score` | PASS |
| T5: sweep.py loads, SweepConfig correct | PASS |
| T6: main.py regression | PASS |
| T7: Z-value analysis | PASS |
| T8: Overlay video frame analysis | PASS |
| T9: MPJVE spec requirement | FAIL (Opt MPJVE regresses for Ex0; Det MPJVE ties for both) |
| T10: Other metrics vs round-4 | PASS with observations (det metrics identical; opt MPJPE slight regression for Ex0) |

---

## Verdict

**NO** -- The current state does not fully meet the spec's Phase 1 definition of done.

**What works:**
- All code cleanup changes are correct and complete: 4 dead config constants removed, `motionbert_to_camera_space_batch` removed, `_get_sigma` removed, `use_real_heatmaps` removed from scoring, sweep.py updated correctly.
- Z-values are physically reasonable (no negative Z, no > 8 m, no catastrophic jitter).
- Det MPJPE and Det P-MPJPE are identical to round-4 (correct recovery from solvePnP regression).
- Overlay videos look correct visually.
- Ex5 Opt MPJVE improves (0.48 -> 0.38 cm/f).

**What fails:**
- Spec: "MPJVE should be improved for ALL test videos." This is not met:
  - Ex0 Opt MPJVE: 1.05 (round-4) -> 1.56 cm/f (P1-02) = **regression of +0.51 cm/f**
  - Det MPJVE for both examples: tied at round-4, not improved

The Opt MPJVE regression for Example 0 is the primary blocker. The root cause is known: 100 optimization steps with no blur causes frame-to-frame jitter that was not present in the round-4 20-step config. The Phase 1.1 sweep (TESTER_REPORT_P1_01.md) already showed this -- every configuration except the 20-step no-blur baseline had Opt MPJVE > 1.05 for Ex0.

**To meet the spec**, the next iteration needs to either:
1. Find a config where Opt MPJVE < 1.05 for Ex0 AND < 0.48 for Ex5, OR
2. Change the definition of "MPJVE improved" to include only the detector output (Det MPJVE), not the optimizer output. The developer's report interpreted "MPJVE improved" to mean Det MPJVE recovered to round-4 levels (= baseline), but strictly the spec says "improved" not "recovered."
