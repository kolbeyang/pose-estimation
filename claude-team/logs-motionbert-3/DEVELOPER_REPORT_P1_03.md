# Developer Report: Phase 1, Iteration 3

**Date:** 2026-03-17
**Plan:** `claude-team/logs/ARCHITECT_PLAN_P1_03.md`
**Phase:** 1, Iteration 3

---

## What Was Implemented

Single config change: `HEATMAP_BLUR_SIGMA` set from `0.0` to `4.0` in `motionbert-pose/config.py`. No other files were modified.

Per the architect's instructions, sigma=8.0 was also tried when sigma=4.0 failed to achieve Opt MPJVE < Det MPJVE for Example 0. Sigma=8.0 also failed. The final value left in config.py is `4.0` (the primary change requested by the spec).

---

## Files Changed

- `motionbert-pose/config.py`: `HEATMAP_BLUR_SIGMA: float = 0.0` -> `HEATMAP_BLUR_SIGMA: float = 4.0`

---

## Commands Run

### Step 1: Verify blur application path in optimize.py
Read-only check of `optimize.py` lines 159-161. Confirmed:
```python
if heatmaps_t is not None and heatmap_blur_schedule is None and cfg.HEATMAP_BLUR_SIGMA > 0:
    heatmaps_t = _apply_blur_torch(heatmaps_t, cfg.HEATMAP_BLUR_SIGMA)
```
Path is correct. Blur applied once before optimization loop.

### Step 2: Run Example 0 with sigma=4
```bash
cd motionbert-pose && uv run python -c "
from main import process_example
import config as cfg, os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-blur4')
os.makedirs(run_dir, exist_ok=True)
seq, cam, start, nf, pidx = cfg.EXAMPLES[0]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```
**Result:** Completed without crash. Det MPJVE = 0.94 cm/f, Opt MPJVE = **1.67 cm/f** (fails spec target < 0.94 cm/f).

### Step 3: Try sigma=8 on Example 0
Temporarily changed to `HEATMAP_BLUR_SIGMA = 8.0`, ran same command with `run_dir = 'p1-blur8'`.
**Result:** Opt MPJVE = **1.72 cm/f** (also fails; slightly worse than sigma=4).

Reverted config to sigma=4.0 as the primary specified change.

### Step 4: Run Example 5 with sigma=4
```bash
uv run python -c "
from main import process_example
import config as cfg, os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-blur4')
seq, cam, start, nf, pidx = cfg.EXAMPLES[5]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```
**Result:** Completed without crash. Det MPJVE = 0.51 cm/f, Opt MPJVE = **0.53 cm/f** (slight regression from P1-02's 0.38 cm/f, fails spec target < 0.51 cm/f).

---

## Results

### Example 0 (171204_pose1_sample, frames 0-99)

| Metric | Round-4 | P1-02 (no blur) | P1-03 (blur=4) | Delta vs Round-4 |
|--------|---------|-----------------|----------------|------------------|
| Det MPJPE (cm) | 30.98 | 30.98 | 30.98 | 0.00 |
| Opt MPJPE (cm) | 30.44 | 31.67 | 31.11 | +0.67 (regression) |
| Det P-MPJPE (cm) | 28.57 | 28.57 | 28.57 | 0.00 |
| Opt P-MPJPE (cm) | 28.42 | 28.21 | 28.31 | -0.11 (slight improvement) |
| Det MPJVE (cm/f) | 0.94 | 0.94 | 0.94 | 0.00 |
| Opt MPJVE (cm/f) | 1.05 | 1.56 | **1.67** | +0.62 (regression) |

Also tested sigma=8 on Example 0:
- Opt MPJPE = 30.78 cm (+0.34 vs round-4, improvement over sigma=4)
- Opt MPJVE = 1.72 cm/f (slightly worse than sigma=4)

### Example 5 (171204_pose3, frames 4000-4149)

| Metric | Round-4 | P1-02 (no blur) | P1-03 (blur=4) | Delta vs Round-4 |
|--------|---------|-----------------|----------------|------------------|
| Det MPJPE (cm) | 15.85 | 15.85 | 15.85 | 0.00 |
| Opt MPJPE (cm) | 15.44 | 15.52 | 15.66 | +0.22 (slight regression) |
| Det P-MPJPE (cm) | 20.33 | 20.33 | 20.33 | 0.00 |
| Opt P-MPJPE (cm) | 20.22 | 20.15 | 20.15 | -0.07 (slight improvement) |
| Det MPJVE (cm/f) | 0.51 | 0.51 | 0.51 | 0.00 |
| Opt MPJVE (cm/f) | 0.48 | 0.38 | **0.53** | +0.05 (regression) |

---

## Decisions Made

### Decision 1: Final config value is sigma=4.0

The task specified sigma=4 as the primary change. When sigma=4 failed to achieve Opt MPJVE < Det MPJVE for Example 0, the plan said to try sigma=8. Sigma=8 was tested and also failed (Opt MPJVE 1.72 cm/f vs Det MPJVE 0.94 cm/f). The plan said "try sigma=8" but did not specify to use sigma=8 as the final value if it also fails. Since sigma=4 is the "primary change" the spec called for, and sigma=8 shows no benefit over sigma=4 on MPJVE, I reverted to sigma=4.0 for the final config.

### Decision 2: Overlay video shows unblurred heatmaps (by design)

The architect's plan explicitly notes that the overlay video in `main.py` uses the raw (unblurred) heatmaps — the blur is applied inside `run_optimization()` to a local copy. The plan says "Do NOT change the overlay video behavior in this iteration -- leave that for Phase 2 as intended." No change was made to `overlay_video.py` or to how heatmaps are passed in `main.py`.

---

## Concerns

### Concern 1: Blur fails to achieve Opt MPJVE < Det MPJVE for BOTH examples

The P1-01 sweep (TESTER_REPORT_P1_01.md) already demonstrated this: every blur configuration increases MPJVE relative to the no-blur 20-step baseline. The round-3 sweep numbers cited in the architect's plan (blur4_100s = 1.62 cm/f, blur8_100s = 1.66 cm/f) are consistent with what was measured here (1.67 and 1.72 cm/f respectively — slightly worse, possibly due to codebase differences or run-to-run variance).

The sweep result is definitive: **no blur-only configuration achieves Opt MPJVE < 1.05 cm/f (round-4 level) for Example 0, let alone < 0.94 cm/f (Det MPJVE level)**. The MPJVE requirement cannot be met with blur alone. The architect's plan itself acknowledged this risk (Risk 1).

Example 5 is now also slightly regressed: Opt MPJVE went from 0.38 cm/f (P1-02, no blur) to 0.53 cm/f (P1-03, blur=4), which is worse than Det MPJVE (0.51 cm/f). P1-02 with no blur was actually meeting the spec for Example 5.

### Concern 2: Opt MPJPE also regresses for Example 5 with blur

P1-02 had Opt MPJPE = 15.52 cm for Example 5. With blur=4, it is 15.66 cm — worse, not better. The sweep ran on Example 0 only; the gains from blur do not appear to generalize to Example 5 (a simpler, lower-motion sequence). This was noted as a risk in the architect's plan.

### Concern 3: The spec target Opt MPJVE < Det MPJVE is unachievable with current loss formulation

All blur and no-blur configurations produce Opt MPJVE > Det MPJVE for Example 0. The only exception in the entire P1-01 sweep was the 20-step no-blur baseline (round-4 config), which matched round-4's 1.05 cm/f but that is already worse than Det MPJVE (0.94 cm/f). The underlying issue is that the per-frame FK optimizer has no explicit velocity term — it optimizes each frame semi-independently (only the `POSITION_PENALTY_WEIGHT` and `ROTATION_PENALTY_SCALAR` couple adjacent frames implicitly, but these are anchor penalties vs the initialization, not explicit velocity/smoothness terms). A dedicated temporal smoothness loss or post-optimization filtering would be needed to achieve Opt MPJVE < Det MPJVE.

---

## Deviations from Plan

### Deviation 1: Spec target not achieved

The plan stated: "If sigma=4 doesn't achieve Opt MPJVE < Det MPJVE on Example 0, try sigma=8." Both were tried. Neither achieves the target. No further deviation from the plan is warranted at this stage — reporting the results accurately is the correct action. The next iteration will need a structural change (e.g., temporal smoothing loss or post-optimization filtering).

---

## Summary

The single config change `HEATMAP_BLUR_SIGMA = 4.0` was implemented. Both sigma=4 and sigma=8 were tested on Example 0. Neither achieves the spec target of Opt MPJVE < Det MPJVE (< 0.94 cm/f for Ex0). The P1-02 no-blur configuration (Opt MPJVE 1.56 cm/f for Ex0, 0.38 cm/f for Ex5) was actually better for Example 5 MPJVE; blur worsens it to 0.53 cm/f (above the 0.51 cm/f Det baseline).

The fundamental problem: the FK optimizer lacks a temporal velocity term. Blur widens the gradient basin (improving 3D accuracy) but the optimizer still fits each frame independently to the blurred heatmap targets. Without an explicit inter-frame smoothness loss or post-processing filter, Opt MPJVE cannot be brought below Det MPJVE.

The config.py is left with `HEATMAP_BLUR_SIGMA = 4.0`. No crashes, no import errors.
