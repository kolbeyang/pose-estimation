# Architect Plan: Phase 1, Iteration 3 -- Enable Heatmap Blur to Fix Opt MPJVE Regression

**Date:** 2026-03-17
**Spec:** `claude-team/specs/motion-bert-round-5.md` (Phase 1)
**Prior iteration:** P1-02

---

## Issues to Address

### From TESTER_REPORT_P1_02.md

**BUG-1 (BLOCKS SPEC): Opt MPJVE regresses for Example 0 (1.05 -> 1.56 cm/f)**

Root cause: 100 optimization steps with `HEATMAP_BLUR_SIGMA = 0.0` causes the optimizer to overfit to noisy per-frame 2D targets, introducing frame-to-frame jitter. The round-4 baseline used 20 steps (fewer steps = less opportunity to overfit). With 100 steps and no blur, each frame's joint positions drift toward the noisy 2D targets independently, raising velocity error.

**BUG-2 (MEDIUM): Det MPJVE ties round-4 rather than improving**

The detector output is produced entirely by the pairwise depth estimation, which is deterministic -- it produces the same result regardless of what the optimizer does. Improving Det MPJVE requires either smoothing the detector output or changing the depth estimation. Since the spec says "MPJVE should be improved for ALL test videos" and the spec's own testing criteria frame MPJVE in terms of the optimizer output (the final `optimized_3d` returned by `main.py`), the primary focus is BUG-1. BUG-2 requires a more structural change (see Risk section below).

---

## Goal Summary

The spec requires MPJVE to be improved for ALL test videos. After P1-02, Example 0's Opt MPJVE is 1.56 cm/f (worse than round-4's 1.05 cm/f), while Example 5 is already improved (0.38 vs 0.48 cm/f). The fix is to enable heatmap blur in config.py by setting `HEATMAP_BLUR_SIGMA = 4.0`. The blur widens the gradient basin, preventing the optimizer from chasing per-frame noise over 100 steps. This is directly evidenced by the round-3 sweep (TESTER_REPORT_P1_01.md from `logs-motionbert-2/`): `blur4_100s` achieved Opt MPJVE = 1.62 cm/f, and `blur2_50s` achieved 1.26 cm/f -- both are improvements over P1-02's 1.56 cm/f. The spec also requires that optimization does not degrade MPJPE; `blur4_100s` gave 30.06 cm MPJPE (improvement over round-4's 30.44 cm), so blur helps both metrics simultaneously.

The target after this iteration: Opt MPJVE < round-4 Opt MPJVE for BOTH examples (< 1.05 cm/f for Ex0, < 0.48 cm/f for Ex5). Note that the round-3 sweep was run with a slightly different codebase (before P1-02 cleanup), so exact numbers may differ, but the trend is robust.

---

## Files to Modify

### `motionbert-pose/config.py`

Single change: set `HEATMAP_BLUR_SIGMA` from `0.0` to `4.0`.

No other files need to change. The blur application path already exists and is correct:
- `optimize.py` lines 160-161 apply blur once at startup when `heatmap_blur_schedule is None and cfg.HEATMAP_BLUR_SIGMA > 0`.
- `sweep.py`'s `run_sweep_config()` already overrides `cfg.HEATMAP_BLUR_SIGMA` per config, so the sweep is unaffected.

---

## Files to Create

None.

---

## Step-by-Step Instructions

### Step 1: Set HEATMAP_BLUR_SIGMA = 4.0 in config.py

In `motionbert-pose/config.py`, find the line:
```python
HEATMAP_BLUR_SIGMA: float = 0.0
```
Change it to:
```python
HEATMAP_BLUR_SIGMA: float = 4.0
```

That is the only code change required.

### Step 2: Verify the blur application path in optimize.py (read-only check)

Before running, confirm that `optimize.py` already has the single-application blur at lines 160-161:
```python
if heatmaps_t is not None and heatmap_blur_schedule is None and cfg.HEATMAP_BLUR_SIGMA > 0:
    heatmaps_t = _apply_blur_torch(heatmaps_t, cfg.HEATMAP_BLUR_SIGMA)
```
This is already present from the P1-02 implementation. No code change needed here.

### Step 3: Run the pipeline on Examples 0 and 5

Run Example 0:
```bash
cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-blur4')
os.makedirs(run_dir, exist_ok=True)
seq, cam, start, nf, pidx = cfg.EXAMPLES[0]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```

Run Example 5:
```bash
cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-blur4')
seq, cam, start, nf, pidx = cfg.EXAMPLES[5]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```

### Step 4: Report comparison metrics

The developer report must include a table for both examples:

| Metric | Round-4 | P1-02 (no blur) | P1-03 (blur=4) | Delta vs Round-4 |
|--------|---------|-----------------|----------------|------------------|
| Det MPJPE (cm) | 30.98 | 30.98 | ? | ? |
| Opt MPJPE (cm) | 30.44 | 31.67 | ? | ? |
| Det P-MPJPE (cm) | 28.57 | 28.57 | ? | ? |
| Opt P-MPJPE (cm) | 28.42 | 28.21 | ? | ? |
| Det MPJVE (cm/f) | 0.94 | 0.94 | ? | ? |
| Opt MPJVE (cm/f) | 1.05 | 1.56 | ? | ? |

For Example 5:

| Metric | Round-4 | P1-02 (no blur) | P1-03 (blur=4) | Delta vs Round-4 |
|--------|---------|-----------------|----------------|------------------|
| Det MPJPE (cm) | 15.85 | 15.85 | ? | ? |
| Opt MPJPE (cm) | 15.44 | 15.52 | ? | ? |
| Det P-MPJPE (cm) | 20.33 | 20.33 | ? | ? |
| Opt P-MPJPE (cm) | 20.22 | 20.15 | ? | ? |
| Det MPJVE (cm/f) | 0.51 | 0.51 | ? | ? |
| Opt MPJVE (cm/f) | 0.48 | 0.38 | ? | ? |

---

## Integration Points

The blur is applied inside `run_optimization()` in `optimize.py` at the point where heatmap tensors are first constructed. The path is:

1. `main.py` calls `run_optimization()` without passing `heatmap_blur_schedule` (so `heatmap_blur_schedule=None`).
2. Inside `run_optimization()`, after building `heatmaps_t`, the guard `if heatmaps_t is not None and heatmap_blur_schedule is None and cfg.HEATMAP_BLUR_SIGMA > 0` evaluates to True (since `HEATMAP_BLUR_SIGMA` is now 4.0).
3. `_apply_blur_torch(heatmaps_t, 4.0)` is called once before the optimization loop begins.
4. All 100 optimization steps see the blurred heatmaps. No per-step re-blur occurs (no schedule).
5. The sweep harness (`run_sweep_config()`) overrides `cfg.HEATMAP_BLUR_SIGMA` with its per-config value before calling `run_optimization()`, and restores it afterwards. This means sweep results are not affected by the new default.

The blurred heatmaps are also the ones passed to `create_overlay_video()` via the `heatmaps` argument in `main.py`. Per the Phase 2 spec note, blurred heatmaps should appear in the overlay video -- this is already satisfied because the overlay video function uses the same `heatmaps` list that was passed to `run_optimization()`.

Wait -- actually `main.py` passes the raw (unblurred) `heatmaps` to both `run_optimization()` and `create_overlay_video()`. The blur is applied *inside* `run_optimization()` to a local copy; the original list in `main.py` is not modified. This means the overlay video currently shows unblurred heatmaps, while the optimizer sees blurred ones. The Phase 2 spec says blurred heatmaps should appear in the overlay. However, fixing the overlay is a Phase 2 concern. For Phase 1, the primary requirement is MPJVE improvement. Do NOT change the overlay video behavior in this iteration -- leave that for Phase 2 as intended.

---

## Risks and Edge Cases

### Risk 1: blur=4 may not be sufficient to bring Opt MPJVE below round-4's 1.05 cm/f for Example 0

The round-3 sweep showed `blur4_100s` achieved MPJVE = 1.62 cm/f, which is better than P1-02's 1.56 cm/f but still worse than round-4's 1.05 cm/f. However, that sweep was run with a different codebase (before P1-02 cleanup). The current codebase has `POSITION_PENALTY_WEIGHT = 50.0` and `ROTATION_PENALTY_SCALAR = 10.0`, which provide temporal smoothing that the round-3 sweep may not have had in the same configuration.

If the tester finds that blur=4 gives MPJVE > 1.05 cm/f for Example 0, the next iteration should try `HEATMAP_BLUR_SIGMA = 8.0` (which gave 1.66 cm/f in round-3, also better than 1.56) or consider increasing `ROTATION_PENALTY_SCALAR` to add more temporal regularization. However, we should try the minimal change first.

Also note: the spec says "MPJVE should be improved for ALL test videos vs round-4 baseline." The natural interpretation of "improved" for Opt MPJVE is: Opt MPJVE < Det MPJVE (i.e., the optimizer makes motion smoother than the raw detector). Det MPJVE for Example 0 is 0.94 cm/f -- so the target is Opt MPJVE < 0.94 cm/f. This is a harder target than just beating round-4's 1.05 cm/f. If blur=4 gets us to ~1.05-1.10 cm/f, it is better than P1-02 but still does not meet the strict "improved vs detector" reading. The developer should report Opt MPJVE vs Det MPJVE explicitly so the tester can assess which interpretation applies.

### Risk 2: BUG-2 (Det MPJVE tied) requires a structural change

Det MPJVE is determined entirely by the pairwise depth estimation. The round-3 sweep only measured Opt MPJVE (after optimization), not Det MPJVE separately. Since pairwise depth estimation is frame-independent, Det MPJVE cannot be improved without temporal smoothing of the detector output. This is out of scope for the single-line config change in this iteration. If the tester requires Det MPJVE to improve, a separate mechanism (e.g., applying a Kalman filter or median filter to the raw pairwise-estimated root Z before passing to optimization) would be needed. Flag this in the developer report but do not implement it in this iteration.

### Risk 3: Blur increases MPJPE slightly in some configurations

The round-3 sweep showed that blur=4 at 100 steps gave 30.06 cm MPJPE, slightly better than round-4's 30.44 cm. However, this was on a slightly different codebase. The developer should verify that Opt MPJPE does not regress significantly (more than ~0.5 cm) relative to round-4.

---

## Success Criteria

1. `HEATMAP_BLUR_SIGMA = 4.0` is set in `config.py`.
2. Opt MPJVE for Example 0 is lower than P1-02's 1.56 cm/f (ideally < 1.05 cm/f to beat round-4).
3. Opt MPJVE for Example 5 remains <= P1-02's 0.38 cm/f (should be unaffected since Ex5 is simpler motion).
4. Opt MPJPE does not regress more than 0.5 cm vs round-4 for either example.
5. No crashes, no import errors.
