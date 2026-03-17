# Architect Plan: Phase 1, Iteration 2 -- Revert to Pairwise Depth, Remove Dead Config

**Date:** 2026-03-17
**Spec:** `claude-team/specs/motion-bert-round-5.md` (Phase 1)
**Prior iteration:** P1-01

---

## Summary of Situation

Phase 1, Iteration 1 attempted to fix solvePnP depth jitter by applying a temporal median filter to the translation vector. This achieved the internal target (root Z std 0.147 -> 0.043 m) but failed the spec requirement: MPJVE was NOT improved versus the round-4 baseline for either tested example (3.69 cm/f vs 1.05 cm/f for Example 0). The developer correctly diagnosed that the jitter is coming from the per-frame solvePnP *rotation*, not just the translation.

The user has therefore decided to revert the depth estimation strategy to the Phase 0 pairwise-only approach (which produced MPJVE ~1.05 cm/f vs solvePnP's 3.38-3.69 cm/f). In addition, the user has tagged several config parameters as dead/unwanted via TODO comments in `config.py`:

1. **Revert depth estimation** -- remove solvePnP entirely from `motionbert_to_camera_space`; remove `motionbert_to_camera_space_batch`; restore the pairwise tz estimation from Phase 0.
2. **Remove SIGMA_SCHEDULE** -- use only the fixed `SIGMA` value.
3. **Remove HEATMAP_BLUR_SCHEDULE** -- use only the fixed `HEATMAP_BLUR_SIGMA`.
4. **Remove FK_TARGET_CONF_THRESHOLD** -- remove the mechanism that replaced low-confidence 2D targets with MotionBERT projections; always use raw Stacked Hourglass detections as optimization targets.
5. **Remove USE_REAL_HEATMAPS** -- always use real heatmaps; remove the analytical Gaussian fallback path.
6. **Fix sweep.py** -- update to match the simplified interface (no batch function, no schedule params, no `FK_TARGET_CONF_THRESHOLD` target-building logic, no `sigma_schedule` field in `SweepConfig`).
7. **Fix main.py** -- revert to per-frame camera-space conversion.

The goal of this cleanup iteration is to produce a simpler, more maintainable codebase where every code path is actually used, then re-run on Examples 0 and 5 to confirm that metrics have recovered to at least the round-4 baseline.

---

## Files to Modify

### `motionbert-pose/config.py`
- Remove `SIGMA_SCHEDULE` constant and its comment block.
- Remove `HEATMAP_BLUR_SCHEDULE` constant and its comment block.
- Remove `FK_TARGET_CONF_THRESHOLD` constant and its comment block.
- Remove `USE_REAL_HEATMAPS` constant and its comment block.

### `motionbert-pose/detect.py`
- Replace the body of `motionbert_to_camera_space()` with the pairwise depth estimation logic from Phase 0 (no solvePnP, no rotation matrix, no cv2 dependency for this function).
- Remove `motionbert_to_camera_space_batch()` entirely.

### `motionbert-pose/optimize.py`
- Remove `_get_sigma()` function and its call; replace with direct use of `cfg.SIGMA`.
- Remove `_get_blur_sigma()` function.
- Remove `heatmap_blur_schedule` parameter from `run_optimization()`.
- Remove the schedule-based blur update block (the `heatmaps_t_orig` storage and per-step `_get_blur_sigma` call).
- Remove the `use_real_heatmaps` argument from the `compute_total_score()` call; always pass real heatmaps when available.
- Remove the `USE_REAL_HEATMAPS` guard in heatmap tensor construction.
- Simplify the step logging to remove the `blur=` field (since there is no dynamic schedule).

### `motionbert-pose/scoring.py`
- Remove the `use_real_heatmaps` parameter from `compute_total_score()`.
- Simplify the `_use_real` logic: always use real heatmaps when `heatmaps_list is not None and affine is not None`; remove the boolean gate.

### `motionbert-pose/main.py`
- Replace the `motionbert_to_camera_space_batch` import with `motionbert_to_camera_space`.
- Restore the per-frame conversion loop (call `motionbert_to_camera_space()` per frame).
- Remove the `improved_target_2d` loop that used `FK_TARGET_CONF_THRESHOLD`; pass `kp_2d` directly as `target_2d` to `run_optimization()`.

### `motionbert-pose/sweep.py`
- Remove `sigma_schedule` field from `SweepConfig` dataclass.
- Remove the `orig_sigma_schedule` save/restore block in `run_sweep_config()`.
- Remove the `cfg.SIGMA_SCHEDULE = config.sigma_schedule` assignment in `run_sweep_config()`.
- Remove the `improved_target_2d` loop that used `FK_TARGET_CONF_THRESHOLD` in `load_example()`; store `kp_2d` directly as the target.
- Remove the `improved_target_2d` key from the returned dict in `load_example()`.
- Update `run_sweep_config()` to pass `data["kp_2d"]` (not `data["improved_target_2d"]`) to `run_optimization()` and `compute_comparison_with_optimization()`.
- Update `motionbert_to_camera_space_batch` import to `motionbert_to_camera_space`.
- Replace the batch camera-space conversion call in `load_example()` with a per-frame loop.

## Files to Create

None.

---

## Step-by-Step Instructions

### Step 1: Simplify `config.py`

Open `motionbert-pose/config.py` and make these removals:

**1a.** Remove the `SIGMA_SCHEDULE` block entirely (lines 49-53). That is, delete:
```
# TODO: remove we don't want a sigma scheduler
# Coarse-to-fine sigma schedule: (fraction_of_steps, sigma)
SIGMA_SCHEDULE: list[tuple[float, float]] = [
    (1.0, 80.0),  # Constant coarse sigma -- prevents overfitting to noisy 2D targets
]
```
The fixed `SIGMA: float = 50.0` constant on line 47 stays.

**1b.** Remove the `HEATMAP_BLUR_SCHEDULE` block entirely (lines 60-67). That is, delete:
```
# TODO: remove this we don't want a heatmap blur schedule
# Coarse-to-fine heatmap blur schedule: (fraction_of_steps, blur_sigma)
# Starts with wide blur for coarse alignment, narrows for precision.
# Set to None to use fixed HEATMAP_BLUR_SIGMA instead.
HEATMAP_BLUR_SCHEDULE: list[tuple[float, float]] | None = [
    (0.5, 8.0),  # First 50%: wide blur for coarse alignment
    (1.0, 2.0),  # Last 50%: narrow blur for precision
]
```
The fixed `HEATMAP_BLUR_SIGMA: float = 0.0` on line 58 stays.

**1c.** Remove the `FK_TARGET_CONF_THRESHOLD` block entirely (lines 106-111). That is, delete:
```
# TODO: remove this and all associated functionality, unneeded complexity
# Confidence threshold for replacing FK optimization 2D targets.
# For joints below this threshold, use MotionBERT's projected 2D
# instead of (potentially garbage) Stacked Hourglass detections.
# This does NOT affect MotionBERT's input (MOTIONBERT_CONF_THRESHOLD controls that).
FK_TARGET_CONF_THRESHOLD: float = 0.1
```

**1d.** Remove the `USE_REAL_HEATMAPS` block entirely (lines 117-122). That is, delete:
```
# TODO: remove this and all associated functionality, unneeded complexity, we should ALWAYS use real heatmaps
# Use real Stacked Hourglass heatmaps for optimization scoring.
# When True, the optimizer samples from the actual (16, 64, 64) heatmaps
# produced by Stacked Hourglass instead of analytical Gaussian approximations.
# When False, uses the old analytical Gaussian approach.
USE_REAL_HEATMAPS: bool = True
```

### Step 2: Rewrite `motionbert_to_camera_space()` in `detect.py` -- pairwise-only

Replace the entire body of `motionbert_to_camera_space()` (lines 618-722) with the pairwise depth estimation strategy from Phase 0. Keep the function signature identical:

```python
def motionbert_to_camera_space(
    positions_3d_norm: np.ndarray,
    kp_2d: np.ndarray,
    scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist_coeffs: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
) -> np.ndarray:
```

The new body should:

1. Compute `root_relative = positions_3d_norm - positions_3d_norm[0:1]` (root-relative structure).
2. Compute bone-scale factor exactly as before (arm-bones-preferred IQR-filtered median of ref_bl/det_bl ratios), producing `root_relative_m = root_relative * bone_scale`.
3. Estimate `tz` using pairwise joint separation ratios:
   - For every pair `(i, j)` where `i != j`, if both joints have valid 2D detections (pixel distance > 1.0 in kp_2d) and the 3D separation in `root_relative_m` is > 1 mm:
     - Compute `dy_3d = abs(root_relative_m[i, 1] - root_relative_m[j, 1])` (vertical world separation)
     - Compute `dv_2d = abs(kp_2d[i, 1] - kp_2d[j, 1])` (vertical pixel separation)
     - If `dv_2d > 5.0`: append `tz_candidate = fy * dy_3d / dv_2d`
   - Collect all `tz_candidates`; apply IQR-filtered median to get `tz`; clip to `[1.0, 8.0]`
   - Fall back to `tz = 3.0` if fewer than 2 candidates remain.
4. Solve for `tx, ty` from the root joint 2D projection: `tx = (kp_2d[0,0] - cx) * tz / fx`, `ty = (kp_2d[0,1] - cy) * tz / fy`.
5. Apply bone-length constraints via `_enforce_bone_lengths(root_relative_m, PARENTS, DEFAULT_BONE_LENGTHS, max_ratio=1.3)`.
6. Translate: `cam_3d = root_relative_corrected.copy(); cam_3d[:, 0] += tx; cam_3d[:, 1] += ty; cam_3d[:, 2] += tz`.
7. Return `cam_3d.astype(np.float64)`.

**Note on pairwise strategy detail:** Use all joint pairs (not just vertical ones) but restrict `tz` estimation to pairs where the projected vertical separation `dv_2d` is at least 5 pixels. This is the same filter the Phase 0 code used to avoid dividing by near-zero pixel separations. The horizontal separation (`dx`) is deliberately not used -- depth estimation from horizontal parallax is unreliable for single-camera setups. Keep the IQR filter with `k=1.5`.

**Note on `dist_coeffs`:** The parameter exists for interface compatibility but is not used in the pairwise approach. No solvePnP means no camera matrix inversion is needed. Leave the parameter in place and simply ignore it.

### Step 3: Remove `motionbert_to_camera_space_batch()` from `detect.py`

Delete the entire `motionbert_to_camera_space_batch()` function (lines 725-896). This includes the function definition, docstring, all internal phases, and the diagnostics printout.

After deletion, the file should jump directly from the end of `motionbert_to_camera_space()` to the `# Pipeline Wrapper` section and `detect_poses()`.

### Step 4: Simplify `optimize.py`

**4a.** Remove `_get_sigma()` (lines 17-23). This is the coarse-to-fine sigma scheduler. It reads from `cfg.SIGMA_SCHEDULE` which is being removed.

**4b.** Remove `_get_blur_sigma()` (lines 26-32). This is the coarse-to-fine heatmap blur scheduler.

**4c.** Remove the `heatmap_blur_schedule` parameter from `run_optimization()`. Change the signature from:
```python
def run_optimization(
    ...
    heatmap_blur_schedule: list[tuple[float, float]] | None = None,
) -> ...:
```
to:
```python
def run_optimization(
    ...
) -> ...:
```

**4d.** In the heatmap tensor construction block (around line 160), remove the `cfg.USE_REAL_HEATMAPS` guard. Change:
```python
if heatmaps is not None and cfg.USE_REAL_HEATMAPS:
    heatmaps_t = [...]
    ...
    print(f"    Using real Stacked Hourglass heatmaps for scoring")
else:
    print(f"    Using analytical Gaussian heatmaps for scoring")
```
to:
```python
if heatmaps is not None:
    heatmaps_t = [
        torch.tensor(hm, dtype=torch.float32) for hm in heatmaps
    ]
    if affine is not None:
        affine_t = torch.tensor(affine, dtype=torch.float32)
    print(f"    Using real Stacked Hourglass heatmaps for scoring")
```

**4e.** Remove the entire `heatmaps_t_orig` block (the storage and initial-blur section, currently lines 171-181). The fixed-blur path in `run_optimization` applies blur to heatmaps before they are passed in (in `run_sweep_config()`), so the dynamic re-blur inside the optimizer is only needed for the schedule, which is gone.

Apply fixed blur inside `run_optimization` directly: after building `heatmaps_t`, if `cfg.HEATMAP_BLUR_SIGMA > 0`, apply blur once before the optimization loop:
```python
if heatmaps_t is not None and cfg.HEATMAP_BLUR_SIGMA > 0:
    heatmaps_t = _apply_blur_torch(heatmaps_t, cfg.HEATMAP_BLUR_SIGMA)
```

**4f.** Remove the `current_blur_sigma` variable and the per-step schedule update block (the `if heatmaps_t_orig is not None` block inside the training loop, currently around lines 231-239).

**4g.** Replace the `_get_sigma(step, num_steps)` call (currently line 242) with simply `cfg.SIGMA`.

**4h.** Remove the `use_real_heatmaps=cfg.USE_REAL_HEATMAPS` argument from the `compute_total_score()` call.

**4i.** Simplify the step logging line to remove `blur={current_blur_sigma:.1f}`. The relevant `print` statement (around line 271) currently reads:
```python
f"loss={loss.item():.1f}  "
f"heatmap={details['heatmap']:.1f}  "
f"sigma={sigma:.0f}  "
f"blur={current_blur_sigma:.1f}  "
f"pos_p={details['pos_penalty']:.4f}  "
f"rot_p={details['rot_penalty']:.4f}"
```
Change to:
```python
f"loss={loss.item():.1f}  "
f"heatmap={details['heatmap']:.1f}  "
f"sigma={sigma:.0f}  "
f"pos_p={details['pos_penalty']:.4f}  "
f"rot_p={details['rot_penalty']:.4f}"
```
Where `sigma` is now just `cfg.SIGMA` (a constant).

### Step 5: Simplify `scoring.py`

**5a.** Remove the `use_real_heatmaps: bool = False` parameter from `compute_total_score()`.

**5b.** Inside `compute_total_score()`, simplify the `_use_real` boolean. Replace:
```python
_use_real: bool = (
    use_real_heatmaps
    and heatmaps_list is not None
    and affine is not None
)
```
with:
```python
_use_real: bool = (
    heatmaps_list is not None
    and affine is not None
)
```

**5c.** Update the docstring for `compute_total_score()` to remove the `use_real_heatmaps` parameter description.

### Step 6: Update `main.py`

**6a.** On the import line (line 21), change:
```python
from detect import detect_poses, motionbert_to_camera_space_batch
```
to:
```python
from detect import detect_poses, motionbert_to_camera_space
```

**6b.** Replace the single batch call (lines 171-175) with a per-frame loop:
```python
det_cam_positions: list[np.ndarray] = []
for i in range(len(frames_rgb)):
    pos_cam: np.ndarray = motionbert_to_camera_space(
        positions_3d_norm[i], kp_2d[i], scale, fx, fy, cx, cy,
        dist_coeffs=dist_coeffs,
        visibility=visibility[i],
    )
    det_cam_positions.append(pos_cam)
```

**6c.** Remove the `improved_target_2d` loop (lines 208-215):
```python
improved_target_2d: list[np.ndarray] = []
for i in range(len(frames_rgb)):
    target: np.ndarray = kp_2d[i].copy()
    mb_projected: np.ndarray = camera.world_to_image(det_cam_positions[i])
    for j in range(NUM_JOINTS):
        if visibility[i][j] < cfg.FK_TARGET_CONF_THRESHOLD:
            target[j] = mb_projected[j]
    improved_target_2d.append(target)
```

**6d.** In the `run_optimization()` call, replace `target_2d=improved_target_2d` with `target_2d=kp_2d`.

**6e.** In the `compute_comparison_with_optimization()` call, replace `detections_2d=improved_target_2d` with `detections_2d=kp_2d`.

**6f.** The step printout that references `improved_target_2d` is the overlay video call at the bottom. There is no reference to `improved_target_2d` in the overlay or graph calls -- only in `run_optimization` and `compute_comparison_with_optimization`, which are already updated above.

### Step 7: Update `sweep.py`

**7a.** In the import line (line 18), change:
```python
from detect import detect_poses, motionbert_to_camera_space_batch
```
to:
```python
from detect import detect_poses, motionbert_to_camera_space
```

**7b.** In `SweepConfig` dataclass, remove the `sigma_schedule` field:
```python
sigma_schedule: list[tuple[float, float]] = field(
    default_factory=lambda: [(1.0, 80.0)]
)
```
This field was always set to the constant `[(1.0, 80.0)]` in every config and is now unused.

**7c.** In `run_sweep_config()`, remove the `orig_sigma_schedule` save/restore and assignment:
- Remove: `orig_sigma_schedule = cfg.SIGMA_SCHEDULE`
- Remove: `cfg.SIGMA_SCHEDULE = config.sigma_schedule`
- Remove: `cfg.SIGMA_SCHEDULE = orig_sigma_schedule` (in the `finally` block)

**7d.** In `load_example()`, replace the batch camera-space conversion call (lines 103-107):
```python
det_cam_positions = motionbert_to_camera_space_batch(
    positions_3d_norm, kp_2d, scale, fx, fy, cx, cy,
    dist_coeffs=dist_coeffs,
    visibility_list=visibility,
)
```
with a per-frame loop:
```python
det_cam_positions: list[np.ndarray] = []
for i in range(len(frames_rgb)):
    pos_cam = motionbert_to_camera_space(
        positions_3d_norm[i], kp_2d[i], scale, fx, fy, cx, cy,
        dist_coeffs=dist_coeffs,
        visibility=visibility[i],
    )
    det_cam_positions.append(pos_cam)
```

**7e.** In `load_example()`, remove the `improved_target_2d` loop (lines 110-117):
```python
improved_target_2d: list[np.ndarray] = []
for i in range(len(frames_rgb)):
    target = kp_2d[i].copy()
    mb_projected = camera.world_to_image(det_cam_positions[i])
    for j in range(NUM_JOINTS):
        if visibility[i][j] < cfg.FK_TARGET_CONF_THRESHOLD:
            target[j] = mb_projected[j]
    improved_target_2d.append(target)
```

**7f.** In the `return` dict of `load_example()`, change `"improved_target_2d": improved_target_2d` to `"target_2d": kp_2d` (or just remove the `improved_target_2d` key and add `"target_2d": kp_2d`).

**7g.** In `run_sweep_config()`, change all references from `data["improved_target_2d"]` to `data["target_2d"]`. There are two:
- Line 187: `target_2d=data["improved_target_2d"]` in `run_optimization()`
- Line 201: `detections_2d=data["improved_target_2d"]` in `compute_comparison_with_optimization()`

**7h.** In `get_phase1_1_configs()` and `get_phase1_2_configs()`, remove `sigma_schedule` from any `SweepConfig` constructors that set it explicitly. Looking at the code, neither function currently sets `sigma_schedule` explicitly (they use the dataclass default), so no changes are needed in these functions -- but the field removal in 7b will cleanly handle this.

**7i.** In `run_sweep_config()`, also remove `heatmap_blur_schedule` from `run_optimization()` call... wait, actually `run_optimization()` in `optimize.py` still needs to accept the `heatmap_blur_schedule` parameter from `sweep.py`'s `run_sweep_config()` at line 193 (`heatmap_blur_schedule=config.heatmap_blur_schedule`). Let me clarify the correct approach:

Since `HEATMAP_BLUR_SCHEDULE` is removed from config, but sweep.py's `SweepConfig` still has `heatmap_blur_schedule` as a field (the sweep tests different blur modes), the `run_optimization()` should NOT have this parameter removed. Instead:

**Correction to Step 4c:** Do NOT remove `heatmap_blur_schedule` from `run_optimization()`. The sweep still needs to test dynamic schedules via `SweepConfig.heatmap_blur_schedule`. What is removed is only `cfg.HEATMAP_BLUR_SCHEDULE` from config.py (the global default). The per-sweep dynamic schedule is still a valid feature of the sweep harness and `run_optimization()`.

**Correction to Step 4e/4f:** Keep the `heatmaps_t_orig` storage and the per-step `_get_blur_sigma` call in place; they are still needed for sweep. Also keep `_get_blur_sigma()`. The simplification is:
- Remove `_get_sigma()` only (the sigma scheduler).
- Keep `_get_blur_sigma()`.
- Keep the dynamic schedule update block.
- Keep `heatmap_blur_schedule` parameter in `run_optimization()`.
- The fixed-blur-from-config (`cfg.HEATMAP_BLUR_SIGMA`) should be applied in the new single location: if no schedule is passed to `run_optimization()`, apply `cfg.HEATMAP_BLUR_SIGMA` once at startup.

Revised Step 4e: After building `heatmaps_t`, add a single-application of the fixed blur:
```python
if heatmaps_t is not None and heatmap_blur_schedule is None and cfg.HEATMAP_BLUR_SIGMA > 0:
    heatmaps_t = _apply_blur_torch(heatmaps_t, cfg.HEATMAP_BLUR_SIGMA)
```
Then the `heatmaps_t_orig` block that follows (for schedule use) should only run when `heatmap_blur_schedule is not None`:
```python
heatmaps_t_orig: list[torch.Tensor] | None = None
if heatmaps_t is not None and heatmap_blur_schedule is not None:
    heatmaps_t_orig = [hm.clone() for hm in heatmaps_t]
    initial_blur = _get_blur_sigma(0, num_steps, heatmap_blur_schedule)
    if initial_blur > 0:
        heatmaps_t = _apply_blur_torch(heatmaps_t_orig, initial_blur)
```
This is actually identical to the existing logic -- the only change is that the `cfg.HEATMAP_BLUR_SIGMA` application now happens inside `run_optimization` rather than outside in `sweep.py`'s `run_sweep_config()`. **This means the fixed-blur path in `run_sweep_config()` can also be removed**, since `run_optimization` will apply it. Update `run_sweep_config()`:

Remove the existing fixed-blur pre-processing block in `run_sweep_config()` (lines 171-183 in sweep.py):
```python
heatmaps = data["heatmaps"]
if config.heatmap_blur_sigma > 0 and config.heatmap_blur_schedule is None:
    # Fixed blur only if no dynamic schedule
    import scipy.ndimage
    heatmaps = [
        np.stack([
            scipy.ndimage.gaussian_filter(hm[c], sigma=config.heatmap_blur_sigma)
            for c in range(hm.shape[0])
        ])
        for hm in heatmaps
    ]
```

And instead, make `run_sweep_config()` temporarily override `cfg.HEATMAP_BLUR_SIGMA` just like it overrides other cfg fields:
- Save: `orig_blur_sigma = cfg.HEATMAP_BLUR_SIGMA`
- Set: `cfg.HEATMAP_BLUR_SIGMA = config.heatmap_blur_sigma`
- Restore in finally: `cfg.HEATMAP_BLUR_SIGMA = orig_blur_sigma`

This is cleaner and consistent with how all other cfg overrides work in the sweep harness.

### Step 8: Test on Examples 0 and 5

Run the pipeline on Example 0:
```bash
cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-pairwise-revert')
os.makedirs(run_dir, exist_ok=True)
seq, cam, start, nf, pidx = cfg.EXAMPLES[0]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```

Run the pipeline on Example 5:
```bash
cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-pairwise-revert')
seq, cam, start, nf, pidx = cfg.EXAMPLES[5]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```

### Step 9: Verify sweep.py loads without crashes

```bash
cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python -c "
from sweep import SweepConfig, get_phase1_2_configs
configs = get_phase1_2_configs()
print(f'{len(configs)} configs')
assert not hasattr(configs[0], 'sigma_schedule'), 'sigma_schedule not removed'
assert not hasattr(configs[0], 'all_joints_smooth_weight'), 'old field present'
print('sweep.py loads OK')
"
```

### Step 10: Report comparison metrics

The developer report should include:

| Metric | Round-4 (pairwise) | P1-00 (raw solvePnP) | P1-01 (smoothed PnP) | P1-02 (pairwise revert) |
|--------|--------------------|-----------------------|----------------------|-------------------------|
| Det MPJPE (cm) | 30.98 | 34.72 | 34.72 | ? |
| Opt MPJPE (cm) | 30.44 | 34.12 | 33.90 | ? |
| Det P-MPJPE (cm) | 28.57 | 28.57 | 28.57 | ? |
| Opt P-MPJPE (cm) | 28.42 | 28.30 | 28.31 | ? |
| Det MPJVE (cm/f) | 0.94 | 3.69 | 3.69 | ? |
| Opt MPJVE (cm/f) | 1.05 | 3.38 | 3.37 | ? |
| Root Z range (det) | -- | 1.44-2.45 m | 2.21-2.38 m | ? |

For Example 5 as well.

---

## Integration Points

- `motionbert_to_camera_space()` returns a single `(16, 3)` array -- identical to what Phase 0 returned. All downstream code (optimize, evaluate, graphs, overlay video) is unchanged.
- `compute_total_score()` loses the `use_real_heatmaps` parameter. The only caller is `optimize.py` (line 245-258), which is also being updated.
- `run_optimization()` keeps the `heatmap_blur_schedule` parameter for sweep compatibility.
- `sweep.py` passes `target_2d=data["target_2d"]` (was `improved_target_2d`). `compute_comparison_with_optimization()` uses this for 2D-vs-detection reprojection error, so using raw `kp_2d` here is correct.

---

## Risks and Edge Cases

### Risk 1: Pairwise tz uses all joint pairs -- potential for degenerate geometry
For poses where all joints happen to have very similar Y pixel coordinates (e.g., person viewed from above, extreme camera angle), there may be no pairs with `dv_2d > 5.0`. The fallback `tz = 3.0` handles this gracefully.

### Risk 2: The `run_optimization()` sigma parameter
After removing `_get_sigma()`, the `sigma` variable in the optimization loop becomes just `cfg.SIGMA` (a constant). This is a simplification that matches the current config (which has `SIGMA_SCHEDULE = [(1.0, 80.0)]` -- constant 80.0 throughout). Note that `cfg.SIGMA = 50.0` but the schedule was using 80.0. After this change, the optimizer will use 50.0. This is fine -- 50.0 is the intended value; the schedule was effectively locked at a single value anyway, and the single constant in config.py is what the user wants to use.

### Risk 3: Fixed blur now applied inside `run_optimization()` not outside it
Previously `sweep.py`'s `run_sweep_config()` applied the fixed blur before calling `run_optimization()`. After this change, `run_optimization()` applies it internally based on `cfg.HEATMAP_BLUR_SIGMA`. This requires that `run_sweep_config()` temporarily overrides `cfg.HEATMAP_BLUR_SIGMA` (as described in Step 7). If this override is missed, all sweep configs will use the global default blur sigma (0.0) instead of the per-config value. The save/restore pattern is already established in `run_sweep_config()` for other parameters; this is low risk.

### Risk 4: `dist_coeffs` parameter on `motionbert_to_camera_space()`
The parameter is kept for interface compatibility but is unused in the pairwise approach. Callers (main.py and sweep.py) pass it but it is silently ignored. This is fine and avoids a cascade of call-site changes.

### Risk 5: `scoring.py` docstring for `compute_total_score()`
The docstring still mentions the analytical Gaussian path in the module-level docstring (lines 1-8). The developer should update or simplify the module docstring to reflect that only real heatmaps are used now.

---

## Expected Outcomes

The pairwise approach was the Phase 0 baseline. With the exact same pairwise logic and the best blur config from P1-01 (which showed 30.00 cm MPJPE at 100 steps with sigma=8 blur), we expect:

- **Det MPJPE**: ~30-31 cm (matches or slightly improves on round-4's 30.98 cm)
- **Opt MPJPE**: ~30 cm (matches round-4's 30.44 cm, potentially better with blur)
- **Det MPJVE**: ~0.94 cm/f (recovers to round-4 level -- no rotation jitter from solvePnP)
- **Opt MPJVE**: ~1.05 cm/f (recovers to round-4 level)

The spec requirement ("MPJVE should be improved for ALL test videos") should be achievable from this cleaner baseline. The simplification also removes ~200 lines of code and eliminates 4 config variables.

---

## Success Criteria

1. `sweep.py` imports and runs without crashes.
2. Det MPJVE recovers to <= round-4 levels for both Example 0 and Example 5.
3. Det MPJPE does not regress vs round-4 by more than 1 cm.
4. All removed config constants (`SIGMA_SCHEDULE`, `HEATMAP_BLUR_SCHEDULE`, `FK_TARGET_CONF_THRESHOLD`, `USE_REAL_HEATMAPS`) are gone from `config.py`.
5. `motionbert_to_camera_space_batch` is gone from `detect.py`.
