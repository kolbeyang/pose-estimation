# Architect Plan: Phase 1, Iteration 1 -- Fix solvePnP Depth Jitter + Clean Up sweep.py

**Date:** 2026-03-17
**Spec:** `claude-team/specs/motion-bert-round-5.md` (Phase 1)
**Prior iteration:** P1-00

## Issues to Address

### From TESTER_REPORT_P1_00.md

**BUG-1 (Medium):** `sweep.py` still references removed `ALL_JOINTS_SMOOTH_WEIGHT` at lines 40, 163, 170, 223, 399. Running `sweep.py` will crash with `AttributeError`.

**BUG-2 (HIGH, blocks spec):** solvePnP produces WORSE MPJVE and MPJPE than the previous pairwise method. Per-frame solvePnP is inherently noisy -- each frame gets an independent depth solve. Example 0 showed:
- MPJVE regressed 1.05 -> 3.38 cm/f (3.2x worse)
- Root Z range 1.44-2.45m (detection) vs GT 2.57-2.63m (0.06m range)
- Root Z velocity std 0.076 m/f vs GT 0.003 m/f (23x worse)
- Catastrophic ~1m Z drop around frames 40-45
- P-MPJPE unchanged, confirming issue is purely depth/translation, not pose shape

**Spec requirement not met:** "MPJVE should be improved for ALL test videos."

## Root Cause Analysis

The solvePnP approach is correct in principle (the mediapipe-pose pipeline uses the same technique successfully), but it has a critical flaw in this context: **MotionBERT's root-relative 3D structure is not perfectly consistent frame-to-frame**, so per-frame solvePnP produces independent, noisy depth estimates. The mediapipe-pose pipeline gets away with this because (a) MediaPipe's 3D structure is more stable, and (b) the mediapipe optimization runs for 300 steps with stronger temporal penalties.

The real problem is that per-frame solvePnP has no temporal context. The previous pairwise method was actually more stable because it used IQR filtering across many joint pairs, producing a smoother depth signal.

## Strategy: Temporal Smoothing of solvePnP Output

Rather than reverting to the pairwise method (which the spec asked us to replace), we will **keep solvePnP but apply temporal smoothing to the translation vector (tvec) before using it**. This preserves the spec's intent ("solvePnP to get everything into camera coordinates") while fixing the jitter.

The approach:
1. Run solvePnP on every frame independently (as currently implemented)
2. Collect the per-frame tvec (translation) and rvec (rotation) estimates
3. Apply a **running median filter** to tz (and optionally tx, ty) across the temporal window
4. Use the smoothed translations with the per-frame rotations to produce camera-space positions

Why running median (not mean): The median is robust to outliers. The catastrophic ~1m Z drop at frame 40-45 in Example 0 is exactly the kind of outlier a median filter handles well -- a 5-frame window will ignore the spike.

This requires restructuring `motionbert_to_camera_space` from a per-frame function into a batch function (or adding a new batch wrapper), since smoothing requires seeing multiple frames at once.

## Files to Modify

### 1. `motionbert-pose/detect.py`
- Add a new function `motionbert_to_camera_space_batch()` that processes all frames at once, runs solvePnP per frame, then applies temporal median smoothing to the translations before producing camera-space output
- Keep the existing `motionbert_to_camera_space()` as-is for backward compatibility

### 2. `motionbert-pose/main.py`
- Replace the per-frame `motionbert_to_camera_space()` loop with a single call to the new `motionbert_to_camera_space_batch()`

### 3. `motionbert-pose/sweep.py`
- Fix BUG-1: Remove all references to `ALL_JOINTS_SMOOTH_WEIGHT`
- Update to use batch camera-space conversion
- Fix `_ROT_MULTIPLIERS` array which has 17 entries (should be 16 after Head removal)
- Fix `range(17)` in `load_example` (should be `range(NUM_JOINTS)`)

### 4. `motionbert-pose/config.py`
- No changes needed

## Files to Create

None.

## Step-by-Step Instructions

### Step 1: Fix BUG-1 -- Remove ALL_JOINTS_SMOOTH_WEIGHT from sweep.py

In `motionbert-pose/sweep.py`:

1a. Remove the `all_joints_smooth_weight` field from the `SweepConfig` dataclass (line 40):
```python
# DELETE this line:
all_joints_smooth_weight: float = 0.0
```

1b. In `run_sweep_config()`, remove these three lines:
- Line 163: `orig_smooth = cfg.ALL_JOINTS_SMOOTH_WEIGHT` -- DELETE
- Line 170: `cfg.ALL_JOINTS_SMOOTH_WEIGHT = config.all_joints_smooth_weight` -- DELETE
- Line 223: `cfg.ALL_JOINTS_SMOOTH_WEIGHT = orig_smooth` -- DELETE

1c. In the print statement around line 399, remove the `smooth=` portion. Change:
```python
f"anchor={config.init_anchor_weight}, smooth={config.all_joints_smooth_weight}, "
```
To:
```python
f"anchor={config.init_anchor_weight}, "
```

### Step 2: Fix _ROT_MULTIPLIERS in sweep.py

The `_ROT_MULTIPLIERS` array (lines 49-52) currently has 17 entries (includes the old Head joint). After the Head removal in P0, the skeleton has 16 joints. The current incorrect array:
```python
_ROT_MULTIPLIERS: list[float] = [
    3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2,
    1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1,
]
```

This has 17 entries. Remove the extra entry at position 10 (old Head, multiplier 0.5). Comparing with `config.py` lines 76-96, the correct 16-entry array matching the current skeleton is:
```python
_ROT_MULTIPLIERS: list[float] = [
    3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2,  # Hip, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle
    1.0, 1.0, 0.5,                         # Spine, Thorax, Neck
    0.5, 0.3, 0.1, 0.5, 0.3, 0.1,         # LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist
]
```

### Step 3: Fix range(17) in sweep.py load_example

In `load_example()`, line 122 has `for j in range(17)`. Change to `for j in range(NUM_JOINTS)`.

Add `NUM_JOINTS` to the imports. Currently line 18 does not import `NUM_JOINTS` from skeleton. Add it -- either at the top of the file or locally in `load_example`. Since `NUM_JOINTS` is also used in detect.py already, import it at the top:

Add to the existing imports (or create a new import line):
```python
from skeleton import NUM_JOINTS
```

### Step 4: Add motionbert_to_camera_space_batch() to detect.py

Add the following function after the existing `motionbert_to_camera_space()` function (after line 722):

```python
def motionbert_to_camera_space_batch(
    positions_3d_norm: np.ndarray,
    kp_2d_list: list[np.ndarray],
    scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist_coeffs: np.ndarray | None = None,
    visibility_list: list[np.ndarray] | None = None,
    smooth_window: int = 5,
) -> list[np.ndarray]:
    """Convert MotionBERT normalized output to camera-space meters with temporal smoothing.

    Runs solvePnP per frame to estimate a rigid transform (rotation + translation),
    then applies a running median filter to the translation vector (tx, ty, tz) across
    frames to remove depth jitter. This produces much smoother trajectories than
    per-frame solvePnP alone, while preserving the per-frame rotation estimates.

    Args:
        positions_3d_norm: (N, 16, 3) normalized MotionBERT output (before pixel denorm).
        kp_2d_list: List of N (16, 2) 2D detections in pixel coordinates.
        scale: crop_scale scale parameter.
        fx, fy, cx, cy: Camera intrinsics.
        dist_coeffs: Optional distortion coefficients from camera calibration.
        visibility_list: Optional list of N (16,) confidence scores per frame.
        smooth_window: Size of the median filter window for temporal smoothing.
            Must be odd. Default 5 (covers ~0.5s at 10fps).

    Returns:
        List of N (16, 3) camera-space positions in meters.
    """
    from skeleton import PARENTS, DEFAULT_BONE_LENGTHS

    n_frames: int = positions_3d_norm.shape[0]

    K: np.ndarray = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    dc: np.ndarray = dist_coeffs if dist_coeffs is not None else np.zeros(4, dtype=np.float64)

    # Phase 1: Run solvePnP independently per frame to get raw rvec/tvec
    raw_rvecs: list[np.ndarray] = []
    raw_tvecs: list[np.ndarray] = []
    root_relatives_m: list[np.ndarray] = []
    pnp_success: list[bool] = []

    for i in range(n_frames):
        kp_2d: np.ndarray = kp_2d_list[i]
        vis: np.ndarray | None = visibility_list[i] if visibility_list is not None else None

        # Get root-relative structure in meters via bone-length matching
        root_relative: np.ndarray = positions_3d_norm[i] - positions_3d_norm[i, 0:1]

        reliable_ratios: list[float] = []
        all_ratios: list[float] = []
        for j in range(1, NUM_JOINTS):
            p: int = int(PARENTS[j])
            det_bl: float = float(np.linalg.norm(root_relative[j] - root_relative[p]))
            ref_bl: float = float(DEFAULT_BONE_LENGTHS[j])
            if det_bl > 1e-4 and ref_bl > 1e-4:
                ratio: float = ref_bl / det_bl
                all_ratios.append(ratio)
                if j in _RELIABLE_BONES_FOR_SCALE:
                    reliable_ratios.append(ratio)

        if len(reliable_ratios) >= 4:
            bone_scale: float = _iqr_filtered_median(np.array(reliable_ratios))
        elif all_ratios:
            bone_scale = _iqr_filtered_median(np.array(all_ratios))
        else:
            bone_scale = 1.0

        root_relative_m: np.ndarray = root_relative * bone_scale
        root_relatives_m.append(root_relative_m)

        # solvePnP
        valid: np.ndarray = np.linalg.norm(kp_2d, axis=1) > 1.0
        if vis is not None:
            valid = valid & (vis > 0.1)

        if valid.sum() >= 4:
            obj_pts: np.ndarray = root_relative_m[valid].astype(np.float64)
            img_pts: np.ndarray = kp_2d[valid].astype(np.float64)

            success: bool
            rvec: np.ndarray
            tvec: np.ndarray
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, dc, flags=cv2.SOLVEPNP_SQPNP,
            )

            if success:
                root_z: float = float(tvec[2, 0])
                if root_z > 0.5 and root_z < 15.0:
                    raw_rvecs.append(rvec.flatten())
                    raw_tvecs.append(tvec.flatten())
                    pnp_success.append(True)
                    continue

        # Fallback: estimate tvec from person-height heuristic
        tz: float = 3.0
        thorax_idx: int = 8
        ankle_mid_2d: np.ndarray = (kp_2d[3] + kp_2d[6]) / 2.0
        pixel_height: float = abs(kp_2d[thorax_idx, 1] - ankle_mid_2d[1])
        height_3d: float = float(np.linalg.norm(
            root_relative_m[thorax_idx] - (root_relative_m[3] + root_relative_m[6]) / 2
        ))
        if pixel_height > 20 and height_3d > 0.1:
            tz = fy * height_3d / pixel_height
            tz = float(np.clip(tz, 1.0, 8.0))

        u_root: float = float(kp_2d[0, 0])
        v_root: float = float(kp_2d[0, 1])
        if abs(u_root) > 1.0 or abs(v_root) > 1.0:
            tx: float = (u_root - cx) * tz / fx
            ty: float = (v_root - cy) * tz / fy
        else:
            tx = 0.0
            ty = 0.0

        raw_rvecs.append(np.zeros(3))  # identity rotation for fallback
        raw_tvecs.append(np.array([tx, ty, tz]))
        pnp_success.append(False)

    # Phase 2: Temporal smoothing of translations via running median filter
    from scipy.ndimage import median_filter

    raw_tvecs_arr: np.ndarray = np.array(raw_tvecs)  # (N, 3)
    smoothed_tvecs: np.ndarray = np.zeros_like(raw_tvecs_arr)
    for dim in range(3):
        smoothed_tvecs[:, dim] = median_filter(
            raw_tvecs_arr[:, dim], size=smooth_window, mode='nearest'
        )

    # Phase 3: Apply smoothed translations to produce camera-space positions
    result: list[np.ndarray] = []
    for i in range(n_frames):
        rr_m: np.ndarray = root_relatives_m[i]

        if pnp_success[i]:
            # Use per-frame rotation but smoothed translation
            R_pnp: np.ndarray
            R_pnp, _ = cv2.Rodrigues(raw_rvecs[i])
            cam_3d: np.ndarray = (R_pnp @ rr_m.T).T + smoothed_tvecs[i]
        else:
            # Fallback frame: translate only (identity rotation)
            cam_3d = rr_m.copy()
            cam_3d += smoothed_tvecs[i]

        # Enforce bone-length constraints
        cam_root: np.ndarray = cam_3d[0].copy()
        cam_rr: np.ndarray = cam_3d - cam_root
        cam_rr_fixed: np.ndarray = _enforce_bone_lengths(
            cam_rr, PARENTS, DEFAULT_BONE_LENGTHS, max_ratio=1.3,
        )
        cam_3d = cam_rr_fixed + cam_root
        result.append(cam_3d.astype(np.float64))

    # Diagnostics
    n_pnp: int = sum(pnp_success)
    print(f"    solvePnP: {n_pnp}/{n_frames} frames succeeded")
    print(f"    Raw root Z range:      {raw_tvecs_arr[:, 2].min():.2f} to {raw_tvecs_arr[:, 2].max():.2f} m")
    print(f"    Smoothed root Z range: {smoothed_tvecs[:, 2].min():.2f} to {smoothed_tvecs[:, 2].max():.2f} m")
    raw_z_std: float = float(np.std(raw_tvecs_arr[:, 2]))
    smooth_z_std: float = float(np.std(smoothed_tvecs[:, 2]))
    print(f"    Raw Z std: {raw_z_std:.4f} m, Smoothed Z std: {smooth_z_std:.4f} m "
          f"({raw_z_std / max(smooth_z_std, 1e-6):.1f}x reduction)")

    return result
```

**Key design decisions explained:**

1. **`smooth_window=5`**: A 5-frame median window. At the video's effective frame rate (30fps native, possibly subsampled), 5 frames covers a short enough window to track genuine motion but long enough to filter single-frame outliers like the ~1m Z drop at frame 40-45 in Example 0. The tester showed GT root Z std was 0.016m -- a 5-frame median should bring the smoothed std well below the raw 0.147m.

2. **`mode='nearest'`**: Edge handling for `scipy.ndimage.median_filter`. Extends the first/last values rather than padding with zeros, which would pull the depth toward 0 at sequence boundaries.

3. **Only translation is smoothed, NOT rotation**: The tester confirmed P-MPJPE was unchanged (pose shape is correct). The per-frame rotation from solvePnP captures legitimate camera-skeleton alignment and should be preserved. Only the translation (global position, especially depth) needs smoothing.

4. **Bone-scale is computed per-frame**: MotionBERT's root-relative structure varies slightly frame-to-frame, so the bone-length-matching scale factor should be per-frame. This is the same behavior as the existing per-frame function.

5. **`scipy.ndimage.median_filter` is already a project dependency** (scipy is used for gaussian_filter in optimize.py and sweep.py).

### Step 5: Update main.py to use batch conversion

In `main.py`:

5a. Update the import on line 21. Change:
```python
from detect import detect_poses, motionbert_to_camera_space
```
To:
```python
from detect import detect_poses, motionbert_to_camera_space_batch
```

5b. Replace the per-frame loop at lines 170-178. Change:
```python
    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        dist_coeffs: np.ndarray | None = cam_calib.get("distCoef")
        pos_cam: np.ndarray = motionbert_to_camera_space(
            positions_3d_norm[i], kp_2d[i], scale, fx, fy, cx, cy,
            dist_coeffs=dist_coeffs,
            visibility=visibility[i],
        )
        det_cam_positions.append(pos_cam)
```
To:
```python
    dist_coeffs: np.ndarray | None = cam_calib.get("distCoef")
    det_cam_positions: list[np.ndarray] = motionbert_to_camera_space_batch(
        positions_3d_norm, kp_2d, scale, fx, fy, cx, cy,
        dist_coeffs=dist_coeffs,
        visibility_list=visibility,
    )
```

### Step 6: Update sweep.py to use batch conversion

In `sweep.py`:

6a. Update the import on line 18. Change:
```python
from detect import detect_poses, motionbert_to_camera_space
```
To:
```python
from detect import detect_poses, motionbert_to_camera_space_batch
```

6b. In `load_example()`, replace the per-frame loop at lines 99-115. Change:
```python
    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        dist_coeffs = cam_calib.get("distCoef")
        pos_cam = motionbert_to_camera_space(
            positions_3d_norm[i],
            kp_2d[i],
            scale,
            fx,
            fy,
            cx,
            cy,
            dist_coeffs=dist_coeffs,
            visibility=visibility[i],
        )
        det_cam_positions.append(pos_cam)
```
To:
```python
    dist_coeffs = cam_calib.get("distCoef")
    det_cam_positions = motionbert_to_camera_space_batch(
        positions_3d_norm, kp_2d, scale, fx, fy, cx, cy,
        dist_coeffs=dist_coeffs,
        visibility_list=visibility,
    )
```

Note: `positions_3d_norm` from `detect_poses()` is already `(N, 16, 3)` -- exactly what the batch function expects.

### Step 7: Test on examples 0 and 5

Run the pipeline on Example 0 and Example 5:

```bash
cd motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os

run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-solvepnp-smooth')
os.makedirs(run_dir, exist_ok=True)
seq, cam, start, nf, pidx = cfg.EXAMPLES[0]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```

```bash
cd motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os

run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-solvepnp-smooth')
seq, cam, start, nf, pidx = cfg.EXAMPLES[5]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```

### Step 8: Verify sweep.py runs without crashes

```bash
cd motionbert-pose && uv run python -c "
from sweep import SweepConfig, get_phase1_2_configs, load_example
configs = get_phase1_2_configs()
print(f'{len(configs)} configs')
print(f'Config fields: {list(configs[0].__dict__.keys())}')
assert not hasattr(configs[0], 'all_joints_smooth_weight'), 'BUG-1 not fixed'
print('sweep.py loads correctly, BUG-1 fixed')
"
```

### Step 9: Report comparison metrics

The developer should report in their report the following table for both examples:

| Metric | Round-4 (pairwise) | P1-00 (raw solvePnP) | P1-01 (smoothed) |
|--------|--------------------|-----------------------|-------------------|
| Det MPJPE (cm) | | | |
| Opt MPJPE (cm) | | | |
| Det P-MPJPE (cm) | | | |
| Opt P-MPJPE (cm) | | | |
| Det MPJVE (cm/f) | | | |
| Opt MPJVE (cm/f) | | | |
| Root Z range (det) | | | |
| Root Z std (det) | | | |
| Raw vs Smoothed Z std | N/A | N/A | (from diagnostics) |

The round-4 baseline numbers are (from DEVELOPER_REPORT_RUN_ALL.md):
- Example 0: Det MPJPE 30.98, Opt MPJPE 30.44, Det MPJVE 0.94, Opt MPJVE 1.05
- Example 5: Det MPJPE 15.85, Opt MPJPE 15.44, Det MPJVE 0.51, Opt MPJVE 0.48

The P1-00 numbers are (from DEVELOPER_REPORT_P1_00.md):
- Example 0: Det MPJPE 34.72, Opt MPJPE 34.12, Det MPJVE 3.69, Opt MPJVE 3.38
- Example 5: Det MPJPE 18.26, Opt MPJPE 18.65, Det MPJVE 0.93, Opt MPJVE 0.54

## Expected Outcomes

| Metric | Round-4 | P1-00 (raw PnP) | P1-01 Target |
|--------|---------|------------------|--------------|
| Ex0 Det MPJPE (cm) | 30.98 | 34.72 | ~30-32 |
| Ex0 Opt MPJVE (cm/f) | 1.05 | 3.38 | < 1.05 |
| Ex0 Root Z std (det) | -- | 0.147 m | < 0.05 m |
| Ex5 Opt MPJVE (cm/f) | 0.48 | 0.54 | < 0.48 |

The median filter should:
- **Eliminate the ~1m Z drop** at frame 40-45 (5-frame median rejects single-frame outliers)
- **Reduce Z std by 3-5x** (from 0.147m toward ~0.03-0.05m)
- **Recover MPJPE** to near round-4 levels (smoother depth = better global alignment)
- **Improve MPJVE** beyond round-4 (solvePnP gives better global placement than pairwise method, smoothing removes jitter)

## Integration Points

- `motionbert_to_camera_space_batch()` takes `positions_3d_norm` as `(N, 16, 3)` -- this matches the output of `detect_poses()` via `run_motionbert()`.
- The existing per-frame `motionbert_to_camera_space()` is NOT removed -- it remains available if any code path still needs per-frame conversion.
- The batch function returns `list[np.ndarray]` with each element `(16, 3)` -- identical type to what the per-frame loop produced, so all downstream code (optimization, evaluation, graphs, overlay video) works unchanged.
- `scipy.ndimage.median_filter` is part of scipy which is already a project dependency (used for `gaussian_filter` in optimize.py).

## Risks and Edge Cases

### Risk 1: Median filter window too large for short sequences
For sequences with fewer than 5 frames, the 5-frame median window covers the entire sequence, effectively smoothing to a near-constant depth. This is acceptable -- for very short sequences, constant depth is a reasonable assumption, and the optimizer can still adjust root_pos freely.

### Risk 2: Smoothing might lag behind genuine depth changes
If the person walks toward/away from the camera, the median filter introduces a ~2-frame lag (half the window). At 10fps effective rate, this is 0.2 seconds -- negligible. The optimizer's root position parameters have full freedom to adjust depth during optimization, so any lag in the initialization is correctable.

### Risk 3: Per-frame rotation not smoothed
We smooth only translation, not rotation. If solvePnP rotation estimates are noisy, this could cause residual jitter in joint positions (since the rotation rotates the entire root-relative structure). However, the tester showed P-MPJPE was unchanged, meaning the rotation is consistent enough. If this becomes an issue in a future iteration, rotation smoothing (e.g., averaging quaternions in a sliding window) can be added.

### Risk 4: The batch function has a different interface than the per-frame function
`motionbert_to_camera_space_batch` takes `(N, 16, 3)` for positions_3d_norm and lists for kp_2d/visibility, while `motionbert_to_camera_space` takes single-frame `(16, 3)` inputs. Both main.py and sweep.py must be updated. The plan covers both.

### Risk 5: _ROT_MULTIPLIERS mismatch could silently produce wrong results
The 17-entry `_ROT_MULTIPLIERS` in sweep.py being used with the 16-entry skeleton will produce a 17-element rotation penalty array when multiplied by the scalar. This will cause a shape mismatch when `ROTATION_PENALTY_PER_JOINT` is used in the optimization (which expects 16 entries). This could either crash or produce incorrect penalties. Step 2 fixes this.

## Success Criteria

1. `sweep.py` imports and runs without crashes (BUG-1 fixed, _ROT_MULTIPLIERS fixed)
2. **MPJVE improves vs round-4 baseline for both tested examples** (spec requirement: "MPJVE should be improved for ALL test videos")
3. MPJPE does not regress significantly vs round-4 (< 2cm regression acceptable)
4. Root Z std is dramatically reduced vs P1-00 raw solvePnP (target: < 0.05m vs P1-00's 0.147m)
5. Overlay video shows stable skeleton placement with no visible depth jumps
