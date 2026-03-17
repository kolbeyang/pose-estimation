# Architect Plan: Phase 1, Iteration 0 -- Smooth Motion via solvePnP + Remove ALL_JOINTS_SMOOTH_WEIGHT

**Date:** 2026-03-17
**Spec:** `claude-team/specs/motion-bert-round-5.md` (Phase 1)
**Round:** 5

## Goal Summary

The spec asks for two changes:

1. **Remove `ALL_JOINTS_SMOOTH_WEIGHT` and all code that uses it.** Temporal smoothness should come solely from the root position penalty and rotation change penalties -- smooth rotations imply smooth positions for all downstream joints.

2. **Use OpenCV `solvePnP` to estimate depth (and full camera-space placement) of the MotionBERT output BEFORE optimization, replacing the current `motionbert_to_camera_space()` pairwise-separation depth estimation.** This mirrors what the mediapipe-pose pipeline does in `mediapipe_3d_to_camera()`. The result is that the optimizer only ever works in camera coordinates and has full freedom to smooth the z values during optimization.

The testing requirement is: run on a simple example, pull out a heatmap video frame, analyze z values, and confirm MPJVE improves.

## Depth Estimation Strategy (Explanation Required by Spec)

### Current approach (to be replaced)

`motionbert_to_camera_space()` in `detect.py` currently:
1. Gets root-relative 3D structure from MotionBERT's normalized output
2. Scales it to meters using bone-length ratio matching (reliable arm bones with IQR filtering)
3. Estimates depth `tz` from pairwise joint separation ratios (3D distance / 2D pixel distance) with IQR filtering
4. Computes `tx, ty` from 2D projections at estimated depth
5. Enforces bone-length constraints by clamping extreme bones

This approach is fragile: the pairwise separation method produces noisy depth estimates, especially for challenging poses, and the bone-length clamping is a band-aid.

### New approach: solvePnP (modeled on mediapipe-pose)

The mediapipe-pose pipeline (`mediapipe-pose/detect.py`, function `mediapipe_3d_to_camera()`) does this:
1. Takes MediaPipe's hip-relative 3D predictions as "object points" (known 3D shape)
2. Takes the 2D pixel detections as "image points"
3. Calls `cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)` which solves for the rigid transform (rotation R + translation t) that best explains the 2D observations given the 3D shape
4. Applies `cam_3d = (R @ obj_pts.T).T + t.T` to get all joints in camera space

For motionbert-pose, we will do the same:
1. Take MotionBERT's root-relative 3D output (already scaled to meters via bone-length matching, which is the good part of the current code)
2. Use the 2D Stacked Hourglass keypoints as image points
3. Call `cv2.solvePnP` with SQPNP to solve for the rigid transform
4. Apply the transform to get camera-space positions

**Why this is better:** solvePnP minimizes reprojection error globally across all visible joints simultaneously, producing a single consistent rigid placement. The current pairwise method estimates depth from individual joint pairs and takes a filtered median, which is inherently less constrained. solvePnP also naturally handles the rotation between MotionBERT's coordinate frame and the camera frame (MotionBERT is trained on H3.6M camera-space data so the rotation should be near-identity, but solvePnP will correct any residual misalignment).

**Fallback:** If fewer than 4 joints are visible or solvePnP fails, fall back to the existing depth heuristic (estimate tz from person height in pixels, as the mediapipe version does).

## Files to Modify

### 1. `motionbert-pose/config.py`
- Remove `ALL_JOINTS_SMOOTH_WEIGHT` constant

### 2. `motionbert-pose/scoring.py`
- Remove `motion_penalty_all_joints()` function
- Remove `all_joints_smooth_weight` parameter and related logic from `compute_total_score()`

### 3. `motionbert-pose/optimize.py`
- Remove the `all_joints_smooth_weight=cfg.ALL_JOINTS_SMOOTH_WEIGHT` argument from the `compute_total_score()` call

### 4. `motionbert-pose/detect.py`
- Replace the depth estimation portion of `motionbert_to_camera_space()` with solvePnP
- Remove unused helper functions `_enforce_bone_lengths_with_2d()` and `_reconstruct_from_2d()`

## Files to Create

None.

## Step-by-Step Instructions

### Step 1: Remove ALL_JOINTS_SMOOTH_WEIGHT from config.py

Delete this line (currently line 118):
```python
ALL_JOINTS_SMOOTH_WEIGHT: float = 0.0
```

Note: it is already 0.0, so this is purely a cleanup. But the spec explicitly asks for removal.

### Step 2: Remove motion_penalty_all_joints from scoring.py

2a. Delete the `motion_penalty_all_joints()` function entirely (lines 208-222).

2b. In `compute_total_score()`:
- Remove parameter `all_joints_smooth_weight: float = 0.0` from the function signature
- Remove `total_all_joints_smooth: torch.Tensor = torch.tensor(0.0)` initialization
- Remove the entire `if all_joints_smooth_weight > 0.0:` block inside the frame loop (the block that calls `motion_penalty_all_joints`)
- Remove `- all_joints_smooth_weight * total_all_joints_smooth` from the `total_score` calculation
- Remove `"all_joints_smooth": float(total_all_joints_smooth.item()),` from the `details` dict

### Step 3: Remove all_joints_smooth_weight from optimize.py

In `run_optimization()`, in the `compute_total_score()` call (around line 258), remove the keyword argument:
```python
all_joints_smooth_weight=cfg.ALL_JOINTS_SMOOTH_WEIGHT,
```

### Step 4: Implement solvePnP in motionbert_to_camera_space()

This is the main change. Replace the body of `motionbert_to_camera_space()` in `detect.py`.

**Keep unchanged:** The function signature, the bone-length scaling step (Step 1 of current code), and the `_iqr_filtered_median`, `_RELIABLE_BONES_FOR_SCALE`, `_enforce_bone_lengths` helpers.

**Replace:** Everything after `root_relative_m` is computed (currently line 853 onward).

The new implementation after `root_relative_m = root_relative * bone_scale`:

```python
    # Step 2: Use solvePnP to estimate rigid transform placing skeleton in camera space.
    # This mirrors the mediapipe-pose approach in mediapipe_3d_to_camera().
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    dc = dist_coeffs if dist_coeffs is not None else np.zeros(4, dtype=np.float64)

    # Filter to joints with valid 2D detections and sufficient confidence
    valid = np.linalg.norm(kp_2d, axis=1) > 1.0
    if visibility is not None:
        valid = valid & (visibility > 0.1)

    if valid.sum() >= 4:
        obj_pts = root_relative_m[valid].astype(np.float64)
        img_pts = kp_2d[valid].astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dc, flags=cv2.SOLVEPNP_SQPNP,
        )

        if success:
            R_pnp, _ = cv2.Rodrigues(rvec)
            cam_3d = (R_pnp @ root_relative_m.T).T + tvec.T

            # Safety check: root Z should be positive and reasonable
            root_z = float(cam_3d[0, 2])
            if root_z > 0.5 and root_z < 15.0:
                # Enforce bone-length constraints as safety clamp
                cam_root = cam_3d[0].copy()
                cam_rr = cam_3d - cam_root
                cam_rr_fixed = _enforce_bone_lengths(
                    cam_rr, PARENTS, DEFAULT_BONE_LENGTHS, max_ratio=1.3,
                )
                cam_3d = cam_rr_fixed + cam_root
                return cam_3d.astype(np.float64)

    # Fallback: depth heuristic when solvePnP fails or produces unreasonable results
    tz = 3.0
    u_root = float(kp_2d[0, 0])
    v_root = float(kp_2d[0, 1])

    # Try person-height heuristic for better depth estimate
    thorax_idx = 8
    ankle_mid_2d = (kp_2d[3] + kp_2d[6]) / 2.0
    pixel_height = abs(kp_2d[thorax_idx, 1] - ankle_mid_2d[1])
    height_3d = float(np.linalg.norm(
        root_relative_m[thorax_idx] - (root_relative_m[3] + root_relative_m[6]) / 2
    ))
    if pixel_height > 20 and height_3d > 0.1:
        tz = fy * height_3d / pixel_height
        tz = float(np.clip(tz, 1.0, 8.0))

    if abs(u_root) > 1.0 or abs(v_root) > 1.0:
        tx = (u_root - cx) * tz / fx
        ty = (v_root - cy) * tz / fy
    else:
        tx = 0.0
        ty = 0.0

    # Enforce bone-length constraints before translation
    root_relative_corrected = _enforce_bone_lengths(
        root_relative_m, PARENTS, DEFAULT_BONE_LENGTHS, max_ratio=1.3,
    )

    cam_3d = root_relative_corrected.copy()
    cam_3d[:, 0] += tx
    cam_3d[:, 1] += ty
    cam_3d[:, 2] += tz
    return cam_3d.astype(np.float64)
```

**Important:** The `import cv2` is already present at line 6 of `detect.py`. Also import `PARENTS` and `DEFAULT_BONE_LENGTHS` from skeleton -- check these are already imported in the existing `motionbert_to_camera_space` (they are imported inside the function body at line 826: `from skeleton import PARENTS, DEFAULT_BONE_LENGTHS`). Move this to the top-level imports for cleanliness, or keep it as-is.

### Step 5: Remove unused helper functions from detect.py

Delete these functions which are no longer called:
- `_enforce_bone_lengths_with_2d()` (lines 582-689)
- `_reconstruct_from_2d()` (lines 692-788)

Keep these functions which are still used:
- `_iqr_filtered_median()` (used in bone-scale estimation)
- `_RELIABLE_BONES_FOR_SCALE` (used in bone-scale estimation)
- `_enforce_bone_lengths()` (used as post-solvePnP safety clamp)

### Step 6: Test on a simple example

Run `process_example()` on example 0 (`171204_pose1_sample`, 100 frames):

```bash
cd motionbert-pose
uv run python -c "
from main import process_example
import config as cfg
import os

run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-solvepnp-test')
os.makedirs(run_dir, exist_ok=True)
seq, cam, start, nf, pidx = cfg.EXAMPLES[0]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```

Check the console output for:
- Root Z range should be in a reasonable range (typically 2-5m for Panoptic)
- MPJPE and MPJVE values
- No errors

### Step 7: Run a second example for comparison

Run example 5 (`171204_pose3_4000`, one of the best-performing in round 4):

```bash
cd motionbert-pose
uv run python -c "
from main import process_example
import config as cfg
import os

run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-solvepnp-test')
seq, cam, start, nf, pidx = cfg.EXAMPLES[5]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```

### Step 8: Compare metrics against the round-4 baseline

The round-4 baseline (from `DEVELOPER_REPORT_RUN_ALL.md`) had:

| Example | Det MPJPE | Opt MPJPE | Det MPJVE | Opt MPJVE |
|---|---|---|---|---|
| 171204_pose1_sample_0 | 30.98 | 30.44 | 0.94 | 1.05 |
| 171204_pose3_4000 | 15.85 | 15.44 | 0.51 | 0.48 |
| **Mean (10 examples)** | **32.99** | **32.65** | **3.13** | **3.11** |

The solvePnP change primarily affects the **detector** (pre-optimization) z-value quality. Better initialization should lead to:
- Similar or better Det MPJPE (better initial depth)
- Better MPJVE (because the z values are more consistent frame-to-frame when estimated via a global rigid fit rather than noisy pairwise ratios)
- Similar or better Opt MPJPE (optimizer starts from a better initial point)

Report: the developer should note in their report the Det MPJPE, Opt MPJPE, Det MPJVE, Opt MPJVE for each tested example, the root Z range, and whether solvePnP succeeded on all frames or fell back to the heuristic.

## Integration Points

- `motionbert_to_camera_space()` is called from `main.py` line 173 in a per-frame loop. Its signature and return type do not change.
- `compute_total_score()` is called from `optimize.py` line 245. The removed parameter `all_joints_smooth_weight` had a default value of 0.0 and was already behaviorally inactive. Removing it is a safe cleanup.
- The `scoring.py` changes only remove dead code (the weight was already 0.0 in config), so no behavioral change from Steps 2-3 alone.

## Risks and Edge Cases

### Risk 1: solvePnP rotation estimate may be wrong
MotionBERT's coordinate frame may not perfectly match camera coordinates. MotionBERT is trained on H3.6M camera-space data, so the coordinates should already be roughly camera-aligned (Y-down, Z-forward). However, solvePnP will estimate a rotation that could include a spurious 180-degree flip if the coordinate axes are mismatched.

**Mitigation:** The safety check `if root_z > 0.5 and root_z < 15.0` catches cases where solvePnP places the skeleton behind the camera or unreasonably far away. If this fires, we fall back to the height heuristic. The developer should print `float(np.linalg.norm(rvec))` (rotation magnitude in radians) to verify it's small (ideally < 0.5 rad). Large rotations suggest a coordinate frame mismatch.

### Risk 2: solvePnP sensitive to outlier 2D detections
If some 2D keypoints are badly detected (ankles in particular), solvePnP may produce a poor fit.

**Mitigation:** Use the visibility-based filtering already in place. Joints below visibility 0.1 are excluded. The SQPNP method is reasonably robust to moderate outliers. If results are poor, a follow-up iteration could further restrict to reliable joints only (arms + torso, exclude ankles).

### Risk 3: Per-frame solvePnP produces jittery z values
Since solvePnP runs independently per frame, the estimated depth may jump between frames.

**Mitigation:** This is exactly what the FK optimization is designed to fix. The root position penalty and rotation penalties will smooth these values during optimization. The key insight from the spec is that we should let the optimizer handle temporal smoothness in camera coordinates.

### Risk 4: The ALL_JOINTS_SMOOTH_WEIGHT removal could increase MPJVE
The weight is already 0.0 in config, so removing it has zero behavioral effect. This is purely code cleanup.

## Success Criteria

1. Pipeline runs without errors on at least 2 examples
2. Z values for root joint are in a reasonable range (1-10m) and roughly match ground truth range
3. MPJVE should be comparable to or better than baseline (the spec says "MPJVE should be improved for ALL test videos")
4. MPJPE should not regress significantly (small regression is acceptable if MPJVE improves)
5. Overlay video looks reasonable -- skeleton placement in the scene should be correct
