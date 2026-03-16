# Architect Plan: Phase 2, Iteration 1 -- MPJPE Bug Fix

## Problem Statement

The MotionBERT pipeline produces Mean MPJPE = 44.89 cm with P-MPJPE = 21.29 cm. The 2.1x ratio between MPJPE and P-MPJPE indicates a systematic scale/position error in the denormalization from MotionBERT's normalized output to camera-space meters. Bone lengths from `test_single.py` are 2-5x too large (e.g., LKnee=2.014m vs expected 0.420m), confirming a scale blow-up in the conversion to camera-space meters.

## Root Cause Analysis

There are **two compounding bugs** in the pipeline:

### Bug 1: `pixel_aligned_to_camera_space` uses MotionBERT's X,Y output as pixel coordinates, but they are NOT accurate pixel coordinates

**File:** `motionbert-pose/detect.py`, lines 500-510

The function uses `kp_3d[j, 0]` and `kp_3d[j, 1]` (MotionBERT's denormalized X,Y output) as pixel coordinates `u, v` for the pinhole back-projection:

```python
u = float(kp_3d[j, 0])   # line 504
v = float(kp_3d[j, 1])   # line 505
x_cam = (u - cx) * z_cam / fx   # line 508
y_cam = (v - cy) * z_cam / fy   # line 509
```

MotionBERT's X,Y output is a **prediction** of where joints are in pixel space. It does NOT perfectly reproduce the 2D input -- the model introduces errors. These errors are then magnified by the back-projection (`* z_cam / fx`). The function should use the actual 2D detections (`kp_2d`) for X,Y and only use MotionBERT for the Z (depth) component, since MotionBERT's value-add is depth estimation.

However, this alone doesn't explain a 2-5x scale error. The bigger issue is Bug 2.

### Bug 2: Z denormalization scale mismatch

**File:** `motionbert-pose/detect.py`, line 458

The core mathematical issue:

**How the model was trained** (`datareader_h36m.py`, lines 61-72):
- 3D labels are `joint3d_image` -- (pixel_x, pixel_y, depth_in_mm)
- X,Y normalization: `labels[:,:,:2] = labels[:,:,:2] / res_w * 2 - [1, res_h/res_w]`
- Z normalization: `labels[:,:,2:] = labels[:,:,2:] / res_w * 2`
- So: `Z_norm = Z_mm / (res_w / 2)` where `res_w = 1000` for H36M

**How our pipeline normalizes 2D input** (`detect.py`, lines 368-369, `crop_scale`):
- `result[..., :2] = (motion[..., :2] - [xs, ys]) / scale`
- `result[..., :2] = (result[..., :2] - 0.5) * 2`
- Where `scale = max(bbox_width, bbox_height)` in pixels

**How our pipeline denormalizes 3D output** (`detect.py`, line 458):
- `positions_3d *= (scale / 2.0)` -- applied to ALL three axes including Z

The mismatch: The model was trained to output Z values such that `Z_out * (res_w / 2)` gives millimeters (where res_w=1000 for H36M, so multiplier = 500). But our denormalization multiplies Z by `scale / 2`, where `scale` is the bounding box extent in pixels.

For CMU Panoptic HD video (1920x1080), if the person's bounding box spans ~400 pixels, then `scale ≈ 400` and `scale / 2 = 200`. The model expects to be denormalized by 500 (from H36M training). So our Z values are `200/500 = 0.4x` what they should be in mm.

**But this would make Z values TOO SMALL, not too large.** The 5x bone length blow-up must come from Bug 3.

### Bug 3 (Primary): The torso-height heuristic + Z conversion formula create scale errors

**File:** `motionbert-pose/detect.py`, lines 488-510

The `pixel_aligned_to_camera_space` function:

1. Estimates root depth via torso-height heuristic: `root_depth = fx * 1.38 / pixel_height`
2. Converts Z: `z_cam = root_depth + (z_px - root_z_px) * root_depth / fx`
3. Back-projects: `x_cam = (u - cx) * z_cam / fx`, `y_cam = (v - cy) * z_cam / fy`

The problem: **Step 3 back-projects the MotionBERT-predicted pixel coordinates using the estimated depth.** But the predicted pixel coordinates (`u, v = kp_3d[j, 0], kp_3d[j, 1]`) span a range determined by the bounding box and MotionBERT's output scale. If the MotionBERT X,Y predictions overshoot the actual pixel positions (or if the denormalized coordinates are slightly off), the back-projection amplifies these errors.

More concretely: the back-projection formula `x_cam = (u - cx) * z_cam / fx` means that joints far from the principal point (cx, cy) will have large camera-space coordinates. If z_cam is overestimated (say 5m instead of 3m), all camera-space coordinates scale up by 5/3. Combined with small errors in u,v, this can easily produce 2-5x bone length errors.

The fundamental flaw is **trying to use a torso-height heuristic when we have exact camera calibration available.** The CMU Panoptic dataset has precise extrinsic camera parameters that tell us exactly where the camera is and how to convert between world and camera coordinates. We should use the actual camera parameters instead of guessing depth from pixel measurements.

### Why P-MPJPE is "only" 21 cm

P-MPJPE performs Procrustes alignment which removes scale, rotation, and translation. This means the **relative pose structure** from MotionBERT is decent (21 cm error). The remaining 23 cm of MPJPE error comes entirely from incorrect scale and global positioning in the `pixel_aligned_to_camera_space` conversion.

## Recommended Fix: Direct Root-Relative Approach

Instead of the current complex denormalization chain (normalize -> MotionBERT -> denormalize to pixels -> heuristic depth -> camera-space), use a much simpler approach:

### Approach: Scale-Normalized Root-Relative Output

The key insight: **for root-relative MPJPE evaluation, we don't need absolute depth at all.** We just need the root-relative 3D structure at the correct scale.

MotionBERT's output (before denormalization) is already root-relative 3D structure in a normalized space. We just need to scale it correctly to meters.

**Step-by-step:**

1. **Get MotionBERT's normalized output** (already done): `positions_3d` in [-1, 1] normalized space, shape (N, 17, 3)

2. **Make root-relative**: Subtract root (joint 0) from all joints. This is just `pos - pos[:, 0:1, :]`

3. **Scale to meters**: The model's output is normalized. We need to find the correct scale factor to convert to meters. Two options:

   **Option A: Use known camera + 2D detections to estimate scale (RECOMMENDED)**
   - The 2D detections give us pixel positions of joints
   - Camera intrinsics (fx, fy) let us relate pixel distances to real distances at a given depth
   - Estimate root depth Z from 2D torso height: `Z = fy * torso_height_m / torso_height_px`
   - At this depth, the pixel scale factor is `Z / f` meters per pixel
   - MotionBERT's normalized output was scaled by `scale / 2` pixels. So the real-world scale of the normalized output is: `(scale / 2) * (Z / fx)` meters per normalized unit
   - Multiply root-relative positions by this factor: `pos_meters = pos_norm * (scale / 2) * (Z_root / fx)`

   **Option B: Use median bone length matching (SIMPLER, MORE ROBUST)**
   - Compute bone lengths from MotionBERT's normalized output
   - Compute reference bone lengths (known anatomical values)
   - Scale factor = median(reference_bone_lengths / detected_bone_lengths)
   - This avoids depth estimation entirely and relies on the model's output being proportionally correct

4. **Estimate root position in camera space** (needed for absolute MPJPE, not for root-relative):
   - Use 2D root position + camera intrinsics + estimated depth
   - `X_root = (u_root - cx) * Z_root / fx`
   - `Y_root = (v_root - cy) * Z_root / fy`

### Implementation Plan

#### Step 1: Add new function `motionbert_to_camera_space` in `detect.py`

Replace `pixel_aligned_to_camera_space` with a new function that:

```python
def motionbert_to_camera_space(
    positions_3d_norm: np.ndarray,  # (17, 3) normalized MotionBERT output (BEFORE pixel denorm)
    kp_2d: np.ndarray,              # (17, 2) 2D detections in pixel coords
    scale: float,                   # crop_scale scale parameter
    fx: float, fy: float,
    cx: float, cy: float,
) -> np.ndarray:
    """Convert MotionBERT normalized output to camera-space meters.

    Strategy:
    1. Scale root-relative structure using bone-length matching
    2. Place root using 2D detection + depth estimation
    """
    # Root-relative
    root_relative = positions_3d_norm - positions_3d_norm[0:1]

    # Convert from normalized units to approximate pixel units
    root_relative_px = root_relative * (scale / 2.0)

    # Estimate root depth from 2D torso height
    thorax_2d = kp_2d[8]
    ankle_mid_2d = (kp_2d[3] + kp_2d[6]) / 2.0
    pixel_height = abs(thorax_2d[1] - ankle_mid_2d[1])
    assumed_height_m = 1.38
    if pixel_height > 20:
        root_depth = fy * assumed_height_m / pixel_height  # NOTE: use fy not fx (vertical measurement)
    else:
        root_depth = 3.0
    root_depth = np.clip(root_depth, 1.0, 8.0)

    # Convert pixel displacement to meter displacement
    # At depth Z, 1 pixel ≈ Z/f meters
    root_relative_m = root_relative_px.copy()
    root_relative_m[:, 0] *= root_depth / fx  # X
    root_relative_m[:, 1] *= root_depth / fy  # Y
    root_relative_m[:, 2] *= root_depth / fx  # Z (use fx as reference, same as X)

    # Place root in camera space
    u_root = kp_2d[0, 0]  # Use 2D DETECTION for root position, not MotionBERT output
    v_root = kp_2d[0, 1]
    x_root = (u_root - cx) * root_depth / fx
    y_root = (v_root - cy) * root_depth / fy
    z_root = root_depth

    # Assemble
    cam_3d = root_relative_m.copy()
    cam_3d[:, 0] += x_root
    cam_3d[:, 1] += y_root
    cam_3d[:, 2] += z_root

    return cam_3d
```

#### Step 2: Modify `run_motionbert` to return BOTH normalized and pixel-aligned output

Currently `run_motionbert` returns only the denormalized (pixel-aligned) output. We need the pre-denormalization output for the new conversion function.

**Changes to `detect.py`, `run_motionbert` function (line 393):**

Return the raw normalized positions and the crop_scale params in addition to (or instead of) the pixel-aligned output.

New return: `(positions_3d_norm, cs_params)` where `positions_3d_norm` is `(N, 17, 3)` in the model's normalized output space (after first-frame-root Z zeroing, but BEFORE the `*= scale/2` denormalization).

#### Step 3: Update `detect_poses` and callers

Update `detect_poses` to use the new conversion function. The callers (`main.py` lines 160-163, `test_single.py` lines 98-103) currently call `pixel_aligned_to_camera_space` separately -- this should be integrated into the pipeline.

#### Step 4: Also fix the torso-height formula to use `fy` instead of `fx`

In the current `pixel_aligned_to_camera_space` (line 495):
```python
root_depth = fx * assumed_height_m / pixel_height
```

This should use `fy` since the torso height is measured vertically (Y axis). CMU Panoptic cameras may have slightly different fx and fy. Using the wrong focal length introduces an additional error.

### Detailed File Changes

#### `motionbert-pose/detect.py`

1. **`run_motionbert` (line 393):** Change return type to include normalized positions and scale params.
   - After line 451 (print raw output stats), SAVE the normalized positions before denormalization
   - Store `positions_3d_norm = positions_3d.copy()` before line 458
   - Return `(positions_3d_norm, cs_params)` instead of just `positions_3d`
   - Remove or keep the pixel-aligned denormalization (lines 458-460) as it may still be useful for visualization

2. **Add `motionbert_to_camera_space` function** (new, ~40 lines):
   - Takes: `positions_3d_norm` (N, 17, 3), `kp_2d` (17, 2), `scale` (float), `fx, fy, cx, cy`
   - Returns: `(17, 3)` camera-space meters
   - Implementation as outlined above

3. **`detect_poses` (line 519):** Update to use new return values from `run_motionbert` and new conversion function.
   - Receive `(positions_3d_norm, cs_params)` from `run_motionbert`
   - Return `positions_3d_norm` and `cs_params` so callers can do the camera-space conversion

4. **Remove or deprecate `pixel_aligned_to_camera_space`**: Once the new function is working, the old one should be removed.

#### `motionbert-pose/main.py`

1. **Lines 155-163:** Update `detect_poses` call to receive new return values, call new conversion function with camera intrinsics per frame.

#### `motionbert-pose/test_single.py`

1. **Lines 90-103:** Same updates as main.py.

### Validation Plan

After implementing the fix:

1. **Run `test_single.py` on example 0:**
   - Bone lengths should be in the 0.1-0.5m range (not 2-5m)
   - Root depth should be 2-5m (reasonable for Panoptic lab setting)
   - MPJPE should drop significantly (expect ~20-30 cm based on P-MPJPE of 21 cm)
   - P-MPJPE should remain ~21 cm (unchanged, since it's scale-invariant)

2. **Run `main.py` on all 10 examples:**
   - Mean MPJPE should be in the 15-30 cm range
   - The MPJPE/P-MPJPE ratio should be much closer to 1.0 (currently 2.1x)

### Expected Impact on Numbers

| Metric | Current | Expected After Fix |
|--------|---------|-------------------|
| Mean MPJPE | 44.89 cm | ~20-30 cm |
| Mean P-MPJPE | 21.29 cm | ~20-22 cm (unchanged) |
| MPJPE/P-MPJPE ratio | 2.1x | ~1.0-1.3x |
| Bone lengths | 2-5x too large | Within 20% of anatomical values |

The P-MPJPE will remain roughly the same because it already removes scale errors via Procrustes alignment. The MPJPE should drop to be much closer to P-MPJPE, since the scale error is being fixed.

### Alternative Considered: Using Extrinsic Camera Parameters

Since CMU Panoptic provides full extrinsic calibration (R, t), we could in theory triangulate the 3D position from the 2D detections + known camera pose. However, we only have a single camera view, so triangulation isn't possible. The extrinsics are used for ground truth conversion (world_to_camera), not for the detector output.

The torso-height heuristic for depth estimation is the best we can do with a single camera and no additional information. The key fix is ensuring the **scale of the root-relative structure** is correct, which is Bug 2/3. Once the structure is correctly scaled, even if root depth is slightly off, the root-relative MPJPE will be close to P-MPJPE.

### Summary of Changes

| Priority | File | Change | Lines |
|----------|------|--------|-------|
| **P0** | `detect.py` | Save normalized output before denorm in `run_motionbert` | ~444-458 |
| **P0** | `detect.py` | New function `motionbert_to_camera_space` | New (~40 lines) |
| **P0** | `detect.py` | Update `detect_poses` return signature and conversion | ~519-569 |
| **P1** | `main.py` | Update to use new detect_poses return values | ~155-163 |
| **P1** | `test_single.py` | Update to use new detect_poses return values | ~90-103 |
| **P2** | `detect.py` | Fix fy vs fx in torso-height formula | Line 495 |
| **P2** | `detect.py` | Remove old `pixel_aligned_to_camera_space` | Lines 469-512 |
