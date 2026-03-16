# Architect Plan: Phase 2, Iteration 3 -- Procrustes-Based Scale/Rotation Correction

## Current State

- Detector baseline: 29.09 cm MPJPE, 21.98 cm P-MPJPE
- FK optimization: 28.21 cm MPJPE (+0.88 cm improvement, marginal)
- The 7.11 cm gap between MPJPE and P-MPJPE is the biggest opportunity
- More FK optimization steps actively hurt because the optimizer overfits to noisy 2D targets
- The 2D reprojection error at initialization is 87-334 pixels -- the projected 3D skeleton is far from the 2D detections

## Root Cause Analysis

After reading all the code, I have identified the core problem and a high-impact fix.

### The Core Problem: MPJPE vs P-MPJPE Gap = Wrong Scale/Rotation/Translation

P-MPJPE (21.98 cm) performs Procrustes alignment -- optimal scale, rotation, and translation -- before measuring error. MPJPE (29.09 cm) does not. The 7.11 cm gap means **the root-relative 3D shape is decent, but it's at the wrong scale and/or rotation**.

This happens because:

1. **`motionbert_to_camera_space` uses bone-length matching for scale** (`detect.py` lines 548-573). The bone-length-based scale factor maps MotionBERT's normalized output to meters using `DEFAULT_BONE_LENGTHS`. But MotionBERT was trained on H3.6M, and the subjects in CMU Panoptic may have different proportions. Also, the `DEFAULT_BONE_LENGTHS` are approximations.

2. **The root-relative structure is placed naively in camera space**: root X,Y come from back-projecting the 2D hip detection at the estimated depth, and relative Z offsets use MotionBERT's predicted depth. Any error in root depth estimation propagates to all joints via the back-projection.

3. **MotionBERT outputs in a coordinate frame that may not align with camera coordinates**: The model was trained on H3.6M data where the coordinate convention may differ from CMU Panoptic's camera convention. A global rotation mismatch would cause systematic MPJPE error that Procrustes removes.

### Why FK Optimization Can't Fix This

The FK optimizer uses 2D reprojection loss -- it projects the 3D skeleton to 2D and compares against Stacked Hourglass detections. But:
- 2D reprojection is invariant to depth scale (a skeleton at 2x depth and 2x size projects identically)
- The 2D targets from Stacked Hourglass don't match MotionBERT's predicted 2D positions (87-334 px gap), so optimizing toward them moves joints AWAY from their correct 3D positions
- More steps = more overfitting to wrong 2D targets = worse 3D accuracy

### The Big Insight: We Can Compute the Optimal Procrustes Transform

We know P-MPJPE is 22 cm. That means if we applied the per-frame Procrustes alignment that the P-MPJPE metric uses, our MPJPE would be 22 cm. We obviously can't use ground truth for this at test time, but we can use a **self-Procrustes** approach:

**Instead of using ground truth to find the optimal alignment, we can use the 2D detections + camera intrinsics to solve for the best rigid transform (scale + rotation + translation) that places the root-relative 3D skeleton in camera space.**

This is exactly what `cv2.solvePnP` does -- it's a Perspective-n-Point solver. Given:
- 3D points in object frame (MotionBERT's root-relative output, scaled to meters)
- 2D projections of those points (Stacked Hourglass 2D detections)
- Camera intrinsics (known exactly from CMU Panoptic calibration)

It finds the rotation R and translation t such that `projection(R @ pts_3d + t) = pts_2d`.

**The mediapipe-pose reference already does exactly this** (`mediapipe-pose/detect.py` lines 98-146). It uses `cv2.solvePnP(..., flags=cv2.SOLVEPNP_SQPNP)` to find the rigid transform from MediaPipe's hip-relative 3D to camera space.

## Proposed Fix: Replace Heuristic Depth Estimation with solvePnP

### Why This Will Work

1. **solvePnP finds the globally optimal placement** of the 3D skeleton in camera space, given the 2D observations. It solves for rotation + translation simultaneously, handling both the depth estimation and any coordinate frame misalignment.

2. **It uses ALL visible joints** for the alignment, not just the torso height. This is far more robust than the single torso-height measurement.

3. **It naturally handles coordinate convention differences** between MotionBERT's output frame and the camera frame. Any systematic rotation gets absorbed by the R matrix.

4. **The mediapipe-pose reference proves it works**: MediaPipe uses the same approach and achieves 18.89 cm MPJPE. The MediaPipe 3D output is also hip-relative and needs to be placed in camera space -- solvePnP is how it does it.

### Expected Impact

The 7.11 cm gap between MPJPE and P-MPJPE represents scale/rotation/translation error. solvePnP should close most of this gap. Realistic target: **23-25 cm MPJPE** (not quite reaching P-MPJPE because solvePnP uses noisy 2D detections while Procrustes uses exact 3D ground truth).

After fixing the placement, the FK optimization may also become more effective, since the projected 3D joints will be much closer to the 2D targets (no more 87-334 px gap), reducing the overfitting problem.

## Implementation Plan

### Change 1 (P0): Add solvePnP-based 3D placement in `detect.py`

Replace `motionbert_to_camera_space` with a new version that uses solvePnP.

**File: `motionbert-pose/detect.py`**

Replace the `motionbert_to_camera_space` function (lines 523-599) with:

```python
def motionbert_to_camera_space(
    positions_3d_norm: np.ndarray,
    kp_2d: np.ndarray,
    scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Convert MotionBERT normalized output to camera-space meters using solvePnP.

    Uses a two-step approach:
    1. Scale the root-relative 3D structure using bone length matching.
    2. Use solvePnP to find the optimal rigid transform (rotation + translation)
       that aligns the 3D skeleton to the 2D detections via the camera model.

    This replaces the heuristic depth estimation with a principled approach that
    jointly optimizes rotation and translation using all visible joints.

    Args:
        positions_3d_norm: (17, 3) normalized MotionBERT output (before pixel denorm).
        kp_2d: (17, 2) 2D detections in pixel coordinates.
        scale: crop_scale scale parameter.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (17, 3) camera-space meters.
    """
    import cv2
    from skeleton import PARENTS, DEFAULT_BONE_LENGTHS

    # Step 1: Get root-relative structure in meters via bone-length matching
    root_relative: np.ndarray = positions_3d_norm - positions_3d_norm[0:1]

    # Compute scale factor via median bone length ratio
    detected_bone_lengths: list[float] = []
    reference_bone_lengths: list[float] = []
    for j in range(1, 17):
        p: int = int(PARENTS[j])
        det_bl: float = float(np.linalg.norm(root_relative[j] - root_relative[p]))
        ref_bl: float = float(DEFAULT_BONE_LENGTHS[j])
        if det_bl > 1e-4 and ref_bl > 1e-4:
            detected_bone_lengths.append(det_bl)
            reference_bone_lengths.append(ref_bl)

    if detected_bone_lengths:
        ratios: np.ndarray = np.array(reference_bone_lengths) / np.array(detected_bone_lengths)
        bone_scale: float = float(np.median(ratios))
    else:
        bone_scale = 1.0

    root_relative_m: np.ndarray = root_relative * bone_scale

    # Step 2: Use solvePnP to find optimal placement in camera space
    K: np.ndarray = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    dist_coeffs: np.ndarray = np.zeros(4, dtype=np.float64)

    # Use joints with nonzero 2D detections
    valid: np.ndarray = np.linalg.norm(kp_2d, axis=1) > 1.0
    n_valid: int = int(valid.sum())

    if n_valid >= 4:
        obj_pts: np.ndarray = root_relative_m[valid].astype(np.float64)
        img_pts: np.ndarray = kp_2d[valid].astype(np.float64)

        success: bool
        rvec: np.ndarray
        tvec: np.ndarray
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP,
        )

        if success:
            R_pnp: np.ndarray
            R_pnp, _ = cv2.Rodrigues(rvec)
            cam_3d: np.ndarray = (R_pnp @ root_relative_m.T).T + tvec.T
            return cam_3d.astype(np.float64)

    # Fallback: original heuristic approach if solvePnP fails
    thorax_2d: np.ndarray = kp_2d[8]
    ankle_mid_2d: np.ndarray = (kp_2d[3] + kp_2d[6]) / 2.0
    pixel_height: float = abs(float(thorax_2d[1] - ankle_mid_2d[1]))
    assumed_height_m: float = 1.38
    if pixel_height > 20:
        root_depth: float = fy * assumed_height_m / pixel_height
    else:
        root_depth = 3.0
    root_depth = float(np.clip(root_depth, 1.0, 8.0))

    u_root: float = float(kp_2d[0, 0])
    v_root: float = float(kp_2d[0, 1])
    x_root: float = (u_root - cx) * root_depth / fx
    y_root: float = (v_root - cy) * root_depth / fy
    z_root: float = root_depth

    cam_3d = root_relative_m.copy()
    cam_3d[:, 0] += x_root
    cam_3d[:, 1] += y_root
    cam_3d[:, 2] += z_root

    return cam_3d
```

### Change 2 (P1): Use Panoptic distortion coefficients for solvePnP

The CMU Panoptic calibration includes distortion coefficients (`distCoef`). Using them in solvePnP will give a more accurate solution. This requires passing the distortion coefficients through the pipeline.

**File: `motionbert-pose/detect.py`** - Add an optional `dist_coeffs` parameter to `motionbert_to_camera_space`.

**File: `motionbert-pose/main.py`** - Pass `cam_calib["distCoef"]` when calling `motionbert_to_camera_space`.

This is a minor improvement but easy to do. The current code uses `np.zeros(4)` for distortion.

### Change 3 (P1): After solvePnP fix, re-evaluate whether FK optimization helps

With solvePnP placing the skeleton correctly, the projected 3D joints should be much closer to the 2D targets. This may make FK optimization effective. However, the current 20-step optimization with constant sigma=80 was tuned for the old (wrong) initialization.

Recommendation: **Keep the existing optimization settings for now** (20 steps, sigma=80). If solvePnP closes the 2D gap significantly, the optimizer should naturally work better without parameter changes. If needed, a subsequent iteration can tune the optimization parameters for the new initialization quality.

### Change 4 (P2, optional): Add RANSAC-based solvePnP for robustness

If some 2D detections are outliers, they'll corrupt the solvePnP solution. Using `cv2.solvePnPRansac` instead of `cv2.solvePnP` would be more robust. However, with 17 joints and typically good Stacked Hourglass detections, this is unlikely to be necessary. Only add if testing reveals problems on specific examples.

## Summary of Changes

| Priority | File | Change | Impact |
|----------|------|--------|--------|
| **P0** | `detect.py` | Replace heuristic depth with solvePnP in `motionbert_to_camera_space` | Close 7cm MPJPE-P-MPJPE gap |
| **P1** | `detect.py`, `main.py` | Pass distortion coefficients to solvePnP | Minor accuracy improvement |
| **P1** | (none) | Re-evaluate FK optimization with better initialization | Potential further gains |
| **P2** | `detect.py` | RANSAC variant of solvePnP | Robustness to outlier 2D detections |

## Expected Results

| Metric | Current | Expected After Fix |
|--------|---------|-------------------|
| Mean MPJPE (detector) | 29.09 cm | ~23-25 cm |
| Mean P-MPJPE (detector) | 21.98 cm | ~21-22 cm (unchanged) |
| MPJPE/P-MPJPE ratio | 1.32x | ~1.05-1.15x |
| Mean MPJPE (optimized) | 28.21 cm | ~22-24 cm |
| Mean improvement from FK | +0.88 cm | ~1-3 cm (may improve) |

## Validation Plan

1. Run `test_single.py` on example 0
   - Check that solvePnP succeeds (not falling back to heuristic)
   - Root depth should be similar to GT root Z
   - 2D reprojection error should be MUCH smaller (expect <20px vs current 87-334px)
   - MPJPE should drop significantly
   - P-MPJPE should remain ~21-22 cm

2. Run `main.py` on all 10 examples
   - Mean MPJPE should be ~23-25 cm
   - No example should regress in P-MPJPE
   - Check that solvePnP succeeds on all examples (>= 4 valid joints)

3. If solvePnP reduces MPJPE by >3 cm, evaluate whether FK optimization still helps
   - If yes, the improvement from FK may be larger now
   - If no, consider disabling FK (returning detector output as final)

## What NOT to Change

- **FK code (`fk.py`)**: Roundtrip error is 0.0000 cm. The FK is correct.
- **Scoring function (`scoring.py`)**: Working correctly.
- **Evaluation code (`evaluate.py`)**: Working correctly.
- **Skeleton definition (`skeleton.py`)**: Correct.
- **Optimization hyperparameters** (`config.py`): Keep current settings for now. Re-tune in a subsequent iteration if needed.
- **Detection pipeline** (Stacked Hourglass, MotionBERT, YOLO): Don't change upstream detection.
