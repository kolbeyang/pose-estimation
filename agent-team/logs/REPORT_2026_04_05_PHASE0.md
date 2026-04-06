# Phase 0 Diagnostic Report

**Date:** 2026-04-05
**Branch:** refactor-2026-04-04
**Run analyzed:** run_2026_04_05_18_46 (single example: 171204_pose1_sample, 100 frames @ 10fps = 34 GT frames, sigma=16, lr=0.0005, steps=10)

---

## 0A: VW-SI-MPJPE Calculation Audit

### Observation
VW-SI-MPJPE is the primary evaluation metric. Need to verify correctness and understand behavior.

### Verification Method
- Code review of `evaluate.py`
- Ran `experiment/audit_vw_si_mpjpe.py` on both MotionBERT and MediaPipe trajectories
- Manual recomputation to verify library output

### Verification Results

**How `compute_visibility_weights()` works:**
- Uses **GT** 3D positions projected to 2D via the camera
- Binary weights: 1 if GT joint projects inside image bounds [0, width) x [0, height), else 0
- This means visibility is determined by whether the *ground truth* joint is visible, not the predicted joint
- This is **correct behavior** -- we want to evaluate only where we have meaningful GT data

**How VW-SI-MPJPE is calculated:**
- Finds a **single global scale factor** `s` across ALL frames that minimizes `sum(weights * ||s*pred - gt||) / sum(weights)`
- Uses `scipy.optimize.minimize_scalar` with bounds **[0.5, 2.0]**
- Applies `s*pred` uniformly to all coordinates (x, y, z) of all joints in camera space
- The weighted sum/weighted-count formulation correctly handles per-joint per-frame weights

**Key finding -- scale operates in camera space, not root-relative:**
The scale factor is applied to full camera-space coordinates. This means it simultaneously adjusts:
1. Skeleton size (bone lengths)
2. Position/depth (distance from camera)

For this example:
- MotionBERT: scale = 1.112 (pred too small/close by ~11%)
- MediaPipe: scale = 1.155 (pred too small/close by ~15%)

Neither hits the bounds [0.5, 2.0], so no saturation issue.

**Manual recomputation matched exactly** for both pipelines (difference < 1e-8).

**Notable behavior -- ankles get zero weight:**
In this example, both ankles (RAnkle, LAnkle) have 0% visibility because GT projects below the image bottom edge (y > 1080). This means ankle errors are **completely excluded** from VW-SI-MPJPE. The MotionBERT ankle errors of 232cm and 238cm are invisible to this metric.

### Potential Issues Found

1. **Scale bound [0.5, 2.0] could be too narrow** for pathological cases. Not an issue in current data.

2. **Camera-space scale is suboptimal.** Comparing `vw_si_mpjpe` (camera-space SI) vs root-relative SI:
   - MotionBERT: 23.43 cm camera-space vs 17.20 cm root-relative (26% better)
   - MediaPipe: 15.66 cm camera-space vs 11.06 cm root-relative (29% better)
   
   The camera-space scale must simultaneously fix depth offset AND skeleton scale, which is a compromise. Root-relative evaluation separates these concerns. However, camera-space is more honest -- it penalizes depth errors.

3. **No bugs found** in the implementation. Code is clean and correct.

### Recommended Solution
- No code changes needed for correctness
- Consider adding root-relative SI-MPJPE as a supplementary metric for analysis
- The ankle visibility exclusion is by design, but worth noting when interpreting results

---

## 0B: MotionBERT Knee Issue

### Observation
MotionBERT per-joint VW-SI-MPJPE shows massive lower body errors:
- RKnee: 77.73 cm, LKnee: 79.89 cm (after scale)
- RAnkle: 231.74 cm, LAnkle: 237.84 cm (invisible to metric due to 0% visibility)
- Detected RKnee bone length: 0.99m vs GT: 0.44m (2.25x too long)

### Hypothesis
The issue originates in MotionBERT's raw 3D predictions, not in the scaling code. MotionBERT's lower body predictions are geometrically distorted.

### Verification Method
Ran `experiment/audit_motionbert_knee.py` and `experiment/audit_fk_roundtrip.py` on MotionBERT trajectories.

### Verification Results

**Bone length analysis reveals systematic lower body distortion:**

| Joint | GT | Pred Mean | Pred Std | Ratio |
|-------|------|-----------|----------|-------|
| RHip | 10.2cm | 9.9cm | 0.6cm | 0.97x |
| RKnee | 44.1cm | 58.4cm | **28.8cm** | 1.32x |
| RAnkle | 39.7cm | **107.8cm** | **53.7cm** | **2.72x** |
| LKnee | 44.5cm | 60.9cm | **31.5cm** | 1.37x |
| LAnkle | 38.9cm | **113.7cm** | **53.8cm** | **2.92x** |
| LShoulder | 16.2cm | 17.7cm | 1.3cm | 1.09x |
| LElbow | 29.0cm | 27.4cm | 1.5cm | 0.94x |
| RElbow | 30.1cm | 27.1cm | 0.6cm | 0.90x |

**Key findings:**
1. **Upper body is accurate** (ratios 0.90-1.09x, low std)
2. **Lower body is wildly wrong** (ratios 1.32-2.92x, high std)
3. The high std on knee/ankle bones (28-54cm!) shows **frame-to-frame instability**
4. In some frames (e.g., frame 0), knee direction is **inverted** (cos_sim = -0.84), meaning MotionBERT thinks the knee is pointing up when it should point down

**The `_RELIABLE_BONES_FOR_SCALE` mechanism works correctly:**
- It only uses joints {10, 11, 12, 13, 14, 15} (shoulders, elbows, wrists) for scale
- These give median ratios very close to 1.0 (0.998-1.002)
- If legs were included, scale would be distorted

**Root cause:** MotionBERT's 2D-to-3D lifting produces distorted lower body poses. The Spine bone is also wrong (1.72x), and Neck is too short (0.34x). This is likely because:
- The person's legs are partially out of frame (ankles project to y > 1080)
- MotionBERT was trained on H36M which has full-body visibility
- The 2D input from SH already has poor leg keypoints (467px reprojection error for knees)

**2D reprojection errors (raw MotionBERT):**
- Upper body: 27-84 px
- RKnee: **467 px**, LKnee: **472 px**
- This confirms the 3D lift is bad, not just the scaling

### Recommended Solution
1. **Do not trust MotionBERT lower body when legs are partially out of frame.** Consider per-joint confidence based on 2D keypoint quality.
2. The optimizer reduces knee errors by ~30% (77cm -> 52cm) through heatmap alignment, but cannot fix the fundamental 3D lift error.
3. A potential fix: use MediaPipe's depth for lower body when MotionBERT's is unreliable, or downweight lower body joints in the optimization loss.

---

## 0C: MediaPipe Regression

### Observation
MediaPipe goes from 15.66 cm raw to 15.86 cm optimized VW-SI-MPJPE -- a 1.2% regression.

### Hypothesis
The optimizer is moving wrists/ankles to match heatmaps, but the heatmaps are at different locations than the GT, causing 3D error to increase for those joints.

### Verification Method
Ran `experiment/audit_fk_roundtrip.py` on MediaPipe trajectories.

### Verification Results

**FK roundtrip error is negligible** (max 0.0005 cm). The FK parameterization is NOT the source of error.

**Per-joint analysis shows exactly which joints regress:**

| Joint | Raw VW-SI | Opt VW-SI | Delta | 2D Raw | 2D Opt |
|-------|-----------|-----------|-------|--------|--------|
| RAnkle | 15.49cm | 16.80cm | **+1.31cm** | N/A | N/A |
| LAnkle | 21.30cm | 22.13cm | **+0.83cm** | N/A | N/A |
| LWrist | 15.93cm | 17.73cm | **+1.80cm** | 28.9px | **50.6px** |
| RWrist | 13.66cm | 15.30cm | **+1.64cm** | 25.4px | **44.9px** |

**Key findings:**
1. The **wrists get significantly worse** (+11-12% error), and the 2D reprojection error nearly doubles (25-29px -> 45-51px)
2. The **ankles get moderately worse** (+4-8%)
3. All other joints either improve or stay flat (Pelvis, Hips, Knees, Neck, Shoulders, Elbows all improve slightly by 0.1-2.3%)
4. The net effect: small improvements on 11 joints are outweighed by large regressions on 4 joints

**Root cause analysis:**
- The optimizer maximizes heatmap log-likelihood. The SH heatmaps for wrists may have peaks at slightly wrong 2D locations compared to GT
- When the optimizer moves the wrist 3D position to match the heatmap peak, the 3D error vs GT increases
- The wrists have **low rotation penalty multipliers** (0.1) making them highly mobile during optimization
- MediaPipe's raw 3D is already quite good (P-MPJPE 7.35cm), so the optimization has very little headroom and is more likely to regress

**Bone length analysis for MediaPipe:**
- Knee bones are 0.73-0.80x of GT (too short)
- Ankle bones are 0.75-0.77x of GT (too short)
- Upper body is 0.80-1.05x (reasonable)
- MediaPipe systematically underestimates limb lengths, but consistently (low std)

### Recommended Solution
1. **Increase rotation penalty for wrists and ankles** to prevent over-fitting to heatmap noise
2. **Reduce optimization steps or learning rate** for MediaPipe -- it needs gentler optimization than MotionBERT
3. Consider **early stopping** based on a validation metric
4. Investigate whether the SH heatmap peaks actually align with MediaPipe's 2D projections -- the discrepancy suggests MediaPipe and SH disagree on wrist/ankle locations

---

## 0D: Heatmap Blur Sigma Discrepancy

### Observation
- Default in `config.py`: `heatmap_blur_sigma: float = 4.0`
- Configs in use: most JSON configs set `heatmap_blur_sigma: 16.0`
- `motionbert-single.json` was recently modified from 16.0 to 32.0 (uncommitted change)
- The run we analyzed (run_2026_04_05_18_46) used sigma=16.0

### Verification Method
- Code review of `scoring.py::apply_blur()` and `optimize/__init__.py`
- Compared config values across all JSON configs

### Verification Results

**What blur does:**
- `apply_blur()` applies a Gaussian blur to the SH heatmaps (64x64 resolution) with the specified sigma
- Heatmaps are per-channel (one per MPII joint), shape (F, 16, 64, 64)
- The blur uses `scipy.ndimage.gaussian_filter` -- this is in heatmap pixel space
- At sigma=16 on a 64x64 heatmap, this creates an **extremely wide** blur (sigma = 25% of image width)
- At sigma=4 (default), the blur is moderate (6.25% of image width)
- At sigma=32, the heatmap is essentially a uniform field -- **all spatial information is destroyed**

**Config values across files:**

| Config File | heatmap_blur_sigma |
|-------------|-------------------|
| config.py (default) | 4.0 |
| both-local-single.json | 16.0 |
| motionbert-local-25-examples.json | 16.0 |
| server-one-each.json | 16.0 |
| server-full-single.json | 16.0 |
| motionbert-single.json (working copy) | **32.0** |

**Impact analysis:**
- sigma=4: Moderate smoothing, heatmap peaks remain localized
- sigma=16: Heavy smoothing, peaks spread across ~50% of the heatmap. This turns the heatmap from a precise localization signal into a very broad "is the joint roughly in this region" signal
- sigma=32: Nearly uniform heatmap. The optimizer gets almost no spatial gradient, so it barely moves

**Why high sigma might have been used:**
The blur makes the loss landscape smoother, reducing the chance of local minima. With sigma=4, if the initial projection is far from the heatmap peak, gradients may be zero (projected point lands in a flat zero region). With sigma=16, gradients propagate further. But at sigma=16, the heatmap signal is so diffuse that the optimizer can only roughly localize joints, not precisely.

### Recommended Solution
1. **The uncommitted change to sigma=32 in motionbert-single.json should be reverted** -- it destroys spatial information
2. Consider a **blur schedule**: start with high sigma (e.g., 16) for first few steps, then anneal to low sigma (e.g., 2-4) for final steps. This gets the benefit of smooth gradients early and precise localization late.
3. The default of 4.0 in config.py is reasonable for well-initialized poses. The override to 16.0 in run configs compensates for poor MotionBERT initialization where projections land far from heatmap peaks.
4. MediaPipe and MotionBERT may need **different blur settings** -- MediaPipe starts much closer to correct, so it needs less blur.

---

## Summary of Key Findings

| Finding | Severity | Impact |
|---------|----------|--------|
| VW-SI-MPJPE is correct but excludes ankles (0% visibility) | Low | Metric underreports lower body issues |
| MotionBERT knee/ankle bones are 1.3-2.9x too long | **Critical** | Dominates MotionBERT error (51% of weighted error from knees alone) |
| MotionBERT lower body has inverted directions in some frames | **Critical** | FK optimizer cannot fix structural 3D errors |
| MediaPipe wrists regress +11-12% after optimization | High | Net regression on MediaPipe |
| MediaPipe ankles regress +4-8% after optimization | Medium | Contributes to net regression |
| Heatmap blur sigma=16 is very aggressive for 64x64 heatmaps | Medium | Limits precision of optimization |
| Uncommitted sigma=32 change destroys heatmap signal | High | Would make optimization nearly useless |
| Camera-space scale is a compromise (depth + skeleton) | Low | ~26-29% worse than root-relative SI |

## Experiment Scripts Created
- `experiment/audit_fk_roundtrip.py` -- FK roundtrip error and raw-vs-opt per-joint comparison
- `experiment/audit_motionbert_knee.py` -- MotionBERT bone length distortion analysis
