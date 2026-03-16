# Research Report: Why MotionBERT MPJPE Is So High and How to Fix It

## Summary of Findings

Our MotionBERT pipeline gets 20-60 cm MPJPE while the paper reports ~40 mm (~4 cm). This 5-15x discrepancy comes from **three compounding problems**: (1) MotionBERT outputs in a normalized coordinate space that is fundamentally different from camera-space meters, and our denormalization pipeline introduces massive scale/depth errors; (2) the paper's reported ~40mm MPJPE uses a **completely different evaluation protocol** (root-relative, on H36M with known camera parameters and a `2.5d_factor` scale correction); and (3) our in-the-wild scenario on CMU Panoptic data is inherently harder than H36M benchmarking.

---

## 1. What Coordinate Space Does MotionBERT Output In?

### The "2.5D Image Space" -- NOT Camera-Space Meters

This is the single most important finding. MotionBERT does NOT output in camera-space meters. It outputs in a **normalized 2.5D image coordinate space** where:

- **X, Y** are normalized to roughly [-1, 1], corresponding to pixel coordinates divided by image width and shifted (see `datareader_h36m.py` lines 61-84)
- **Z** (depth) is normalized by the **same pixel scale factor** as X and Y -- i.e., `z_normalized = z_camera / (res_w / 2)`

Specifically, during training the 3D labels (`joint3d_image`) are prepared as:
```python
# For X,Y: map pixel coords to [-1, 1]
labels[:, :2] = labels[:, :2] / res_w * 2 - [1, res_h / res_w]
# For Z: scale by the SAME factor (no offset)
labels[:, 2:] = labels[:, 2:] / res_w * 2
```

And during evaluation, the inverse is applied:
```python
# Denormalize
data[:, :, :2] = (data[:, :, :2] + [1, res_h / res_w]) * res_w / 2
data[:, :, 2:] = data[:, :, 2:] * res_w / 2
```

After denormalization, **X and Y are in pixels** and **Z is in the same scale as pixels** -- NOT in millimeters or meters. The Z values are effectively `Z_camera_mm * (res_w / focal_length)` because H36M's `joint3d_image` is computed by projecting camera-space 3D joints onto the image plane.

### The Critical `2.5d_factor`

In the evaluation code (`train.py` line 87-121), after denormalization, results are multiplied by a **per-clip `2.5d_factor`**:
```python
factor = factor_clips[idx][:,None,None]
pred *= factor
```

This `2.5d_factor` converts from the 2.5D image space back to millimeters for MPJPE computation. It is stored in the H36M dataset file and is essentially `focal_length / res_w` (or similar), bridging the gap between pixel-scaled coordinates and real-world millimeters.

**We do not have access to this factor for in-the-wild inference.** This is the fundamental scale ambiguity problem.

### The Official `infer_wild.py` Confirms This

The official wild inference code (lines 93-96) shows:
```python
if opts.pixel:
    results_all = results_all * (min(vid_size) / 2.0)
    results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
```

This converts from normalized [-1,1] space to pixel coordinates for visualization only. There is **no conversion to metric space** in the official code -- they simply visualize the relative 3D structure.

The official author confirmed in GitHub Issue #9: "For 3D pose estimation, the network output is in normalized coordinates (roughly [-1,1]) and can be associated with the pixel coordinates."

---

## 2. How the Paper's Evaluation Protocol Differs From Ours

### Paper's Protocol (H36M Benchmark)
1. Input: 2D detections normalized to [-1, 1] using image resolution
2. Output: 3D predictions in normalized 2.5D image space
3. Denormalize using known image resolution (multiply by `res_w/2`, add pixel offset)
4. Multiply by `2.5d_factor` to convert to millimeters
5. **Root-relative**: subtract root joint from both prediction and GT (`pred - pred[:,0:1,:]`)
6. Compute MPJPE in millimeters

Key points:
- The `2.5d_factor` is derived from **known camera intrinsics** of the H36M cameras
- Root-relative comparison eliminates all absolute position/depth errors
- The model never needs to estimate absolute scale or depth

### Our Protocol (CMU Panoptic, In-the-Wild)
1. Input: 2D detections from Stacked Hourglass, normalized via `crop_scale()` to [-1, 1]
2. Output: 3D predictions in normalized space
3. **We attempt to convert to camera-space meters** using bone-length matching and depth heuristics
4. Compare against GT in camera-space meters (root-relative)

Key differences:
- We do NOT have the `2.5d_factor`, so we invent our own scale recovery
- Our `crop_scale()` normalization is **different from H36M's resolution-based normalization** -- it uses the bounding box extent, not the image resolution
- Our scale recovery (bone-length matching, torso-height heuristic) introduces large errors because MotionBERT's bone proportions are often wrong (CV > 0.3 for arm bones across sequences)

### What MPJPE Should We Actually Expect?

If we did root-relative evaluation in the **normalized output space** (no denormalization at all), MotionBERT should give reasonable relative joint positions. The P-MPJPE values in our test report (~26 cm after Procrustes) suggest the joint structure is roughly correct but scale/translation is wrong.

The paper's ~40mm is achievable ONLY with:
- Known camera parameters for proper denormalization
- The `2.5d_factor` for scale conversion
- H36M data that the model was trained on

For truly in-the-wild data, **50-100mm root-relative MPJPE is more realistic** even with perfect denormalization.

---

## 3. Why Our MPJPE Is So High -- Specific Hypotheses

### Hypothesis 1: Scale Recovery Is the Dominant Error Source (HIGH CONFIDENCE)
Our `motionbert_to_camera_space()` tries to recover absolute scale from bone lengths. MotionBERT's normalized bone lengths vary wildly (arm bone scale ratios of 0.31 to 2.26 across sequences, per P2_07 report). Any bone-length-based scale is unreliable.

### Hypothesis 2: crop_scale vs Resolution-Based Normalization Mismatch (HIGH CONFIDENCE)
The H36M training normalizes 2D inputs by `x_norm = x_pixel / res_w * 2 - 1`. Our pipeline uses `crop_scale()` which normalizes by the bounding box extent: `x_norm = (x - xs) / scale * 2 - 1`. These are different transformations. The bounding box changes per video, so the relationship between normalized coords and real-world scale is inconsistent.

The model was fine-tuned with the resolution-based normalization. Using bbox-based normalization means the input distribution is slightly different from training, potentially degrading the 3D predictions.

### Hypothesis 3: Depth (Z) Estimation Is Fundamentally Ambiguous (HIGH CONFIDENCE)
MotionBERT outputs Z in the same normalized space as X and Y. Converting this to meters requires knowing the camera focal length and the `2.5d_factor`. Without this, any Z estimate is a guess. The P2_07 report shows the pairwise depth estimation ranges widely (blended 40/60 with heuristic), confirming this is fragile.

### Hypothesis 4: Stacked Hourglass 2D Quality on Panoptic (MEDIUM CONFIDENCE)
The Stacked Hourglass model was trained on MPII, which has different joint definitions and appearance distribution than CMU Panoptic. Ankle/knee confidence is often near zero. However, MotionBERT was designed to handle noisy 2D input, so this is a secondary issue.

### Hypothesis 5: Input Normalization Clipping (LOW CONFIDENCE)
Our `crop_scale()` clips to [-1, 1] after normalization. If joints fall outside the bounding box (e.g., during fast motion), they get clipped, corrupting the input. The official code also clips, so this is probably minor.

---

## 4. Z-Normalization and Scale Strategies

### What the Paper Does (H36M Benchmark)
- **No explicit Z-normalization** -- Z is implicitly normalized by the same pixel-to-normalized transformation as X and Y
- **`2.5d_factor`** bridges normalized space to millimeters using known camera parameters
- For the "global" model (rootrel=False), only the first frame's root Z is zeroed: `predicted[:,0,0,2] = 0`, leaving subsequent frame depths as offsets from frame 0
- For the "rootrel" model (rootrel=True), all root joints are zeroed: `predicted[:,:,0,:] = 0`

### What Others Do (In-the-Wild Lifting)

**PoseLifter (Chang et al. 2019):** Estimates absolute 3D pose from a single noisy 2D pose, using a lifting network that directly predicts camera-space coordinates. Requires camera focal length as input.

**VideoPose3D (Pavllo et al. 2019):** Uses the same 2.5D normalization. For wild inference, they note that camera parameters must be approximated. Their FAQ suggests using the video resolution to approximate focal length.

**Common approaches for in-the-wild scale recovery:**
1. **Assume focal length**: Use `f = image_width` (a common heuristic for phone cameras with ~50mm equivalent)
2. **Anthropometric priors**: Assume average human height (~1.7m) or torso length (~0.55m) and solve for depth
3. **Procrustes alignment**: If GT is available, align with Procrustes (this is P-MPJPE). This factors out scale entirely
4. **Relative evaluation only**: Report root-relative MPJPE in normalized space, not metric space

### What We Currently Do
Our pipeline uses a complex chain: arm-bone-length matching -> IQR-filtered median scale -> pairwise depth estimation -> torso-height heuristic blending -> bone-length clamping. Each step introduces error that compounds.

---

## 5. Recommended Fixes

### Fix 1: Evaluate Root-Relative in Normalized Space (IMMEDIATE, HIGH IMPACT)
Before trying to convert to meters, evaluate MotionBERT's raw normalized output against GT converted to the same normalized space. This tells us how good the model actually is, independent of our broken denormalization.

Steps:
1. Take GT 3D joints in camera space
2. Project to 2D using camera intrinsics -> pixel coordinates
3. Apply the same normalization as the 2D input (either resolution-based or crop_scale)
4. For Z: `z_norm = z_camera * (2 / res_w)` or equivalent for crop_scale
5. Make both root-relative
6. Compute MPJPE in normalized units

This should give us numbers comparable to the paper (after accounting for dataset differences).

### Fix 2: Use Resolution-Based Normalization Instead of crop_scale (HIGH IMPACT)
Match the H36M training normalization more closely:
```python
# Instead of crop_scale(), use:
scale = min(image_width, image_height) / 2.0
kp_2d[:,:,:2] = (kp_2d[:,:,:2] - [image_width/2, image_height/2]) / scale
```
This matches the `WildDetDataset` path when `vid_size` is provided (dataset_wild.py lines 79-83), which is the `--pixel` mode that keeps relative scale with pixel coordinates.

The crop_scale path (with `scale_range=[1,1]`) is the OTHER option in infer_wild.py, and it normalizes by bounding box -- we should try both and compare.

### Fix 3: Proper Scale Recovery Using Camera Intrinsics (HIGH IMPACT)
Since we have CMU Panoptic camera calibration (fx, fy, cx, cy), we can compute the equivalent `2.5d_factor`:
```python
# The 2.5d_factor converts from pixel-scale Z to real Z
# In H36M: z_pixel_scale = z_camera * fx / res_w (roughly)
# So: z_camera = z_pixel_scale * res_w / fx
# After MotionBERT denorm: z_denorm = z_norm * (res_w / 2)
# Therefore: z_camera = z_denorm * (1 / fx) ... approximately
# More precisely, 2.5d_factor = f / (res_w / 2)
factor = focal_length  # This IS approximately the 2.5d_factor for our camera
```

The key insight: after denormalizing MotionBERT output to pixel-scale (multiply by res_w/2), the Z values are in "pixel depth" which equals `Z_camera * fx / (res_w/2)`. To get camera-space Z in mm: `Z_camera = Z_pixel_depth * (res_w/2) / fx`.

### Fix 4: Two-Stage Approach (MEDIUM IMPACT)
1. Use MotionBERT ONLY for root-relative 3D structure (joint angles/proportions)
2. Use a separate method for absolute positioning:
   - Triangulation if multiple cameras available
   - PnP with known bone lengths
   - Simple pinhole projection with assumed height

### Fix 5: Consider the rootrel Model Instead of Global (LOW EFFORT, WORTH TESTING)
The config shows `rootrel: False` (global model). The rootrel model (`MB_ft_h36m.yaml`) zeros out all root positions and focuses purely on relative joint structure. This avoids the depth estimation problem entirely and may give better root-relative results.

### Fix 6: P-MPJPE as Primary Metric (IMMEDIATE)
Since our scale recovery is unreliable, P-MPJPE (Procrustes-aligned) is a much more meaningful metric for our use case. It factors out scale, translation, and rotation, measuring pure pose quality. Our P-MPJPE of ~26 cm is already much better than our MPJPE of ~33 cm, confirming that scale/depth is the dominant error.

---

## 6. Other Relevant Findings

### The `crop_scale` Input Normalization Is Correct But Suboptimal
Our `crop_scale()` implementation matches the official one in `utils_data.py`. However, for inference with known image size, the resolution-based normalization (from `dataset_wild.py` with `vid_size` set) is more appropriate because it preserves the relationship between pixel scale and depth that the model learned during training.

### Global vs RootRel Model Behavior
- **Global model** (what we use, `rootrel: False`): Zeros only `predicted[:,0,0,2]=0` (first frame root Z). Subsequent frames have Z offsets. The model tries to predict absolute depth structure.
- **RootRel model** (`rootrel: True`): Zeros all root joints `predicted[:,:,0,:]=0`. All output is relative to root. No depth prediction needed.

For in-the-wild inference without camera parameters, the rootrel model might actually be better since it doesn't attempt the impossible task of predicting absolute depth from normalized 2D input.

### Training Data Distribution Mismatch
MotionBERT was fine-tuned on H36M with Stacked Hourglass 2D detections. H36M has:
- Fixed camera positions (4 cameras, known intrinsics)
- Indoor lab setting with consistent lighting
- Fixed image resolution (~1000x1000)
- Subjects at 2-5 meter range

CMU Panoptic has different camera positions, resolutions, and subject distances, contributing to distribution shift.

### The Bone-Length Scale Problem Is Well-Known
MotionBERT (and all lifting networks) produce outputs in a scale-ambiguous space. The P2_07 developer report documents that arm bone ratios vary from 0.31 to 2.26 across sequences -- a 7x range. This makes bone-length-based scale estimation fundamentally unreliable as a single-sequence approach. A better approach would be to use the known camera intrinsics directly.

---

## References

- [MotionBERT Paper (Zhu et al., ICCV 2023)](https://arxiv.org/abs/2210.06551)
- [MotionBERT GitHub - Issue #9: Coordinate type of 3D pose](https://github.com/Walter0807/MotionBERT/issues/9)
- [MotionBERT GitHub - Issue #121: Units of measurement](https://github.com/Walter0807/MotionBERT/issues/121)
- [MotionBERT Official infer_wild.py](https://github.com/Walter0807/MotionBERT/blob/main/infer_wild.py)
- [PoseLifter: Absolute 3D Pose Lifting (Chang et al. 2019)](https://arxiv.org/abs/1910.12029)
- [VideoPose3D (Pavllo et al. 2019)](https://github.com/facebookresearch/VideoPose3D)
