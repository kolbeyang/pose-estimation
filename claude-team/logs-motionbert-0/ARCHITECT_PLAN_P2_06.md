# Architect Plan: Phase 2, Iteration 6 -- Detection Pipeline Bugs

## Current State

| Metric | Value |
|--------|-------|
| Detector MPJPE | 29.09 cm |
| Detector P-MPJPE | 21.98 cm |
| Optimized MPJPE | 28.20 cm |
| Optimization ceiling | +0.89 cm (29.09 -> 28.20) |

After 5 iterations of FK optimization tuning with negligible improvement, the focus shifts to the **detection pipeline itself**. A character-by-character comparison of our `detect.py` against the official MotionBERT code revealed two concrete bugs and one high-impact configuration issue.

## Bugs Found

### BUG 1 (HIGH IMPACT): Missing Stacked Hourglass Color Normalization

**Location:** `detect.py`, lines 251-253 in `run_hourglass()`

**What we do:**
```python
img = cropped.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
inp = torch.from_numpy(img).unsqueeze(0).to(device)
```

**What the official HumanPosePredictor does:**
```python
image /= 255.0
image = color_normalize(image, rgb_mean, rgb_stddev)
# where rgb_mean = [0.4404, 0.4440, 0.4327]
# and color_normalize just subtracts mean (NO division by std)
```

**Impact:** Every frame fed to Stacked Hourglass has all three channels shifted by ~+0.44 relative to what the model was trained on. This is a significant distribution shift that degrades heatmap quality and 2D keypoint accuracy. Since MotionBERT's 3D predictions are downstream of these 2D keypoints, this bug corrupts the ENTIRE pipeline.

**Note:** The `color_normalize` function in the stacked_hourglass package ONLY subtracts the mean. It does NOT divide by stddev despite the parameter being named `rgb_stddev`. The stddev parameter is unused. So the fix is just mean subtraction.

**Fix:** After `/255.0`, subtract the channel means:
```python
img = cropped.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
# Apply same normalization as the official HumanPosePredictor
img[0] -= 0.4404  # R
img[1] -= 0.4440  # G
img[2] -= 0.4327  # B
```

Apply the same fix to the flipped image path as well (lines 261-263).

### BUG 2 (LOW IMPACT): Wrong LayerNorm epsilon in DSTformer

**Location:** `detect.py`, lines 317-322 in `load_motionbert_model()`

**What we do:**
```python
model = DSTformer(
    dim_in=3, dim_out=3,
    dim_feat=256, dim_rep=512,
    depth=5, num_heads=8, mlp_ratio=4,
    num_joints=17, maxlen=243,
)
```

**What the official `load_backbone()` does:**
```python
model_backbone = DSTformer(
    dim_in=3, dim_out=3,
    dim_feat=args.dim_feat, dim_rep=args.dim_rep,
    depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    maxlen=args.maxlen, num_joints=args.num_joints,
)
```

The official code passes `norm_layer=partial(nn.LayerNorm, eps=1e-6)`. Our code uses the default `norm_layer=nn.LayerNorm` which has `eps=1e-5`. The LayerNorm weight and bias parameters load correctly from the checkpoint, but the epsilon hyperparameter is wrong.

**Impact:** Likely negligible. The difference between eps=1e-5 and eps=1e-6 almost never affects outputs. But fix it anyway for correctness.

**Fix:** Add the norm_layer parameter:
```python
from functools import partial
import torch.nn as nn

model = DSTformer(
    dim_in=3, dim_out=3,
    dim_feat=256, dim_rep=512,
    depth=5, num_heads=8, mlp_ratio=4,
    num_joints=17, maxlen=243,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)
```

### CONFIG ISSUE (HIGH IMPACT): Too few input frames for MotionBERT

**Location:** `config.py`, line 30

**Current:** `TARGET_FPS = 10.0` -- subsamples 30fps video to 10fps, yielding:
- 100 raw frames -> ~34 frames fed to MotionBERT
- 150 raw frames -> ~50 frames fed to MotionBERT

**Problem:** MotionBERT was trained on 243-frame clips from H36M at 50fps (~4.86 seconds). Our clips are:
- 34 frames at 10fps = 3.4 seconds (decent duration, but very sparse temporal sampling)
- 50 frames at 10fps = 5.0 seconds (good duration, but sparse)

At 30fps (native, no subsampling):
- 100 frames at 30fps = 3.3 seconds
- 150 frames at 30fps = 5.0 seconds

The key advantage of 30fps is MORE FRAMES (100-150 vs 34-50), giving MotionBERT much more temporal context. The model uses self-attention over the temporal dimension, so more frames = better predictions. It handles variable-length input natively via `temp_embed[:,:F,:,:]`.

**Fix:** Change `TARGET_FPS` from 10.0 to 30.0 (or use the native video FPS directly).

**Trade-off:** 3x more frames means 3x slower Stacked Hourglass inference (the bottleneck). But MotionBERT itself handles the full sequence in one forward pass regardless of frame count (up to 243).

## Implementation Plan

### Priority Order

1. **BUG 1 fix (color normalization)** -- This is the single highest-impact change. Every 2D keypoint we've been computing is degraded by the missing normalization.

2. **CONFIG change (TARGET_FPS = 30.0)** -- More temporal context for MotionBERT. Test this SEPARATELY from bug 1 to measure each improvement independently.

3. **BUG 2 fix (LayerNorm eps)** -- Low impact but trivial to fix. Do it alongside bug 1.

### Testing Protocol

Run `test_single.py` on examples 0 and 1 for each configuration:

| Config | Description |
|--------|-------------|
| A (baseline) | Current code, no changes |
| B | + color normalization fix + LayerNorm eps fix |
| C | + TARGET_FPS=30.0 (without color norm fix) |
| D | + color normalization + TARGET_FPS=30.0 + LayerNorm eps |

Then run `main.py` with the best config on all 10 examples.

### Expected Impact

**BUG 1 (color normalization):**
- The Stacked Hourglass model has been receiving systematically shifted inputs. Fixing this should improve 2D keypoint accuracy across ALL joints, particularly for harder joints like ankles.
- Better 2D keypoints -> better MotionBERT input -> better 3D predictions.
- Expected: 2-5 cm MPJPE improvement (hard to estimate without testing, but color normalization is a fundamental preprocessing step that affects every joint in every frame).

**CONFIG (more frames):**
- MotionBERT's temporal transformer should produce smoother, more accurate predictions with 100-150 frames vs 34-50.
- Expected: 1-3 cm MPJPE improvement, particularly in temporal consistency.

**Combined:** MPJPE could improve from 29.09 cm to potentially 22-26 cm.

## What Was Verified Correct

- `crop_scale()` normalization: matches official code exactly
- `flip_data()` joint indices: identical to official
- DSTformer constructor parameters: all match the YAML config
- `no_conf: False`: we correctly pass 3-channel input (x, y, confidence)
- `rootrel: False`: we correctly zero first-frame root Z
- MPII-to-H36M joint mapping: consistent with the official Halpe-to-H36M mapping (different source joints but same target semantics)
- COCO19-to-H36M ground truth mapping: correct
- Denormalization math: correct inverse of crop_scale
- Flip augmentation on Stacked Hourglass: correct MPII flip pairs

## Files to Change

1. **`detect.py`** -- Add color normalization in `run_hourglass()` (both original and flipped paths). Add `norm_layer` parameter in `load_motionbert_model()`.
2. **`config.py`** -- Change `TARGET_FPS` from 10.0 to 30.0.

No other files need changes. The denormalization, joint mappings, and evaluation code are all correct.

## Risk Assessment

- **BUG 1 fix:** Zero risk. This is adding a preprocessing step that the model was trained with. It can only improve or maintain performance.
- **BUG 2 fix:** Zero risk. Trivial change.
- **CONFIG change:** Low risk. More frames is strictly more information. The only downside is ~3x slower Stacked Hourglass inference.
