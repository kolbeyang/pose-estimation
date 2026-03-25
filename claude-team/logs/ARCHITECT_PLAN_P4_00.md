# Architect Plan: Phase 4, Iteration 0 -- Batch SH Inference + GPU Acceleration

## Goal Summary

Phase 3 reduced optimization from 98% to 2% of pipeline time. Detection (YOLO + Stacked Hourglass + MotionBERT) now dominates at ~41s per example (~98%). This phase targets the detection bottleneck with two optimizations:

**Phase 4a** -- Batch the Stacked Hourglass forward pass. Currently `run_hourglass()` loops over frames one at a time, calling `model(inp)` and `model(inp_flip)` per frame (2 forward passes per frame). The SH model is a standard CNN that natively supports batch dimensions. We should preprocess all frames, stack them into a batch tensor, and run a single (or few) forward pass(es).

**Phase 4b** -- Move SH and MotionBERT to GPU (MPS on Apple Silicon). Currently both models use `torch.device("cuda" if torch.cuda.is_available() else "cpu")`, which falls back to CPU on Apple Silicon. MPS (Metal Performance Shaders) is available on Apple Silicon Macs and provides GPU acceleration for PyTorch.

## Key Discovery: HumanPosePredictor Already Supports Batched + Flip

The `stacked_hourglass` library ships a `HumanPosePredictor` class (in `predictor.py`) with an `estimate_heatmaps(images, flip=True)` method that:
- Accepts a list/batch of images
- Handles preprocessing (resize to 256x256, color normalization)
- Runs batched forward pass
- Handles flip augmentation internally (fliplr, forward, flip_back, average)
- Returns `(B, 16, 64, 64)` heatmaps

However, we should NOT use `HumanPosePredictor` because:
1. Its `prepare_image()` uses `color_normalize(image, rgb_mean, rgb_stddev)` which divides by stddev. Our code subtracts mean only (no stddev division). Switching would change numeric results.
2. We need raw heatmaps for our optimization pipeline, and we need fine control over the flip pair swapping (our `_flip_heatmaps` uses MPII_FLIP_PAIRS).

Instead, we replicate the batching pattern ourselves in `run_hourglass()`.

## Files to Modify

### 1. `motionbert-pose/detect.py`
All changes are in this file. Specifically:
- `run_hourglass()` -- batch SH inference
- `run_motionbert()` -- GPU device selection
- `run_hourglass()` device selection -- GPU device selection
- A new helper `_get_device()` for unified device selection

### 2. `motionbert-pose/config.py`
- Add `SH_BATCH_SIZE` config parameter

### 3. `motionbert-pose/benchmark.py`
- Add sub-stage timing within detection (YOLO time, SH time, MotionBERT time) so we can measure the speedup

## Files to Create

None.

## Step-by-Step Instructions

### Step 1: Add device selection helper to `detect.py`

Add a function at the top of `detect.py` (after imports, before `detect_person_bbox`):

```python
def _get_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### Step 2: Add `SH_BATCH_SIZE` to `config.py`

Add after the existing config parameters:

```python
# Stacked Hourglass batch size for 2D pose inference.
# Higher = faster but more memory. 256x256x3 float32 = 768KB per frame.
# Batch of 32: ~25MB GPU memory for input tensor alone.
# Batch of 150 (max frames): ~115MB. Safe for most GPUs.
SH_BATCH_SIZE: int = 32
```

32 is a conservative default. 150 frames at 256x256 is only ~115MB for the input tensor. The model activations are the larger concern (HG8 has 8 stacks), but 32 frames should be safe even on 8GB Apple Silicon.

### Step 3: Rewrite `run_hourglass()` for batched inference

Replace the per-frame loop in `run_hourglass()` with batched processing. The new flow:

1. **Preprocess all frames** -- Crop, resize, normalize all frames upfront. Store as a list of numpy arrays.
2. **Stack into batches** -- Group frames into batches of `SH_BATCH_SIZE`.
3. **Run batched forward pass** -- For each batch, stack the original images into `(B, 3, 256, 256)`, run `model(batch)`, then stack the flipped images, run `model(batch_flip)`.
4. **Post-process** -- Average original + flipped heatmaps, parse keypoints, scale to original coords.

Here is the detailed rewrite of `run_hourglass()`:

```python
def run_hourglass(
    frames_rgb: list[np.ndarray],
    bbox: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    from stacked_hourglass import hg8
    import config as cfg

    device: torch.device = _get_device()

    # Load pretrained model (same loading logic as before)
    try:
        model = hg8(pretrained=True)
    except RuntimeError:
        model = hg8(pretrained=False)
        cached_path = os.path.join(
            torch.hub.get_dir(), "checkpoints", "bearpaw_hg8-90e5d470.pth"
        )
        if os.path.exists(cached_path):
            state_dict = torch.load(cached_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError("Stacked Hourglass weights not found.")

    model = model.to(device)
    model.eval()
    print(f"  Loaded Stacked Hourglass (8-stack, pretrained) on {device}")

    # --- Phase 1: Preprocess all frames ---
    preprocessed: list[np.ndarray] = []       # normalized CHW arrays
    preprocessed_flip: list[np.ndarray] = []  # flipped normalized CHW arrays
    shared_affine: np.ndarray | None = None

    for frame in frames_rgb:
        cropped, affine = crop_and_resize(frame, bbox, target_size=256)
        if shared_affine is None:
            shared_affine = affine

        # Normalize: [0,1], HWC->CHW, subtract RGB means
        img = cropped.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img[0] -= 0.4404
        img[1] -= 0.4440
        img[2] -= 0.4327
        preprocessed.append(img)

        # Flipped version
        cropped_flip = cropped[:, ::-1].copy()
        img_flip = cropped_flip.astype(np.float32) / 255.0
        img_flip = np.transpose(img_flip, (2, 0, 1))
        img_flip[0] -= 0.4404
        img_flip[1] -= 0.4440
        img_flip[2] -= 0.4327
        preprocessed_flip.append(img_flip)

    assert shared_affine is not None
    n_frames = len(frames_rgb)
    batch_size = cfg.SH_BATCH_SIZE

    # --- Phase 2: Batched inference ---
    all_heatmaps: list[np.ndarray] = []

    print(f"  Running Stacked Hourglass on {n_frames} frames (batch_size={batch_size})...")

    with torch.no_grad():
        for start in tqdm(range(0, n_frames, batch_size), desc="  2D Pose"):
            end = min(start + batch_size, n_frames)
            batch_imgs = preprocessed[start:end]
            batch_flips = preprocessed_flip[start:end]

            # Stack into tensor and run forward pass
            inp = torch.from_numpy(np.stack(batch_imgs)).to(device)
            output = model(inp)
            heatmaps_batch = output[-1].cpu().numpy()  # (B, 16, 64, 64)

            # Flipped forward pass
            inp_flip = torch.from_numpy(np.stack(batch_flips)).to(device)
            output_flip = model(inp_flip)
            heatmaps_flip_batch = output_flip[-1].cpu().numpy()  # (B, 16, 64, 64)

            # Flip heatmaps and swap joints
            for i in range(end - start):
                hm_flip = _flip_heatmaps(heatmaps_flip_batch[i])
                hm_avg = (heatmaps_batch[i] + hm_flip) / 2.0
                all_heatmaps.append(hm_avg)

    # --- Phase 3: Parse keypoints ---
    all_keypoints_2d: list[np.ndarray] = []
    for heatmaps in all_heatmaps:
        keypoints_64 = _parse_heatmaps(heatmaps)
        keypoints_orig = keypoints_64.copy()
        keypoints_orig[:, :2] *= 4  # 64 -> 256
        for j in range(16):
            x_256 = keypoints_orig[j, 0]
            y_256 = keypoints_orig[j, 1]
            keypoints_orig[j, 0] = shared_affine[0, 0] * x_256 + shared_affine[0, 2]
            keypoints_orig[j, 1] = shared_affine[1, 1] * y_256 + shared_affine[1, 2]
        all_keypoints_2d.append(keypoints_orig)

    return all_keypoints_2d, all_heatmaps, shared_affine
```

Key differences from the original:
- Preprocessing is fully separated from inference (no interleaving)
- `np.stack()` builds batch tensors from pre-processed arrays
- Two model forward passes per batch (original + flipped) instead of two per frame
- `_flip_heatmaps()` still applied per-frame (it swaps joint indices, can't easily vectorize)
- Device is `_get_device()` instead of hardcoded cuda/cpu

### Step 4: Update `run_motionbert()` to use `_get_device()`

In `run_motionbert()`, replace:
```python
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
with:
```python
device: torch.device = _get_device()
```

Also in `load_motionbert_model()`, keep `map_location="cpu"` for loading weights (this is correct -- load to CPU first, then move to device).

Add a print statement after model.to(device) in `run_motionbert()`:
```python
print(f"  MotionBERT on {device}")
```

### Step 5: Add sub-stage timing to `benchmark.py`

Currently `benchmark.py` times the entire `detect_poses()` call as one "detection_time_s" block. To measure the speedup from batching SH and GPU acceleration, we need to break detection into sub-stages.

Modify `detect_poses()` in `detect.py` to return timing info. Add an optional parameter:

```python
def detect_poses(
    frames_rgb: list[np.ndarray],
    return_timing: bool = False,
) -> tuple[...]:
```

Inside `detect_poses()`, wrap each sub-call with `time.perf_counter()`:

```python
import time

t0 = time.perf_counter()
union_bbox = detect_person_bbox(frames_rgb)
t1 = time.perf_counter()

all_keypoints_2d, all_heatmaps, affine = run_hourglass(frames_rgb, union_bbox)
t2 = time.perf_counter()

positions_3d_norm = run_motionbert(all_keypoints_2d)
t3 = time.perf_counter()

# ... existing post-processing ...
t4 = time.perf_counter()
```

If `return_timing` is True, append a timing dict to the return tuple:
```python
timing = {
    "yolo_s": round(t1 - t0, 3),
    "stacked_hourglass_s": round(t2 - t1, 3),
    "motionbert_s": round(t3 - t2, 3),
    "postprocess_s": round(t4 - t3, 3),
}
```

Update `benchmark_example()` in `benchmark.py` to pass `return_timing=True` and include the sub-stage timing in the result dict.

### Step 6: Update YOLO device selection (bonus, low effort)

In `detect_person_bbox()`, YOLO uses its own device selection. This is fine -- `ultralytics` YOLO auto-detects GPU. No change needed here.

## Integration Points

- `run_hourglass()` signature is unchanged. All callers (`detect_poses()`, and transitively `main.py`, `benchmark.py`) work without modification.
- `run_motionbert()` signature is unchanged.
- `detect_poses()` gains an optional `return_timing` parameter with default `False`, so existing callers are unaffected.
- `config.SH_BATCH_SIZE` is a new config value with a sensible default.

## Risks and Edge Cases

### 1. MPS Compatibility
MPS support in PyTorch is still maturing. Some operations may not be supported or may produce different numeric results. Specific concerns:
- **BatchNorm2d on MPS**: Should work fine in eval mode (no running stats update).
- **grid_sample on MPS**: Used in optimization scoring, not in SH inference. Not a concern here.
- **MotionBERT's DSTformer on MPS**: Uses LayerNorm, attention, etc. These should work on MPS but may need testing.
- **Mitigation**: If MPS fails for a specific model, fall back to CPU for that model only. The developer should wrap the device selection in a try/except for the actual forward pass and fall back gracefully.

### 2. Memory Constraints
- 150 frames * 256x256 * 3 * float32 = ~113MB for input tensor.
- HG8 intermediate activations are larger (~30-50MB per image at peak). Batch of 32 could use ~1-1.5GB.
- With `torch.no_grad()`, no gradient storage needed, so memory is just activations + parameters.
- HG8 parameters: ~25M params * 4 bytes = ~100MB.
- **Total estimate for batch=32**: ~1.5-2GB. Safe for 8GB Apple Silicon (leaves plenty for OS + other processes).
- **Batch=150 could use ~5-7GB**: Potentially tight on 8GB machines. The `SH_BATCH_SIZE=32` default is conservative.

### 3. Flip Heatmap Post-Processing
`_flip_heatmaps()` operates per-frame because it swaps joint channels. This could be vectorized with fancy indexing on the batch dimension, but it's a trivial cost (numpy array slicing on 16x64x64) compared to the forward pass. Not worth optimizing.

### 4. YOLO Batching
YOLO already processes frames one at a time in a loop. Batching YOLO is possible but YOLO is already fast (~0.05s/frame) and is a small fraction of detection time compared to SH. Not worth the complexity in this phase.

## Expected Speedup

From the Phase 3 final report, detection takes ~41s for ~150 frames. The breakdown is approximately:
- YOLO: ~7-10s (runs per-frame but is lightweight)
- Stacked Hourglass: ~25-30s (the heavyweight -- 8-stack CNN, 2 passes per frame)
- MotionBERT: ~3-5s (single forward pass on all frames, already efficient)

**Phase 4a (Batched SH)**: Eliminating per-frame Python loop overhead and enabling GPU batching should reduce SH time by 2-5x on CPU alone (better GPU utilization, fewer kernel launches). Expected SH time: ~5-15s.

**Phase 4b (MPS GPU)**: CNN inference on MPS is typically 2-4x faster than CPU on Apple Silicon. Combined with batching, SH could drop to ~3-8s. MotionBERT (transformer) may see 1.5-3x speedup on MPS.

**Combined expected total detection time**: ~15-25s (down from ~41s), a 1.6-2.7x overall speedup.

## Definition of Done

1. `run_hourglass()` processes frames in batches of `SH_BATCH_SIZE` with 2 forward passes per batch (original + flip), not per frame.
2. Both SH and MotionBERT use MPS when available on Apple Silicon.
3. `benchmark.py` reports sub-stage detection timing (YOLO, SH, MotionBERT).
4. Run benchmark on at least 2 examples comparing before/after. Report timing breakdown.
5. Verify MPJPE is unchanged (numeric results should be identical or within floating-point tolerance).
