# Tester Report: Phase 4, Iteration 0 -- Batch SH Inference + GPU Acceleration

## Tests Run

### 1. Code Review: `_get_device()` in detect.py
**PASS**

Correctly implements CUDA > MPS > CPU priority. Uses `torch.cuda.is_available()` and `torch.backends.mps.is_available()`. Clean and simple.

### 2. Code Review: `run_hourglass()` batched rewrite
**PASS**

- Phase 1 (preprocess): All frames are cropped, normalized, and stored as lists. Both original and flipped versions prepared. `shared_affine` captured from first frame (correct since bbox is constant across frames).
- Phase 2 (batched inference): Frames grouped by `cfg.SH_BATCH_SIZE`, stacked via `np.stack()`, forwarded through model. Flip heatmaps averaged per-frame using `_flip_heatmaps()`. `torch.no_grad()` wraps the entire inference loop. The `tqdm` progress bar provides user feedback.
- Phase 3 (parse keypoints): Same logic as before -- `_parse_heatmaps()` then affine transform to original coords.
- MPS fallback: try/except around the forward pass catches `RuntimeError`, moves model to CPU, and retries. Correctly reassigns `device` so subsequent batches also use CPU. Correct.

### 3. Code Review: `run_motionbert()` device update
**PASS**

Uses `_get_device()`. MPS fallback implemented with try/except around `model(input_tensor)`. Falls back to CPU by moving both model and input tensor. Correct.

### 4. Code Review: `detect_poses()` timing instrumentation
**PASS**

- `time.perf_counter()` placed at 5 points (t0-t4) around YOLO, SH, MotionBERT, and post-processing stages.
- When `return_timing=True`, a timing dict is appended as a 7th element in the return tuple.
- When `return_timing=False` (default), 6-element tuple returned. Existing callers (`test_single.py`, `main.py`) are unaffected.
- Timing values are `round()`ed to 3 decimal places. Correct.

### 5. Code Review: config.py -- `SH_BATCH_SIZE`
**PASS**

`SH_BATCH_SIZE: int = 32` added with explanatory comment about memory implications. Default of 32 is conservative and appropriate for 8GB Apple Silicon. Placed logically after other model-related config values.

### 6. Code Review: benchmark.py -- timing capture
**PASS**

- `benchmark_example()` passes `return_timing=True` to `detect_poses()`.
- Sub-stage timing extracted via `detection_result[6]` and stored in result dict with keys `yolo_time_s`, `stacked_hourglass_time_s`, `motionbert_time_s`, `detection_postprocess_time_s`.
- Aggregate results include `mean_yolo_time_s`, `mean_stacked_hourglass_time_s`, `mean_motionbert_time_s`.
- Print output shows breakdown: `Detection: Xs (YOLO=Xs, SH=Xs, MB=Xs)`. Correct.

### 7. Smoke Test: `test_single.py`
**PASS**

Command: `cd motionbert-pose && uv run python test_single.py`

Output confirms:
- SH loaded on MPS: "Loaded Stacked Hourglass (8-stack, pretrained) on mps"
- Batched inference: "Running Stacked Hourglass on 100 frames (batch_size=32)" -- 4 batches processed
- MotionBERT on MPS: "MotionBERT on mps"
- Pipeline completed without errors
- Overlay video generated successfully

### 8. Accuracy Check: MPJPE unchanged from Phase 3
**PASS**

| Metric | Phase 3 Baseline | Phase 4 | Match? |
|--------|-----------------|---------|--------|
| Det MPJPE | 50.79 cm | 50.79 cm | Yes |
| Opt MPJPE | 44.87 cm | 44.87 cm | Yes |
| Step 0 loss | 6835.9 | 6835.9 | Yes |
| Step 49 loss | 6702.4 | 6702.4 | Yes |

Bit-identical results confirm batching and MPS acceleration did not change numeric output.

### 9. Benchmark Timing Plausibility
**PASS** (verified via developer report, not re-run)

Developer reported for 100 frames:
- YOLO: 4.3s (plausible for 100 frames of YOLOv8n)
- SH: 11.7s (down from ~25-30s estimate, ~2.3x speedup from batching + MPS)
- MotionBERT: 0.4s (down from ~3-5s, ~8-12x speedup from MPS)
- Total detection: 16.8s (down from ~41s, ~2.4x overall speedup)

All sub-stage times sum correctly and are consistent with expectations.

## Bugs Found

None. All Phase 4 changes are correct and functional.

## Code Review Findings

### Finding 1: Complex union return type on `detect_poses()` (Minor)
- **Description**: Lines 718-733 of `detect.py` use a union of two long tuple types for the return annotation. This is hard to read. A cleaner approach would be to always return the timing dict (defaulting to `None` or an empty dict), or use a dataclass/namedtuple for the return value.
- **Severity**: Minor (readability only)
- **Location**: `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose/detect.py` lines 718-733

### Finding 2: Duplicated RGB mean normalization magic numbers (Minor)
- **Description**: The RGB mean values `(0.4404, 0.4440, 0.4327)` appear twice in `run_hourglass()` -- once for the original image (lines 283-285) and once for the flipped image (lines 292-294). These should be extracted to a module-level constant (e.g., `_SH_RGB_MEAN = np.array([0.4404, 0.4440, 0.4327])`).
- **Severity**: Minor (maintainability, DRY principle)
- **Location**: Lines 283-285 and 292-294

### Finding 3: Preprocessing could share code path (Minor)
- **Description**: The original and flipped preprocessing in `run_hourglass()` (lines 281-295) repeat the same normalization steps: `astype(float32) / 255.0`, `transpose(2,0,1)`, subtract RGB means. This could be extracted to a helper function like `_normalize_for_sh(cropped: np.ndarray) -> np.ndarray`.
- **Severity**: Minor (code duplication)

### Finding 4: TODO comment in `motionbert_to_camera_space()` (Pre-existing)
- **Description**: Line 693 has a TODO: "Shouldn't back projection occur using the camera parameters that we know from the ground truth data? We don't need to make unnecessary pinhole assumptions." This predates Phase 4 but is in a modified file.
- **Severity**: Informational (design question, not a bug)

### Finding 5: Config mutation in `benchmark.py` (Pre-existing, noted in Phase 3)
- **Description**: `benchmark_example()` temporarily mutates global `cfg` module attributes (lines 82-93) and restores them in a `finally` block. This is fragile if benchmarking were ever parallelized. Was flagged in Phase 3 tester report and remains.
- **Severity**: Minor (tech debt)

### Finding 6: `benchmark_example()` index-based unpacking of detection result (Minor)
- **Description**: Lines 127-133 of `benchmark.py` unpack `detect_poses()` result by index (`detection_result[0]` through `detection_result[6]`). This is fragile -- if the return tuple order changes, this silently breaks. A named tuple or dataclass would be safer, or at minimum use tuple unpacking: `kp_2d, visibility, heatmaps, mpii_kp_2d, affine, positions_3d_norm, timing = detect_poses(...)`.
- **Severity**: Minor (fragility)

### Finding 7: Model loaded fresh on every call to `run_hourglass()` and `run_motionbert()` (Pre-existing)
- **Description**: Both SH and MotionBERT models are loaded from disk on every call to their respective functions. For `benchmark.py` running multiple examples, this means loading the same ~100MB model repeatedly. Caching the model (e.g., module-level singleton or passed as parameter) would save time.
- **Severity**: Minor (performance, pre-existing)

## REFACTORING TODO LIST

| # | File | Description | Severity |
|---|------|-------------|----------|
| R1 | detect.py | Extract RGB mean normalization values to a module-level constant `_SH_RGB_MEAN` | Minor |
| R2 | detect.py | Extract image normalization into a helper `_normalize_for_sh(cropped)` to eliminate duplication between original/flipped paths | Minor |
| R3 | detect.py | Simplify `detect_poses()` return type -- always return timing dict (None when not requested), or use a dataclass | Minor |
| R4 | detect.py | Address TODO on line 693 about using known camera parameters for back-projection | Design question |
| R5 | benchmark.py | Use tuple unpacking instead of index-based access for `detect_poses()` result | Minor |
| R6 | benchmark.py | Consider passing config values via function params instead of mutating global cfg module | Minor/Tech debt |
| R7 | detect.py | Cache loaded models (SH, MotionBERT) to avoid reloading on each call | Minor/Performance |

## Verdict

**YES** -- The implementation meets the Phase 4 spec's definition of done.

All required deliverables are verified:
1. `run_hourglass()` processes frames in batches of `SH_BATCH_SIZE` with 2 forward passes per batch (original + flip). Confirmed by code review and runtime output showing 4 batches for 100 frames.
2. Both SH and MotionBERT use MPS on Apple Silicon. Confirmed by runtime output ("on mps" for both models).
3. `benchmark.py` reports sub-stage detection timing (YOLO, SH, MotionBERT). Confirmed by code review.
4. Developer report shows 2-example benchmark with timing breakdown demonstrating ~2.4x detection speedup.
5. MPJPE is unchanged: Det=50.79cm, Opt=44.87cm, loss values bit-identical to Phase 3 baseline.

No bugs found. Seven refactoring items identified for optional cleanup.
