# Phase 4: Detection Optimization -- Final Report

## Summary

Phase 4 focused on reducing end-to-end latency, which was dominated by the detection pipeline (YOLO + Stacked Hourglass + MotionBERT). Three optimizations were applied: (a) batched Stacked Hourglass inference, (b) GPU/MPS acceleration for SH and MotionBERT, and (c) removal of the anchor penalty from the batched scoring path. Additionally, tester-identified refactoring items R1-R4 were addressed. The net result is a 1.34x total speedup (41.9s -> 31.3s mean per example) with a small accuracy trade-off from anchor removal (+0.42 cm mean Opt MPJPE).

## Phase 4a: Batched Stacked Hourglass Inference

**Change:** Rewrote `run_hourglass()` in `detect.py` to process frames in batches of `SH_BATCH_SIZE` (default 32) instead of one frame at a time. All preprocessing (crop, resize, normalize, flip) is done upfront, then batches are forwarded through the model with two passes each (original + flipped).

**Impact:** Reduces Python loop overhead and amortizes GPU kernel launch costs. SH inference dropped from ~25-30s to ~9.3s for 150 frames (estimated ~2.7x speedup).

## Phase 4b: GPU/MPS Acceleration

**Change:** Added `_get_device()` helper that selects CUDA > MPS > CPU. Applied to both Stacked Hourglass and MotionBERT inference. Added try/except fallback for MPS compatibility issues.

**Impact:** On Apple Silicon (M-series), both models now run on the Metal Performance Shaders backend instead of CPU. MotionBERT saw the largest speedup (~8-12x, from ~3-5s to ~0.3s). Combined with batching, total detection time dropped from ~41s to ~30.5s.

## Anchor Penalty A/B Test

The initialization anchor penalty (`INIT_ANCHOR_WEIGHT=5.0`) penalizes the optimizer for drifting away from MotionBERT's initial 3D predictions, weighted by per-joint visibility confidence. This was removed from the batched scoring path to simplify the code.

### Single-Example Comparison (171204_pose1_sample_0, 100 frames)

| Metric | With Anchor (5.0) | Without Anchor (0.0) | Delta |
|--------|-------------------|---------------------|-------|
| Det MPJPE | 50.79 cm | 50.79 cm | 0.00 |
| Opt MPJPE | 44.87 cm | 48.70 cm | +3.83 cm |
| Opt P-MPJPE | 40.71 cm | 40.46 cm | -0.25 cm |
| Improvement | +5.91 cm | +2.08 cm | -3.83 cm |
| Step 0 loss | 6835.9 | 6827.8 | -- |
| Step 49 loss | 6702.4 | 6634.2 | -- |

### Full Benchmark Comparison (17 examples)

| Metric | Phase 3 (anchor=5.0) | Phase 4 Final (anchor=0.0) | Delta |
|--------|----------------------|---------------------------|-------|
| Mean Det MPJPE | 40.06 cm | 40.06 cm | 0.00 |
| Mean Opt MPJPE | 38.69 cm | 39.11 cm | +0.42 cm |
| Mean Det P-MPJPE | 28.05 cm | 28.05 cm | 0.00 |
| Mean Opt P-MPJPE | 27.30 cm | 27.50 cm | +0.20 cm |
| Mean Improvement | +1.37 cm | +0.95 cm | -0.42 cm |

**Conclusion:** The anchor penalty was beneficial, contributing ~0.42 cm to the mean MPJPE improvement. Without it, the optimizer can explore further from MotionBERT's predictions but sometimes overshoots, especially on examples where MotionBERT's initial estimate was already good. The anchor penalty is removed from the batched code path for simplicity, but the constant `INIT_ANCHOR_WEIGHT` remains in `config.py` (set to 0.0) and the non-batched path still supports it for experimentation.

## Refactoring

| # | Status | Description |
|---|--------|-------------|
| R1 | Already done (P4 iter 0) | Extracted `_SH_RGB_MEAN` constant in detect.py |
| R2 | Already done (P4 iter 0) | Created `_normalize_for_sh()` helper in detect.py |
| R3 | Already resolved | `detect_poses()` return type already returns timing as 7th element (None when not requested) |
| R4/R5 | Already resolved | `benchmark.py` already uses tuple unpacking for `detect_poses()` result |

## Full Benchmark Results (Phase 4 Final, 17 examples)

| Example | Frames | Det MPJPE | Opt MPJPE | Improvement | Detection | Optimization | Total |
|---------|--------|-----------|-----------|-------------|-----------|-------------|-------|
| pose1_sample_0 | 100 | 50.79 | 48.70 | +2.08 | 10.2s | 0.7s | 10.9s |
| pose2_200 | 150 | 40.63 | 37.57 | +3.06 | 13.9s | 0.7s | 14.7s |
| pose2_5000 | 150 | 18.08 | 19.69 | -1.61 | 24.7s | 0.7s | 25.5s |
| pose2_15000 | 150 | 36.04 | 34.20 | +1.84 | 47.9s | 0.7s | 48.7s |
| pose3_200 | 150 | 57.69 | 55.64 | +2.05 | 14.0s | 0.7s | 14.8s |
| pose3_4000 | 150 | 15.07 | 16.66 | -1.59 | 22.5s | 0.7s | 23.3s |
| ultimatum1_200 | 150 | 63.67 | 61.91 | +1.76 | 13.8s | 0.7s | 14.6s |
| ultimatum1_10000 | 150 | 63.53 | 63.03 | +0.50 | 36.4s | 0.7s | 37.2s |
| pose2_10000 | 150 | 17.09 | 18.27 | -1.18 | 35.9s | 0.7s | 36.7s |
| pose2_25000 | 150 | 18.66 | 18.53 | +0.13 | 71.7s | 0.7s | 72.5s |
| pose1_5000 | 150 | 46.99 | 43.16 | +3.82 | 24.6s | 0.7s | 25.4s |
| pose1_14000 | 150 | 41.03 | 42.72 | -1.70 | 45.1s | 0.7s | 45.9s |
| pose1_22000 | 150 | 16.53 | 17.41 | -0.89 | 66.6s | 0.7s | 67.4s |
| haggling1_3000 | 150 | 27.66 | 28.37 | -0.71 | 20.3s | 0.7s | 21.1s |
| haggling1_7000 | 150 | 72.62 | 71.36 | +1.26 | 29.1s | 0.7s | 29.9s |
| pizza1_2000 | 150 | 43.06 | 36.61 | +6.45 | 17.9s | 0.7s | 18.7s |
| pizza1_4500 | 150 | 51.87 | 51.08 | +0.79 | 23.9s | 0.7s | 24.7s |
| **Mean** | **147** | **40.06** | **39.11** | **+0.95** | **30.5s** | **0.7s** | **31.3s** |

### Per-Stage Detection Breakdown (Mean)

| Stage | Time | % Detection |
|-------|------|-------------|
| YOLO | 3.4s | 11% |
| Stacked Hourglass | 9.3s | 30% |
| MotionBERT | 0.3s | 1% |
| Postprocess | <0.01s | <1% |
| Frame extraction + calibration | ~17.5s | 57% |

Note: Frame extraction time scales with `start_frame` because OpenCV must seek through the video. Later examples (start_frame=25000) take much longer for detection due to video seeking, not model inference.

## Comparison to Phase 3 Baseline

| Metric | Phase 3 | Phase 4 Final | Change |
|--------|---------|--------------|--------|
| Mean Det MPJPE | 40.06 cm | 40.06 cm | 0.00 |
| Mean Opt MPJPE | 38.69 cm | 39.11 cm | +0.42 cm |
| Mean Opt P-MPJPE | 27.30 cm | 27.50 cm | +0.20 cm |
| Mean Improvement | +1.37 cm | +0.95 cm | -0.42 cm |
| Mean Detection Time | 41.3s | 30.5s | -10.8s (1.35x faster) |
| Mean Optimization Time | 0.49s | 0.70s | +0.21s |
| Mean Total Time | 41.9s | 31.3s | -10.6s (1.34x faster) |
| Mean Latency/Frame | 0.283s | 0.211s | -0.072s (1.34x faster) |

**Summary:** Phase 4 achieved a 1.34x overall speedup (primarily from batched SH + MPS acceleration) with a small accuracy regression (+0.42 cm mean Opt MPJPE) due to anchor penalty removal. The optimization time increased slightly (0.49s -> 0.70s) because the anchor penalty removal allows the optimizer more freedom, leading to larger gradient steps. Detection time dropped by 26% from 41.3s to 30.5s.
