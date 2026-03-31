# Testing Report: Phase 3 — Model Loading Optimization

**Evaluator:** Eve (QA)
**Date:** 2026-03-31

## Verdict: PASS

All checks pass. Models are loaded once and reused across examples. No issues found.

## 1. Code Review

### `pose-optimizer/run_motionbert/detect.py`

- **`MotionBertModels` dataclass** (line 39): Holds `yolo`, `hourglass`, `motionbert`, and `device`. Clean, no global state.
- **`load_all_models()`** (line 52): Loads all three models, moves to device, calls `.eval()` on hourglass. Prints one line per model load. Returns dataclass.
- **`detect_person_bbox()`** (line 127): Accepts `yolo: Any = None`. Falls back to loading YOLO internally when None. Correct.
- **`run_hourglass()`** (line 275): Accepts `model` and `device` with None defaults. Falls back to internal loading when either is None. Correct.
- **`run_motionbert()`** (line 499): Same pattern -- `model` and `device` default to None, falls back to loading. Correct.
- **`detect_poses()`** (line 748): Accepts `models: MotionBertModels | None = None`. Unpacks and passes `models.yolo`, `models.hourglass`, `models.motionbert`, `models.device` through to sub-functions. Falls back to None (triggering per-call loading) when not provided. Correct.

### `pose-optimizer/run_motionbert/__init__.py`

- Line 46-49: Imports `detect_poses`, `load_all_models`, `motionbert_to_camera_space`, `MotionBertModels`.
- `run_pipeline()` (line 404): Calls `load_all_models()` at line 428, before the example loop. Passes `models=models` to `process_example()` at line 441.
- `process_example()` (line 105): Accepts `models: MotionBertModels | None = None`, passes to `detect_poses()` at line 185.

### `pose-optimizer/run_mediapipe/detect.py`

- **`load_landmarker()`** (line 40): Creates PoseLandmarker with IMAGE mode. Prints "Loaded MediaPipe PoseLandmarker".
- **`detect_poses()`** (line 58): Accepts `landmarker: vision.PoseLandmarker | None = None`. Uses `should_close` flag to track ownership -- only closes landmarker if it was created internally. Correct lifecycle management via try/finally.

### `pose-optimizer/run_mediapipe/__init__.py`

- Line 45: Imports `detect_poses`, `load_landmarker`, `mediapipe_3d_to_camera`.
- `run_pipeline()` (line 403): Calls `load_landmarker()` at line 427, passes to `process_example()` at line 440, calls `landmarker.close()` at line 450. Correct lifecycle.
- `process_example()` (line 101): Accepts `landmarker=None`, passes through at line 173.

### Cross-cutting checks

- **Backward compatibility**: All new parameters default to None. When None, each function falls back to loading models internally. Any caller that does not pass models will still work correctly (just slower).
- **No global state**: No module-level model variables. All models passed explicitly via function parameters.
- **No singletons or caching**: Models are plain objects owned by `run_pipeline` scope.

## 2. Smoke Test Results

Ran both pipelines with a 2-example config (171204_pose1_sample, frames 0-14 and 30-44, 3 optimization steps).

### MediaPipe Pipeline
- Both examples processed successfully (2/2).
- Example 1: Det MPJPE 12.31 cm, Opt MPJPE 12.19 cm.
- Example 2: Det MPJPE 12.81 cm, Opt MPJPE 13.21 cm.
- No errors.

### MotionBERT Pipeline
- Both examples processed successfully (2/2).
- Example 1: Det MPJPE 18.23 cm, Opt MPJPE 18.31 cm.
- Example 2: Det MPJPE 18.19 cm, Opt MPJPE 18.22 cm.
- No errors.

## 3. Single-Load Verification

Counted model loading messages in output logs:

| Message | Expected | Actual |
|---|---|---|
| "Loaded MediaPipe PoseLandmarker" | 1 | 1 |
| "Loaded YOLOv8n" | 1 | 1 |
| "Loaded Stacked Hourglass" | 1 | 1 |
| "Loaded MotionBERT-Lite" | 1 | 1 |
| "All models loaded on mps" | 1 | 1 |

All model loading messages appear exactly once, at pipeline startup before the example loop. No model loading occurs during per-example processing.

## 4. Definition of Done Checklist

- [x] `MotionBertModels` dataclass and `load_all_models()` exist in `run_motionbert/detect.py`
- [x] `load_landmarker()` exists in `run_mediapipe/detect.py`
- [x] `run_pipeline()` in both `__init__.py` files loads models once before the example loop
- [x] `process_example()` in both pipelines accepts and forwards pre-loaded models
- [x] `detect_person_bbox`, `run_hourglass`, `run_motionbert` accept pre-loaded models (fall back to internal loading when None)
- [x] `detect_poses` in MediaPipe does not create a new PoseLandmarker when one is provided
- [x] Running a 2-example config produces "Loaded" log lines exactly once per model
- [x] No global/module-level model state -- all model references passed explicitly

## Issues Found

None. No new items for TODO.md.
