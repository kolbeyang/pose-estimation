# Testing Report 00: Comprehensive QA

**Date:** 2026-03-29
**Tester:** Evaluator Eve
**Branch:** feat/combined-motionbert-mediapipe
**Commit:** b00dbd0

---

## 1. Smoke Tests

### Unit Tests: PASS
All 55 unit tests pass (`uv run python -m pytest experiment/ -v`):
- Phase 1 (28 tests): Skeleton, Camera, Config, CMU Data
- Phase 2 (22 tests): FK, Evaluation metrics, Scoring
- Phase 3 (5 tests): Optimizer core

### MotionBert Pipeline: PASS
`uv run python motionbert.py motionbert/run_single_config.json` completes without error.
- 34 frames processed (171204_pose1_sample, subsampled to 10fps)
- Output directory: `output/motionbert_2026_03_29_19_50/171204_pose1_sample_0/`

### MediaPipe Pipeline: PASS
`uv run python mediapipe.py mediapipe/run_single_config.json` completes without error.
- 34 frames processed (same example)
- Output directory: `output/mediapipe_2026_03_29_19_50/171204_pose1_sample_0/`

### Output Files Present: PASS
Both pipelines produce identical output structures:
- `summary.png` -- 4-panel summary (loss, per-joint error, per-frame MPJPE, bone lengths)
- `overlay_video.mp4` -- valid mp4 (1920x1080, 5fps, 34 frames, ~4MB)
- `results.json` -- all metrics and config
- `graphs/` -- individual graph images + `trajectories/` subdirectory with per-joint XYZ plots
- `trajectories.json` -- GT, raw, optimized trajectories + camera data

---

## 2. Edge Case Tests

### All 9 Metrics Present and Valid: PASS
Both pipelines produce all 18 metric values (9 det_ + 9 opt_), all finite positive numbers.

### SI-MPJPE <= MPJPE: PASS
- MotionBert det: SI-MPJPE 0.5047 <= MPJPE 0.5145
- MotionBert opt: SI-MPJPE 0.4928 <= MPJPE 0.5120
- MediaPipe det: SI-MPJPE 0.1280 <= MPJPE 0.1539
- MediaPipe opt: SI-MPJPE 0.1202 <= MPJPE 0.1509

### SI-MPJVE vs MPJVE: NOTE
SI-MPJVE can exceed MPJVE (e.g., MediaPipe det: 0.0444 > 0.0386). This is mathematically correct -- the scale optimized for positions is applied to velocities, and a scale != 1.0 can increase velocity error. Not a bug.

### Optimizer Does Not Dramatically Worsen Results: PASS (with notes)
**MediaPipe:** Optimizer improves all metrics. VW-SI-MPJPE: 11.74 -> 11.04 cm (6% improvement).

**MotionBert:** Mixed results on this single example:
- VW-SI-MPJPE increased marginally: 18.81 -> 19.20 cm (+2.1%)
- VW-MPJVE improved significantly: 3.16 -> 2.73 cm/f (-13.8%)
- VW-SI-MPJVE improved: 3.23 -> 2.79 cm/f (-13.6%)

The optimizer trades marginal position accuracy for substantially better velocity (smoothness). This is expected behavior with motion penalties on a short clip. Should be validated across the full 25-example run.

### Overlay Videos Valid: PASS
Both overlay videos are valid mpeg4 files confirmed by ffprobe (1920x1080, 5fps, 6.8s duration).

### Overlay Video Visual Inspection: PASS
Extracted mid-frame (frame 17) from both:

**MotionBert overlay:** Shows human subject in CMU Panoptic geodesic dome with arms outstretched. Three skeleton overlays visible: green (raw MotionBert), red (optimized), blue (GT). The green skeleton's lower body diverges significantly from GT (legs positioned incorrectly), consistent with the high MPJPE. Upper body tracking is much closer. Heatmaps are subtle but present. Frame label and legend visible.

**MediaPipe overlay:** Same scene with tighter skeleton alignment overall. Synthetic Gaussian heatmap blobs clearly visible as warm-colored overlays centered on joint locations. All three skeletons (green/red/blue) track the subject well. The closer alignment is consistent with MediaPipe's lower MPJPE.

---

## 3. Code Review

### Optimizer (`optimize/__init__.py`): PASS
- Clean pure-function design: `optimize(raw_3d, camera, config, ...) -> (improved_3d, bone_lengths, loss_history)`
- Supports both heatmap modes: real SH (MotionBert) and synthetic Gaussian (MediaPipe)
- 3 param groups with separate LRs (root_pos at 1.5x, angles at 1x, bone_lengths at separate LR)
- Batched FK for vectorized optimization
- Bone length clamping (min=0.01) after each step
- Input validation: raises ValueError if neither heatmaps nor target_2d provided

### Evaluation (`evaluate.py`): PASS
- All 9 metrics implemented correctly
- VW uses frame-boundary visibility via `compute_visibility_weights(gt_3d_cam, camera)` which projects GT 3D to 2D and calls `camera.is_in_frame()` -- binary 0/1 weights, NOT confidence-based
- SI uses single global scalar across all frames (via `scipy.optimize.minimize_scalar`)
- VW-SI-MPJPE uses `_optimal_scale_weighted` which respects visibility in scale optimization
- Velocity weights use `np.minimum(weights[:-1], weights[1:])` -- both frames must be visible
- `evaluate()` function applies root-relative transform and eval joint filtering

### Camera (`camera.py`): PASS
All required methods present:
- `world_to_camera(points_world)` -- R @ pts + t
- `camera_to_world(points_cam)` -- R^T @ (pts - t)
- `camera_to_image(points_cam)` -- pinhole projection (numpy)
- `camera_to_image_torch(points_cam)` -- differentiable version
- `is_in_frame(points_2d)` -- frame boundary check
- `from_panoptic_calibration()` class method
- `to_dict()` / `from_dict()` serialization
- Legacy aliases preserved (`world_to_image`, `world_to_image_torch`)

### Original Directories Untouched: PASS
`git diff --stat HEAD -- mediapipe-pose/ motionbert-pose/` returns empty -- no modifications.

---

## 4. Output Structure Verification

### Directory Structure: PASS
```
output/
    motionbert_2026_03_29_19_50/        # Timestamped run directory
        171204_pose1_sample_0/          # Per-video subdirectory
            summary.png                 # 4-panel summary
            overlay_video.mp4           # Skeleton overlay video
            results.json                # All metrics + config
            trajectories.json           # GT/raw/opt trajectories + camera
            graphs/
                loss_curve.png
                per_joint_error.png
                per_frame_mpjpe.png
                bone_lengths.png
                summary.png             # Same as parent summary.png
                trajectories/
                    Hip_X.png, Hip_Y.png, Hip_Z.png, ...
        aggregate_summary.png           # Cross-example summary
```

Both pipelines produce identical structures.

### results.json Format: MINOR DEVIATION
The plan (Phase 6) specifies separate top-level keys: `metrics`, `raw_metrics`, `per_joint`, `config`. The actual implementation uses a flat structure:
```json
{
    "model": "motionbert",
    "example": "171204_pose1_sample_0",
    "num_frames": 34,
    "metrics": {
        "det_mpjpe": ..., "opt_mpjpe": ...,
        "det_per_joint": [...], "opt_per_joint": [...],
        "det_per_frame_mpjpe": [...], "opt_per_frame_mpjpe": [...],
        "gt_bone_lengths": [...], "det_bone_lengths": [...]
    },
    "config": { ... }
}
```
Functionally equivalent but differs from the plan's nested structure. Logged in TODO.md.

### trajectories.json Format: PASS
Matches spec exactly:
```json
{
    "joint_names": ["Hip", "RHip", ...],  // 16 joints
    "ground_truth": [[...], ...],          // 34 frames
    "raw_prediction": [[...], ...],
    "optimized_prediction": [[...], ...],
    "camera": { "fx": ..., "fy": ..., ... }
}
```

---

## Summary

| Test Category | Result |
|---|---|
| Unit tests (55) | PASS |
| MotionBert pipeline | PASS |
| MediaPipe pipeline | PASS |
| Output files present | PASS |
| All 9 metrics present/valid | PASS |
| SI-MPJPE <= MPJPE | PASS |
| Optimizer reasonableness | PASS (with notes) |
| Overlay videos valid | PASS |
| Visual inspection | PASS |
| Optimizer pure-function design | PASS |
| evaluate.py VW frame-boundary | PASS |
| camera.py completeness | PASS |
| Original dirs untouched | PASS |
| Output structure | PASS |
| results.json format | MINOR DEVIATION |
| trajectories.json format | PASS |

### Issues Logged (see TODO.md)
1. **results.json format deviation** -- flat `metrics` dict vs planned nested structure
2. **MotionBert VW-SI-MPJPE regression** -- 2.1% increase on single example, needs validation on full run
3. **MotionBert high absolute MPJPE** -- 51.45 cm likely from depth/scale estimation, needs investigation

### Notes to Developer
- The codebase is clean and well-structured. All core functionality works end-to-end.
- The optimizer's trade-off of marginal position accuracy for significantly improved velocity smoothness is expected behavior and aligns with the spec's emphasis on VW-SI-MPJVE.
- The MotionBert high MPJPE is dominated by the depth estimation step. The VW-SI-MPJPE (19 cm) is much more reasonable and the relevant metric per the spec.
- SI-MPJVE exceeding MPJVE is mathematically correct (scale optimized for positions, not velocities) and not a bug.
- The full 25-example runs (Phase 7) should be executed to validate aggregate metric trends, particularly whether the MotionBert VW-SI-MPJPE regression persists at scale.
