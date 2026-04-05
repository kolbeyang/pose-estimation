# Report 00: Phases 1-3 Implementation

## Summary

Implemented the foundation, evaluation, and optimizer core for the unified `pose-optimizer/` package. All 55 tests pass across all three phases.

## Files Created

```
pose-optimizer/
    pyproject.toml              # uv project config with all dependencies
    skeleton.py                 # Unified 16-joint skeleton + all mappings
    camera.py                   # Camera with intrinsics, extrinsics, is_in_frame
    cmu_data.py                 # CMU Panoptic data loading + download helpers
    config.py                   # Pydantic config models with JSON serialization
    fk.py                       # Differentiable FK (single + batch) + inverse FK
    evaluate.py                 # All 9 evaluation metrics
    scoring.py                  # Unified scoring with grid_sample + synthetic heatmaps
    optimize/__init__.py        # Pure-function optimizer
    experiment/test_phase1.py   # 28 tests
    experiment/test_phase2.py   # 22 tests
    experiment/test_phase3.py   # 5 tests
    data/.gitignore
    output/.gitignore
```

## Critical Decisions

### Skeleton Joint 9: "Neck" (MotionBert convention)
Joint index 9 is named "Neck" following the MotionBert convention. In practice:
- COCO19 GT maps Nose (index 1) here
- MPII/SH maps MPII Head (index 9) here
- MediaPipe maps landmark 0 (Nose) here

These are all reasonable approximations of the upper head/neck region. The key requirement is consistency within each pipeline (GT mapping matches detector mapping), which holds.

### Eval Joints: 12 joints excluding midpoints
Joints 0 (Hip), 7 (Spine), 8 (Thorax), 9 (Neck) are excluded from evaluation because they are computed midpoints that differ between detector, GT, and FK. This matches motionbert-pose behavior.

### Visibility Weighting: Frame-boundary based
VW metrics use binary visibility from projecting GT 3D to 2D and checking frame bounds. This replaces the previous SH-confidence-based weighting. For velocity metrics, both the current and previous frame must be visible (minimum of the two weights).

### Camera: Unified with extrinsics
The Camera class stores both intrinsics (K) and optional extrinsics (R, t). Methods:
- `world_to_camera()` / `camera_to_world()` for extrinsic transforms
- `camera_to_image()` / `camera_to_image_torch()` for intrinsic projection
- `is_in_frame()` for visibility checks
- Legacy aliases `world_to_image()` preserved for compat

### Scoring: Unified grid_sample pipeline
Both heatmap modes (real SH and synthetic Gaussian) use the same `grid_sample`-based scoring. The `use_mpii_mapping` flag controls whether MPII joint remapping is applied (MotionBert: True, scores 14 joints) or direct H36M indexing is used (MediaPipe: False, scores all 16).

### Synthetic Heatmaps: Pixel-to-heatmap affine
For MediaPipe mode, a (2, 3) affine maps [0, hm_size-1] to [0, image_size-1]. Gaussian sigma is converted from pixel space to heatmap space. The crop_to_hm_ratio is 1.0 for synthetic (vs 4.0 for SH's 256->64 reduction).

### Optimizer: 3 param groups
Following MotionBert's batched optimizer:
- root_pos at 1.5x learning rate
- root_rot + local_rots at 1x
- bone_lengths at separate (lower) LR

Shared bone lengths via per-frame median. Clamped to min=0.01 after each step.

## Test Results

**Phase 1 (28 tests):** All pass. Covers skeleton mappings, camera projection/roundtrip/is_in_frame, config serialization, and real CMU data loading.

**Phase 2 (22 tests):** All pass. FK roundtrip < 0.01 cm on real data. All 9 metrics produce finite positive values. SI-MPJPE <= MPJPE. P-MPJPE <= MPJPE. Scoring produces nonzero gradients.

**Phase 3 (5 tests):** All pass. Loss decreases during optimization. Bone lengths stay positive. On real CMU data with 2cm noise, optimizer does not make MPJPE worse.

## Issues and Notes

- The `camera_to_image` numpy method handles (..., 3) -> (..., 2) shapes generically, which was needed for the evaluate module to project (F, K, 3) directly.
- The scoring module has extensive comments about the affine/crop/heatmap coordinate mapping. The SH pipeline uses a 256->64 crop (ratio=4), while synthetic uses ratio=1. This is handled by the `use_mpii_mapping` flag which doubles as a crop ratio selector.
- The 25 examples list in `cmu_data.py` includes multiple windows from the same sequences. Some sequences (like `160226_haggling1`) are excluded because they are multi-person. The `count_persons_in_frame()` helper is available for runtime validation if needed.
- Download helpers exist but are untested (would need network access). The data symlinks to `../data/panoptic-toolbox/` which is shared with existing implementations.
