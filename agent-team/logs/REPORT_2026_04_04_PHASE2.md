# Phase 2 Report: Evaluation Metrics in Camera Coordinates

## What Was Changed

Removed all `root_relative()` calls from evaluation paths. Previously, both `evaluate()` in `evaluate.py` and the per-joint metric code in `main.py`, `run_motionbert/__init__.py`, and `run_mediapipe/__init__.py` subtracted the pelvis position before computing metrics. This made the pelvis error always 0 and hid translation errors.

### Files Modified
- `pose-optimizer/evaluate.py` - `evaluate()` no longer calls `root_relative()`, uses camera-space positions directly
- `pose-optimizer/main.py` - Per-joint/per-frame metrics use camera coordinates
- `pose-optimizer/run_motionbert/__init__.py` - Same change
- `pose-optimizer/run_mediapipe/__init__.py` - Same change

## Before/After Metric Comparison

Test: `motionbert-single.json` (171204_pose1_sample, 34 frames)

### Optimized Metrics
| Metric | Before (root-relative) | After (camera coords) | Change |
|--------|----------------------|---------------------|--------|
| MPJPE | 40.67 cm | 59.62 cm | +47% |
| P-MPJPE | 42.47 cm | 42.47 cm | 0% |
| SI-MPJPE | 40.57 cm | 48.22 cm | +19% |
| VW-MPJPE | 13.59 cm | 33.14 cm | +144% |
| VW-SI-MPJPE | 13.00 cm | 18.86 cm | +45% |
| MPJVE | 2.29 cm/f | 4.83 cm/f | +111% |
| SI-MPJVE | 2.26 cm/f | 5.27 cm/f | +133% |

### Raw Detector Metrics
| Metric | Before (root-relative) | After (camera coords) | Change |
|--------|----------------------|---------------------|--------|
| MPJPE | 43.52 cm | 62.35 cm | +43% |
| VW-SI-MPJPE | 17.20 cm | 23.43 cm | +36% |

### Key Observations
- P-MPJPE is unchanged (Procrustes alignment already handles translation)
- All other metrics are higher, reflecting the true camera-space error including translation
- Pelvis per-joint error: 0.00 cm (before) -> 33.3 cm (after), confirming the fix
- The relative improvement from optimization is preserved (optimizer still helps)

## Issues
None. The change was straightforward.
