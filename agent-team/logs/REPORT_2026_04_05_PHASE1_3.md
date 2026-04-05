# Phase 1-3 Implementation Report

**Date:** 2026-04-05
**Branch:** refactor-2026-04-04
**Commit:** 207e34b

---

## Phase 1: Per-joint Visible Frame Counts

### Changes
- **`main.py`**: Added `per_joint_visible_frames` computation after `compute_visibility_weights()`. This is a list of 15 ints (one per eval joint) indicating how many frames that joint's GT projection fell within the image bounds.
- **`main.py`**: Added field to `_save_results()` under the `per_joint` section.

### Verification
Results.json now contains e.g.:
```json
"per_joint_visible_frames": [34, 34, 34, 0, 34, 34, 0, 34, 34, 34, 34, 34, 34, 34, 34]
```
Both ankles (RAnkle, LAnkle) show 0 visible frames for this example, confirming the Phase 0 finding that ankle errors are excluded from VW-SI-MPJPE.

---

## Phase 2: Fix MotionBERT Knee/Lower-Body Issue

### Root Cause
MotionBERT's 3D lift produces wildly wrong lower body: knee bones 1.3x, ankle bones 2.7-2.9x too long. The median bone length init preserved these bad values, and bone_length_lr=0.0001 was too slow to correct in 10 steps.

### Changes

1. **Bone length clamping** (`optimize/__init__.py`):
   - After computing median bone lengths across frames, clamp each bone to [0.5x, 1.5x] of `DEFAULT_BONE_LENGTHS`.
   - This prevents pathological initialization. MotionBERT's ankle bones at 1.08m get clamped to 0.60m (1.5x of default 0.40m), much closer to GT 0.40m.
   - MediaPipe bones (0.73-0.80x of GT) are NOT clamped because they fall within the [0.5x, 1.5x] range.

2. **Increased bone_length_lr** (`config.py`):
   - Default bumped from 0.0001 to 0.001 (10x faster bone length learning).

3. **Per-pipeline config** (`configs/both-local-single.json`):
   - MotionBERT gets 50 steps, lr=0.001, bone_length_lr=0.005, blur annealing 16->4.

### Results
- **Baseline**: Raw 23.43 cm -> Opt 18.86 cm (19.5% improvement)
- **After fix**: Raw 23.43 cm -> Opt 16.83 cm (28.1% improvement)
- Improvement over baseline optimized: 2.03 cm (10.8% better)

---

## Phase 3: Fix MediaPipe Optimization Regression

### Root Cause Analysis
The regression had TWO components:
1. **FK shared bone length error (~0.4 cm)**: Converting per-frame positions to FK params with shared median bone lengths introduces error. With 0 optimization steps, MediaPipe already goes from 15.66 to ~16.07.
2. **Heatmap-driven drift**: The SH heatmaps disagree with GT at wrist/ankle positions. The optimizer faithfully moves joints toward heatmap peaks, increasing 3D error. Wrist errors increased 11-12%, ankle errors 4-8%.

### Critical Decision: Anchor to Raw Positions
The plan suggested anchoring to FK-init positions. I tested both:
- **Anchor to FK-init**: 15.66 -> 15.96 (still regresses because FK-init already has bone length error)
- **Anchor to raw positions**: 15.66 -> 15.40 at anchor_weight=1000 (improvement!)

Anchoring to raw positions is the correct choice because it penalizes deviation from the detector's output, not the FK reconstruction. This lets the optimizer improve motion smoothness and bone proportions while preventing heatmap-driven joint drift.

### Changes

1. **Anchor penalty** (`scoring.py`):
   - Added `anchor_positions` and `anchor_weight` parameters to `compute_total_score_batch()`.
   - Anchor penalty = `anchor_weight * sum((positions - anchor_positions)^2)`.
   - Default anchor_weight=0.0 (no effect unless configured).

2. **Anchor position computation** (`optimize/__init__.py`):
   - When anchor_weight > 0, uses the raw input positions (not FK reconstruction) as anchor targets.

3. **Blur sigma annealing** (`optimize/__init__.py`):
   - New config params: `heatmap_blur_sigma_start` and `heatmap_blur_sigma_end`.
   - When both are set, sigma linearly interpolates from start to end over optimization steps.
   - Backward compatible: if only `heatmap_blur_sigma` is set, uses fixed sigma as before.

4. **Per-pipeline optimization configs** (`config.py`):
   - `PipelineConfig` now has optional `optimization` dict for overrides.
   - `RunConfig.optimization_for_pipeline(name)` merges pipeline-specific overrides on top of shared config.
   - Full backward compatibility: if no pipeline overrides, uses shared config.

5. **Config tuning** (`configs/both-local-single.json`):
   - MediaPipe: 25 steps, lr=0.0005, sigma=16 (fixed), anchor_weight=1000

### Rotation Penalty Multiplier Decision
I initially bumped wrist/ankle multipliers from 0.1/0.2 to 0.5/0.5, but this made MediaPipe WORSE (16.06 vs 15.86 with old values). The higher penalties over-constrained the skeleton even at the FK init stage. **Reverted to original defaults.** The anchor penalty is a better mechanism for preventing drift.

### Results
- **Baseline**: Raw 15.66 cm -> Opt 15.86 cm (1.3% REGRESSION)
- **After fix**: Raw 15.66 cm -> Opt 15.40 cm (1.7% improvement)
- Swing from regression to improvement: 0.46 cm

### Anchor Weight Sweep (MediaPipe, 10 steps, sigma=16, lr=0.0005)
| anchor_weight | Opt VW-SI-MPJPE |
|---------------|----------------|
| 0 (baseline)  | 16.06 cm       |
| 100           | 15.70 cm       |
| 500           | 15.54 cm       |
| 1000          | 15.52 cm       |
| 2000          | 15.53 cm       |

With 25 steps + anchor_weight=1000: **15.40 cm** (best).

---

## Summary of All Changes

| File | Changes |
|------|---------|
| `config.py` | Added `heatmap_blur_sigma_start`, `heatmap_blur_sigma_end`, `anchor_weight` to OptimizationConfig. Added `optimization` dict to PipelineConfig. Added `optimization_for_pipeline()` to RunConfig. Bumped `bone_length_lr` default 0.0001 -> 0.001. |
| `scoring.py` | Added `anchor_positions` and `anchor_weight` params to `compute_total_score_batch()`. |
| `optimize/__init__.py` | Added bone length clamping to [0.5x, 1.5x] of defaults. Added blur sigma annealing. Added anchor position computation and passing to scoring. |
| `main.py` | Added `per_joint_visible_frames` to metrics/results. Updated optimize calls to use `optimization_for_pipeline()`. |
| `configs/both-local-single.json` | Added per-pipeline optimization overrides. |
| `configs/motionbert-single.json` | Reverted sigma=32 to use blur annealing 16->4. |

## Final Metrics (Single Example: 171204_pose1_sample)

| Pipeline | Raw VW-SI-MPJPE | Old Opt | New Opt | Improvement |
|----------|-----------------|---------|---------|-------------|
| MotionBERT | 23.43 cm | 18.86 cm | **16.83 cm** | +2.03 cm vs old opt |
| MediaPipe | 15.66 cm | 15.86 cm (WORSE) | **15.40 cm** | Fixed regression, now improves |

## Open Issues
- MediaPipe improvement is only 1.7% on this single example; need >2% average on 25 examples (Phase 4/5)
- Blur annealing re-blurs all heatmaps every step via scipy -- slow for many steps
- 25-example configs not yet updated with new per-pipeline settings
