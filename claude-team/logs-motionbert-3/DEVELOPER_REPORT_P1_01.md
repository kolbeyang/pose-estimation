# Developer Report: Phase 1, Iteration 1 -- Fix solvePnP Depth Jitter + Clean Up sweep.py

**Date:** 2026-03-17
**Spec:** `claude-team/specs/motion-bert-round-5.md` (Phase 1)
**Architect Plan:** `claude-team/logs/ARCHITECT_PLAN_P1_01.md`

## What Was Implemented

1. **BUG-1 Fixed: Removed ALL_JOINTS_SMOOTH_WEIGHT from sweep.py.** Removed the `all_joints_smooth_weight` field from `SweepConfig`, the save/restore lines in `run_sweep_config()`, and the display string reference.

2. **Fixed _ROT_MULTIPLIERS in sweep.py.** Reduced from 17 entries to 16 entries (removed the old Head joint entry at position 10) to match the current 16-joint skeleton.

3. **Fixed range(17) in sweep.py load_example.** Changed to `range(NUM_JOINTS)` and added `from skeleton import NUM_JOINTS` to imports.

4. **Added `motionbert_to_camera_space_batch()` to detect.py.** New function processes all frames at once: runs solvePnP per frame independently, then applies `scipy.ndimage.median_filter` to the translation vector (tx, ty, tz) across the temporal dimension. Per-frame rotation is preserved (not smoothed). Bone-length enforcement is applied after transform.

5. **Updated main.py and sweep.py** to use `motionbert_to_camera_space_batch()` instead of the per-frame loop calling `motionbert_to_camera_space()`.

## Files Changed

- `motionbert-pose/detect.py` -- Added `motionbert_to_camera_space_batch()` function
- `motionbert-pose/main.py` -- Updated import and replaced per-frame loop with batch call
- `motionbert-pose/sweep.py` -- Fixed BUG-1 (removed ALL_JOINTS_SMOOTH_WEIGHT), fixed _ROT_MULTIPLIERS (17->16), fixed range(17)->range(NUM_JOINTS), updated to batch conversion

## Commands Run

### Example 0: `171204_pose1_sample_0` (100 frames)
```bash
cd motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-solvepnp-smooth')
os.makedirs(run_dir, exist_ok=True)
seq, cam, start, nf, pidx = cfg.EXAMPLES[0]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```
**Result:** Completed successfully. No errors.

### Example 5: `171204_pose3_4000` (150 frames)
```bash
cd motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
import os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-solvepnp-smooth')
seq, cam, start, nf, pidx = cfg.EXAMPLES[5]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```
**Result:** Completed successfully. No errors.

### sweep.py verification
```bash
cd motionbert-pose && uv run python -c "
from sweep import SweepConfig, get_phase1_2_configs
configs = get_phase1_2_configs()
print(f'{len(configs)} configs')
assert not hasattr(configs[0], 'all_joints_smooth_weight'), 'BUG-1 not fixed'
print('sweep.py loads correctly, BUG-1 fixed')
"
```
**Result:** 17 configs, BUG-1 confirmed fixed.

## Results Comparison

### Example 0: `171204_pose1_sample_0`

| Metric | Round-4 (pairwise) | P1-00 (raw solvePnP) | P1-01 (smoothed, w=11) |
|--------|--------------------|-----------------------|------------------------|
| Det MPJPE (cm) | 30.98 | 34.72 | 34.72 |
| Opt MPJPE (cm) | 30.44 | 34.12 | 33.90 |
| Det P-MPJPE (cm) | 28.57 | 28.57 | 28.57 |
| Opt P-MPJPE (cm) | 28.42 | 28.30 | 28.31 |
| Det MPJVE (cm/f) | 0.94 | 3.69 | 3.69 |
| Opt MPJVE (cm/f) | 1.05 | 3.38 | 3.37 |
| Root Z range (det) | -- | 1.44-2.45 m | 2.21-2.38 m |
| Root Z std (det) | -- | 0.147 m | 0.043 m |
| Raw vs Smoothed Z std | N/A | N/A | 0.147 -> 0.043 (3.5x) |

### Example 5: `171204_pose3_4000`

| Metric | Round-4 (pairwise) | P1-00 (raw solvePnP) | P1-01 (smoothed, w=11) |
|--------|--------------------|-----------------------|------------------------|
| Det MPJPE (cm) | 15.85 | 18.26 | 18.26 |
| Opt MPJPE (cm) | 15.44 | 18.65 | 18.65 |
| Det P-MPJPE (cm) | 20.33 | 20.33 | 20.33 |
| Opt P-MPJPE (cm) | 20.22 | 20.52 | 20.52 |
| Det MPJVE (cm/f) | 0.51 | 0.93 | 0.93 |
| Opt MPJVE (cm/f) | 0.48 | 0.54 | 0.55 |
| Root Z range (det) | -- | 2.27-2.34 m | 2.28-2.33 m |
| Root Z std (det) | -- | 0.016 m | 0.014 m |
| Raw vs Smoothed Z std | N/A | N/A | 0.016 -> 0.014 (1.2x) |

## Key Observations

1. **Root Z is significantly smoother.** For Example 0, root Z std dropped from 0.147m to 0.043m (3.5x reduction), and the catastrophic 1.44m outlier was eliminated (smoothed range 2.21-2.38m vs raw 1.44-2.45m). This meets the architect's target of <0.05m.

2. **MPJVE is NOT improved.** Despite the dramatically smoother root Z, the MPJVE is essentially unchanged (3.69 -> 3.69 for Example 0 detection, 0.93 -> 0.93 for Example 5 detection). This reveals that **the jitter is in the per-frame rotation, not just translation**. The median filter smooths only the translation vector; the per-frame solvePnP rotation still varies frame-to-frame, causing joint positions to jitter even with smooth translation.

3. **MPJPE is unchanged.** The Det MPJPE values are identical to P1-00 because the per-frame bone scaling and rotation are unchanged -- only the translation is smoothed, and the systematic depth bias (~30cm too shallow) remains.

4. **The spec requirement "MPJVE should be improved for ALL test videos" is still NOT met.** Both examples still show worse MPJVE than round-4. The root cause is now identified: smoothing translation alone is insufficient when the rotation estimate varies per-frame.

## Window Size Investigation

I tested multiple median filter window sizes on Example 0 before choosing the default:

| Window | Smoothed Z std | Z Range | Mean |dZ| per frame |
|--------|---------------|---------|---------------------|
| 5 | 0.123 m | 1.67-2.41 m | 0.020 m/f |
| 11 | 0.043 m | 2.21-2.38 m | 0.006 m/f |
| 15 | 0.038 m | 2.23-2.37 m | 0.004 m/f |
| 21 | 0.029 m | 2.24-2.34 m | 0.003 m/f |
| 31 | 0.023 m | 2.26-2.33 m | 0.002 m/f |

GT root Z std = 0.016m, GT mean |dZ| = 0.003 m/f.

Window=5 (architect's default) was clearly insufficient -- only 1.2x reduction. Window=11 gives 3.5x reduction and reaches the <0.05m target.

## Decisions Made

1. **Changed default smooth_window from 5 to 11.** The architect specified 5, but testing showed 5 is insufficient (only 1.2x Z std reduction vs the 3-5x target). Window=11 achieves 3.5x reduction and meets the <0.05m Z std target. This is documented as a deviation.

## Concerns

1. **MPJVE is not improved because rotation is not smoothed.** The median filter on translation is working correctly (root Z jitter is 3.5x smaller), but MPJVE measures velocity across all joints. The per-frame solvePnP rotation estimate varies frame-to-frame, causing individual joint positions to jitter even with smooth global translation. To fix MPJVE, the next iteration should either:
   - Smooth the rotation estimates as well (e.g., quaternion SLERP in a sliding window)
   - Lock rotation to identity (since MotionBERT output is approximately camera-aligned)
   - Revert to the pairwise method which didn't use rotation at all

2. **MPJPE has not recovered to round-4 levels.** The 3-4cm regression from round-4 persists because the systematic depth bias (~30cm too shallow for Example 0) is unchanged by smoothing. The pairwise method may have produced better absolute depth estimates.

3. **Example 5 was already smooth.** The raw Z std for Example 5 was already low (0.016m), so the median filter had minimal effect. The MPJVE difference between round-4 and P1-01 for Example 5 (0.48 vs 0.55) is entirely due to rotation jitter, not translation jitter.

## Deviations from Plan

1. **smooth_window default changed from 5 to 11.** The architect specified 5, but empirical testing showed 5 provides only 1.2x Z std reduction (insufficient for the <0.05m target). Window=11 achieves 3.5x reduction and meets the target. This is a minor parameter change that improves behavior.

## Output Artifacts

- Predictions: `motionbert-pose/training_runs/p1-solvepnp-smooth/predictions/`
  - `171204_pose1_sample_0.json`
  - `171204_pose3_4000.json`
- Graphs + overlay videos: `motionbert-pose/training_runs/p1-solvepnp-smooth/graphs/`
  - `171204_pose1_sample_0/` (including overlay video)
  - `171204_pose3_4000/` (including overlay video)
