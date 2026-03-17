# Developer Report P0-00: Remove Head Joint + Simplify Bone Scaling

## What Was Implemented

Phase 0 of the round-5 spec: removed the Head joint (old index 10) from the 17-joint H36M skeleton, reducing to 16 joints, and simplified `motionbert_to_camera_space` by removing the torso-height heuristic and the deprecated `pixel_aligned_to_camera_space` function.

### Summary of Changes

1. **Skeleton reduced from 17 to 16 joints.** Head joint removed; all indices >= 10 shifted down by 1. The kinematic chain Thorax -> Neck still exists, but Neck is now a leaf node.

2. **MotionBERT still receives 17-joint input** (`mpii_to_h36m` unchanged), but its output is stripped to 16 joints via a new `h36m_17_to_16()` utility in `skeleton.py`. This preserves compatibility with the pretrained model.

3. **Ground truth** (`coco19_to_h36m`) now directly produces 16 joints (no Head extrapolation).

4. **Torso-height heuristic removed** from `motionbert_to_camera_space`. The blending line `tz = 0.4 * tz + 0.6 * tz_heuristic` and the entire torso-height fallback block were removed. Now uses only pairwise separation for depth estimation with a simple `tz = 3.0` fallback when fewer than 4 valid joints.

5. **Deprecated function removed**: `pixel_aligned_to_camera_space` deleted from `detect.py`.

## Files Changed

1. **`motionbert-pose/skeleton.py`** -- NUM_JOINTS=16, removed Head from JOINT_NAMES, PARENTS, DEFAULT_BONE_LENGTHS, REST_DIRECTIONS, BODY_GROUPS, EVAL_JOINTS. Updated `coco19_to_h36m` to produce (16, 3). Added `h36m_17_to_16()` helper. `mpii_to_h36m` unchanged (still produces 17 for MotionBERT).

2. **`motionbert-pose/config.py`** -- ROTATION_PENALTY_PER_JOINT reduced from 17 to 16 entries (Head entry removed, comments renumbered).

3. **`motionbert-pose/scoring.py`** -- H36M_TO_MPII_HEATMAP reduced from 17 to 16 entries (Head entry removed).

4. **`motionbert-pose/detect.py`** -- Updated `_RELIABLE_BONES_FOR_SCALE` indices (11-16 -> 10-15). Updated `_DEPTH_RELIABLE_JOINTS` (removed old 10/Head, shifted). Changed `range(1, 17)` to `range(1, NUM_JOINTS)` in `_enforce_bone_lengths`, `_enforce_bone_lengths_with_2d`, `_reconstruct_from_2d`. Removed torso-height heuristic blending and fallback in `motionbert_to_camera_space`. Removed deprecated `pixel_aligned_to_camera_space`. Added `h36m_17_to_16` calls in `run_motionbert` (output) and `detect_poses` (2D keypoints + visibility). Updated docstrings.

5. **`motionbert-pose/evaluate.py`** -- No code changes needed; uses imports from skeleton which auto-update.

6. **`motionbert-pose/graphs.py`** -- Updated `key_joints` in `generate_summary` from `[0, 8, 9, 10, 13, 3]` to `[0, 8, 9, 12, 15, 3]` (replaced Head with RWrist).

7. **`motionbert-pose/main.py`** -- Changed `for j in range(17)` to `for j in range(NUM_JOINTS)` in FK target replacement loop. Added `NUM_JOINTS` import.

8. **`motionbert-pose/models.py`** -- Updated comment from `# always 17` to `# always 16`.

9. **`motionbert-pose/panoptic.py`** -- Updated docstring in `load_ground_truth_sequence` to say "16-joint".

10. **`motionbert-pose/overlay_video.py`** -- No changes needed (uses BONES and NUM_JOINTS from imports).

11. **`motionbert-pose/optimize.py`** -- No changes needed (uses NUM_JOINTS from imports).

12. **`motionbert-pose/fk.py`** -- No changes needed (uses NUM_JOINTS, PARENTS, REST_DIRECTIONS from imports).

## Commands Run

### 1. Skeleton assertion test
```bash
cd motionbert-pose && uv run python -c "from skeleton import ...; assert NUM_JOINTS == 16; ..."
```
**Result:** All assertions passed. Parent chain verified correct manually:
- Hip(0) -> root
- RHip(1) -> Hip(0), RKnee(2) -> RHip(1), RAnkle(3) -> RKnee(2)
- LHip(4) -> Hip(0), LKnee(5) -> LHip(4), LAnkle(6) -> LKnee(5)
- Spine(7) -> Hip(0), Thorax(8) -> Spine(7), Neck(9) -> Thorax(8)
- LShoulder(10) -> Thorax(8), LElbow(11) -> LShoulder(10), LWrist(12) -> LElbow(11)
- RShoulder(13) -> Thorax(8), RElbow(14) -> RShoulder(13), RWrist(15) -> RElbow(14)

### 2. FK roundtrip + module import test
```bash
cd motionbert-pose && uv run python -c "...FK roundtrip, h36m_17_to_16, coco19_to_h36m, scoring, config checks..."
```
**Result:** FK roundtrip error 0.000224 cm (well under threshold). All module checks passed.

### 3. Full pipeline smoke test (example 0: 171204_pose1_sample, 100 frames)
```bash
cd motionbert-pose && uv run python -c "from main import process_example; process_example(*cfg.EXAMPLES[0], 'training_runs/test_p0')"
```
**Result:** Pipeline completed end-to-end successfully. No crashes.
- Det MPJPE: 30.98 cm
- Opt MPJPE: 30.82 cm (improvement: +0.16 cm)
- Det P-MPJPE: 28.57 cm, Opt P-MPJPE: 28.23 cm
- Det MPJVE: 0.94 cm/f, Opt MPJVE: 1.53 cm/f
- Overlay video, graphs, prediction JSON all saved successfully.

## Decisions Made

1. **`flip_data` indices unchanged.** The architect noted `flip_data` operates on 17-joint MotionBERT input/output (before `h36m_17_to_16` is applied), so the left/right joint indices `[4,5,6,11,12,13]` / `[1,2,3,14,15,16]` remain in 17-joint format. This is correct because `flip_data` is only called inside `run_motionbert()` on 17-joint data before stripping.

2. **Key joints in summary.** Replaced Head (old index 10) with RWrist (new index 15) to maintain 6 representative joints from different body parts: Hip, Thorax, Neck, LWrist, RWrist, RAnkle.

3. **Simple fallback for `motionbert_to_camera_space`.** When `n_valid_tz < 4`, use `tz = 3.0` and compute `tx, ty` from root 2D keypoint. This follows the architect's plan exactly.

## Concerns

1. **MPJPE is relatively high (~31 cm)** for this example, but this is consistent with previous results before the Head removal. The Head removal itself should not significantly change MPJPE since Head was already excluded from EVAL_JOINTS.

2. **Torso-height heuristic removal** may cause slightly less robust depth estimation in edge cases where very few upper-body joints are detected. The simple `tz = 3.0` fallback is acceptable but could be wrong for unusual camera setups (very close or very far subjects).

3. **No hardcoded 17s remain** in the modified files. All joint-count-dependent code now uses `NUM_JOINTS` from skeleton or the explicit 16-joint arrays.

## Deviations from Plan

None. All steps in the architect's plan were followed precisely.
