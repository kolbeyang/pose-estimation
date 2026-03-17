# Architect Plan P0-00: Omit Head Joint + Simplify Bone Scaling

## Goal Summary

Phase 0 has two tasks: (1) Remove the Head joint (index 10) entirely from the skeleton, evaluation, visualization, and all related logic -- the Head mapping from COCO19/MPII was unreliable (extrapolated from neck-to-nose direction) and has been excluded from eval since the beginning. Now we formalize this by dropping it completely so it does not appear in FK, scoring, overlay videos, or graphs. (2) Simplify `motionbert_to_camera_space` in `detect.py` by removing the torso-height heuristic fallback and the blended `tz` computation -- keep only the pairwise separation method for depth estimation.

## Files to Modify

1. **`motionbert-pose/skeleton.py`** -- Reduce from 17 to 16 joints by removing Head (index 10). Update all arrays, mappings, and constants.
2. **`motionbert-pose/detect.py`** -- Remove torso-height heuristic from `motionbert_to_camera_space`. Remove the deprecated `pixel_aligned_to_camera_space` function entirely. Update `_DEPTH_RELIABLE_JOINTS` to exclude joint 10. Update `flip_data` L/R joint indices.
3. **`motionbert-pose/fk.py`** -- Update `positions_to_fk_params` to use 16 joints. Update `forward_kinematics` (uses `NUM_JOINTS` from skeleton, should adapt automatically but verify).
4. **`motionbert-pose/scoring.py`** -- Update `H36M_TO_MPII_HEATMAP` to have 16 entries (remove Head entry at index 10).
5. **`motionbert-pose/evaluate.py`** -- Update `EVAL_JOINTS`, `EVAL_JOINTS_NO_ANKLES` indices for the new 16-joint indexing.
6. **`motionbert-pose/overlay_video.py`** -- Uses `BONES` and `NUM_JOINTS` from skeleton; should adapt automatically. Verify no hardcoded 17s.
7. **`motionbert-pose/optimize.py`** -- Uses `NUM_JOINTS` from skeleton; should adapt automatically. Verify no hardcoded 17s.
8. **`motionbert-pose/config.py`** -- `ROTATION_PENALTY_PER_JOINT` has 17 entries; reduce to 16 (remove Head entry at index 10).
9. **`motionbert-pose/graphs.py`** -- Uses constants from skeleton; should adapt automatically. Verify `key_joints` in `generate_summary` (currently includes joint 10 = Head).
10. **`motionbert-pose/main.py`** -- Verify no hardcoded 17s or references to Head.
11. **`motionbert-pose/models.py`** -- Update comment on line 31 (`num_joints: int  # always 17` -> `# always 16`).
12. **`motionbert-pose/panoptic.py`** -- Update docstrings that say "17-joint" to "16-joint" (lines 88, 91).

## Files to Create

None.

## Step-by-Step Instructions

### Step 1: Update `skeleton.py` -- Remove Head Joint

The Head joint is currently at index 10. Removing it shifts all joints above 10 down by one. The new 16-joint skeleton becomes:

```
Old index -> New index:
 0: Hip       ->  0: Hip
 1: RHip      ->  1: RHip
 2: RKnee     ->  2: RKnee
 3: RAnkle    ->  3: RAnkle
 4: LHip      ->  4: LHip
 5: LKnee     ->  5: LKnee
 6: LAnkle    ->  6: LAnkle
 7: Spine     ->  7: Spine
 8: Thorax    ->  8: Thorax
 9: Neck      ->  9: Neck
10: Head      ->  REMOVED
11: LShoulder -> 10: LShoulder
12: LElbow    -> 11: LElbow
13: LWrist    -> 12: LWrist
14: RShoulder -> 13: RShoulder
15: RElbow    -> 14: RElbow
16: RWrist    -> 15: RWrist
```

Specific changes in `skeleton.py`:

1. Set `NUM_JOINTS = 16`
2. Remove `"Head"` from `JOINT_NAMES` (remove index 10 entry).
3. Update `EVAL_JOINTS` to use new indices. Old: `[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]`. New: `[1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]`.
4. Update `PARENTS` array. Old: `[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]`. Remove entry at index 10 (Head's parent was 9=Neck), then shift all parent references above 10 down by 1. New 16-element array: `[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]`.
   - Key: LShoulder(10) parent was Thorax(old 8) -> Thorax(new 8). Correct.
   - RShoulder(13) parent was Thorax(old 8) -> Thorax(new 8). Correct.
   - Neck(9) parent was Thorax(old 8) -> Thorax(new 8). Correct.
   - Head was child of Neck -- now removed entirely.
5. Remove entry at index 10 from `DEFAULT_BONE_LENGTHS` (the `0.12` for Neck -> Head). New array has 16 entries.
6. Remove entry at index 10 from `REST_DIRECTIONS` (the `[0, -1, 0]` for Neck -> Head). New array has 16 entries.
7. Update `BODY_GROUPS`:
   - `"spine"`: old `[0, 7, 8, 9, 10]` -> new `[0, 7, 8, 9]` (remove 10, which was Head; Neck is still 9).
   - `"left_arm"`: old `[11, 12, 13]` -> new `[10, 11, 12]`.
   - `"right_arm"`: old `[14, 15, 16]` -> new `[13, 14, 15]`.
   - `"left_leg"` and `"right_leg"` unchanged (indices < 10).
8. Update `mpii_to_h36m()`: currently produces (17, D). Must produce (16, D). Remove the `h36m[10] = keypoints_mpii[9]` line. Shift all indices 11-16 down to 10-15:
   - `h36m[10] = keypoints_mpii[13]`  (LShoulder, was h36m[11])
   - `h36m[11] = keypoints_mpii[14]`  (LElbow, was h36m[12])
   - `h36m[12] = keypoints_mpii[15]`  (LWrist, was h36m[13])
   - `h36m[13] = keypoints_mpii[12]`  (RShoulder, was h36m[14])
   - `h36m[14] = keypoints_mpii[11]`  (RElbow, was h36m[15])
   - `h36m[15] = keypoints_mpii[10]`  (RWrist, was h36m[16])
   - Change output array size from `(17, ndim)` to `(16, ndim)`.
   - Update Thorax mapping: `h36m[8] = keypoints_mpii[8]` stays. Note: old MPII mapping had Thorax=MPII Neck and Neck=MPII Head. With Head removed, Thorax(8) and Neck(9) still both exist and their mappings are unchanged.
9. Update `coco19_to_h36m()`: currently produces (17, 3). Must produce (16, 3). Remove the `h36m[10]` line (Head extrapolation). Shift indices 11-16 down to 10-15:
   - `h36m[10] = joints19[3]`  (LShoulder)
   - `h36m[11] = joints19[4]`  (LElbow)
   - `h36m[12] = joints19[5]`  (LWrist)
   - `h36m[13] = joints19[9]`  (RShoulder)
   - `h36m[14] = joints19[10]` (RElbow)
   - `h36m[15] = joints19[11]` (RWrist)
   - Change output array from `(NUM_JOINTS, 3)` (which will be 16) and update docstring.

### Step 2: Update `config.py` -- Remove Head from Rotation Penalty

1. `ROTATION_PENALTY_PER_JOINT`: Remove entry at index 10 (`ROTATION_PENALTY_SCALAR * 0.5,  # 10: Head`). Renumber comments for indices 10-15 (LShoulder through RWrist). Array now has 16 entries.

### Step 3: Update `scoring.py` -- Remove Head from Heatmap Mapping

1. `H36M_TO_MPII_HEATMAP`: Remove entry at index 10 (`9,  # 10: Head`). Renumber comments. New list has 16 entries:
   ```
   None,  # 0: Hip
   2,     # 1: RHip
   1,     # 2: RKnee
   0,     # 3: RAnkle
   3,     # 4: LHip
   4,     # 5: LKnee
   5,     # 6: LAnkle
   None,  # 7: Spine
   7,     # 8: Thorax
   8,     # 9: Neck
   13,    # 10: LShoulder
   14,    # 11: LElbow
   15,    # 12: LWrist
   12,    # 13: RShoulder
   11,    # 14: RElbow
   10,    # 15: RWrist
   ```

### Step 4: Update `detect.py` -- New Joint Indices + Simplified Bone Scaling

**4a. Update joint references throughout `detect.py`:**

1. `flip_data()`: `left_joints` was `[4, 5, 6, 11, 12, 13]`, becomes `[4, 5, 6, 10, 11, 12]`. `right_joints` was `[1, 2, 3, 14, 15, 16]`, becomes `[1, 2, 3, 13, 14, 15]`.
2. `_RELIABLE_BONES_FOR_SCALE`: was `{11, 12, 13, 14, 15, 16}`, becomes `{10, 11, 12, 13, 14, 15}`.
3. `_DEPTH_RELIABLE_JOINTS` inside `motionbert_to_camera_space`: was `{0, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}`. Remove 10 (old Head) and shift. Becomes `{0, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15}` (i.e., all 16 joints minus knees 2,5 and ankles 3,6).
4. All hardcoded `range(1, 17)` -> `range(1, 16)`. All `range(17)` -> `range(16)`. All `np.zeros((17, ...))` -> `np.zeros((16, ...))`.
5. Joint index references for 2D keypoints inside `motionbert_to_camera_space`:
   - `kp_2d[11]` (LShoulder) -> `kp_2d[10]`
   - `kp_2d[14]` (RShoulder) -> `kp_2d[13]`
   - `kp_2d[4]` (LHip) -> `kp_2d[4]` (unchanged)
   - `kp_2d[1]` (RHip) -> `kp_2d[1]` (unchanged)
   - `kp_2d[8]` (Thorax) -> `kp_2d[8]` (unchanged in `pixel_aligned_to_camera_space`)
   - `kp_2d[3]` and `kp_2d[6]` (ankles) -> unchanged

6. `run_motionbert()`: The input to MotionBERT must still be 17 joints (the model expects H36M 17-joint input). However, `mpii_to_h36m()` now produces 16 joints. **Important decision**: We need to feed MotionBERT 17 joints as input (the model was trained on 17-joint H36M data), then strip the Head joint from the output. Therefore:
   - Add a helper function or inline logic in `run_motionbert()` that temporarily pads the 16-joint input back to 17 joints (inserting a zero/copy at index 10) before feeding to MotionBERT, then strips index 10 from the 17-joint output.
   - Actually, looking more carefully: `mpii_to_h36m` is called in `run_motionbert` to convert MPII 16 -> H36M. MotionBERT's model expects `(N, 17, 3)` input. The simplest approach: **keep `mpii_to_h36m` producing 17 joints for MotionBERT input**, but create a separate `strip_head_joint(arr_17)` utility that removes index 10 to go from 17->16. Apply `strip_head_joint` to all MotionBERT outputs before returning from `run_motionbert`.

   **Revised approach for `mpii_to_h36m`**: Keep it producing `(17, D)` as-is (unchanged), since it is only used as input to MotionBERT. Instead, add a new function:

   ```python
   def h36m_17_to_16(arr: np.ndarray) -> np.ndarray:
       """Remove Head joint (index 10) from H36M 17-joint array.
       Args:
           arr: (..., 17, D) array.
       Returns:
           (..., 16, D) array with Head joint removed.
       """
       return np.delete(arr, 10, axis=-2)
   ```

   Place this in `skeleton.py`.

   Then in `detect.py`:
   - `run_motionbert()` returns `positions_3d_pixel` and `positions_3d_norm` both as `(N, 17, 3)`. After the denormalization, apply `h36m_17_to_16` to both outputs before returning. The return shapes become `(N, 16, 3)`.
   - `detect_poses()`: After converting MPII 2D to H36M 2D using `mpii_to_h36m`, apply `h36m_17_to_16` to the 2D keypoints and visibility. The `kp_2d_list` entries become `(16, 2)` and `visibility_list` entries become `(16,)`.
   - `crop_scale()`: Operates on `(N, 17, 3)` -- this is used for MotionBERT input, keep as 17 joints.

   **Wait -- re-reading `detect_poses()` more carefully**: `mpii_to_h36m` is used in two places:
   1. Inside `run_motionbert()` to prepare MotionBERT input -- must stay at 17 joints.
   2. Inside `detect_poses()` at line 1096 to convert MPII 2D to H36M 2D for downstream use -- should produce 16 joints.

   So: keep `mpii_to_h36m` producing 17 joints (rename nothing, it stays as-is). Add `h36m_17_to_16` to `skeleton.py`. Apply it in `detect_poses()` after the MPII-to-H36M conversion for 2D keypoints/visibility, and in `run_motionbert()` to the 3D outputs.

   **But also**: `coco19_to_h36m` is used for ground truth in `panoptic.py` or `main.py`. This function should now produce 16 joints directly. Since GT doesn't go through MotionBERT, change `coco19_to_h36m` to produce `(16, 3)` as described in Step 1.

   **Revised plan for `mpii_to_h36m`**: Keep producing 17 joints (only used for MotionBERT input). Do NOT change it.

   **Revised plan for `coco19_to_h36m`**: Change to produce 16 joints (used for GT, which never goes through MotionBERT).

**4b. Simplify `motionbert_to_camera_space` -- remove torso-height heuristic:**

Inside `motionbert_to_camera_space`:

1. Remove the torso-height cross-check block (lines 960-979). Specifically, remove the code that computes `shoulder_mid_2d`, `hip_mid_2d`, `torso_pixel_height`, `tz_heuristic`, and the blending line `tz = 0.4 * tz + 0.6 * tz_heuristic`. After the pairwise `tz` is computed (line 956), it should be used directly without blending.

2. Remove the entire fallback block (lines 1008-1041) that uses the torso-height heuristic when `n_valid_tz < 4`. Replace it with a simple fallback: if `n_valid_tz < 4`, use `tz = 3.0` (a reasonable default depth) and compute `tx`, `ty` from the root 2D keypoint, then apply `_enforce_bone_lengths`.

3. Remove the deprecated `pixel_aligned_to_camera_space` function (lines 503-548) entirely. It is marked DEPRECATED and not called anywhere.

### Step 5: Update `evaluate.py` -- New Joint Indices

1. `EVAL_JOINTS`: was `[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]`. Now: `[1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]`.
2. `EVAL_JOINTS_NO_ANKLES`: was `[j for j in EVAL_JOINTS if j not in (3, 6)]`. The comprehension will auto-update based on the new `EVAL_JOINTS`, so no code change needed beyond `EVAL_JOINTS` itself.
3. All `range(17)` loops -> verify they use `NUM_JOINTS` from skeleton. The code already uses `NUM_JOINTS` from imports, so this should auto-update.

### Step 6: Update `graphs.py` -- Fix Key Joints in Summary

1. `generate_summary()`: `key_joints = [0, 8, 9, 10, 13, 3]` includes old index 10 (Head) and old index 13 (LWrist). Update to new indices: `key_joints = [0, 8, 9, 12, 15, 3]` (Hip, Thorax, Neck, LWrist, RWrist, RAnkle). Or pick a similar representative set without Head, e.g., `[0, 8, 9, 12, 3, 6]` (Hip, Thorax, Neck, LWrist, RAnkle, LAnkle).

   Recommended: `key_joints = [0, 8, 9, 12, 15, 3]` -- replaces Head with RWrist to maintain 6 representative joints from different body parts.

### Step 7: Update `overlay_video.py` -- Verify No Hardcoded Indices

The file uses `BONES` and `NUM_JOINTS` from skeleton. With the skeleton change, `BONES` will auto-update (generated from `PARENTS`). `NUM_JOINTS` will be 16. The `_draw_skeleton_2d` function iterates `range(NUM_JOINTS)` -- this will adapt. No hardcoded 17s found. No changes needed.

### Step 8: Update `optimize.py` -- Verify No Hardcoded Indices

Uses `NUM_JOINTS` from skeleton. The `range(17)` in `_enforce_bone_lengths` and `_enforce_bone_lengths_with_2d` (which are in `detect.py`, not `optimize.py`) need updating. `optimize.py` itself uses `NUM_JOINTS` everywhere. No hardcoded 17s found. No changes needed.

### Step 9: Update `main.py` -- Verify Joint Index References

1. Line 215: `for j in range(17)` in the FK target replacement loop. Change to `for j in range(16)` or better `for j in range(NUM_JOINTS)` -- import `NUM_JOINTS` from skeleton.
2. Verify all other references use imported constants rather than hardcoded 17.

### Step 10: Update `detect.py` helper functions -- Range Fixes

1. `_enforce_bone_lengths()`: `range(1, 17)` -> `range(1, NUM_JOINTS)`. Import `NUM_JOINTS` from skeleton if not already imported.
2. `_enforce_bone_lengths_with_2d()`: Same -- `range(1, 17)` -> `range(1, NUM_JOINTS)`. `np.zeros((17, 3))` -> `np.zeros((NUM_JOINTS, 3))`.
3. `_reconstruct_from_2d()`: `range(1, 17)` -> `range(1, NUM_JOINTS)`. `np.zeros((17, 3))` -> `np.zeros((NUM_JOINTS, 3))`.
4. `run_motionbert()`: `np.zeros((n_frames, 17, 3))` stays as 17 since this is MotionBERT input. But after MotionBERT output, apply `h36m_17_to_16`.
5. `detect_poses()`: The MotionBERT output `kp_3d_array` will now be `(N, 16, 3)` after Step 4. Downstream slicing should work.

### Step 11: Verify `fk.py`

1. `positions_to_fk_params()`: Uses `NUM_JOINTS` from skeleton and `PARENTS`, `REST_DIRECTIONS`. With updated skeleton, this auto-adapts. Line 197: `positions[8] - positions[0]` (Thorax - Hip) -- Thorax is still index 8. OK. Line 198: `positions[1] - positions[4]` (RHip - LHip) -- unchanged. OK.
2. `forward_kinematics()`: Uses `NUM_JOINTS`, `PARENTS`, `REST_DIRECTIONS`. Auto-adapts.

### Step 12: Run a Quick Test

After all changes, run:
```bash
cd motionbert-pose && uv run python -c "
from skeleton import NUM_JOINTS, JOINT_NAMES, PARENTS, EVAL_JOINTS, BONES, DEFAULT_BONE_LENGTHS, REST_DIRECTIONS
print(f'NUM_JOINTS={NUM_JOINTS}')
print(f'JOINT_NAMES={JOINT_NAMES}')
print(f'PARENTS={PARENTS}')
print(f'EVAL_JOINTS={EVAL_JOINTS}')
print(f'BONES={BONES}')
print(f'len(DEFAULT_BONE_LENGTHS)={len(DEFAULT_BONE_LENGTHS)}')
print(f'len(REST_DIRECTIONS)={len(REST_DIRECTIONS)}')
assert NUM_JOINTS == 16
assert 'Head' not in JOINT_NAMES
assert len(PARENTS) == 16
assert len(DEFAULT_BONE_LENGTHS) == 16
assert len(REST_DIRECTIONS) == 16
assert len(BONES) == 15
print('All skeleton assertions passed')
"
```

Then run the pipeline on example 0 to verify end-to-end:
```bash
cd motionbert-pose && uv run python -c "
from main import process_example
import config as cfg
process_example(*cfg.EXAMPLES[0], 'training_runs/test_p0')
"
```

## Integration Points

- **MotionBERT model**: Still expects `(N, 17, 3)` input. The `mpii_to_h36m` function continues to produce 17 joints. The `h36m_17_to_16` function strips the Head joint from MotionBERT output before downstream processing.
- **Ground truth**: `coco19_to_h36m` now produces 16 joints directly. No Head extrapolation needed.
- **FK system**: `forward_kinematics` and `positions_to_fk_params` use the 16-joint skeleton. The kinematic chain from Thorax -> Neck still exists but Neck is now a leaf (was previously Neck -> Head).
- **Scoring/Optimization**: All scoring functions operate on 16 joints. The Head heatmap (MPII index 9) is no longer mapped and will not be scored.

## Risks and Edge Cases

1. **MotionBERT input must remain 17 joints.** The model was trained on 17-joint H36M data. We must NOT change `mpii_to_h36m` or `crop_scale` to produce 16 joints, only strip the Head from the output.

2. **Joint index cascade.** Removing an index in the middle of the array shifts all subsequent indices. Every hardcoded index >= 10 (for old indices >= 11) must be decremented. The most error-prone places are `detect.py` (many hardcoded joint indices for shoulder/hip 2D lookups) and `config.py` (penalty weights).

3. **PARENTS array correctness is critical.** One wrong parent index will cause FK to produce garbage. The developer must verify the new PARENTS array by checking each joint's parent name manually.

4. **The fallback in `motionbert_to_camera_space` is important.** When removing the torso-height heuristic fallback, we must still handle the case where `n_valid_tz < 4`. A simple `tz = 3.0` default is reasonable but may be wrong for unusual camera setups. This is acceptable because: (a) it is rare (most frames have >4 valid upper-body joints), and (b) the optimization can correct the depth afterward.

5. **`flip_data` in detect.py** uses hardcoded L/R joint indices for MotionBERT flip augmentation. These indices must match the **17-joint** H36M format (since flip_data operates on MotionBERT input/output). However, if `flip_data` is also called on 16-joint data elsewhere, it needs a separate version. **Check**: `flip_data` is only called inside `run_motionbert()` on the 17-joint MotionBERT input/output. So keep `flip_data` using 17-joint indices. BUT -- `run_motionbert` returns 17-joint data that gets stripped to 16 joints by `h36m_17_to_16` before returning. So `flip_data` is fine with 17-joint indices as long as it is called before the stripping.

6. **No change to `panoptic.py`** -- it loads raw COCO19 data. The conversion `coco19_to_h36m` is in `skeleton.py` and called from `main.py`'s ground truth loading path via `panoptic.py`'s `load_ground_truth_sequence`. Wait -- let me check. Actually, `coco19_to_h36m` is called in the GT loading path. Looking at `main.py` line 186-194: `load_ground_truth_sequence` returns COCO19 joints, then `world_to_camera` transforms them, but `coco19_to_h36m` is not called in main.py. Let me check `panoptic.py`.

   **Confirmed**: `coco19_to_h36m` is called inside `panoptic.py` at line 100 within `load_ground_truth_sequence`. The GT is already in H36M format when `main.py` receives it. With the updated `coco19_to_h36m` producing 16 joints, the GT path will automatically produce 16-joint arrays. Also update the docstring in `panoptic.py` at lines 88 and 91 (says "H36M 17-joint" -- change to "H36M 16-joint").
