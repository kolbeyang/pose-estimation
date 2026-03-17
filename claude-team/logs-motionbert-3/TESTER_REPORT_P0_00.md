# Tester Report P0-00: Verify Head Joint Removal + Simplified Bone Scaling

## Tests Run

| Test | Description | Result |
|------|-------------|--------|
| T1 | Skeleton constants (NUM_JOINTS=16, no Head in JOINT_NAMES, array lengths) | PASS |
| T1b | BODY_GROUPS updated (no old Head index, correct arm indices) | PASS |
| T2 | PARENTS array semantic correctness (each joint's parent name verified) | PASS |
| T3 | `h36m_17_to_16()` removes exactly index 10, preserves others | PASS |
| T4 | `coco19_to_h36m()` produces (16, 3) with correct mappings | PASS |
| T5 | `mpii_to_h36m()` still produces (17, 3) for MotionBERT input | PASS |
| T6 | `ROTATION_PENALTY_PER_JOINT` has 16 entries | PASS |
| T7 | `H36M_TO_MPII_HEATMAP` has 16 entries, no MPII Head (idx 9) mapped | PASS |
| T8 | FK roundtrip error = 0.000224 cm (well under 0.01 cm threshold) | PASS |
| T9 | No hardcoded `17` in detect.py logic paths (only in MotionBERT-specific code and docstrings) | PASS (see findings) |
| T10 | Torso-height heuristic code removed from `motionbert_to_camera_space` | PASS |
| T11 | `pixel_aligned_to_camera_space` removed from detect.py | PASS |
| T12 | `flip_data` uses 17-joint indices (correct for MotionBERT data) | PASS |
| T13 | `graphs.py` key_joints = [0, 8, 9, 12, 15, 3] -- no old Head index | PASS |
| T14 | Output JSON has 16-joint data everywhere (verified developer's test_p0 run) | PASS |
| R1 | MotionBERT receives (N, 17, 3) input -- verified in `run_motionbert()` | PASS |
| R2 | EVAL_JOINTS resolves to correct 12 joint names (no Head, no spine/hip/thorax/neck) | PASS |
| Import | All 11 source modules import without error | PASS |

## Bugs Found

None. No functional bugs were identified.

## Code Review Findings

### CR-1: Stale Docstrings Referencing 17 Joints (Minor)

Many docstrings still say `(17, 3)` or `(17,)` when the actual runtime data is now `(16, 3)` or `(16,)`. These are cosmetic but could confuse future developers.

**Affected files and locations:**

- **`detect.py`**: `_enforce_bone_lengths` docstring says `(17, 3)`, `(17,)` (lines 557-564). `_enforce_bone_lengths_with_2d` docstring says `(17, 3)`, `(17, 2)` (lines 613-624). `_reconstruct_from_2d` docstring says `(17, 2)`, `(17,)`, `(17, 3)` (lines 710-718). `detect_poses` docstring says `(N,17,3)` (line 974).
- **`scoring.py`**: `motion_penalty_position` says `(17, 3)` (lines 150-151). `initialization_penalty` says `(17, 3)`, `(17,)` (lines 195-197, 203). `motion_penalty_all_joints` says `(17, 3)` (lines 215-216). `compute_total_score` says `(17, 3)`, `(17, 2)`, `(17,)` (lines 247-255).
- **`evaluate.py`**: `reprojection_error_vs_detections` says `(17, 3)`, `(17, 2)`, `(17,)` (lines 117-119). `compute_comparison` says `(17, 3)` (lines 206-207). `compute_comparison_with_optimization` says `(17, 3)` (lines 283-285).
- **`optimize.py`**: `run_optimization` docstring says `(17, 3)`, `(17, 2)`, `(17,)` (lines 64-74).
- **`overlay_video.py`**: `_draw_skeleton_2d` says `(17, 2)` (line 150). `generate_overlay_video` says `(17, 3)` (lines 198-199, 203).
- **`graphs.py`**: Multiple functions say `(17, 3)` or `(17,)` (lines 44-46, 239, 241-242, 291-293, 296).
- **`fk.py`**: Module docstring says "H36M 17-joint skeleton" (line 1). `positions_to_fk_params` says `(17, 3)`, `(17,)` (lines 178, 184-185).

**Severity:** Minor. The code itself correctly uses `NUM_JOINTS` (16) everywhere, so the runtime behavior is correct. Only the documentation is stale.

**Suggested fix:** Find-and-replace `(17,` with `(16,` and `17-joint` with `16-joint` in docstrings across these files, but only where the docstring describes data that has already been stripped to 16 joints (not MotionBERT internal 17-joint data).

### CR-2: `detect.py` Helper Function Docstrings Say 17 But Loop Uses NUM_JOINTS (Minor)

`_enforce_bone_lengths`, `_enforce_bone_lengths_with_2d`, and `_reconstruct_from_2d` all have docstrings saying `(17, 3)` and `(17,)` but the actual loop uses `range(1, NUM_JOINTS)` which is `range(1, 16)`. The functions work correctly at runtime. The docstrings are simply stale.

**Severity:** Minor (cosmetic).

### CR-3: No Remaining Hardcoded 17s in Logic (Confirmed Good)

All remaining literal `17` references in detect.py are specifically in MotionBERT-related code where 17-joint arrays are intentionally used (model config `num_joints=17`, input array allocation `np.zeros((n_frames, 17, 3))`, confidence thresholding `range(17)`, and count `n_total = n_frames * 17`). These are correct because MotionBERT requires 17-joint input.

### CR-4: `flip_data` Correctly Uses 17-Joint Indices (Confirmed Good)

`flip_data` uses `left_joints = [4, 5, 6, 11, 12, 13]` and `right_joints = [1, 2, 3, 14, 15, 16]` which are 17-joint H36M indices. This is correct because `flip_data` is only called inside `run_motionbert()` on 17-joint data before `h36m_17_to_16` strips the Head.

### CR-5: Simplified Bone Scaling (Confirmed Good)

`motionbert_to_camera_space` no longer contains any torso-height heuristic, blending, or the old `pixel_aligned_to_camera_space` function. The depth estimation uses only pairwise separation with IQR filtering, falling back to `tz = 3.0` when fewer than 4 valid joints are available.

## Verdict

**YES** -- The implementation meets the Phase 0 definition of done.

1. **Head joint is fully omitted.** Removed from `JOINT_NAMES`, `PARENTS`, `DEFAULT_BONE_LENGTHS`, `REST_DIRECTIONS`, `BODY_GROUPS`, `EVAL_JOINTS`, `H36M_TO_MPII_HEATMAP`, `ROTATION_PENALTY_PER_JOINT`. The `coco19_to_h36m` function produces 16 joints directly. MotionBERT output is stripped from 17 to 16 joints via `h36m_17_to_16`. All downstream code (FK, scoring, optimization, evaluation, visualization) operates on 16 joints.

2. **Bone scaling simplified to pairwise separation only.** The torso-height heuristic and blending code have been completely removed. The deprecated `pixel_aligned_to_camera_space` is also gone.

3. **No functional bugs found.** All tests pass. The pipeline produces correct 16-joint outputs.

4. **One minor finding:** Many docstrings across multiple files still reference `(17, 3)` shapes when the runtime data is now `(16, 3)`. This is cosmetic and does not affect correctness, but should be cleaned up when convenient.
