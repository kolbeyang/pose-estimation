# Nit-pick Review: Post-Implementation Verification

**Date:** 2026-03-30 19:00
**Reviewer:** Nit-pick Nathan (QA, iteration 2)
**Scope:** Verify `git diff HEAD~4` against the SPEC at `claude-team/specs/2026-03-30.md` and user feedback.

---

## Methodology

1. Read the SPEC line by line and extracted every requirement.
2. Read the actual changed code (`skeleton.py`, `scoring.py`, `config.py`, `cmu_data.py`, `run_mediapipe/detect.py`, `run_motionbert/detect.py`, `run_mediapipe/__init__.py`, `run_motionbert/__init__.py`, `experiment/test_projection.py`, `README.md`).
3. Cross-referenced every joint mapping index against the SPEC tables.
4. Grepped the entire `pose-optimizer/` tree for `h36m`, `H36M`, `../`, `compare.py`, `Thorax`, old function names, and old directory names.
5. Compared the user's critical feedback (5 rules) against the implementation.

---

## User Feedback Compliance

### 1. "Simple renames only: mediapipe/ -> run_mediapipe/, motionbert/ -> run_motionbert/. Files stay inside."

**Status: PASS**

- `run_mediapipe/` and `run_motionbert/` directories exist with correct names (underscores, not hyphens).
- Files (`__init__.py`, `detect.py`, config JSONs, model files) are inside the directories.
- Entry point scripts renamed: `mediapipe.py` -> `run_mediapipe.py`, `motionbert.py` -> `run_motionbert.py`.
- Old directories do not exist.

### 2. "No H36M references: Zero h36m in the codebase."

**Status: FAIL (4 remaining in .py files, 3 in README.md)**

In Python files (all in `run_motionbert/` and are the upstream checkpoint filename -- acknowledged as unfixable):
- `run_motionbert/detect.py:278` -- `"motionbert_lite_h36m.bin"`
- `run_motionbert/setup_models.py:50` -- comment `# 2. Download MotionBERT-Lite H36M checkpoint`
- `run_motionbert/setup_models.py:51` -- `"motionbert_lite_h36m.bin"`
- `run_motionbert/setup_models.py:54` -- HuggingFace URL containing `h36m`

In README.md (stale, should be fixed):
- Line 70: `"16-joint H36M skeleton definition and joint mappings"`
- Line 139: `"16-joint H36M skeleton (Head removed). No extrapolated joints."`
- Line 143: Old joint names `8=Thorax  9=Neck` (should be `8=Neck  9=Head`)

**Verdict:** The Python file references are the external checkpoint filename (unfixable without re-hosting). The README references are stale documentation that should be updated. Added to TODO.

### 3. "No backwards compatibility: No old aliases or compat shims."

**Status: PASS**

- `camera.py` had backward-compat aliases removed (`world_to_image`, `world_to_image_torch`).
- No old function names (`mpii_to_h36m`, `coco19_to_h36m`, etc.) exist anywhere in the codebase.
- No import shims or deprecation wrappers.

### 4. "Self-contained: No references outside pose-optimizer/."

**Status: PASS**

- Grepped `*.py` and `*.json` for `../` -- only hit is a stale COMMENT in `cmu_data.py:16` that says `../data/panoptic-toolbox`. The actual code resolves to `data/panoptic-toolbox` within `pose-optimizer/`. The comment is wrong but the behavior is correct.
- `DEFAULT_DATA_ROOT` resolves to `pose-optimizer/data/panoptic-toolbox`.
- `data/panoptic-toolbox` exists as a real directory (not symlink) within `pose-optimizer/`.

### 5. "Don't over-engineer."

**Status: PASS**

No unnecessary abstractions, no backward compat shims, no elaborate migration code. Changes are direct renames and mapping updates.

---

## SPEC Requirement Verification

### Joint Mapping: Category 1 (12 Direct Eval Joints)

**Status: PASS -- all indices verified**

Checked every index in three functions against SPEC table:

| Skel Joint | `mpii_to_skeleton` | SPEC (MPII) | `mediapipe_to_skeleton` | SPEC (MP) | `coco19_to_skeleton` | SPEC (COCO19) |
|---|---|---|---|---|---|---|
| 1 RHip | mpii[2] | MPII[2] | MP[24] | MP[24] | COCO[12] | COCO[12] |
| 2 RKnee | mpii[1] | MPII[1] | MP[26] | MP[26] | COCO[13] | COCO[13] |
| 3 RAnkle | mpii[0] | MPII[0] | MP[28] | MP[28] | COCO[14] | COCO[14] |
| 4 LHip | mpii[3] | MPII[3] | MP[23] | MP[23] | COCO[6] | COCO[6] |
| 5 LKnee | mpii[4] | MPII[4] | MP[25] | MP[25] | COCO[7] | COCO[7] |
| 6 LAnkle | mpii[5] | MPII[5] | MP[27] | MP[27] | COCO[8] | COCO[8] |
| 13 RShoulder | mpii[12] | MPII[12] | MP[12] | MP[12] | COCO[9] | COCO[9] |
| 14 RElbow | mpii[11] | MPII[11] | MP[14] | MP[14] | COCO[10] | COCO[10] |
| 15 RWrist | mpii[10] | MPII[10] | MP[16] | MP[16] | COCO[11] | COCO[11] |
| 10 LShoulder | mpii[13] | MPII[13] | MP[11] | MP[11] | COCO[3] | COCO[3] |
| 11 LElbow | mpii[14] | MPII[14] | MP[13] | MP[13] | COCO[4] | COCO[4] |
| 12 LWrist | mpii[15] | MPII[15] | MP[15] | MP[15] | COCO[5] | COCO[5] |

Note: For `mpii_to_skeleton`, output is 17-joint, then `strip_head_joint` removes index 10 to produce 16-joint. The shoulder/arm indices (11-16 in 17-joint) shift to (10-15 in 16-joint), which matches the 16-joint skeleton layout.

### Joint Mapping: Category 2 (2 Synthesized Eval Joints)

**Status: PASS**

| Joint | MotionBert (MPII) | MediaPipe | CMU GT (COCO19) |
|---|---|---|---|
| 0 Pelvis | mpii[6] direct | midpoint(MP[23], MP[24]) | COCO[2] direct |
| 8 Neck | mpii[8] direct | midpoint(MP[11], MP[12]) | COCO[0] direct |

All match SPEC exactly. The previous implementation computed Pelvis as `midpoint(mpii[2], mpii[3])` -- this has been corrected to `mpii[6]` (direct).

### Joint Mapping: Category 3 (Mode-Specific Opt-Only)

**Status: PASS**

- MotionBert: Skeleton joint 7 (Spine) from MPII[7] (what SPEC calls "Thorax"), joint 9 (Head) from MPII[9] (what SPEC calls "Head Top"). Both are used in optimization via `SKELETON_TO_MPII_HEATMAP` but excluded from eval.
- MediaPipe: Skeleton joint 9 (Head) from MP[0] (Nose). Excluded from eval.

Note on naming: SPEC uses "Thorax" for MPII[7] but the skeleton calls it "Spine" (joint 7). This is a naming discrepancy between the SPEC and implementation, but the actual index mapping is correct.

### 3 Categories Correct?

**Status: PASS**

- 12 direct eval: joints [1,2,3,4,5,6,10,11,12,13,14,15] -- all in EVAL_JOINTS
- 2 synthesized eval: joints [0,8] -- both in EVAL_JOINTS
- Mode-specific opt-only: joints [7,9] -- both excluded from EVAL_JOINTS

### EVAL_JOINTS = exactly 14?

**Status: PASS**

`EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15]` -- 14 entries. Correct.

### "No inferred/computed midpoints in the optimizer's skeleton"

**Status: PASS (with caveat)**

The optimizer's FK chain (`fk.py`, `optimize/`) has zero midpoint computations. However:
- `mediapipe_to_skeleton()` computes Pelvis and Neck as midpoints (this is the SPEC's Cat2 definition for MediaPipe -- required by design).
- `coco19_to_skeleton()` computes Spine as `midpoint(COCO[2], COCO[0])` -- Spine is FK-internal and excluded from eval, so this is acceptable. There is no direct COCO19 "Spine" joint.
- `mpii_to_skeleton()` uses MPII[6] and MPII[7] directly -- no midpoints. This was fixed from the old code.

The principle is about the optimizer, not the detection/GT mapping. The optimizer skeleton has no midpoints.

### Projection Tests with REAL Camera Parameters

**Status: PASS**

`experiment/test_projection.py` uses `fx=1395, fy=1395, cx=960, cy=540, (1080, 1920)` which are typical CMU Panoptic HD camera intrinsics. Tests 3-4 (`TestWorldToCamera`, `TestFullPipelineProjection`) load actual calibration from `171204_pose1_sample` and skip if data is unavailable. 19 tests total across 7 test classes.

### Heatmap Fix (Circular Gaussians in Pixel Space)

**Status: PASS**

`scoring.py:generate_synthetic_heatmaps()` now uses per-axis sigma: `sigma_hm_x = sigma / sx`, `sigma_hm_y = sigma / sy`. The Gaussian blob is circular in pixel space (elliptical in heatmap space for non-square images). Verified by `test_non_square_circular_in_pixel_space` which asserts asymmetric heatmap values at equal heatmap distances from center.

---

## New Issues Found

### Issue 1: README.md is completely stale (HIGH PRIORITY)

The README has not been updated to reflect ANY of the skeleton changes:
- Line 70: Still says "16-joint H36M skeleton definition"
- Line 139: Still says "16-joint H36M skeleton (Head removed)"
- Line 143: Lists old joint names `8=Thorax  9=Neck` (should be `8=Neck  9=Head`)
- Line 147: Says "Evaluation uses 12 joints" (should be 14)
- Lines 20, 52: References `compare.py` which was deleted

This is the most visible documentation in the project and directly contradicts the code.

### Issue 2: config.py rotation_penalty_multipliers uses old joint names (LOW PRIORITY)

Comments at lines 35, 43-44 say "Hip", "Thorax", "Neck" instead of "Pelvis", "Neck (Base of Neck)", "Head". Functionally harmless but misleading.

### Issue 3: cmu_data.py stale comment (LOW PRIORITY)

Line 16 comment says the default data root is `../data/panoptic-toolbox` but code resolves to `data/panoptic-toolbox` within `pose-optimizer/`. Comment is wrong.

### Issue 4: scoring.py mixed naming convention (COSMETIC)

Line 24 says `# 7: Spine -- MPII Thorax`. Mixes the skeleton name and MPII name in one comment. Not wrong but could be clearer.

---

## Pre-existing TODO Items Status

| Item | Status |
|------|--------|
| results.json format deviation | Checked-off, closed |
| MotionBert VW-SI-MPJPE regression | Open, needs full-run data |
| MotionBert high absolute MPJPE | Open, needs investigation |
| Multi-person sequences | Checked-off, closed |
| generate_per_frame_mpjve dead import | Checked-off, closed |
| data/ directory | Checked-off, closed (data/panoptic-toolbox exists) |
| Output directory timestamp format | Open, cosmetic |
| Duplicate gaussian_sigma/heatmap_sigma | Checked-off, closed (gaussian_sigma removed) |
| 9 examples fail due to OOB frames | Open |
| 171204_pose1_30000 no ground truth | Open |
| MotionBert 171204_pose3_200 outlier | Open |
| mediapipe.py shadows mediapipe package | Checked-off, closed (renamed to run_mediapipe.py) |
| Unused imports EVAL_JOINT_NAMES etc | Checked-off, closed |
| h36m in checkpoint filename | Open (unfixable, external) |

---

## Verdict

The implementation correctly satisfies all functional SPEC requirements. All 36 joint mapping indices (12x3 systems) verified correct. The 3 categories are properly separated. Eval is exactly 14 joints. The optimizer has no midpoints. Heatmaps are circular in pixel space. Projection tests use real camera parameters.

**7 new TODO items added**, all documentation/comment issues. Zero functional bugs found.

The most important action item is updating `README.md` which has stale H36M references, old joint names, wrong eval joint count, and references to the deleted `compare.py`.
