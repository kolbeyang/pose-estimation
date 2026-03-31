# Nit-pick Review: Skeleton Rework, Directory Restructure, and Isolation

**Date:** 2026-03-30 17:30
**Reviewer:** Nathan (QA, iteration 1)

---

## Methodology

Read the spec line by line, extracted every requirement, then verified each against the actual code (skeleton.py, scoring.py, evaluate.py, cmu_data.py, detect_motionbert.py, detect_mediapipe.py, pipeline_mediapipe.py, pipeline_motionbert.py, config files, and test files). Ran `git diff` and grepped the full `pose-optimizer` tree for violations.

---

## Requirement-by-Requirement Verification

### 1. Skeleton Joint Mapping (Category 1: 12 eval joints)

**Status: PASS**

Verified every single index mapping in `mpii_to_motionbert_skeleton()`, `mediapipe_to_skeleton()`, and `coco19_to_motionbert_skeleton()` / `coco19_to_mediapipe_skeleton()` against the spec table. All 12 MPII, MediaPipe, and COCO19 indices match exactly.

### 2. Skeleton Joint Mapping (Category 2: 2 synthesized joints)

**Status: PASS**

- Pelvis: MPII[6] direct (MotionBert), midpoint(MP[23], MP[24]) (MediaPipe), COCO[2] direct (GT) -- all correct
- BaseOfNeck: MPII[8] direct (MotionBert), midpoint(MP[11], MP[12]) (MediaPipe), COCO[0] direct (GT) -- all correct

### 3. Skeleton Joint Mapping (Category 3: optimization-only)

**Status: PASS**

- MotionBert: Thorax(14) from MPII[7], HeadTop(15) from MPII[9] -- correct
- MediaPipe: Nose(14) from MP[0] -- correct

### 4. No Computed Midpoints in Optimizer

**Status: PASS**

Grepped `optimize/` for `midpoint` and `/ 2.0` -- zero matches. The optimizer's FK chain has no midpoint computations. Only the detector mapping functions (`mediapipe_to_skeleton()`) compute midpoints for Pelvis and BaseOfNeck, exactly as spec requires.

### 5. Evaluation is Strictly 14 Joints

**Status: PASS**

- Both skeletons have `eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` (14 joints)
- Both pipelines pass `_SKEL.eval_joints` to `evaluate()`
- `evaluate()` slices to `eval_joints` before computing all metrics

### 6. Directory Rename (run-mediapipe / run-motionbert with hyphens)

**Status: PASS**

- `run-mediapipe/` and `run-motionbert/` directories exist with hyphens (not underscores)
- Both contain only JSON config files, no Python packages
- Old `mediapipe/` and `motionbert/` Python package directories are removed

### 7. Checkpoints in pose-optimizer/checkpoints/

**Status: PASS**

- `checkpoints/` exists with `pose_landmarker_lite.task`
- `detect_mediapipe.py` line 20: `_MODEL_PATH = os.path.join(_SCRIPT_DIR, "checkpoints", "pose_landmarker_lite.task")`
- `detect_motionbert.py` line 73: `CHECKPOINTS_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")`
- `setup_models.py` downloads to `checkpoints/` within `pose-optimizer`

### 8. pose-optimizer Completely Isolated

**Status: PASS (with one minor note)**

- Grepped for `../../`, `mediapipe-pose`, `motionbert-pose` -- zero matches
- Grepped for `../` -- only one hit: a stale comment in `cmu_data.py` line 21 that says `../data/panoptic-toolbox` but the actual code correctly uses `data/panoptic-toolbox` (relative to script dir, within `pose-optimizer`)
- All config files use `data_root: "data/panoptic-toolbox"` (within `pose-optimizer`)
- `DEFAULT_DATA_ROOT` resolves to `pose-optimizer/data/panoptic-toolbox`
- `data/panoptic-toolbox` directory exists (appears to be a symlink or actual data)

### 9. Synthetic Heatmaps Circular (not stretched)

**Status: PASS**

`scoring.py:generate_synthetic_heatmaps()` correctly uses separate `sigma_hm_x = sigma / sx` and `sigma_hm_y = sigma / sy` so that the Gaussian is circular in pixel space even for non-square images. Test `test_heatmap_circular_for_nonsquare` validates this.

### 10. Projection Tests with Real Camera Parameters

**Status: PASS**

`experiment/test_projection.py` has 12 tests including:
- Tests using real CMU Panoptic calibration data (`_get_real_camera()` loads from `171204_pose1_sample`)
- World-camera roundtrip with real calibration
- GT projects inside frame
- Numpy/torch consistency on real GT
- End-to-end scoring pathway tests

Tests appropriately skip if CMU data is not available.

---

## Issues Found

### NEW ISSUES

#### Issue 1: `compare.py` uses hardcoded OLD joint layout (BLOCKING for cross-pipeline comparison)

`compare.py` lines 25-30 define:
```python
JOINT_NAMES = ["Hip", "RHip", "RKnee", ..., "RShoulder", "RElbow", "RWrist"]  # old 16-joint
EVAL_JOINTS = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]  # old indices
```

This is the OLD H36M-16 joint layout. The new skeleton has different joint names, different ordering (Pelvis instead of Hip, BaseOfNeck instead of Thorax, etc.), and `EVAL_JOINTS = [0..13]`. If `compare.py` is run on outputs from the new skeleton, it will:
- Mislabel joints in graphs
- Evaluate the wrong subset of joints
- Produce incorrect per-joint comparisons

Both Dan and Eve noted this. It needs to be updated before any cross-pipeline comparison.

#### Issue 2: Stale comment in `cmu_data.py` line 21

The comment says `../data/panoptic-toolbox` but the code resolves to `data/panoptic-toolbox` (within `pose-optimizer`). Minor documentation inconsistency.

#### Issue 3: Stale docstring in `scoring.py` heatmap_score_batch

Line 65 says "scores all 16 joints directly (heatmaps have 16 channels matching H36M indices)". The MediaPipe skeleton has 15 joints, not 16. The code is shape-agnostic and works correctly, but the docstring is misleading.

#### Issue 4: `coco19_to_motionbert_skeleton` maps Thorax as midpoint

Line 483: `out[14] = (joints19[2] + joints19[0]) / 2.0  # Thorax (midpoint for vis, not eval)`

The spec says "No inferred/computed midpoints in the optimizer's skeleton." This IS a midpoint computation. However, it is in the GT loading function (not the optimizer's FK chain), and Thorax is opt-only (excluded from eval). The spec's principle is about the optimizer's FK chain, not GT loading. But it is worth noting: the GT for Thorax is an approximation, so if Thorax GT is ever used for loss computation, the approximation quality matters.

Eve correctly flagged this as "Acceptable" in her report. I concur that it does not violate the spec's intent (which is about the optimizer's skeleton, not GT approximations), but it should be documented.

#### Issue 5: HeadTop in GT is actually Nose

In `coco19_to_motionbert_skeleton()`, line 484: `out[15] = joints19[1]  # HeadTop <- Nose (COCO19)`

COCO19 index 1 is "Nose" in CMU Panoptic. The new skeleton calls this joint "HeadTop". The spec acknowledges this mismatch in Category 3: "COCO19 has Nose, not Head Top (~10-15cm apart)". Since HeadTop is excluded from evaluation, this is fine, but the joint name "HeadTop" is misleading when the GT data is actually Nose. This is a known trade-off documented in the spec.

---

## Pre-existing TODO Items Verified

The TODO has several pre-existing items. Quick status check:
- `compare.py` stale joint layout: Still present (Issue 1 above)
- MotionBert VW-SI-MPJPE regression: Still needs monitoring
- MotionBert high absolute MPJPE: Still needs investigation
- 9 failing examples in full config: Pre-existing, not in scope for this iteration
- 171204_pose1_30000 no GT: Pre-existing
- MotionBert outlier 171204_pose3_200: Pre-existing

---

## Verdict

**The implementation correctly implements all spec requirements.** The joint mappings, directory structure, heatmap fix, projection tests, self-containment, and evaluation are all verified.

The only new actionable item is `compare.py` needing an update to the new joint layout before it can be used. The other findings are minor documentation issues that do not affect correctness.
