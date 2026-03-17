# Tester Plan P0-00: Verify Head Joint Removal + Simplified Bone Scaling

## Phase 0 Goals
1. Head joint (old index 10) is omitted entirely from skeleton definition, evaluation, visualization, everything
2. Bone scaling in `motionbert_to_camera_space` simplified to pairwise separation only (no torso height heuristic)

## Tests

### T1: Skeleton Constants Verification
**What:** Verify NUM_JOINTS=16, Head not in JOINT_NAMES, PARENTS/DEFAULT_BONE_LENGTHS/REST_DIRECTIONS all have 16 entries, BONES has 15 entries, EVAL_JOINTS uses correct new indices.
**How:** `uv run python -c "from skeleton import ...; assert ..."`

### T2: PARENTS Array Correctness
**What:** Verify each joint's parent is semantically correct after index shift.
**How:** Python script that prints each joint's name and parent name, manually verify chain.

### T3: h36m_17_to_16 Utility
**What:** Verify the strip function removes exactly index 10 and preserves other joints.
**How:** Create a (17, 3) test array, apply function, check output shape and that correct row was removed.

### T4: coco19_to_h36m Produces 16 Joints
**What:** Verify GT conversion produces (16, 3) with no Head joint.
**How:** Create synthetic COCO19 data, convert, check shape and mapping correctness.

### T5: mpii_to_h36m Still Produces 17 Joints
**What:** Verify MotionBERT input path unchanged -- mpii_to_h36m still outputs (17, D).
**How:** `uv run python -c "from skeleton import mpii_to_h36m; ..."`

### T6: config.py ROTATION_PENALTY_PER_JOINT Has 16 Entries
**What:** Verify penalty array length matches NUM_JOINTS.
**How:** `uv run python -c "import config as cfg; assert len(cfg.ROTATION_PENALTY_PER_JOINT) == 16"`

### T7: scoring.py H36M_TO_MPII_HEATMAP Has 16 Entries
**What:** Verify heatmap mapping length matches NUM_JOINTS, and no entry maps to MPII index 9 (old Head).
**How:** `uv run python -c "from scoring import H36M_TO_MPII_HEATMAP; assert len(...) == 16"`

### T8: FK Roundtrip Test
**What:** Verify forward_kinematics(positions_to_fk_params(positions)) roundtrips with < 0.01 cm error.
**How:** Generate synthetic 16-joint positions, run inverse FK then forward FK, measure error.

### T9: detect.py -- No Hardcoded 17s
**What:** Grep for literal "17" in detect.py to check for remaining hardcoded references to old joint count.
**How:** Grep + manual review of flagged lines (17 in MotionBERT-specific code is expected).

### T10: Torso-Height Heuristic Removed
**What:** Verify `motionbert_to_camera_space` does not contain torso-height blending code.
**How:** Grep for "torso", "heuristic", "tz_heuristic", "0.4 * tz" in detect.py.

### T11: pixel_aligned_to_camera_space Removed
**What:** Verify deprecated function is gone.
**How:** Grep for "pixel_aligned_to_camera_space" in detect.py.

### T12: flip_data Uses 17-Joint Indices
**What:** Verify flip_data still operates on 17-joint MotionBERT data (indices go up to 16).
**How:** Read code -- left_joints should include 11,12,13 and right_joints should include 14,15,16.

### T13: graphs.py key_joints Updated
**What:** Verify key_joints no longer includes old Head index 10.
**How:** Read code.

### T14: Smoke Test -- Full Pipeline
**What:** Run process_example on example 0 and verify it completes without error. Check output JSON has 16-joint data.
**How:** `uv run python -c "from main import process_example; import config as cfg; process_example(*cfg.EXAMPLES[0], 'training_runs/test_p0_tester')"`

### T15: Docstring/Comment Consistency
**What:** Verify docstrings that previously said "17-joint" are updated to "16-joint" where appropriate.
**How:** Grep for "17-joint" and "17," in all source files.

## Regression Checks

### R1: MotionBERT Input Shape
**What:** Ensure MotionBERT still receives (N, 17, 3) input. The model requires this.
**How:** Verified via code review of run_motionbert() -- keypoints_h36m shape is (n_frames, 17, 3).

### R2: EVAL_JOINTS Contains Correct Joint Names
**What:** Verify EVAL_JOINTS resolves to the expected 12 joint names (no spine/hip/thorax/neck, no Head).
**How:** Print EVAL_JOINT_NAMES and compare to expected set.
