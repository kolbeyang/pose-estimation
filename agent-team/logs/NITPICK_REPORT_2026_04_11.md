# Nitpick Report: Joint Alignment Revision

**Date**: 2026-04-11
**Reviewer**: Nathan (QA Nitpick)
**Commit under review**: `27da70f`
**SPEC**: `agent-team/specs/joint-alignment.md`
**Developer report**: `agent-team/logs/REPORT_2026_04_11_JOINT_ALIGNMENT.md`
**Testing report**: `agent-team/logs/TESTING_REPORT_2026_04_11.md`

---

## Method

Went through the SPEC line by line, extracted every requirement, and verified each against the actual code (reading files + git diff `5853a3c..27da70f`). Checked all `.py` files in `pose-optimizer/` and `pose-optimizer/experiment/` for stale references to old joint semantics.

---

## SPEC Requirement Verification

### Req 1: "Skeleton representations for MotionBert connects the shoulders to the new neck joint (which is called 'spine')"

**PASS.**

The SPEC wants shoulders to connect to the thorax-level joint. In MotionBERT's terminology, this is their "Spine" (MB joint 7). In our skeleton, shoulders (indices 10, 13) connect to joint 8 via `PARENTS`. Joint 8 now receives data from MPII[7] Thorax (= MB "Spine"). So shoulders connect to the correct joint.

`PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]`
- `PARENTS[10]` = 8 (LShoulder -> Neck)
- `PARENTS[13]` = 8 (RShoulder -> Neck)

Joint 8 = thorax/shoulder level. Correct.

### Req 2: "The joint that is called 'neck' for motionbert and the ground truth can be omitted from the pipeline because it does not exist in mediapipe"

**PASS.**

The anatomical neck (SH[8] UpperNeck, MB joint 8 "Neck") is no longer mapped to any skeleton joint. It appears exactly once in the code: `skeleton.py` line 168, inside the Nose synthesis formula `keypoints_mpii[8] + 0.3 * (keypoints_mpii[9] - keypoints_mpii[8])`. This is an internal computation using raw MPII indices, not a skeleton joint assignment. The SPEC allows this (Category 3 explicitly says Nose is synthesized from SH[8] and SH[9]).

MPII heatmap channel 8 (UpperNeck) is no longer scored against any skeleton joint (`SKELETON_TO_MPII_HEATMAP` has no entry mapping to channel 8). Correct.

### Req 3: Neck joint mapping per source

SPEC says Neck (SPEC #14) maps to:

| Source | SPEC says | Code does | Verdict |
|--------|-----------|-----------|---------|
| SH (MPII 16) | SH[7] Thorax | `skel[8] = keypoints_mpii[7]` | **PASS** |
| MotionBERT | MB joint 7 "Spine" | MotionBERT receives 2D Thorax at joint 8 in 17-joint input; its output at joint 8 is its H36M "Neck" prediction (= midpoint of shoulders in training) | **PASS** |
| MediaPipe | midpoint(LShoulder, RShoulder) | `skel[8] = (landmarks[11] + landmarks[12]) / 2.0` | **PASS** (unchanged) |
| CMU GT (COCO19) | COCO[0] Neck | `skel[8] = joints19[0]` | **PASS** (unchanged) |

### Req 4: PARENTS consistency

**PASS.** PARENTS unchanged. Shoulders connect to joint 8 (now thorax-level). Spine(7) connects to Pelvis(0). Neck(8) connects to Spine(7). Nose(9) connects to Neck(8). All correct.

### Req 5: Stale comments, variable names, docstrings

**Two issues found in experiment scripts (not in core code).**

Core code (`skeleton.py`, `scoring.py`, `config.py`) -- all comments updated correctly. The diff shows "Base of Neck" replaced with "Thorax / shoulder level" everywhere it appeared.

Issue 1: `experiment/verify_keypoint_mapping.py` line 132:
```python
"MB Opt": {9: "synth: 30% Neck->HeadTop"},
```
"Neck" here is ambiguous. In our skeleton, "Neck" now means thorax/shoulder level. But the Nose is synthesized from raw MPII[8] (UpperNeck), not from our skeleton's Neck. The label should say "30% UpperNeck(MPII[8])->HeadTop(MPII[9])".

Issue 2: `experiment/verify_keypoint_mapping.py` line 228:
```python
note = "H36M 'Neck/Nose' -- 2D input was synth mid(Neck,HeadTop)"
```
Two problems: (a) says "mid" but the formula is 30% not midpoint, (b) "Neck" is now ambiguous. Pre-existing inaccuracy worsened by the rename.

### Req 6: Overlay video / visualization correctness

**PASS.** `overlay_video.py` uses `BONES` from `skeleton.py`, which derives from `PARENTS`. Since PARENTS is unchanged, bone connectivity is correct. The overlay draws Pelvis->Spine->Neck->Nose and Neck->shoulders correctly with the new joint positions.

No labels or legends reference specific joint names or indices. The legend just says "Green=MotionBERT Red=Optimized Blue=GT".

### Req 7: Experiment scripts with hardcoded joint indices

**One pre-existing bug found; no new issues from the joint alignment change.**

- `experiment/single_frame_heatmap.py` line 12 imports `_EVAL_MPII_CHANNELS` from `overlay_video.py`, but this symbol does not exist in `overlay_video.py`. The script will crash on import. This is a pre-existing bug unrelated to the joint alignment change.

- `experiment/audit_fk_roundtrip.py` line 164: `if j != 7: # skip Spine` -- still correct (Spine is still FK-internal).

- `experiment/verify_keypoint_mapping.py` lines 64-82: `H36M_17_NAMES` list with H36M-specific joint names -- these describe MotionBERT's native naming, not our skeleton. Still correct as H36M reference data.

---

## Additional Findings

### Finding A: REST_DIRECTIONS[9] suboptimal for new Neck position (Low priority)

`REST_DIRECTIONS[9]` (Neck->Nose) is `[0, -0.5, -0.866]`, encoding "mostly forward (Z=-0.866), slightly upward (Y=-0.5)" in camera convention (Y-down). With Neck now at thorax/shoulder level, the physical direction from thorax to nose is predominantly upward with some forward lean -- roughly `[0, -0.866, -0.5]` (or similar).

The current rest direction means the FK solver starts with a poor estimate for bone 9's orientation and must learn a larger rotation to compensate. This affects initialization quality but not final convergence (the optimizer adjusts).

The PLAN explicitly noted "REST_DIRECTIONS: No change needed. Physical directions unchanged." This assessment is incorrect -- the physical direction from thorax to nose IS different from the direction from anatomical-neck to nose. However, the testing results show the optimizer still converges well, so this is a minor optimization opportunity, not a correctness bug.

**Severity**: Low. Would improve FK initialization for the Nose joint.

### Finding B: MotionBERT 3D output joint 7 is not re-synthesized (Non-issue)

The PLAN says "Joint 7 (Spine) becomes synthesized as midpoint(Pelvis, joint 8) for ALL sources." For the MotionBERT pipeline, joint 7 in the 2D input is correctly synthesized as midpoint(Pelvis, Neck). However, MotionBERT's 3D output at joint 7 is its own learned prediction (not re-synthesized by us).

This is not a problem because: (a) joint 7 is excluded from evaluation (EVAL_JOINTS), (b) the optimizer uses FK to compute all joint positions from root+rotations+bone_lengths, so the initial joint 7 position from MotionBERT's prediction only affects initialization, (c) MotionBERT's "Spine" prediction is approximately midpoint(root, neck) per H36M convention anyway.

**Severity**: Non-issue. Documenting for completeness.

### Finding C: Dual heatmap scoring at channel 7 (Acknowledged, non-issue)

Both skeleton joints 7 (Spine) and 8 (Neck) now score against MPII heatmap channel 7 (Thorax). This was noted in both the developer report and testing report. The double gradient signal toward Thorax is acceptable because joint 7 is constrained by FK (it's between Pelvis and Neck), so the heatmap pull just reinforces the correct position.

MPII channel 8 (UpperNeck) is now orphaned -- no skeleton joint scores against it. This is a minor signal loss but is correct per the SPEC since the anatomical neck has no MediaPipe equivalent.

**Severity**: Non-issue. By design.

### Finding D: Bone length update for bone 9 could be more precise (Minor)

`DEFAULT_BONE_LENGTHS[9]` was changed from 0.08 to 0.18. The PLAN discussed this at length and arrived at 0.18m as a reasonable estimate. The clamping range becomes [0.09, 0.27]m.

For GT (COCO), the Neck-to-Nose distance (COCO[0] to COCO[1]) is typically ~0.12-0.15m. For MotionBERT with Thorax at joint 8, the distance is larger (~0.20-0.25m). The chosen default of 0.18m with range [0.09, 0.27] accommodates both. The testing results show the optimizer converges, so this is adequate.

**Severity**: Non-issue. Chosen value is reasonable.

---

## Summary

| Category | Count |
|----------|-------|
| SPEC violations found | 0 |
| TODO items added (should-fix) | 2 (stale labels in verify_keypoint_mapping.py) |
| TODO items added (nice-to-have) | 1 (REST_DIRECTIONS[9] suboptimal) |
| TODO items added (pre-existing) | 1 (single_frame_heatmap.py broken import) |
| Non-issues documented | 3 |

**Overall assessment**: The implementation correctly satisfies the SPEC. The core code changes are minimal and accurate. The two stale labels in `verify_keypoint_mapping.py` are cosmetic issues in an experiment/debug script. The suboptimal REST_DIRECTIONS[9] is a minor FK initialization quality issue that does not affect correctness. No blocking issues found.
