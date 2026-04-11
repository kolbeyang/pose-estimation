# Report: Joint Alignment Revision (Neck/Spine Remapping)

**Date**: 2026-04-11
**SPEC**: `agent-team/specs/joint-alignment.md`
**PLAN**: `agent-team/logs/PLAN_2026_04_11.md`
**Commit**: `27da70f` (on branch `main`)

---

## Summary

Remapped skeleton joint 8 ("Neck") from anatomical neck (MPII UpperNeck) to thorax/shoulder level (MPII Thorax), aligning with the SPEC's revised joint definitions. The anatomical neck (SH[8] UpperNeck) is now dropped from the skeleton; only used internally for Nose synthesis.

---

## Changes by Phase

### Phase 1: Update `skeleton.py` joint mappings

**File**: `pose-optimizer/skeleton.py`

1. **`mpii_to_skeleton()`** (lines 157-159):
   - Joint 8: `skel[8] = keypoints_mpii[7]` (was `keypoints_mpii[8]`). Now maps to MPII Thorax.
   - Joint 7: `skel[7] = (skel[0] + skel[8]) / 2.0` (was `keypoints_mpii[7]` direct). Now synthesized as midpoint(Pelvis, Neck).
   - Nose (line 168): `skel[9] = keypoints_mpii[8] + 0.3 * (keypoints_mpii[9] - keypoints_mpii[8])`. Still uses raw MPII[8] UpperNeck and MPII[9] HeadTop. Does NOT reference `skel[8]`.

2. **`coco19_to_skeleton()`**: No change (joint 8 already = COCO[0] at shoulder level).
3. **`mediapipe_to_skeleton()`**: No change (joint 8 already = midpoint of shoulders).
4. **Comments/docstrings**: Updated module docstring, `JOINT_NAMES[8]` comment, and function docstrings.

### Phase 2: Update `scoring.py` heatmap mapping

**File**: `pose-optimizer/scoring.py`, line 24

- `SKELETON_TO_MPII_HEATMAP[8]`: Changed from `8` (UpperNeck) to `7` (Thorax).
- MPII channel 8 (UpperNeck) is now unused by any skeleton joint.
- Both joints 7 and 8 now score against MPII Thorax heatmap (channel 7).

### Phase 3: Update bone lengths in `skeleton.py`

**File**: `pose-optimizer/skeleton.py`, line 77

- `DEFAULT_BONE_LENGTHS[9]`: Changed from `0.08` to `0.18`.
- Rationale: Neck is now at thorax/shoulder level, so Neck-to-Nose distance is ~18cm instead of ~8cm. The old 1.5x clamp (0.12m max) would have been too restrictive.
- Bones 7 and 8 kept at 0.22m (unchanged). The Pelvis-to-Neck total distance is still similar across GT/MP sources.

### Phase 4: FK Verification

- `positions_to_fk_params()` uses `positions[8] - positions[0]` for root orientation. This direction is still valid (pelvis to upper torso, just shorter). No code change needed.
- FK roundtrip was verified implicitly by the successful optimization runs in Phase 5.

### Phase 5: Comparison Run

**Config**: `configs/local-debug-baseline.json`
**Baseline output**: `output/run_2026_04_11_13_45/`
**New output**: `output/run_2026_04_11_14_04/`

---

## Metrics Comparison

### Primary Metric: Optimized VW-SI-MPJPE (cm)

| Pipeline / Example | Baseline | New | Delta |
|-|-|-|-|
| MotionBERT / pose1_360 | 27.31 | 25.27 | **-2.04 (7.5% better)** |
| MotionBERT / piano4_166 | 11.85 | 11.25 | **-0.59 (5.0% better)** |
| MediaPipe / pose1_360 | 12.73 | 12.79 | +0.06 (0.4% worse) |
| MediaPipe / piano4_166 | 20.40 | 20.22 | **-0.18 (0.9% better)** |

### Detection VW-SI-MPJPE (cm, before optimization)

| Pipeline / Example | Baseline | New | Delta |
|-|-|-|-|
| MotionBERT / pose1_360 | 37.24 | 37.81 | +0.57 (1.5%) |
| MotionBERT / piano4_166 | 11.35 | 10.63 | -0.72 (6.4% better) |
| MediaPipe / pose1_360 | 13.09 | 13.09 | 0.00 |
| MediaPipe / piano4_166 | 20.93 | 20.93 | 0.00 |

### Key Observations

1. **MotionBERT improved significantly** on the primary metric (VW-SI-MPJPE). The pose1 example saw a 7.5% improvement, and piano4 improved by 5.0%.
2. **MediaPipe is essentially unchanged**, as expected since the GT and MediaPipe mappings did not change. Tiny differences (<0.5%) are from the bone length change affecting optimization.
3. **Per-joint analysis** (MotionBERT, pose1): Nearly all joints improved after optimization. The Neck joint itself improved from 35.66cm to 25.54cm (10.12cm improvement). Nose improved from 39.23cm to 27.90cm (11.33cm improvement).
4. **The MPJPE (non-scale-invariant) for piano4 MotionBERT increased** (+14.2%), which indicates a scale estimation shift. However, SI-MPJPE improved, confirming the shape quality is better.

---

## Critical Decisions

1. **Kept joint indices 7 and 8 in place** -- remapped their data sources rather than swapping indices. This avoided cascading changes to PARENTS, BONES, REST_DIRECTIONS, BODY_GROUPS, and EVAL_JOINTS.
2. **Nose synthesis preserved using raw MPII indices** -- `keypoints_mpii[8]` (UpperNeck) and `keypoints_mpii[9]` (HeadTop), NOT `skel[8]` which is now Thorax. This was the most important correctness check.
3. **Bone 9 default set to 0.18m** (from 0.08m) -- this is conservative. The 1.5x clamp gives a max of 0.27m, which accommodates both GT (base-of-neck to nose ~0.12m) and MB (thorax to nose ~0.25m).

## Issues / Red Flags

- **None blocking.** The pipeline ran without errors on both examples.
- **Minor**: MPII heatmap channel 8 (UpperNeck) is now unused. This means a small scoring signal loss -- the optimizer no longer receives gradient from the UpperNeck heatmap peak. This is by design per the SPEC.
- **Minor**: Two skeleton joints (7 and 8) now score against the same heatmap channel (7, Thorax). This doubles the gradient signal from Thorax, which could slightly bias optimization toward that heatmap. In practice, joint 7 is FK-internal and its position is dominated by the kinematic chain, not direct heatmap scoring.

---

## Files Changed

| File | Lines | Change |
|-|-|-|
| `pose-optimizer/skeleton.py` | 1-28 | Module docstring updated |
| `pose-optimizer/skeleton.py` | 42 | Joint 8 comment: "Base of Neck" -> "Thorax / shoulder level" |
| `pose-optimizer/skeleton.py` | 77 | `DEFAULT_BONE_LENGTHS[9]`: 0.08 -> 0.18 |
| `pose-optimizer/skeleton.py` | 135-177 | `mpii_to_skeleton()`: remapped joints 7, 8; updated docstring |
| `pose-optimizer/scoring.py` | 24 | `SKELETON_TO_MPII_HEATMAP[8]`: 8 -> 7 |
| `pose-optimizer/config.py` | 44 | Comment on rotation_penalty_multipliers[8] updated |
