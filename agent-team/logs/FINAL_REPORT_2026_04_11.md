# Final Report: Joint Alignment Revision (2026-04-11)

## Summary

Remapped the "Neck" joint (skeleton index 8) from anatomical neck to thorax/shoulder level across the MPII/MotionBERT pipeline. GT and MediaPipe mappings were already correct and unchanged. The anatomical neck (SH[8] UpperNeck) is now unused in the skeleton but still used internally for Nose synthesis.

## What changed (3 files, 5 edits)

### `skeleton.py`

**`mpii_to_skeleton()`** (line 161-163):
```python
# Before:
skel[7] = keypoints_mpii[7]   # Spine = Thorax direct
skel[8] = keypoints_mpii[8]   # Neck = UpperNeck direct

# After:
skel[8] = keypoints_mpii[7]   # Neck = Thorax (shoulder level)
skel[7] = (skel[0] + skel[8]) / 2.0  # Spine = midpoint(Pelvis, Neck)
```

Nose synthesis (line 165) unchanged — still uses raw `keypoints_mpii[8]` (UpperNeck) and `keypoints_mpii[9]` (HeadTop).

**`DEFAULT_BONE_LENGTHS[9]`** (line 77): `0.08` → `0.18` (Neck→Nose distance increased since Neck is now lower at thorax level).

**`REST_DIRECTIONS[9]`** (line 98): `[0, -0.5, -0.866]` → `[0, -0.866, -0.5]` (Neck→Nose direction is now mostly upward).

### `scoring.py`

**`SKELETON_TO_MPII_HEATMAP[8]`** (line ~24): `8` → `7` (joint 8 now scores against Thorax heatmap instead of UpperNeck).

### `config.py`

Comment update only on `rotation_penalty_multipliers[8]`.

## What did NOT change (confirmed)

- `coco19_to_skeleton()` — GT already mapped COCO[0] to joint 8
- `mediapipe_to_skeleton()` — MP already mapped midpoint(shoulders) to joint 8
- `PARENTS`, `BONES`, `EVAL_JOINTS` — hierarchy unchanged, shoulders still connect to joint 8
- `fk.py`, `evaluate.py`, `overlay_video.py`, `optimize/`, `main.py` — no changes needed

## Metrics comparison

| Pipeline / Example | Baseline VW-SI-MPJPE | New VW-SI-MPJPE | Change |
|-|-|-|-|
| MB / pose1_360 | 27.31 cm | 25.27 cm | **-7.5%** |
| MB / piano4_166 | 11.85 cm | 11.25 cm | **-5.0%** |
| MP / pose1_360 | 12.73 cm | 12.79 cm | +0.4% |
| MP / piano4_166 | 20.40 cm | 20.22 cm | -0.9% |

MotionBERT improved on both examples. Eve's review found this is because the new mapping (joint 8 = Thorax) better matches MotionBERT's own `coco2h36m()` definition where joint 8 = midpoint(shoulders). MediaPipe unchanged as expected.

## Critical decisions for review

1. **Joint 8 in GT still maps to COCO[0] "Neck"** which is anatomical neck, while for MB it's now Thorax (lower). This means GT and MB "Neck" are at slightly different physical positions. The SPEC explicitly listed this mapping so it was implemented as specified.

2. **MPII heatmap channel 8 (UpperNeck) is now unused** by any skeleton joint for scoring. Channel 7 (Thorax) is used by both joints 7 and 8. Minor scoring signal loss.

3. **Nose synthesis still uses raw MPII[8]** (UpperNeck), not the skeleton's new Neck (joint 8 = Thorax). This is correct — the Nose position should be synthesized from the anatomical reference points, not from the remapped skeleton joint.

## Issues with the SPEC

None blocking. The SPEC was clear and the "only change is the Neck joint" claim was confirmed accurate.

## Subagent notes

- Architect Alex produced a thorough plan that correctly identified the minimal change set
- Developer Dan implemented cleanly in one pass, no iteration needed
- Evaluator Eve caught the positive insight about MotionBERT's `coco2h36m()` alignment
- Nit-pick Nathan found 4 items (3 stale labels, 1 pre-existing broken import) — all fixed by Tom directly
