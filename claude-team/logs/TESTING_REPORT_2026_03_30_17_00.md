# Testing Report: Skeleton Rework, Directory Restructure, and Isolation

**Date:** 2026-03-30 17:00
**Evaluator:** Eve (iteration 1)

---

## 1. Unit Tests

**Result: ALL 59 TESTS PASS**

```
59 passed in 1.99s
```

Breakdown:
- 15 skeleton tests (joint counts, eval joints, FK chains, mappings, backward compat)
- 7 camera tests
- 3 config tests
- 3 CMU data tests
- 6 FK tests (identity, roundtrip, batch, gradients)
- 5 evaluation/scoring tests
- 4 optimizer smoke tests (MotionBert + MediaPipe output shape, loss decrease, error on empty input)
- 12 projection tests (basic, numpy/torch consistency, roundtrips, real CMU data, scoring pathway)
- 2 heatmap tests (synthetic generation, circular for nonsquare)

---

## 2. Skeleton Correctness

**Result: PASS -- all mappings match the spec exactly.**

### Category 1 (12 direct eval joints)

All MPII, MediaPipe, and COCO19 index mappings verified against the spec table:

| Joint | MPII | MediaPipe | COCO19 | Implementation |
|-------|------|-----------|--------|----------------|
| RHip | MPII[2] | MP[24] | COCO[12] | Correct |
| RKnee | MPII[1] | MP[26] | COCO[13] | Correct |
| RAnkle | MPII[0] | MP[28] | COCO[14] | Correct |
| LHip | MPII[3] | MP[23] | COCO[6] | Correct |
| LKnee | MPII[4] | MP[25] | COCO[7] | Correct |
| LAnkle | MPII[5] | MP[27] | COCO[8] | Correct |
| RShoulder | MPII[12] | MP[12] | COCO[9] | Correct |
| RElbow | MPII[11] | MP[14] | COCO[10] | Correct |
| RWrist | MPII[10] | MP[16] | COCO[11] | Correct |
| LShoulder | MPII[13] | MP[11] | COCO[3] | Correct |
| LElbow | MPII[14] | MP[13] | COCO[4] | Correct |
| LWrist | MPII[15] | MP[15] | COCO[5] | Correct |

### Category 2 (2 synthesized)

- Pelvis: MPII[6] direct, midpoint(MP[23],MP[24]) for MediaPipe, COCO[2] direct -- Correct
- BaseOfNeck: MPII[8] direct, midpoint(MP[11],MP[12]) for MediaPipe, COCO[0] direct -- Correct

### Category 3 (mode-specific)

- MotionBert: Thorax(14) from MPII[7], HeadTop(15) from MPII[9] -- Correct
- MediaPipe: Nose(14) from MP[0] -- Correct

### No computed midpoints for Hip, Spine, or Thorax

- `mpii_to_motionbert_skeleton()`: All direct mappings, no midpoints -- Correct
- `mediapipe_to_skeleton()`: Only Pelvis and BaseOfNeck synthesized -- Correct
- `coco19_to_motionbert_skeleton()`: Thorax uses midpoint(Pelvis, Neck) for GT approximation, but Thorax is opt-only (index 14, not in eval_joints) -- Acceptable

Note: The old `mpii_to_h36m()` still computes midpoints, but it is only used internally by the MotionBERT lifting network (H36M 17-joint format), not by the optimizer skeleton.

### eval_joints

- Both skeletons: `eval_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]` (all 14 shared joints) -- Correct

### FK chains

- MotionBert: Pelvis(0) -> Thorax(14) -> BaseOfNeck(7) -- Correct (parent[14]=0, parent[7]=14)
- MediaPipe: Pelvis(0) -> BaseOfNeck(7) -- Correct (parent[7]=0, no Thorax)

---

## 3. Directory Structure

**Result: PASS**

- `run-mediapipe/`: Contains only `run_full_config.json`, `run_single_config.json` (no Python files)
- `run-motionbert/`: Contains only `run_full_config.json`, `run_single_config.json` (no Python files)
- No `mediapipe/` or `motionbert/` Python package directories exist
- `checkpoints/` exists with `pose_landmarker_lite.task`
- Old entrypoints (`mediapipe.py`, `motionbert.py`) are removed
- New entrypoints (`run_mediapipe.py`, `run_motionbert.py`) exist

---

## 4. Self-Containment

**Result: PASS**

- Zero matches for `../../`, `mediapipe-pose`, or `motionbert-pose` in any Python file
- No imports reference old `mediapipe.detect` or `motionbert.detect` package structure
- The only `from mediapipe` imports are from the Google MediaPipe library itself (in `detect_mediapipe.py`)

---

## 5. Heatmap Fix

**Result: PASS**

`scoring.py:generate_synthetic_heatmaps()` uses separate `sigma_hm_x` and `sigma_hm_y`:
```python
sigma_hm_x = sigma / sx
sigma_hm_y = sigma / sy
sq_dist = ((xx - cx_hm) / sigma_hm_x) ** 2 + ((yy - cy_hm) / sigma_hm_y) ** 2
```

Test `test_heatmap_circular_for_nonsquare` validates this for non-square images.

---

## 6. Code Review

**Result: PASS**

- All imports updated -- no references to old `mediapipe.detect` or `motionbert.detect`
- `config.py` defaults to 16-joint MotionBert multipliers; MediaPipe pipeline overrides with 15 elements
- `evaluate.py` accepts `eval_joints` parameter (defaults to `EVAL_JOINTS` from skeleton), not hardcoded

---

## 7. Smoke Test Imports

**Result: ALL PASS**

```
Skeleton OK: 16 15
FK OK
Scoring OK
Eval OK
```

---

## Issues Found

**No new blocking issues found.** The implementation matches the spec across all verification criteria.

### Minor Observations (non-blocking)

1. **Thorax midpoint in GT loading**: `coco19_to_motionbert_skeleton()` computes Thorax as midpoint(Pelvis, Neck) for GT. Since Thorax is opt-only and excluded from eval_joints, this has no impact on evaluation metrics. However, if Thorax GT is ever used for optimization loss targeting, the approximation quality should be noted.

2. **`compare.py` stale joint layout**: As Dan noted in his report, `compare.py` still uses the old 16-joint layout with hardcoded `EVAL_JOINTS = [1,2,3,4,5,6,10,11,12,13,14,15]`. This will produce incorrect results if used with outputs from the new skeleton. This was pre-existing and not in scope for this iteration, but should be addressed before the next cross-pipeline comparison.

3. **Backward-compat aliases**: The global `EVAL_JOINTS`, `PARENTS`, etc. all alias the MotionBert skeleton. Any code that uses these globals while running in MediaPipe mode would get wrong values. The pipeline code appears to pass skeleton configs explicitly, so this is safe, but the globals are a latent footgun.

---

## Verdict

**PASS.** All tests pass, all spec requirements verified, no blocking issues. The implementation is clean, well-structured, and correctly implements the 3-category joint mapping system with mode-aware skeletons. Ready for integration testing with real pipeline runs.
