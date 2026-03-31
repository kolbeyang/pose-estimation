# Testing Report: Phase 1 QA Evaluation

**Evaluator:** Evaluator Eve
**Date:** 2026-03-31
**Scope:** Phase 1 (Docstrings + Skeleton Format Tags) from spec `2026-03-31-02-31.md`
**Commits reviewed:** `3774924`, `b2df7c8`

## 1. Smoke Test

### Compilation
All 14 in-scope files pass `py_compile.compile(doraise=True)`:
- `skeleton.py`, `scoring.py`, `fk.py`, `camera.py`, `config.py`, `cmu_data.py`, `evaluate.py`, `overlay_video.py`, `graphs.py`, `optimize/__init__.py`, `run_motionbert/__init__.py`, `run_motionbert/detect.py`, `run_mediapipe/__init__.py`, `run_mediapipe/detect.py`

**Result: PASS**

### Core Imports
```
from run_motionbert import run_pipeline  # OK
from run_mediapipe import run_pipeline   # OK
from optimize import optimize            # OK
from evaluate import evaluate            # OK
```

Note: The evaluation spec listed `optimize_sequence` and `evaluate_sequence` as import targets but the actual exports are `optimize` and `evaluate`. This is a spec-side mismatch, not a code issue.

**Result: PASS**

## 2. Completeness Check

### Docstrings
- All public functions in the 9 directly-importable modules (`skeleton`, `camera`, `config`, `cmu_data`, `evaluate`, `scoring`, `fk`, `overlay_video`, `graphs`) confirmed to have docstrings via `inspect.getmembers` + `__doc__` check.
- All functions (public and private) in the 5 subpackage files (`run_motionbert/__init__.py`, `run_motionbert/detect.py`, `run_mediapipe/__init__.py`, `run_mediapipe/detect.py`, `optimize/__init__.py`) confirmed to have docstrings via AST parsing.

**Result: PASS** -- every function has a docstring.

### Format Tags
Tag counts per file (matches for `[DIM:JOINTS]` patterns including inline comments and docstrings):

| File | Tag Count |
|------|-----------|
| `skeleton.py` | 17 |
| `run_motionbert/detect.py` | 63 |
| `run_motionbert/__init__.py` | 17 |
| `run_mediapipe/detect.py` | 20 |
| `run_mediapipe/__init__.py` | 17 |
| `optimize/__init__.py` | 12 |
| `evaluate.py` | 13 |
| `scoring.py` | 22 |
| `fk.py` | 10 |
| `camera.py` | 11 |
| `cmu_data.py` | 7 |
| `overlay_video.py` | 14 |
| `graphs.py` | 19 |
| `config.py` | 1 (`[SKELETON_16]` in comment) |

All pose-carrying functions have format tags in their Args/Returns sections. Inline tags are present at key conversion points in the pipeline entrypoints.

**Result: PASS**

## 3. Correctness Check

### Tag Accuracy Verified Against Code
Verified the following critical mappings:

1. **`skeleton.py` JOINT_NAMES** has exactly 16 entries (Pelvis through RWrist) -- matches `[SKELETON_16]` tags.

2. **`EVAL_JOINTS = [0,1,2,3,4,5,6,8,10,11,12,13,14,15]`** -- 14 joints, excludes Spine(7) and Head(9). Matches `[SKELETON_16_EVAL]` tags.

3. **`mpii_to_skeleton`** correctly tagged `[2D:MPII_16]` -> `[2D:MPII_17]` (16,D) -> (17,D).

4. **`coco19_to_skeleton`** correctly tagged `[3D:COCO19]` (19,3) -> `[3D:SKELETON_16]` (16,3).

5. **`mediapipe_to_skeleton`** correctly tagged with `[2D:MEDIAPIPE_33]`/`[3D:MEDIAPIPE_33]` -> `[*:SKELETON_16]`.

6. **`strip_head_joint`** correctly tagged `[*:SKELETON_16]` output (generic over dimensionality).

7. **`detect_poses` in `run_motionbert/detect.py`** return tuple annotations match actual return types and conversions in the function body.

8. **`motionbert_to_camera_space`** input/output tags match actual shapes (`[3D:SKELETON_16]` normalized -> camera-space).

9. **`mediapipe_3d_to_camera`** tags match actual shapes and data flow.

10. **`optimize()`** correctly distinguishes `[HEATMAP:MPII_16]` (MotionBERT) vs `[HEATMAP:SKELETON_16]` (MediaPipe) in both docstring and inline comments.

11. **`forward_kinematics`/`forward_kinematics_batch`** correctly tagged `[FK_PARAMS]` inputs and `[3D:SKELETON_16]` outputs.

12. **`evaluate()`** correctly shows `[3D:SKELETON_16]` inputs sliced to `[3D:SKELETON_16_EVAL]` internally.

13. **All tags used** are from the approved vocabulary in the plan (cross-referenced against the plan's table).

### Issues Found

1. **Minor: `graphs.py:generate_per_joint_error_bar`** -- `det_per_joint` tagged `[3D:SKELETON_16_EVAL]` but these are scalar error lists (floats), not 3D position arrays. The tag correctly indicates they follow eval joint ordering but incorrectly implies `(14, 3)` 3D data. Should use a notation like "indexed by SKELETON_16_EVAL order" without the `[3D:...]` prefix.

**Result: PASS with 1 minor issue** (logged to TODO.md)

## 4. Diagnostics Issues

The three Pyright issues from the evaluation spec were investigated:

1. **`detect_mediapipe.py:17` -- `MEDIAPIPE_SKELETON` unknown import**: File `detect_mediapipe.py` does not exist. The actual file is `run_mediapipe/detect.py` and it does not import `MEDIAPIPE_SKELETON`. This issue is **stale/non-applicable**.

2. **`evaluate.py:21` -- `MOTIONBERT_SKELETON` unknown import**: Line 21 of `evaluate.py` is `from skeleton import EVAL_JOINTS`, not `MOTIONBERT_SKELETON`. This issue is **stale/non-applicable**.

3. **`config.py:5` -- `Any` is not accessed**: Confirmed. `from typing import Any` on line 5 is unused. This is a **pre-existing issue** not introduced by Phase 1. Logged to TODO.md.

**Result: Only the `Any` import is actionable; the other two are stale references to old file names/symbols.**

## 5. Code Review

### No Logic Changes
Reviewed the full diff of both commits (`3774924` and `b2df7c8`) against the pre-Phase-1 codebase. All changes are:
- New or expanded docstrings
- Inline `# [DIM:JOINTS]` comments
- Minor docstring wording improvements (e.g., "hip-centered" -> "Pelvis-centered")
- No function signatures, logic, imports, or control flow were modified.

**Result: PASS** -- docstrings and comments only.

### Pre-existing Issue Noted
`graphs.py` lines 133-134 have chart title "12 eval joints" but there are 14 eval joints. This is pre-existing, not introduced by Dan. Logged to TODO.md for future fix.

## Summary

| Check | Result |
|-------|--------|
| Compilation | PASS |
| Core imports | PASS |
| Docstring completeness | PASS |
| Format tag completeness | PASS |
| Format tag correctness | PASS (1 minor issue) |
| Diagnostics issues | 3/3 stale or pre-existing |
| No logic changes | PASS |

**Overall verdict: PASS.** Phase 1 work is complete and correct. Two minor issues and one pre-existing bug logged to TODO.md. None are blockers.
