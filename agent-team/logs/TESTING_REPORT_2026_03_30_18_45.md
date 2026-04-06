# Testing Report: 2026-03-30 18:45

**Evaluator:** Eve (QA)
**Scope:** Verify Developer Dan's 2026-03-30 18:30 implementation report

## Summary

**76 tests pass. 1 critical runtime bug found. 2 minor issues.**

The core logic (skeleton, scoring, FK, optimization, projection) is solid. The directory renames and h36m removal are complete. However, the MediaPipe pipeline is broken at runtime due to a module naming collision that was already present before the rename but has now become a hard crash.

---

## 1. Unit Tests

**Result: PASS (76/76)**

```
76 passed in 2.15s
```

All 4 test files pass: `test_phase1.py`, `test_phase2.py`, `test_phase3.py`, `test_projection.py` (19 new tests).

---

## 2. Pyright Issues Verification

| # | Issue | Verdict | Details |
|---|-------|---------|---------|
| 1 | `run_mediapipe/detect.py:55` -- `_get_mediapipe` not defined | **FALSE POSITIVE** | The function was correctly deleted. Line 55 is now `options = vision.PoseLandmarkerOptions(...)`. The old `_get_mediapipe()` hack is gone. |
| 2 | `run_mediapipe/detect.py:100-102` -- `mediapipe_to_h36m` not defined | **FALSE POSITIVE** | Correctly renamed to `mediapipe_to_skeleton` and `mediapipe_visibility_to_skeleton`. Import on line 21 matches usage on lines 98-100. |
| 3 | `run_mediapipe/__init__.py:45` -- `detect_poses` and `mediapipe_3d_to_camera` unknown | **FALSE POSITIVE** | Both functions exist in `run_mediapipe/detect.py` (lines 40 and 105). The import is correct. |
| 4 | `run_mediapipe/__init__.py:49` -- `EVAL_JOINT_NAMES` and `DEFAULT_BONE_LENGTHS` unused | **TRUE -- minor** | Both are imported but never referenced in the file body. Same issue in `run_motionbert/__init__.py:48`. |
| 5 | `motionbert.py:20` -- Module is not callable | **FALSE POSITIVE (Pyright-only)** | `run_pipeline` is a function in `run_motionbert/__init__.py:354`. Runtime import works. Pyright likely cannot resolve through the `__init__.py` re-export. |

---

## 3. Smoke Test Imports

| Import | Result |
|--------|--------|
| `from run_mediapipe.detect import detect_poses` | **FAIL -- circular import** |
| `from run_motionbert.detect import detect_poses` | **PASS** |
| `from skeleton import JOINT_NAMES, EVAL_JOINTS` | **PASS** |
| `from scoring import generate_synthetic_heatmaps` | **PASS** |

### Critical Bug: `mediapipe.py` shadows pip package

When Python resolves `import mediapipe as mp` in `run_mediapipe/detect.py`, it finds `pose-optimizer/mediapipe.py` (the CLI entrypoint) instead of the `mediapipe` pip package. This causes:

1. `mediapipe.py` imports `from run_mediapipe import run_pipeline`
2. `run_mediapipe/__init__.py` imports `from run_mediapipe.detect import detect_poses`
3. `run_mediapipe/detect.py` does `import mediapipe as mp` -- finds `mediapipe.py` (circular!)
4. Crash: `ModuleNotFoundError: No module named 'mediapipe.tasks'; 'mediapipe' is not a package`

**Root cause:** The old `_get_mediapipe()` sys.path hack in `detect.py` was working around this exact problem by manipulating `sys.modules`. Dan correctly removed the hack but did not rename `mediapipe.py`, so the underlying naming collision is now exposed.

**Fix:** Rename `mediapipe.py` to a non-colliding name (e.g., `run_mediapipe_cli.py`).

Note: `motionbert.py` does NOT have this problem because there is no pip package called `motionbert` -- the MotionBERT code is vendored in `run_motionbert/external/`.

---

## 4. Remaining h36m References

**Result: PASS (acceptable)**

Only 4 occurrences, all in checkpoint filename references:
- `run_motionbert/setup_models.py:50-51` -- download URL and local path for `motionbert_lite_h36m.bin`
- `run_motionbert/setup_models.py:54` -- upstream checkpoint path containing `h36m`
- `run_motionbert/detect.py:278` -- local checkpoint path

These are upstream external filenames that cannot be renamed. Acceptable per Dan's report.

---

## 5. External Path References

| Pattern | Result |
|---------|--------|
| `../../` | **ZERO matches** |
| `mediapipe-pose` | **ZERO matches** |
| `motionbert-pose` | **ZERO matches** |

Self-containment requirement satisfied.

---

## 6. compare.py Deletion

**Result: PASS** -- File does not exist.

---

## 7. Directory Structure

**Result: PASS**

| Check | Status |
|-------|--------|
| `run_mediapipe/` exists with Python files | Yes: `__init__.py`, `detect.py` |
| `run_motionbert/` exists with Python files | Yes: `__init__.py`, `detect.py`, `setup_models.py` |
| Old `mediapipe/` directory gone | Yes |
| Old `motionbert/` directory gone | Yes |
| No flattened modules at root | Correct -- no `pipeline_mediapipe.py`, `detect_mediapipe.py`, etc. |

---

## Issues Added to TODO

1. **CRITICAL: mediapipe.py shadows mediapipe package** -- Entire MediaPipe pipeline is broken at import time.
2. **Minor: Unused imports** -- `EVAL_JOINT_NAMES` and `DEFAULT_BONE_LENGTHS` in both `run_mediapipe/__init__.py` and `run_motionbert/__init__.py`.
3. **Known: h36m in checkpoint filename** -- Upstream filename, no action needed.
