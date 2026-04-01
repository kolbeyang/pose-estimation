# Nitpick Report 05 — Remote Server Execution Prep (2026-04-01)

**Spec:** `claude-team/specs/2026-04-01-12-26.md`
**Commits reviewed:** `6cfd593` (feat) and `b4f25b5` (logging refactor), both on top of `bbc6915`
**Reviewer:** Nit-pick Nathan

---

## Methodology

1. Read the spec line by line.
2. Ran `git diff bbc6915 HEAD --name-only` to enumerate all changed files.
3. Read full diffs for each changed file.
4. Verified spec requirements against actual code.
5. Investigated Pyright-flagged issues specifically called out in the task.
6. Checked pre-existing issues against their scope.

---

## Spec Requirements Checklist

### Data Loading

- [x] `examples` defaults to `[]` — empty list triggers auto-discovery. (`config.py` line 61)
- [x] Auto-discovery enumerates subdirs of `data_root`, checks for `hdVideos/hd_00_00.mp4`. (`cmu_data.py:discover_examples`)
- [x] Auto-discovery uses `start_frame=0`, `num_frames=total_frames`, `person_idx=0`, `camera="00_00"`. All correct.
- [x] Both `run_pipeline()` functions call `discover_examples()` when `config.examples` is empty.
- [x] When `examples` is non-empty, existing behavior is preserved.

### Data Saving

- [x] `output_dir` config field is `str | None = None`. (`config.py` line 63)
- [x] If `output_dir` is set (non-null, non-empty), it is used as `run_dir` directly.
- [x] If `output_dir` is null, defaults to `output/motionbert_{timestamp}` or `output/mediapipe_{timestamp}`.
- [x] Both `run_pipeline()` functions implement this logic identically.

### generate_graphs / generate_video Flags

- [x] `generate_graphs: bool = True` added to `RunConfig`. (`config.py` line 72)
- [x] `generate_video: bool = True` added to `RunConfig`. (`config.py` line 73)
- [x] All graph calls in both `__init__.py` files are guarded with `if config.generate_graphs:`.
- [x] Overlay video call is guarded with `if config.generate_video:`.
- [x] `generate_synthetic_heatmaps()` in MediaPipe pipeline moved inside `if config.generate_video:` block — no longer called when video is disabled. This is correct and optimal.
- [x] `graphs_dir` creation moved inside the `if config.generate_graphs:` block. No unnecessary directory creation.

### New run_all_config.json Files

- [x] `run_motionbert/run_all_config.json` created with `examples=[]`, `generate_graphs=false`, `generate_video=false`.
- [x] `run_mediapipe/run_all_config.json` created with `examples=[]`, `generate_graphs=false`, `generate_video=false`.
- [x] Both have `target_fps: 10.0` and `num_steps: 50` (as noted in REPORT_05).
- [x] Existing `run_single_config.json` and `run_full_config.json` for both pipelines updated with explicit `generate_graphs: true, generate_video: true`. No functional change (Pydantic defaults already covered it), but the configs are now self-documenting.

### Logging

- [x] `logging.basicConfig()` added to both top-level entrypoints (`run_motionbert.py`, `run_mediapipe.py`).
- [x] Both `__init__.py` files use `logger = logging.getLogger(__name__)`.
- [x] Most `print()` calls converted to `logger.info()` / `logger.warning()` / `logger.error()`.
- [x] Per-example VW-SI-MPJPE summary remains a bare `print()` — intentional, always-visible.
- [x] Submodule `print()` calls in `run_motionbert/detect.py`, `run_mediapipe/detect.py`, `optimize/__init__.py`, `overlay_video.py` all converted (Iteration 2 fix, commit `b4f25b5`).

---

## Issues Found

### Issue 1 — Stale imports in both `__init__.py` files (Pyright-flagged, INTRODUCED BY THIS SPEC)

**File:** `pose-optimizer/run_motionbert/__init__.py` lines 29, 32, 34
**File:** `pose-optimizer/run_mediapipe/__init__.py` lines 29, 32, 34

```python
from config import RunConfig, load_config          # load_config unused in __init__.py
from evaluate import (
    evaluate,
    compute_visibility_weights,                    # unused in __init__.py
    mpjpe_per_joint,
    mpjve_per_joint,                               # unused in __init__.py
    root_relative,
)
```

**Status:** INTRODUCED BY THIS SPEC. The refactor added `discover_examples` and `verbose=False` changes but did not audit the pre-existing import list. All three names appear only on their import lines — no call sites in either `__init__.py`.

- `load_config` is used in the top-level entrypoints (`run_motionbert.py`, `run_mediapipe.py`) which import it directly from `config`. It is redundantly imported in `__init__.py` but never called there.
- `compute_visibility_weights` is used only in `evaluate.py` internally, not exposed to callers.
- `mpjve_per_joint` is genuinely unused — there is no per-joint velocity error metric reported by the pipeline (only aggregate `mpjve` is computed via `mpjve_per_frame()`).

**Verify once fixed:** `pyright pose-optimizer/run_motionbert/__init__.py pose-optimizer/run_mediapipe/__init__.py` should show zero "not accessed" warnings for these names. Alternatively: `grep -n "load_config\|compute_visibility_weights\|mpjve_per_joint" pose-optimizer/run_motionbert/__init__.py` should show each only on its import line — confirm none remain.

---

### Issue 2 — `traceback.print_exc()` bypasses logging (minor, INTRODUCED BY THIS SPEC)

**File:** `pose-optimizer/run_motionbert/__init__.py` line 467
**File:** `pose-optimizer/run_mediapipe/__init__.py` line 465

```python
except Exception as e:
    logger.error("ERROR processing %s_%d: %s", example.sequence, example.start_frame, e)
    import traceback
    traceback.print_exc()   # writes directly to stderr, bypasses logging
    continue
```

**Status:** INTRODUCED BY THIS SPEC (the `logger.error()` call was added, but `traceback.print_exc()` was retained unchanged). The spec says "most of the current messages should be INFO messages" — exception tracebacks writing to raw stderr will always appear on the server even when log level is set to WARNING or ERROR. For a large batch run this is acceptable (stack traces are essential debugging information), but it is inconsistent with the logging-based output system.

The fix would be `logger.exception("ERROR processing %s_%d", example.sequence, example.start_frame)` which logs the full traceback at ERROR level via the logging system.

**Severity:** Low. In practice, server-side logging infrastructure often captures stderr alongside stdout, and exceptions are rare. Not blocking.

**Verify once fixed:** Replace `logger.error(...)` + `traceback.print_exc()` with `logger.exception(...)`. Confirm that `traceback.print_exc()` and `import traceback` lines are removed.

---

### Issue 3 — Per-example summary not printed when no GT is available

**File:** `pose-optimizer/run_motionbert/__init__.py` lines 298-303
**File:** `pose-optimizer/run_mediapipe/__init__.py` lines 279-284

The spec says:
> After each example is processed I would like to see the following:
> - Base Model (Mediapipe/Motionbert) VW-SI-MPJPE
> - Optimizer VW-SI-MPJPE

The implementation:

```python
if gt_indices:   # <-- only enters here if GT is available
    ...
    print(
        f"[{name}] "
        f"Det VW-SI-MPJPE: {metrics['det_vw_si_mpjpe'] * 100:.2f} cm  "
        f"Opt VW-SI-MPJPE: {metrics['opt_vw_si_mpjpe'] * 100:.2f} cm"
    )
else:
    logger.info("    No ground truth available for evaluation.")  # silent at WARNING level
```

If an auto-discovered sequence has no matching ground truth JSON files, the per-example summary is silently skipped. The user sees only an INFO-level log line that disappears when log level is WARNING. This means for sequences with no GT, the operator has no confirmation that the example completed successfully.

**Severity:** Medium for the "no GT" case. In practice, all CMU Panoptic sequences have GT, so this only affects unusual sequences. However, for a server run, an operator watching for per-example progress lines would see gaps without explanation.

**Verify once fixed:** A fallback print like `print(f"[{name}] No GT available — skipped metrics")` should appear in the `else` branch, so every processed example produces at least one summary line.

---

## Investigation: Experiment Sweep Files (Pyright-Flagged)

The task asked about `experiment/rotation_penalty_sweep.py`, `experiment/num_steps_sweep.py`, `experiment/quick_steps_sweep.py` and whether the refactor broke them.

**Finding:** These files do not exist. They were deleted in commit `f6c06a9` ("chore: clean up temporary sweep scripts"), which predates this spec's commits by several commits. The files were temporary scripts created during Phase 2 optimization sweeps and intentionally removed before this spec was implemented.

**Conclusion:** This is NOT an issue introduced by this spec. The Pyright warnings about a missing `models` argument in those files are moot — the files are gone. No action required.

---

## Pre-existing Issues (Outside This Spec's Scope)

The following are pre-existing issues documented in TODO.md that were not introduced by this spec and are clearly outside its scope. Listed for completeness but NOT added to TODO:

- MPJPE degradation with solvePnP on pose1 (Phase 2 issue, from previous iteration)
- Minor depth clip asymmetry between MotionBERT and MediaPipe solvePnP paths (Phase 2 issue)
- Missing type annotation on `landmarker` param in `run_mediapipe/__init__.py:109` (Phase 3 issue)
- `landmarker.close()` not in `try/finally` (Phase 3 issue)
- `or` vs `and` in fallback guards in `run_motionbert/detect.py` (Phase 3 issue)

---

## TODO.md Additions

Two new items added to TODO.md (see below).

---

## Verdict

**NEEDS FIXES**

Issue 1 (stale imports) is a clean code hygiene issue introduced by this spec that Pyright explicitly flags. It should be fixed. Issues 2 and 3 are lower severity but represent spec compliance gaps (logging contract, per-example output contract).

Blocking: **Issue 1** (stale imports — directly called out in the task brief as Pyright-flagged).
Non-blocking but should fix: **Issue 3** (missing summary line when no GT).
Low priority: **Issue 2** (`traceback.print_exc()` bypassing logging).

---

## Final Approval Check

**Reviewer:** Nit-pick Nathan
**Iteration:** 3 (Developer Dan's fixes to Issues 1, 2, 3 from the report above)

### Verification of the Three Fixes

**Fix 1 — Stale imports removed**

Searched both `run_motionbert/__init__.py` and `run_mediapipe/__init__.py` for `load_config`, `compute_visibility_weights`, and `mpjve_per_joint`. Zero matches in both files. The import blocks now read only:

```python
from config import RunConfig
from evaluate import (
    evaluate,
    mpjpe_per_joint,
    root_relative,
)
```

All three stale names are gone. CONFIRMED FIXED.

**Fix 2 — `traceback.print_exc()` replaced by `logger.exception()`**

Searched both files for `traceback`. Zero matches. The exception handlers now read:

```python
except Exception:
    logger.exception("ERROR processing %s_%d", example.sequence, example.start_frame)
    continue
```

`import traceback` is also absent from both files. CONFIRMED FIXED.

**Fix 3 — Per-example summary fires for every example**

The summary `print()` is now unconditional and placed outside the `if len(gt_indices) >= 2:` block in both files. It uses `.get()` with a `None` sentinel and renders `N/A` when no GT metrics were computed:

```python
# Per-example summary (always printed regardless of GT availability)
det_vw = metrics.get("det_vw_si_mpjpe")
opt_vw = metrics.get("opt_vw_si_mpjpe")
det_str = f"{det_vw * 100:.2f} cm" if det_vw is not None else "N/A"
opt_str = f"{opt_vw * 100:.2f} cm" if opt_vw is not None else "N/A"
print(f"[{name}] Det VW-SI-MPJPE: {det_str}  Opt VW-SI-MPJPE: {opt_str}")
```

Both `run_motionbert/__init__.py` (line 304) and `run_mediapipe/__init__.py` (line 285) have this pattern. CONFIRMED FIXED.

### Quick Final Spec Sweep

Rechecked the remaining spec requirements against current code:

- `data_root`, `output_dir`, `generate_graphs`, `generate_video` all present in `config.py` with correct defaults. PASS.
- `run_all_config.json` exists for both pipelines, with `examples: []`, `generate_graphs: false`, `generate_video: false`. PASS.
- Both pipelines call `discover_examples()` when `config.examples` is empty. PASS.
- Graph and video generation gated on their respective config flags. PASS.
- `logger.basicConfig` in entrypoints; `logging.getLogger(__name__)` in both `__init__.py` modules. PASS.
- No remaining `print()` calls in submodules that bypass logging (confirmed clean by NITPICK_REPORT_05 Iteration 2). PASS.

No new issues identified in the final sweep.

### Verdict

**APPROVE**

All three issues from the previous NITPICK_REPORT_05 are correctly fixed. The implementation is clean, spec-compliant, and consistent across both pipelines. No new issues were introduced by the fixes.
