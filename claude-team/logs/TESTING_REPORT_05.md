# Testing Report 05 — Remote Server Execution Prep (2026-04-01)

**Evaluator:** Eve (QA)
**Spec:** `claude-team/specs/2026-04-01-12-26.md`
**Dev report:** `claude-team/logs/REPORT_05.md`

---

## Step 1: Code Review

### config.py

All spec-required fields are present and correctly typed:
- `examples: list[ExampleConfig] = Field(default_factory=lambda: [])` — empty default ✓
- `data_root: str = "data/panoptic-toolbox"` ✓
- `output_dir: str | None = None` ✓
- `generate_graphs: bool = True` ✓
- `generate_video: bool = True` ✓
- `target_fps`, `optimization`, `sh_batch_size`, `motionbert_conf_threshold` all present ✓

Minor observation: `model_config = {"json_schema_extra": {"examples": []}}` — the key name `"examples"` here matches the field name on the model. This is valid Pydantic usage (it sets the OpenAPI schema example), not a field collision.

### cmu_data.py — discover_examples()

Implementation is correct:
- Enumerates subdirectories of `data_root`, checks for `hdVideos/hd_00_00.mp4` ✓
- Returns `ExampleConfig(sequence=..., camera="00_00", start_frame=0, num_frames=total_frames, person_idx=0)` ✓
- Sorted alphabetically ✓
- Returns empty list if `data_root` does not exist (graceful) ✓
- Local test: found 8 sequences correctly

One observation: `discover_examples` does not use logging (no `import logging`). If `data_root` is wrong, it silently returns an empty list, and then `run_pipeline()` logs "Found 0 sequences" and exits cleanly. This is acceptable but could be confusing. The dev noted this as a suggestion for `data_root` validation in their report.

### run_motionbert/__init__.py

- All `print()` calls replaced with `logger.info/warning/error()` in the orchestration layer ✓
- Graph guard `if config.generate_graphs:` wraps all graph generation including `graphs_dir` creation ✓
- Video guard `if config.generate_video:` wraps overlay video generation ✓
- Auto-discovery: `if not examples: examples = discover_examples(config.data_root)` ✓
- VW-SI-MPJPE summary `print()` preserved (correct per spec — always appears) ✓
- `output_dir` handling: `if config.output_dir: run_dir = config.output_dir` ✓

**Bug found: missing `landmarker` type annotation** (pre-existing TODO): `run_mediapipe/__init__.py:113` — `landmarker` parameter has no type annotation. Low severity, not new in this phase.

### run_mediapipe/__init__.py

Same structure as MotionBERT. All spec requirements implemented.

**Bug found: generate_synthetic_heatmaps moved correctly** — now inside `if config.generate_video:` block, not called unconditionally. This is correct per the developer's Phase 3 notes.

### Incomplete Logging Refactor (NEW BUG)

The spec says: "I would prefer to use proper logging and most of the current messages should be INFO messages."

The logging refactor was only applied to the orchestration layer (`__init__.py` files). Several submodule files still use bare `print()` that cannot be silenced by log level:

| File | Lines | Impact |
|------|-------|--------|
| `run_motionbert/detect.py` | 66, 83, 89, 147, 167, 314, 332, 422, 501, 519, 523, 532, 539, 549, 550 | Prints model loading status, YOLO detections, SH/MotionBERT progress |
| `run_mediapipe/detect.py` | 54 | "Loaded MediaPipe PoseLandmarker" |
| `optimize/__init__.py` | 87, 119, 163, 200 | FK init, roundtrip error, optimizer step progress |
| `overlay_video.py` | 245, 322 | Video save status |

The `optimize/__init__.py` prints are gated on `verbose=True` (the default), but the pipeline `__init__.py` files call `optimize()` without passing `verbose=False`. So optimizer step-progress lines always print.

**Observed in run_mediapipe single config output:**
```
  Loaded MediaPipe PoseLandmarker         ← detect.py print()
  Initializing FK parameters...           ← optimize/__init__.py print()
    FK roundtrip error: ...               ← optimize/__init__.py print()
  Optimising 34 frames for 50 steps...   ← optimize/__init__.py print()
    Step 0/50 loss=...                    ← optimize/__init__.py print()
  Saved overlay video: ...               ← overlay_video.py print()
```

These lines appear mixed with properly-formatted INFO log lines, creating inconsistent output. For large server runs with 50+ sequences, this is significant noise.

---

## Step 2: Config Smoke Test

All 6 JSON configs load without error:

```
run_motionbert/run_all_config.json:   examples=0, graphs=False, video=False ✓
run_mediapipe/run_all_config.json:    examples=0, graphs=False, video=False ✓
run_motionbert/run_single_config.json: examples=1, graphs=True, video=True ✓
run_mediapipe/run_single_config.json:  examples=1, graphs=True, video=True ✓
run_motionbert/run_full_config.json:   examples=25, graphs=True, video=True ✓
run_mediapipe/run_full_config.json:    examples=25, graphs=True, video=True ✓
```

Command verified: `uv run python -c "from config import RunConfig; import json; RunConfig(**json.load(open('config.json')))"` for each config.

---

## Step 3: Existing Config (Single Example) Test

**Command:** `uv run python run_mediapipe.py run_mediapipe/run_single_config.json`

**Result: PASS — completed without error in ~8 seconds.**

Full output confirms:
- Logging format correct: `HH:MM:SS [INFO] message` ✓
- All 7 pipeline stages log correctly ✓
- Secondary metrics go to INFO: `Det MPJPE: 14.07 cm`, `Opt MPJPE: 13.79 cm`, etc. ✓
- VW-SI-MPJPE summary printed (unformatted, per spec): `[171204_pose1_sample_0] Det VW-SI-MPJPE: 10.58 cm  Opt VW-SI-MPJPE: 10.07 cm` ✓
- Graphs generated (graphs_dir created, summary.png copied) ✓
- Overlay video generated ✓
- Results saved to timestamped output directory ✓

---

## Step 4: run_all_config Test (Kick Off, Observe, Kill)

**Command:** `uv run python run_mediapipe.py run_mediapipe/run_all_config.json`

**Initial logging behavior verified:**
```
HH:MM:SS [INFO] No examples specified — auto-discovering sequences in data/panoptic-toolbox
HH:MM:SS [INFO] Found 8 sequences
HH:MM:SS [INFO] ============================================================
HH:MM:SS [INFO] MediaPipe Pose Estimation Pipeline
HH:MM:SS [INFO] 8 examples to process
HH:MM:SS [INFO] Run: output/mediapipe_2026_04_01_HH_MM
HH:MM:SS [INFO] ============================================================
```

Auto-discovery and logging format are correct.

**Critical observation: First sequence too large to observe in test.**

The 8 locally-discovered sequences sorted alphabetically begin with `160226_haggling1` (14,591 frames → 4,864 frames to process at 10 fps). This sequence takes far longer than 60 seconds to complete. The VW-SI-MPJPE summary line for the first example could **not** be observed in a reasonable kill cycle.

The `generate_graphs=false` and `generate_video=false` flags in `run_all_config.json` are correctly applied (confirmed by code review — graph/video blocks gated correctly). However, their effectiveness for speed cannot be confirmed from the run because the bottleneck is frame extraction and MediaPipe inference, not graph generation.

**MotionBERT run_all_config:** Same result — first example is 160226_haggling1, cannot observe VW-SI-MPJPE line.

**Workaround for spec testing:** The spec says "wait for one example to be processed." To actually observe this, `run_all_config.json` should temporarily have a known-short sequence in `examples`, OR the test should be run against `run_single_config.json` (confirmed working) and then `run_all_config.json` kicked off with the understanding that the first summary line will take 10+ minutes on local data.

---

## Issues Found

### Blocking for Nathan

None. All spec requirements are implemented correctly in the orchestration layer. The pipeline runs end-to-end, configs load, auto-discovery works, and logging format matches spec.

### Non-Blocking Issues (added to TODO)

**Issue 1: Incomplete logging refactor (submodule print() calls)**

Severity: Medium for large runs (noise in output), not a correctness bug.

`run_motionbert/detect.py`, `run_mediapipe/detect.py`, `optimize/__init__.py`, and `overlay_video.py` still use bare `print()`. For a 50-sequence server run, each example will produce ~20 extra `print()` lines (model loading, step progress) that cannot be suppressed. The optimizer step progress is controlled by `verbose=True` default in `optimize()`, but the pipelines don't pass `verbose=False`.

Fix: In both `run_motionbert/__init__.py` and `run_mediapipe/__init__.py`, change the `optimize(...)` call to pass `verbose=False`. Then separately convert `detect.py` print calls to logging.

**Issue 2: First auto-discovered example is very large (testing concern)**

Severity: Low for production (it will eventually complete), Medium for spec testing protocol.

Auto-discovery sorts alphabetically. On local data, the first sequence is `160226_haggling1` (14,591 frames). The testing instruction "wait for one example to be processed" cannot be fulfilled in a quick smoke test. This is a dataset property, not a code bug. On the remote server, sequences may be sorted differently.

---

## Notes to Developer

1. **Logging refactor is ~80% complete.** The orchestration layer (`__init__.py` files) is clean. The submodules are not. For a large server run, you'll see interleaved `HH:MM:SS [INFO]` lines and bare-print lines. The most impactful quick fix: pass `verbose=False` to `optimize()` in both pipeline files — this silences the 5-6 step-progress prints per example.

2. **The `run_all_config` generates very large intermediate data.** 8 sequences × up to 4864 frames × MediaPipe inference is a multi-hour run locally. The `generate_graphs=false, generate_video=false` flags are correctly implemented, but `trajectories.json` is always saved (no guard). For a 4864-frame sequence, `trajectories.json` will be large (16 joints × 3 coords × 4864 frames × 2 models × JSON = several MB per example). This is probably fine but worth knowing.

3. **Auto-discovery on the remote server vs locally.** `discover_examples()` finds only sequences with `hdVideos/hd_00_00.mp4`. If the server has sequences with a different camera arrangement, they will be skipped. The dev report already noted this.

4. **`data_root` uses a relative path default** (`"data/panoptic-toolbox"` not `"./data/panoptic-toolbox"`). This is functionally equivalent but means it resolves relative to the current working directory at runtime, not relative to the script file. Since the scripts must be run from `pose-optimizer/`, this is correct. Note that if called programmatically from a different cwd, discovery will fail silently (returns empty list, which then triggers "Found 0 sequences" log).

5. **The developer's REPORT_05.md suggestions for resumability and aggregate CSV are valuable.** For a 50+ sequence server run, a crashed mid-run job currently loses all progress. A simple `results.json` existence check at the start of the loop would be the highest-value quick addition.

---

## Verdict

**Ready to proceed to Nathan: YES, with caveats.**

The core spec requirements are implemented and functional:
- Config schema ✓
- Auto-discovery ✓
- Graph/video guards ✓
- Logging (orchestration layer) ✓
- VW-SI-MPJPE summary line ✓
- New `run_all_config.json` files ✓
- Existing configs updated and working ✓

The incomplete logging refactor (submodule `print()` calls) is noisy but not a correctness issue. Nathan will see some unformatted output lines alongside the INFO messages. The pipeline still runs correctly and produces correct results. File the submodule print-to-logging conversion as a cleanup task for the next cycle.
