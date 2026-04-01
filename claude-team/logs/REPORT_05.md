# Report 05 — Remote Server Execution Prep (2026-04-01)

## Summary

Implemented all 5 phases from PLAN.md to prepare the pipeline for large-scale remote server runs. All changes are in a single commit (`6cfd593`).

---

## Phase 1: Config Schema Update (`config.py`)

**Changes:**
- `examples` default changed from `lambda: [ExampleConfig()]` to `lambda: []` (empty = auto-discover).
- Added `generate_graphs: bool = True` and `generate_video: bool = True` to `RunConfig`.

**Verified:** `RunConfig()` gives `examples=[]`, both flags default to `True`. Existing explicit-example configs unaffected since they all set `examples` explicitly.

---

## Phase 2: Auto-Discovery (`cmu_data.py`)

**Changes:**
- Added `discover_examples(data_root: str) -> list[ExampleConfig]` in `cmu_data.py`.
- Enumerates subdirectories of `data_root`, checks for `hdVideos/hd_00_00.mp4`, calls `get_video_frame_count()`, and returns `ExampleConfig(sequence=..., camera="00_00", start_frame=0, num_frames=total_frames, person_idx=0)` for each valid sequence, sorted alphabetically.
- Both `run_pipeline()` functions now call `discover_examples()` when `config.examples` is empty.

**Verified:** `discover_examples("data/panoptic-toolbox")` found 8 sequences locally with correct frame counts.

---

## Phase 3: Graph and Video Guards

**Changes in both `__init__.py` files:**
- All graph generation calls are now wrapped in `if config.generate_graphs:`. The `graphs_dir` creation was moved inside this block (no unnecessary directory creation).
- The overlay video call is now wrapped in `if config.generate_video:`.
- In `run_mediapipe/__init__.py`, the `generate_synthetic_heatmaps()` call (previously at the top of section 5 unconditionally) was moved entirely inside the `if config.generate_video:` block — it is not needed by the optimizer and was pure overhead during large runs.

---

## Phase 4: Replace print with logging

**Changes:**
- `run_motionbert.py` and `run_mediapipe.py`: added `logging.basicConfig(level=INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")` before `run_pipeline()`.
- Both `__init__.py` files: added `import logging; logger = logging.getLogger(__name__)` at module level.
- All `print()` calls converted to `logger.info()`, `logger.warning()`, or `logger.error()` as appropriate.
- **Exception:** one `print()` per example is kept for the VW-SI-MPJPE summary, in the format:
  ```
  [171204_pose1_sample_0] Det VW-SI-MPJPE: X.XX cm  Opt VW-SI-MPJPE: X.XX cm
  ```
  This always appears regardless of log level configuration.
- The 6-8 secondary metric lines (MPJPE, SI-MPJPE, MPJVE) now go to `logger.info()`.

---

## Phase 5: New run_all_config.json + Existing Config Updates

**Created:**
- `run_motionbert/run_all_config.json`: `examples=[]`, `generate_graphs=false`, `generate_video=false`, `num_steps=50`.
- `run_mediapipe/run_all_config.json`: same structure, mediapipe-appropriate optimization params.

**Updated existing configs** (all 4) to explicitly include `"generate_graphs": true, "generate_video": true`. No functional change — Pydantic defaults already covered it.

**Verified:** All 6 configs load cleanly:
```
run_motionbert/run_all_config.json:   examples=0, graphs=False, video=False
run_mediapipe/run_all_config.json:    examples=0, graphs=False, video=False
run_motionbert/run_single_config.json: examples=1, graphs=True,  video=True
run_mediapipe/run_single_config.json:  examples=1, graphs=True,  video=True
run_motionbert/run_full_config.json:   examples=25, graphs=True, video=True
run_mediapipe/run_full_config.json:    examples=25, graphs=True, video=True
```

---

## Critical Decisions for Future Review

1. **`num_frames` in auto-discovery uses `CAP_PROP_FRAME_COUNT`** — as noted in the PLAN, this OpenCV property can be unreliable for some MP4 encodings. The existing full-config configs already document that pose2 and pose3 have corrupt video frames well before the reported count. When running `run_all_config.json` on server data, sequences like these will produce a `WARNING: Only got X/Y frames` log line rather than crashing. The optimizer handles early video end gracefully. However, if a sequence is systematically corrupt early (e.g. pose3), processing will still be attempted. Consider adding a `--resume` flag to skip sequences with existing `results.json` for large runs.

2. **`examples` default change** — any code that constructs `RunConfig()` with no arguments and expects a default example will now get an empty list. The `default_single_config()` helper in `config.py` explicitly sets `examples=[ExampleConfig()]` so it is safe. The `default_full_config()` helper also sets examples explicitly. External code that calls `RunConfig()` bare should be audited.

3. **Logging configuration** — `logging.basicConfig()` is only called in the top-level entrypoints (`run_motionbert.py`, `run_mediapipe.py`). If either pipeline is imported and `run_pipeline()` called programmatically (e.g. from a notebook or test), logging will not be configured and output will be suppressed unless the caller configures logging. This is standard Python behavior but worth noting.

4. **`run_all_config.json` optimization params** — the `num_steps=50` value was taken from the PLAN spec. The single_config uses `num_steps=10` for speed. For a full server run, 50 steps is the right default but adds ~5x wall time per example compared to the local quick-test config.

---

## Issues / Red Flags

- No issues encountered during implementation.
- All config tests pass cleanly.
- Auto-discovery test found 8 sequences including sequences not in the hand-curated list (haggling1, ultimatum1, pizza1). These have calibration files but were not previously benchmarked — they may or may not have the expected GT format. When running `run_all_config.json`, failures on these sequences will be caught by the `except Exception` block in `run_pipeline()` and logged as errors without stopping the run.

---

## Suggestions for Improving the Code (from spec)

From PLAN.md (reproduced and expanded):

1. **Resumability.** Add a `--resume` flag or `resume: bool` config field that skips examples whose `results.json` already exists. Single check at the top of the loop: `if os.path.exists(os.path.join(run_dir, name, "results.json")): continue`. Critical for 50+ sequence runs where crashes or kills are likely.

2. **Parallel sequence processing.** Each sequence is independent. `concurrent.futures.ProcessPoolExecutor(max_workers=N)` would reduce wall time proportionally. Main complication: MediaPipe landmarker and MotionBERT GPU models may not be picklable. Approach: pre-initialize workers with models via initializer function, or switch to thread-based parallelism for I/O-heavy parts.

3. **Structured aggregate CSV output.** For 50+ sequences, per-sequence `results.json` files need pandas to aggregate. A final `aggregate_results.csv` (one row per sequence, metric columns) generated by `run_pipeline()` would make spreadsheet analysis trivial. The `generate_aggregate_summary()` function generates an image but no CSV.

4. **Progress file.** Write `progress.json` to the run directory as each example completes. Format: `{"completed": [{"name": ..., "det_vw_si_mpjpe": ..., "opt_vw_si_mpjpe": ...}], "failed": [...]}`. Enables `watch cat progress.json` during long runs and survives kills.

5. **`data_root` validation on startup.** Currently fails at first example if `data_root` is wrong. A check at the start of `run_pipeline()` — `if not os.path.isdir(config.data_root): raise ValueError(...)` — gives a fast, clear error before models are loaded.

6. **Frame count reliability.** `get_video_frame_count()` uses `CAP_PROP_FRAME_COUNT` which returns metadata-reported count, not actual readable frames (pose2, pose3 are known examples where these differ). A fallback — if count exceeds some sanity threshold or is 0, scan sequentially — would improve robustness. Alternatively, document that auto-discovery frame counts are upper bounds.
