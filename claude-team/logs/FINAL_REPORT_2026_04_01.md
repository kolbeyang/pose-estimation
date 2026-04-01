# Final Report — Remote Server Execution Prep (2026-04-01)

**Spec:** `claude-team/specs/2026-04-01-12-26.md`
**Commits:** `6cfd593`, `b4f25b5`, `207e2ba` (3 commits on top of `bbc6915`)

---

## 1. What Was Built

Five changes to prepare the pipeline for large-scale remote server evaluation runs:

1. **Config schema additions** (`config.py`) — `generate_graphs: bool = True`, `generate_video: bool = True` added to `RunConfig`; `examples` default changed from `[ExampleConfig()]` to `[]` (empty = auto-discover).
2. **Auto-discovery of sequences** (`cmu_data.py`) — `discover_examples(data_root)` enumerates subdirectories of `data_root`, checks for `hdVideos/hd_00_00.mp4`, and returns `ExampleConfig` objects covering the full video at camera `00_00`, `person_idx=0`. Called by both `run_pipeline()` functions when `config.examples` is empty.
3. **Graph and video output guards** (both `__init__.py`) — All graph generation and the overlay video call are wrapped in `if config.generate_graphs:` / `if config.generate_video:` blocks. In the MediaPipe pipeline, `generate_synthetic_heatmaps()` was moved inside the video guard — it was previously called unconditionally even when the video was never generated.
4. **Python logging throughout** (`run_motionbert.py`, `run_mediapipe.py`, both `__init__.py`, `run_motionbert/detect.py`, `run_mediapipe/detect.py`, `optimize/__init__.py`, `overlay_video.py`) — `logging.basicConfig(level=INFO)` in entrypoints; `logger = logging.getLogger(__name__)` in all submodules; all `print()` calls converted to `logger.info/warning/error/exception`. One `print()` per example is kept for the always-visible VW-SI-MPJPE summary line.
5. **New `run_all_config.json` files** for both pipelines — `examples=[]`, `generate_graphs=false`, `generate_video=false`, `num_steps=50`. All four existing configs updated to explicitly include the new boolean fields (`true`).

---

## 2. How It Works

### Auto-discovery data flow

```
run_pipeline(config)
  → if not config.examples:
        examples = discover_examples(config.data_root)   # cmu_data.py
  → for example in examples:
        process_example(example, config, ...)
```

`discover_examples` (`pose-optimizer/cmu_data.py`, added at end of file):
- `sorted(os.listdir(data_root))` — alphabetical for determinism
- Checks `os.path.exists(get_video_path(data_root, seq, "00_00"))` — skips sequences without the expected camera
- `get_video_frame_count(video_path)` via `cv2.CAP_PROP_FRAME_COUNT` — sets `num_frames`
- Returns `ExampleConfig(sequence=seq, camera="00_00", start_frame=0, num_frames=total_frames, person_idx=0)`

### Graph/video guard wiring

Both `process_example()` functions (`run_motionbert/__init__.py` and `run_mediapipe/__init__.py`):

```python
if config.generate_graphs:
    os.makedirs(graphs_dir, exist_ok=True)
    generate_trajectory_graphs(...)
    generate_loss_curve(...)
    # ... remaining graph calls ...
    generate_summary(...)

if config.generate_video:
    # MediaPipe only: synthetic heatmap generation is here, not above
    overlay_path = os.path.join(example_dir, "overlay_video.mp4")
    generate_overlay_video(...)
```

`results.json` and `trajectories.json` are always saved regardless of the flags.

### Logging configuration

Entrypoints (`run_motionbert.py`, `run_mediapipe.py`) before `run_pipeline()`:
```python
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
```

Per-example summary (always visible, outside the GT guard block):
```python
# run_motionbert/__init__.py:304 and run_mediapipe/__init__.py:285
det_str = f"{det_vw * 100:.2f} cm" if det_vw is not None else "N/A"
opt_str = f"{opt_vw * 100:.2f} cm" if opt_vw is not None else "N/A"
print(f"[{name}] Det VW-SI-MPJPE: {det_str}  Opt VW-SI-MPJPE: {opt_str}")
```

Exception handling uses `logger.exception(...)` (includes traceback through the logging framework, not raw stderr).

---

## 3. Critical Decisions

**`CAP_PROP_FRAME_COUNT` for auto-discovery frame counts.** OpenCV's `CAP_PROP_FRAME_COUNT` reads MP4 metadata rather than scanning frames. Known CMU sequences (pose2, pose3) already have corrupt/truncated video files where actual readable frames are fewer than the metadata count. Auto-discovery will set `num_frames` to the metadata value; the pipeline produces a `WARNING: Only got X/Y frames` log line rather than crashing, and the optimizer handles early video end. For large server runs with unknown sequences, corrupt videos will be attempted and logged as warnings — not failures. Worth adding a minimum viable frame check (e.g. skip sequences with fewer than 30 readable frames) if this proves noisy.

**`examples` default changed from one example to empty list.** Any code that constructs `RunConfig()` with no arguments and expects a default example will now get auto-discovery behavior. The `default_single_config()` and `default_full_config()` helpers in `config.py` both set `examples` explicitly, so they are safe. External callers should be audited if they exist.

**Logging only configured in entrypoints.** If either pipeline's `run_pipeline()` is called programmatically (notebook, test harness), logging will not be configured and output will be suppressed. Standard Python behavior but worth noting for any future programmatic use.

**`num_steps=50` in `run_all_config.json`.** This is 5x the `run_single_config.json` setting (`num_steps=10`). Appropriate for a proper evaluation run but significantly increases per-example wall time. Callers should be aware this is intentional.

---

## 4. Issues / Contradictions with Spec

**Spec testing protocol vs. local dataset ordering.** The spec says "wait for one example to be processed." Auto-discovery sorts alphabetically; on the local dataset the first sequence is `160226_haggling1` (14,591 frames, ~4,864 at 10fps). This is a multi-minute run that cannot be observed in a quick smoke test. The spec's testing expectation implicitly assumed a shorter sequence would be first. Testing was validated against `run_single_config.json` instead for the end-to-end check. Not a code issue, but the test protocol assumption does not hold for the local dataset.

**Spec says "most of the current messages should be INFO."** The initial implementation (commit `6cfd593`) applied logging only to the orchestration layer; the submodules (`detect.py`, `optimize/__init__.py`, `overlay_video.py`) still used `print()`. This was caught by QA and fixed in the second commit (`b4f25b5`). The spec requirement is now fully met — no remaining bare `print()` calls in the pipeline except the intentional VW-SI-MPJPE summary line.

**`optimize()` `verbose` parameter.** The spec said to convert prints to logging but the initial implementation did not pass `verbose=False` to `optimize()`, causing the optimizer step-progress lines (gated on `verbose=True` default) to print unformatted. Fixed in the logging refactor commit.

---

## 5. Subagent Performance Notes

**Three-commit resolution pattern worked well.** The implementation commit (`6cfd593`) was followed by a QA-identified issue (incomplete submodule logging) that required a second commit (`b4f25b5`), followed by nitpick-identified issues (stale imports, traceback bypass, silent summary) fixed in a third commit (`207e2ba`). This is expected; the issues found were real.

**Developer missed submodule scope.** The spec said "most of the current messages should be INFO messages" without specifying files. Developer Dan interpreted this as "the `__init__.py` orchestration layer" and missed `detect.py`, `optimize/__init__.py`, and `overlay_video.py`. The plan explicitly listed specific files, but the developer did not follow through on all of them in the first pass. Recommendation: when the plan lists submodule files as targets for a cross-cutting change (logging, imports, etc.), require the developer to confirm each file was touched before marking the phase done.

**QA caught the right issue immediately.** Eve's testing report correctly identified the incomplete logging scope, estimated the severity accurately (medium for server runs, not a correctness bug), and provided exact file/line references for the fix. QA process functioned well.

**Nitpick caught three real issues.** The stale import issue was Pyright-flagged and directly introduced by the refactor. The `traceback.print_exc()` bypass was a partial fix (added `logger.error` but left the `print_exc`). The conditional summary was a spec compliance gap. All three were non-trivial and worth catching. Nathan's methodology (reading the spec line-by-line, then diffs, then Pyright investigation) is effective and should be kept.

**Recommendation for TOM.md:** Add a step in the developer task template requiring explicit verification of "all files listed in the plan as targets for this phase" before marking a phase complete. Specifically for cross-cutting changes (logging, import audits), the developer should produce a grep/list of remaining instances and confirm zero are left.

---

## 6. Suggestions for Improving the Code

Given the goals of speed and structural robustness for large server runs:

**Resumability (highest priority).** A crash halfway through a 50-sequence run loses all progress. A `resume: bool` config field or `--resume` CLI flag that skips examples whose `results.json` already exists in the output directory would fix this. Implementation: single `os.path.exists` check at the top of the `for example in examples` loop. The output path for each example is deterministic given sequence name and `start_frame`, so a resumed run can reuse the same `run_dir`.

**Aggregate CSV output.** Per-sequence `results.json` files require scripting to aggregate. A `aggregate_results.csv` (one row per sequence: `sequence, start_frame, det_vw_si_mpjpe, opt_vw_si_mpjpe, ...`) written by `run_pipeline()` at the end (or appended after each example) would make spreadsheet analysis immediate. `generate_aggregate_summary()` currently writes only an image.

**Progress file.** Write a `progress.json` to the run directory after each example completes: `{"completed": [{"name": ..., "det_vw_si_mpjpe": ..., "opt_vw_si_mpjpe": ...}], "failed": [...]}`. Enables `watch cat output/.../progress.json` during long runs and survives kills. Trivial to implement alongside the per-example summary print.

**`data_root` validation on startup.** Currently fails at the first example if `data_root` is wrong (after models are loaded). `if not os.path.isdir(config.data_root): raise ValueError(...)` at the start of `run_pipeline()` gives a fast, clear error before any expensive model loading. Similarly, `discover_examples()` silently returns `[]` for a missing `data_root`, logging only "Found 0 sequences" — a dedicated check with an explicit error message would prevent confusion.

**Frame count reliability.** `get_video_frame_count()` uses `CAP_PROP_FRAME_COUNT` which reads MP4 metadata. Known issue on pose2/pose3 locally. A fallback that scans actual frames if the metadata count exceeds a sanity threshold (e.g. > 30,000) or if count is 0 would improve robustness. Alternatively, document that auto-discovery frame counts are upper bounds and the pipeline handles early video end via the existing `WARNING: Only got X/Y frames` path.

**Parallel sequence processing.** Each sequence is independent. `concurrent.futures.ProcessPoolExecutor(max_workers=2-4)` would reduce wall time proportionally. Main constraint: GPU models (MotionBERT) and MediaPipe's landmarker may not be picklable across processes. Pre-initializing worker processes with models via the `initializer` argument to `ProcessPoolExecutor` is the cleanest path. Not trivial but high value for large evaluation runs.

**`trajectories.json` size for long sequences.** This file is always written regardless of flags. For a 4,864-frame sequence (16 joints × 3 coords × 4,864 × 2 models), it can be several MB. For a 50-sequence run this may be acceptable, but if disk space or aggregation speed becomes an issue, a `generate_trajectories: bool = True` flag following the same pattern as `generate_graphs` would allow disabling it for large runs.
