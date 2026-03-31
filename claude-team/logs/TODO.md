# TODO — Spec 2026-03-31-02-31

## Phase 1 Issues (from QA evaluation)

- [x] Unused `Any` import in `config.py`: Line 5 imports `from typing import Any` but `Any` is never used in the file. Remove the import. Verify: `uv run python -c "import py_compile; py_compile.compile('pose-optimizer/config.py', doraise=True)"` and grep for `Any` usage.

- [x] Inaccurate format tag in `graphs.py:generate_per_joint_error_bar`: The `det_per_joint` and `opt_per_joint` args are tagged `[3D:SKELETON_16_EVAL]` but they are lists of scalar errors (one per eval joint), not (14, 3) 3D positions. The tag should indicate they follow SKELETON_16_EVAL joint ordering without implying 3D shape. Verify: read the docstring and confirm the tag reflects the actual data type (list of floats indexed by eval joint order).

- [x] Stale "12 eval joints" label in `graphs.py`: Lines 133-134 say "12 eval joints" in the chart title, but `EVAL_JOINTS` has 14 joints (12 Cat1 + 2 Cat2). This is a pre-existing bug, not introduced by Phase 1, but should be fixed. Verify: check that chart titles say "14 eval joints".

- [x] Diagnostics issues from spec are stale/non-applicable: The three Pyright issues listed in the evaluation spec (`detect_mediapipe.py:17 MEDIAPIPE_SKELETON`, `evaluate.py:21 MOTIONBERT_SKELETON`, `config.py:5 Any`) do not match current file structure. `detect_mediapipe.py` does not exist (renamed to `run_mediapipe/detect.py`), `evaluate.py:21` imports `EVAL_JOINTS` not `MOTIONBERT_SKELETON`, and the `Any` import issue in config.py is pre-existing. Only the `Any` import is actionable (see first item above).
