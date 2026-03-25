# Developer Report — Round 6 Type Fix

## What was implemented

Fixed 3 Pyright type errors in `motionbert-pose/sweep.py` where `**base` dict unpacking caused Pyright to infer that `float` values could be assigned to the `heatmap_blur_schedule: list[tuple[float, float]] | None` parameter of `SweepConfig.__init__`.

## Approach

Replaced the `base` dict + `**base` unpacking pattern with local variables (`pw`, `rs`, `aw`) passed as explicit keyword arguments. This gives Pyright full visibility into each argument's type.

A `dict[str, float]` annotation was attempted first but does not help — Pyright still sees `**dict[str, float]` as potentially matching any `str`-keyed parameter, including `heatmap_blur_schedule`.

## Files changed

- `motionbert-pose/sweep.py` — Replaced `base = dict(...)` + `**base` with explicit keyword args in `get_phase1_2_configs()`.

## Commands run

```
cd motionbert-pose && uv run pyright sweep.py
```

Result: The 3 `**base`-related errors are resolved. One pre-existing error remains on line 81 (`ndarray` vs `tuple[int, int]` for `resolution` parameter in `from_panoptic_calibration`) — this is unrelated to the task scope.

## Decisions made

- **Explicit kwargs over TypedDict**: Could have used a `TypedDict` subclass to type the base dict precisely, but explicit keyword args are simpler, more readable, and avoid adding a new type just for internal use in one function. The repetition (`position_penalty_weight=pw, rotation_penalty_scalar=rs, init_anchor_weight=aw`) is kept on a single line per call to minimize visual noise.

## Concerns

- The pre-existing line 81 error (`ndarray` assigned to `tuple[int, int]`) should be fixed separately.

## Deviations from plan

None.
