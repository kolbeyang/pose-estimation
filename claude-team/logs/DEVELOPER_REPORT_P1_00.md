# Developer Report: Phase 1, Iteration 0

## What was implemented

Added a limb length over time graph showing 4 arm limb segments (R Shoulder-Elbow, R Elbow-Wrist, L Shoulder-Elbow, L Elbow-Wrist) for 3 sources (GT in blue, detector in green, optimized in red), totaling 12 lines. Line style encodes which limb segment; color encodes which source.

The graph is generated both as a standalone PNG per example and as a miniature subplot in the summary grid (axes[3,3], which was previously empty).

## Files changed

- `/motionbert-pose/graphs.py` -- Added `generate_limb_length_graph()` function (~65 lines) after `generate_bone_lengths_graph`. Also added a miniature limb length plot to the summary grid at axes[3,3].
- `/motionbert-pose/main.py` -- Added `generate_limb_length_graph` to imports and added a call to it in `process_example()` after the per-frame MPJVE graph generation.

## Commands run

```bash
cd motionbert-pose && uv run python -c "..." # Single-example test with EXAMPLES[0]
```

Result: Success. Both `limb_lengths.png` and `summary.png` generated correctly.

## Sanity checks (visual verification)

1. GT limb lengths (blue) are nearly constant across frames -- correct, real arms don't change length.
2. Optimized (red) lines are flatter than detector (green) lines -- confirmed, FK optimization enforces shared bone lengths.
3. All limb lengths fall in ~0.24-0.34m range -- reasonable for adult arm segments.
4. All 12 lines visible in legend.
5. Colors match spec: blue=GT, green=Det, red=Opt.

## Decisions made

- **Summary grid legend placement**: Used `bbox_to_anchor=(1.0, 1.0)` with `loc="upper left"` and `fontsize=5` for the miniature plot in the summary grid. The abbreviated limb names (e.g., "R Sh-El" instead of "R Shoulder-Elbow") keep the legend compact.
- **NaN handling for missing GT**: Used `np.nan` for frames where `gt_3d[i]` is `None`, which matplotlib renders as gaps in the line.

## Concerns

- The summary grid's bottom-right subplot with 12 legend entries is quite dense at the miniature scale. The `fontsize=5` legend is legible but tight. The standalone `limb_lengths.png` is much clearer.
- The legend in the summary mini-plot extends slightly outside the subplot area due to `bbox_to_anchor`. This is handled by `bbox_inches="tight"` in `_save()`.

## Deviations from plan

None. All steps followed exactly as specified.
