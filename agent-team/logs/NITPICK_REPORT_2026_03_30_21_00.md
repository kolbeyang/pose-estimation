# Nitpick Report: Comparison Script & Output Review

**Date:** 2026-03-30
**Reviewer:** Nit-pick Nathan (QA)

## Spec Reminder

> "Run all 25 examples for both mediapipe and motionbert. I want to see the comparison graphs essentially answering the question in a multitude of ways, 'Which is better Motionbert or Mediapipe'."

## Verdict

The work is solid. 25/25 examples ran for both pipelines, 9 PNG graphs + summary.json produced. The script is clean and straightforward (~450 lines). A few issues below, ranging from "should fix" to "nice to have."

---

## Issues Found

### ISSUE 1 (Medium): Per-joint graph uses raw MPJPE, not scale-invariant

The `per_joint_comparison.png` graph uses raw MPJPE (not SI-MPJPE or VW-SI-MPJPE). This makes MotionBert's ankle errors look catastrophic (80-90 cm) when they are largely a depth/scale estimation artifact. The primary metric everywhere else is VW-SI-MPJPE. This graph could mislead the viewer into thinking MotionBert's ankles are 3x worse than they actually are after scale correction.

**Fix:** The per-joint data stored in results.json is raw MPJPE only (no SI variant exists per-joint). Either: (a) add a note to the graph title saying "(Raw MPJPE, not scale-corrected)" so the viewer isn't misled, or (b) compute SI-MPJPE per-joint in evaluate.py and use that instead.

### ISSUE 2 (Low): summary.json lacks a plain-English conclusion

The user wants to know "which is better." The summary.json has raw numbers but no `"conclusion"` or `"winner"` field. A human reading the JSON has to interpret the numbers themselves. Adding a simple `"conclusion": "MediaPipe wins on accuracy (14/25 examples, lower mean error). MotionBert wins on smoothness (22/25 examples, lower velocity error)."` would directly answer the question.

### ISSUE 3 (Low): Scatter plot annotations are hard to read

The scatter plot annotates each point with just the frame number suffix (e.g., "1000", "8000"). With 25 points clustered at low values and a few outliers, the annotations overlap heavily in the bottom-left corner. The graph is readable for the outliers but the dense cluster is a mess. Consider either: (a) removing annotations entirely (the overall pattern is what matters), or (b) only annotating outliers above a threshold.

### ISSUE 4 (Low): No "headline" graph

There are 9 graphs that each tell part of the story, but no single graph that a user could look at and immediately get the answer. The `win_loss.png` is closest but it shows 4 separate metrics. A simple one-chart summary (e.g., a radar/spider chart, or a single bar chart of "MediaPipe vs MotionBert" on the 2-3 most important metrics with clear "WINNER" annotation) would make the answer jump off the screen.

### ISSUE 5 (Cosmetic): Optimization improvement graph is confusing

The `optimization_improvement.png` shows many negative bars, meaning optimization made things worse for some examples. This is surprising and potentially alarming. The graph title says "positive = optimization helped" but the visual impression is "optimization hurts half the time." This needs either: (a) a note explaining why (short clips, over-smoothing penalties), or (b) a rethink of whether this graph answers the core question at all. It is about optimization quality, not pipeline comparison.

---

## Checklist: Does the comparison cover enough angles?

| Angle | Covered? | Graph(s) |
|---|---|---|
| Overall accuracy (lower error) | YES | aggregate_metrics, per_example_vw_si_mpjpe |
| Per-example breakdown (who wins more) | YES | win_loss, scatter_mpjpe |
| Per-joint breakdown (which joints differ) | YES (with caveat, see ISSUE 1) | per_joint_comparison |
| Temporal quality (velocity/smoothness) | YES | velocity_comparison |
| Distribution/variance (consistency) | YES | error_distribution (box plot) |
| Optimization benefit | YES | optimization_improvement |

All requested comparison dimensions are present.

## Code Quality: compare.py

- Clean, well-structured, ~450 lines
- No unnecessary complexity
- Proper axis labels, titles, legends on all graphs
- Consistent color scheme (orange = MotionBert, blue = MediaPipe) across all graphs
- Uses matplotlib Agg backend correctly for headless rendering
- Handles missing data gracefully (None checks)
- No issues with the code itself

## Summary

The core deliverable is met: 25 examples, both pipelines, 9 comparison graphs, summary data. The graphs collectively answer "which is better" from multiple angles. The main actionable issue is ISSUE 1 (per-joint graph could mislead due to raw vs scale-invariant metric). The rest are polish items.
