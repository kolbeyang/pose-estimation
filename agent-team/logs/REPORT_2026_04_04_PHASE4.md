# Phase 4 Report: Joint Evaluation and Graphs

## Summary

Implemented all Phase 4 deliverables: cross-pipeline comparison graphs (config-controlled), per-joint VW-SI metric breakdowns, and two experiment scripts.

## Changes

### evaluate.py
- Added `vw_si_mpjpe_per_joint()`: per-joint VW-SI-MPJPE using global optimal scale then per-joint weighted mean.
- Added `vw_si_mpjve_per_joint()`: per-joint VW-SI-MPJVE using global optimal scale then per-joint weighted velocity error.

### graphs.py
- Added consistent color scheme constants: GT=Blue, MB Raw=Light Orange, MB Opt=Orange, MP Raw=Light Red, MP Opt=Red.
- `generate_cross_pipeline_per_joint_position()`: grouped bar chart, 4 bars per joint, Y=VW-SI-MPJPE (cm).
- `generate_cross_pipeline_per_joint_velocity()`: grouped bar chart, 4 bars per joint, Y=VW-SI-MPJVE (cm/frame).
- `generate_cross_pipeline_metrics_comparison()`: grouped bar chart, 4 bars per metric group (VW-SI-MPJPE, VW-SI-MPJVE), averaged across all examples.

### main.py
- `_evaluate_and_collect_metrics()` now computes `det_vw_si_mpjpe_per_joint`, `opt_vw_si_mpjpe_per_joint`, `det_vw_si_mpjve_per_joint`, `opt_vw_si_mpjve_per_joint` and stores them in metrics.
- `_save_results()` now persists these per-joint VW-SI breakdowns in results.json.
- Cross-pipeline graph generation (Section 6) is now fully wired up, controlled by `config.graphs` toggles. Logs a warning if `graphs` is set but only one pipeline is configured.

### experiment/elbow_smoothness.py
- Takes MB and MP trajectories.json files as input.
- Plots 15 lines: 5 trajectories (GT, MB Raw, MB Opt, MP Raw, MP Opt) x 3 dimensions (X, Y, Z).
- Trajectories differentiated by color, dimensions by line style (solid, wide dash, narrow dash).
- Configurable joint name (default: RElbow), FPS, output directory.

### experiment/bone_length_variation.py
- Takes MB and MP trajectories.json files as input.
- Plots 5 lines: GT, MB Raw, MB Opt, MP Raw, MP Opt forearm bone length over time.
- Optimized lines are perfectly horizontal (std=0.0000) confirming shared bone length parameter works correctly.
- Configurable side (L/R), FPS, output directory. Prints statistics table.

## Testing

Ran `configs/both-local-single.json` end-to-end. All three cross-pipeline graphs generated successfully in `cross_pipeline_graphs/` directory. Verified graph colors, labels, axes, and groupings visually.

Ran both experiment scripts with R and L variants:
- `elbow_smoothness.py` with RElbow and LElbow
- `bone_length_variation.py` with R and L forearm

All outputs confirmed correct.

## Metrics (single example: 171204_pose1_sample_0)

| Pipeline | Det VW-SI-MPJPE | Opt VW-SI-MPJPE |
|----------|-----------------|-----------------|
| MotionBERT | 23.43 cm | 18.86 cm |
| MediaPipe | 15.66 cm | 15.86 cm |

## Commit
`f3db7e3` on branch `refactor-2026-04-04`
