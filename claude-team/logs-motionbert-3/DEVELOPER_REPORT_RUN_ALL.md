# Developer Report: Full Pipeline Run (All 10 Examples)

**Date:** 2026-03-16
**Run ID:** `motionbert-run-20260316-185953`
**Output:** `motionbert-pose/training_runs/motionbert-run-20260316-185953/`

## Summary

All 10/10 CMU Panoptic examples processed successfully with zero errors or warnings.

## Per-Example Results (all values in cm)

| Example | Det MPJPE | Opt MPJPE | Improv | Det P-MPJPE | Opt P-MPJPE | Det MPJVE | Opt MPJVE |
|---|---|---|---|---|---|---|---|
| 171204_pose1_sample_0 | 30.98 | 30.44 | +0.54 | 28.57 | 28.42 | 0.94 | 1.05 |
| 171204_pose2_200 | 39.60 | 39.38 | +0.22 | 30.15 | 29.92 | 4.80 | 4.49 |
| 171204_pose2_5000 | 18.65 | 18.13 | +0.52 | 21.36 | 21.32 | 1.09 | 1.20 |
| 171204_pose2_15000 | 24.95 | 24.57 | +0.38 | 27.03 | 26.85 | 2.50 | 2.37 |
| 171204_pose3_200 | 56.54 | 55.93 | +0.61 | 36.54 | 36.54 | 3.87 | 3.47 |
| 171204_pose3_4000 | 15.85 | 15.44 | +0.42 | 20.33 | 20.22 | 0.51 | 0.48 |
| 160422_ultimatum1_200 | 50.89 | 50.98 | -0.09 | 34.91 | 34.92 | 10.91 | 11.00 |
| 160422_ultimatum1_10000 | 57.66 | 57.29 | +0.37 | 26.34 | 26.26 | 3.27 | 3.31 |
| 171204_pose2_10000 | 16.86 | 16.61 | +0.25 | 18.17 | 18.29 | 1.44 | 1.56 |
| 171204_pose2_25000 | 17.93 | 17.76 | +0.17 | 20.89 | 20.59 | 2.01 | 2.14 |
| **MEAN** | **32.99** | **32.65** | **+0.34** | **26.43** | **26.33** | **3.13** | **3.11** |

## Aggregate Metrics

- **Mean Det MPJPE:** 32.99 cm
- **Mean Opt MPJPE:** 32.65 cm (optimization improved by +0.34 cm on average)
- **Mean Det P-MPJPE:** 26.43 cm
- **Mean Opt P-MPJPE:** 26.33 cm
- **Mean Det MPJVE:** 3.13 cm/frame
- **Mean Opt MPJVE:** 3.11 cm/frame

## Observations

1. **Optimization helps modestly.** FK optimization improved MPJPE in 9/10 examples, with the largest gain on 171204_pose3_200 (+0.61 cm) and the only regression on 160422_ultimatum1_200 (-0.09 cm).

2. **Best examples** are 171204_pose3_4000 (15.44 cm), 171204_pose2_10000 (16.61 cm), and 171204_pose2_25000 (17.76 cm) -- all in the pose2/pose3 sequences where the person is relatively close to camera.

3. **Worst examples** are 160422_ultimatum1_10000 (57.29 cm) and 171204_pose3_200 (55.93 cm). The ultimatum sequences have multiple people and more challenging poses. The large 2D MPJPE (1107-1209 px) on these suggests the 2D detector struggles.

4. **Ankle error dominates.** Comparing MPJPE vs MPJPE-no-ankles shows ankles add 10-20 cm of error on average (e.g., 171204_pose1_sample: 30.44 cm full vs 13.85 cm without ankles).

5. **P-MPJPE vs MPJPE gap** is large on the worst examples (e.g., 160422_ultimatum1_10000: 57.29 vs 26.26), indicating significant global alignment error that Procrustes corrects.

6. **No errors or warnings** during the entire run.

## Generated Outputs

Per example (10 directories under `graphs/`):
- Overlay video (MP4) with heatmaps + skeleton projections
- Per-joint trajectory graphs (X, Y, Z for all 17 joints)
- Per-frame MPJPE and MPJVE plots
- Per-joint error bar charts
- Bone length comparison graphs
- Loss curves
- Summary dashboard

Aggregate (under `graphs/`):
- `aggregate_mpjpe.png`
- `aggregate_p_mpjpe.png`
- `aggregate_mpjve.png`

Predictions (10 JSON files under `predictions/`):
- Full per-frame 3D predictions, ground truth, 2D detections, visibility, metrics
