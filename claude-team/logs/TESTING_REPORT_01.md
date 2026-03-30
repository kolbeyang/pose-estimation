# Testing Report 01 -- Full Pipeline Runs (2026-03-29)

## Overview

Ran both MotionBert and MediaPipe full pipelines with 25 examples each.
Both pipelines completed without crashing. Each processed 16/25 examples; the same 9 examples failed in both pipelines due to data issues (not code bugs).

## Output Directories

- **MotionBert**: `pose-optimizer/output/motionbert_2026_03_29_20_41/`
- **MediaPipe**: `pose-optimizer/output/mediapipe_2026_03_29_20_48/`

## Results Summary

### Successful Examples (16/25)

Both pipelines produced results for these 16 examples. Of those, 15 have full metrics (all 9 metrics, all finite values). One example (`171204_pose1_30000`) ran but has empty metrics because ground truth was unavailable for all 50 frames.

### Failed Examples (9/25)

All 9 failures are due to "Need at least 2 frames" -- the video extraction returned 0 frames, meaning the requested frame range exceeds the available video data.

| Example | Failure Reason |
|---|---|
| `171204_pose3_8000` | 0 frames extracted (video too short for start_frame=8000) |
| `171204_pose1_38000` | 0 frames extracted (video too short for start_frame=38000) |
| `171204_pose2_30000` | 0 frames extracted (video too short for start_frame=30000) |
| `171204_pose3_6000` | 0 frames extracted (video too short for start_frame=6000) |
| `female_example_01_350` | 0 frames extracted (video too short or missing) |
| `female_example_01_1000` | 0 frames extracted |
| `female_example_01_2000` | 0 frames extracted |
| `female_example_01_3000` | 0 frames extracted |
| `female_example_01_4000` | 0 frames extracted |

All `female_example_01` examples failed (5/5), suggesting the video file may be missing or empty.
Several high-`start_frame` examples from `pose1`, `pose2`, `pose3` also failed, indicating those start frames exceed the video duration.

### Empty Metrics (1/25)

- `171204_pose1_30000`: Pipeline ran successfully (optimization completed) but ground truth was available for 0/50 frames, so no evaluation metrics could be computed.

## Metric Averages (15 examples with valid metrics)

| Metric | MotionBert | MediaPipe |
|---|---|---|
| mpjpe | 0.3222 | 0.1664 |
| p_mpjpe | 0.2920 | 0.1050 |
| si_mpjpe | 0.3070 | 0.1593 |
| vw_mpjpe | 0.1851 | 0.1242 |
| vw_si_mpjpe | 0.1666 | 0.1203 |
| mpjve | 0.0362 | 0.0462 |
| si_mpjve | 0.0303 | 0.0490 |
| vw_mpjve | 0.0313 | 0.0398 |
| vw_si_mpjve | 0.0280 | 0.0394 |

Units are in meters (based on the magnitudes).

### Per-Example Metrics

#### MotionBert (optimized)

| Example | MPJPE | SI-MPJPE | VW-MPJPE | MPJVE |
|---|---|---|---|---|
| 171204_pose1_10000 | 0.3688 | 0.3406 | 0.1390 | 0.0474 |
| 171204_pose1_14000 | 0.4164 | 0.4148 | 0.2850 | 0.0361 |
| 171204_pose1_18000 | 0.2583 | 0.2571 | 0.0824 | 0.0053 |
| 171204_pose1_22000 | 0.1730 | 0.1728 | 0.0848 | 0.0126 |
| 171204_pose1_5000 | 0.4092 | 0.3958 | 0.2708 | 0.0668 |
| 171204_pose1_sample_0 | 0.5120 | 0.4928 | 0.1932 | 0.0354 |
| 171204_pose2_10000 | 0.2096 | 0.2064 | 0.0917 | 0.0336 |
| 171204_pose2_15000 | 0.4968 | 0.4746 | 0.1376 | 0.0713 |
| 171204_pose2_200 | 0.3579 | 0.3552 | 0.1877 | 0.0731 |
| 171204_pose2_20000 | 0.2929 | 0.2810 | 0.0941 | 0.0261 |
| 171204_pose2_25000 | 0.1878 | 0.1866 | 0.0895 | 0.0426 |
| 171204_pose2_5000 | 0.2181 | 0.2127 | 0.1045 | 0.0214 |
| 171204_pose3_200 | 0.6107 | 0.4942 | 0.8402 | 0.0631 |
| 171204_pose3_2000 | 0.1498 | 0.1497 | 0.0974 | 0.0045 |
| 171204_pose3_4000 | 0.1717 | 0.1710 | 0.0789 | 0.0035 |

#### MediaPipe (optimized)

| Example | MPJPE | SI-MPJPE | VW-MPJPE | MPJVE |
|---|---|---|---|---|
| 171204_pose1_10000 | 0.1357 | 0.1342 | 0.0976 | 0.0659 |
| 171204_pose1_14000 | 0.3736 | 0.3736 | 0.2363 | 0.0923 |
| 171204_pose1_18000 | 0.1244 | 0.1214 | 0.1100 | 0.0061 |
| 171204_pose1_22000 | 0.1111 | 0.1073 | 0.0854 | 0.0317 |
| 171204_pose1_5000 | 0.1447 | 0.1427 | 0.1438 | 0.0336 |
| 171204_pose1_sample_0 | 0.1509 | 0.1202 | 0.1297 | 0.0371 |
| 171204_pose2_10000 | 0.1504 | 0.1417 | 0.1291 | 0.0514 |
| 171204_pose2_15000 | 0.1414 | 0.1223 | 0.1109 | 0.0545 |
| 171204_pose2_200 | 0.2053 | 0.2052 | 0.1124 | 0.0795 |
| 171204_pose2_20000 | 0.1448 | 0.1423 | 0.1148 | 0.0405 |
| 171204_pose2_25000 | 0.1606 | 0.1575 | 0.1447 | 0.0417 |
| 171204_pose2_5000 | 0.1550 | 0.1376 | 0.1294 | 0.0319 |
| 171204_pose3_200 | 0.2096 | 0.2077 | 0.1415 | 0.0550 |
| 171204_pose3_2000 | 0.1353 | 0.1270 | 0.0919 | 0.0526 |
| 171204_pose3_4000 | 0.1536 | 0.1495 | 0.0862 | 0.0197 |

## Key Observations

1. **MediaPipe significantly outperforms MotionBert** on position metrics (MPJPE 0.17 vs 0.32, nearly 2x better). This is consistent across almost all examples.

2. **MotionBert has lower velocity error** (MPJVE 0.036 vs 0.046), suggesting smoother temporal predictions, though both are in a similar range.

3. **Outlier**: `171204_pose3_200` has unusually high VW-MPJPE (0.84) for MotionBert, indicating a severe depth/scale estimation error for that sequence.

4. **Data issues are the primary failure mode** -- not code bugs. The 9 failures are all due to video frame ranges exceeding available data. The `female_example_01` sequence appears to have no usable video at all.

5. **All computed metrics are finite** -- no NaN or Inf values in any results.json.

## Recommendations

- Remove or fix the 9 failing examples in `run_full_config.json` for both pipelines (frame ranges exceed video length)
- Investigate why `female_example_01` has 0 extractable frames for all requested ranges
- Investigate `171204_pose1_30000` -- video exists but GT is missing for those frames
- Consider investigating the `171204_pose3_200` MotionBert outlier (VW-MPJPE=0.84)
