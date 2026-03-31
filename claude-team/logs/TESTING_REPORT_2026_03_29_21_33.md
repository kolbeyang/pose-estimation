# Testing Report 02 -- Full Pipeline Comparison (25 Examples)

**Date:** 2026-03-29
**Run by:** Evaluator Eve (QA)

## Summary

Both MotionBert and MediaPipe pipelines ran successfully on all 25 examples with fixed frame-range configs. Comparison outputs generated without errors. No NaN or Inf values found anywhere.

## Pipeline Runs

### MotionBert
- **Output directory:** `output/motionbert_2026_03_29_21_24/`
- **Result:** 25/25 examples completed successfully
- **All 25 results.json files present**

### MediaPipe
- **Output directory:** `output/mediapipe_2026_03_29_21_33/`
- **Result:** 25/25 examples completed successfully
- **All 25 results.json files present**

## Comparison Outputs

**Output directory:** `output/comparison/`

All expected files present:
- `single_video/` -- 36 trajectory plots (12 joints x 3 axes) + 1 summary grid (37 files total)
- `per_video_metrics.png` + `per_video_metrics.json`
- `aggregate_metrics.png` + `aggregate_metrics.json`

## Aggregate Metrics (cm / cm-per-frame for velocity)

| Metric       | MediaPipe Raw | MediaPipe Opt | MotionBert Raw | MotionBert Opt |
|-------------|--------------|--------------|----------------|----------------|
| MPJPE       | 16.29        | 16.09        | 26.78          | 28.05          |
| P-MPJPE     | 10.93        | 10.15        | 26.51          | 26.35          |
| SI-MPJPE    | 15.73        | 15.26        | 25.58          | 26.93          |
| VW-MPJPE    | 12.46        | 12.38        | 14.00          | 15.67          |
| VW-SI-MPJPE | 12.08        | 11.91        | 12.63          | 14.38          |
| MPJVE       | 5.62         | 4.67         | 2.74           | 3.05           |
| SI-MPJVE    | 5.81         | 4.94         | 2.49           | 2.70           |
| VW-MPJVE    | 4.87         | 4.06         | 2.18           | 2.59           |
| VW-SI-MPJVE | 4.77         | 4.04         | 2.01           | 2.39           |

## Sanity Checks

### 1. All 9 metrics present and finite
**PASS.** All 4 variants (MediaPipe Raw/Opt, MotionBert Raw/Opt) have exactly 9 metrics, all finite.

### 2. SI-MPJPE <= MPJPE for all variants
**PASS.**
- MediaPipe Raw: 15.73 <= 16.29
- MediaPipe Opt: 15.26 <= 16.09
- MotionBert Raw: 25.58 <= 26.78
- MotionBert Opt: 26.93 <= 28.05

### 3. Does optimization improve velocity metrics (MPJVE)?
**MIXED.**
- MediaPipe: YES -- MPJVE drops from 5.62 to 4.67 cm/f (17% improvement)
- MotionBert: NO -- MPJVE increases from 2.74 to 3.05 cm/f (11% worse)

This is a notable finding. The FK optimization improves MediaPipe's temporal smoothness but degrades MotionBert's. MotionBert already produces very smooth trajectories (2.74 cm/f), and the optimization adds noise rather than helping. This pattern holds across all 4 velocity metrics.

### 4. NaN or outlier check
**No NaN or Inf values** in any per-video or aggregate metric.

**Outliers identified:**
- `171204_pose3_200` has very high MotionBert VW-SI-MPJPE (63.5 cm raw, 62.7 cm opt) -- likely a difficult sequence or detection failure
- `171204_pose1_sample_0` has MotionBert SI-MPJPE ~50 cm (raw and opt)
- `171204_pose1_5000` and `171204_pose2_15000` also show elevated MotionBert SI-MPJPE (>40 cm)
- All MediaPipe values stay below 39 cm SI-MPJPE

## Key Findings

1. **MediaPipe wins on positional accuracy** across all metrics (MPJPE, SI-MPJPE, VW-SI-MPJPE). The gap is roughly 10 cm in MPJPE, narrowing to ~2 cm when visibility-weighted (VW-SI-MPJPE).

2. **MotionBert wins on temporal smoothness** with MPJVE roughly half of MediaPipe's (2.74 vs 5.62 cm/f).

3. **FK optimization helps MediaPipe** modestly on both position and velocity. For MotionBert, optimization slightly worsens all metrics.

4. **Visibility weighting narrows the gap** substantially between pipelines, suggesting MotionBert's main errors are on occluded/less-visible joints.

5. **No pipeline failures** -- all 25 examples ran cleanly for both pipelines.
