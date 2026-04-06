# Report 04 -- Config Fixes & Comparison Script (2026-03-29)

## Task 1: Fix Full Run Configs

### Problem
10 of 25 examples in both `motionbert/run_full_config.json` and `mediapipe/run_full_config.json` were broken:
- 5x `female_example_01` -- video only has 327 frames and GT starts at frame 300, making it unusable
- `171204_pose3_8000` and `171204_pose3_6000` -- pose3 video is corrupt/truncated after ~frame 4600 (despite metadata claiming 9204 frames)
- `171204_pose2_30000` -- pose2 video is corrupt after ~frame 28100 (despite metadata claiming 40689 frames)
- `171204_pose1_38000` -- exceeds pose1's 31661 video frames
- `171204_pose1_30000` -- video readable but GT only available through frame 27678, so 0/50 frames had GT

### Data Investigation
Verified actual readable frame limits by extracting test frames:

| Sequence | Video Frames (metadata) | Readable Up To | GT Range | Safe max start_frame |
|---|---|---|---|---|
| 171204_pose1 | 31,661 | ~31,650 | 118-27,678 | 27,530 |
| 171204_pose1_sample | 101 | 101 | 0-100 | 0 |
| 171204_pose2 | 40,689 | ~28,100 | 140-37,890 | 27,950 |
| 171204_pose3 | 9,204 | ~4,600 | 137-9,056 | 4,450 |
| female_example_01 | 327 | 327 | 300-600 | unusable |

Key finding: pose2 and pose3 videos have corrupt/truncated data well before their reported frame count. The `ffprobe`-reported frame counts are misleading -- OpenCV's sequential read fails at the corrupt boundary.

### Solution
Replaced all 10 broken examples with valid windows spread across the three working sequences:
- **pose1**: 11 windows (1000, 5000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 26000)
- **pose2**: 9 windows (200, 2000, 5000, 8000, 10000, 15000, 20000, 25000, 27000)
- **pose3**: 4 windows (200, 2000, 3000, 4000)
- **pose1_sample**: 1 window (0, num_frames=100)
- Total: 25 examples, all verified within safe video and GT ranges

Both config files are identical in their example lists.

## Task 2: Cross-Pipeline Comparison Script

Created `pose-optimizer/compare.py` -- a standalone script that loads results from two pipeline output directories.

### Usage
```
uv run python compare.py output/motionbert_DIR/ output/mediapipe_DIR/ --output output/comparison/
```

### Output Structure
```
comparison/
  single_video/          # Level 1
    RWrist_X.png         # 36 individual joint/axis plots
    ...
    summary_grid.png     # 12x3 grid of all joints/axes
  per_video_metrics.png  # Level 2: heatmap
  per_video_metrics.json
  aggregate_metrics.png  # Level 3: bar chart
  aggregate_metrics.json
```

### Level 1: Single-Video Trajectories
- Picks the first example with trajectories in both pipelines
- For each of 12 eval joints x 3 axes: ground truth (blue), raw/optimized for both pipelines
- Summary grid shows all 36 plots on one page

### Level 2: Per-Video Metrics Heatmap
- Shows 4 key metrics (VW-SI-MPJPE, SI-MPJPE, VW-SI-MPJVE, MPJVE) x 4 variants
- Color-coded heatmap with annotated values in cm
- Makes outliers immediately visible (e.g., pose3_200 MotionBert VW-SI-MPJPE)

### Level 3: Aggregate Bar Chart
- All 9 metrics compared across 4 variants (MediaPipe Raw/Opt, MotionBert Raw/Opt)
- Values labeled on each bar in cm

### Tested
Ran successfully against existing output directories (15 common examples). All visualizations render correctly with clean legends, titles, and readable text.
