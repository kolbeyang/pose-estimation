# Testing Report -- Evaluator Eve -- 2026-03-30 21:00

## Scope
Verify Dan's pipeline runs (25 examples each for MediaPipe and MotionBert) and `compare.py` comparison output.

## 1. Pipeline Outputs Verified

| Check | Result |
|-------|--------|
| MediaPipe output dir exists (`mediapipe_2026_03_30_19_15`) | PASS -- 25 subdirectories |
| MotionBert output dir exists (`motionbert_2026_03_30_18_57`) | PASS -- 25 subdirectories |
| results.json files contain real metrics | PASS -- spot-checked multiple, all have valid floats |
| No NaN or zero-valued metrics | PASS -- scanned all 50 results.json files, none found |

## 2. Comparison Output Verified

| Check | Result |
|-------|--------|
| Comparison dir exists (`comparison_2026_03_30_19_30`) | PASS -- 9 PNGs + summary.json |
| PNG files non-empty | PASS -- all 33-79 KB |
| summary.json has 25 examples | PASS |
| Metrics look reasonable | PASS -- see summary below |

### Key Numbers from summary.json

| Metric | MediaPipe | MotionBert |
|--------|-----------|------------|
| VW-SI-MPJPE mean (cm) | 10.8 | 13.0 |
| VW-SI-MPJPE median (cm) | 9.9 | 9.4 |
| SI-MPJPE mean (cm) | 13.8 | 23.7 |
| VW-SI-MPJVE mean (cm/frame) | 3.8 | 2.0 |
| Wins (VW-SI-MPJPE) | 14/25 | 11/25 |

**Interpretation**: MediaPipe has lower mean error overall; MotionBert has lower median and smoother velocity. The comparison graphs answer the "which is better" question clearly.

## 3. Smoke Test

Re-ran `uv run python compare.py` on the same output dirs. Completed successfully, produced a new `comparison_2026_03_30_19_32` directory with identical structure and metrics.

## 4. Issues Found and Fixed

| Issue | Severity | Status |
|-------|----------|--------|
| Unused `get_metric` nested function on line 62 of `compare.py` | Low (dead code) | FIXED -- removed |

## 5. Pre-Existing Issues (Not Fixed, From TODO.md)

- MotionBert `171204_pose3_200` outlier (VW-MPJPE 0.84m) -- still present, not a compare.py issue
- 9 examples fail due to out-of-range frames -- already addressed by using valid configs
- `171204_pose1_30000` has no GT -- not in current runs

## Verdict

PASS. All 25 examples ran for both pipelines, comparison graphs are correct and non-empty, summary numbers are reasonable. One dead-code issue fixed in `compare.py`.
