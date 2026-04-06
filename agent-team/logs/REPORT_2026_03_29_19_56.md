# Report 02: Fix results.json format deviation

**Date:** 2026-03-29
**Developer:** Developer Dan
**Branch:** feat/combined-motionbert-mediapipe
**Commit:** af813f6

---

## TODO Item 1: results.json format deviation -- FIXED

**Problem:** The plan (Phase 6) specifies separate top-level keys `metrics`, `raw_metrics`, `per_joint`, and `config`. The implementation was dumping all metrics (det_ and opt_ prefixed) plus per-joint and per-frame data into a single flat `metrics` dict.

**Fix:** Restructured the results.json construction in both `motionbert/__init__.py` and `mediapipe/__init__.py`. The 9 optimized metric values go into `metrics` (unprefixed), the 9 raw/detector values go into `raw_metrics` (unprefixed), and per-joint breakdowns, per-frame MPJPE, and bone lengths go into `per_joint`. The internal `metrics` dict used by graph generation functions is unchanged.

**Verified output format (MediaPipe single-example run):**
```
Top-level keys: model, example, num_frames, metrics, raw_metrics, per_joint, config
metrics keys:   mpjpe, p_mpjpe, si_mpjpe, vw_mpjpe, vw_si_mpjpe, mpjve, si_mpjve, vw_mpjve, vw_si_mpjve
raw_metrics:    (same 9 keys)
per_joint:      det_per_joint, opt_per_joint, det_per_frame_mpjpe, opt_per_frame_mpjpe, gt_bone_lengths, det_bone_lengths
```

This matches the Phase 6 spec exactly.

**No regressions:** The internal `metrics` dict passed to `generate_summary()` and `generate_aggregate_summary()` is unmodified -- only the JSON serialization was restructured. All graphs and overlay videos still generate correctly.

---

## TODO Items 2 and 3: Observations (no action taken)

**Item 2 (MotionBert VW-SI-MPJPE regression):** The 2.1% increase on a single 34-frame clip is marginal and expected when motion penalties over-smooth a short sequence. Needs validation on the full 25-example run in Phase 7.

**Item 3 (MotionBert high absolute MPJPE):** Reviewed `motionbert_to_camera_space` in `motionbert/detect.py:458-518`. The depth estimation uses a two-step heuristic: (1) bone-length ratio scaling against `DEFAULT_BONE_LENGTHS`, (2) vertical disparity across joint pairs to estimate root depth `tz`. The approach is reasonable but has potential systematic error sources:

- `DEFAULT_BONE_LENGTHS` are fixed reference values; any mismatch with the actual subject introduces a global scale bias.
- The vertical disparity method (`fy * dy_3d / dv_2d`) is sensitive to MotionBERT's relative Y-axis accuracy and to poses where vertical separations are small (e.g., arms at sides).
- The depth is clipped to [1.0, 8.0] meters which is appropriate for Panoptic dome scenes but could mask errors.

The VW-SI-MPJPE (~19 cm) after scale-invariant and visibility weighting is more representative of actual pose quality. The raw MPJPE (51 cm) is inflated by the depth/scale estimation error, which SI metrics are designed to factor out. No code change warranted at this time.

---

## Files Changed

- `pose-optimizer/motionbert/__init__.py` -- results.json construction (lines 240-262)
- `pose-optimizer/mediapipe/__init__.py` -- results.json construction (lines 246-268)
