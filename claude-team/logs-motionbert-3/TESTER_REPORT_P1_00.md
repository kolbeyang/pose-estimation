# Tester Report: Phase 1, Iteration 0 -- solvePnP + Remove ALL_JOINTS_SMOOTH_WEIGHT

**Date:** 2026-03-17
**Spec:** Phase 1 of `claude-team/specs/motion-bert-round-5.md`
**Developer Report:** `claude-team/logs/DEVELOPER_REPORT_P1_00.md`

## Tests Run

### T1: ALL_JOINTS_SMOOTH_WEIGHT fully removed -- PARTIAL FAIL

- **config.py**: No references found. PASS.
- **scoring.py**: No references to `ALL_JOINTS_SMOOTH_WEIGHT` or `motion_penalty_all_joints`. PASS.
- **optimize.py**: No references. PASS.
- **sweep.py**: 3 stale references found at lines 40, 163, 170, 223, and 399:
  - Line 40: `SweepConfig.all_joints_smooth_weight: float = 0.0`
  - Line 163: `orig_smooth = cfg.ALL_JOINTS_SMOOTH_WEIGHT` -- will throw `AttributeError` at runtime
  - Line 170: `cfg.ALL_JOINTS_SMOOTH_WEIGHT = config.all_joints_smooth_weight` -- will throw `AttributeError`
  - Line 223: `cfg.ALL_JOINTS_SMOOTH_WEIGHT = orig_smooth` -- will throw `AttributeError`
  - Line 399: display string referencing the field

**Result**: The removal was done in the three core files but `sweep.py` was missed. Running `sweep.py` will crash.

### T2: solvePnP implementation code review -- PASS

Reviewed `detect.py` lines 582-722. The implementation follows the architect's plan exactly:
- Camera matrix K is correctly constructed as `[[fx,0,cx],[0,fy,cy],[0,0,1]]`
- Visibility filtering: joints with `norm(kp_2d) > 1.0` AND `visibility > 0.1`
- Requires >= 4 valid joints for solvePnP, otherwise fallback
- Uses `cv2.SOLVEPNP_SQPNP` (correct choice for general PnP)
- Safety check: root Z must be in (0.5, 15.0)m, otherwise fallback
- Fallback uses person-height depth heuristic (thorax-to-ankle pixel height)
- Bone-length enforcement applied post-solvePnP (max_ratio=1.3)

No correctness issues found in the code itself. The code is well-structured.

### T3: Smoke test -- PASS

Developer's predictions exist at:
- `motionbert-pose/training_runs/p1-solvepnp-test/predictions/171204_pose1_sample_0.json`
- `motionbert-pose/training_runs/p1-solvepnp-test/predictions/171204_pose3_4000.json`

Both contain valid metrics and per-frame data. `main.py` imports without errors.

### T4: Z-value trajectory analysis -- FAIL (critical finding)

**Example 0 (171204_pose1_sample_0):**
| Metric | Detection | Optimized | Ground Truth |
|--------|-----------|-----------|--------------|
| Root Z mean | 2.285 m | 2.294 m | 2.588 m |
| Root Z std | 0.147 m | 0.128 m | 0.016 m |
| Root Z range | 1.44 - 2.45 m | 1.56 - 2.40 m | 2.57 - 2.63 m |
| Z velocity std | 0.1067 m/f | 0.0756 m/f | 0.0033 m/f |
| Z velocity max | 0.898 m/f | 0.638 m/f | 0.018 m/f |
| Mean |Z error| | 30.28 cm | 29.34 cm | -- |

**Problems:**
1. Root Z is systematically ~30cm too shallow (2.29 vs 2.59m GT)
2. Z velocity std is 23x ground truth (0.076 vs 0.003 m/f) -- extremely jittery
3. Max Z velocity is 0.64 m/f optimized (0.90 m/f detection) vs 0.018 m/f GT -- nearly 1m frame-to-frame jumps
4. Visual inspection of Hip_Z.png shows a catastrophic ~1m Z drop around frames 40-45 (person extends arms)

**Example 5 (171204_pose3_4000):**
| Metric | Detection | Optimized | Ground Truth |
|--------|-----------|-----------|--------------|
| Root Z mean | 2.307 m | 2.301 m | 2.405 m |
| Root Z std | 0.016 m | 0.009 m | 0.003 m |
| Z velocity std | 0.0112 m/f | 0.0022 m/f | 0.0012 m/f |
| Mean |Z error| | 9.73 cm | 10.40 cm | -- |

Example 5 is much better -- Z std is only 3x GT (vs 23x for Example 0). The optimizer does successfully smooth Z here. However, there is still a systematic 10cm depth bias.

### T5: MPJPE comparison vs round-4 baseline -- CONFIRMED REGRESSION

| Metric | Ex 0 Round-4 | Ex 0 Round-5 | Ex 5 Round-4 | Ex 5 Round-5 |
|--------|-------------|-------------|-------------|-------------|
| Det MPJPE (cm) | 30.98 | 34.72 | 15.85 | 18.26 |
| Opt MPJPE (cm) | 30.44 | 34.12 | 15.44 | 18.65 |
| Det P-MPJPE (cm) | 28.57 | 28.57 | 20.33 | 20.33 |
| Opt P-MPJPE (cm) | 28.42 | 28.30 | 20.22 | 20.52 |

MPJPE regressed by 3.7 cm (ex 0) and 3.2 cm (ex 5). P-MPJPE is unchanged -- this confirms the regression is purely a global depth/translation issue, not a pose shape issue.

### T6: MPJVE comparison -- FAIL (spec requirement not met)

| Metric | Ex 0 Round-4 | Ex 0 Round-5 | Ex 5 Round-4 | Ex 5 Round-5 |
|--------|-------------|-------------|-------------|-------------|
| Det MPJVE (cm/f) | 0.94 | 3.69 | 0.51 | 0.93 |
| Opt MPJVE (cm/f) | 1.05 | 3.38 | 0.48 | 0.54 |

**MPJVE is significantly worse for both examples.**
- Example 0 Det MPJVE: 0.94 -> 3.69 (3.9x worse)
- Example 0 Opt MPJVE: 1.05 -> 3.38 (3.2x worse)
- Example 5 Det MPJVE: 0.51 -> 0.93 (1.8x worse)
- Example 5 Opt MPJVE: 0.48 -> 0.54 (1.1x worse)

The spec explicitly requires "MPJVE should be improved for ALL test videos." This requirement is not met for either example.

### T7: Developer claim verification -- CONFIRMED

The developer reported that solvePnP produced worse MPJPE and MPJVE. My independent analysis of the prediction JSONs confirms the exact same numbers. The developer's report is accurate.

### T8: Heatmap/overlay video frame analysis -- PASS (informational)

Extracted frames 0 and 41 from the overlay video for Example 0:
- Frame 0: Skeletons project onto the person reasonably. Green/red/blue all roughly aligned on the torso; arm estimation has notable errors.
- Frame 41: Person has arms outstretched. Detector and optimized skeletons show significant arm projection errors. This correlates with the catastrophic Z drop visible in the Hip_Z trajectory plot -- solvePnP produces a bad depth estimate when the pose is unusual.

Artifacts saved to:
- `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/artifacts/overlay_frame_ex0_f0.png`
- `/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/artifacts/overlay_frame_ex0_f41.png`

## Bugs Found

### BUG-1: sweep.py crashes due to stale ALL_JOINTS_SMOOTH_WEIGHT references (Severity: Medium)
- **Description:** `sweep.py` lines 163, 170, 223 reference `cfg.ALL_JOINTS_SMOOTH_WEIGHT` which was removed from `config.py`. Calling `run_sweep_config()` will throw `AttributeError`.
- **How observed:** Grep of entire `motionbert-pose/` for `ALL_JOINTS_SMOOTH`.
- **Severity:** Medium -- sweep is not part of the main pipeline but is a development tool that will crash if used.
- **Fix direction:** Remove the `all_joints_smooth_weight` field from `SweepConfig`, and remove lines 163, 170, 223 from `run_sweep_config()`. Remove the display reference at line 399.

### BUG-2: solvePnP produces worse MPJPE and MPJVE than the previous pairwise method (Severity: HIGH -- blocks spec)
- **Description:** solvePnP's per-frame independent depth estimation produces noisier Z values than the old pairwise separation method. Example 0 shows a catastrophic ~1m Z drop around frame 40-45 and MPJVE degrades 3.2x. Both tested examples show worse MPJPE (3-4cm regression) and worse MPJVE.
- **How observed:** Z trajectory analysis (T4), MPJVE comparison (T6).
- **Severity:** HIGH -- the spec requires MPJVE improvement for ALL test videos, and this change makes it significantly worse.
- **Root cause analysis:** solvePnP runs independently per frame with no temporal context. The old pairwise method, while theoretically less principled, was more stable because it estimated depth from multiple joint pair ratios with IQR filtering. solvePnP is sensitive to the exact 2D-3D correspondences and can produce large depth errors when the pose changes (e.g., arm extension at frame 40-45).
- **Contributing factor:** The ~20-degree rotation from solvePnP suggests MotionBERT's coordinate frame is not perfectly camera-aligned. The rotation correction may be introducing additional noise.
- **Fix direction options (for architect):**
  1. Restrict solvePnP to translation-only (fix rotation to identity) since MotionBERT output is already approximately camera-aligned
  2. Apply temporal median filtering to solvePnP depth estimates before optimization
  3. Revert to the pairwise method and address smoothing through other means
  4. Use solvePnP depth as an additional optimization constraint rather than as initialization

## Code Review Findings

### CR-1: solvePnP allows full rotation, which may be unnecessary (Minor/Informational)
The architect's plan noted the ~20-degree rotation as a risk. MotionBERT is trained on H3.6M camera-space data, so the output should already be roughly camera-aligned. Allowing solvePnP to estimate rotation may be adding noise rather than correcting misalignment. A translation-only variant (fixing R=I) might produce more stable results.

### CR-2: No temporal filtering of solvePnP output (Informational)
The architect's Risk 3 acknowledged that per-frame solvePnP would produce jittery Z values and noted that the optimizer should handle smoothing. However, the data shows the optimizer cannot fully compensate -- Example 0 still has 0.638 m/f max Z velocity after optimization vs 0.018 m/f for GT. A temporal filter (e.g., median over a small window) on the solvePnP output before optimization could help.

## Verdict

**NO** -- Phase 1 goals are NOT met.

1. **ALL_JOINTS_SMOOTH_WEIGHT removal**: Partially done. Core files are clean, but `sweep.py` has 3 stale references that will crash at runtime. (BUG-1)

2. **solvePnP for depth estimation**: Implemented correctly per the architect's plan, but produces significantly worse results than the previous method. MPJPE regressed 3-4cm on both examples. MPJVE regressed 3.2x on Example 0. (BUG-2)

3. **MPJVE improvement requirement**: The spec explicitly requires "MPJVE should be improved for ALL test videos." Both tested examples show WORSE MPJVE. This is a blocking failure.

The developer followed the architect's plan faithfully. The issue is that solvePnP is fundamentally noisier per-frame than the old pairwise method for this application. The next iteration should either add temporal stabilization to solvePnP output or consider a hybrid approach.
