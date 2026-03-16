# Tester Report P1-00: Parameter Sweep Verification

## Tests Run

### T1: test_single.py displays new 2D-vs-Det metric -- PASS
- **Command**: `cd motionbert-pose && uv run python test_single.py 0`
- **Result**: Output contains both lines:
  ```
  Det 2D-vs-Det: 38.1 px
  Opt 2D-vs-Det: 33.8 px
  ```
- Values match sweep baseline (38.06 and 33.78 px in JSON, rounded to 38.1 and 33.8).

### T2: Sweep results JSON well-formed -- PASS
- **File**: `motionbert-pose/training_runs/sweep_results/sweep_171204_pose1_sample_0.json`
- **Result**: 20 entries present. All required fields (`config`, `det_mpjpe_cm`, `opt_mpjpe_cm`, `improvement_cm`, `opt_p_mpjpe_cm`, `det_2d_det_mpjpe_px`, `opt_2d_det_mpjpe_px`, `opt_mpjve_cm`) are non-null for all entries.

### T3: det_2d_det_mpjpe_px constant across configs -- PASS
- All 20 entries have `det_2d_det_mpjpe_px` = 38.059 px (identical to 15 decimal places).
- This is correct: the detector baseline does not change across sweep configs.

### T4: Code review -- reprojection_error_vs_detections() -- PASS
- Correctly uses `camera.world_to_image(positions_3d[i])` to project 3D predictions to 2D.
- Compares against `detections_2d[i]` (SH keypoints), NOT GT projections. This is the correct metric per the spec.
- Visibility masking at threshold 0.5 correctly excludes low-confidence joints.
- Edge case: if no joints are visible for a frame, appends 0.0 (not NaN). Acceptable behavior.

### T5: Code review -- sweep.py config restoration -- PASS
- `run_sweep_config()` saves all 6 config fields before modification and restores them in a `finally` block.
- `ROTATION_PENALTY_PER_JOINT` is saved via `.copy()` (numpy array), preventing aliasing bugs.

### T6: sweep.py import -- PASS
- **Command**: `uv run python -c "from sweep import get_phase1_1_configs; configs = get_phase1_1_configs(); print(f'{len(configs)} configs')"`
- **Result**: `20 configs`

### R1: main.py import regression -- PASS
- **Command**: `uv run python -c "from main import process_example; print('OK')"`
- **Result**: `OK`

## Bugs Found

None. All code is correct and functional.

## Code Review Findings

### CR-1: Sweep uses `improved_target_2d` as the "detections" reference (Minor, not a bug)
The 2D-vs-Det metric compares against `improved_target_2d`, which is SH keypoints for high-confidence joints but MotionBERT-projected 2D for low-confidence joints (below `FK_TARGET_CONF_THRESHOLD`). This means the metric is not purely "distance from SH detections" -- for low-confidence joints, it measures distance from the MotionBERT projection (which is also the optimization's initial target for those joints). This is **consistent** with how the optimization target works, so it's a reasonable choice, but the user should be aware that this is not a pure "SH 2D distance" metric.

### CR-2: MPJVE not printed in sweep table (Minor)
The sweep summary table header includes only MPJPE, P-MPJPE, and 2D-vs-Det columns. MPJVE is saved to JSON (`opt_mpjve_cm`) but not printed in the console table. The spec lists MPVPE (velocity error) as a required test metric. Data is captured in JSON, so this is a display-only gap.

## Phase 1.1 Results Analysis

### Sweep Data Summary (20-step configs)

| Parameter | Range Tested | Opt MPJPE Range (cm) | Effect |
|-----------|-------------|---------------------|--------|
| pos_w | 0 -- 5000 | 30.39 -- 30.45 | 0.06 cm spread |
| rot_s | 0 -- 1000 | 30.39 -- 30.45 | 0.06 cm spread |
| anchor | 0 -- 500 | 30.43 -- 30.55 | 0.12 cm spread |
| heatmap_only | all=0 | 30.44 | Same as baseline |
| all_low | 0.5/0.1/0.05 | 30.44 | Same as baseline |

### Key Finding: Penalties are irrelevant at 20 steps
At 20 optimization steps, every parameter configuration produces nearly the same result (30.39--30.55 cm). The maximum spread is 0.16 cm. Even `heatmap_only` (all penalties = 0) matches the baseline. **This means 20 steps is insufficient for penalty terms to meaningfully influence the optimization trajectory.**

### Key Finding: More steps improve 2D fit but hurt 3D accuracy

| Config | Steps | Opt MPJPE (cm) | Opt P-MPJPE (cm) | Opt 2D-vs-Det (px) | MPJVE (cm/f) |
|--------|-------|----------------|-------------------|---------------------|--------------|
| baseline | 20 | 30.44 | 28.42 | 33.8 | 1.05 |
| baseline_100steps | 100 | 31.10 | 28.30 | 28.2 | 1.36 |
| all_low_100steps | 100 | 31.42 | 28.29 | 27.2 | 1.84 |

At 100 steps:
- **3D MPJPE gets WORSE** (+0.66 to +0.98 cm)
- **2D alignment gets BETTER** (-5.6 to -6.6 px)
- **P-MPJPE slightly improves** (-0.12 to -0.13 cm) -- the pose shape is better, just worse positioned
- **MPJVE gets WORSE** (+0.31 to +0.79 cm/f) -- temporal jitter increases significantly

This is the classic depth-ambiguity problem: the optimizer fits 2D detections well but sacrifices depth accuracy. Low penalties (`all_low_100steps`) exacerbate this -- without regularization, the optimizer overfits to noisy 2D targets, causing both worse absolute 3D error AND worse temporal smoothness.

### Phase 1.1 Goal Assessment
**Goal: Coarse-to-fine parameter tuning to understand how parameters affect output.**
- **Met**: The sweep comprehensively demonstrated that penalty weights have negligible effect at 20 steps, and that increasing steps creates a 2D-vs-3D accuracy tradeoff.
- **Actionable insight**: The core problem is not penalty weight tuning -- it's that heatmap-based optimization inherently sacrifices depth for 2D alignment. More steps = better 2D fit but worse depth.

## Recommendations for Phase 1.2 (Heatmap Blur)

1. **Heatmap blur should be tested at moderate step counts (50-100 steps)**. At 20 steps nothing matters. The interesting regime is where optimization has enough steps to move but needs a wider gradient basin.

2. **Blur may help the 2D-3D tradeoff**. The hypothesis is that blurred heatmaps provide gentler gradients that guide joints toward the right region without demanding pixel-perfect 2D alignment. This could reduce the "overfitting to noisy 2D" problem seen at 100 steps.

3. **Include MPJVE in the sweep console output**. The velocity error is critical for understanding temporal stability and is already saved in JSON. The 100-step results show dramatically worse MPJVE (1.84 vs 1.05 cm/f), which is important context.

4. **Test blur sigma values of 1, 2, 4, 8 at both 50 and 100 steps**. The architect's plan already includes this. Prioritize the 50-step + blur={2,4} regime as the most likely sweet spot.

5. **Consider a depth regularization strategy**. The data shows the fundamental issue: optimizing toward 2D heatmaps hurts depth. Phase 1.2 blur alone may not solve this. If blur helps 2D alignment without hurting 3D as much, that's a win. If not, a depth-preserving constraint (e.g., penalizing Z deviation from MotionBERT's depth estimate) may be needed.

## Verdict

**YES** -- Phase 1.1 goals are met. The sweep infrastructure works correctly, the new 2D-vs-detection metric is properly implemented, and the results provide clear, actionable insights. The code is clean, correctly restores config state, and saves all required metrics. Ready to proceed to Phase 1.2 (heatmap blur).
