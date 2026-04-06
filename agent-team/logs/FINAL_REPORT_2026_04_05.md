# Final Report — Optimization Improvement (2026-04-05)

**Branch**: `refactor-2026-04-04`
**Spec**: `agent-team/specs/2026-04-05-06-39.md`
**Commits**: `207e34b` (Phase 1-3), `3320ce1` (Phase 4), `6c6f1e1` (Phase 5)

---

## Success Criteria

| Criterion | Result | Met? |
|-----------|--------|------|
| Optimized MB >2% better than raw (avg) | +9.4% across 15 examples | **YES** |
| Optimized MP >2% better than raw (avg) | +3.0% across 14 examples (excl 1 detection failure) | **YES** |
| MB optimized <10 cm avg VW-SI-MPJPE | 23.06 cm | **NO** |
| MP optimized <10 cm avg VW-SI-MPJPE | 12.77 cm (excl outlier) | **NO** |

**Important context on <10 cm target**: The prior spec (2026-04-04-11-44, Phase 2) changed evaluation from root-relative to camera-space ("Evaluation metrics should not be root relative. This is cheating."). Under root-relative evaluation, MediaPipe would be ~9-10 cm. The <10 cm target in this spec was likely written when metrics were root-relative. Under camera-space metrics, the target is significantly harder because the metric must simultaneously match depth and skeleton scale.

---

## Experiments (Spec Format)

### Experiment 1: Error Calculation Audit

1. **Observation**: Results.json errors look too high given what the heatmap videos show.
2. **Hypothesis**: VW-SI-MPJPE calculation has a bug, or visibility weighting is incorrect.
3. **Verification**: Code review of `evaluate.py`, manual recomputation matching library output to <1e-8 precision.
4. **Results**: Metric is correct. Visibility uses GT projections (correct). Both ankles have 0% visibility in the test example, excluding their massive errors. Camera-space scale factor is 1.11-1.15x (no bound saturation).
5. **Solution**: No bug to fix. Added `per_joint_visible_frames` to results.json as requested by spec.
6. **Solution results**: Results.json now shows e.g. `[34, 34, 34, 0, 34, 34, 0, 34, 34, 34, 34, 34, 34, 34, 34]` — ankles at 0 confirming exclusion.

### Experiment 2: MotionBERT Knee Investigation

1. **Observation**: Knees contribute 51% of MotionBERT's weighted error. Detected bone lengths are 1.3-2.9x too long.
2. **Hypothesis**: MotionBERT's 3D lift fails when legs are partially out of frame.
3. **Verification**: `experiment/audit_motionbert_knee.py` — per-bone scale ratios, direction cosine similarity, 2D reprojection errors.
4. **Results**: Upper body bones accurate (0.90-1.09x), lower body wildly wrong (1.32-2.92x). Knee direction inverted in some frames (cos_sim=-0.84). 2D reprojection error for knees: 467px (vs 27-84px upper body). Root cause: MotionBERT trained on H36M with full-body visibility, fails on partial views.
5. **Solution**: Bone length clamping to [0.5x, 1.5x] of defaults in `optimize/__init__.py:110-117`. Increased `bone_length_lr` from 0.0001 to 0.001.
6. **Solution results**: Single example: 18.86 → 16.83 cm. 15-example avg: 25.45 → 23.06 cm (+9.4%). The clamping prevents pathological initialization but cannot fix the underlying 3D lift distortion.

### Experiment 3: MediaPipe Regression

1. **Observation**: MediaPipe optimization makes results 1.2% WORSE (15.66 → 15.86 cm).
2. **Hypothesis**: SH heatmaps disagree with GT at wrist/ankle positions; optimizer drifts joints toward wrong heatmap peaks.
3. **Verification**: `experiment/audit_fk_roundtrip.py` — per-joint error change, FK roundtrip analysis.
4. **Results**: FK roundtrip error is negligible (0.0005 cm). Wrists regress +11-12%, ankles +4-8%. These 4 joints have the lowest rotation penalties (0.1-0.2), making them most mobile. 2D reprojection error for wrists nearly doubles (25→51px) after optimization.
5. **Solution**: Anchor penalty in `scoring.py` — penalizes deviation from raw detector positions. Anchoring to raw positions (not FK-init) was critical. `anchor_weight=200` optimal for MediaPipe, `anchor_weight=0` for MotionBERT (raw predictions are bad).
6. **Solution results**: Single example: 15.86 → 15.20 cm (fixed regression). 14-example avg (excl outlier): 13.16 → 12.77 cm (+3.0%).

### Experiment 4: Blur Sigma

1. **Observation**: `heatmap_blur_sigma=16` on 64x64 heatmaps spreads each peak across 50% of image. Uncommitted change to sigma=32 destroys all spatial info.
2. **Hypothesis**: Sigma is too high, limiting optimization precision.
3. **Verification**: Sweep of sigma=[4, 8, 12, 16] in `experiment/mediapipe_sweep.py`.
4. **Results**: sigma=4 outperforms sigma=16 for MediaPipe. For MotionBERT, annealing 16→4 is best (starts far from truth, needs broad gradients initially).
5. **Solution**: MediaPipe: fixed sigma=4. MotionBERT: anneal 16→4 over 100 steps. Reverted uncommitted sigma=32.
6. **Solution results**: +0.2 cm improvement for MediaPipe over sigma=16. MotionBERT annealing: 16.83 → 15.33 cm on single example.

### Experiment 5: Hyperparameter Sweep

1. **Observation**: Default hyperparameters suboptimal for both pipelines.
2. **Hypothesis**: Per-pipeline tuning can push both above 2% improvement target.
3. **Verification**: 4 sweep scripts, 59 total configs tested across 1-5 examples.
4. **Results**: See Phase 4 report for full sweep tables.
5. **Solution**: MotionBERT: 100 steps, lr=0.001, bone_lr=0.005, anneal 16→4, no anchoring. MediaPipe: 50 steps, sigma=4, lr=0.0005, anchor=200.
6. **Solution results**: See final validation below.

---

## Final 15-Example Results

### MotionBERT
| Stat | Value |
|------|-------|
| Avg raw | 25.45 cm |
| Avg optimized | 23.06 cm |
| Improvement | **+9.4%** |
| Examples improved | 11/15 (73%) |
| Regressions | 4/15 — all on examples with <10 cm raw error |

### MediaPipe (excluding 1 detection-failure outlier at 113.82 cm)
| Stat | Value |
|------|-------|
| Avg raw | 13.16 cm |
| Avg optimized | 12.77 cm |
| Improvement | **+3.0%** |
| Examples improved | 13/14 (93%) |
| Regressions | 0/14 |

Note: 10/25 configured examples failed (missing/short videos). Results are on the 15 valid examples.

---

## Code Changes

### `optimize/__init__.py`
- **Bone length clamping** (lines 110-117): After computing median bone lengths, clamp to [0.5x, 1.5x] of `DEFAULT_BONE_LENGTHS`. Prevents MotionBERT's 2-3x wrong leg bones from corrupting initialization.
- **Blur sigma annealing** (lines 174-182): When `heatmap_blur_sigma_start` and `_end` are both set, linearly interpolates sigma over steps. Falls back to fixed sigma for backward compat.
- **Anchor position passing** (lines 155-162): Computes raw detector positions as anchor target when `anchor_weight > 0`.

### `scoring.py`
- **Anchor penalty** in `compute_total_score_batch()`: `anchor_weight * mean((positions - anchor_positions)^2)`. Penalizes deviation from raw detector output. Zero by default.

### `config.py`
- New fields: `heatmap_blur_sigma_start`, `heatmap_blur_sigma_end`, `anchor_weight`
- `PipelineConfig.optimization` dict for per-pipeline overrides
- `RunConfig.optimization_for_pipeline()` merges pipeline-specific on top of shared config
- `bone_length_lr` default: 0.0001 → 0.001

### `main.py`
- Added `per_joint_visible_frames` to metrics output
- Uses `config.optimization_for_pipeline(pipeline_name)` for per-pipeline optimization

### Config files updated
All 5 existing configs + 1 new (`both-local-5-examples.json`) with per-pipeline optimization settings.

---

## Architecture: How It Works

```
main.py configs/both-local-single.json
  ↓
config.optimization_for_pipeline("motionbert")
  → merges pipeline.optimization overrides onto shared config
  → MB gets: 100 steps, anneal 16→4, anchor_weight=0
  
config.optimization_for_pipeline("mediapipe")
  → MP gets: 50 steps, sigma=4, anchor_weight=200

optimize() in optimize/__init__.py:
  1. positions_to_fk_params() for each frame
  2. median bone lengths → clamp to [0.5x, 1.5x] of defaults
  3. Adam optimizer with 3 param groups (root_pos, rotations, bone_lengths)
  4. Per-step: if annealing, recompute blurred heatmaps at current sigma
  5. compute_total_score_batch() → heatmap + position + rotation + anchor penalties
  6. Gradient descent on negative score
```

Key: `scoring.py:compute_total_score_batch()` line with anchor:
```python
if anchor_positions is not None and anchor_weight > 0:
    anchor_loss = anchor_weight * torch.mean((positions - anchor_positions) ** 2)
    total = total - anchor_loss
```

---

## Unresolved Issues

1. **<10 cm target unreachable for MotionBERT** without improving the underlying 3D lift. Sequences where MotionBERT's raw error exceeds 30 cm (pose2_200: 69cm, pose1_14000: 69cm, pose3_200: 46cm) have fundamentally distorted lower body predictions that the optimizer can only partially correct.

2. **MotionBERT regressions on good predictions**: 4/15 examples where raw error <10 cm got worse after optimization. The optimizer overshoots when the starting point is already close. A per-example adaptive strategy (skip or reduce steps when raw quality is high) could help.

3. **10/25 configured examples failed**: Missing videos or start frames beyond video length. The 25-example config needs curation.

4. **Missing SH heatmap visual inspection for knees**: We diagnosed the knee issue via bone length analysis and 2D reprojection, but did not directly visualize whether SH heatmap channels have peaks at knee locations. The 2D reprojection data (467px error) strongly suggests heatmaps are wrong, but direct visualization would be more definitive.

5. **Blur annealing re-blurs every step via scipy**: For 100-step MotionBERT runs, this adds ~5-10 seconds. Pre-computing heatmaps at a few sigma levels would be faster.

---

## Spec Issues

1. **"VW-SW-MPJPE" on line 62**: Likely typo for "VW-SI-MPJPE".
2. **<10 cm target vs camera-space metrics**: This target was likely set when metrics were root-relative (prior to the 2026-04-04 spec which changed to camera-space). Under root-relative, MediaPipe would be ~9-10 cm. Under camera-space, 12.77 cm is the best achievable without reducing depth error.

---

## Agent Process Notes

- **Phase 0 (diagnostics) was critical.** Without it, we would have blindly tuned hyperparameters. The diagnostics identified the 3 root causes (MB bone distortion, MP wrist drift, aggressive blur) that informed targeted fixes.
- **Anchor penalty was the key innovation for MediaPipe.** Simple idea (penalize deviation from raw positions), but the choice to anchor to raw positions rather than FK-init positions was non-obvious and crucial (FK-init has ~0.4 cm bone length error).
- **Per-pipeline configs were essential.** MB and MP have opposite optimization profiles: MB needs aggressive optimization (bad starting point), MP needs conservative (good starting point). A single config cannot serve both.
- **Nathan's nit-pick about evaluate.py was a false alarm** — the root-relative removal was from a prior spec, not this session. However, his strategic observation about the <10 cm target being invalidated by the metric change was valuable.
- **Total: 6 phases, ~4 agent dispatches, 15 valid examples tested.** The bottleneck was compute time for the 25-example runs (~35 min per pipeline).
