# TODO

## Phase 0 Findings (2026-04-05)

### Critical
- [x] **MotionBERT lower body distortion**: Fixed in Phase 2 via bone length clamping [0.5x, 1.5x] of defaults + increased bone_length_lr. VW-SI-MPJPE improved from 18.86 to 16.83 cm.
- [x] **MediaPipe wrist/ankle regression**: Fixed in Phase 3 via anchor penalty (anchor_weight=1000) anchoring to raw detector positions. VW-SI-MPJPE improved from 15.86 (regression) to 15.40 cm (improvement).

### High Priority
- [x] **Implement blur sigma annealing**: Implemented in Phase 3. Linear interpolation from sigma_start to sigma_end over steps. Used by MotionBERT (16->4).
- [x] **Revert motionbert-single.json**: Reverted sigma=32 to use blur annealing 16->4.
- [x] **Per-pipeline optimization configs**: Implemented in Phase 3. PipelineConfig.optimization dict merged on top of shared config.
- [x] **MediaPipe improvement now 2.7% avg on 5 examples**: Tuned to sigma=4, anchor=200, 50 steps. Validated on 5 examples. All examples positive except none negative with this config.
- [ ] **Blur annealing performance**: Re-blurs all heatmaps via scipy every step during annealing. For 50 steps this adds several seconds. Consider pre-computing a few sigma levels or caching. Verify by timing optimize() with and without annealing.

### Medium Priority
- [ ] **Add root-relative SI-MPJPE as supplementary metric**: Camera-space SI is 26-29% worse than root-relative SI because it must simultaneously fix depth and skeleton scale. Root-relative helps diagnose which is the bottleneck.
- [ ] **Investigate SH heatmap quality for lower body**: MotionBERT knees have 467px 2D reprojection error. Need to visually inspect SH heatmaps for knee channels when legs are partially out of frame.
- [x] **25-example configs updated**: Per-pipeline optimization settings applied to mediapipe-local-25-examples.json and motionbert-local-25-examples.json.
- [ ] **Run full 25-example validation**: Configs updated but 25-example run not yet executed. Need to verify aggregate metrics meet targets (>2% improvement, <10 cm average).

### Low Priority
- [ ] **Consider early stopping**: Monitor VW-SI-MPJPE during optimization and stop if it plateaus or regresses.
- [ ] **Audit visibility weights for fairness**: Currently ankles get 0% visibility weight, meaning their massive errors are invisible to the primary metric. This is by design but should be documented.
