# TODO

## Resolved
- [x] Add per_joint_visible_frames to results.json (Phase 1)
- [x] Fix MotionBERT knee/ankle bone length initialization (Phase 2)
- [x] Fix MediaPipe optimization regression (Phase 3)
- [x] Per-pipeline optimization configs (Phase 3c)
- [x] Blur sigma annealing support (Phase 3a)
- [x] Anchor penalty for good-quality detectors (Phase 3)

## Open
- [ ] Phase 4: Hyperparameter sweep on 25 examples
- [ ] Phase 5: Final validation on full 25 examples
- [ ] MediaPipe improvement is only 1.7% on single example -- need to verify >2% on 25 examples
- [ ] Consider per-joint anchor weights (currently uniform across all joints)
- [ ] Blur annealing is slow (re-blurs all heatmaps every step via scipy) -- consider caching or GPU blur
- [ ] Investigate whether MotionBERT can benefit from anchor penalty too (currently anchor_weight=0)
- [ ] The 25-example configs need updating with new per-pipeline optimization settings
