# TODO

## Joint Alignment Revision (2026-04-11)

All items resolved.

- [x] **REST_DIRECTIONS[9] suboptimal**: Fixed — changed from `[0, -0.5, -0.866]` to `[0, -0.866, -0.5]` (mostly up, somewhat forward).
- [x] **Stale Nose synthesis label in verify_keypoint_mapping.py**: Fixed — now reads `"synth: 30% UpperNeck(MPII[8])→HeadTop(MPII[9])"`.
- [x] **Stale annotation in verify_keypoint_mapping.py draw_panel_17j**: Fixed — now reads `"30% UpperNeck(MPII[8])→HeadTop(MPII[9])"`.
- [x] **single_frame_heatmap.py broken import**: Fixed — computes `_EVAL_MPII_CHANNELS` from `SKELETON_TO_MPII_HEATMAP` and `EVAL_JOINTS` instead of importing non-existent symbol.
