---
name: batched_full
num_steps: 100
learning_rate: 0.001
bone_length_lr: 0.0001
sigma: 50.0
heatmap_blur_sigma: 4.0
position_penalty_weight: 50.0
init_anchor_weight: 5.0
mean_opt_mpjpe_cm: 38.69
mean_opt_p_mpjpe_cm: 27.18
mean_opt_mpjve_cm: 3.75
mean_latency_per_frame_s: 0.520
mean_optimization_time_s: 0.961
---

## Description
Fully vectorized optimization: batch FK across frames (forward_kinematics_batch), batch grid_sample across all frames and joints, vectorized motion/rotation penalties. Same hyperparameters as baseline. All 17 successful examples.

## Results
- Mean Opt MPJPE: 38.69 cm (17 examples)
- Mean Det MPJPE: 40.06 cm (improvement: +1.4 cm)
- Optimization time: 0.96s (vs 61.5s unbatched = 64x faster)
- Per-step: ~10ms (vs ~608ms = 60x faster)
- Total time: 76.6s (dominated by detection at 75.5s = 98.6% of time)
- Latency/frame: 0.52s

## Conclusion
Unequivocal improvement. Vectorization produces identical loss curves and final MPJPE while reducing optimization time from 61.5s to <1s. Detection now dominates end-to-end time at 98.6%. Further optimization gains require reducing detection overhead.
