---
name: batched_50step_lr002_full
num_steps: 50
learning_rate: 0.002
bone_length_lr: 0.0001
sigma: 50.0
heatmap_blur_sigma: 4.0
position_penalty_weight: 50.0
init_anchor_weight: 5.0
mean_opt_mpjpe_cm: 38.69
mean_opt_p_mpjpe_cm: 27.30
mean_opt_mpjve_cm: 3.80
mean_latency_per_frame_s: 0.283
mean_optimization_time_s: 0.488
---

## Description
Batched optimization with reduced steps (50 vs 100) and 2x learning rate (0.002 vs 0.001). Tests whether convergence is reached faster with a higher LR.

## Results
- Mean Opt MPJPE: 38.69 cm (identical to 100-step baseline)
- Mean Opt P-MPJPE: 27.30 cm (vs 27.18 cm, effectively the same)
- Optimization time: 0.49s (vs 0.96s, 2x faster)
- Total time: 41.9s (detection dominates)
- Latency/frame: 0.28s

## Conclusion
50 steps at LR=0.002 achieves the same accuracy as 100 steps at LR=0.001, at half the optimization cost. Since optimization is already <1% of total time after batching, this is a minor improvement in absolute terms but confirms the optimizer converges well before 100 steps.
