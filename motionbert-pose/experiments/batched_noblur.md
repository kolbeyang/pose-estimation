---
name: batched_noblur
num_steps: 100
learning_rate: 0.001
bone_length_lr: 0.0001
sigma: 50.0
heatmap_blur_sigma: 0.0
position_penalty_weight: 50.0
init_anchor_weight: 5.0
mean_opt_mpjpe_cm: 41.65
mean_opt_p_mpjpe_cm: 34.59
mean_latency_per_frame_s: 0.430
mean_optimization_time_s: 0.8
---

## Description
Same as batched baseline but with heatmap blur sigma set to 0 (no Gaussian blur on heatmaps). Tested on 2 examples only.

## Results
- Mean Opt MPJPE: 41.65 cm (vs 41.21 cm with blur = +0.44 cm worse)
- Optimization time: 0.8s (slightly faster, blur is minimal overhead)

## Conclusion
Removing heatmap blur slightly degrades accuracy. The blur widens the gradient basin, helping optimization converge better. Keep blur at 4.0.
