---
name: baseline_unbatched
num_steps: 100
learning_rate: 0.001
bone_length_lr: 0.0001
sigma: 50.0
heatmap_blur_sigma: 4.0
position_penalty_weight: 50.0
init_anchor_weight: 5.0
mean_opt_mpjpe_cm: 44.99
mean_opt_p_mpjpe_cm: 40.85
mean_opt_mpjve_cm: ~
mean_latency_per_frame_s: 1.354
mean_optimization_time_s: 61.5
---

## Description
Original non-batched optimization baseline. FK and scoring loop over frames in Python, each building individual autograd graphs. Profiled on 1 example only.

## Results
- Opt MPJPE: 44.99 cm (1 example: 171204_pose1_sample_0)
- Optimization time: 61.5s for 100 frames (608 ms/step)
- Per-step breakdown: FK+Proj=128ms, Scoring=100ms, Backward=363ms
- Detection time: 73.8s, Total: 135.4s

## Conclusion
The backward pass dominates step time (60%) because of the large autograd graph built from sequential Python loops. This is the primary optimization target.
