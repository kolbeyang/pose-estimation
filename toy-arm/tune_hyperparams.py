"""Hyperparameter tuning script for pose optimization."""

import logging
from itertools import product

from optimization import evaluate_results, run_optimization
from utils import OptimizationConfig
from video import Video

logging.basicConfig(level=logging.WARNING)

VIDEO_PATH = "/Users/kolbeyang/documents/School/spring_2026/capstone/pose-estimation/images/arm-video-20260204_143220_995716"

# Final comparison: explore around both promising regions
STEPS = 100

configs = [
    # Current baseline region (pos=0.4)
    {"name": "p=0.4,ab=0.5,bc=0.3", "config": OptimizationConfig(num_steps=STEPS, position_penalty_weight=0.4, ab_rotation_penalty_weight=0.5, bc_rotation_penalty_weight=0.3)},
    {"name": "p=0.4,ab=0.3,bc=0.2", "config": OptimizationConfig(num_steps=STEPS, position_penalty_weight=0.4, ab_rotation_penalty_weight=0.3, bc_rotation_penalty_weight=0.2)},
    {"name": "p=0.4,ab=0.4,bc=0.2", "config": OptimizationConfig(num_steps=STEPS, position_penalty_weight=0.4, ab_rotation_penalty_weight=0.4, bc_rotation_penalty_weight=0.2)},
    {"name": "p=0.5,ab=0.4,bc=0.2", "config": OptimizationConfig(num_steps=STEPS, position_penalty_weight=0.5, ab_rotation_penalty_weight=0.4, bc_rotation_penalty_weight=0.2)},
    {"name": "p=0.3,ab=0.4,bc=0.2", "config": OptimizationConfig(num_steps=STEPS, position_penalty_weight=0.3, ab_rotation_penalty_weight=0.4, bc_rotation_penalty_weight=0.2)},
    # Higher penalty region
    {"name": "p=2.0,ab=0.5,bc=0.8", "config": OptimizationConfig(num_steps=STEPS, position_penalty_weight=2.0, ab_rotation_penalty_weight=0.5, bc_rotation_penalty_weight=0.8)},
    {"name": "p=1.5,ab=0.5,bc=0.8", "config": OptimizationConfig(num_steps=STEPS, position_penalty_weight=1.5, ab_rotation_penalty_weight=0.5, bc_rotation_penalty_weight=0.8)},
    {"name": "p=2.0,ab=0.4,bc=0.6", "config": OptimizationConfig(num_steps=STEPS, position_penalty_weight=2.0, ab_rotation_penalty_weight=0.4, bc_rotation_penalty_weight=0.6)},
]

video = Video.load(VIDEO_PATH)
print(f"Testing {len(configs)} configurations (steps={STEPS})...\n")

results = []
for exp in configs:
    name = exp["name"]
    config = exp["config"]

    opt_result = run_optimization(video, config)
    eval_result = evaluate_results(opt_result, video, config)

    results.append((name, eval_result.mpjpe, config))
    print(f"{name:25s} MPJPE: {eval_result.mpjpe:.4f}")

print("\n" + "=" * 50)
print("Top 10 by MPJPE:")
print("=" * 50)
for name, mpjpe, _ in sorted(results, key=lambda x: x[1])[:10]:
    print(f"{name:25s} MPJPE: {mpjpe:.4f}")
