"""
Refined experiments to beat -15.12.
Based on findings: SGD + cosine annealing + no/tiny noise + position_penalty=0.4 works best.
"""
import argparse
import json
import logging
import os
from dataclasses import replace
from datetime import datetime

from experiment import ExperimentConfig, get_current_config, run_experiment
from utils import OptimizationConfig
from video import Video

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run refined experiments")
    parser.add_argument("folder_path", help="Path to saved video folder")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    # Load video
    video = Video.load(args.folder_path)
    logger.info(f"Loaded video from: {args.folder_path}")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"refined-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Saving results to: {exp_dir}")

    # Best config so far: SGD, no noise, cosine annealing, position_penalty=0.4
    base_cfg = get_current_config()
    base_cfg = replace(base_cfg, noise_temperature=0.0, num_runs=1)

    experiments = [
        # Baseline best
        ExperimentConfig(
            name="best_baseline",
            opt_config=base_cfg,
            description="Current best: SGD + cosine annealing + no noise + pos_penalty=0.4"
        ),

        # Test lower position penalty
        ExperimentConfig(
            name="pos_penalty_0.3",
            opt_config=replace(base_cfg, position_penalty_weight=0.3),
            description="Lower position penalty"
        ),
        ExperimentConfig(
            name="pos_penalty_0.2",
            opt_config=replace(base_cfg, position_penalty_weight=0.2),
            description="Even lower position penalty"
        ),
        ExperimentConfig(
            name="pos_penalty_0.5",
            opt_config=replace(base_cfg, position_penalty_weight=0.5),
            description="Higher position penalty"
        ),

        # Test different rotation penalties
        ExperimentConfig(
            name="ab_penalty_0.4",
            opt_config=replace(base_cfg, ab_rotation_penalty_weight=0.4),
            description="Lower AB rotation penalty"
        ),
        ExperimentConfig(
            name="bc_penalty_0.2",
            opt_config=replace(base_cfg, bc_rotation_penalty_weight=0.2),
            description="Lower BC rotation penalty"
        ),

        # Test more steps
        ExperimentConfig(
            name="steps_1500",
            opt_config=replace(base_cfg, num_steps=1500),
            description="More training steps"
        ),
        ExperimentConfig(
            name="steps_2000",
            opt_config=replace(base_cfg, num_steps=2000),
            description="Even more training steps"
        ),

        # Test tiny noise
        ExperimentConfig(
            name="tiny_noise_1e6",
            opt_config=replace(base_cfg, noise_temperature=1e-6),
            description="Extremely tiny SGLD noise"
        ),

        # Combination: lower penalties + more steps
        ExperimentConfig(
            name="combo_low_penalties_1500",
            opt_config=replace(base_cfg,
                             position_penalty_weight=0.3,
                             ab_rotation_penalty_weight=0.4,
                             bc_rotation_penalty_weight=0.2,
                             num_steps=1500),
            description="Combined: lower penalties + 1500 steps"
        ),
    ]

    # Run all experiments
    results = []
    for exp_config in experiments:
        result = run_experiment(video, exp_config, exp_dir)
        results.append(result)

    # Summary
    logger.info("=" * 80)
    logger.info("REFINED EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    # Sort by score (descending, higher is better)
    results.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"{'Name':<30s} | {'Score':>8s} | {'MPJPE':>8s} | {'vs best':>8s}")
    logger.info("-" * 80)

    best_score = results[0]["score"]
    for i, result in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        improvement = result["score"] - results[-1]["score"]
        logger.info(
            f"{result['name']:<30s} | {result['score']:8.2f} | {result['mpjpe']:8.4f} | "
            f"{improvement:+8.2f}{marker}"
        )

    # Save summary
    summary_file = os.path.join(exp_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nBest configuration: {results[0]['name']}")
    logger.info(f"Best score: {best_score:.2f}")
    logger.info(f"Target score: -16.00")
    logger.info(f"Gap to target: {best_score - (-16.0):.2f}")


if __name__ == "__main__":
    main()
