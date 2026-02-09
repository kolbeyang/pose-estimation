"""
Systematic experiments to optimize SGLD performance.
Run ablation studies to understand what's degrading performance.
"""
import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime

import torch

from optimization import evaluate_results, run_optimization
from utils import OptimizationConfig
from video import Video

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    opt_config: OptimizationConfig
    description: str = ""


def run_experiment(video: Video, exp_config: ExperimentConfig, exp_dir: str):
    """Run a single experiment and log results."""
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT: {exp_config.name}")
    logger.info(f"Description: {exp_config.description}")
    logger.info("=" * 80)

    # Run optimization
    opt_result = run_optimization(video, exp_config.opt_config)
    eval_result = evaluate_results(opt_result, video, exp_config.opt_config)

    # Log results
    logger.info(f"Results: score={eval_result.pred_total_score:.4f}, MPJPE={eval_result.mpjpe:.4f}")

    # Save results
    result = {
        "name": exp_config.name,
        "description": exp_config.description,
        "config": asdict(exp_config.opt_config),
        "score": eval_result.pred_total_score,
        "mpjpe": eval_result.mpjpe,
        "gt_score": eval_result.gt_total_score,
    }

    result_file = os.path.join(exp_dir, f"{exp_config.name}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def get_baseline_config() -> OptimizationConfig:
    """Get baseline config (no SGLD, like commit e2b1777)."""
    return OptimizationConfig(
        num_steps=100,
        learning_rate=0.1,
        lr_min=0.1,  # No annealing
        noise_temperature=0.0,  # No SGLD noise
        num_runs=1,
        position_penalty_weight=1.0,
        ab_rotation_penalty_weight=0.5,
        bc_rotation_penalty_weight=0.3,
    )


def get_current_config() -> OptimizationConfig:
    """Get current config (with SGLD)."""
    return OptimizationConfig(
        num_steps=1000,
        learning_rate=0.1,
        lr_min=1e-5,
        noise_temperature=0.01,
        num_runs=1,
        position_penalty_weight=0.4,
        ab_rotation_penalty_weight=0.5,
        bc_rotation_penalty_weight=0.3,
    )


def main():
    parser = argparse.ArgumentParser(description="Run systematic experiments")
    parser.add_argument("folder_path", help="Path to saved video folder")
    parser.add_argument("--experiment", "-e",
                       choices=["baseline", "all", "noise", "optimizer", "penalty", "schedule"],
                       default="all",
                       help="Which experiment(s) to run")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    # Load video
    video = Video.load(args.folder_path)
    logger.info(f"Loaded video from: {args.folder_path}")
    logger.info(f"Frame count: {video.get_frame_count()}")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"exp-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Saving results to: {exp_dir}")

    # Define experiments
    baseline_cfg = get_baseline_config()
    current_cfg = get_current_config()

    experiments = []

    # Always run baseline for comparison
    if args.experiment in ["baseline", "all"]:
        experiments.append(ExperimentConfig(
            name="baseline",
            opt_config=baseline_cfg,
            description="Baseline (no SGLD, Adam-like with SGD lr=0.1 constant)"
        ))

    if args.experiment in ["all", "noise"]:
        # Hypothesis 1: Noise scale too high
        for temp in [0.001, 0.0001, 0.00001]:
            experiments.append(ExperimentConfig(
                name=f"noise_temp_{temp}",
                opt_config=replace(current_cfg, noise_temperature=temp),
                description=f"SGLD with noise_temperature={temp}"
            ))

    if args.experiment in ["all", "optimizer"]:
        # Hypothesis 2: Optimizer mismatch
        # Note: We can't easily switch to Adam in current code, but we can test SGD without SGLD
        experiments.append(ExperimentConfig(
            name="sgd_no_noise",
            opt_config=replace(current_cfg, noise_temperature=0.0),
            description="SGD with cosine annealing, no SGLD noise"
        ))

    if args.experiment in ["all", "penalty"]:
        # Hypothesis 3: Penalty weight mismatch
        experiments.append(ExperimentConfig(
            name="penalty_1.0",
            opt_config=replace(current_cfg, position_penalty_weight=1.0),
            description="SGLD with position_penalty_weight=1.0 (baseline value)"
        ))
        experiments.append(ExperimentConfig(
            name="penalty_0.6",
            opt_config=replace(current_cfg, position_penalty_weight=0.6),
            description="SGLD with position_penalty_weight=0.6 (intermediate)"
        ))

    if args.experiment in ["all", "schedule"]:
        # Hypothesis 4: LR schedule incompatibility
        experiments.append(ExperimentConfig(
            name="constant_lr",
            opt_config=replace(current_cfg, lr_min=0.1),
            description="SGLD with constant LR (no annealing)"
        ))
        experiments.append(ExperimentConfig(
            name="higher_lr_min",
            opt_config=replace(current_cfg, lr_min=0.01),
            description="SGLD with higher lr_min=0.01"
        ))

    # Run all experiments
    results = []
    for exp_config in experiments:
        result = run_experiment(video, exp_config, exp_dir)
        results.append(result)

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)

    best_score = results[0]["score"]
    for i, result in enumerate(results):
        marker = " <-- BEST" if i == 0 else ""
        improvement = result["score"] - results[-1]["score"]
        logger.info(f"{result['name']:25s} | score={result['score']:7.2f} | MPJPE={result['mpjpe']:.4f} | Δ={improvement:+6.2f}{marker}")

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
