"""
Experiment: optimize with noise_temperature=0.1 using noise_start_step.
Deterministic SGD converges first, then SGLD noise kicks in.
"""
import argparse
import json
import logging
import os
from dataclasses import replace
from datetime import datetime

from experiment import ExperimentConfig, run_experiment
from utils import OptimizationConfig
from video import Video

logger = logging.getLogger(__name__)

VIDEO_PATH = "/Users/kolbeyang/documents/School/spring_2026/capstone/pose-estimation/images/arm-video-20260204_143220_995716"


def phase1_noise_start_step(video, exp_dir):
    """Find optimal noise_start_step with noise_temperature=0.1."""
    logger.info("=" * 80)
    logger.info("PHASE 1: noise_start_step sweep")
    logger.info("=" * 80)

    base = OptimizationConfig(
        num_steps=100,
        learning_rate=0.5,
        lr_min=1e-5,
        noise_temperature=0.1,
        num_runs=1,
        position_penalty_weight=0.30,
        ab_rotation_penalty_weight=0.50,
        bc_rotation_penalty_weight=0.40,
    )

    results = []
    for start_step in [0, 20, 40, 50, 60, 70, 80, 90]:
        cfg = replace(base, noise_start_step=start_step)
        exp = ExperimentConfig(
            name=f"start_{start_step}",
            opt_config=cfg,
            description=f"noise_start_step={start_step}, noise_temp=0.1, lr=0.5, 100 steps",
        )
        result = run_experiment(video, exp, exp_dir)
        results.append(result)

    # Find best
    best = min(results, key=lambda r: r["mpjpe"])
    logger.info(f"Phase 1 best: {best['name']} (MPJPE={best['mpjpe']:.4f}, score={best['score']:.2f})")
    return results, best


def phase2_lr_steps(video, exp_dir, best_start_step):
    """Retune LR and steps with best noise_start_step."""
    logger.info("=" * 80)
    logger.info(f"PHASE 2: LR + steps sweep (noise_start_step={best_start_step})")
    logger.info("=" * 80)

    configs = [
        (0.3, 100), (0.5, 100),
        (0.3, 200), (0.5, 200),
        (0.3, 300), (0.5, 300),
    ]

    results = []
    for lr, steps in configs:
        # Scale noise_start_step proportionally to total steps
        scaled_start = int(best_start_step * steps / 100)
        cfg = OptimizationConfig(
            num_steps=steps,
            learning_rate=lr,
            lr_min=1e-5,
            noise_temperature=0.1,
            num_runs=1,
            noise_start_step=scaled_start,
            position_penalty_weight=0.30,
            ab_rotation_penalty_weight=0.50,
            bc_rotation_penalty_weight=0.40,
        )
        exp = ExperimentConfig(
            name=f"lr{lr}_steps{steps}_start{scaled_start}",
            opt_config=cfg,
            description=f"lr={lr}, steps={steps}, noise_start_step={scaled_start}",
        )
        result = run_experiment(video, exp, exp_dir)
        results.append(result)

    best = min(results, key=lambda r: r["mpjpe"])
    logger.info(f"Phase 2 best: {best['name']} (MPJPE={best['mpjpe']:.4f}, score={best['score']:.2f})")
    return results, best


def phase3_ensemble(video, exp_dir, best_config):
    """Test ensemble with best config."""
    logger.info("=" * 80)
    logger.info("PHASE 3: Ensemble scaling")
    logger.info("=" * 80)

    results = []
    for num_runs in [5, 10, 20]:
        cfg = replace(best_config, num_runs=num_runs)
        exp = ExperimentConfig(
            name=f"ensemble_{num_runs}",
            opt_config=cfg,
            description=f"Ensemble with {num_runs} runs",
        )
        result = run_experiment(video, exp, exp_dir)
        results.append(result)

    best = min(results, key=lambda r: r["mpjpe"])
    logger.info(f"Phase 3 best: {best['name']} (MPJPE={best['mpjpe']:.4f}, score={best['score']:.2f})")
    return results, best


def main():
    parser = argparse.ArgumentParser(description="High noise (0.1) experiments")
    parser.add_argument("folder_path", nargs="?", default=VIDEO_PATH, help="Path to saved video folder")
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    video = Video.load(args.folder_path)
    logger.info(f"Loaded video: {video.get_frame_count()} frames")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"high-noise-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    all_results = {}

    if args.phase in ["1", "all"]:
        p1_results, p1_best = phase1_noise_start_step(video, exp_dir)
        all_results["phase1"] = p1_results
        best_start_step = int(p1_best["name"].split("_")[1])

    if args.phase in ["2", "all"]:
        if "best_start_step" not in dir():
            best_start_step = 70  # Default if skipping phase 1
        p2_results, p2_best = phase2_lr_steps(video, exp_dir, best_start_step)
        all_results["phase2"] = p2_results
        best_cfg = p2_best["config"]
        best_opt_config = OptimizationConfig(**{
            k: v for k, v in best_cfg.items()
            if k in OptimizationConfig.__dataclass_fields__
        })

    if args.phase in ["3", "all"]:
        if "best_opt_config" not in dir():
            best_opt_config = OptimizationConfig(
                noise_temperature=0.1, noise_start_step=70,
                learning_rate=0.5, num_steps=100,
            )
        p3_results, p3_best = phase3_ensemble(video, exp_dir, best_opt_config)
        all_results["phase3"] = p3_results

    # Save summary
    summary_file = os.path.join(exp_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
