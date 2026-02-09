"""
Phase 2: SGLD Noise-Penalty Grid Search

Systematic grid search to find optimal noise_temperature + penalty combination
that achieves MPJPE ≤ 0.45 with SGLD enabled.
"""
import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from itertools import product

from optimization import evaluate_results, run_optimization
from utils import OptimizationConfig
from video import Video

logger = logging.getLogger(__name__)


def run_grid_experiment(
    video: Video,
    name: str,
    config: OptimizationConfig,
    exp_dir: str,
):
    """Run a single grid search experiment."""
    logger.info(f"Running: {name}")

    # Run optimization
    opt_result = run_optimization(video, config)
    eval_result = evaluate_results(opt_result, video, config)

    # Log results
    logger.info(
        f"  MPJPE={eval_result.mpjpe:.4f}, score={eval_result.pred_total_score:.4f}"
    )

    # Save results
    result = {
        "name": name,
        "config": asdict(config),
        "mpjpe": eval_result.mpjpe,
        "score": eval_result.pred_total_score,
        "gt_score": eval_result.gt_total_score,
    }

    result_file = os.path.join(exp_dir, f"{name}.json")
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_core_noise_sweep(video: Video, exp_dir: str):
    """
    Phase 2.1: Core noise sweep
    Fix penalties at (pos=0.2, ab=0.5, bc=0.3), lr=0.5, sweep noise
    """
    logger.info("=" * 80)
    logger.info("PHASE 2.1: Core Noise Sweep")
    logger.info("=" * 80)

    noise_levels = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    results = []

    for noise_temp in noise_levels:
        config = OptimizationConfig(
            num_steps=100,
            learning_rate=0.5,
            lr_min=1e-5,
            noise_temperature=noise_temp,
            num_runs=1,
            position_penalty_weight=0.2,
            ab_rotation_penalty_weight=0.5,
            bc_rotation_penalty_weight=0.3,
        )

        name = f"noise_{noise_temp:.0e}"
        result = run_grid_experiment(video, name, config, exp_dir)
        results.append(result)

    return results


def run_penalty_tuning(video: Video, exp_dir: str, best_noise_temps: list[float]):
    """
    Phase 2.2: Penalty tuning
    Take best 1-2 noise levels, grid search pos × ab × bc
    """
    logger.info("=" * 80)
    logger.info("PHASE 2.2: Penalty Tuning")
    logger.info(f"Using noise temperatures: {best_noise_temps}")
    logger.info("=" * 80)

    position_weights = [0.15, 0.2, 0.25, 0.3]
    ab_rotation_weights = [0.4, 0.5, 0.6]
    bc_rotation_weights = [0.2, 0.3, 0.4]

    results = []

    for noise_temp, pos, ab, bc in product(
        best_noise_temps, position_weights, ab_rotation_weights, bc_rotation_weights
    ):
        config = OptimizationConfig(
            num_steps=100,
            learning_rate=0.5,
            lr_min=1e-5,
            noise_temperature=noise_temp,
            num_runs=1,
            position_penalty_weight=pos,
            ab_rotation_penalty_weight=ab,
            bc_rotation_penalty_weight=bc,
        )

        name = f"noise_{noise_temp:.0e}_pos_{pos:.2f}_ab_{ab:.2f}_bc_{bc:.2f}"
        result = run_grid_experiment(video, name, config, exp_dir)
        results.append(result)

    return results


def run_lr_tuning(video: Video, exp_dir: str, best_config: dict):
    """
    Phase 2.3: LR fine-tune
    Test lr variants with best noise+penalty combo
    """
    logger.info("=" * 80)
    logger.info("PHASE 2.3: Learning Rate Tuning")
    logger.info("=" * 80)

    learning_rates = [0.3, 0.4, 0.5]
    results = []

    base_config = best_config["config"]

    for lr in learning_rates:
        config = OptimizationConfig(
            num_steps=100,
            learning_rate=lr,
            lr_min=1e-5,
            noise_temperature=base_config["noise_temperature"],
            num_runs=1,
            position_penalty_weight=base_config["position_penalty_weight"],
            ab_rotation_penalty_weight=base_config["ab_rotation_penalty_weight"],
            bc_rotation_penalty_weight=base_config["bc_rotation_penalty_weight"],
        )

        name = f"lr_{lr:.1f}"
        result = run_grid_experiment(video, name, config, exp_dir)
        results.append(result)

    return results


def print_summary(results: list[dict], phase_name: str):
    """Print summary table sorted by MPJPE."""
    logger.info("=" * 80)
    logger.info(f"{phase_name} SUMMARY")
    logger.info("=" * 80)

    # Sort by MPJPE
    sorted_results = sorted(results, key=lambda x: x["mpjpe"])

    logger.info(f"{'Name':<50s} | {'MPJPE':>8s} | {'Score':>8s}")
    logger.info("-" * 80)

    for i, result in enumerate(sorted_results[:10]):  # Top 10
        marker = " <-- BEST" if i == 0 else ""
        logger.info(
            f"{result['name']:<50s} | {result['mpjpe']:8.4f} | {result['score']:8.2f}{marker}"
        )

    return sorted_results


def main():
    parser = argparse.ArgumentParser(
        description="SGLD grid search for optimal noise + penalty combination"
    )
    parser.add_argument("folder_path", help="Path to saved video folder")
    parser.add_argument(
        "--phase",
        choices=["core_noise", "penalty_tuning", "lr_tuning", "all"],
        default="all",
        help="Which phase to run",
    )
    parser.add_argument(
        "--best-noise",
        type=float,
        nargs="+",
        help="Best noise temperature(s) from core_noise phase (for penalty_tuning)",
    )
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
    exp_dir = os.path.join("experiments", f"sgld-grid-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Saving results to: {exp_dir}")

    all_results = []
    best_noise_temps = None

    # Phase 2.1: Core noise sweep
    if args.phase in ["core_noise", "all"]:
        noise_results = run_core_noise_sweep(video, exp_dir)
        all_results.extend(noise_results)
        sorted_noise = print_summary(noise_results, "CORE NOISE SWEEP")

        # Save intermediate summary
        with open(os.path.join(exp_dir, "core_noise_summary.json"), "w") as f:
            json.dump(sorted_noise, f, indent=2)

        # Automatically select best 2 noise levels for penalty tuning
        best_noise_temps = [r["config"]["noise_temperature"] for r in sorted_noise[:2]]
        logger.info(f"\nBest noise temperatures: {best_noise_temps}")

    # Phase 2.2: Penalty tuning
    if args.phase in ["penalty_tuning", "all"]:
        if args.phase == "penalty_tuning" and not args.best_noise:
            logger.error(
                "Must provide --best-noise when running penalty_tuning phase alone"
            )
            return

        if args.phase == "penalty_tuning":
            best_noise_temps = args.best_noise
        # else: best_noise_temps already set from phase 2.1

        if best_noise_temps is None:
            logger.error("No noise temperatures available for penalty tuning")
            return

        penalty_results = run_penalty_tuning(video, exp_dir, best_noise_temps)
        all_results.extend(penalty_results)
        sorted_penalty = print_summary(penalty_results, "PENALTY TUNING")

        # Save intermediate summary
        with open(os.path.join(exp_dir, "penalty_tuning_summary.json"), "w") as f:
            json.dump(sorted_penalty, f, indent=2)

    # Phase 2.3: LR tuning
    if args.phase in ["lr_tuning", "all"]:
        if args.phase == "all":
            # Use best config from all previous experiments
            best_config = sorted(all_results, key=lambda x: x["mpjpe"])[0]
        else:
            # Must load best config from previous run
            logger.error(
                "lr_tuning phase requires running core_noise and penalty_tuning first"
            )
            return

        lr_results = run_lr_tuning(video, exp_dir, best_config)
        all_results.extend(lr_results)
        print_summary(lr_results, "LEARNING RATE TUNING")

    # Final summary
    if all_results:
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY (All Experiments)")
        logger.info("=" * 80)

        sorted_all = sorted(all_results, key=lambda x: x["mpjpe"])

        logger.info(f"{'Name':<50s} | {'MPJPE':>8s} | {'Score':>8s}")
        logger.info("-" * 80)

        for i, result in enumerate(sorted_all[:15]):  # Top 15
            marker = " <-- BEST" if i == 0 else ""
            logger.info(
                f"{result['name']:<50s} | {result['mpjpe']:8.4f} | {result['score']:8.2f}{marker}"
            )

        # Target comparison
        best_mpjpe = sorted_all[0]["mpjpe"]
        baseline_mpjpe = 0.401  # From no-noise baseline
        target_mpjpe = 0.45

        logger.info("")
        logger.info(f"Best MPJPE achieved: {best_mpjpe:.4f}")
        logger.info(f"No-noise baseline: {baseline_mpjpe:.4f}")
        logger.info(f"Target MPJPE: {target_mpjpe:.4f}")

        if best_mpjpe <= target_mpjpe:
            logger.info(f"✓ TARGET ACHIEVED! ({best_mpjpe:.4f} ≤ {target_mpjpe})")
        else:
            gap = best_mpjpe - target_mpjpe
            logger.info(f"✗ Target not reached (gap: {gap:.4f})")

        # Save final summary
        summary_file = os.path.join(exp_dir, "summary.json")
        with open(summary_file, "w") as f:
            json.dump(
                {
                    "best_mpjpe": best_mpjpe,
                    "baseline_mpjpe": baseline_mpjpe,
                    "target_mpjpe": target_mpjpe,
                    "best_config": sorted_all[0],
                    "all_results": sorted_all,
                },
                f,
                indent=2,
            )

        logger.info(f"\nComplete results saved to: {summary_file}")


if __name__ == "__main__":
    main()
