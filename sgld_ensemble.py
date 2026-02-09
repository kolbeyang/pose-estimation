"""
Phase 4: SGLD Multi-Run Ensemble

Leverage SGLD's exploration to generate diverse solutions, then select or average
for better MPJPE performance.
"""
import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime

import numpy as np
import torch

from model.arm import Arm
from optimization import evaluate_results, run_optimization
from utils import OptimizationConfig
from video import Video

logger = logging.getLogger(__name__)


def compute_pairwise_diversity(all_coords_list: list[list[dict]]):
    """
    Compute pairwise MPJPE between runs to measure diversity.

    Args:
        all_coords_list: List of runs, each containing list of frame coords

    Returns:
        Mean pairwise MPJPE standard deviation across frames
    """
    num_runs = len(all_coords_list)
    num_frames = len(all_coords_list[0])

    frame_diversities = []

    for frame_idx in range(num_frames):
        # Get all predictions for this frame
        frame_preds = [run_coords[frame_idx] for run_coords in all_coords_list]

        # Compute pairwise distances
        pairwise_mpjpe = []
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                # Compute MPJPE between run i and run j for this frame
                mpjpe = 0.0
                for joint in ["a", "b", "c"]:
                    diff = frame_preds[i][joint] - frame_preds[j][joint]
                    mpjpe += np.linalg.norm(diff)
                mpjpe /= 3
                pairwise_mpjpe.append(mpjpe)

        # Standard deviation of pairwise distances for this frame
        frame_diversities.append(np.std(pairwise_mpjpe))

    return np.mean(frame_diversities)


def ensemble_average(all_coords_list: list[list[dict]]):
    """
    Average joint predictions across all runs.

    Args:
        all_coords_list: List of runs, each containing list of frame coords

    Returns:
        List of averaged frame coords
    """
    num_runs = len(all_coords_list)
    num_frames = len(all_coords_list[0])

    averaged_coords = []

    for frame_idx in range(num_frames):
        avg_frame = {}
        for joint in ["a", "b", "c"]:
            # Average this joint across all runs
            joint_sum = np.zeros(3)
            for run_coords in all_coords_list:
                joint_sum += run_coords[frame_idx][joint]
            avg_frame[joint] = joint_sum / num_runs
        averaged_coords.append(avg_frame)

    return averaged_coords


def compute_mpjpe_from_coords(pred_coords_list: list[dict], gt_coords_list: list[dict]):
    """Compute MPJPE from coordinate lists."""
    num_frames = len(pred_coords_list)
    total_error = 0.0

    for i in range(num_frames):
        frame_error = 0.0
        for joint in ["a", "b", "c"]:
            pred = pred_coords_list[i][joint]
            gt = gt_coords_list[i][joint]
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().cpu().numpy()
            frame_error += np.linalg.norm(pred - gt)
        total_error += frame_error / 3

    return total_error / num_frames


def main():
    parser = argparse.ArgumentParser(
        description="SGLD ensemble strategies: best-of-N and averaging"
    )
    parser.add_argument("folder_path", help="Path to saved video folder")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of SGLD runs to generate (default: 10)",
    )
    parser.add_argument(
        "--noise-temperature",
        type=float,
        default=1e-6,
        help="Noise temperature to use (from grid search results)",
    )
    parser.add_argument(
        "--position-penalty",
        type=float,
        default=0.2,
        help="Position penalty weight",
    )
    parser.add_argument(
        "--ab-rotation-penalty",
        type=float,
        default=0.5,
        help="AB rotation penalty weight",
    )
    parser.add_argument(
        "--bc-rotation-penalty",
        type=float,
        default=0.3,
        help="BC rotation penalty weight",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.5,
        help="Learning rate",
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
    exp_dir = os.path.join("experiments", f"sgld-ensemble-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Saving results to: {exp_dir}")

    # Configure optimization
    config = OptimizationConfig(
        num_steps=100,
        learning_rate=args.learning_rate,
        lr_min=1e-5,
        noise_temperature=args.noise_temperature,
        num_runs=1,  # We'll run multiple times manually
        position_penalty_weight=args.position_penalty,
        ab_rotation_penalty_weight=args.ab_rotation_penalty,
        bc_rotation_penalty_weight=args.bc_rotation_penalty,
    )

    logger.info("=" * 80)
    logger.info(f"Running {args.num_runs} SGLD optimization runs...")
    logger.info(f"Config: noise_temp={config.noise_temperature:.0e}, lr={config.learning_rate}")
    logger.info("=" * 80)

    # Run multiple optimizations
    all_results = []
    all_eval_results = []
    all_coords_list = []

    for run_idx in range(args.num_runs):
        logger.info(f"\nRun {run_idx + 1}/{args.num_runs}")
        opt_result = run_optimization(video, config)
        eval_result = evaluate_results(opt_result, video, config)

        all_results.append(opt_result)
        all_eval_results.append(eval_result)
        all_coords_list.append(opt_result.pred_coords)

        logger.info(
            f"  MPJPE={eval_result.mpjpe:.4f}, score={eval_result.pred_total_score:.4f}"
        )

    # Strategy 1: Best-of-N
    logger.info("=" * 80)
    logger.info("STRATEGY 1: Best-of-N Selection")
    logger.info("=" * 80)

    best_idx = min(range(args.num_runs), key=lambda i: all_eval_results[i].mpjpe)
    best_mpjpe = all_eval_results[best_idx].mpjpe
    best_score = all_eval_results[best_idx].pred_total_score

    logger.info(f"Best run: {best_idx + 1}")
    logger.info(f"Best MPJPE: {best_mpjpe:.4f}")
    logger.info(f"Best score: {best_score:.4f}")

    # Strategy 2: Ensemble Average
    logger.info("=" * 80)
    logger.info("STRATEGY 2: Ensemble Average")
    logger.info("=" * 80)

    averaged_coords = ensemble_average(all_coords_list)

    # Get GT coords for comparison
    gt_coords_list = [all_results[0].gt_coords[i] for i in range(video.get_frame_count())]

    avg_mpjpe = compute_mpjpe_from_coords(averaged_coords, gt_coords_list)

    logger.info(f"Ensemble MPJPE: {avg_mpjpe:.4f}")

    # Compute diversity
    diversity = compute_pairwise_diversity(all_coords_list)
    logger.info(f"Pairwise diversity (std): {diversity:.4f}")

    # Summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    baseline_mpjpe = 0.401  # No-noise baseline
    target_mpjpe = 0.40  # Stretch goal

    # Individual run statistics
    all_mpjpe = [r.mpjpe for r in all_eval_results]
    mean_mpjpe = np.mean(all_mpjpe)
    std_mpjpe = np.std(all_mpjpe)

    logger.info(f"Individual runs:")
    logger.info(f"  Mean MPJPE: {mean_mpjpe:.4f} ± {std_mpjpe:.4f}")
    logger.info(f"  Best MPJPE: {best_mpjpe:.4f} (run {best_idx + 1})")
    logger.info(f"  Worst MPJPE: {max(all_mpjpe):.4f}")
    logger.info("")
    logger.info(f"Ensemble strategies:")
    logger.info(f"  Best-of-{args.num_runs}: {best_mpjpe:.4f}")
    logger.info(f"  Average: {avg_mpjpe:.4f}")
    logger.info("")
    logger.info(f"Baselines:")
    logger.info(f"  No-noise baseline: {baseline_mpjpe:.4f}")
    logger.info(f"  Stretch target: {target_mpjpe:.4f}")
    logger.info("")

    # Check if targets met
    if best_mpjpe < baseline_mpjpe:
        logger.info(f"✓ Best-of-N BEATS no-noise baseline! ({best_mpjpe:.4f} < {baseline_mpjpe:.4f})")
    elif best_mpjpe <= target_mpjpe:
        logger.info(f"✓ Best-of-N meets stretch target ({best_mpjpe:.4f} ≤ {target_mpjpe:.4f})")
    else:
        gap = best_mpjpe - baseline_mpjpe
        logger.info(f"Best-of-N gap to baseline: {gap:+.4f}")

    if avg_mpjpe < baseline_mpjpe:
        logger.info(f"✓ Ensemble average BEATS no-noise baseline! ({avg_mpjpe:.4f} < {baseline_mpjpe:.4f})")
    elif avg_mpjpe <= target_mpjpe:
        logger.info(f"✓ Ensemble average meets stretch target ({avg_mpjpe:.4f} ≤ {target_mpjpe:.4f})")
    else:
        gap = avg_mpjpe - baseline_mpjpe
        logger.info(f"Ensemble average gap to baseline: {gap:+.4f}")

    # Diversity validation
    logger.info("")
    if diversity > 0.05:
        logger.info(f"✓ Runs show good diversity (std={diversity:.4f} > 0.05)")
    else:
        logger.info(f"✗ Low diversity (std={diversity:.4f} ≤ 0.05) - SGLD may not be exploring effectively")

    # Save results
    results = {
        "config": asdict(config),
        "num_runs": args.num_runs,
        "individual_mpjpe": all_mpjpe,
        "mean_mpjpe": mean_mpjpe,
        "std_mpjpe": std_mpjpe,
        "best_of_n_mpjpe": best_mpjpe,
        "best_run_idx": best_idx,
        "ensemble_average_mpjpe": avg_mpjpe,
        "diversity_std": diversity,
        "baseline_mpjpe": baseline_mpjpe,
        "target_mpjpe": target_mpjpe,
    }

    summary_file = os.path.join(exp_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {summary_file}")


if __name__ == "__main__":
    main()
