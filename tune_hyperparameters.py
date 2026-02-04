import argparse
import itertools
import logging

import numpy as np
import torch

from optimization import OptimizationConfig, run_optimization
from video import Video

logger = logging.getLogger(__name__)

# Grid search values for each hyperparameter
PARAM_GRID = {
    "learning_rate": [0.005, 0.01, 0.02],
    "position_penalty_weight": [0.5, 1.0, 2.0],
    "ab_rotation_penalty_weight": [0.25, 0.5, 1.0],
    "bc_rotation_penalty_weight": [0.15, 0.3, 0.6],
}


def evaluate_config(video: Video, config: OptimizationConfig) -> float:
    """Run optimization and return negative mean error (higher is better)."""
    torch.manual_seed(42)  # Fixed seed for reproducibility
    result = run_optimization(video, config)

    # Compute mean Euclidean distance between predicted and ground truth joints
    total_error = 0.0
    num_joints = 0

    for pred_coords, gt_coords in zip(result.pred_coords, result.gt_coords):
        for joint in ["a", "b", "c"]:
            pred = pred_coords[joint]
            gt = gt_coords[joint]
            if isinstance(gt, torch.Tensor):
                gt = gt.detach().numpy()
            total_error += np.linalg.norm(pred - gt)
            num_joints += 1

    mean_error = total_error / num_joints
    return -mean_error  # Negative so higher is better


def grid_search(video: Video) -> tuple[OptimizationConfig, float]:
    """Run grid search over hyperparameter combinations."""
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())

    combinations = list(itertools.product(*param_values))
    total = len(combinations)

    logger.info("Grid search: %d combinations", total)

    best_config = None
    best_score = float("-inf")

    for i, values in enumerate(combinations):
        params = dict(zip(param_names, values, strict=True))
        config = OptimizationConfig(**params)  # type: ignore[arg-type]

        score = evaluate_config(video, config)

        mean_error = -score  # Convert back to positive error
        logger.info(
            "[%d/%d] mean_error=%.4f | lr=%.3f pos=%.2f ab=%.2f bc=%.2f",
            i + 1, total, mean_error,
            params["learning_rate"],
            params["position_penalty_weight"],
            params["ab_rotation_penalty_weight"],
            params["bc_rotation_penalty_weight"],
        )

        if score > best_score:
            best_score = score
            best_config = config

    assert best_config is not None
    return best_config, best_score


def main():
    parser = argparse.ArgumentParser(description="Tune optimization hyperparameters via grid search")
    parser.add_argument("folder_path", help="Path to saved video folder")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    video = Video.load(args.folder_path)
    logger.info("Loaded video: %s (%d frames)", args.folder_path, video.get_frame_count())

    best_config, best_score = grid_search(video)

    logger.info("=" * 50)
    logger.info("Best mean error: %.4f", -best_score)
    logger.info("Best config:")
    logger.info("  learning_rate: %.4f", best_config.learning_rate)
    logger.info("  position_penalty_weight: %.4f", best_config.position_penalty_weight)
    logger.info("  ab_rotation_penalty_weight: %.4f", best_config.ab_rotation_penalty_weight)
    logger.info("  bc_rotation_penalty_weight: %.4f", best_config.bc_rotation_penalty_weight)


if __name__ == "__main__":
    main()
