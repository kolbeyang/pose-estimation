"""
Fast iteration experiments with 50 steps.
Test key hypotheses quickly.
"""
import argparse
import json
import logging
from dataclasses import replace

from experiment import ExperimentConfig, get_current_config, run_experiment
from video import Video

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fast 50-step experiments")
    parser.add_argument("folder_path", help="Path to saved video folder")
    args = parser.parse_args()

    video = Video.load(args.folder_path)
    logger.info(f"Loaded video: {video.get_frame_count()} frames")

    # Base: 50 steps, no noise, cosine annealing
    base = get_current_config()
    base = replace(base, num_steps=50, noise_temperature=0.0, num_runs=1)

    experiments = [
        # Test noise levels
        ExperimentConfig("noise_0.0", replace(base, noise_temperature=0.0), "No noise"),
        ExperimentConfig("noise_1e-5", replace(base, noise_temperature=1e-5), "Tiny noise"),
        ExperimentConfig("noise_1e-3", replace(base, noise_temperature=1e-3), "Small noise"),
        ExperimentConfig("noise_0.01", replace(base, noise_temperature=0.01), "Original noise"),

        # Test position penalties (no noise)
        ExperimentConfig("pos_0.1", replace(base, position_penalty_weight=0.1), "Very low penalty"),
        ExperimentConfig("pos_0.2", replace(base, position_penalty_weight=0.2), "Low penalty"),
        ExperimentConfig("pos_0.3", replace(base, position_penalty_weight=0.3), "Medium penalty"),
        ExperimentConfig("pos_0.4", replace(base, position_penalty_weight=0.4), "Default penalty"),
        ExperimentConfig("pos_0.6", replace(base, position_penalty_weight=0.6), "High penalty"),

        # Combinations
        ExperimentConfig("best_guess",
            replace(base, position_penalty_weight=0.2, noise_temperature=1e-6),
            "pos=0.2 + tiny noise"),
    ]

    results = []
    for exp in experiments:
        result = run_experiment(video, exp, "/tmp")
        results.append(result)
        logger.info(f"  {exp.name:20s} -> score={result['score']:7.2f}, mpjpe={result['mpjpe']:.4f}")

    # Summary
    print("\n" + "="*70)
    print("FAST EXPERIMENT RESULTS (50 steps)")
    print("="*70)
    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results):
        marker = " ⭐ BEST" if i == 0 else ""
        vs_target = r['score'] - (-16.0)
        status = "✅" if vs_target > 0 else "❌"
        print(f"{status} {r['name']:20s} | score={r['score']:7.2f} | Δ target={vs_target:+6.2f}{marker}")

    print(f"\nBest: {results[0]['name']} with score {results[0]['score']:.2f}")
    print(f"Target: -16.00")
    print(f"Gap: {results[0]['score'] - (-16.0):+.2f}")


if __name__ == "__main__":
    main()
