"""
Test configs for faster convergence in fewer steps.
Try higher LR, different schedules, etc.
"""
import argparse
import logging
from dataclasses import replace

from experiment import ExperimentConfig, get_current_config, run_experiment
from video import Video

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fast convergence experiments")
    parser.add_argument("folder_path", help="Path to saved video folder")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps")
    args = parser.parse_args()

    video = Video.load(args.folder_path)
    logger.info(f"Testing with {args.steps} steps")

    base = get_current_config()
    base = replace(base, num_steps=args.steps, num_runs=1)

    experiments = [
        # Higher LR for faster convergence
        ExperimentConfig("lr_0.3_pos0.2",
            replace(base, learning_rate=0.3, position_penalty_weight=0.2, noise_temperature=0.0),
            "High LR, low penalty"),

        ExperimentConfig("lr_0.5_pos0.2",
            replace(base, learning_rate=0.5, position_penalty_weight=0.2, noise_temperature=0.0),
            "Very high LR, low penalty"),

        # No annealing (constant LR)
        ExperimentConfig("const_lr0.1_pos0.2",
            replace(base, learning_rate=0.1, lr_min=0.1, position_penalty_weight=0.2, noise_temperature=0.0),
            "Constant LR=0.1"),

        ExperimentConfig("const_lr0.2_pos0.2",
            replace(base, learning_rate=0.2, lr_min=0.2, position_penalty_weight=0.2, noise_temperature=0.0),
            "Constant LR=0.2"),

        # Slower annealing
        ExperimentConfig("lr_0.1_min0.05_pos0.2",
            replace(base, learning_rate=0.1, lr_min=0.05, position_penalty_weight=0.2, noise_temperature=0.0),
            "Slower annealing"),

        # Even lower penalty
        ExperimentConfig("lr_0.1_pos0.1",
            replace(base, learning_rate=0.1, position_penalty_weight=0.1, noise_temperature=0.0),
            "Very low penalty"),

        # Tiny noise + low penalty + higher LR
        ExperimentConfig("lr_0.2_pos0.2_noise1e6",
            replace(base, learning_rate=0.2, position_penalty_weight=0.2, noise_temperature=1e-6),
            "Combo: higher LR + low penalty + tiny noise"),
    ]

    results = []
    for exp in experiments:
        result = run_experiment(video, exp, "/tmp")
        results.append(result)
        logger.info(f"  {exp.name:25s} -> score={result['score']:7.2f}")

    # Summary
    print("\n" + "="*70)
    print(f"FAST CONVERGENCE RESULTS ({args.steps} steps)")
    print("="*70)
    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results):
        marker = " ⭐" if i == 0 else ""
        vs_target = r['score'] - (-16.0)
        status = "✅" if vs_target > 0 else "❌"
        print(f"{status} {r['name']:30s} | score={r['score']:7.2f} | Δ={vs_target:+6.2f}{marker}")

    print(f"\nBest: {results[0]['name']} = {results[0]['score']:.2f}")
    if results[0]['score'] > -16.0:
        print("✅ BEATS TARGET!")
    else:
        print(f"❌ Still {abs(results[0]['score'] - (-16.0)):.2f} away from target")


if __name__ == "__main__":
    main()
