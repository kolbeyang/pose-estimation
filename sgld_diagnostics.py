"""
Phase 1: SGLD Gradient Escape Diagnostics

Validates the hypothesis that SGLD noise causes optimization to "fall off"
sparse heatmap gradients by tracking gradient norms, heatmap scores, and
pixel displacements over the optimization trajectory.
"""
import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime

import matplotlib.pyplot as plt

from optimization import evaluate_results, run_optimization
from utils import OptimizationConfig
from video import HEATMAP_POINT_STD, Video

logger = logging.getLogger(__name__)


def run_diagnostic_experiment(
    video: Video, name: str, noise_temperature: float, exp_dir: str
):
    """Run a single diagnostic experiment with tracking enabled."""
    logger.info("=" * 80)
    logger.info(f"Running diagnostic: {name} (noise_temperature={noise_temperature})")
    logger.info("=" * 80)

    # Configure optimization with diagnostics enabled
    config = OptimizationConfig(
        num_steps=100,
        learning_rate=0.5,
        lr_min=1e-5,
        noise_temperature=noise_temperature,
        num_runs=1,
        position_penalty_weight=0.2,
        ab_rotation_penalty_weight=0.5,
        bc_rotation_penalty_weight=0.3,
        track_diagnostics=True,
        diagnostic_interval=10,
    )

    # Run optimization
    opt_result = run_optimization(video, config)
    eval_result = evaluate_results(opt_result, video, config)

    # Log results
    logger.info(
        f"Results: MPJPE={eval_result.mpjpe:.4f}, score={eval_result.pred_total_score:.4f}"
    )

    # Save diagnostic data
    diagnostic_data = {
        "name": name,
        "noise_temperature": noise_temperature,
        "config": asdict(config),
        "mpjpe": eval_result.mpjpe,
        "score": eval_result.pred_total_score,
        "gradient_norms": opt_result.gradient_norms,
        "heatmap_scores": opt_result.heatmap_scores,
        "pixel_displacements": opt_result.pixel_displacements,
    }

    result_file = os.path.join(exp_dir, f"{name}.json")
    with open(result_file, "w") as f:
        json.dump(diagnostic_data, f, indent=2)

    return diagnostic_data


def plot_diagnostics(diagnostics_list: list[dict], exp_dir: str):
    """Generate diagnostic plots comparing different noise levels."""
    logger.info("Generating diagnostic plots...")

    # Calculate gradient basin radius (from HEATMAP_POINT_STD)
    # Gaussian gradient effectively zero beyond ~5 standard deviations
    gradient_radius_px = 5 * HEATMAP_POINT_STD

    # 1. Gradient norms over time
    fig, ax = plt.subplots(figsize=(10, 6))
    for diag in diagnostics_list:
        if diag["gradient_norms"]:
            steps = [
                i * diag["config"]["diagnostic_interval"]
                for i in range(len(diag["gradient_norms"]))
            ]
            ax.plot(
                steps,
                diag["gradient_norms"],
                label=f"{diag['name']} (noise={diag['noise_temperature']:.0e})",
                marker="o",
                markersize=4,
            )
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Gradient Norms Over Time (Higher = Stronger Signal)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.savefig(os.path.join(exp_dir, "gradient_norms.png"), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: gradient_norms.png")

    # 2. Heatmap scores over time
    fig, ax = plt.subplots(figsize=(10, 6))
    for diag in diagnostics_list:
        if diag["heatmap_scores"]:
            steps = [
                i * diag["config"]["diagnostic_interval"]
                for i in range(len(diag["heatmap_scores"]))
            ]
            ax.plot(
                steps,
                diag["heatmap_scores"],
                label=f"{diag['name']} (noise={diag['noise_temperature']:.0e})",
                marker="o",
                markersize=4,
            )
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Total Heatmap Score (penalty-free)")
    ax.set_title("Heatmap Fit Over Time (Higher = Better Fit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(exp_dir, "heatmap_scores.png"), dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: heatmap_scores.png")

    # 3. Summary table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    table_data = [
        [
            "Experiment",
            "Noise Temp",
            "MPJPE",
            "Score",
            "Gradient Collapse?",
            "Expected Displacement",
        ]
    ]

    for diag in diagnostics_list:
        # Check for gradient collapse (>50% drop from initial)
        if diag["gradient_norms"] and len(diag["gradient_norms"]) > 1:
            initial_norm = diag["gradient_norms"][0]
            final_norm = diag["gradient_norms"][-1]
            collapse = "YES" if final_norm < 0.5 * initial_norm else "NO"
        else:
            collapse = "N/A"

        # Estimate expected pixel displacement from noise
        # noise_scale = sqrt(2 * noise_temp * lr)
        # For typical parameter (angle ~1 rad), displacement scales with noise_scale
        lr = diag["config"]["learning_rate"]
        noise_temp = diag["noise_temperature"]
        noise_scale = (2 * noise_temp * lr) ** 0.5
        expected_displacement_px = (
            noise_scale * 100
        )  # Rough estimate (100px per unit angle)

        table_data.append(
            [
                diag["name"],
                f"{diag['noise_temperature']:.0e}",
                f"{diag['mpjpe']:.4f}",
                f"{diag['score']:.2f}",
                collapse,
                f"{expected_displacement_px:.1f}px",
            ]
        )

    table = ax.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.15, 0.12, 0.12, 0.12, 0.17, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor("#40466e")
        cell.set_text_props(weight="bold", color="white")

    ax.set_title(
        f"SGLD Diagnostic Summary\nGradient Basin Radius: ~{gradient_radius_px:.0f}px (5σ, σ={HEATMAP_POINT_STD})",
        fontsize=12,
        weight="bold",
        pad=20,
    )

    fig.savefig(os.path.join(exp_dir, "diagnostic_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: diagnostic_summary.png")


def main():
    parser = argparse.ArgumentParser(
        description="Run SGLD diagnostic experiments to validate gradient escape hypothesis"
    )
    parser.add_argument("folder_path", help="Path to saved video folder")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    # Load video
    video = Video.load(args.folder_path)
    logger.info(f"Loaded video from: {args.folder_path}")
    logger.info(f"Frame count: {video.get_frame_count()}")
    logger.info(
        f"Heatmap gradient basin radius: ~{5 * HEATMAP_POINT_STD:.0f}px (σ={HEATMAP_POINT_STD})"
    )

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments", f"sgld-diagnostics-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Saving results to: {exp_dir}")

    # Run diagnostic experiments
    experiments = [
        ("no_noise", 0.0),
        ("tiny_noise", 1e-6),
        ("problematic_noise", 0.01),
    ]

    diagnostics_list = []
    for name, noise_temp in experiments:
        diag = run_diagnostic_experiment(video, name, noise_temp, exp_dir)
        diagnostics_list.append(diag)

    # Generate plots
    plot_diagnostics(diagnostics_list, exp_dir)

    # Print summary
    logger.info("=" * 80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Gradient basin radius: ~{5 * HEATMAP_POINT_STD:.0f}px (beyond this, gradients are effectively zero)"
    )
    logger.info("")

    # Sort by MPJPE
    diagnostics_list.sort(key=lambda x: x["mpjpe"])

    for i, diag in enumerate(diagnostics_list):
        marker = " <-- BEST MPJPE" if i == 0 else ""
        logger.info(
            f"{diag['name']:20s} | noise={diag['noise_temperature']:8.0e} | "
            f"MPJPE={diag['mpjpe']:.4f} | score={diag['score']:7.2f}{marker}"
        )

    logger.info("")
    logger.info("Analysis:")
    logger.info("  - Check gradient_norms.png: Does high noise cause gradient collapse?")
    logger.info(
        "  - Check heatmap_scores.png: Does high noise prevent score improvement?"
    )
    logger.info(
        "  - Check diagnostic_summary.png: Compare expected displacement vs gradient basin"
    )

    # Save summary JSON
    summary_file = os.path.join(exp_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "gradient_basin_radius_px": 5 * HEATMAP_POINT_STD,
                "heatmap_std": HEATMAP_POINT_STD,
                "experiments": diagnostics_list,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
