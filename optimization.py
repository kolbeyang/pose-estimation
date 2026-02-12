import argparse
import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm

from model.arm import Arm
from model.environment import Environment
from utils import (
    EvaluationResult,
    OptimizationConfig,
    OptimizationResult,
    prepare_heatmaps,
    score,
    score_pose_against_heatmap,
)
from video import Video

logger = logging.getLogger(__name__)


def run_optimization(
    video: Video,
    config: OptimizationConfig,
) -> OptimizationResult:
    """
    Run pose optimization on all frames of a video.

    Args:
        video: Video object containing frames and ground truth
        config: Optimization hyperparameters

    Returns:
        OptimizationResult with predicted and ground truth data
    """
    num_frames = video.get_frame_count()
    camera = video.camera

    # Initialize shared length parameters (optimizable, shared across all frames)
    a_b_length = torch.tensor(
        video.a_b_length + torch.randn(1).item() * config.length_init_noise,
        dtype=torch.float32,
    )
    b_c_length = torch.tensor(
        video.b_c_length + torch.randn(1).item() * config.length_init_noise,
        dtype=torch.float32,
    )
    a_b_length.requires_grad_(True)
    b_c_length.requires_grad_(True)

    # Initialize per-frame parameters with jittered ground truth
    all_a_pos = []
    all_a_b_polar = []
    all_b_c_theta = []
    all_gt_coords = []
    all_init_coords = []

    logger.info("Initializing parameters from jittered ground truth...")
    for frame_idx in range(num_frames):
        gt_arm = video.get_frame_arm(frame_idx)
        gt_coords = gt_arm.get_coordinates()
        all_gt_coords.append(gt_coords)

        # Jitter ground truth for initialization
        a_pos = (
            gt_arm.a_pos.detach().clone().float()
            + torch.randn(3) * config.position_init_noise
        )
        a_b_polar = (
            gt_arm.a_b_polar.detach().clone().float()
            + torch.randn(3) * config.angle_init_noise
        )
        b_c_theta = (
            gt_arm.b_c_theta.detach().clone().float()
            + torch.randn(1).item() * config.angle_init_noise
        )

        # Store initial coordinates before enabling gradients
        init_arm = Arm(
            a_pos, a_b_length.detach(), a_b_polar, b_c_length.detach(), b_c_theta
        )
        all_init_coords.append(init_arm.get_coordinates_numpy())

        a_pos.requires_grad_(True)
        a_b_polar.requires_grad_(True)
        b_c_theta.requires_grad_(True)

        all_a_pos.append(a_pos)
        all_a_b_polar.append(a_b_polar)
        all_b_c_theta.append(b_c_theta)

    # Setup optimizer (includes shared lengths)
    all_params = all_a_pos + all_a_b_polar + all_b_c_theta + [a_b_length, b_c_length]
    logger.info("Optimizing %d parameters", len(all_params))
    optimizer = torch.optim.SGD(all_params, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_steps, eta_min=config.lr_min
    )

    # Track parameters over training
    a_b_length_history = []
    b_c_length_history = []
    mid = num_frames // 2
    mid_a_pos_history = []
    mid_a_b_polar_history = []
    mid_b_c_theta_history = []

    # Load heatmaps
    logger.info("Loading heatmaps...")
    all_log_heatmaps = [
        prepare_heatmaps(video.get_frame_heatmaps(i)) for i in range(num_frames)
    ]

    # Optimization loop
    logger.info("Starting batch optimization (%d steps)...", config.num_steps)
    logger.info(
        "Penalty weights: position=%.2f, ab_rotation=%.2f, bc_rotation=%.2f",
        config.position_penalty_weight,
        config.ab_rotation_penalty_weight,
        config.bc_rotation_penalty_weight,
    )

    pbar = tqdm(range(config.num_steps), desc="Optimizing", unit="step")
    for step in pbar:
        optimizer.zero_grad()

        # Record current parameter values
        a_b_length_history.append(a_b_length.item())
        b_c_length_history.append(b_c_length.item())
        mid_a_pos_history.append(all_a_pos[mid].detach().tolist())
        mid_a_b_polar_history.append(all_a_b_polar[mid].detach().tolist())
        mid_b_c_theta_history.append(all_b_c_theta[mid].detach().item())

        arms = [
            Arm(
                all_a_pos[i], a_b_length, all_a_b_polar[i], b_c_length, all_b_c_theta[i]
            )
            for i in range(num_frames)
        ]

        total_score = score(all_log_heatmaps, arms, camera, config)

        loss = -total_score
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        # SGLD: inject Langevin noise scaled by current LR (only after noise_start_step)
        current_lr = scheduler.get_last_lr()[0]
        if step >= config.noise_start_step:
            with torch.no_grad():
                noise_scale = (2.0 * config.noise_temperature * current_lr) ** 0.5
                for param in all_params:
                    param.add_(torch.randn_like(param) * noise_scale)
        # scheduler.step()

        pbar.set_postfix(score=f"{total_score.item():.2f}", lr=f"{current_lr:.1e}")

    logger.info("Optimization complete.")

    # Build final predicted arms
    pred_arms = []
    pred_coords = []
    gt_arms = []

    with torch.no_grad():
        for i in range(num_frames):
            pred_arm = Arm(
                all_a_pos[i], a_b_length, all_a_b_polar[i], b_c_length, all_b_c_theta[i]
            )
            pred_arms.append(pred_arm)
            pred_coords.append(pred_arm.get_coordinates_numpy())
            gt_arms.append(video.get_frame_arm(i))

    return OptimizationResult(
        pred_arms=pred_arms,
        pred_coords=pred_coords,
        init_coords=all_init_coords,
        gt_arms=gt_arms,
        gt_coords=all_gt_coords,
        a_b_length_history=a_b_length_history,
        b_c_length_history=b_c_length_history,
        mid_frame_idx=mid,
        mid_a_pos_history=mid_a_pos_history,
        mid_a_b_polar_history=mid_a_b_polar_history,
        mid_b_c_theta_history=mid_b_c_theta_history,
    )


def evaluate_results(
    opt_result: OptimizationResult,
    video: Video,
    config: OptimizationConfig,
) -> EvaluationResult:
    """
    Evaluate optimization results against ground truth.

    Args:
        opt_result: Results from run_optimization
        video: Video object for heatmaps
        config: Config for score computation

    Returns:
        EvaluationResult with scores
    """
    num_frames = video.get_frame_count()
    camera = video.camera

    all_log_heatmaps = [
        prepare_heatmaps(video.get_frame_heatmaps(i)) for i in range(num_frames)
    ]

    gt_per_frame = []
    pred_per_frame = []
    mpjpe_per_frame = []

    with torch.no_grad():
        for i in range(num_frames):
            gt_score = score_pose_against_heatmap(
                all_log_heatmaps[i], opt_result.gt_arms[i], camera
            ).item()
            pred_score = score_pose_against_heatmap(
                all_log_heatmaps[i], opt_result.pred_arms[i], camera
            ).item()
            gt_per_frame.append(gt_score)
            pred_per_frame.append(pred_score)

            # Compute per-frame MPJPE
            gt_coords = opt_result.gt_arms[i].get_coordinates()
            pred_coords = opt_result.pred_arms[i].get_coordinates()
            frame_mpjpe = (
                sum(
                    torch.norm(pred_coords[j] - gt_coords[j]).item()
                    for j in ["a", "b", "c"]
                )
                / 3
            )
            mpjpe_per_frame.append(frame_mpjpe)

        gt_total = score(all_log_heatmaps, opt_result.gt_arms, camera, config).item()
        pred_total = score(
            all_log_heatmaps, opt_result.pred_arms, camera, config
        ).item()

    mpjpe = sum(mpjpe_per_frame) / len(mpjpe_per_frame)

    return EvaluationResult(
        gt_total_score=gt_total,
        pred_total_score=pred_total,
        gt_per_frame_scores=gt_per_frame,
        pred_per_frame_scores=pred_per_frame,
        mpjpe=mpjpe,
        mpjpe_per_frame=mpjpe_per_frame,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Optimize arm parameters to fit heatmaps using PyTorch"
    )
    parser.add_argument("folder_path", help="Path to saved video folder")
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Visualize results after optimization",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    # Load video
    video = Video.load(args.folder_path)
    logger.info("Loaded video from: %s", args.folder_path)
    logger.info("Frame count: %d", video.get_frame_count())
    logger.info(
        "Segment lengths: a_b=%.2f, b_c=%.2f", video.a_b_length, video.b_c_length
    )

    # Run optimization (possibly multiple runs)
    config = OptimizationConfig()
    num_runs = config.num_runs

    results = []
    eval_results = []
    for run_idx in range(num_runs):
        if num_runs > 1:
            logger.info("=" * 50)
            logger.info("Starting run %d/%d", run_idx + 1, num_runs)
        result = run_optimization(video, config)
        eval_result = evaluate_results(result, video, config)
        results.append(result)
        eval_results.append(eval_result)
        logger.info(
            "Run %d: MPJPE=%.4f, score=%.4f",
            run_idx + 1,
            eval_result.mpjpe,
            eval_result.pred_total_score,
        )

    # Find best run (lowest MPJPE)
    best_idx = min(range(num_runs), key=lambda i: eval_results[i].mpjpe)

    # Log summary
    logger.info("=" * 50)
    logger.info("Summary (%d runs):", num_runs)
    for i in range(num_runs):
        marker = " <-- best" if i == best_idx else ""
        logger.info(
            "  Run %d: MPJPE=%.4f, score=%.4f%s",
            i + 1,
            eval_results[i].mpjpe,
            eval_results[i].pred_total_score,
            marker,
        )
    logger.info("Best run: %d (MPJPE=%.4f)", best_idx + 1, eval_results[best_idx].mpjpe)

    # Save graphs to training_runs/
    import matplotlib
    import matplotlib.pyplot as plt

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("training_runs", f"run-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    num_frames = video.get_frame_count()
    frames = list(range(num_frames))
    steps = list(range(len(results[0].a_b_length_history)))
    mid = results[0].mid_frame_idx
    gt_mid_arm = results[0].gt_arms[mid]

    # Color setup for multi-run graphs
    cmap = matplotlib.colormaps["tab10"]

    def run_style(run_i):
        """Return (color, alpha, linewidth) for a run index."""
        c = cmap(run_i % 10)
        if run_i == best_idx:
            return c, 1.0, 2.0
        return c, 0.5, 1.0

    # 1. Heatmap scores per frame (single-value: tab10 per run)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        frames,
        eval_results[0].gt_per_frame_scores,
        color="green",
        linestyle="--",
        alpha=0.7,
        label="Ground Truth",
        marker="o",
        markersize=4,
    )
    for ri in range(num_runs):
        c, a, lw = run_style(ri)
        label = f"Run {ri + 1}" if num_runs > 1 else "Predicted"
        ax.plot(
            frames,
            eval_results[ri].pred_per_frame_scores,
            color=c,
            alpha=a,
            linewidth=lw,
            label=label,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Heatmap Score")
    ax.set_title("Heatmap Scores per Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(run_dir, "heatmap_scores.png"), dpi=150)
    plt.close(fig)

    # 2. MPJPE per frame (single-value: tab10 per run)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ri in range(num_runs):
        c, a, lw = run_style(ri)
        label = f"Run {ri + 1}" if num_runs > 1 else "MPJPE"
        ax.plot(
            frames,
            eval_results[ri].mpjpe_per_frame,
            color=c,
            alpha=a,
            linewidth=lw,
            label=label,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Frame")
    ax.set_ylabel("MPJPE (distance)")
    ax.set_title("MPJPE per Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(run_dir, "mpjpe.png"), dpi=150)
    plt.close(fig)

    # 3. Segment lengths over training (single-value: tab10 per run)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(
        y=video.a_b_length, color="blue", linestyle="--", alpha=0.5, label="GT a_b"
    )
    ax.axhline(
        y=video.b_c_length, color="orange", linestyle="--", alpha=0.5, label="GT b_c"
    )
    for ri in range(num_runs):
        c, a, lw = run_style(ri)
        ab_label = f"Run {ri + 1} a_b" if num_runs > 1 else "a_b_length"
        bc_label = f"Run {ri + 1} b_c" if num_runs > 1 else "b_c_length"
        ax.plot(
            steps,
            results[ri].a_b_length_history,
            color=c,
            alpha=a,
            linewidth=lw,
            label=ab_label,
            linestyle="-",
        )
        ax.plot(
            steps,
            results[ri].b_c_length_history,
            color=c,
            alpha=a,
            linewidth=lw,
            label=bc_label,
            linestyle=":",
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Length")
    ax.set_title("Segment Lengths over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(run_dir, "segment_lengths.png"), dpi=150)
    plt.close(fig)

    # 4. Middle frame a_pos over training (per-component: rgb, alpha per run)
    gt_a_pos = gt_mid_arm.a_pos.detach()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(
        y=gt_a_pos[0].item(), color="red", linestyle="--", alpha=0.5, label="GT x"
    )
    ax.axhline(
        y=gt_a_pos[1].item(), color="green", linestyle="--", alpha=0.5, label="GT y"
    )
    ax.axhline(
        y=gt_a_pos[2].item(), color="blue", linestyle="--", alpha=0.5, label="GT z"
    )
    for ri in range(num_runs):
        a = 1.0 if ri == best_idx else 0.5
        lw = 2.0 if ri == best_idx else 1.0
        x_data = [p[0] for p in results[ri].mid_a_pos_history]
        y_data = [p[1] for p in results[ri].mid_a_pos_history]
        z_data = [p[2] for p in results[ri].mid_a_pos_history]
        # Only add legend entries from first run
        ax.plot(
            steps,
            x_data,
            color="red",
            alpha=a,
            linewidth=lw,
            label="x" if ri == 0 else None,
        )
        ax.plot(
            steps,
            y_data,
            color="green",
            alpha=a,
            linewidth=lw,
            label="y" if ri == 0 else None,
        )
        ax.plot(
            steps,
            z_data,
            color="blue",
            alpha=a,
            linewidth=lw,
            label="z" if ri == 0 else None,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Position")
    ax.set_title(f"a_pos over Training (frame {mid})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(run_dir, "mid_frame_a_pos.png"), dpi=150)
    plt.close(fig)

    # 5. Middle frame a_b_polar over training (per-component: rgb, alpha per run)
    gt_polar = gt_mid_arm.a_b_polar.detach()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(
        y=gt_polar[0].item(), color="red", linestyle="--", alpha=0.5, label="GT azimuth"
    )
    ax.axhline(
        y=gt_polar[1].item(),
        color="green",
        linestyle="--",
        alpha=0.5,
        label="GT elevation",
    )
    ax.axhline(
        y=gt_polar[2].item(), color="blue", linestyle="--", alpha=0.5, label="GT roll"
    )
    for ri in range(num_runs):
        a = 1.0 if ri == best_idx else 0.5
        lw = 2.0 if ri == best_idx else 1.0
        az_data = [p[0] for p in results[ri].mid_a_b_polar_history]
        el_data = [p[1] for p in results[ri].mid_a_b_polar_history]
        roll_data = [p[2] for p in results[ri].mid_a_b_polar_history]
        ax.plot(
            steps,
            az_data,
            color="red",
            alpha=a,
            linewidth=lw,
            label="azimuth" if ri == 0 else None,
        )
        ax.plot(
            steps,
            el_data,
            color="green",
            alpha=a,
            linewidth=lw,
            label="elevation" if ri == 0 else None,
        )
        ax.plot(
            steps,
            roll_data,
            color="blue",
            alpha=a,
            linewidth=lw,
            label="roll" if ri == 0 else None,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Angle (radians)")
    ax.set_title(f"a_b_polar over Training (frame {mid})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(run_dir, "mid_frame_a_b_polar.png"), dpi=150)
    plt.close(fig)

    # 6. Middle frame b_c_theta over training (single-value: tab10 per run)
    gt_theta = gt_mid_arm.b_c_theta.detach().item()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(
        y=gt_theta, color="blue", linestyle="--", alpha=0.5, label="GT b_c_theta"
    )
    for ri in range(num_runs):
        c, a, lw = run_style(ri)
        label = f"Run {ri + 1}" if num_runs > 1 else "b_c_theta"
        ax.plot(
            steps,
            results[ri].mid_b_c_theta_history,
            color=c,
            alpha=a,
            linewidth=lw,
            label=label,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Angle (radians)")
    ax.set_title(f"b_c_theta over Training (frame {mid})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(run_dir, "mid_frame_b_c_theta.png"), dpi=150)
    plt.close(fig)

    # 7. a_pos z coordinate across frames (GT vs init (best only) vs optimized)
    gt_a_z = [results[0].gt_coords[i]["a"][2].item() for i in range(num_frames)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        frames, gt_a_z, color="green", label="Ground Truth", marker="o", markersize=4
    )
    # Show init only for best run to avoid clutter
    best_init_a_z = [
        results[best_idx].init_coords[i]["a"][2] for i in range(num_frames)
    ]
    ax.plot(
        frames,
        best_init_a_z,
        color="orange",
        label="Initialized (best)",
        marker="s",
        markersize=3,
    )
    for ri in range(num_runs):
        c, a, lw = run_style(ri)
        label = f"Run {ri + 1}" if num_runs > 1 else "Optimized"
        pred_a_z = [results[ri].pred_coords[i]["a"][2] for i in range(num_frames)]
        ax.plot(
            frames,
            pred_a_z,
            color=c,
            alpha=a,
            linewidth=lw,
            label=label,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Frame")
    ax.set_ylabel("a_pos z")
    ax.set_title("a_pos z Coordinate across Frames")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(run_dir, "a_pos_z_across_frames.png"), dpi=150)
    plt.close(fig)

    logger.info("Graphs saved to: %s", run_dir)

    # Visualization with -v flag
    if args.visualize:
        import sys

        from vpython import rate

        from visualize import Visualizer

        env = Environment(cube_size=10.0)

        # Build coords_list: GT arm + N predicted arms
        coords_list = [results[0].gt_coords[0]] + [
            results[i].pred_coords[0] for i in range(num_runs)
        ]
        opacities = [1.0] + [0.1] * num_runs
        labels = ["Ground Truth"] + [f"Run {i + 1}" for i in range(num_runs)]

        visualizer = Visualizer(
            env,
            coords_list,
            camera=video.camera,
            fps=2,
            opacities=opacities,
            labels=labels,
        )
        logger.info("Visualization: green=ground truth, others=predicted runs")
        logger.info("Animating through all frames...")

        try:
            while True:
                for frame_i in range(video.get_frame_count()):
                    rate(visualizer.fps)
                    frame_coords = [results[0].gt_coords[frame_i]] + [
                        results[ri].pred_coords[frame_i] for ri in range(num_runs)
                    ]
                    visualizer.update(frame_coords)
        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
