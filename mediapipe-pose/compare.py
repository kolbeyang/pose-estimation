"""Visualization and comparison of MediaPipe vs optimized arm poses."""

import numpy as np
import matplotlib.pyplot as plt

from model.camera import Camera
from optimize import OptimizationResult
from scoring import prepare_heatmaps, score_pose_against_heatmap


def show_comparison_graphs(
    result: OptimizationResult,
    cameras: list[Camera],
    heatmaps: list[dict[str, np.ndarray]],
):
    """
    Show matplotlib comparison graphs.

    Plots:
    1. Per-frame heatmap scores (MediaPipe vs optimized)
    2. Per-frame joint position deltas
    """
    n_frames = len(result.mediapipe_arms)

    # Compute per-frame heatmap scores
    mp_scores = []
    opt_scores = []
    all_log_heatmaps = [prepare_heatmaps(h) for h in heatmaps]

    for i in range(n_frames):
        mp_s = score_pose_against_heatmap(
            all_log_heatmaps[i], result.mediapipe_arms[i], cameras[i]
        )
        mp_scores.append(mp_s.item())

        opt_s = score_pose_against_heatmap(
            all_log_heatmaps[i], result.optimized_arms[i], cameras[i]
        )
        opt_scores.append(opt_s.item())

    # Compute per-frame joint position deltas
    deltas = {"a": [], "b": [], "c": []}
    for i in range(n_frames):
        for name in ["a", "b", "c"]:
            diff = np.linalg.norm(
                result.optimized_coords[i][name] - result.mediapipe_coords[i][name]
            )
            deltas[name].append(diff)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Heatmap scores
    ax = axes[0]
    frames = range(n_frames)
    ax.plot(frames, mp_scores, "g-o", label="MediaPipe", markersize=3)
    ax.plot(frames, opt_scores, "r-o", label="Optimized", markersize=3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Heatmap Score (log)")
    ax.set_title("Per-Frame Heatmap Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Joint position deltas
    ax = axes[1]
    ax.plot(frames, deltas["a"], "b-o", label="Shoulder (A)", markersize=3)
    ax.plot(frames, deltas["b"], "g-o", label="Elbow (B)", markersize=3)
    ax.plot(frames, deltas["c"], "r-o", label="Wrist (C)", markersize=3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position Delta (meters)")
    ax.set_title("Joint Position Change: MediaPipe vs Optimized")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def show_3d_comparison(result: OptimizationResult):
    """
    VPython 3D visualization: animate MediaPipe arm (green) vs optimized arm (red).

    Steps through frames with arrow keys.
    """
    import vpython as vp

    n_frames = len(result.mediapipe_coords)

    scene = vp.canvas(title="MediaPipe (green) vs Optimized (red)", width=800, height=600)
    scene.center = vp.vector(0, 0, 0)

    # Create arm visual objects
    mp_spheres = {name: vp.sphere(radius=0.02, color=vp.color.green) for name in ["a", "b", "c"]}
    mp_rods = {
        "ab": vp.cylinder(radius=0.008, color=vp.color.green),
        "bc": vp.cylinder(radius=0.008, color=vp.color.green),
    }

    opt_spheres = {name: vp.sphere(radius=0.02, color=vp.color.red) for name in ["a", "b", "c"]}
    opt_rods = {
        "ab": vp.cylinder(radius=0.008, color=vp.color.red),
        "bc": vp.cylinder(radius=0.008, color=vp.color.red),
    }

    frame_label = vp.label(pos=vp.vector(0, 0.3, 0), text="Frame 0")

    def update_frame(idx):
        mp = result.mediapipe_coords[idx]
        opt = result.optimized_coords[idx]

        for name in ["a", "b", "c"]:
            mp_pos = mp[name]
            opt_pos = opt[name]
            mp_spheres[name].pos = vp.vector(float(mp_pos[0]), float(mp_pos[1]), float(mp_pos[2]))
            opt_spheres[name].pos = vp.vector(float(opt_pos[0]), float(opt_pos[1]), float(opt_pos[2]))

        # Update rods
        for prefix, coords, rods in [("mp", mp, mp_rods), ("opt", opt, opt_rods)]:
            a, b, c = coords["a"], coords["b"], coords["c"]
            rods["ab"].pos = vp.vector(float(a[0]), float(a[1]), float(a[2]))
            rods["ab"].axis = vp.vector(float(b[0] - a[0]), float(b[1] - a[1]), float(b[2] - a[2]))
            rods["bc"].pos = vp.vector(float(b[0]), float(b[1]), float(b[2]))
            rods["bc"].axis = vp.vector(float(c[0] - b[0]), float(c[1] - b[1]), float(c[2] - b[2]))

        frame_label.text = f"Frame {idx}/{n_frames - 1}"

    update_frame(0)

    # Animate through frames
    current_frame = 0
    print("Press 'n' for next frame, 'p' for previous, 'a' to auto-play, 'q' to quit")

    auto_play = False
    while True:
        vp.rate(30)

        if auto_play:
            current_frame = (current_frame + 1) % n_frames
            update_frame(current_frame)
            vp.rate(10)  # ~10 fps playback

        keys = vp.keysdown()
        if "n" in keys:
            current_frame = min(current_frame + 1, n_frames - 1)
            update_frame(current_frame)
            vp.rate(5)
        elif "p" in keys:
            current_frame = max(current_frame - 1, 0)
            update_frame(current_frame)
            vp.rate(5)
        elif "a" in keys:
            auto_play = not auto_play
            vp.rate(5)
        elif "q" in keys:
            break
