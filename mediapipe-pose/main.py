"""Orchestrator: record → optimize → compare."""

import argparse

import numpy as np

from record import Recording, record_webcam, ARM_INDICES
from heatmaps import generate_frame_heatmaps
from model.camera import Camera
from convert import estimate_segment_lengths, mediapipe_frame_to_arm
from scoring import OptimizationConfig
from optimize import run_optimization


def fit_cameras(recording: Recording) -> list[Camera]:
    """Fit a camera for each frame from all visible MediaPipe landmarks."""
    cameras = []
    h, w = recording.image_size

    for i, frame in enumerate(recording.frames):
        # Use all landmarks with visibility > 0.5
        visible = frame.visibility > 0.5
        if np.sum(visible) < 6:
            print(f"  Frame {i}: only {np.sum(visible)} visible landmarks, using all")
            visible = np.ones(33, dtype=bool)

        pts_3d = frame.landmarks_3d[visible]
        pts_2d_norm = frame.landmarks_2d[visible]
        pts_2d_px = pts_2d_norm * np.array([w, h], dtype=np.float32)

        try:
            cam = Camera.fit_from_correspondences(pts_2d_px, pts_3d, (h, w))
            cameras.append(cam)
        except RuntimeError as e:
            print(f"  Frame {i}: camera fit failed ({e}), reusing previous")
            if cameras:
                cameras.append(cameras[-1])
            else:
                raise RuntimeError(f"First frame camera fit failed: {e}")

    # Print reprojection error for first frame
    frame = recording.frames[0]
    visible = frame.visibility > 0.5
    pts_3d = frame.landmarks_3d[visible]
    pts_2d_norm = frame.landmarks_2d[visible]
    pts_2d_px = pts_2d_norm * np.array([w, h], dtype=np.float32)
    reproj = cameras[0].world_to_image(pts_3d)
    error = np.mean(np.linalg.norm(reproj - pts_2d_px, axis=1))
    print(f"  Frame 0 reprojection error: {error:.2f} pixels")

    return cameras


def main():
    parser = argparse.ArgumentParser(description="MediaPipe left arm pose optimization")
    parser.add_argument("--fps", type=float, default=10.0, help="Recording FPS")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("-v", action="store_true", help="Show VPython 3D visualization")
    parser.add_argument("--graph", action="store_true", help="Show matplotlib graphs")
    args = parser.parse_args()

    # 1. Record webcam
    print("=== Recording ===")
    recording = record_webcam(target_fps=args.fps)
    if len(recording.frames) < 2:
        print("Need at least 2 frames. Exiting.")
        return

    # 2. Generate heatmaps per frame
    print("\n=== Generating heatmaps ===")
    all_heatmaps = []
    for frame in recording.frames:
        hm = generate_frame_heatmaps(frame, ARM_INDICES, recording.image_size)
        all_heatmaps.append(hm)
    print(f"  Generated heatmaps for {len(all_heatmaps)} frames")

    # 3. Fit camera per frame
    print("\n=== Fitting cameras ===")
    cameras = fit_cameras(recording)
    print(f"  Fitted {len(cameras)} cameras")

    # 4. Estimate median segment lengths
    print("\n=== Estimating segment lengths ===")
    a_b_length, b_c_length = estimate_segment_lengths(recording.frames)
    print(f"  Upper arm (AB): {a_b_length:.4f} m")
    print(f"  Forearm  (BC): {b_c_length:.4f} m")

    # 5. Convert MediaPipe 3D → initial Arm objects
    print("\n=== Converting to Arm parameters ===")
    initial_arms = []
    for frame in recording.frames:
        arm = mediapipe_frame_to_arm(frame, a_b_length, b_c_length)
        initial_arms.append(arm)

    # Verify IK round-trip for first frame
    frame0 = recording.frames[0]
    arm0_coords = initial_arms[0].get_coordinates_numpy()
    mp_shoulder = frame0.landmarks_3d[ARM_INDICES["a"]]
    mp_elbow = frame0.landmarks_3d[ARM_INDICES["b"]]
    mp_wrist = frame0.landmarks_3d[ARM_INDICES["c"]]
    print(f"  IK round-trip errors (frame 0):")
    print(f"    Shoulder: {np.linalg.norm(arm0_coords['a'] - mp_shoulder):.6f} m")
    print(f"    Elbow:    {np.linalg.norm(arm0_coords['b'] - mp_elbow):.6f} m")
    print(f"    Wrist:    {np.linalg.norm(arm0_coords['c'] - mp_wrist):.6f} m")

    # 6. Run optimization
    print("\n=== Optimizing ===")
    config = OptimizationConfig(num_steps=args.steps)
    result = run_optimization(initial_arms, all_heatmaps, cameras, a_b_length, b_c_length, config)

    # 7. Show results
    if args.graph:
        print("\n=== Showing graphs ===")
        from compare import show_comparison_graphs
        show_comparison_graphs(result, cameras, all_heatmaps)

    if args.v:
        print("\n=== Showing 3D visualization ===")
        from compare import show_3d_comparison
        show_3d_comparison(result)

    if not args.graph and not args.v:
        print("\nDone. Use --graph for matplotlib plots or -v for 3D visualization.")


if __name__ == "__main__":
    main()
