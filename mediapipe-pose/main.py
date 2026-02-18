"""Orchestrator: video → MediaPipe → heatmaps → optimize → evaluate."""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import cv2
import mediapipe as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.arm import Arm
from model.camera import Camera
from scoring import OptimizationConfig
from optimize import run_optimization, OptimizationResult

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# MediaPipe landmark indices for left arm
SHOULDER_IDX = 11
ELBOW_IDX = 13
WRIST_IDX = 15
ARM_INDICES = {"a": SHOULDER_IDX, "b": ELBOW_IDX, "c": WRIST_IDX}


# --- Data classes ---

@dataclass
class FrameData:
    landmarks_2d: np.ndarray      # (33, 2) normalized [0,1]
    landmarks_3d: np.ndarray      # (33, 3) world coords in meters
    visibility: np.ndarray         # (33,) scores
    timestamp_ms: int


# --- Video processing ---

def process_video(video_path: str, target_fps: float = 10.0) -> tuple[list[FrameData], tuple[int, int]]:
    """
    Read a video file and run MediaPipe pose detection at target FPS.

    Returns:
        (frames, image_size) where image_size is (height, width)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(round(video_fps / target_fps)))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_size = (h, w)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    frames: list[FrameData] = []
    frame_idx = 0
    timestamp_ms = 0
    frame_interval_ms = int(1000 / target_fps)

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                timestamp_ms += frame_interval_ms
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and result.pose_world_landmarks:
                    lm_2d = result.pose_landmarks[0]
                    lm_3d = result.pose_world_landmarks[0]

                    landmarks_2d = np.array([[l.x, l.y] for l in lm_2d], dtype=np.float32)
                    landmarks_3d = np.array([[l.x, l.y, l.z] for l in lm_3d], dtype=np.float32)
                    visibility = np.array([l.visibility for l in lm_2d], dtype=np.float32)

                    frames.append(FrameData(
                        landmarks_2d=landmarks_2d,
                        landmarks_3d=landmarks_3d,
                        visibility=visibility,
                        timestamp_ms=timestamp_ms,
                    ))

            frame_idx += 1

    cap.release()
    print(f"  Processed {len(frames)} frames from {video_path} ({image_size[1]}x{image_size[0]})")
    return frames, image_size


# --- Heatmap generation ---

def generate_heatmap(x: float, y: float, height: int, width: int, sigma: float = 3.0) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (x, y). Returns uint8 [0, 255]."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return (gaussian * 255).astype(np.uint8)


def generate_frame_heatmaps(
    frame: FrameData,
    arm_indices: dict[str, int],
    image_size: tuple[int, int],
    sigma: float = 3.0,
) -> dict[str, np.ndarray]:
    """Generate heatmaps for arm joints from MediaPipe 2D landmarks."""
    h, w = image_size
    heatmaps = {}
    for name, idx in arm_indices.items():
        px = frame.landmarks_2d[idx, 0] * w
        py = frame.landmarks_2d[idx, 1] * h
        heatmaps[name] = generate_heatmap(px, py, h, w, sigma=sigma)
    return heatmaps


def save_heatmaps(all_heatmaps: list[dict[str, np.ndarray]], output_dir: str):
    """Save heatmaps as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, hm in enumerate(all_heatmaps):
        for name, heatmap in hm.items():
            path = os.path.join(output_dir, f"frame_{i:04d}_{name}.png")
            cv2.imwrite(path, heatmap)
    print(f"  Saved heatmaps to {output_dir}")


# --- Camera fitting ---

def fit_cameras(frames: list[FrameData], image_size: tuple[int, int]) -> list[Camera]:
    """Fit a camera for each frame from all visible MediaPipe landmarks."""
    cameras = []
    h, w = image_size

    for i, frame in enumerate(frames):
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
    frame = frames[0]
    visible = frame.visibility > 0.5
    pts_3d = frame.landmarks_3d[visible]
    pts_2d_norm = frame.landmarks_2d[visible]
    pts_2d_px = pts_2d_norm * np.array([w, h], dtype=np.float32)
    reproj = cameras[0].world_to_image(pts_3d)
    error = np.mean(np.linalg.norm(reproj - pts_2d_px, axis=1))
    print(f"  Frame 0 reprojection error: {error:.2f} pixels")

    return cameras


# --- IK conversion ---

def estimate_segment_lengths(frames: list[FrameData]) -> tuple[float, float]:
    """Estimate median segment lengths (shoulder-elbow, elbow-wrist) across all frames."""
    ab_lengths = []
    bc_lengths = []
    for frame in frames:
        shoulder = frame.landmarks_3d[SHOULDER_IDX]
        elbow = frame.landmarks_3d[ELBOW_IDX]
        wrist = frame.landmarks_3d[WRIST_IDX]
        ab_lengths.append(np.linalg.norm(elbow - shoulder))
        bc_lengths.append(np.linalg.norm(wrist - elbow))
    return float(np.median(ab_lengths)), float(np.median(bc_lengths))


def positions_to_arm_params(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    a_b_length: float,
    b_c_length: float,
) -> dict:
    """Convert 3D joint positions to Arm polar parameters with fixed segment lengths."""
    ab_vec = elbow - shoulder
    ab_norm = np.linalg.norm(ab_vec)
    if ab_norm < 1e-8:
        ab_dir = np.array([1.0, 0.0, 0.0])
    else:
        ab_dir = ab_vec / ab_norm

    azimuth = np.arctan2(ab_dir[1], ab_dir[0])
    elevation = np.arcsin(np.clip(ab_dir[2], -1.0, 1.0))

    bc_vec = wrist - elbow
    bc_norm = np.linalg.norm(bc_vec)
    if bc_norm < 1e-8:
        bc_dir = ab_dir
    else:
        bc_dir = bc_vec / bc_norm

    local_x = ab_dir
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)
    local_y_no_roll = np.array([-sin_az, cos_az, 0.0])

    cos_el = np.cos(elevation)
    sin_el = np.sin(elevation)
    local_z_no_roll = np.array([
        -cos_az * sin_el,
        -sin_az * sin_el,
        cos_el,
    ])
    local_z_no_roll = local_z_no_roll / (np.linalg.norm(local_z_no_roll) + 1e-10)

    bc_along_x = np.dot(bc_dir, local_x)
    bc_along_y = np.dot(bc_dir, local_y_no_roll)
    bc_along_z = np.dot(bc_dir, local_z_no_roll)

    theta = np.arctan2(bc_along_x, np.sqrt(bc_along_y**2 + bc_along_z**2 + 1e-10))
    roll = np.arctan2(bc_along_z, bc_along_y)

    return {
        "a_pos": shoulder.copy(),
        "a_b_polar": (float(azimuth), float(elevation), float(roll)),
        "b_c_theta": float(theta),
    }


def mediapipe_frame_to_arm(
    frame: FrameData,
    a_b_length: float,
    b_c_length: float,
) -> Arm:
    """Convert one frame's MediaPipe landmarks to an Arm object."""
    shoulder = frame.landmarks_3d[SHOULDER_IDX]
    elbow = frame.landmarks_3d[ELBOW_IDX]
    wrist = frame.landmarks_3d[WRIST_IDX]

    params = positions_to_arm_params(shoulder, elbow, wrist, a_b_length, b_c_length)

    return Arm(
        a_pos=params["a_pos"],
        a_b_length=a_b_length,
        a_b_polar=params["a_b_polar"],
        b_c_length=b_c_length,
        b_c_theta=params["b_c_theta"],
    )


# --- Evaluation graphs ---

JOINT_NAMES = {"a": "Shoulder", "b": "Elbow", "c": "Wrist"}
COORD_NAMES = {0: "x", 1: "y", 2: "z"}


def generate_evaluation_graphs(result: OptimizationResult, frames: list[FrameData], output_dir: str):
    """Generate and save evaluation PNG graphs."""
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(result.mediapipe_coords)
    frame_indices = list(range(n_frames))

    # 9 coordinate trajectory graphs
    for joint_key, joint_name in JOINT_NAMES.items():
        for coord_idx, coord_name in COORD_NAMES.items():
            mp_vals = [result.mediapipe_coords[i][joint_key][coord_idx] for i in range(n_frames)]
            opt_vals = [result.optimized_coords[i][joint_key][coord_idx] for i in range(n_frames)]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(frame_indices, mp_vals, "g-", label="MediaPipe", linewidth=1.5)
            ax.plot(frame_indices, opt_vals, "r-", label="Optimized", linewidth=1.5)
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{coord_name} (meters)")
            ax.set_title(f"{joint_name} {coord_name.upper()} Trajectory")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = f"{joint_key}_{coord_name}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=100)
            plt.close(fig)

    # 2 bone length curves
    for bone_key, bone_label in [("a_b", "Upper Arm"), ("b_c", "Forearm")]:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = list(range(len(result.bone_length_history[bone_key])))
        ax.plot(steps, result.bone_length_history[bone_key], "b-", linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Length (meters)")
        ax.set_title(f"Bone Length: {bone_label}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"bone_length_{bone_key}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close(fig)

    # 2 MediaPipe per-frame bone length graphs (from raw landmarks)
    bone_segments = [
        (SHOULDER_IDX, ELBOW_IDX, "Upper Arm", "a_b"),
        (ELBOW_IDX, WRIST_IDX, "Forearm", "b_c"),
    ]
    for idx_a, idx_b, bone_label, bone_key in bone_segments:
        mp_lengths = [
            np.linalg.norm(frames[i].landmarks_3d[idx_b] - frames[i].landmarks_3d[idx_a])
            for i in range(n_frames)
        ]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frame_indices, mp_lengths, "g-", label="MediaPipe", linewidth=1.5)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Length (meters)")
        ax.set_title(f"MediaPipe {bone_label} Length per Frame")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"mp_bone_length_{bone_key}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close(fig)

    # Rename coordinate graphs to match plan naming
    rename_map = {}
    for joint_key, joint_name in JOINT_NAMES.items():
        for coord_idx, coord_name in COORD_NAMES.items():
            old = f"{joint_key}_{coord_name}.png"
            new = f"{joint_name.lower()}_{coord_name}.png"
            if old != new:
                rename_map[old] = new

    for old_name, new_name in rename_map.items():
        old_path = os.path.join(output_dir, old_name)
        new_path = os.path.join(output_dir, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)

    print(f"  Saved 13 evaluation graphs to {output_dir}")


# --- Main pipeline ---

def main():
    parser = argparse.ArgumentParser(description="MediaPipe left arm pose optimization")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS for processing")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("-v", action="store_true", help="Show VPython 3D visualization")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 1. Process video
    print("=== Processing Video ===")
    frames, image_size = process_video(args.video, target_fps=args.fps)
    if len(frames) < 2:
        print("Need at least 2 frames. Exiting.")
        return

    # 2. Generate + save heatmaps
    print("\n=== Generating Heatmaps ===")
    all_heatmaps = []
    for frame in frames:
        hm = generate_frame_heatmaps(frame, ARM_INDICES, image_size)
        all_heatmaps.append(hm)
    print(f"  Generated heatmaps for {len(all_heatmaps)} frames")

    heatmap_dir = os.path.join("images", f"mediapipe-heatmaps-{timestamp}")
    save_heatmaps(all_heatmaps, heatmap_dir)

    # 3. Fit per-frame cameras
    print("\n=== Fitting Cameras ===")
    cameras = fit_cameras(frames, image_size)
    print(f"  Fitted {len(cameras)} cameras")

    # 4. Estimate median segment lengths
    print("\n=== Estimating Segment Lengths ===")
    a_b_length, b_c_length = estimate_segment_lengths(frames)
    print(f"  Upper arm (AB): {a_b_length:.4f} m")
    print(f"  Forearm  (BC): {b_c_length:.4f} m")

    # 5. Extract raw MediaPipe 3D coords for 3 joints
    mp_3d_coords = []
    for frame in frames:
        coords = {
            "a": frame.landmarks_3d[SHOULDER_IDX].copy(),
            "b": frame.landmarks_3d[ELBOW_IDX].copy(),
            "c": frame.landmarks_3d[WRIST_IDX].copy(),
        }
        mp_3d_coords.append(coords)

    # 6. Convert to initial Arm objects via IK
    print("\n=== Converting to Arm Parameters ===")
    initial_arms = []
    for frame in frames:
        arm = mediapipe_frame_to_arm(frame, a_b_length, b_c_length)
        initial_arms.append(arm)

    # Verify IK round-trip
    arm0_coords = initial_arms[0].get_coordinates_numpy()
    print(f"  IK round-trip errors (frame 0):")
    print(f"    Shoulder: {np.linalg.norm(arm0_coords['a'] - mp_3d_coords[0]['a']):.6f} m")
    print(f"    Elbow:    {np.linalg.norm(arm0_coords['b'] - mp_3d_coords[0]['b']):.6f} m")
    print(f"    Wrist:    {np.linalg.norm(arm0_coords['c'] - mp_3d_coords[0]['c']):.6f} m")

    # 7. Run optimization
    print("\n=== Optimizing ===")
    config = OptimizationConfig(num_steps=args.steps)
    result = run_optimization(initial_arms, all_heatmaps, cameras, a_b_length, b_c_length, config)

    # 8. Save evaluation graphs
    print("\n=== Generating Evaluation Graphs ===")
    run_dir = os.path.join("training_runs", f"mediapipe-run-{timestamp}")
    generate_evaluation_graphs(result, frames, run_dir)

    # 9. VPython visualization
    if args.v:
        print("\n=== VPython Visualization ===")
        from vpython import rate as vp_rate
        from model.environment import Environment
        from visualize import Visualizer

        env = Environment(cube_size=2.0)

        # Initial frame
        init_coords = [mp_3d_coords[0], result.optimized_coords[0]]
        vis = Visualizer(env, init_coords, camera=cameras[0], fps=10)

        n_frames = len(frames)
        current_frame = 0
        print(f"  Showing {n_frames} frames. Close browser tab to exit.")

        try:
            while True:
                vp_rate(10)
                frame_coords = [mp_3d_coords[current_frame], result.optimized_coords[current_frame]]
                vis.update(frame_coords)
                current_frame = (current_frame + 1) % n_frames
        except KeyboardInterrupt:
            import os as _os
            _os._exit(0)
    else:
        print(f"\nDone. Results saved to {run_dir}")
        print("Use -v for VPython 3D visualization.")


if __name__ == "__main__":
    main()
