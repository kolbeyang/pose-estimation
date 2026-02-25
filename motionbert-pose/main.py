"""Orchestrator: video → Stacked Hourglass → MotionBERT → heatmaps → optimize → evaluate."""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.arm import Arm
from model.camera import Camera
from scoring import OptimizationConfig
from optimize import run_optimization, OptimizationResult
from detect import (
    detect_persons,
    run_hourglass,
    run_motionbert,
    MPII_LEFT_ARM,
    H36M_LEFT_ARM,
)


# MediaPipe-equivalent indices for left arm (using H36M from MotionBERT)
SHOULDER_IDX = H36M_LEFT_ARM["a"]  # 11
ELBOW_IDX = H36M_LEFT_ARM["b"]    # 12
WRIST_IDX = H36M_LEFT_ARM["c"]    # 13
ARM_JOINT_NAMES = {"a": "Shoulder", "b": "Elbow", "c": "Wrist"}

# MacBook Pro 14" webcam: ~65° horizontal FOV at 1280x720 → fx ≈ 1005
FOCAL_LENGTH = 1005.0

# Approximate head-to-ankle height for root depth estimation (meters)
ASSUMED_HEIGHT = 1.50


# --- Data classes ---

@dataclass
class FrameData:
    """Per-frame pose data from Stacked Hourglass + MotionBERT."""
    keypoints_2d: np.ndarray       # (16, 3) MPII keypoints (x, y, conf) in pixel coords
    heatmaps: np.ndarray           # (16, H, W) full-resolution heatmaps
    landmarks_3d: np.ndarray       # (17, 3) H36M 3D positions from MotionBERT
    landmarks_2d_norm: np.ndarray  # (17, 2) normalized [0,1] 2D positions (for camera fitting)


# --- Heatmap extraction ---

def extract_arm_heatmaps(
    frame: FrameData,
    image_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Extract left arm heatmaps from full Stacked Hourglass output.

    Returns dict with keys "a", "b", "c" mapping to uint8 heatmaps (H, W).
    """
    h, w = image_size
    heatmaps = {}
    for name, mpii_idx in MPII_LEFT_ARM.items():
        hm = frame.heatmaps[mpii_idx]
        # Normalize to [0, 255] uint8
        hm_min, hm_max = hm.min(), hm.max()
        if hm_max > hm_min:
            hm_norm = ((hm - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
        else:
            hm_norm = np.zeros((h, w), dtype=np.uint8)
        heatmaps[name] = hm_norm
    return heatmaps


def save_heatmaps(all_heatmaps: list[dict[str, np.ndarray]], output_dir: str):
    """Save heatmaps as PNG files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, hm in enumerate(all_heatmaps):
        for name, heatmap in hm.items():
            path = os.path.join(output_dir, f"frame_{i:04d}_{name}.png")
            cv2.imwrite(path, heatmap)
    print(f"  Saved heatmaps to {output_dir}")


# --- Root depth estimation ---

# H36M joint indices for height estimation
H36M_HEAD = 9
H36M_LANKLE = 6
H36M_RANKLE = 3


def pixel_to_camera_space(
    positions_3d_pixel: np.ndarray,
    bboxes: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert pixel-aligned 3D positions to camera space (meters).

    MotionBERT outputs pixel-aligned positions where:
      X, Y = pixel coordinates matching the 2D input
      Z = pixel-proportional depth (same scale as X,Y pixel offsets)

    This function:
      1. Estimates absolute root depth per frame from YOLO bbox height
      2. Uses perspective back-projection to convert each joint to camera space

    The conversion guarantees that camera.world_to_image(cam_pos) ≈ pixel (X, Y),
    so 2D reprojection error should be near zero.

    Returns:
        (positions_cam, root_depths) where positions_cam is (N, 17, 3) in meters
    """
    h, w = image_size
    cx, cy = w / 2.0, h / 2.0
    n_frames = positions_3d_pixel.shape[0]

    # Estimate per-frame root depth from YOLO bbox
    root_depths = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        bbox_height = bboxes[i][3] - bboxes[i][1]
        if bbox_height > 20:
            root_depths[i] = FOCAL_LENGTH * ASSUMED_HEIGHT / bbox_height
        else:
            root_depths[i] = 3.0
    root_depths = np.clip(root_depths, 1.0, 6.0)

    # Convert pixel-aligned positions to camera space
    positions_cam = np.zeros_like(positions_3d_pixel)
    for i in range(n_frames):
        z_root = root_depths[i]
        root_z_px = positions_3d_pixel[i, 0, 2]  # Root pixel-Z

        for j in range(17):
            u = positions_3d_pixel[i, j, 0]   # pixel X
            v = positions_3d_pixel[i, j, 1]   # pixel Y
            z_px = positions_3d_pixel[i, j, 2]  # pixel-proportional Z

            # Joint depth = root depth + Z offset converted from pixels to meters
            dz_px = z_px - root_z_px
            Z_cam = z_root + dz_px * z_root / FOCAL_LENGTH

            # Perspective back-projection
            X_cam = (u - cx) * Z_cam / FOCAL_LENGTH
            Y_cam = (v - cy) * Z_cam / FOCAL_LENGTH

            positions_cam[i, j] = [X_cam, Y_cam, Z_cam]

    return positions_cam, root_depths


# --- IK conversion ---

def estimate_segment_lengths(frames: list[FrameData]) -> tuple[float, float]:
    """Estimate median segment lengths from MotionBERT 3D positions."""
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


def frame_to_arm(frame: FrameData) -> Arm:
    """Convert one frame's MotionBERT 3D landmarks to an Arm object.

    Uses actual per-frame segment lengths (not median) so the IK
    round-trip is exact. The optimizer uses shared learnable bone
    lengths, so these per-frame lengths only affect initialization.
    """
    shoulder = frame.landmarks_3d[SHOULDER_IDX]
    elbow = frame.landmarks_3d[ELBOW_IDX]
    wrist = frame.landmarks_3d[WRIST_IDX]

    actual_ab = float(np.linalg.norm(elbow - shoulder))
    actual_bc = float(np.linalg.norm(wrist - elbow))

    params = positions_to_arm_params(shoulder, elbow, wrist, actual_ab, actual_bc)

    return Arm(
        a_pos=params["a_pos"],
        a_b_length=actual_ab,
        a_b_polar=params["a_b_polar"],
        b_c_length=actual_bc,
        b_c_theta=params["b_c_theta"],
    )


# --- Evaluation graphs ---

COORD_NAMES = {0: "x", 1: "y", 2: "z"}


def generate_evaluation_graphs(result: OptimizationResult, frames: list[FrameData], output_dir: str):
    """Generate and save evaluation PNG graphs."""
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(result.mediapipe_coords)  # "mediapipe" here means MotionBERT initial
    frame_indices = list(range(n_frames))

    # 9 coordinate trajectory graphs
    for joint_key, joint_name in ARM_JOINT_NAMES.items():
        for coord_idx, coord_name in COORD_NAMES.items():
            initial_vals = [result.mediapipe_coords[i][joint_key][coord_idx] for i in range(n_frames)]
            opt_vals = [result.optimized_coords[i][joint_key][coord_idx] for i in range(n_frames)]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(frame_indices, initial_vals, "b-", label="MotionBERT", linewidth=1.5)
            ax.plot(frame_indices, opt_vals, "r-", label="Optimized", linewidth=1.5)
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{coord_name} (meters)")
            ax.set_title(f"{joint_name} {coord_name.upper()} Trajectory")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = f"{joint_name.lower()}_{coord_name}.png"
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

    # Loss curve
    if result.loss_history:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result.loss_history, "b-", linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Optimization Loss Curve")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=100)
        plt.close(fig)

    # 2 MotionBERT per-frame bone length graphs
    bone_segments = [
        (SHOULDER_IDX, ELBOW_IDX, "Upper Arm", "a_b"),
        (ELBOW_IDX, WRIST_IDX, "Forearm", "b_c"),
    ]
    for idx_a, idx_b, bone_label, bone_key in bone_segments:
        lengths = [
            np.linalg.norm(frames[i].landmarks_3d[idx_b] - frames[i].landmarks_3d[idx_a])
            for i in range(n_frames)
        ]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frame_indices, lengths, "b-", label="MotionBERT", linewidth=1.5)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Length (meters)")
        ax.set_title(f"MotionBERT {bone_label} Length per Frame")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"mb_bone_length_{bone_key}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close(fig)

    # Summary grid: all 13 graphs in one image
    fig_summary, axes = plt.subplots(4, 4, figsize=(28, 20))
    axes = axes.flatten()

    # Row 0-2: 9 coordinate trajectory graphs (joint x coord)
    plot_idx = 0
    for joint_key, joint_name in ARM_JOINT_NAMES.items():
        for coord_idx, coord_name in COORD_NAMES.items():
            ax = axes[plot_idx]
            initial_vals = [result.mediapipe_coords[i][joint_key][coord_idx] for i in range(n_frames)]
            opt_vals = [result.optimized_coords[i][joint_key][coord_idx] for i in range(n_frames)]
            ax.plot(frame_indices, initial_vals, "b-", label="MotionBERT", linewidth=1)
            ax.plot(frame_indices, opt_vals, "r-", label="Optimized", linewidth=1)
            ax.set_title(f"{joint_name} {coord_name.upper()}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if plot_idx == 0:
                ax.legend(fontsize=7)
            plot_idx += 1

    # Row 3 cols 0-1: bone length training curves
    for bone_key, bone_label in [("a_b", "Upper Arm"), ("b_c", "Forearm")]:
        ax = axes[plot_idx]
        steps = list(range(len(result.bone_length_history[bone_key])))
        ax.plot(steps, result.bone_length_history[bone_key], "b-", linewidth=1)
        ax.set_title(f"Bone Length: {bone_label}", fontsize=10)
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Row 3 cols 2-3: MotionBERT per-frame bone lengths
    for idx_a, idx_b, bone_label, bone_key in bone_segments:
        ax = axes[plot_idx]
        lengths = [
            np.linalg.norm(frames[i].landmarks_3d[idx_b] - frames[i].landmarks_3d[idx_a])
            for i in range(n_frames)
        ]
        ax.plot(frame_indices, lengths, "b-", linewidth=1)
        ax.set_title(f"MB {bone_label} Length/Frame", fontsize=10)
        ax.set_xlabel("Frame", fontsize=8)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused subplot slots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig_summary.suptitle("Optimization Summary", fontsize=16, fontweight="bold")
    fig_summary.tight_layout(rect=[0, 0, 1, 0.96])
    fig_summary.savefig(os.path.join(output_dir, "summary.png"), dpi=150)
    plt.close(fig_summary)

    print(f"  Saved 13 evaluation graphs + summary.png to {output_dir}")


# --- 2D Overlay ---

JOINT_ORDER = ["a", "b", "c"]  # shoulder -> elbow -> wrist

def save_overlay_frames(
    frames_rgb: list[np.ndarray],
    bboxes: list[np.ndarray],
    cameras: list[Camera],
    initial_coords: list[dict],
    optimized_coords: list[dict],
    keypoints_2d: list[np.ndarray],
    output_dir: str,
):
    """Reproject 3D arm joints to 2D and overlay on video frames.

    White box = YOLO crop, Cyan = SH 2D detections, Green = MotionBERT 3D reprojected,
    Red = Optimized 3D reprojected.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(frames_rgb)
    joint_radius = 5
    line_thickness = 2

    for i in range(n_frames):
        frame_bgr = cv2.cvtColor(frames_rgb[i].copy(), cv2.COLOR_RGB2BGR)
        cam = cameras[i]

        # Draw YOLO bounding box in white
        bbox = bboxes[i].astype(int)
        cv2.rectangle(frame_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

        # Draw SH 2D detections in cyan (ground truth alignment check)
        kp = keypoints_2d[i]  # (16, 3) MPII format
        sh_idx = [MPII_LEFT_ARM["a"], MPII_LEFT_ARM["b"], MPII_LEFT_ARM["c"]]
        for j in range(len(sh_idx) - 1):
            p1 = tuple(kp[sh_idx[j], :2].astype(int))
            p2 = tuple(kp[sh_idx[j + 1], :2].astype(int))
            cv2.line(frame_bgr, p1, p2, (255, 255, 0), line_thickness, cv2.LINE_AA)
        for j in sh_idx:
            center = tuple(kp[j, :2].astype(int))
            cv2.circle(frame_bgr, center, joint_radius, (255, 255, 0), -1, cv2.LINE_AA)

        # Draw 3D reprojections
        for coords, color in [(initial_coords[i], (0, 200, 0)), (optimized_coords[i], (0, 0, 255))]:
            pts_3d = np.array([coords[k] for k in JOINT_ORDER])  # (3, 3)
            pts_2d = cam.world_to_image(pts_3d)  # (3, 2)

            for j in range(len(JOINT_ORDER) - 1):
                p1 = tuple(pts_2d[j].astype(int))
                p2 = tuple(pts_2d[j + 1].astype(int))
                cv2.line(frame_bgr, p1, p2, color, line_thickness, cv2.LINE_AA)

            for j in range(len(JOINT_ORDER)):
                center = tuple(pts_2d[j].astype(int))
                cv2.circle(frame_bgr, center, joint_radius, color, -1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), frame_bgr)

    print(f"  Saved {n_frames} overlay frames to {output_dir}")


# --- Main pipeline ---

def main():
    parser = argparse.ArgumentParser(description="MotionBERT left arm pose optimization")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS for processing")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("-v", action="store_true", help="Show VPython 3D visualization")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 1. Detect persons in video
    print("=== Person Detection (YOLOv8) ===")
    frames_rgb, bboxes, image_size = detect_persons(args.video, target_fps=args.fps)
    if len(frames_rgb) < 2:
        print("Need at least 2 frames. Exiting.")
        return

    # 1b. Save cropped person detections
    crops_dir = os.path.join("training_runs", f"motionbert-run-{timestamp}", "crops")
    os.makedirs(crops_dir, exist_ok=True)
    for i, (frame_rgb, bbox) in enumerate(zip(frames_rgb, bboxes)):
        x1, y1, x2, y2 = bbox.astype(int)
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame_rgb[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(crops_dir, f"frame_{i:04d}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    print(f"  Saved {len(frames_rgb)} cropped frames to {crops_dir}")

    # 2. Run Stacked Hourglass for 2D pose + heatmaps
    print("\n=== 2D Pose Estimation (Stacked Hourglass) ===")
    all_heatmaps_fullres, all_keypoints_2d = run_hourglass(frames_rgb, bboxes, image_size)

    # 3. Run MotionBERT for 3D lifting
    print("\n=== 3D Pose Lifting (MotionBERT) ===")
    positions_3d = run_motionbert(all_keypoints_2d, image_size)

    # 4. Build FrameData for each frame
    print("\n=== Building Frame Data ===")
    h, w = image_size
    frames: list[FrameData] = []
    for i in range(len(frames_rgb)):
        # Compute normalized 2D positions from keypoints
        # Map MPII keypoints to H36M for 2D positions
        from detect import mpii_to_h36m
        kp_h36m_2d = mpii_to_h36m(all_keypoints_2d[i])  # (17, 3)
        landmarks_2d_norm = kp_h36m_2d[:, :2].copy()
        landmarks_2d_norm[:, 0] /= w
        landmarks_2d_norm[:, 1] /= h

        frames.append(FrameData(
            keypoints_2d=all_keypoints_2d[i],
            heatmaps=all_heatmaps_fullres[i],
            landmarks_3d=positions_3d[i],
            landmarks_2d_norm=landmarks_2d_norm,
        ))

    # 5. Extract and save arm heatmaps
    print("\n=== Extracting Arm Heatmaps ===")
    all_arm_heatmaps = []
    for frame in frames:
        hm = extract_arm_heatmaps(frame, image_size)
        all_arm_heatmaps.append(hm)
    print(f"  Extracted heatmaps for {len(all_arm_heatmaps)} frames")

    heatmap_dir = os.path.join("images", f"motionbert-heatmaps-{timestamp}")
    save_heatmaps(all_arm_heatmaps, heatmap_dir)

    # 6. Convert pixel-aligned 3D to camera space (meters).
    # MotionBERT output is pixel-aligned (X,Y in pixels, Z pixel-proportional).
    # This converts to camera coordinates using perspective back-projection.
    print("\n=== Pixel → Camera Space Conversion ===")
    positions_3d_abs, root_depths = pixel_to_camera_space(positions_3d, bboxes, image_size)
    print(f"  Root depth range: {root_depths.min():.2f} - {root_depths.max():.2f} m")
    print(f"  Root depth mean: {root_depths.mean():.2f} m")

    # Update frames with absolute 3D
    for i, frame in enumerate(frames):
        frame.landmarks_3d = positions_3d_abs[i]

    # Perspective camera with known focal length (same for all frames)
    camera = Camera.from_focal_length(FOCAL_LENGTH, image_size)
    cameras = [camera] * len(frames)

    # Check reprojection quality (SH 2D keypoints vs 3D→2D reprojection)
    from detect import mpii_to_h36m as _mpii_to_h36m
    for i in range(min(3, len(frames))):
        kp_h36m = _mpii_to_h36m(all_keypoints_2d[i])
        arm_2d = kp_h36m[[SHOULDER_IDX, ELBOW_IDX, WRIST_IDX], :2]
        arm_3d = frames[i].landmarks_3d[[SHOULDER_IDX, ELBOW_IDX, WRIST_IDX]]
        error = camera.reprojection_error(arm_2d, arm_3d)
        print(f"  Frame {i} arm reproj error: {error:.1f}px")

    # 7. Estimate median segment lengths
    print("\n=== Estimating Segment Lengths ===")
    a_b_length, b_c_length = estimate_segment_lengths(frames)
    print(f"  Upper arm (AB): {a_b_length:.4f} m")
    print(f"  Forearm  (BC): {b_c_length:.4f} m")

    # 8. Extract 3D coords for left arm
    mb_3d_coords = []
    for frame in frames:
        coords = {
            "a": frame.landmarks_3d[SHOULDER_IDX].copy(),
            "b": frame.landmarks_3d[ELBOW_IDX].copy(),
            "c": frame.landmarks_3d[WRIST_IDX].copy(),
        }
        mb_3d_coords.append(coords)

    # 9. Convert to initial Arm objects via IK (per-frame actual lengths)
    print("\n=== Converting to Arm Parameters ===")
    initial_arms = []
    for frame in frames:
        arm = frame_to_arm(frame)
        initial_arms.append(arm)

    # Verify IK round-trip
    arm0_coords = initial_arms[0].get_coordinates_numpy()
    print(f"  IK round-trip errors (frame 0):")
    print(f"    Shoulder: {np.linalg.norm(arm0_coords['a'] - mb_3d_coords[0]['a']):.6f} m")
    print(f"    Elbow:    {np.linalg.norm(arm0_coords['b'] - mb_3d_coords[0]['b']):.6f} m")
    print(f"    Wrist:    {np.linalg.norm(arm0_coords['c'] - mb_3d_coords[0]['c']):.6f} m")

    # 10. Run optimization
    print("\n=== Optimizing ===")
    config = OptimizationConfig(num_steps=args.steps)
    result = run_optimization(initial_arms, all_arm_heatmaps, cameras, a_b_length, b_c_length, config)

    # 11. Save evaluation graphs
    print("\n=== Generating Evaluation Graphs ===")
    run_dir = os.path.join("training_runs", f"motionbert-run-{timestamp}")
    generate_evaluation_graphs(result, frames, run_dir)

    # 12. Save 2D overlay frames (reproject 3D -> 2D onto video)
    print("\n=== Saving 2D Overlay Frames ===")
    overlay_dir = os.path.join(run_dir, "overlays")
    save_overlay_frames(frames_rgb, bboxes, cameras, mb_3d_coords, result.optimized_coords, all_keypoints_2d, overlay_dir)

    # 13. VPython visualization
    if args.v:
        print("\n=== VPython Visualization ===")
        from vpython import rate as vp_rate
        from model.environment import Environment
        from visualize import Visualizer

        env = Environment(cube_size=2.0)

        init_coords = [mb_3d_coords[0], result.optimized_coords[0]]
        vis = Visualizer(env, init_coords, camera=cameras[0], fps=10)

        n_frames = len(frames)
        current_frame = 0
        print(f"  Showing {n_frames} frames. Close browser tab to exit.")

        try:
            while True:
                vp_rate(10)
                frame_coords = [mb_3d_coords[current_frame], result.optimized_coords[current_frame]]
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
