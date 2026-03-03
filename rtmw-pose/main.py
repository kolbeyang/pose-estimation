"""Orchestrator: video → RTMW3D → heatmaps → optimize → evaluate."""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rtmlib.tools.pose_estimation.rtmpose3d import RTMPose3d

from model.arm import Arm
from model.camera import Camera
from scoring import OptimizationConfig
from optimize import run_optimization, OptimizationResult

# COCO-WholeBody indices for left arm
SHOULDER_IDX = 5  # left_shoulder
ELBOW_IDX = 7  # left_elbow
WRIST_IDX = 9  # left_wrist
ARM_INDICES = {"a": SHOULDER_IDX, "b": ELBOW_IDX, "c": WRIST_IDX}

# Body keypoint indices (0-16) used for camera fitting
BODY_INDICES = list(range(17))

# RTMW3D model dimensions
# model_input_size in rtmlib is (width, height) = (288, 384)
MODEL_WIDTH = 288
MODEL_HEIGHT = 384
# Depth dimension matches width in the training codec: input_size=(288, 384, 288)
MODEL_DEPTH = 288
Z_RANGE = 2.1744869
SIMCC_SPLIT_RATIO = 2.0

# Anthropometric prior: average adult bi-acromial shoulder width (meters)
SHOULDER_WIDTH_M = 0.36
L_SHOULDER_IDX = 5
R_SHOULDER_IDX = 6


# --- Data classes ---


@dataclass
class FrameData:
    landmarks_2d: np.ndarray  # (133, 2) normalized [0,1]
    landmarks_3d: np.ndarray  # (133, 3) normalized 3D coords
    visibility: np.ndarray  # (133,) confidence scores
    timestamp_ms: int


# --- Video processing ---


def process_video(
    video_path: str, target_fps: float = 10.0
) -> tuple[list[FrameData], tuple[int, int], list[dict]]:
    """
    Read a video file and run RTMW3D pose detection at target FPS.

    Returns:
        (frames, image_size, debug_raw) where image_size is (height, width)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(round(video_fps / target_fps)))

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_size = (h, w)

    # Initialize RTMW3D pose model directly (no YOLOX detection/cropping)
    POSE_MODEL_URL = "https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx"
    print("  Loading RTMW3D model (no detection, full-frame input)...")
    model = RTMPose3d(
        POSE_MODEL_URL,
        model_input_size=(288, 384),
        backend="onnxruntime",
        device="cpu",
    )

    # Camera intrinsics (standard assumption for typical cameras)
    fx = fy = float(max(h, w))
    cx, cy = w / 2.0, h / 2.0
    prev_z0 = 2.0  # fallback reference depth (meters)

    frames: list[FrameData] = []
    _debug_raw: list[dict] = []
    frame_idx = 0
    timestamp_ms = 0
    frame_interval_ms = int(1000 / target_fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            timestamp_ms += frame_interval_ms

            # RTMW3D inference
            keypoints, scores, keypoints_simcc, keypoints_2d = model(frame)

            if len(keypoints) > 0:
                # Take first detected person
                kp_simcc = keypoints_simcc[0]  # (133, 3): raw SimCC coords / split_ratio
                kp_scores = scores[0]  # (133,)
                kp_2d = keypoints_2d[0]  # (133, 2): pixel coords in original image

                # Normalize 2D to [0, 1] (same convention as MediaPipe)
                landmarks_2d = kp_2d / np.array([w, h], dtype=np.float32)

                # --- 3D reconstruction: unproject 2D + SimCC depth ---

                # 1. Decode root-relative z from SimCC (meters)
                #    rtmlib uses height (384) but correct depth dim is 288
                z_rel = (kp_simcc[:, 2] / (MODEL_DEPTH / 2) - 1) * Z_RANGE

                # 2. Estimate reference depth from shoulder width
                l_sh_px = kp_2d[L_SHOULDER_IDX]
                r_sh_px = kp_2d[R_SHOULDER_IDX]
                sh_px_dist = np.linalg.norm(l_sh_px - r_sh_px)
                if sh_px_dist > 10:
                    z0 = fx * SHOULDER_WIDTH_M / sh_px_dist
                else:
                    z0 = prev_z0  # fallback to previous frame

                prev_z0 = z0

                # 3. Absolute depth per joint
                z_abs = z0 + z_rel  # (133,)

                # 4. Unproject (u, v, Z) → (X, Y, Z) in camera frame
                landmarks_3d = np.zeros((133, 3), dtype=np.float32)
                landmarks_3d[:, 0] = (kp_2d[:, 0] - cx) * z_abs / fx
                landmarks_3d[:, 1] = (kp_2d[:, 1] - cy) * z_abs / fy
                landmarks_3d[:, 2] = z_abs

                # --- DEBUG: Store raw data for first few frames ---
                if len(frames) < 5:
                    _debug_raw.append({
                        "frame_idx": len(frames),
                        "kp_simcc_body": kp_simcc[:17].copy(),
                        "kp_2d_body": kp_2d[:17].copy(),
                        "kp_scores_body": kp_scores[:17].copy(),
                        "landmarks_3d_body": landmarks_3d[:17].copy(),
                        "z_rel_body": z_rel[:17].copy(),
                        "z0": z0,
                        "sh_px_dist": sh_px_dist,
                    })

                frames.append(
                    FrameData(
                        landmarks_2d=landmarks_2d.astype(np.float32),
                        landmarks_3d=landmarks_3d.astype(np.float32),
                        visibility=kp_scores.astype(np.float32),
                        timestamp_ms=timestamp_ms,
                    )
                )

        frame_idx += 1

    cap.release()
    print(
        f"  Processed {len(frames)} frames from {video_path} ({image_size[1]}x{image_size[0]})"
    )
    return frames, image_size, _debug_raw


# --- Depth diagnostics ---

COCO_BODY_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]


def diagnose_depth(frames: list[FrameData], _debug_raw: list[dict]):
    """Print depth diagnostics for the unprojection-based 3D reconstruction."""
    print("\n" + "=" * 70)
    print("DEPTH DIAGNOSTICS (unprojection approach)")
    print("=" * 70)

    # --- 1. Depth estimation from shoulder width ---
    print("\n--- 1. Reference depth (Z0) from shoulder width ---")
    for d in _debug_raw[:5]:
        i = d["frame_idx"]
        print(f"  Frame {i}: Z0={d['z0']:.3f}m  shoulder_px_dist={d['sh_px_dist']:.1f}px")

    # --- 2. SimCC z_rel values (root-relative depth) ---
    print("\n--- 2. SimCC root-relative depth (z_rel) for body joints, frame 0 ---")
    if _debug_raw:
        d = _debug_raw[0]
        z_rel = d["z_rel_body"]
        simcc = d["kp_simcc_body"]
        print(f"  {'Joint':<14s} {'simcc_z':>8s} {'z_rel(m)':>10s} {'score':>6s}")
        for j in range(17):
            print(f"  {COCO_BODY_NAMES[j]:<14s} {simcc[j, 2]:8.1f} {z_rel[j]:+10.4f} {d['kp_scores_body'][j]:6.3f}")

    # --- 3. Unprojected 3D coordinates ---
    print("\n--- 3. Unprojected 3D body landmarks (camera frame, meters) ---")
    for d in _debug_raw[:3]:
        i = d["frame_idx"]
        l3d = d["landmarks_3d_body"]
        print(f"\n  Frame {i} (Z0={d['z0']:.3f}m):")
        print(f"    x: min={l3d[:, 0].min():.4f}  max={l3d[:, 0].max():.4f}  range={l3d[:, 0].max() - l3d[:, 0].min():.4f}")
        print(f"    y: min={l3d[:, 1].min():.4f}  max={l3d[:, 1].max():.4f}  range={l3d[:, 1].max() - l3d[:, 1].min():.4f}")
        print(f"    z: min={l3d[:, 2].min():.4f}  max={l3d[:, 2].max():.4f}  range={l3d[:, 2].max() - l3d[:, 2].min():.4f}")

    # --- 4. Left arm bone lengths ---
    print("\n--- 4. Left arm bone lengths ---")
    print("  (Expected: upper arm ~0.25-0.35m, forearm ~0.22-0.28m)")
    ab_lens = []
    bc_lens = []
    for i, frame in enumerate(frames[:10]):
        shoulder = frame.landmarks_3d[SHOULDER_IDX]
        elbow = frame.landmarks_3d[ELBOW_IDX]
        wrist = frame.landmarks_3d[WRIST_IDX]
        ab_vec = elbow - shoulder
        bc_vec = wrist - elbow
        ab_len = np.linalg.norm(ab_vec)
        bc_len = np.linalg.norm(bc_vec)
        ab_z_frac = abs(ab_vec[2] - 0) / ab_len if ab_len > 0 else 0
        bc_z_frac = abs(bc_vec[2] - 0) / bc_len if bc_len > 0 else 0
        ab_lens.append(ab_len)
        bc_lens.append(bc_len)
        if i < 5:
            print(f"\n  Frame {i}:")
            print(f"    Shoulder: [{shoulder[0]:+.4f}, {shoulder[1]:+.4f}, {shoulder[2]:+.4f}]")
            print(f"    Elbow:    [{elbow[0]:+.4f}, {elbow[1]:+.4f}, {elbow[2]:+.4f}]")
            print(f"    Wrist:    [{wrist[0]:+.4f}, {wrist[1]:+.4f}, {wrist[2]:+.4f}]")
            print(f"    Upper arm: {ab_len:.4f}m  (delta_z={ab_vec[2]:+.4f}m, z_contrib={ab_z_frac:.1%})")
            print(f"    Forearm:   {bc_len:.4f}m  (delta_z={bc_vec[2]:+.4f}m, z_contrib={bc_z_frac:.1%})")

    if ab_lens:
        print(f"\n  Summary (first {len(ab_lens)} frames):")
        print(f"    Upper arm: median={np.median(ab_lens):.4f}m  std={np.std(ab_lens):.4f}m")
        print(f"    Forearm:   median={np.median(bc_lens):.4f}m  std={np.std(bc_lens):.4f}m")

    # --- 5. IK round-trip check ---
    print("\n--- 5. IK round-trip for frame 0 ---")
    if frames:
        frame = frames[0]
        shoulder = frame.landmarks_3d[SHOULDER_IDX]
        elbow = frame.landmarks_3d[ELBOW_IDX]
        wrist = frame.landmarks_3d[WRIST_IDX]
        ab_len = np.linalg.norm(elbow - shoulder)
        bc_len = np.linalg.norm(wrist - elbow)
        params = positions_to_arm_params(shoulder, elbow, wrist, ab_len, bc_len)
        arm = Arm(a_pos=params["a_pos"], a_b_length=ab_len,
                  a_b_polar=params["a_b_polar"], b_c_length=bc_len,
                  b_c_theta=params["b_c_theta"])
        coords = arm.get_coordinates_numpy()
        print(f"  Shoulder error: {np.linalg.norm(coords['a'] - shoulder):.6f}m")
        print(f"  Elbow error:    {np.linalg.norm(coords['b'] - elbow):.6f}m")
        print(f"  Wrist error:    {np.linalg.norm(coords['c'] - wrist):.6f}m")

    print("\n" + "=" * 70)


# --- Heatmap generation ---


def generate_heatmap(
    x: float, y: float, height: int, width: int, sigma: float = 50.0
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (x, y). Returns uint8 [0, 255]."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return (gaussian * 255).astype(np.uint8)


def generate_frame_heatmaps(
    frame: FrameData,
    arm_indices: dict[str, int],
    image_size: tuple[int, int],
    sigma: float = 3.0,
) -> dict[str, np.ndarray]:
    """Generate heatmaps for arm joints from RTMW3D 2D landmarks."""
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
    """Fit a camera for each frame from visible body keypoints."""
    cameras = []
    h, w = image_size

    for i, frame in enumerate(frames):
        # Use body keypoints (0-16) with sufficient confidence
        body_scores = frame.visibility[BODY_INDICES]
        visible_mask = body_scores > 0.5
        visible_indices = np.array(BODY_INDICES)[visible_mask]

        if len(visible_indices) < 6:
            print(f"  Frame {i}: only {len(visible_indices)} visible body landmarks, using all")
            visible_indices = np.array(BODY_INDICES)

        pts_3d = frame.landmarks_3d[visible_indices]
        pts_2d_norm = frame.landmarks_2d[visible_indices]
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
    body_scores = frame.visibility[BODY_INDICES]
    visible_mask = body_scores > 0.5
    visible_indices = np.array(BODY_INDICES)[visible_mask]
    pts_3d = frame.landmarks_3d[visible_indices]
    pts_2d_norm = frame.landmarks_2d[visible_indices]
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
    local_z_no_roll = np.array(
        [
            -cos_az * sin_el,
            -sin_az * sin_el,
            cos_el,
        ]
    )
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


def frame_to_arm(
    frame: FrameData,
    a_b_length: float,
    b_c_length: float,
) -> Arm:
    """Convert one frame's RTMW3D landmarks to an Arm object."""
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


def generate_evaluation_graphs(
    result: OptimizationResult, frames: list[FrameData], output_dir: str
):
    """Generate and save evaluation PNG graphs."""
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(result.mediapipe_coords)
    frame_indices = list(range(n_frames))

    # 9 coordinate trajectory graphs
    for joint_key, joint_name in JOINT_NAMES.items():
        for coord_idx, coord_name in COORD_NAMES.items():
            init_vals = [
                result.mediapipe_coords[i][joint_key][coord_idx]
                for i in range(n_frames)
            ]
            opt_vals = [
                result.optimized_coords[i][joint_key][coord_idx]
                for i in range(n_frames)
            ]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(frame_indices, init_vals, "g-", label="RTMW3D", linewidth=1.5)
            ax.plot(frame_indices, opt_vals, "r-", label="Optimized", linewidth=1.5)
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{coord_name} (normalized)")
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
        ax.set_ylabel("Length (normalized)")
        ax.set_title(f"Bone Length: {bone_label}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"bone_length_{bone_key}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close(fig)

    # 2 RTMW3D per-frame bone length graphs
    bone_segments = [
        (SHOULDER_IDX, ELBOW_IDX, "Upper Arm", "a_b"),
        (ELBOW_IDX, WRIST_IDX, "Forearm", "b_c"),
    ]
    for idx_a, idx_b, bone_label, bone_key in bone_segments:
        lengths = [
            np.linalg.norm(
                frames[i].landmarks_3d[idx_b] - frames[i].landmarks_3d[idx_a]
            )
            for i in range(n_frames)
        ]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frame_indices, lengths, "g-", label="RTMW3D", linewidth=1.5)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Length (normalized)")
        ax.set_title(f"RTMW3D {bone_label} Length per Frame")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"rtmw_bone_length_{bone_key}.png"
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


# --- Reprojection overlay ---


def render_reprojection_video(
    video_path: str,
    target_fps: float,
    cameras: list[Camera],
    initial_3d_coords: list[dict[str, np.ndarray]],
    result: OptimizationResult,
    output_path: str,
):
    """Render video with reprojected arm skeletons overlaid on original frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(round(video_fps / target_fps)))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (w, h))

    joint_order = ["a", "b", "c"]
    pose_frame_idx = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0 and pose_frame_idx < len(cameras):
            cam = cameras[pose_frame_idx]

            # Project initial RTMW3D coords (green)
            init_pts = []
            for name in joint_order:
                pt_3d = initial_3d_coords[pose_frame_idx][name]
                pt_2d = cam.world_to_image(pt_3d)
                init_pts.append((int(pt_2d[0]), int(pt_2d[1])))

            # Project optimized coords (red)
            opt_pts = []
            for name in joint_order:
                pt_3d = result.optimized_coords[pose_frame_idx][name]
                pt_2d = cam.world_to_image(pt_3d)
                opt_pts.append((int(pt_2d[0]), int(pt_2d[1])))

            # Draw RTMW3D skeleton (green)
            for i in range(len(init_pts) - 1):
                cv2.line(frame, init_pts[i], init_pts[i + 1], (0, 255, 0), 2)
            for pt in init_pts:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)

            # Draw optimized skeleton (red, BGR)
            for i in range(len(opt_pts) - 1):
                cv2.line(frame, opt_pts[i], opt_pts[i + 1], (0, 0, 255), 2)
            for pt in opt_pts:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

            # Labels
            cv2.putText(
                frame, "RTMW3D", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.putText(
                frame, "Optimized", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

            out.write(frame)
            pose_frame_idx += 1

        frame_idx += 1

    cap.release()
    out.release()
    print(f"  Saved reprojection video to {output_path}")


# --- Main pipeline ---


def main():
    parser = argparse.ArgumentParser(description="RTMW3D left arm pose optimization")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Target FPS for processing"
    )
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("-v", action="store_true", help="Show VPython 3D visualization")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 1. Process video
    print("=== Processing Video ===")
    frames, image_size, _debug_raw = process_video(args.video, target_fps=args.fps)
    if len(frames) < 2:
        print("Need at least 2 frames. Exiting.")
        return

    # Depth diagnostics (before any downstream processing)
    diagnose_depth(frames, _debug_raw)

    # 2. Generate + save heatmaps
    print("\n=== Generating Heatmaps ===")
    all_heatmaps = []
    for frame in frames:
        hm = generate_frame_heatmaps(frame, ARM_INDICES, image_size)
        all_heatmaps.append(hm)
    print(f"  Generated heatmaps for {len(all_heatmaps)} frames")

    heatmap_dir = os.path.join("images", f"rtmw-heatmaps-{timestamp}")
    save_heatmaps(all_heatmaps, heatmap_dir)

    # 3. Fit per-frame cameras
    print("\n=== Fitting Cameras ===")
    cameras = fit_cameras(frames, image_size)
    print(f"  Fitted {len(cameras)} cameras")

    # 4. Estimate median segment lengths
    print("\n=== Estimating Segment Lengths ===")
    a_b_length, b_c_length = estimate_segment_lengths(frames)
    print(f"  Upper arm (AB): {a_b_length:.4f}")
    print(f"  Forearm  (BC): {b_c_length:.4f}")

    # 5. Extract raw RTMW3D 3D coords for 3 joints
    initial_3d_coords = []
    for frame in frames:
        coords = {
            "a": frame.landmarks_3d[SHOULDER_IDX].copy(),
            "b": frame.landmarks_3d[ELBOW_IDX].copy(),
            "c": frame.landmarks_3d[WRIST_IDX].copy(),
        }
        initial_3d_coords.append(coords)

    # 6. Convert to initial Arm objects via IK
    print("\n=== Converting to Arm Parameters ===")
    initial_arms = []
    for frame in frames:
        arm = frame_to_arm(frame, a_b_length, b_c_length)
        initial_arms.append(arm)

    # Verify IK round-trip
    arm0_coords = initial_arms[0].get_coordinates_numpy()
    print(f"  IK round-trip errors (frame 0):")
    print(
        f"    Shoulder: {np.linalg.norm(arm0_coords['a'] - initial_3d_coords[0]['a']):.6f}"
    )
    print(
        f"    Elbow:    {np.linalg.norm(arm0_coords['b'] - initial_3d_coords[0]['b']):.6f}"
    )
    print(
        f"    Wrist:    {np.linalg.norm(arm0_coords['c'] - initial_3d_coords[0]['c']):.6f}"
    )

    # 7. Run optimization
    print("\n=== Optimizing ===")
    config = OptimizationConfig(num_steps=args.steps)
    result = run_optimization(
        initial_arms, all_heatmaps, cameras, a_b_length, b_c_length, config
    )

    # 8. Save evaluation graphs
    print("\n=== Generating Evaluation Graphs ===")
    run_dir = os.path.join("training_runs", f"rtmw-run-{timestamp}")
    generate_evaluation_graphs(result, frames, run_dir)

    # 9. Reprojection overlay video
    print("\n=== Rendering Reprojection Video ===")
    reproj_path = os.path.join(run_dir, "reprojection.mp4")
    render_reprojection_video(
        args.video, args.fps, cameras, initial_3d_coords, result, reproj_path
    )

    # 10. VPython visualization
    if args.v:
        print("\n=== VPython Visualization ===")
        from vpython import rate as vp_rate
        from model.environment import Environment
        from visualize import Visualizer

        env = Environment(cube_size=2.0)

        # Initial frame
        init_coords = [initial_3d_coords[0], result.optimized_coords[0]]
        vis = Visualizer(env, init_coords, camera=cameras[0], fps=10)

        n_frames = len(frames)
        current_frame = 0
        print(f"  Showing {n_frames} frames. Close browser tab to exit.")

        try:
            while True:
                vp_rate(10)
                frame_coords = [
                    initial_3d_coords[current_frame],
                    result.optimized_coords[current_frame],
                ]
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
