"""Orchestrator: video → MediaPipe → heatmaps → optimize → evaluate."""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
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
    landmarks_2d: np.ndarray  # (33, 2) normalized [0,1]
    landmarks_3d: np.ndarray  # (33, 3) world coords in meters
    visibility: np.ndarray  # (33,) scores
    timestamp_ms: int


# --- Video processing ---


def process_video(
    video_path: str, target_fps: float = 10.0
) -> tuple[list[FrameData], list[np.ndarray], tuple[int, int], int]:
    """
    Read a video file and run MediaPipe pose detection at target FPS.

    Returns:
        (frames, raw_bgr_frames, image_size, frame_skip)
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
    raw_bgr_frames: list[np.ndarray] = []
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

                    landmarks_2d = np.array(
                        [[l.x, l.y] for l in lm_2d], dtype=np.float32
                    )
                    landmarks_3d = np.array(
                        [[l.x, l.y, l.z] for l in lm_3d], dtype=np.float32
                    )
                    visibility = np.array(
                        [l.visibility for l in lm_2d], dtype=np.float32
                    )

                    frames.append(
                        FrameData(
                            landmarks_2d=landmarks_2d,
                            landmarks_3d=landmarks_3d,
                            visibility=visibility,
                            timestamp_ms=timestamp_ms,
                        )
                    )
                    raw_bgr_frames.append(frame.copy())

            frame_idx += 1

    cap.release()
    print(
        f"  Processed {len(frames)} frames from {video_path} ({image_size[1]}x{image_size[0]})"
    )
    return frames, raw_bgr_frames, image_size, frame_skip


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


# --- CMU Panoptic ground truth loading ---

# CMU COCO19 joint indices
CMU_BODY_CENTER_IDX = 2  # hip midpoint, used as origin
CMU_LSHOULDER_IDX = 3
CMU_LELBOW_IDX = 4
CMU_LWRIST_IDX = 5


def load_cmu_gt(
    gt_dir: str,
    n_frames: int,
    frame_skip: int = 1,
    camera_rotation: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    """
    Load CMU Panoptic 3D ground truth for left arm joints.

    Args:
        gt_dir: Path to hdPose3d_stage1_coco19/ directory
        n_frames: Number of frames to load (matched to video frame count)
        frame_skip: Take every Nth GT file to match video subsampling
        camera_rotation: 3x3 rotation matrix to transform from dome global
                         coords into the viewing camera's frame (from calibration JSON)

    Returns:
        List of dicts with keys "a", "b", "c" as np.ndarray positions in meters
    """
    gt_files = sorted(
        f
        for f in os.listdir(gt_dir)
        if f.startswith("body3DScene_") and f.endswith(".json")
    )
    if not gt_files:
        raise RuntimeError(f"No body3DScene_*.json files found in {gt_dir}")

    # Subsample at the same rate as the video
    gt_files = gt_files[::frame_skip]

    coords = []
    for f in gt_files[:n_frames]:
        with open(os.path.join(gt_dir, f)) as fh:
            data = json.load(fh)

        if not data["bodies"]:
            if coords:
                coords.append(coords[-1])
            else:
                raise RuntimeError(f"No bodies in {f} and no previous frame to copy")
            continue

        joints = data["bodies"][0]["joints19"]
        # Extract hip center as origin, convert cm -> m
        hip = (
            np.array(joints[CMU_BODY_CENTER_IDX * 4 : CMU_BODY_CENTER_IDX * 4 + 3])
            / 100.0
        )
        # Extract left arm joints, convert cm -> m, make hip-relative
        shoulder = (
            np.array(joints[CMU_LSHOULDER_IDX * 4 : CMU_LSHOULDER_IDX * 4 + 3]) / 100.0
            - hip
        )
        elbow = (
            np.array(joints[CMU_LELBOW_IDX * 4 : CMU_LELBOW_IDX * 4 + 3]) / 100.0 - hip
        )
        wrist = (
            np.array(joints[CMU_LWRIST_IDX * 4 : CMU_LWRIST_IDX * 4 + 3]) / 100.0 - hip
        )

        # Rotate from dome global frame into camera frame
        if camera_rotation is not None:
            shoulder = camera_rotation @ shoulder
            elbow = camera_rotation @ elbow
            wrist = camera_rotation @ wrist

        coords.append({"a": shoulder, "b": elbow, "c": wrist})

    print(f"  Loaded {len(coords)} CMU GT frames from {gt_dir}")
    return coords


def load_cmu_camera_rotation(calib_path: str, camera_name: str = "00_00") -> np.ndarray:
    """Load a camera's rotation matrix from CMU Panoptic calibration JSON."""
    with open(calib_path) as f:
        calib = json.load(f)
    for cam in calib["cameras"]:
        if cam["name"] == camera_name:
            return np.array(cam["R"], dtype=np.float64)
    raise RuntimeError(f"Camera {camera_name} not found in {calib_path}")


# --- Evaluation graphs ---

JOINT_NAMES = {"a": "Shoulder", "b": "Elbow", "c": "Wrist"}
COORD_NAMES = {0: "x", 1: "y", 2: "z"}


def generate_evaluation_graphs(
    result: OptimizationResult,
    frames: list[FrameData],
    output_dir: str,
    cmu_gt_coords: list[dict[str, np.ndarray]] | None = None,
):
    """Generate and save evaluation PNG graphs."""
    os.makedirs(output_dir, exist_ok=True)
    n_frames = len(result.mediapipe_coords)
    frame_indices = list(range(n_frames))
    has_gt = cmu_gt_coords is not None and len(cmu_gt_coords) >= n_frames

    # 9 coordinate trajectory graphs
    for joint_key, joint_name in JOINT_NAMES.items():
        for coord_idx, coord_name in COORD_NAMES.items():
            mp_vals = [
                result.mediapipe_coords[i][joint_key][coord_idx]
                for i in range(n_frames)
            ]
            opt_vals = [
                result.optimized_coords[i][joint_key][coord_idx]
                for i in range(n_frames)
            ]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(frame_indices, mp_vals, "g-", label="MediaPipe", linewidth=1.5)
            ax.plot(frame_indices, opt_vals, "r-", label="Optimized", linewidth=1.5)
            if has_gt:
                gt_vals = [
                    cmu_gt_coords[i][joint_key][coord_idx] for i in range(n_frames)
                ]
                ax.plot(frame_indices, gt_vals, "y-", label="CMU GT", linewidth=1.5)
            ax.set_xlabel("Frame")
            ax.set_ylabel(f"{coord_name} (meters)")
            ax.set_title(f"{joint_name} {coord_name.upper()} Trajectory")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = f"{joint_key}_{coord_name}.png"
            fig.savefig(os.path.join(output_dir, filename), dpi=100)
            plt.close(fig)

    # 2 bone length curves (training)
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

    # Per-frame bone length graphs
    bone_pairs = [("a", "b", "Upper Arm", "a_b"), ("b", "c", "Forearm", "b_c")]
    bone_mp_indices = [
        (SHOULDER_IDX, ELBOW_IDX),
        (ELBOW_IDX, WRIST_IDX),
    ]
    for (jk_a, jk_b, bone_label, bone_key), (idx_a, idx_b) in zip(
        bone_pairs, bone_mp_indices
    ):
        mp_lengths = [
            np.linalg.norm(
                frames[i].landmarks_3d[idx_b] - frames[i].landmarks_3d[idx_a]
            )
            for i in range(n_frames)
        ]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frame_indices, mp_lengths, "g-", label="MediaPipe", linewidth=1.5)
        if has_gt:
            gt_lengths = [
                np.linalg.norm(cmu_gt_coords[i][jk_b] - cmu_gt_coords[i][jk_a])
                for i in range(n_frames)
            ]
            ax.plot(frame_indices, gt_lengths, "y-", label="CMU GT", linewidth=1.5)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Length (meters)")
        ax.set_title(f"{bone_label} Length per Frame")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"mp_bone_length_{bone_key}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=100)
        plt.close(fig)

    # MPJPE over training steps (vs MediaPipe)
    if result.mpjpe_history:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            range(len(result.mpjpe_history)), result.mpjpe_history, "b-", linewidth=1.5
        )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MPJPE (meters)")
        ax.set_title("MPJPE vs MediaPipe Over Training")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "mpjpe.png"), dpi=100)
        plt.close(fig)

    # Per-frame MPJPE vs GT
    if has_gt:
        mp_mpjpe = []
        opt_mpjpe = []
        for i in range(n_frames):
            gt = cmu_gt_coords[i]
            mp_err = np.mean(
                [
                    np.linalg.norm(result.mediapipe_coords[i][k] - gt[k])
                    for k in ("a", "b", "c")
                ]
            )
            opt_err = np.mean(
                [
                    np.linalg.norm(result.optimized_coords[i][k] - gt[k])
                    for k in ("a", "b", "c")
                ]
            )
            mp_mpjpe.append(mp_err)
            opt_mpjpe.append(opt_err)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(frame_indices, mp_mpjpe, "g-", label="MediaPipe vs GT", linewidth=1.5)
        ax.plot(frame_indices, opt_mpjpe, "r-", label="Optimized vs GT", linewidth=1.5)
        ax.set_xlabel("Frame")
        ax.set_ylabel("MPJPE (meters)")
        ax.set_title("Per-Frame MPJPE vs CMU Ground Truth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "mpjpe_vs_gt.png"), dpi=100)
        plt.close(fig)

        avg_mp = np.mean(mp_mpjpe)
        avg_opt = np.mean(opt_mpjpe)
        print(f"  MPJPE vs GT — MediaPipe: {avg_mp:.4f} m, Optimized: {avg_opt:.4f} m")

        # Per-frame MPJVE (Mean Per-Joint Velocity Error) vs GT
        # Velocity = displacement between consecutive frames per joint
        mp_mpjve = []
        opt_mpjve = []
        for i in range(1, n_frames):
            gt_vel = {
                k: cmu_gt_coords[i][k] - cmu_gt_coords[i - 1][k]
                for k in ("a", "b", "c")
            }
            mp_vel = {
                k: result.mediapipe_coords[i][k] - result.mediapipe_coords[i - 1][k]
                for k in ("a", "b", "c")
            }
            opt_vel = {
                k: result.optimized_coords[i][k] - result.optimized_coords[i - 1][k]
                for k in ("a", "b", "c")
            }
            mp_err = np.mean(
                [np.linalg.norm(mp_vel[k] - gt_vel[k]) for k in ("a", "b", "c")]
            )
            opt_err = np.mean(
                [np.linalg.norm(opt_vel[k] - gt_vel[k]) for k in ("a", "b", "c")]
            )
            mp_mpjve.append(mp_err)
            opt_mpjve.append(opt_err)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            range(1, n_frames), mp_mpjve, "g-", label="MediaPipe vs GT", linewidth=1.5
        )
        ax.plot(
            range(1, n_frames), opt_mpjve, "r-", label="Optimized vs GT", linewidth=1.5
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("MPJVE (meters/frame)")
        ax.set_title("Per-Frame MPJVE vs CMU Ground Truth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "mpjve_vs_gt.png"), dpi=100)
        plt.close(fig)

        avg_mp_vel = np.mean(mp_mpjve)
        avg_opt_vel = np.mean(opt_mpjve)
        print(
            f"  MPJVE vs GT — MediaPipe: {avg_mp_vel:.4f} m/f, Optimized: {avg_opt_vel:.4f} m/f"
        )

    # Score components over training steps
    if result.score_history:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = range(len(result.score_history["total"]))
        ax.plot(
            steps,
            result.score_history["total"],
            "k-",
            label="Total Score",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            result.score_history["heatmap"],
            "g-",
            label="Heatmap Score",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            result.score_history["position"],
            "r-",
            label="Position Penalty (×400)",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            result.score_history["ab_rotation"],
            "m-",
            label="AB Rotation Penalty (×50)",
            linewidth=1.5,
        )
        ax.plot(
            steps,
            result.score_history["bc_rotation"],
            "c-",
            label="BC Rotation Penalty (×30)",
            linewidth=1.5,
        )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Score")
        ax.set_title("Score Components Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "score.png"), dpi=100)
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

    print(f"  Saved evaluation graphs to {output_dir}")


def _render_config_image(
    config: OptimizationConfig, height: int, width: int
) -> np.ndarray:
    """Render optimization config as a matplotlib image matching the given dimensions."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.axis("off")
    ax.set_title("Optimization Config", fontsize=14, fontweight="bold")

    config_dict = asdict(config)
    text = "\n".join(f"{k}: {v}" for k, v in config_dict.items())
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
    )
    plt.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return cv2.resize(img_bgr, (width, height))


def generate_summary_image(output_dir: str, config: OptimizationConfig):
    """Combine all evaluation PNGs in output_dir into a single summary.png."""
    # Collect all PNG files except summary.png itself
    png_files = sorted(
        f for f in os.listdir(output_dir) if f.endswith(".png") and f != "summary.png"
    )
    if not png_files:
        return

    images = []
    for f in png_files:
        img = cv2.imread(os.path.join(output_dir, f))
        if img is not None:
            images.append(img)

    if not images:
        return

    # Use first image dimensions as target
    target_h = images[0].shape[0]
    target_w = images[0].shape[1]

    # Append config as the last tile
    images.append(_render_config_image(config, target_h, target_w))

    # Arrange in a grid: 3 columns
    n_cols = 3
    n_rows = (len(images) + n_cols - 1) // n_cols

    resized = []
    for img in images:
        resized.append(cv2.resize(img, (target_w, target_h)))

    # Pad to fill grid
    blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    while len(resized) < n_rows * n_cols:
        resized.append(blank.copy())

    rows = []
    for r in range(n_rows):
        row_imgs = resized[r * n_cols : (r + 1) * n_cols]
        rows.append(np.hstack(row_imgs))
    summary = np.vstack(rows)

    cv2.imwrite(os.path.join(output_dir, "summary.png"), summary)
    print(f"  Saved summary.png ({n_rows}x{n_cols} grid)")


def generate_overlay_frames(
    raw_frames: list[np.ndarray],
    frames: list[FrameData],
    all_heatmaps: list[dict[str, np.ndarray]],
    result: OptimizationResult,
    cameras: list[Camera],
    image_size: tuple[int, int],
    heatmap_alpha: float = 0.3,
    cmu_gt_coords: list[dict[str, np.ndarray]] | None = None,
) -> list[np.ndarray]:
    """Generate overlay frames with heatmaps, MediaPipe (green), optimized (red), and optionally GT (yellow) skeletons."""
    h, w = image_size
    overlay_frames = []

    for i in range(len(raw_frames)):
        vis_frame = raw_frames[i].copy()
        camera = cameras[i]

        # Overlay heatmaps
        combined_heatmap = np.zeros((h, w), dtype=np.float32)
        for name in ["a", "b", "c"]:
            combined_heatmap = np.maximum(
                combined_heatmap, all_heatmaps[i][name].astype(np.float32)
            )
        heatmap_color = cv2.applyColorMap(
            combined_heatmap.astype(np.uint8), cv2.COLORMAP_HOT
        )
        mask = combined_heatmap > 5  # only blend where heatmap is visible
        mask_3ch = np.stack([mask] * 3, axis=-1)
        vis_frame = np.where(
            mask_3ch,
            cv2.addWeighted(
                vis_frame, 1.0 - heatmap_alpha, heatmap_color, heatmap_alpha, 0
            ),
            vis_frame,
        )

        # Heatmap 2D points (yellow dots, no lines)
        for name, idx in ARM_INDICES.items():
            px = int(frames[i].landmarks_2d[idx, 0] * w)
            py = int(frames[i].landmarks_2d[idx, 1] * h)
            cv2.circle(vis_frame, (px, py), 6, (0, 255, 255), -1)  # BGR yellow

        # Project MediaPipe 3D coords -> 2D (green)
        mp_coords = result.mediapipe_coords[i]
        mp_2d = {}
        for name in ["a", "b", "c"]:
            pt = camera.world_to_image(mp_coords[name])
            mp_2d[name] = (int(pt[0]), int(pt[1]))

        cv2.line(vis_frame, mp_2d["a"], mp_2d["b"], (0, 200, 0), 2)
        cv2.line(vis_frame, mp_2d["b"], mp_2d["c"], (0, 200, 0), 2)
        for name in ["a", "b", "c"]:
            cv2.circle(vis_frame, mp_2d[name], 5, (0, 255, 0), -1)

        # Project optimized 3D coords -> 2D (red)
        opt_coords = result.optimized_coords[i]
        opt_2d = {}
        for name in ["a", "b", "c"]:
            pt = camera.world_to_image(opt_coords[name])
            opt_2d[name] = (int(pt[0]), int(pt[1]))

        cv2.line(vis_frame, opt_2d["a"], opt_2d["b"], (0, 0, 200), 2)
        cv2.line(vis_frame, opt_2d["b"], opt_2d["c"], (0, 0, 200), 2)
        for name in ["a", "b", "c"]:
            cv2.circle(vis_frame, opt_2d[name], 5, (0, 0, 255), -1)

        overlay_frames.append(vis_frame)

    return overlay_frames


def save_prediction_overlay_video(
    raw_frames: list[np.ndarray],
    frames: list[FrameData],
    all_heatmaps: list[dict[str, np.ndarray]],
    result: OptimizationResult,
    cameras: list[Camera],
    image_size: tuple[int, int],
    output_path: str,
    fps: float = 10.0,
    heatmap_alpha: float = 0.3,
    cmu_gt_coords: list[dict[str, np.ndarray]] | None = None,
):
    """Save video with heatmap overlay, heatmap points (yellow), MediaPipe (green), optimized (red), and GT (yellow) skeletons."""
    h, w = image_size
    overlay_frames = generate_overlay_frames(
        raw_frames,
        frames,
        all_heatmaps,
        result,
        cameras,
        image_size,
        heatmap_alpha,
        cmu_gt_coords,
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in overlay_frames:
        writer.write(frame)
    writer.release()
    print(f"  Saved prediction overlay video to {output_path}")


# --- Main pipeline ---


def main():
    parser = argparse.ArgumentParser(description="MediaPipe left arm pose optimization")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Target FPS for processing"
    )
    parser.add_argument(
        "-v",
        choices=["vpython", "pyvista"],
        default=None,
        help="3D visualization backend: vpython or pyvista",
    )
    parser.add_argument(
        "--cmu-gt",
        default=None,
        help="Path to CMU Panoptic hdPose3d_stage1_coco19/ directory for ground truth overlay",
    )
    parser.add_argument(
        "--cmu-calib",
        default=None,
        help="Path to CMU Panoptic calibration JSON (required with --cmu-gt)",
    )
    parser.add_argument(
        "--cmu-camera",
        default="00_00",
        help="CMU camera name to align GT coordinates (default: 00_00)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 1. Process video
    print("=== Processing Video ===")
    frames, raw_bgr_frames, image_size, frame_skip = process_video(
        args.video, target_fps=args.fps
    )
    if len(frames) < 2:
        print("Need at least 2 frames. Exiting.")
        return

    # 2. Generate heatmaps (in memory only)
    print("\n=== Generating Heatmaps ===")
    all_heatmaps = []
    for frame in frames:
        hm = generate_frame_heatmaps(frame, ARM_INDICES, image_size, sigma=50.0)
        all_heatmaps.append(hm)
    print(f"  Generated heatmaps for {len(all_heatmaps)} frames")

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
    print(
        f"    Shoulder: {np.linalg.norm(arm0_coords['a'] - mp_3d_coords[0]['a']):.6f} m"
    )
    print(
        f"    Elbow:    {np.linalg.norm(arm0_coords['b'] - mp_3d_coords[0]['b']):.6f} m"
    )
    print(
        f"    Wrist:    {np.linalg.norm(arm0_coords['c'] - mp_3d_coords[0]['c']):.6f} m"
    )

    # 7. Run optimization
    print("\n=== Optimizing ===")
    config = OptimizationConfig()
    result = run_optimization(
        initial_arms, all_heatmaps, cameras, a_b_length, b_c_length, config
    )

    # 8. Load CMU ground truth if provided
    cmu_gt_coords = None
    if args.cmu_gt:
        print("\n=== Loading CMU Ground Truth ===")
        cam_rot = None
        if args.cmu_calib:
            cam_rot = load_cmu_camera_rotation(args.cmu_calib, args.cmu_camera)
            print(f"  Using camera rotation from {args.cmu_camera}")
        cmu_gt_coords = load_cmu_gt(args.cmu_gt, len(frames), frame_skip, cam_rot)

    # 9. Save run metadata + evaluation graphs + summary
    run_dir = os.path.join("training_runs", f"mediapipe-run-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    with open(os.path.join(run_dir, "command-line-args.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    print("\n=== Generating Evaluation Graphs ===")
    generate_evaluation_graphs(result, frames, run_dir, cmu_gt_coords)
    generate_summary_image(run_dir, config)

    # 10. Save overlay video
    print("\n=== Saving Overlay Video ===")
    save_prediction_overlay_video(
        raw_bgr_frames,
        frames,
        all_heatmaps,
        result,
        cameras,
        image_size,
        os.path.join(run_dir, "prediction_overlay.mp4"),
        fps=args.fps,
        cmu_gt_coords=cmu_gt_coords,
    )

    # 12. Visualization
    if args.v == "vpython":
        print("\n=== VPython Visualization ===")
        from vpython import rate as vp_rate
        from model.environment import Environment
        from visualize import Visualizer

        env = Environment(cube_size=2.0)

        # Initial frame
        init_coords = [mp_3d_coords[0], result.optimized_coords[0]]
        if cmu_gt_coords:
            init_coords.append(cmu_gt_coords[0])
        vis = Visualizer(env, init_coords, camera=cameras[0], fps=10)

        n_frames = len(frames)
        current_frame = 0
        print(f"  Showing {n_frames} frames. Close browser tab to exit.")

        try:
            while True:
                vp_rate(10)
                frame_coords = [
                    mp_3d_coords[current_frame],
                    result.optimized_coords[current_frame],
                ]
                if cmu_gt_coords and current_frame < len(cmu_gt_coords):
                    frame_coords.append(cmu_gt_coords[current_frame])
                vis.update(frame_coords)
                current_frame = (current_frame + 1) % n_frames
        except KeyboardInterrupt:
            import os as _os

            _os._exit(0)

    elif args.v == "pyvista":
        print("\n=== PyVista Visualization ===")
        from visualize_pyvista import PoseVisualizer

        overlay_frames = generate_overlay_frames(
            raw_bgr_frames,
            frames,
            all_heatmaps,
            result,
            cameras,
            image_size,
            cmu_gt_coords=cmu_gt_coords,
        )
        vis = PoseVisualizer(
            display_frames=overlay_frames,
            cameras=cameras,
            mp_coords=mp_3d_coords,
            opt_coords=result.optimized_coords,
            cmu_gt_coords=cmu_gt_coords,
            image_size=image_size,
        )
        vis.show()

    print(f"\nSaved results to {run_dir}")


if __name__ == "__main__":
    main()
