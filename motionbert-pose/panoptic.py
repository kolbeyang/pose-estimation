"""CMU Panoptic dataset loading: ground truth, calibration, and video frames."""

import json
import os

import cv2
import numpy as np

from skeleton import coco19_to_h36m, NUM_JOINTS


def load_calibration(sequence_dir: str) -> dict[str, dict[str, np.ndarray]]:
    """Load camera calibration for a sequence.

    Args:
        sequence_dir: Path to the sequence directory.

    Returns:
        Dict mapping camera name (e.g. "00_00") to dict with keys:
            K: (3, 3) intrinsic matrix
            distCoef: (5,) distortion coefficients
            R: (3, 3) rotation matrix (world -> camera)
            t: (3, 1) translation vector
            resolution: (w, h)
    """
    seq_name: str = os.path.basename(sequence_dir)
    calib_path: str = os.path.join(sequence_dir, f"calibration_{seq_name}.json")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration not found: {calib_path}")

    with open(calib_path) as f:
        data: dict = json.load(f)

    cameras: dict[str, dict[str, np.ndarray]] = {}
    for cam in data["cameras"]:
        name: str = cam["name"]
        cameras[name] = {
            "K": np.array(cam["K"], dtype=np.float64).reshape(3, 3),
            "distCoef": np.array(cam["distCoef"], dtype=np.float64).flatten()[:5],
            "R": np.array(cam["R"], dtype=np.float64).reshape(3, 3),
            "t": np.array(cam["t"], dtype=np.float64).reshape(3, 1),
            "resolution": tuple(cam["resolution"]),
        }
    return cameras


def load_ground_truth_frame(
    sequence_dir: str,
    frame_idx: int,
    person_idx: int = 0,
) -> np.ndarray | None:
    """Load one frame of 3D ground truth in COCO19 format.

    Args:
        sequence_dir: Path to the sequence directory.
        frame_idx: Frame index (matching JSON file numbering).
        person_idx: Which person to extract (0 = first).

    Returns:
        (19, 3) array in world coordinates (centimeters), or None if not found.
    """
    gt_dir: str = os.path.join(sequence_dir, "hdPose3d_stage1_coco19")
    json_path: str = os.path.join(gt_dir, f"body3DScene_{frame_idx:08d}.json")

    if not os.path.exists(json_path):
        return None

    with open(json_path) as f:
        data: dict = json.load(f)

    bodies: list[dict] = data.get("bodies", [])
    if person_idx >= len(bodies):
        return None

    joints_flat: list[float] = bodies[person_idx]["joints19"]
    # joints19 is [x0,y0,z0,c0, x1,y1,z1,c1, ...]
    joints: np.ndarray = np.array(joints_flat, dtype=np.float64).reshape(19, 4)
    return joints[:, :3]  # (19, 3) xyz in centimeters


def load_ground_truth_sequence(
    sequence_dir: str,
    frame_indices: list[int],
    person_idx: int = 0,
) -> list[np.ndarray | None]:
    """Load ground truth for a sequence of frames.

    Converts from COCO19 to H36M 17-joint format.

    Returns:
        List of (17, 3) H36M arrays in world coordinates (centimeters),
        or None for missing frames.
    """
    results: list[np.ndarray | None] = []
    for fidx in frame_indices:
        coco19: np.ndarray | None = load_ground_truth_frame(
            sequence_dir, fidx, person_idx
        )
        if coco19 is not None:
            h36m: np.ndarray = coco19_to_h36m(coco19)
            results.append(h36m)
        else:
            results.append(None)
    return results


def world_to_camera(
    points_world: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Transform world coordinates to camera coordinates.

    Args:
        points_world: (..., 3) in centimeters.
        R: (3, 3) rotation matrix (world -> camera).
        t: (3, 1) translation.

    Returns:
        (..., 3) in centimeters (camera frame).
    """
    shape: tuple[int, ...] = points_world.shape
    pts: np.ndarray = points_world.reshape(-1, 3)
    pts_cam: np.ndarray = (R @ pts.T + t).T  # (N, 3)
    return pts_cam.reshape(shape)


def extract_video_frames(
    video_path: str,
    frame_indices: list[int],
) -> list[np.ndarray]:
    """Extract specific frames from an HD video file.

    Args:
        video_path: Path to the MP4 file.
        frame_indices: Sorted list of 0-based frame indices to extract.

    Returns:
        List of RGB frames as (H, W, 3) uint8 arrays.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    idx_set: set[int] = set(frame_indices)
    max_idx: int = max(frame_indices) if frame_indices else 0

    frame_num: int = 0
    while frame_num <= max_idx:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in idx_set:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_num += 1

    cap.release()
    return frames


def get_video_path(
    panoptic_root: str,
    sequence_name: str,
    camera_name: str,
) -> str:
    """Get HD video file path."""
    return os.path.join(
        panoptic_root, sequence_name, "hdVideos", f"hd_{camera_name}.mp4"
    )


def get_sequence_dir(panoptic_root: str, sequence_name: str) -> str:
    """Get sequence directory path."""
    return os.path.join(panoptic_root, sequence_name)
