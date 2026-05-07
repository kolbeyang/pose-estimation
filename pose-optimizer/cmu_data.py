"""CMU Panoptic dataset loading: ground truth, calibration, video frames, and download helpers."""

from __future__ import annotations

import json
import os
import subprocess

import cv2
import numpy as np

from skeleton import coco19_to_skeleton, NUM_JOINTS


# ---------------------------------------------------------------------------
# Default data root (relative to this file -> data/panoptic-toolbox)
# ---------------------------------------------------------------------------
_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT: str = os.path.normpath(
    os.path.join(_THIS_DIR, "data", "panoptic-toolbox")
)


# ---------------------------------------------------------------------------
# 25 single-subject examples for full evaluation runs
# (sequence, camera, start_frame, num_frames, person_idx)
# ---------------------------------------------------------------------------
EXAMPLES: list[tuple[str, str, int, int, int]] = [
    # pose1_sample
    ("171204_pose1_sample", "00_00", 0, 100, 0),
    # pose2 -- multiple windows
    ("171204_pose2", "00_00", 200, 150, 0),
    ("171204_pose2", "00_00", 5000, 150, 0),
    ("171204_pose2", "00_00", 10000, 150, 0),
    ("171204_pose2", "00_00", 15000, 150, 0),
    ("171204_pose2", "00_00", 25000, 150, 0),
    # pose3
    ("171204_pose3", "00_00", 200, 150, 0),
    ("171204_pose3", "00_00", 4000, 150, 0),
    ("171204_pose3", "00_00", 8000, 150, 0),
    # pose1 -- longer sequence, multiple windows
    ("171204_pose1", "00_00", 5000, 150, 0),
    ("171204_pose1", "00_00", 14000, 150, 0),
    ("171204_pose1", "00_00", 22000, 150, 0),
    ("171204_pose1", "00_00", 30000, 150, 0),
    ("171204_pose1", "00_00", 38000, 150, 0),
    # pose1 -- additional windows
    ("171204_pose1", "00_00", 10000, 150, 0),
    ("171204_pose1", "00_00", 18000, 150, 0),
    # pose2 -- additional windows
    ("171204_pose2", "00_00", 20000, 150, 0),
    ("171204_pose2", "00_00", 30000, 150, 0),
    # pose3 -- additional windows
    ("171204_pose3", "00_00", 2000, 150, 0),
    ("171204_pose3", "00_00", 6000, 150, 0),
    # female_example
    ("female_example_01", "00_00", 350, 150, 0),
    ("female_example_01", "00_00", 1000, 150, 0),
    ("female_example_01", "00_00", 2000, 150, 0),
    ("female_example_01", "00_00", 3000, 150, 0),
    ("female_example_01", "00_00", 4000, 150, 0),
]


def get_sequence_dir(data_root: str, sequence_name: str) -> str:
    """Get full sequence directory path.

    Args:
        data_root: Base data directory.
        sequence_name: Sequence name (e.g. "171204_pose1_sample").

    Returns:
        Full path to the sequence directory.
    """
    return os.path.join(data_root, sequence_name)


def get_video_path(
    data_root: str,
    sequence_name: str,
    camera_name: str,
) -> str:
    """Get HD video file path.

    Args:
        data_root: Base data directory.
        sequence_name: Sequence name.
        camera_name: Camera name (e.g. "00_00").

    Returns:
        Full path to the HD video MP4 file.
    """
    return os.path.join(
        data_root, sequence_name, "hdVideos", f"hd_{camera_name}.mp4"
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

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
        (19, 3) array in world coordinates (centimeters), or None. [3D:COCO19]
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
    joints: np.ndarray = np.array(joints_flat, dtype=np.float64).reshape(19, 4)
    return joints[:, :3]  # (19, 3) xyz in centimeters


def count_persons_in_frame(sequence_dir: str, frame_idx: int) -> int:
    """Count the number of detected persons in a GT frame.

    Args:
        sequence_dir: Path to the sequence directory.
        frame_idx: Frame index.

    Returns:
        Number of persons, or 0 if frame not found.
    """
    gt_dir = os.path.join(sequence_dir, "hdPose3d_stage1_coco19")
    json_path = os.path.join(gt_dir, f"body3DScene_{frame_idx:08d}.json")
    if not os.path.exists(json_path):
        return 0
    with open(json_path) as f:
        data = json.load(f)
    return len(data.get("bodies", []))


def load_ground_truth_sequence(
    sequence_dir: str,
    frame_indices: list[int],
    person_idx: int = 0,
) -> list[np.ndarray | None]:
    """Load ground truth for a sequence of frames.

    Converts from [3D:COCO19] to [3D:SKELETON_16] via coco19_to_skeleton().

    Args:
        sequence_dir: Path to the sequence directory.
        frame_indices: List of frame indices to load.
        person_idx: Which person to extract (0 = first).

    Returns:
        List of (16, 3) skeleton arrays in world coordinates (centimeters),
        or None for missing frames. [3D:SKELETON_16]
    """
    results: list[np.ndarray | None] = []
    for fidx in frame_indices:
        coco19 = load_ground_truth_frame(sequence_dir, fidx, person_idx)  # [3D:COCO19] or None
        if coco19 is not None:
            skel = coco19_to_skeleton(coco19)  # [3D:SKELETON_16]
            results.append(skel)
        else:
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    idx_set = set(frame_indices)
    max_idx = max(frame_indices) if frame_indices else 0

    frame_num = 0
    while frame_num <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in idx_set:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_num += 1

    cap.release()
    return frames


def get_video_fps(video_path: str) -> float:
    """Get the FPS of a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Frames per second.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_frame_count(video_path: str) -> int:
    """Get the total frame count of a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Total number of frames.
    """
    cap = cv2.VideoCapture(video_path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def discover_examples(data_root: str) -> list:
    """Discover all processable sequences in data_root.

    Used when config.examples is empty. For each subdirectory in data_root
    that contains hdVideos/hd_00_00.mp4, creates an ExampleConfig covering
    the full video (start_frame=0, num_frames=total_frames, camera="00_00",
    person_idx=0).

    Args:
        data_root: Base data directory (e.g. "/mnt/data/panoptic-toolbox").

    Returns:
        List of ExampleConfig objects, sorted by sequence name.
    """
    from config import ExampleConfig  # local import to avoid circular import

    results = []
    if not os.path.isdir(data_root):
        return results

    for seq_name in sorted(os.listdir(data_root)):
        seq_dir = os.path.join(data_root, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        video_path = get_video_path(data_root, seq_name, "00_00")
        if not os.path.exists(video_path):
            continue
        total_frames = get_video_frame_count(video_path)
        if total_frames < 2:
            continue
        results.append(ExampleConfig(
            sequence=seq_name,
            camera="00_00",
            start_frame=0,
            num_frames=total_frames,
            person_idx=0,
        ))
    return results


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

_CMU_BASE_URL = "http://domedb.perception.cs.cmu.edu/171204_pose1_sample"

def download_sequence(
    sequence_name: str,
    data_root: str | None = None,
    components: list[str] | None = None,
) -> str:
    """Download a CMU Panoptic sequence.

    Uses the CMU Panoptic getData scripts if available, otherwise
    downloads directly via wget/curl.

    Args:
        sequence_name: e.g. "171204_pose1_sample".
        data_root: Where to store data. Defaults to DEFAULT_DATA_ROOT.
        components: List of components to download. Defaults to
            ["hdVideos", "calibration", "hdPose3d_stage1_coco19"].

    Returns:
        Path to the sequence directory.
    """
    if data_root is None:
        data_root = DEFAULT_DATA_ROOT
    if components is None:
        components = ["hdVideos", "calibration", "hdPose3d_stage1_coco19"]

    seq_dir = os.path.join(data_root, sequence_name)
    os.makedirs(seq_dir, exist_ok=True)

    base_url = f"http://domedb.perception.cs.cmu.edu/{sequence_name}"

    for component in components:
        if component == "calibration":
            calib_file = f"calibration_{sequence_name}.json"
            calib_path = os.path.join(seq_dir, calib_file)
            if not os.path.exists(calib_path):
                url = f"{base_url}/{calib_file}"
                print(f"Downloading {url} ...")
                subprocess.run(
                    ["wget", "-q", "-O", calib_path, url],
                    check=True,
                )
        elif component == "hdVideos":
            vid_dir = os.path.join(seq_dir, "hdVideos")
            os.makedirs(vid_dir, exist_ok=True)
            # Download camera 00_00 by default
            vid_file = "hd_00_00.mp4"
            vid_path = os.path.join(vid_dir, vid_file)
            if not os.path.exists(vid_path):
                url = f"{base_url}/{vid_file}"
                print(f"Downloading {url} ...")
                subprocess.run(
                    ["wget", "-q", "-O", vid_path, url],
                    check=True,
                )
        elif component == "hdPose3d_stage1_coco19":
            gt_dir = os.path.join(seq_dir, "hdPose3d_stage1_coco19")
            if not os.path.exists(gt_dir):
                tar_file = f"hdPose3d_stage1_coco19.tar"
                tar_path = os.path.join(seq_dir, tar_file)
                url = f"{base_url}/{tar_file}"
                print(f"Downloading {url} ...")
                subprocess.run(
                    ["wget", "-q", "-O", tar_path, url],
                    check=True,
                )
                subprocess.run(
                    ["tar", "-xf", tar_path, "-C", seq_dir],
                    check=True,
                )
                os.remove(tar_path)

    return seq_dir
