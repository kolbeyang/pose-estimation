import json
import glob
import os
import numpy as np
import cv2


def load_calibration(calib_path):
    """Load calibration JSON, return dict mapping (panel, node) -> cam_dict with numpy arrays."""
    with open(calib_path) as f:
        data = json.load(f)

    cameras = {}
    for cam in data["cameras"]:
        cam_dict = {
            "name": cam["name"],
            "type": cam["type"],
            "resolution": cam["resolution"],
            "K": np.array(cam["K"], dtype=np.float64),
            "R": np.array(cam["R"], dtype=np.float64),
            "t": np.array(cam["t"], dtype=np.float64),
            "distCoef": np.array(cam["distCoef"], dtype=np.float64),
        }
        cameras[(cam["panel"], cam["node"])] = cam_dict

    return cameras


def load_body_poses(pose_dir):
    """Load all body3DScene JSON files. Returns list of frames, each a list of body dicts."""
    files = sorted(glob.glob(os.path.join(pose_dir, "body3DScene_*.json")))
    frames = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        bodies = []
        for body in data.get("bodies", []):
            joints = np.array(body["joints19"], dtype=np.float64).reshape(19, 4)
            bodies.append({"id": body["id"], "joints19": joints})
        frames.append(bodies)
    return frames


def project_points(points_3d, cam):
    """Project 3D points to 2D using camera params with OpenCV distortion.

    Args:
        points_3d: (N, 3) array of 3D points
        cam: camera dict with K, R, t, distCoef

    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    K = cam["K"]
    R = cam["R"]
    t = cam["t"]  # (3, 1)
    Kd = cam["distCoef"]

    # Transform to camera coords: x = R @ X^T + t -> (3, N)
    X = points_3d.T  # (3, N)
    x = R @ X + t

    # Normalize by depth
    x[0:2, :] = x[0:2, :] / x[2, :]

    # Radial and tangential distortion (OpenCV model)
    # Note: must save x0, x1 before modification since they're used in both equations
    r2 = x[0, :] ** 2 + x[1, :] ** 2
    radial = 1 + Kd[0] * r2 + Kd[1] * r2**2 + Kd[4] * r2**3

    x0 = x[0, :].copy()
    x1 = x[1, :].copy()
    x[0, :] = x0 * radial + 2 * Kd[2] * x0 * x1 + Kd[3] * (r2 + 2 * x0**2)
    x[1, :] = x1 * radial + 2 * Kd[3] * x0 * x1 + Kd[2] * (r2 + 2 * x1**2)

    # Apply intrinsics
    u = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    v = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return np.stack([u, v], axis=1)  # (N, 2)


def extract_video_frames(video_path):
    """Yield RGB frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
