"""MediaPipe PoseLandmarker detection with H36M mapping.

Adapted from mediapipe-pose/detect.py for the unified pose-optimizer.
"""

import os
import sys
import urllib.request

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Lazy import of the pip mediapipe package.
# Since our directory is also called 'mediapipe', we load the pip version
# from its known site-packages location using importlib.util.
# ---------------------------------------------------------------------------

_mp = None
_BaseOptions = None
_vision = None


def _get_mediapipe():
    """Lazily import the pip mediapipe package from site-packages."""
    global _mp, _BaseOptions, _vision
    if _mp is not None:
        return _mp, _BaseOptions, _vision

    import importlib.util

    # Find the site-packages path containing the pip mediapipe
    _our_dir = os.path.normpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mp_init = None
    for path_entry in sys.path:
        if "site-packages" not in path_entry:
            continue
        candidate = os.path.join(path_entry, "mediapipe", "__init__.py")
        if os.path.exists(candidate):
            mp_init = candidate
            mp_base = os.path.join(path_entry, "mediapipe")
            break

    if mp_init is None:
        raise ImportError("pip mediapipe not found in site-packages")

    # Temporarily clear our local mediapipe from sys.modules
    saved_mods = {}
    for key in list(sys.modules.keys()):
        if key == "mediapipe" or key.startswith("mediapipe."):
            saved_mods[key] = sys.modules.pop(key)

    # Also temporarily adjust sys.path -- remove both our dir and '' (cwd)
    orig_path = sys.path[:]
    cwd = os.getcwd()
    cwd_norm = os.path.normpath(cwd)
    sys.path = [
        p for p in sys.path
        if os.path.normpath(p or cwd) != _our_dir
        and os.path.normpath(p or cwd) != cwd_norm
    ]

    try:
        # Now import the pip mediapipe -- it won't find our local one
        import mediapipe as mp_mod
        from mediapipe.tasks.python import BaseOptions as BO
        from mediapipe.tasks.python import vision as vis

        # Save references to the pip modules we need
        _mp = mp_mod
        _BaseOptions = BO
        _vision = vis

        # Keep the pip mediapipe modules in sys.modules alongside ours
        # by storing them under aliased names
        pip_mods = {}
        for key in list(sys.modules.keys()):
            if key == "mediapipe" or key.startswith("mediapipe."):
                pip_mods["_pip_" + key] = sys.modules[key]

        return _mp, _BaseOptions, _vision
    finally:
        # Restore path
        sys.path = orig_path
        # Restore our local mediapipe modules
        sys.modules.update(saved_mods)

# Add parent dir to path for skeleton imports
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from skeleton import mediapipe_to_h36m, mediapipe_visibility_to_h36m, NUM_JOINTS

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(SCRIPT_DIR, "pose_landmarker_lite.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)

# Also check mediapipe-pose for the model file as fallback
_MEDIAPIPE_POSE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", "mediapipe-pose")
)


def _ensure_model() -> None:
    """Download the pose landmarker model if missing."""
    if os.path.exists(_MODEL_PATH):
        return

    # Check if model exists in mediapipe-pose directory
    fallback_path = os.path.join(_MEDIAPIPE_POSE_DIR, "pose_landmarker_lite.task")
    if os.path.exists(fallback_path):
        print(f"    Copying model from {fallback_path}")
        import shutil
        shutil.copy2(fallback_path, _MODEL_PATH)
        return

    print(f"    Downloading pose model to {_MODEL_PATH}...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


def detect_poses(
    frames_rgb: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Run MediaPipe PoseLandmarker on a list of RGB frames.

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.

    Returns:
        keypoints_2d: List of (16, 2) pixel coordinates (H36M 16-joint).
        keypoints_3d: List of (16, 3) world coordinates in meters (hip-relative).
        visibility: List of (16,) visibility scores [0, 1].
    """
    _ensure_model()

    mp, BaseOptions, vision = _get_mediapipe()

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_MODEL_PATH),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        output_segmentation_masks=False,
    )

    all_kp_2d: list[np.ndarray] = []
    all_kp_3d: list[np.ndarray] = []
    all_vis: list[np.ndarray] = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for frame in frames_rgb:
            h, w = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect(mp_image)

            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                all_kp_2d.append(np.zeros((NUM_JOINTS, 2), dtype=np.float64))
                all_kp_3d.append(np.zeros((NUM_JOINTS, 3), dtype=np.float64))
                all_vis.append(np.zeros(NUM_JOINTS, dtype=np.float64))
                continue

            pose_lm = result.pose_landmarks[0]
            world_lm = result.pose_world_landmarks[0]

            # 2D pixel landmarks (33, 2)
            mp_2d = np.array(
                [[lm.x * w, lm.y * h] for lm in pose_lm],
                dtype=np.float64,
            )
            # Visibility (33,)
            mp_vis = np.array(
                [lm.visibility for lm in pose_lm],
                dtype=np.float64,
            )
            # 3D world landmarks (33, 3)
            mp_3d = np.array(
                [[lm.x, lm.y, lm.z] for lm in world_lm],
                dtype=np.float64,
            )

            # Convert to H36M 16-joint
            all_kp_2d.append(mediapipe_to_h36m(mp_2d))
            all_kp_3d.append(mediapipe_to_h36m(mp_3d))
            all_vis.append(mediapipe_visibility_to_h36m(mp_vis))

    return all_kp_2d, all_kp_3d, all_vis


def mediapipe_3d_to_camera(
    mp_3d: np.ndarray,
    mp_2d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Convert MediaPipe hip-relative 3D to camera-space 3D using solvePnP.

    Args:
        mp_3d: (16, 3) MediaPipe world coords (hip-relative, meters).
        mp_2d: (16, 2) pixel coordinates.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (16, 3) positions in camera coordinates (meters).
    """
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    valid = np.linalg.norm(mp_2d, axis=1) > 1.0
    if valid.sum() < 4:
        return _depth_heuristic_fallback(mp_3d, mp_2d, fx, fy, cx, cy)

    obj_pts = mp_3d[valid].astype(np.float64)
    img_pts = mp_2d[valid].astype(np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success:
        return _depth_heuristic_fallback(mp_3d, mp_2d, fx, fy, cx, cy)

    R, _ = cv2.Rodrigues(rvec)
    cam_3d = (R @ mp_3d.T).T + tvec.T
    return cam_3d


def _depth_heuristic_fallback(
    mp_3d: np.ndarray,
    mp_2d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Fallback depth estimation when solvePnP fails."""
    thorax_2d = mp_2d[8]
    ankle_mid_2d = (mp_2d[3] + mp_2d[6]) / 2.0
    pixel_height = abs(thorax_2d[1] - ankle_mid_2d[1])

    assumed_height_m = 1.2
    if pixel_height > 20:
        root_depth = fx * assumed_height_m / pixel_height
    else:
        root_depth = 3.0
    root_depth = float(np.clip(root_depth, 1.0, 8.0))

    hip_2d = mp_2d[0]
    root_cam = np.array([
        (hip_2d[0] - cx) * root_depth / fx,
        (hip_2d[1] - cy) * root_depth / fy,
        root_depth,
    ])

    offsets_cam = mp_3d.copy()
    offsets_cam[:, 1] = -mp_3d[:, 1]
    offsets_cam[:, 2] = -mp_3d[:, 2]

    return root_cam + offsets_cam
