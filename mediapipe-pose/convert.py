"""Inverse kinematics: convert MediaPipe 3D positions to Arm polar parameters."""

import numpy as np

from model.arm import Arm
from record import FrameData, SHOULDER_IDX, ELBOW_IDX, WRIST_IDX


def estimate_segment_lengths(frames: list[FrameData]) -> tuple[float, float]:
    """
    Estimate median segment lengths (shoulder-elbow, elbow-wrist) across all frames.

    Uses median for robustness against noisy frames.
    """
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
    """
    Convert 3D joint positions to Arm polar parameters with fixed segment lengths.

    Args:
        shoulder, elbow, wrist: 3D positions (3,)
        a_b_length, b_c_length: Fixed segment lengths

    Returns:
        Dict with keys: a_pos, a_b_polar (azimuth, elevation, roll), b_c_theta
    """
    # AB direction vector
    ab_vec = elbow - shoulder
    ab_norm = np.linalg.norm(ab_vec)
    if ab_norm < 1e-8:
        ab_dir = np.array([1.0, 0.0, 0.0])
    else:
        ab_dir = ab_vec / ab_norm

    # Azimuth and elevation from AB direction
    azimuth = np.arctan2(ab_dir[1], ab_dir[0])
    elevation = np.arcsin(np.clip(ab_dir[2], -1.0, 1.0))

    # BC direction vector
    bc_vec = wrist - elbow
    bc_norm = np.linalg.norm(bc_vec)
    if bc_norm < 1e-8:
        bc_dir = ab_dir
    else:
        bc_dir = bc_vec / bc_norm

    # Build local frame at B (without roll) to solve for roll and theta
    # local_x = along AB
    local_x = ab_dir

    # local_y_no_roll: perpendicular to AB in the horizontal-ish plane
    cos_az = np.cos(azimuth)
    sin_az = np.sin(azimuth)
    local_y_no_roll = np.array([-sin_az, cos_az, 0.0])

    # local_z_no_roll: perpendicular to both
    cos_el = np.cos(elevation)
    sin_el = np.sin(elevation)
    # This matches the Arm._build_local_frame convention
    local_z_no_roll = np.array([
        -cos_az * sin_el,
        -sin_az * sin_el,
        cos_el,
    ])
    # Normalize for safety
    local_z_no_roll = local_z_no_roll / (np.linalg.norm(local_z_no_roll) + 1e-10)

    # BC direction in local frame (without roll):
    # bc_dir = sin(theta) * local_x + cos(theta) * (cos(roll) * local_y_no_roll + sin(roll) * local_z_no_roll)
    # Project bc_dir onto local frame components
    bc_along_x = np.dot(bc_dir, local_x)  # sin(theta)
    bc_along_y = np.dot(bc_dir, local_y_no_roll)  # cos(theta) * cos(roll)
    bc_along_z = np.dot(bc_dir, local_z_no_roll)  # cos(theta) * sin(roll)

    # Solve for theta and roll
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
    """
    Convert one frame's MediaPipe landmarks to an Arm object.

    Args:
        frame: FrameData with 3D world landmarks
        a_b_length, b_c_length: Fixed segment lengths

    Returns:
        Arm object matching the MediaPipe pose
    """
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
