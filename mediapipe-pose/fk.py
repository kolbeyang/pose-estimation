"""Differentiable forward kinematics for the H36M 16-joint skeleton.

Uses axis-angle rotation representation (no gimbal lock).
All operations use PyTorch for autograd support.
"""

import numpy as np
import torch

from skeleton import PARENTS, REST_DIRECTIONS, NUM_JOINTS


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (3,) to 3x3 rotation matrix via Rodrigues' formula."""
    angle = torch.norm(aa)
    if angle < 1e-8:
        return torch.eye(3, dtype=aa.dtype)
    axis = aa / angle
    K = torch.zeros(3, 3, dtype=aa.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]
    R = (
        torch.eye(3, dtype=aa.dtype)
        + torch.sin(angle) * K
        + (1 - torch.cos(angle)) * (K @ K)
    )
    return R


def forward_kinematics(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    local_rots: torch.Tensor,
    bone_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute 3D joint positions from FK parameters.

    Args:
        root_pos: (3,) root hip position in camera coordinates.
        root_rot: (3,) axis-angle rotation for the root joint.
        local_rots: (NUM_JOINTS, 3) axis-angle local rotations per joint.
        bone_lengths: (NUM_JOINTS,) scalar bone lengths.

    Returns:
        (NUM_JOINTS, 3) world positions of each joint.
    """
    rest_dirs = torch.tensor(REST_DIRECTIONS, dtype=torch.float32)

    positions: list[torch.Tensor] = [torch.zeros(3)] * NUM_JOINTS
    rotations: list[torch.Tensor] = [torch.eye(3)] * NUM_JOINTS

    positions[0] = root_pos
    rotations[0] = _axis_angle_to_matrix(root_rot)

    for j in range(1, NUM_JOINTS):
        parent = int(PARENTS[j])
        R_local = _axis_angle_to_matrix(local_rots[j])
        R_world = rotations[parent] @ R_local
        rotations[j] = R_world

        direction = R_world @ rest_dirs[j]
        positions[j] = positions[parent] + bone_lengths[j] * direction

    return torch.stack(positions)  # (NUM_JOINTS, 3)


# ---------------------------------------------------------------------------
# Inverse: positions → FK parameters (for initialization from MediaPipe)
# ---------------------------------------------------------------------------

def _rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix mapping unit vector *v_from* → *v_to*."""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-10)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-10)

    cross = np.cross(v_from, v_to)
    dot = float(np.dot(v_from, v_to))
    sin_angle = np.linalg.norm(cross)

    if sin_angle < 1e-8:
        if dot > 0:
            return np.eye(3)
        # Anti-parallel: 180° rotation around any perpendicular axis.
        # Find a perpendicular axis (pick the coordinate axis least aligned with v_from).
        abs_from = np.abs(v_from)
        if abs_from[0] <= abs_from[1] and abs_from[0] <= abs_from[2]:
            perp = np.array([1.0, 0.0, 0.0])
        elif abs_from[1] <= abs_from[2]:
            perp = np.array([0.0, 1.0, 0.0])
        else:
            perp = np.array([0.0, 0.0, 1.0])
        axis = np.cross(v_from, perp)
        axis = axis / np.linalg.norm(axis)
        # R = -I + 2 * n @ n^T (180° rotation around axis n)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    axis = cross / sin_angle
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to axis-angle (3,)."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)

    # Standard extraction from antisymmetric part
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    axis_norm = np.linalg.norm(axis)

    if axis_norm > 1e-6:
        return axis / axis_norm * angle

    # Near 180°: antisymmetric part vanishes.
    # Extract axis from symmetric part: R + I = 2 * cos(0) * ... doesn't work,
    # but R = -I + 2*nn^T at 180°, so (R + I)/2 = nn^T.
    # The axis is the eigenvector with eigenvalue 1, or the column of (R+I) with largest norm.
    S = R + np.eye(3)
    col_norms = np.linalg.norm(S, axis=0)
    best = np.argmax(col_norms)
    if col_norms[best] < 1e-10:
        return np.zeros(3)
    axis = S[:, best] / col_norms[best]
    return axis * angle


def positions_to_fk_params(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert (NUM_JOINTS, 3) joint positions to FK parameters.

    Returns:
        (root_pos, root_rot, local_rots, bone_lengths) all as numpy arrays.
    """
    root_pos = positions[0].copy()

    # Bone lengths
    bone_lengths = np.zeros(NUM_JOINTS, dtype=np.float64)
    for j in range(1, NUM_JOINTS):
        parent = int(PARENTS[j])
        bone_lengths[j] = np.linalg.norm(positions[j] - positions[parent])

    # Root rotation: align rest-pose body frame with actual skeleton
    # Use spine direction as "up" and hip axis as "right"
    spine_dir = positions[8] - positions[0]  # thorax - hip
    hip_axis = positions[1] - positions[4]   # right hip - left hip

    spine_len = np.linalg.norm(spine_dir)
    hip_len = np.linalg.norm(hip_axis)

    if spine_len > 1e-6 and hip_len > 1e-6:
        up = spine_dir / spine_len
        right = hip_axis / hip_len
        forward = np.cross(up, right)
        forward_len = np.linalg.norm(forward)
        if forward_len > 1e-6:
            forward = forward / forward_len
            right = np.cross(forward, up)
            right = right / (np.linalg.norm(right) + 1e-10)
            # Build rotation mapping rest axes to body axes.
            # Rest convention: right=(-1,0,0), up=(0,-1,0)
            # Implicit rest forward = cross(up_rest, right_rest) = (0,0,-1)
            # R @ (-1,0,0) = right → R[:,0] = -right
            # R @ (0,-1,0) = up    → R[:,1] = -up
            # R @ (0,0,-1) = fwd   → R[:,2] = -forward
            R_root = np.column_stack([-right, -up, -forward])
        else:
            R_root = np.eye(3)
    else:
        R_root = np.eye(3)

    root_rot = _rotation_matrix_to_axis_angle(R_root)

    # Local rotations for each joint
    world_rots = [None] * NUM_JOINTS
    local_rots = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    world_rots[0] = R_root

    rest_dirs = REST_DIRECTIONS.astype(np.float64)

    for j in range(1, NUM_JOINTS):
        parent = int(PARENTS[j])
        bone_vec = positions[j] - positions[parent]
        bl = bone_lengths[j]

        if bl < 1e-6:
            world_rots[j] = world_rots[parent].copy()
            continue

        actual_dir = bone_vec / bl
        rest_dir = rest_dirs[j]

        # World rotation that maps rest_dir → actual_dir
        R_world = _rotation_between_vectors(rest_dir, actual_dir)
        world_rots[j] = R_world

        # Local rotation: R_world = R_parent @ R_local → R_local = R_parent^T @ R_world
        R_local = world_rots[parent].T @ R_world
        local_rots[j] = _rotation_matrix_to_axis_angle(R_local)

    return root_pos, root_rot, local_rots, bone_lengths
