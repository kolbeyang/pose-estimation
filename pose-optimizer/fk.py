"""Differentiable forward kinematics for the H36M 16-joint skeleton.

Uses axis-angle rotation representation (no gimbal lock).
All operations use PyTorch for autograd support.
"""

import numpy as np
import torch

from skeleton import PARENTS, REST_DIRECTIONS, NUM_JOINTS


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (3,) to 3x3 rotation matrix via Rodrigues' formula.

    Args:
        aa: (3,) axis-angle vector.

    Returns:
        (3, 3) rotation matrix.
    """
    angle: torch.Tensor = torch.norm(aa)
    if angle < 1e-8:
        return torch.eye(3, dtype=aa.dtype)
    axis: torch.Tensor = aa / angle
    K: torch.Tensor = torch.zeros(3, 3, dtype=aa.dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]
    R: torch.Tensor = (
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
    rest_dirs: torch.Tensor = torch.tensor(REST_DIRECTIONS, dtype=torch.float32)

    positions: list[torch.Tensor] = [torch.zeros(3)] * NUM_JOINTS
    rotations: list[torch.Tensor] = [torch.eye(3)] * NUM_JOINTS

    positions[0] = root_pos
    rotations[0] = _axis_angle_to_matrix(root_rot)

    for j in range(1, NUM_JOINTS):
        parent: int = int(PARENTS[j])
        R_local: torch.Tensor = _axis_angle_to_matrix(local_rots[j])
        R_world: torch.Tensor = rotations[parent] @ R_local
        rotations[j] = R_world

        direction: torch.Tensor = R_world @ rest_dirs[j]
        positions[j] = positions[parent] + bone_lengths[j] * direction

    return torch.stack(positions)  # (NUM_JOINTS, 3)


def _axis_angle_to_matrix_batch(aa: torch.Tensor) -> torch.Tensor:
    """Batch axis-angle (F, 3) -> (F, 3, 3) rotation matrices via Rodrigues.

    Args:
        aa: (F, 3) axis-angle vectors.

    Returns:
        (F, 3, 3) rotation matrices.
    """
    F_dim: int = aa.shape[0]
    angle: torch.Tensor = torch.norm(aa, dim=-1, keepdim=True)  # (F, 1)
    safe_angle: torch.Tensor = torch.clamp(angle, min=1e-8)
    axis: torch.Tensor = aa / safe_angle  # (F, 3)

    zero: torch.Tensor = torch.zeros(F_dim, dtype=aa.dtype)
    K: torch.Tensor = torch.stack([
        zero, -axis[:, 2], axis[:, 1],
        axis[:, 2], zero, -axis[:, 0],
        -axis[:, 1], axis[:, 0], zero,
    ], dim=-1).reshape(F_dim, 3, 3)

    eye: torch.Tensor = torch.eye(3, dtype=aa.dtype).unsqueeze(0)  # (1, 3, 3)
    sin_a: torch.Tensor = torch.sin(angle).unsqueeze(-1)  # (F, 1, 1)
    cos_a: torch.Tensor = torch.cos(angle).unsqueeze(-1)  # (F, 1, 1)

    R: torch.Tensor = eye + sin_a * K + (1 - cos_a) * (K @ K)

    small: torch.Tensor = (angle.squeeze(-1) < 1e-8).float()  # (F,)
    R = R * (1 - small).reshape(F_dim, 1, 1) + eye * small.reshape(F_dim, 1, 1)

    return R


def forward_kinematics_batch(
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    local_rots: torch.Tensor,
    bone_lengths: torch.Tensor,
) -> torch.Tensor:
    """Batch forward kinematics across frames.

    Args:
        root_pos: (F, 3) root positions.
        root_rot: (F, 3) root axis-angle rotations.
        local_rots: (F, J, 3) local axis-angle rotations per joint.
        bone_lengths: (J,) shared bone lengths.

    Returns:
        (F, J, 3) world positions.
    """
    F_dim: int = root_pos.shape[0]
    rest_dirs: torch.Tensor = torch.tensor(REST_DIRECTIONS, dtype=torch.float32)

    positions: list[torch.Tensor] = [torch.zeros(F_dim, 3)] * NUM_JOINTS
    rotations: list[torch.Tensor] = [torch.eye(3).unsqueeze(0).expand(F_dim, -1, -1)] * NUM_JOINTS

    positions[0] = root_pos
    rotations[0] = _axis_angle_to_matrix_batch(root_rot)

    for j in range(1, NUM_JOINTS):
        parent: int = int(PARENTS[j])
        R_local: torch.Tensor = _axis_angle_to_matrix_batch(local_rots[:, j, :])
        R_world: torch.Tensor = rotations[parent] @ R_local
        rotations[j] = R_world

        direction: torch.Tensor = (R_world @ rest_dirs[j].unsqueeze(-1)).squeeze(-1)
        positions[j] = positions[parent] + bone_lengths[j] * direction

    return torch.stack(positions, dim=1)  # (F, J, 3)


# ---------------------------------------------------------------------------
# Inverse: positions -> FK parameters (for initialization)
# ---------------------------------------------------------------------------


def _rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix mapping unit vector v_from -> v_to.

    Handle anti-parallel case with proper 180-degree rotation.

    Args:
        v_from: (3,) source unit vector.
        v_to: (3,) target unit vector.

    Returns:
        (3, 3) rotation matrix.
    """
    v_from = v_from / (np.linalg.norm(v_from) + 1e-10)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-10)

    cross: np.ndarray = np.cross(v_from, v_to)
    dot: float = float(np.dot(v_from, v_to))
    sin_angle: float = float(np.linalg.norm(cross))

    if sin_angle < 1e-8:
        if dot > 0:
            return np.eye(3)
        abs_from: np.ndarray = np.abs(v_from)
        if abs_from[0] <= abs_from[1] and abs_from[0] <= abs_from[2]:
            perp: np.ndarray = np.array([1.0, 0.0, 0.0])
        elif abs_from[1] <= abs_from[2]:
            perp = np.array([0.0, 1.0, 0.0])
        else:
            perp = np.array([0.0, 0.0, 1.0])
        axis: np.ndarray = np.cross(v_from, perp)
        axis = axis / np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    axis = cross / sin_angle
    angle: float = float(np.arccos(np.clip(dot, -1.0, 1.0)))

    K: np.ndarray = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to axis-angle (3,).

    Handle near-180-degree case where antisymmetric part vanishes.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3,) axis-angle vector.
    """
    angle: float = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))
    if angle < 1e-8:
        return np.zeros(3)

    axis: np.ndarray = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    axis_norm: float = float(np.linalg.norm(axis))

    if axis_norm > 1e-6:
        return axis / axis_norm * angle

    S: np.ndarray = R + np.eye(3)
    col_norms: np.ndarray = np.linalg.norm(S, axis=0)
    best: int = int(np.argmax(col_norms))
    if col_norms[best] < 1e-10:
        return np.zeros(3)
    axis = S[:, best] / col_norms[best]
    return axis * angle


def positions_to_fk_params(
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert (NUM_JOINTS, 3) joint positions to FK parameters (inverse FK).

    Args:
        positions: (16, 3) joint positions.

    Returns:
        (root_pos, root_rot, local_rots, bone_lengths) all as numpy arrays.
    """
    root_pos: np.ndarray = positions[0].copy()

    bone_lengths: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float64)
    for j in range(1, NUM_JOINTS):
        parent: int = int(PARENTS[j])
        bone_lengths[j] = float(np.linalg.norm(positions[j] - positions[parent]))

    spine_dir: np.ndarray = positions[8] - positions[0]
    hip_axis: np.ndarray = positions[1] - positions[4]

    spine_len: float = float(np.linalg.norm(spine_dir))
    hip_len: float = float(np.linalg.norm(hip_axis))

    R_root: np.ndarray
    if spine_len > 1e-6 and hip_len > 1e-6:
        up: np.ndarray = spine_dir / spine_len
        right: np.ndarray = hip_axis / hip_len
        forward: np.ndarray = np.cross(up, right)
        forward_len: float = float(np.linalg.norm(forward))
        if forward_len > 1e-6:
            forward = forward / forward_len
            right = np.cross(forward, up)
            right = right / (float(np.linalg.norm(right)) + 1e-10)
            R_root = np.column_stack([-right, -up, -forward])
        else:
            R_root = np.eye(3)
    else:
        R_root = np.eye(3)

    root_rot: np.ndarray = _rotation_matrix_to_axis_angle(R_root)

    world_rots: list[np.ndarray | None] = [None] * NUM_JOINTS
    local_rots: np.ndarray = np.zeros((NUM_JOINTS, 3), dtype=np.float64)
    world_rots[0] = R_root

    rest_dirs: np.ndarray = REST_DIRECTIONS.astype(np.float64)

    for j in range(1, NUM_JOINTS):
        parent = int(PARENTS[j])
        bone_vec: np.ndarray = positions[j] - positions[parent]
        bl: float = bone_lengths[j]

        if bl < 1e-6:
            world_rots[j] = world_rots[parent].copy()  # type: ignore[union-attr]
            continue

        actual_dir: np.ndarray = bone_vec / bl
        rest_dir: np.ndarray = rest_dirs[j]

        R_world: np.ndarray = _rotation_between_vectors(rest_dir, actual_dir)
        world_rots[j] = R_world

        R_local: np.ndarray = world_rots[parent].T @ R_world  # type: ignore[union-attr]
        local_rots[j] = _rotation_matrix_to_axis_angle(R_local)

    return root_pos, root_rot, local_rots, bone_lengths
