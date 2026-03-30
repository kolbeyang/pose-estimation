"""Unified 16-joint H36M skeleton definition, joint mappings, and body groups.

Head joint (old index 10) has been removed. All indices >= 10 are shifted down by 1.
Biased toward MotionBert naming conventions (joint 9 = "Neck").
"""

import numpy as np

NUM_JOINTS: int = 16

JOINT_NAMES: list[str] = [
    "Hip",          # 0  (root)
    "RHip",         # 1
    "RKnee",        # 2
    "RAnkle",       # 3
    "LHip",         # 4
    "LKnee",        # 5
    "LAnkle",       # 6
    "Spine",        # 7
    "Thorax",       # 8
    "Neck",         # 9
    "LShoulder",    # 10
    "LElbow",       # 11
    "LWrist",       # 12
    "RShoulder",    # 13
    "RElbow",       # 14
    "RWrist",       # 15
]

# Eval joints: exclude Hip(0), Spine(7), Thorax(8), Neck(9)
# because their definitions differ between detector, GT, and FK.
EVAL_JOINTS: list[int] = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]
EVAL_JOINT_NAMES: list[str] = [JOINT_NAMES[j] for j in EVAL_JOINTS]
NUM_EVAL_JOINTS: int = len(EVAL_JOINTS)

# Parent joint index for each joint (-1 = root, no parent)
PARENTS: np.ndarray = np.array(
    [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 8, 13, 14]
)

# Bone connections: list of (parent, child)
BONES: list[tuple[int, int]] = [(int(PARENTS[i]), i) for i in range(1, NUM_JOINTS)]

# Default bone lengths in meters (typical 1.7 m adult)
DEFAULT_BONE_LENGTHS: np.ndarray = np.array([
    0.00,   # 0: root (no bone)
    0.12,   # 1: Hip -> RHip
    0.42,   # 2: RHip -> RKnee
    0.40,   # 3: RKnee -> RAnkle
    0.12,   # 4: Hip -> LHip
    0.42,   # 5: LHip -> LKnee
    0.40,   # 6: LKnee -> LAnkle
    0.22,   # 7: Hip -> Spine
    0.22,   # 8: Spine -> Thorax
    0.12,   # 9: Thorax -> Neck
    0.18,   # 10: Thorax -> LShoulder
    0.28,   # 11: LShoulder -> LElbow
    0.25,   # 12: LElbow -> LWrist
    0.18,   # 13: Thorax -> RShoulder
    0.28,   # 14: RShoulder -> RElbow
    0.25,   # 15: RElbow -> RWrist
], dtype=np.float64)

# Rest-pose bone directions (unit vectors when all local rotations are zero).
# Camera convention: Y-down, person facing camera.
REST_DIRECTIONS: np.ndarray = np.array([
    [0, 0, 0],       # 0: Hip (root, unused)
    [-1, 0, 0],      # 1: Hip -> RHip
    [0, 1, 0],       # 2: RHip -> RKnee
    [0, 1, 0],       # 3: RKnee -> RAnkle
    [1, 0, 0],       # 4: Hip -> LHip
    [0, 1, 0],       # 5: LHip -> LKnee
    [0, 1, 0],       # 6: LKnee -> LAnkle
    [0, -1, 0],      # 7: Hip -> Spine
    [0, -1, 0],      # 8: Spine -> Thorax
    [0, -1, 0],      # 9: Thorax -> Neck
    [1, 0, 0],       # 10: Thorax -> LShoulder
    [0, 1, 0],       # 11: LShoulder -> LElbow
    [0, 1, 0],       # 12: LElbow -> LWrist
    [-1, 0, 0],      # 13: Thorax -> RShoulder
    [0, 1, 0],       # 14: RShoulder -> RElbow
    [0, 1, 0],       # 15: RElbow -> RWrist
], dtype=np.float64)

# Body groups for coloring
BODY_GROUPS: dict[str, list[int]] = {
    "spine":     [0, 7, 8, 9],
    "left_leg":  [4, 5, 6],
    "right_leg": [1, 2, 3],
    "left_arm":  [10, 11, 12],
    "right_arm": [13, 14, 15],
}

GROUP_COLORS_RGB: dict[str, tuple[float, float, float]] = {
    "spine":     (0.9, 0.9, 0.9),
    "left_leg":  (0.0, 0.8, 0.8),
    "right_leg": (1.0, 0.6, 0.0),
    "left_arm":  (0.2, 0.4, 1.0),
    "right_arm": (1.0, 0.2, 0.2),
}

# Map joint index -> group name
JOINT_GROUP: dict[int, str] = {}
for _group_name, _joint_indices in BODY_GROUPS.items():
    for _j in _joint_indices:
        JOINT_GROUP[_j] = _group_name


# ---------------------------------------------------------------------------
# Joint mapping: MPII 16-joint -> H36M 17-joint
# ---------------------------------------------------------------------------

def mpii_to_h36m(keypoints_mpii: np.ndarray) -> np.ndarray:
    """Convert MPII 16-joint keypoints to H36M 17-joint format.

    Hip = midpoint(rhip, lhip). Spine = midpoint(pelv, thrx).
    Head and Neck both map to MPII head (no separate joint).

    Args:
        keypoints_mpii: (16, D) array of MPII keypoints.

    Returns:
        (17, D) array of H36M keypoints.
    """
    ndim: int = keypoints_mpii.shape[-1]
    h36m: np.ndarray = np.zeros((17, ndim), dtype=keypoints_mpii.dtype)

    h36m[0] = (keypoints_mpii[2] + keypoints_mpii[3]) / 2.0  # Hip
    h36m[7] = (keypoints_mpii[6] + keypoints_mpii[7]) / 2.0  # Spine

    h36m[1] = keypoints_mpii[2]   # RHip
    h36m[2] = keypoints_mpii[1]   # RKnee
    h36m[3] = keypoints_mpii[0]   # RAnkle
    h36m[4] = keypoints_mpii[3]   # LHip
    h36m[5] = keypoints_mpii[4]   # LKnee
    h36m[6] = keypoints_mpii[5]   # LAnkle
    h36m[8] = keypoints_mpii[8]   # Thorax = MPII Neck
    h36m[9] = keypoints_mpii[9]   # Neck = MPII Head
    h36m[10] = keypoints_mpii[9]  # Head = MPII Head (same)
    h36m[11] = keypoints_mpii[13]  # LShoulder
    h36m[12] = keypoints_mpii[14]  # LElbow
    h36m[13] = keypoints_mpii[15]  # LWrist
    h36m[14] = keypoints_mpii[12]  # RShoulder
    h36m[15] = keypoints_mpii[11]  # RElbow
    h36m[16] = keypoints_mpii[10]  # RWrist

    return h36m


# ---------------------------------------------------------------------------
# Joint mapping: CMU Panoptic COCO19 -> H36M 16 joints
# ---------------------------------------------------------------------------

def coco19_to_h36m(joints19: np.ndarray) -> np.ndarray:
    """Convert CMU Panoptic COCO19 (19, 3) to H36M (16, 3).

    Args:
        joints19: (19, 3) xyz positions.

    Returns:
        (16, 3) H36M joints.
    """
    h36m: np.ndarray = np.zeros((NUM_JOINTS, 3), dtype=np.float64)

    h36m[0] = joints19[2]                           # Hip <- BodyCenter
    h36m[1] = joints19[12]                           # RHip
    h36m[2] = joints19[13]                           # RKnee
    h36m[3] = joints19[14]                           # RAnkle
    h36m[4] = joints19[6]                            # LHip
    h36m[5] = joints19[7]                            # LKnee
    h36m[6] = joints19[8]                            # LAnkle
    h36m[7] = (joints19[2] + joints19[0]) / 2.0     # Spine
    h36m[8] = joints19[0]                            # Thorax <- Neck
    h36m[9] = joints19[1]                            # Neck <- Nose
    h36m[10] = joints19[3]                           # LShoulder
    h36m[11] = joints19[4]                           # LElbow
    h36m[12] = joints19[5]                           # LWrist
    h36m[13] = joints19[9]                           # RShoulder
    h36m[14] = joints19[10]                          # RElbow
    h36m[15] = joints19[11]                          # RWrist

    return h36m


def h36m_17_to_16(arr: np.ndarray) -> np.ndarray:
    """Remove Head joint (index 10) from H36M 17-joint array.

    Args:
        arr: (..., 17, D) array.

    Returns:
        (..., 16, D) array with Head joint removed.
    """
    return np.delete(arr, 10, axis=-2)


# ---------------------------------------------------------------------------
# Joint mapping: MediaPipe 33 landmarks -> H36M 16 joints
# ---------------------------------------------------------------------------

def mediapipe_to_h36m(landmarks: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 33-landmark array to H36M 16-joint array.

    Args:
        landmarks: (33, D) where D >= 2.

    Returns:
        (16, D) H36M joints.
    """
    d: int = landmarks.shape[1]
    h36m: np.ndarray = np.zeros((NUM_JOINTS, d), dtype=landmarks.dtype)

    # Direct mappings
    h36m[1] = landmarks[24]          # RHip
    h36m[2] = landmarks[26]          # RKnee
    h36m[3] = landmarks[28]          # RAnkle
    h36m[4] = landmarks[23]          # LHip
    h36m[5] = landmarks[25]          # LKnee
    h36m[6] = landmarks[27]          # LAnkle
    h36m[9] = landmarks[0]           # Neck (Nose landmark)
    h36m[10] = landmarks[11]         # LShoulder
    h36m[11] = landmarks[13]         # LElbow
    h36m[12] = landmarks[15]         # LWrist
    h36m[13] = landmarks[12]         # RShoulder
    h36m[14] = landmarks[14]         # RElbow
    h36m[15] = landmarks[16]         # RWrist

    # Computed joints
    h36m[0] = (landmarks[23] + landmarks[24]) / 2.0           # Hip center
    h36m[8] = (landmarks[11] + landmarks[12]) / 2.0           # Thorax
    h36m[7] = (h36m[0] + h36m[8]) / 2.0                      # Spine

    return h36m


def mediapipe_visibility_to_h36m(visibility: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 33-landmark visibility to H36M 16-joint visibility.

    Takes minimum visibility of contributing landmarks for averaged joints.

    Args:
        visibility: (33,) float array.

    Returns:
        (16,) float array.
    """
    h36m_vis: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float64)

    h36m_vis[0] = min(visibility[23], visibility[24])   # Hip center
    h36m_vis[1] = visibility[24]                         # RHip
    h36m_vis[2] = visibility[26]                         # RKnee
    h36m_vis[3] = visibility[28]                         # RAnkle
    h36m_vis[4] = visibility[23]                         # LHip
    h36m_vis[5] = visibility[25]                         # LKnee
    h36m_vis[6] = visibility[27]                         # LAnkle
    h36m_vis[7] = min(visibility[23], visibility[24],
                      visibility[11], visibility[12])    # Spine
    h36m_vis[8] = min(visibility[11], visibility[12])    # Thorax
    h36m_vis[9] = visibility[0]                          # Neck (Nose)
    h36m_vis[10] = visibility[11]                        # LShoulder
    h36m_vis[11] = visibility[13]                        # LElbow
    h36m_vis[12] = visibility[15]                        # LWrist
    h36m_vis[13] = visibility[12]                        # RShoulder
    h36m_vis[14] = visibility[14]                        # RElbow
    h36m_vis[15] = visibility[16]                        # RWrist

    return h36m_vis
