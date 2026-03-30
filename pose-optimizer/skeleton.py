"""Unified 16-joint skeleton definition, joint mappings, and body groups.

Head joint (old index 10) has been removed. All indices >= 10 are shifted down by 1.
"""

import numpy as np

NUM_JOINTS: int = 16

JOINT_NAMES: list[str] = [
    "Pelvis",       # 0  (root)
    "RHip",         # 1
    "RKnee",        # 2
    "RAnkle",       # 3
    "LHip",         # 4
    "LKnee",        # 5
    "LAnkle",       # 6
    "Spine",        # 7
    "Neck",         # 8  (Base of Neck)
    "Head",         # 9  (Head Top / Nose, mode-dependent)
    "LShoulder",    # 10
    "LElbow",       # 11
    "LWrist",       # 12
    "RShoulder",    # 13
    "RElbow",       # 14
    "RWrist",       # 15
]

# Eval joints: 14 joints (12 direct + Pelvis + Neck).
# Excluded: Spine(7) = FK-internal, Head(9) = differs per pipeline.
EVAL_JOINTS: list[int] = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15]
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
    0.12,   # 1: Pelvis -> RHip
    0.42,   # 2: RHip -> RKnee
    0.40,   # 3: RKnee -> RAnkle
    0.12,   # 4: Pelvis -> LHip
    0.42,   # 5: LHip -> LKnee
    0.40,   # 6: LKnee -> LAnkle
    0.22,   # 7: Pelvis -> Spine
    0.22,   # 8: Spine -> Neck
    0.12,   # 9: Neck -> Head
    0.18,   # 10: Neck -> LShoulder
    0.28,   # 11: LShoulder -> LElbow
    0.25,   # 12: LElbow -> LWrist
    0.18,   # 13: Neck -> RShoulder
    0.28,   # 14: RShoulder -> RElbow
    0.25,   # 15: RElbow -> RWrist
], dtype=np.float64)

# Rest-pose bone directions (unit vectors when all local rotations are zero).
# Camera convention: Y-down, person facing camera.
REST_DIRECTIONS: np.ndarray = np.array([
    [0, 0, 0],       # 0: Pelvis (root, unused)
    [-1, 0, 0],      # 1: Pelvis -> RHip
    [0, 1, 0],       # 2: RHip -> RKnee
    [0, 1, 0],       # 3: RKnee -> RAnkle
    [1, 0, 0],       # 4: Pelvis -> LHip
    [0, 1, 0],       # 5: LHip -> LKnee
    [0, 1, 0],       # 6: LKnee -> LAnkle
    [0, -1, 0],      # 7: Pelvis -> Spine
    [0, -1, 0],      # 8: Spine -> Neck
    [0, -1, 0],      # 9: Neck -> Head
    [1, 0, 0],       # 10: Neck -> LShoulder
    [0, 1, 0],       # 11: LShoulder -> LElbow
    [0, 1, 0],       # 12: LElbow -> LWrist
    [-1, 0, 0],      # 13: Neck -> RShoulder
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
# Joint mapping: MPII 16-joint -> 17-joint skeleton
# ---------------------------------------------------------------------------

def mpii_to_skeleton(keypoints_mpii: np.ndarray) -> np.ndarray:
    """Convert MPII 16-joint keypoints to 17-joint skeleton format.

    Pelvis = MPII[6] direct. Spine = MPII[7] direct.
    Head and Neck both map to MPII head (no separate joint).

    Args:
        keypoints_mpii: (16, D) array of MPII keypoints.

    Returns:
        (17, D) array of skeleton keypoints.
    """
    ndim: int = keypoints_mpii.shape[-1]
    skel: np.ndarray = np.zeros((17, ndim), dtype=keypoints_mpii.dtype)

    skel[0] = keypoints_mpii[6]                       # Pelvis (direct)
    skel[7] = keypoints_mpii[7]                        # Spine (direct)

    skel[1] = keypoints_mpii[2]   # RHip
    skel[2] = keypoints_mpii[1]   # RKnee
    skel[3] = keypoints_mpii[0]   # RAnkle
    skel[4] = keypoints_mpii[3]   # LHip
    skel[5] = keypoints_mpii[4]   # LKnee
    skel[6] = keypoints_mpii[5]   # LAnkle
    skel[8] = keypoints_mpii[8]   # Neck = MPII Upper Neck
    skel[9] = keypoints_mpii[9]   # Head = MPII Head Top
    skel[10] = keypoints_mpii[9]  # Head duplicate (removed by strip_head_joint)
    skel[11] = keypoints_mpii[13]  # LShoulder
    skel[12] = keypoints_mpii[14]  # LElbow
    skel[13] = keypoints_mpii[15]  # LWrist
    skel[14] = keypoints_mpii[12]  # RShoulder
    skel[15] = keypoints_mpii[11]  # RElbow
    skel[16] = keypoints_mpii[10]  # RWrist

    return skel


# ---------------------------------------------------------------------------
# Joint mapping: CMU Panoptic COCO19 -> 16-joint skeleton
# ---------------------------------------------------------------------------

def coco19_to_skeleton(joints19: np.ndarray) -> np.ndarray:
    """Convert CMU Panoptic COCO19 (19, 3) to skeleton (16, 3).

    Args:
        joints19: (19, 3) xyz positions.

    Returns:
        (16, 3) skeleton joints.
    """
    skel: np.ndarray = np.zeros((NUM_JOINTS, 3), dtype=np.float64)

    skel[0] = joints19[2]                           # Pelvis <- BodyCenter
    skel[1] = joints19[12]                           # RHip
    skel[2] = joints19[13]                           # RKnee
    skel[3] = joints19[14]                           # RAnkle
    skel[4] = joints19[6]                            # LHip
    skel[5] = joints19[7]                            # LKnee
    skel[6] = joints19[8]                            # LAnkle
    skel[7] = (joints19[2] + joints19[0]) / 2.0     # Spine (midpoint, FK-internal)
    skel[8] = joints19[0]                            # Neck <- COCO Neck
    skel[9] = joints19[1]                            # Head <- Nose
    skel[10] = joints19[3]                           # LShoulder
    skel[11] = joints19[4]                           # LElbow
    skel[12] = joints19[5]                           # LWrist
    skel[13] = joints19[9]                           # RShoulder
    skel[14] = joints19[10]                          # RElbow
    skel[15] = joints19[11]                          # RWrist

    return skel


def strip_head_joint(arr: np.ndarray) -> np.ndarray:
    """Remove Head joint (index 10) from 17-joint array.

    Args:
        arr: (..., 17, D) array.

    Returns:
        (..., 16, D) array with Head joint removed.
    """
    return np.delete(arr, 10, axis=-2)


# ---------------------------------------------------------------------------
# Joint mapping: MediaPipe 33 landmarks -> 16-joint skeleton
# ---------------------------------------------------------------------------

def mediapipe_to_skeleton(landmarks: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 33-landmark array to 16-joint skeleton array.

    Args:
        landmarks: (33, D) where D >= 2.

    Returns:
        (16, D) skeleton joints.
    """
    d: int = landmarks.shape[1]
    skel: np.ndarray = np.zeros((NUM_JOINTS, d), dtype=landmarks.dtype)

    # Direct mappings
    skel[1] = landmarks[24]          # RHip
    skel[2] = landmarks[26]          # RKnee
    skel[3] = landmarks[28]          # RAnkle
    skel[4] = landmarks[23]          # LHip
    skel[5] = landmarks[25]          # LKnee
    skel[6] = landmarks[27]          # LAnkle
    skel[9] = landmarks[0]           # Head (Nose landmark)
    skel[10] = landmarks[11]         # LShoulder
    skel[11] = landmarks[13]         # LElbow
    skel[12] = landmarks[15]         # LWrist
    skel[13] = landmarks[12]         # RShoulder
    skel[14] = landmarks[14]         # RElbow
    skel[15] = landmarks[16]         # RWrist

    # Synthesized joints (Cat2: midpoints for MediaPipe)
    skel[0] = (landmarks[23] + landmarks[24]) / 2.0           # Pelvis
    skel[8] = (landmarks[11] + landmarks[12]) / 2.0           # Neck
    skel[7] = (skel[0] + skel[8]) / 2.0                       # Spine (FK-internal)

    return skel


def mediapipe_visibility_to_skeleton(visibility: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 33-landmark visibility to 16-joint skeleton visibility.

    Takes minimum visibility of contributing landmarks for averaged joints.

    Args:
        visibility: (33,) float array.

    Returns:
        (16,) float array.
    """
    skel_vis: np.ndarray = np.zeros(NUM_JOINTS, dtype=np.float64)

    skel_vis[0] = min(visibility[23], visibility[24])   # Pelvis
    skel_vis[1] = visibility[24]                         # RHip
    skel_vis[2] = visibility[26]                         # RKnee
    skel_vis[3] = visibility[28]                         # RAnkle
    skel_vis[4] = visibility[23]                         # LHip
    skel_vis[5] = visibility[25]                         # LKnee
    skel_vis[6] = visibility[27]                         # LAnkle
    skel_vis[7] = min(visibility[23], visibility[24],
                      visibility[11], visibility[12])    # Spine
    skel_vis[8] = min(visibility[11], visibility[12])    # Neck
    skel_vis[9] = visibility[0]                          # Head (Nose)
    skel_vis[10] = visibility[11]                        # LShoulder
    skel_vis[11] = visibility[13]                        # LElbow
    skel_vis[12] = visibility[15]                        # LWrist
    skel_vis[13] = visibility[12]                        # RShoulder
    skel_vis[14] = visibility[14]                        # RElbow
    skel_vis[15] = visibility[16]                        # RWrist

    return skel_vis
