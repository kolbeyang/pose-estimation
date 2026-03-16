"""H36M 17-joint skeleton definition, joint mappings, and body groups."""

import numpy as np

NUM_JOINTS: int = 17

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
    "Head",         # 10
    "LShoulder",    # 11
    "LElbow",       # 12
    "LWrist",       # 13
    "RShoulder",    # 14
    "RElbow",       # 15
    "RWrist",       # 16
]

# Eval joints: exclude Hip(0), Spine(7), Thorax(8), Neck(9), Head(10)
# because their definitions differ between detector, GT, and FK.
EVAL_JOINTS: list[int] = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
EVAL_JOINT_NAMES: list[str] = [JOINT_NAMES[j] for j in EVAL_JOINTS]
NUM_EVAL_JOINTS: int = len(EVAL_JOINTS)

# Parent joint index for each joint (-1 = root, no parent)
PARENTS: np.ndarray = np.array(
    [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
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
    0.12,   # 10: Neck -> Head
    0.18,   # 11: Thorax -> LShoulder
    0.28,   # 12: LShoulder -> LElbow
    0.25,   # 13: LElbow -> LWrist
    0.18,   # 14: Thorax -> RShoulder
    0.28,   # 15: RShoulder -> RElbow
    0.25,   # 16: RElbow -> RWrist
], dtype=np.float64)

# Body groups for coloring
BODY_GROUPS: dict[str, list[int]] = {
    "spine":     [0, 7, 8, 9, 10],
    "left_leg":  [4, 5, 6],
    "right_leg": [1, 2, 3],
    "left_arm":  [11, 12, 13],
    "right_arm": [14, 15, 16],
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
# MPII order:
#  0=RAnkle, 1=RKnee, 2=RHip, 3=LHip, 4=LKnee, 5=LAnkle,
#  6=Pelvis, 7=Thorax, 8=Neck, 9=Head,
#  10=RWrist, 11=RElbow, 12=RShoulder, 13=LShoulder, 14=LElbow, 15=LWrist

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

    # Computed joints
    h36m[0] = (keypoints_mpii[2] + keypoints_mpii[3]) / 2.0  # Hip = mid(rhip, lhip)
    h36m[7] = (keypoints_mpii[6] + keypoints_mpii[7]) / 2.0  # Spine = mid(pelv, thrx)

    # Direct mappings
    h36m[1] = keypoints_mpii[2]   # RHip
    h36m[2] = keypoints_mpii[1]   # RKnee
    h36m[3] = keypoints_mpii[0]   # RAnkle
    h36m[4] = keypoints_mpii[3]   # LHip
    h36m[5] = keypoints_mpii[4]   # LKnee
    h36m[6] = keypoints_mpii[5]   # LAnkle
    h36m[8] = keypoints_mpii[8]   # Thorax = MPII Neck
    h36m[9] = keypoints_mpii[9]   # Neck = MPII Head
    h36m[10] = keypoints_mpii[9]  # Head = MPII Head (same, no separate joint)
    h36m[11] = keypoints_mpii[13]  # LShoulder
    h36m[12] = keypoints_mpii[14]  # LElbow
    h36m[13] = keypoints_mpii[15]  # LWrist
    h36m[14] = keypoints_mpii[12]  # RShoulder
    h36m[15] = keypoints_mpii[11]  # RElbow
    h36m[16] = keypoints_mpii[10]  # RWrist

    return h36m


# ---------------------------------------------------------------------------
# Joint mapping: CMU Panoptic COCO19 -> H36M 17 joints
# ---------------------------------------------------------------------------
# COCO19 order:
#  0=Neck, 1=Nose, 2=BodyCenter, 3=lShoulder, 4=lElbow, 5=lWrist,
#  6=lHip, 7=lKnee, 8=lAnkle, 9=rShoulder, 10=rElbow, 11=rWrist,
#  12=rHip, 13=rKnee, 14=rAnkle, 15=lEye, 16=lEar, 17=rEye, 18=rEar

def coco19_to_h36m(joints19: np.ndarray) -> np.ndarray:
    """Convert CMU Panoptic COCO19 (19, 3) to H36M (17, 3).

    Head is extrapolated from neck->nose direction.

    Args:
        joints19: (19, 3) xyz positions.

    Returns:
        (17, 3) H36M joints.
    """
    h36m: np.ndarray = np.zeros((NUM_JOINTS, 3), dtype=np.float64)

    h36m[0] = joints19[2]                           # Hip <- BodyCenter
    h36m[1] = joints19[12]                           # RHip
    h36m[2] = joints19[13]                           # RKnee
    h36m[3] = joints19[14]                           # RAnkle
    h36m[4] = joints19[6]                            # LHip
    h36m[5] = joints19[7]                            # LKnee
    h36m[6] = joints19[8]                            # LAnkle
    h36m[7] = (joints19[2] + joints19[0]) / 2.0     # Spine <- mid(BodyCenter, Neck)
    h36m[8] = joints19[0]                            # Thorax <- Neck
    h36m[9] = joints19[1]                            # Neck <- Nose
    # Head: extrapolate nose + (nose - neck)
    h36m[10] = joints19[1] + (joints19[1] - joints19[0])
    h36m[11] = joints19[3]                           # LShoulder
    h36m[12] = joints19[4]                           # LElbow
    h36m[13] = joints19[5]                           # LWrist
    h36m[14] = joints19[9]                           # RShoulder
    h36m[15] = joints19[10]                          # RElbow
    h36m[16] = joints19[11]                          # RWrist

    return h36m
