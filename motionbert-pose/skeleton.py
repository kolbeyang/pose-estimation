"""H36M 17-joint skeleton definition and joint mappings."""

import numpy as np

NUM_JOINTS = 17

JOINT_NAMES = [
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

# Parent joint index for each joint (-1 = root, no parent)
PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

# Bone connections: list of (parent, child)
BONES = [(PARENTS[i], i) for i in range(1, NUM_JOINTS)]

# Rest-pose bone directions (unit vectors when all local rotations are zero).
# Standing pose: legs down, spine up, shoulders out, arms at sides.
REST_DIRECTIONS = np.array([
    [0, 0, 0],       # 0: Hip (root, unused)
    [1, 0, 0],       # 1: Hip → RHip  (right)
    [0, -1, 0],      # 2: RHip → RKnee  (down)
    [0, -1, 0],      # 3: RKnee → RAnkle  (down)
    [-1, 0, 0],      # 4: Hip → LHip  (left)
    [0, -1, 0],      # 5: LHip → LKnee  (down)
    [0, -1, 0],      # 6: LKnee → LAnkle  (down)
    [0, 1, 0],       # 7: Hip → Spine  (up)
    [0, 1, 0],       # 8: Spine → Thorax  (up)
    [0, 1, 0],       # 9: Thorax → Neck  (up)
    [0, 1, 0],       # 10: Neck → Head  (up)
    [-1, 0, 0],      # 11: Thorax → LShoulder  (left)
    [0, -1, 0],      # 12: LShoulder → LElbow  (down)
    [0, -1, 0],      # 13: LElbow → LWrist  (down)
    [1, 0, 0],       # 14: Thorax → RShoulder  (right)
    [0, -1, 0],      # 15: RShoulder → RElbow  (down)
    [0, -1, 0],      # 16: RElbow → RWrist  (down)
], dtype=np.float64)

# Default bone lengths in meters (typical 1.7 m adult)
DEFAULT_BONE_LENGTHS = np.array([
    0.00,   # 0: root (no bone)
    0.12,   # 1: Hip → RHip
    0.42,   # 2: RHip → RKnee
    0.40,   # 3: RKnee → RAnkle
    0.12,   # 4: Hip → LHip
    0.42,   # 5: LHip → LKnee
    0.40,   # 6: LKnee → LAnkle
    0.22,   # 7: Hip → Spine
    0.22,   # 8: Spine → Thorax
    0.12,   # 9: Thorax → Neck
    0.14,   # 10: Neck → Head
    0.18,   # 11: Thorax → LShoulder
    0.28,   # 12: LShoulder → LElbow
    0.25,   # 13: LElbow → LWrist
    0.18,   # 14: Thorax → RShoulder
    0.28,   # 15: RShoulder → RElbow
    0.25,   # 16: RElbow → RWrist
], dtype=np.float64)

# Body groups for coloring
BODY_GROUPS = {
    "spine":     [0, 7, 8, 9, 10],
    "left_leg":  [4, 5, 6],
    "right_leg": [1, 2, 3],
    "left_arm":  [11, 12, 13],
    "right_arm": [14, 15, 16],
}

GROUP_COLORS_RGB = {
    "spine":     (0.9, 0.9, 0.9),
    "left_leg":  (0.0, 0.8, 0.8),
    "right_leg": (1.0, 0.6, 0.0),
    "left_arm":  (0.2, 0.4, 1.0),
    "right_arm": (1.0, 0.2, 0.2),
}

# Map joint index → group name
JOINT_GROUP = {}
for group_name, joint_indices in BODY_GROUPS.items():
    for j in joint_indices:
        JOINT_GROUP[j] = group_name


# ---------------------------------------------------------------------------
# Joint mapping: MediaPipe 33 landmarks → H36M 17 joints
# ---------------------------------------------------------------------------

def mediapipe_to_h36m(landmarks: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 33-landmark array to H36M 17-joint array.

    Args:
        landmarks: (33, D) where D >= 2 (can be 2D, 3D, etc.)

    Returns:
        (17, D) H36M joints
    """
    d = landmarks.shape[1]
    h36m = np.zeros((NUM_JOINTS, d), dtype=landmarks.dtype)

    # Direct mappings
    h36m[1] = landmarks[24]          # RHip
    h36m[2] = landmarks[26]          # RKnee
    h36m[3] = landmarks[28]          # RAnkle
    h36m[4] = landmarks[23]          # LHip
    h36m[5] = landmarks[25]          # LKnee
    h36m[6] = landmarks[27]          # LAnkle
    h36m[9] = landmarks[0]           # Neck (nose)
    h36m[11] = landmarks[11]         # LShoulder
    h36m[12] = landmarks[13]         # LElbow
    h36m[13] = landmarks[15]         # LWrist
    h36m[14] = landmarks[12]         # RShoulder
    h36m[15] = landmarks[14]         # RElbow
    h36m[16] = landmarks[16]         # RWrist

    # Computed joints
    h36m[0] = (landmarks[23] + landmarks[24]) / 2.0           # Hip center
    h36m[8] = (landmarks[11] + landmarks[12]) / 2.0           # Thorax (shoulder midpoint)
    h36m[7] = (h36m[0] + h36m[8]) / 2.0                      # Spine (midpoint hip↔thorax)

    # Head: extrapolate above nose using ear midpoint direction
    ear_mid = (landmarks[7] + landmarks[8]) / 2.0             # midpoint of ears
    nose = landmarks[0]
    head_dir = nose - ear_mid
    head_dir_norm = np.linalg.norm(head_dir)
    if head_dir_norm > 1e-6:
        h36m[10] = nose + head_dir / head_dir_norm * 0.12     # ~12cm above nose
    else:
        h36m[10] = nose + np.array([0, -0.12] + [0] * (d - 2))

    return h36m


def mediapipe_visibility_to_h36m(visibility: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 33-landmark visibility to H36M 17-joint visibility.

    Takes minimum visibility of contributing landmarks for averaged joints.

    Args:
        visibility: (33,) float array

    Returns:
        (17,) float array
    """
    h36m_vis = np.zeros(NUM_JOINTS, dtype=np.float64)

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
    h36m_vis[9] = visibility[0]                          # Neck/nose
    h36m_vis[10] = min(visibility[0], visibility[7],
                       visibility[8])                    # Head
    h36m_vis[11] = visibility[11]                        # LShoulder
    h36m_vis[12] = visibility[13]                        # LElbow
    h36m_vis[13] = visibility[15]                        # LWrist
    h36m_vis[14] = visibility[12]                        # RShoulder
    h36m_vis[15] = visibility[14]                        # RElbow
    h36m_vis[16] = visibility[16]                        # RWrist

    return h36m_vis


# ---------------------------------------------------------------------------
# Joint mapping: CMU Panoptic COCO19 → H36M 17 joints
# ---------------------------------------------------------------------------
# COCO19 order:
#  0=Neck, 1=Nose, 2=BodyCenter, 3=lShoulder, 4=lElbow, 5=lWrist,
#  6=lHip, 7=lKnee, 8=lAnkle, 9=rShoulder, 10=rElbow, 11=rWrist,
#  12=rHip, 13=rKnee, 14=rAnkle, 15=lEye, 16=lEar, 17=rEye, 18=rEar

def coco19_to_h36m(joints19: np.ndarray) -> np.ndarray:
    """Convert CMU Panoptic COCO19 (19, 3) to H36M (17, 3).

    Args:
        joints19: (19, 3) xyz positions

    Returns:
        (17, 3) H36M joints
    """
    h36m = np.zeros((NUM_JOINTS, 3), dtype=np.float64)

    h36m[0] = joints19[2]              # Hip ← BodyCenter
    h36m[1] = joints19[12]             # RHip
    h36m[2] = joints19[13]             # RKnee
    h36m[3] = joints19[14]             # RAnkle
    h36m[4] = joints19[6]              # LHip
    h36m[5] = joints19[7]              # LKnee
    h36m[6] = joints19[8]              # LAnkle
    h36m[7] = (joints19[2] + joints19[0]) / 2.0   # Spine ← mid(BodyCenter, Neck)
    h36m[8] = joints19[0]              # Thorax ← Neck
    h36m[9] = joints19[1]              # Neck ← Nose
    h36m[11] = joints19[3]             # LShoulder
    h36m[12] = joints19[4]             # LElbow
    h36m[13] = joints19[5]             # LWrist
    h36m[14] = joints19[9]             # RShoulder
    h36m[15] = joints19[10]            # RElbow
    h36m[16] = joints19[11]            # RWrist

    # Head: extrapolate above nose using neck→nose direction
    neck = joints19[0]
    nose = joints19[1]
    head_dir = nose - neck
    head_dir_norm = np.linalg.norm(head_dir)
    if head_dir_norm > 1e-6:
        # Extrapolate by the same distance as neck→nose
        h36m[10] = nose + head_dir
    else:
        h36m[10] = nose

    return h36m
