"""Human body pose model using world-space bone directions and bone lengths.

Instead of parent-relative local rotations (FK), each bone is defined by a
world-space unit direction vector and a scalar length. Joint positions are
computed by traversing the skeleton tree and adding direction * length offsets.
"""

import numpy as np

JOINT_NAMES = [
    "Hip",          # 0 - root
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

NUM_JOINTS = len(JOINT_NAMES)

PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

BODY_GROUPS = {
    "Root": [0],
    "Spine": [7, 8, 9, 10],
    "Left Arm": [11, 12, 13],
    "Right Arm": [14, 15, 16],
    "Left Leg": [4, 5, 6],
    "Right Leg": [1, 2, 3],
}

# 16 bones, each as (parent_index, child_index), in topological order
BONE_INDICES = [(PARENTS[j], j) for j in range(1, NUM_JOINTS)]

NUM_BONES = len(BONE_INDICES)

# Default bone lengths in meters (same as FK model)
DEFAULT_BONE_LENGTHS = np.array([
    0.12,   # 0: Hip -> RHip
    0.42,   # 1: RHip -> RKnee
    0.40,   # 2: RKnee -> RAnkle
    0.12,   # 3: Hip -> LHip
    0.42,   # 4: LHip -> LKnee
    0.40,   # 5: LKnee -> LAnkle
    0.22,   # 6: Hip -> Spine
    0.22,   # 7: Spine -> Thorax
    0.12,   # 8: Thorax -> Neck
    0.14,   # 9: Neck -> Head
    0.18,   # 10: Thorax -> LShoulder
    0.28,   # 11: LShoulder -> LElbow
    0.25,   # 12: LElbow -> LWrist
    0.18,   # 13: Thorax -> RShoulder
    0.28,   # 14: RShoulder -> RElbow
    0.25,   # 15: RElbow -> RWrist
], dtype=np.float64)

# Default world-space bone directions for a standing pose (same geometry as FK rest pose)
DEFAULT_BONE_DIRECTIONS = np.array([
    [ 1,  0,  0],   # 0: Hip -> RHip (right)
    [ 0, -1,  0],   # 1: RHip -> RKnee (down)
    [ 0, -1,  0],   # 2: RKnee -> RAnkle (down)
    [-1,  0,  0],   # 3: Hip -> LHip (left)
    [ 0, -1,  0],   # 4: LHip -> LKnee (down)
    [ 0, -1,  0],   # 5: LKnee -> LAnkle (down)
    [ 0,  1,  0],   # 6: Hip -> Spine (up)
    [ 0,  1,  0],   # 7: Spine -> Thorax (up)
    [ 0,  1,  0],   # 8: Thorax -> Neck (up)
    [ 0,  1,  0],   # 9: Neck -> Head (up)
    [-1,  0,  0],   # 10: Thorax -> LShoulder (left)
    [ 0, -1,  0],   # 11: LShoulder -> LElbow (down)
    [ 0, -1,  0],   # 12: LElbow -> LWrist (down)
    [ 1,  0,  0],   # 13: Thorax -> RShoulder (right)
    [ 0, -1,  0],   # 14: RShoulder -> RElbow (down)
    [ 0, -1,  0],   # 15: RElbow -> RWrist (down)
], dtype=np.float64)


def direction_to_spherical(d: np.ndarray) -> tuple[float, float]:
    """Convert unit direction vector to (azimuth, elevation) in radians."""
    azimuth = np.arctan2(d[1], d[0])
    elevation = np.arcsin(np.clip(d[2], -1.0, 1.0))
    return float(azimuth), float(elevation)


def spherical_to_direction(azimuth: float, elevation: float) -> np.ndarray:
    """Convert (azimuth, elevation) in radians to unit direction vector."""
    ce = np.cos(elevation)
    return np.array([
        ce * np.cos(azimuth),
        ce * np.sin(azimuth),
        np.sin(elevation),
    ])


class HumanPoseAnatomy3D:
    def __init__(
        self,
        root_position: np.ndarray,
        bone_directions: np.ndarray,
        bone_lengths: np.ndarray,
    ):
        """
        Args:
            root_position: (3,) root joint world position (x, y, z) in meters
            bone_directions: (16, 3) unit direction vectors in world space
            bone_lengths: (16,) bone lengths in meters
        """
        self.root_position = root_position.copy()
        self.bone_directions = bone_directions.copy()
        self.bone_lengths = bone_lengths.copy()

    def compute_positions(self) -> np.ndarray:
        """Compute 3D joint positions from bone directions and lengths.

        Returns:
            (17, 3) array of joint positions in world coordinates.
        """
        positions = np.zeros((NUM_JOINTS, 3))
        positions[0] = self.root_position

        for bone_i, (parent, child) in enumerate(BONE_INDICES):
            positions[child] = (
                positions[parent]
                + self.bone_lengths[bone_i] * self.bone_directions[bone_i]
            )

        return positions

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        """Return joint positions as {name: (3,) array} dict."""
        positions = self.compute_positions()
        return {JOINT_NAMES[i]: positions[i] for i in range(NUM_JOINTS)}

    @classmethod
    def default_standing(cls) -> "HumanPoseAnatomy3D":
        """Natural standing pose with arms at sides."""
        root_position = np.array([0.0, 0.95, 0.0])
        bone_directions = DEFAULT_BONE_DIRECTIONS.copy()
        bone_lengths = DEFAULT_BONE_LENGTHS.copy()
        return cls(root_position, bone_directions, bone_lengths)
