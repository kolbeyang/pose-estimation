"""Human body pose model using Human3.6M 17-joint skeleton.

Parameterized by root position, root rotation, per-joint local euler angles,
and bone lengths. Forward kinematics computes 3D joint positions via rotation
matrix chain.
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

# Rest direction for each bone in the parent's local frame.
# When all local rotations are zero, this is the direction each bone points.
# Index 0 is unused (root has no parent bone).
REST_DIRECTIONS = np.array([
    [0,  0,  0],   # 0: Hip (root, unused)
    [1,  0,  0],   # 1: RHip - right
    [0, -1,  0],   # 2: RKnee - down
    [0, -1,  0],   # 3: RAnkle - down
    [-1, 0,  0],   # 4: LHip - left
    [0, -1,  0],   # 5: LKnee - down
    [0, -1,  0],   # 6: LAnkle - down
    [0,  1,  0],   # 7: Spine - up
    [0,  1,  0],   # 8: Thorax - up
    [0,  1,  0],   # 9: Neck - up
    [0,  1,  0],   # 10: Head - up
    [-1, 0,  0],   # 11: LShoulder - left
    [0, -1,  0],   # 12: LElbow - down
    [0, -1,  0],   # 13: LWrist - down
    [1,  0,  0],   # 14: RShoulder - right
    [0, -1,  0],   # 15: RElbow - down
    [0, -1,  0],   # 16: RWrist - down
], dtype=np.float64)

# Default bone lengths in meters (typical adult human ~1.7m).
# Index 0 is unused. Indices 1-16 correspond to child joints.
DEFAULT_BONE_LENGTHS = np.array([
    0.00,   # 0: unused
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


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert euler angles (radians) to 3x3 rotation matrix. XYZ convention."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


class HumanPose:
    def __init__(
        self,
        root_position: np.ndarray,
        root_rotation: np.ndarray,
        local_rotations: np.ndarray,
        bone_lengths: np.ndarray,
    ):
        """
        Args:
            root_position: (3,) root joint world position (x, y, z) in meters
            root_rotation: (3,) root rotation euler angles (rx, ry, rz) in radians
            local_rotations: (17, 3) per-joint local euler angles in radians.
                             Index 0 is unused (root uses root_rotation).
            bone_lengths: (17,) bone lengths in meters. Index 0 is unused.
        """
        self.root_position = root_position.copy()
        self.root_rotation = root_rotation.copy()
        self.local_rotations = local_rotations.copy()
        self.bone_lengths = bone_lengths.copy()

    def forward_kinematics(self) -> np.ndarray:
        """Compute 3D joint positions via rotation matrix chain.

        Returns:
            (17, 3) array of joint positions in world coordinates.
        """
        positions = np.zeros((NUM_JOINTS, 3))
        rotations = [None] * NUM_JOINTS

        # Root joint
        positions[0] = self.root_position
        rotations[0] = euler_to_rotation_matrix(*self.root_rotation)

        # Child joints in topological order (indices 1..16 are already valid)
        for j in range(1, NUM_JOINTS):
            parent = PARENTS[j]
            R_local = euler_to_rotation_matrix(*self.local_rotations[j])
            R_world = rotations[parent] @ R_local
            rotations[j] = R_world

            direction = R_world @ REST_DIRECTIONS[j]
            positions[j] = positions[parent] + self.bone_lengths[j] * direction

        return positions

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        """Return joint positions as {name: (3,) array} dict."""
        positions = self.forward_kinematics()
        return {JOINT_NAMES[i]: positions[i] for i in range(NUM_JOINTS)}

    @classmethod
    def default_standing(cls) -> "HumanPose":
        """Natural standing pose with arms at sides. All local rotations zero."""
        root_position = np.array([0.0, 0.95, 0.0])
        root_rotation = np.zeros(3)
        local_rotations = np.zeros((NUM_JOINTS, 3))
        bone_lengths = DEFAULT_BONE_LENGTHS.copy()
        return cls(root_position, root_rotation, local_rotations, bone_lengths)
