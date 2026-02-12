"""BVH file parser with forward kinematics for CMU motion capture data."""

import numpy as np
from dataclasses import dataclass


@dataclass
class BVHData:
    joint_names: list[str]
    parent_indices: list[int]  # -1 for root
    offsets: np.ndarray  # (num_joints, 3)
    channel_names: list[list[str]]  # per-joint channel names (empty for end sites)
    channel_offsets: list[int]  # index into the flat motion data array (-1 for end sites)
    num_frames: int
    frame_time: float
    joint_positions: np.ndarray  # (num_frames, num_joints, 3)


def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ])


def _rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ])


def _rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ])


_ROT_FN = {"Xrotation": _rot_x, "Yrotation": _rot_y, "Zrotation": _rot_z}


def parse_bvh(filepath: str) -> BVHData:
    with open(filepath, "r") as f:
        lines = f.readlines()

    # --- Parse HIERARCHY ---
    joint_names: list[str] = []
    parent_indices: list[int] = []
    offsets: list[list[float]] = []
    channel_names: list[list[str]] = []
    channel_offsets: list[int] = []

    parent_stack: list[int] = []  # stack of parent joint indices
    total_channels = 0
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if line == "MOTION":
            break

        if line.startswith("ROOT") or line.startswith("JOINT"):
            name = line.split()[-1]
            joint_names.append(name)
            parent_idx = parent_stack[-1] if parent_stack else -1
            parent_indices.append(parent_idx)

        elif line == "End Site":
            # End sites get a generated name
            parent_name = joint_names[parent_stack[-1]]
            joint_names.append(f"{parent_name}_End")
            parent_indices.append(parent_stack[-1])

        elif line == "{":
            parent_stack.append(len(joint_names) - 1)

        elif line == "}":
            parent_stack.pop()

        elif line.startswith("OFFSET"):
            parts = line.split()
            offsets.append([float(parts[1]), float(parts[2]), float(parts[3])])
            # If this joint doesn't have channels yet, add empty
            while len(channel_names) < len(offsets):
                channel_names.append([])
                channel_offsets.append(-1)

        elif line.startswith("CHANNELS"):
            parts = line.split()
            num_ch = int(parts[1])
            ch_names = parts[2:2 + num_ch]
            # Replace the last entry (which was set to empty by OFFSET processing)
            joint_idx = len(joint_names) - 1
            # Ensure we have an entry for this joint
            while len(channel_names) <= joint_idx:
                channel_names.append([])
                channel_offsets.append(-1)
            channel_names[joint_idx] = ch_names
            channel_offsets[joint_idx] = total_channels
            total_channels += num_ch

    # Pad channel_names/channel_offsets to match joint count
    while len(channel_names) < len(joint_names):
        channel_names.append([])
        channel_offsets.append(-1)

    num_joints = len(joint_names)
    offsets_arr = np.array(offsets)

    # --- Parse MOTION ---
    # i is now pointing to the line after "MOTION"
    num_frames = 0
    frame_time = 0.0
    motion_data: list[list[float]] = []

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if line.startswith("Frames:"):
            num_frames = int(line.split(":")[1].strip())
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":")[1].strip())
        elif line:
            values = [float(v) for v in line.split()]
            motion_data.append(values)

    assert len(motion_data) == num_frames, \
        f"Expected {num_frames} frames, got {len(motion_data)}"

    # --- Forward Kinematics ---
    joint_positions = np.zeros((num_frames, num_joints, 3))

    for frame_idx in range(num_frames):
        frame_values = motion_data[frame_idx]
        world_transforms = [np.eye(4) for _ in range(num_joints)]

        for j in range(num_joints):
            ch_names = channel_names[j]
            ch_offset = channel_offsets[j]
            offset = offsets_arr[j]

            # Build local rotation matrix
            if ch_names:
                rot = np.eye(3)
                translation = offset.copy()

                for k, ch_name in enumerate(ch_names):
                    val = frame_values[ch_offset + k]
                    if ch_name in _ROT_FN:
                        rot = rot @ _ROT_FN[ch_name](val)
                    elif ch_name == "Xposition":
                        translation[0] += val
                    elif ch_name == "Yposition":
                        translation[1] += val
                    elif ch_name == "Zposition":
                        translation[2] += val

                local = np.eye(4)
                local[:3, :3] = rot
                local[:3, 3] = translation
            else:
                # End site: just translation (offset), no rotation
                local = np.eye(4)
                local[:3, 3] = offset

            parent_idx = parent_indices[j]
            if parent_idx == -1:
                world_transforms[j] = local
            else:
                world_transforms[j] = world_transforms[parent_idx] @ local

            joint_positions[frame_idx, j] = world_transforms[j][:3, 3]

    return BVHData(
        joint_names=joint_names,
        parent_indices=parent_indices,
        offsets=offsets_arr,
        channel_names=channel_names,
        channel_offsets=channel_offsets,
        num_frames=num_frames,
        frame_time=frame_time,
        joint_positions=joint_positions,
    )
