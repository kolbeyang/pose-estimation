"""VPython visualization for CMU motion capture BVH data."""

import sys

from vpython import canvas, sphere, cylinder, vector, color, rate

from bvh_parser import parse_bvh


# Color assignment by body part
LEFT_JOINTS = {
    "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "LeftToeBase_End", "LeftShoulder", "LeftArm", "LeftForeArm",
    "LeftHand", "LeftFingerBase", "LeftHandIndex1", "LeftHandIndex1_End",
    "LThumb", "LThumb_End",
}
RIGHT_JOINTS = {
    "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "RightToeBase_End", "RightShoulder", "RightArm", "RightForeArm",
    "RightHand", "RightFingerBase", "RightHandIndex1", "RightHandIndex1_End",
    "RThumb", "RThumb_End",
}


def joint_color(name: str) -> vector:
    if name in LEFT_JOINTS:
        return color.cyan
    if name in RIGHT_JOINTS:
        return color.red
    return color.green


def bvh_to_vpython(pos) -> vector:
    """Convert BVH Y-up coordinates to VPython Z-up: (x, y, z) -> (x, z, y)."""
    return vector(float(pos[0]), float(pos[2]), float(pos[1]))


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_mocap.py <path_to.bvh>")
        sys.exit(1)

    filepath = sys.argv[1]
    data = parse_bvh(filepath)

    print(f"Joints: {len(data.joint_names)}")
    print(f"Frames: {data.num_frames}")
    fps = round(1.0 / data.frame_time) if data.frame_time > 0 else 120
    print(f"FPS: {fps}")

    # Set up scene
    scene = canvas(
        title=f"CMU Mocap: {filepath}",
        width=1000,
        height=700,
        background=color.gray(0.15),
    )
    scene.up = vector(0, 0, 1)
    scene.forward = vector(-1, -0.5, -0.3)

    # Auto-scale: find the bounding box across all frames
    all_pos = data.joint_positions.reshape(-1, 3)
    center = all_pos.mean(axis=0)
    extent = all_pos.max(axis=0) - all_pos.min(axis=0)
    scale = max(extent) * 0.6
    scene.center = bvh_to_vpython(center)
    scene.range = scale

    joint_radius = scale * 0.015

    # Create VPython objects for the first motion frame (skip T-pose at frame 0)
    init_frame = 1 if data.num_frames > 1 else 0
    frame0 = data.joint_positions[init_frame]
    spheres = []
    cylinders = []

    for j, name in enumerate(data.joint_names):
        pos = bvh_to_vpython(frame0[j])
        col = joint_color(name)
        s = sphere(pos=pos, radius=joint_radius, color=col)
        spheres.append(s)

        parent_idx = data.parent_indices[j]
        if parent_idx >= 0:
            parent_pos = bvh_to_vpython(frame0[parent_idx])
            c = cylinder(
                pos=parent_pos,
                axis=pos - parent_pos,
                radius=joint_radius * 0.4,
                color=col,
            )
            cylinders.append((j, parent_idx, c))

    # Animation loop — skip frame 0 on wrap since it's the rest/T-pose
    start_frame = 1 if data.num_frames > 1 else 0
    frame = start_frame
    try:
        while True:
            rate(fps)
            positions = data.joint_positions[frame]

            for j in range(len(data.joint_names)):
                spheres[j].pos = bvh_to_vpython(positions[j])

            for j, parent_idx, cyl in cylinders:
                p = bvh_to_vpython(positions[parent_idx])
                ch = bvh_to_vpython(positions[j])
                cyl.pos = p
                cyl.axis = ch - p

            frame = frame + 1
            if frame >= data.num_frames:
                frame = start_frame
    except KeyboardInterrupt:
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
