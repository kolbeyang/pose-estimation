"""3D VPython visualization of CMU Panoptic COCO19 body poses."""

import argparse
import os
import sys

from vpython import canvas, sphere, cylinder, vector, color, rate, label

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_cmu_visualization.panoptic_loader import load_body_poses

# COCO19 skeleton edges
BODY_EDGES = [
    [0, 1], [0, 3], [3, 4], [4, 5],       # neck-nose, neck-lShoulder-lElbow-lWrist
    [0, 9], [9, 10], [10, 11],              # neck-rShoulder-rElbow-rWrist
    [0, 2], [2, 6], [6, 7], [7, 8],         # neck-bodyCenter-lHip-lKnee-lAnkle
    [2, 12], [12, 13], [13, 14],            # bodyCenter-rHip-rKnee-rAnkle
    [1, 15], [15, 16], [1, 17], [17, 18],   # nose-lEye-lEar, nose-rEye-rEar
]

LEFT_JOINTS = {3, 4, 5, 6, 7, 8, 15, 16}
RIGHT_JOINTS = {9, 10, 11, 12, 13, 14, 17, 18}

CONFIDENCE_THRESHOLD = 0.1

DATA_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "panoptic-toolbox", "171204_pose1_sample"
)

JOINT_NAMES = [
    "Neck", "Nose", "BodyCenter", "lShoulder", "lElbow", "lWrist",
    "lHip", "lKnee", "lAnkle", "rShoulder", "rElbow", "rWrist",
    "rHip", "rKnee", "rAnkle", "lEye", "lEar", "rEye", "rEar",
]


def joint_color(j):
    if j in LEFT_JOINTS:
        return color.cyan
    elif j in RIGHT_JOINTS:
        return color.red
    return color.green


def edge_color(j1, j2):
    if j1 in LEFT_JOINTS or j2 in LEFT_JOINTS:
        return color.cyan
    elif j1 in RIGHT_JOINTS or j2 in RIGHT_JOINTS:
        return color.red
    return color.green


def to_vec(arr):
    """Convert numpy xyz (in cm) to VPython vector (in meters, Y-up)."""
    # Panoptic coords: X right, Y negative-up, Z towards camera (cm)
    # Negate Y so head is above feet in VPython's Y-up convention
    return vector(float(arr[0]) / 100, -float(arr[1]) / 100, float(arr[2]) / 100)


def main():
    parser = argparse.ArgumentParser(description="3D VPython visualization of Panoptic poses")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to animate")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument("--loop", action="store_true", help="Loop animation")
    args = parser.parse_args()

    pose_dir = os.path.join(DATA_ROOT, "hdPose3d_stage1_coco19")
    print(f"Loading body poses from {pose_dir}")
    all_poses = load_body_poses(pose_dir)
    n_frames = len(all_poses)
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    print(f"Loaded {len(all_poses)} frames, rendering {n_frames}")

    # Figure out how many bodies max we need to visualize
    max_bodies = max(len(frame) for frame in all_poses[:n_frames])
    print(f"Max bodies in a frame: {max_bodies}")

    # Set up scene
    scene = canvas(
        title="CMU Panoptic - COCO19 Skeleton",
        width=1000,
        height=700,
        background=color.gray(0.15),
    )
    scene.up = vector(0, 1, 0)

    # Center camera on first frame's first body
    if all_poses[0]:
        center_pt = all_poses[0][0]["joints19"][:, :3].mean(axis=0)
        scene.center = to_vec(center_pt)
    scene.range = 1.5

    frame_label = label(pos=scene.center + vector(0, 1.5, 0),
                        text="Frame 0", height=16, box=False, color=color.white)

    # Pre-create VPython objects for each body
    body_objects = []  # list of (joint_spheres[19], edge_cylinders[len(BODY_EDGES)])
    for _ in range(max_bodies):
        joint_spheres = []
        for j in range(19):
            s = sphere(pos=vector(0, 0, 0), radius=0.02, color=joint_color(j), visible=False)
            joint_spheres.append(s)

        edge_cyls = []
        for j1, j2 in BODY_EDGES:
            c = cylinder(pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                         radius=0.008, color=edge_color(j1, j2), visible=False)
            edge_cyls.append(c)

        body_objects.append((joint_spheres, edge_cyls))

    def update_frame(frame_idx):
        bodies = all_poses[frame_idx]
        frame_label.text = f"Frame {frame_idx}/{n_frames - 1}"

        for b_idx in range(max_bodies):
            joint_spheres, edge_cyls = body_objects[b_idx]

            if b_idx >= len(bodies):
                # Hide unused body objects
                for s in joint_spheres:
                    s.visible = False
                for c in edge_cyls:
                    c.visible = False
                continue

            joints = bodies[b_idx]["joints19"]  # (19, 4)
            pts = joints[:, :3]
            conf = joints[:, 3]

            # Update joints
            for j in range(19):
                if conf[j] > CONFIDENCE_THRESHOLD:
                    joint_spheres[j].pos = to_vec(pts[j])
                    joint_spheres[j].visible = True
                else:
                    joint_spheres[j].visible = False

            # Update edges
            for e_idx, (j1, j2) in enumerate(BODY_EDGES):
                if conf[j1] > CONFIDENCE_THRESHOLD and conf[j2] > CONFIDENCE_THRESHOLD:
                    p1 = to_vec(pts[j1])
                    p2 = to_vec(pts[j2])
                    edge_cyls[e_idx].pos = p1
                    edge_cyls[e_idx].axis = p2 - p1
                    edge_cyls[e_idx].visible = True
                else:
                    edge_cyls[e_idx].visible = False

    # Initial frame
    update_frame(0)

    # Animate
    try:
        while True:
            for i in range(n_frames):
                rate(args.fps)
                update_frame(i)
            if not args.loop:
                # Hold on last frame
                while True:
                    rate(10)
    except KeyboardInterrupt:
        os._exit(0)


if __name__ == "__main__":
    main()
