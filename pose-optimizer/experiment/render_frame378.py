"""Render raw and optimized skeletons for frame 378 at a fixed camera angle.

Saves two transparent PNGs to ~/Downloads/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from skeleton import BONES

TRAJ_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "output/frame378_v4/171204_pose1_360/mediapipe/trajectories.json",
)
FRAME_IDX = 6

RAW_COLOR = "#EF9A9A"
OPT_COLOR = "#F44336"

# Front view: looking down -Z axis, slight rotation for 3/4 depth
VIEW_ELEV = -28.01
VIEW_AZIM = 145.08
VIEW_ROLL = -70.30


def draw_skeleton(ax, positions, color, linewidth=3, marker_size=35):
    positions = np.array(positions)
    for parent, child in BONES:
        ax.plot3D(
            [positions[parent, 0], positions[child, 0]],
            [positions[parent, 1], positions[child, 1]],
            [positions[parent, 2], positions[child, 2]],
            color=color, linewidth=linewidth, solid_capstyle="round",
        )
    ax.scatter3D(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color=color, s=marker_size, depthshade=False, zorder=5,
    )


def flip_y(positions):
    """Flip Y axis so head points up (camera coords have Y-down)."""
    p = np.array(positions).copy()
    p[:, 1] = -p[:, 1]
    return p


def render_one(positions, color, label, output_path, center, max_range):
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")

    draw_skeleton(ax, positions, color)

    # Equal range on all axes so depth isn't exaggerated
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_box_aspect([1, 1, 1])

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM, roll=VIEW_ROLL)

    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    fig.savefig(output_path, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved {label}: {output_path}")


def main():
    with open(TRAJ_PATH) as f:
        data = json.load(f)

    raw = flip_y(np.array(data["raw_prediction"][FRAME_IDX]))
    opt = flip_y(np.array(data["optimized_prediction"][FRAME_IDX]))

    # Match viewer exactly: compute limits from opt only
    center = opt.mean(axis=0)
    xy_range = np.max(np.abs(opt - center)) * 1.3

    render_one(raw, RAW_COLOR, "Raw MediaPipe",
               os.path.expanduser("~/Downloads/skeleton_raw.png"), center, xy_range)
    render_one(opt, OPT_COLOR, "Optimized",
               os.path.expanduser("~/Downloads/skeleton_optimized.png"), center, xy_range)


if __name__ == "__main__":
    main()
