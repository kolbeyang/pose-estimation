"""Interactive 3D skeleton viewer for frame 378 of 171204_pose1.

Shows optimized skeleton (red). Rotate to your preferred angle, press 'p' to
print camera coordinates to console. Give those back to render_frame378.py.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from skeleton import BONES

TRAJ_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "output/frame378_v6_no_anneal/171204_pose1_360/mediapipe/trajectories.json",
)
FRAME_IDX = 6  # frame 378 = index 6 (start=360, step=3)

OPT_COLOR = "#F44336"


def draw_skeleton(ax, positions, color, linewidth=2.5, marker_size=30):
    positions = np.array(positions)
    # Flip Y so head points up (camera coords have Y-down)
    positions = positions.copy()
    positions[:, 1] = -positions[:, 1]
    for parent, child in BONES:
        ax.plot3D(
            [positions[parent, 0], positions[child, 0]],
            [positions[parent, 1], positions[child, 1]],
            [positions[parent, 2], positions[child, 2]],
            color=color, linewidth=linewidth, solid_capstyle="round",
        )
    ax.scatter3D(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color=color, s=marker_size, depthshade=False,
    )


def main():
    with open(TRAJ_PATH) as f:
        data = json.load(f)

    opt = np.array(data["optimized_prediction"][FRAME_IDX])

    # Flip Y for display
    opt_flip = opt.copy()
    opt_flip[:, 1] = -opt_flip[:, 1]

    center = opt_flip.mean(axis=0)
    xy_range = np.max(np.abs(opt_flip - center)) * 1.3
    z_half = 0.3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    draw_skeleton(ax, opt, OPT_COLOR)

    ax.set_xlim(center[0] - xy_range, center[0] + xy_range)
    ax.set_ylim(center[1] - xy_range, center[1] + xy_range)
    ax.set_zlim(center[2] - xy_range, center[2] + xy_range)
    ax.set_box_aspect([1, 1, 1])

    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")

    def on_key(event):
        if event.key == "p":
            print(f"\n--- Copy this back to Claude ---")
            print(f"elev={ax.elev:.2f}")
            print(f"azim={ax.azim:.2f}")
            print(f"roll={getattr(ax, 'roll', 0):.2f}")
            print(f"--------------------------------\n")

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Rotate the view, then press 'p' to print camera angles.")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
