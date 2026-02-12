import argparse
import json
import sys

import numpy as np
import torch
from vpython import canvas, box, sphere, cylinder, vector, color, rate, checkbox

from model.arm import Arm
from model.camera import Camera
from model.environment import Environment


def to_vpython(arr: np.ndarray | torch.Tensor) -> vector:
    """Convert numpy array or torch tensor to vpython vector."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()
    return vector(float(arr[0]), float(arr[1]), float(arr[2]))


# Color palette for multiple arms (6 colors)
ARM_COLORS = [
    color.green,   # Arm 0 - Ground truth
    color.yellow,  # Arm 1 - Initialization
    color.red,     # Arm 2 - Predicted
    color.blue,    # Arm 3
    color.orange,  # Arm 4
    color.purple,  # Arm 5
]

# Labels for arms (used in toggle checkboxes)
ARM_LABELS = [
    "Ground Truth",
    "Initialization",
    "Predicted",
    "Arm 3",
    "Arm 4",
    "Arm 5",
]


class Visualizer:
    FPS = 12  # Default frames per second

    def __init__(self, env: Environment, coords_list: list[dict], camera: Camera | None = None, fps: int = 12,
                 opacities: list[float] | None = None, labels: list[str] | None = None):
        """
        Args:
            env: Environment object defining the scene bounds
            coords_list: List of coordinate dicts, each with keys "a", "b", "c"
            camera: Optional Camera object for visualization
            fps: Frames per second for animation (default: 12)
            opacities: Per-arm opacity values (default: all 1.0)
            labels: Per-arm labels for toggle checkboxes (default: ARM_LABELS)
        """
        if opacities is not None:
            assert len(opacities) == len(coords_list), \
                f"opacities length ({len(opacities)}) must match coords_list length ({len(coords_list)})"
        if labels is not None:
            assert len(labels) == len(coords_list), \
                f"labels length ({len(labels)}) must match coords_list length ({len(coords_list)})"
        self.fps = fps
        self.scene = canvas(
            title="Arm Visualization",
            width=800,
            height=600,
            center=vector(0, 0, 0),
            background=color.gray(0.2),
        )
        self.scene.up = vector(0, 0, 1)
        self.scene.forward = vector(-1, -0.5, -1)
        self.scene.range = 12

        # Environment cube
        box(
            pos=vector(0, 0, 0),
            size=vector(env.cube_size, env.cube_size, env.cube_size),
            color=color.white,
            opacity=0.1,
        )

        # Camera position (optional)
        if camera is not None:
            camera_pos = to_vpython(camera.position)
            sphere(pos=camera_pos, radius=0.3, color=color.red, opacity=0.2)

        # Create visualization objects for each arm
        self.arms = []  # List of (sphere_a, sphere_b, sphere_c, cyl_ab, cyl_bc) tuples
        self.arm_visible = []  # Track visibility state for each arm

        for i, coords in enumerate(coords_list):
            arm_color = ARM_COLORS[i % len(ARM_COLORS)]
            op = opacities[i] if opacities is not None else 1.0

            a_pos = to_vpython(coords["a"])
            b_pos = to_vpython(coords["b"])
            c_pos = to_vpython(coords["c"])

            sphere_a = sphere(
                pos=a_pos, radius=0.4, color=arm_color, opacity=op
            )
            sphere_b = sphere(
                pos=b_pos, radius=0.3, color=arm_color, opacity=op
            )
            sphere_c = sphere(
                pos=c_pos, radius=0.2, color=arm_color, opacity=op
            )
            cyl_ab = cylinder(
                pos=a_pos, axis=b_pos - a_pos,
                radius=0.05, color=arm_color, opacity=op,
            )
            cyl_bc = cylinder(
                pos=b_pos, axis=c_pos - b_pos,
                radius=0.05, color=arm_color, opacity=op,
            )
            self.arms.append((sphere_a, sphere_b, sphere_c, cyl_ab, cyl_bc))
            self.arm_visible.append(True)

        # Create toggle checkboxes for each arm
        self.scene.append_to_caption("\n\nToggle Arms:\n")
        for i in range(len(coords_list)):
            if labels is not None:
                arm_label = labels[i]
            elif i < len(ARM_LABELS):
                arm_label = ARM_LABELS[i]
            else:
                arm_label = f"Arm {i}"
            checkbox(bind=self._make_toggle_handler(i), text=arm_label, checked=True)
            self.scene.append_to_caption("  ")

    def _make_toggle_handler(self, arm_index: int):
        """Create a toggle handler for a specific arm index."""
        def handler(evt):
            self._set_arm_visible(arm_index, evt.checked)
        return handler

    def _set_arm_visible(self, arm_index: int, visible: bool):
        """Set visibility for a specific arm."""
        if arm_index >= len(self.arms):
            return
        self.arm_visible[arm_index] = visible
        sphere_a, sphere_b, sphere_c, cyl_ab, cyl_bc = self.arms[arm_index]
        sphere_a.visible = visible
        sphere_b.visible = visible
        sphere_c.visible = visible
        cyl_ab.visible = visible
        cyl_bc.visible = visible

    def update(self, coords_list: list[dict]):
        """Update positions for all arms (only updates visible arms)."""
        for i, coords in enumerate(coords_list):
            if i >= len(self.arms):
                break  # Skip if more coords than initialized arms

            # Skip update if arm is hidden (optimization)
            if not self.arm_visible[i]:
                continue

            sphere_a, sphere_b, sphere_c, cyl_ab, cyl_bc = self.arms[i]
            a_pos = to_vpython(coords["a"])
            b_pos = to_vpython(coords["b"])
            c_pos = to_vpython(coords["c"])
            sphere_a.pos = a_pos
            sphere_b.pos = b_pos
            sphere_c.pos = c_pos
            cyl_ab.pos = a_pos
            cyl_ab.axis = b_pos - a_pos
            cyl_bc.pos = b_pos
            cyl_bc.axis = c_pos - b_pos

    def run(self):
        try:
            while True:
                rate(self.fps)
        except KeyboardInterrupt:
            import os
            os._exit(0)


def load_arm_from_json(filepath: str) -> Arm:
    with open(filepath, "r") as f:
        data = json.load(f)

    required_keys = ["a_pos", "a_b_length", "a_b_polar", "b_c_length", "b_c_theta"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if len(data["a_pos"]) != 3:
        raise ValueError("a_pos must have 3 values (x, y, z)")
    if len(data["a_b_polar"]) != 3:
        raise ValueError("a_b_polar must have 3 values (azimuth, elevation, roll)")
    if not isinstance(data["b_c_theta"], (int, float)):
        raise ValueError("b_c_theta must be a single numeric value")

    return Arm(
        a_pos=np.array(data["a_pos"]),
        a_b_length=data["a_b_length"],
        a_b_polar=tuple(data["a_b_polar"]),
        b_c_length=data["b_c_length"],
        b_c_theta=data["b_c_theta"],
    )


def visualize(env: Environment, coords_list: list[dict]):
    vis = Visualizer(env, coords_list)
    vis.run()


def main():
    parser = argparse.ArgumentParser(description="Visualize an arm from a JSON file")
    parser.add_argument("filepath", help="Path to the arm JSON file")
    args = parser.parse_args()

    try:
        arm = load_arm_from_json(args.filepath)
    except FileNotFoundError:
        print(f"Error: File not found: {args.filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid arm data: {e}", file=sys.stderr)
        sys.exit(1)

    coords = arm.get_coordinates()

    env = Environment(cube_size=10.0)

    print(f"Loaded arm from {args.filepath}")
    print(f"Point A: {coords['a']}")
    print(f"Point B: {coords['b']}")
    print(f"Point C: {coords['c']}")

    visualize(env, [coords])


if __name__ == "__main__":
    main()
