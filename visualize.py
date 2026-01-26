import argparse
import json
import sys

import numpy as np
from vpython import canvas, box, sphere, curve, vector, color, rate

from model.arm import Arm
from model.environment import Environment


def numpy_to_vpython(arr: np.ndarray) -> vector:
    return vector(float(arr[0]), float(arr[1]), float(arr[2]))


# Color palette for multiple arms (5 colors)
ARM_COLORS = [
    color.green,  # Arm 0
    color.red,  # Arm 1
    color.blue,  # Arm 2
    color.orange,  # Arm 3
    color.purple,  # Arm 4
]


class Visualizer:
    def __init__(self, env: Environment, coords_list: list[dict], camera=None, fps: int = 12):
        """
        Args:
            env: Environment object defining the scene bounds
            coords_list: List of coordinate dicts, each with keys "a", "b", "c"
            camera: Optional camera object for visualization
            fps: Frames per second for animation (default: 12)
        """
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
            camera_pos = numpy_to_vpython(camera.camera_position)
            sphere(pos=camera_pos, radius=0.3, color=color.red, opacity=0.2)

        # Create visualization objects for each arm
        self.arms = []  # List of (sphere_a, sphere_b, sphere_c, curve) tuples

        for i, coords in enumerate(coords_list):
            arm_color = ARM_COLORS[i % len(ARM_COLORS)]

            sphere_a = sphere(
                pos=numpy_to_vpython(coords["a"]), radius=0.4, color=arm_color
            )
            sphere_b = sphere(
                pos=numpy_to_vpython(coords["b"]), radius=0.3, color=arm_color
            )
            sphere_c = sphere(
                pos=numpy_to_vpython(coords["c"]), radius=0.2, color=arm_color
            )
            arm_curve = curve(
                pos=[
                    numpy_to_vpython(coords["a"]),
                    numpy_to_vpython(coords["b"]),
                    numpy_to_vpython(coords["c"]),
                ],
                radius=0.05,
                color=arm_color,
            )
            self.arms.append((sphere_a, sphere_b, sphere_c, arm_curve))

    def update(self, coords_list: list[dict]):
        """Update positions for all arms."""
        for i, coords in enumerate(coords_list):
            if i >= len(self.arms):
                break  # Skip if more coords than initialized arms

            sphere_a, sphere_b, sphere_c, arm_curve = self.arms[i]
            sphere_a.pos = numpy_to_vpython(coords["a"])
            sphere_b.pos = numpy_to_vpython(coords["b"])
            sphere_c.pos = numpy_to_vpython(coords["c"])
            arm_curve.clear()
            arm_curve.append(numpy_to_vpython(coords["a"]))
            arm_curve.append(numpy_to_vpython(coords["b"]))
            arm_curve.append(numpy_to_vpython(coords["c"]))

    def run(self):
        while True:
            rate(self.fps)


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
