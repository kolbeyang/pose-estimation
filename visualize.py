import argparse
import json
import sys

import numpy as np
from vpython import canvas, box, sphere, curve, vector, color, rate

from arm import Arm
from environment import Environment


def numpy_to_vpython(arr: np.ndarray) -> vector:
    return vector(float(arr[0]), float(arr[1]), float(arr[2]))


def load_arm_from_json(filepath: str) -> Arm:
    with open(filepath, "r") as f:
        data = json.load(f)

    required_keys = ["a_pos", "a_b_length", "a_b_polar", "b_c_length", "b_c_polar"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key}")

    if len(data["a_pos"]) != 3:
        raise ValueError("a_pos must have 3 values (x, y, z)")
    if len(data["a_b_polar"]) != 2:
        raise ValueError("a_b_polar must have 2 values (azimuth, elevation)")
    if len(data["b_c_polar"]) != 2:
        raise ValueError("b_c_polar must have 2 values (azimuth, elevation)")

    return Arm(
        a_pos=np.array(data["a_pos"]),
        a_b_length=data["a_b_length"],
        a_b_polar=tuple(data["a_b_polar"]),
        b_c_length=data["b_c_length"],
        b_c_polar=tuple(data["b_c_polar"]),
    )


def visualize(env: Environment, coords: dict):
    scene = canvas(
        title="Arm Visualization",
        width=800,
        height=600,
        center=vector(0, 0, 0),
        background=color.gray(0.2),
    )
    scene.up = vector(0, 0, 1)
    scene.forward = vector(-1, -0.5, -1)
    scene.range = 12

    # Environment cube
    box(
        pos=vector(0, 0, 0),
        size=vector(env.cube_size, env.cube_size, env.cube_size),
        color=color.white,
        opacity=0.1,
    )

    # Camera position
    camera_pos = numpy_to_vpython(env.camera_position)
    sphere(pos=camera_pos, radius=0.3, color=color.red)

    # Arm points
    a_pos = numpy_to_vpython(coords["a"])
    b_pos = numpy_to_vpython(coords["b"])
    c_pos = numpy_to_vpython(coords["c"])

    sphere(pos=a_pos, radius=0.4, color=color.green)
    sphere(pos=b_pos, radius=0.3, color=color.yellow)
    sphere(pos=c_pos, radius=0.2, color=color.cyan)
    curve(pos=[a_pos, b_pos, c_pos], radius=0.05, color=color.white)

    # Keep running
    while True:
        rate(30)


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

    # Camera rotation: camera at [8,0,0] looking toward origin
    camera_rotation = np.array([
        [0, 1, 0],
        [0, 0, -1],
        [-1, 0, 0],
    ])

    env = Environment(
        cube_size=10.0,
        camera_position=np.array([8.0, 0.0, 0.0]),
        camera_rotation=camera_rotation,
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        image_size=(100, 100),
    )

    print(f"Loaded arm from {args.filepath}")
    print(f"Point A: {coords['a']}")
    print(f"Point B: {coords['b']}")
    print(f"Point C: {coords['c']}")

    visualize(env, coords)


if __name__ == "__main__":
    main()
