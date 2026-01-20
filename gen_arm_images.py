import argparse
import json
from datetime import datetime
import numpy as np
from PIL import Image

from arm import Arm
from environment import Environment


def main():
    parser = argparse.ArgumentParser(description="Generate random arm images")
    parser.add_argument("--n", type=int, default=1, help="Number of images to generate")
    args = parser.parse_args()

    # Camera rotation: camera at [8,0,0] looking toward origin
    camera_rotation = np.array([
        [0, 1, 0],   # camera X = world Y
        [0, 0, -1],  # camera Y = world -Z
        [-1, 0, 0],  # camera Z = world -X
    ])

    env = Environment(
        cube_size=10.0,
        camera_position=np.array([8.0, 0.0, 0.0]),
        camera_rotation=camera_rotation,
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        image_size=(100, 100),
    )

    # Generate a random arm within bounds
    min_bounds = np.array([-3, -3, -3])
    max_bounds = np.array([3, 3, 3])

    for i in range(args.n):
        arm = Arm.random(min_bounds, max_bounds, a_b_length=2, b_c_length=2)
        coords = arm.get_coordinates()

        # Get world coordinates
        world_points = np.array([coords["a"], coords["b"], coords["c"]])

        # Generate and save the image
        image = env.snap_a_photo(world_points)

        # Save using PIL
        img = Image.fromarray(image)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_path = f"/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/images/arm-{timestamp}"

        img.save(f"{base_path}.png")

        # Save arm parameters as JSON
        params = {
            "a_pos": arm.a_pos.tolist(),
            "a_b_length": arm.a_b_length,
            "a_b_polar": list(arm.a_b_polar),
            "b_c_length": arm.b_c_length,
            "b_c_polar": list(arm.b_c_polar),
        }
        with open(f"{base_path}.json", "w") as f:
            json.dump(params, f, indent=2)

        print(f"Saved image {i+1}/{args.n}: {base_path}.png")


if __name__ == "__main__":
    main()
