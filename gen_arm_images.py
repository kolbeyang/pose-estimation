import argparse
import json
import os
from datetime import datetime
import numpy as np
from PIL import Image

from arm import Arm
from environment import Environment

IMAGES_DIR = "/Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/images"


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

    min_bounds = np.array([-1, -1, -1])
    max_bounds = np.array([1, 1, 1])
    height, width = env.image_size

    for i in range(args.n):
        arm = Arm.random(min_bounds, max_bounds, a_b_length=2, b_c_length=2)
        coords = arm.get_coordinates()

        # Create output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder_path = os.path.join(IMAGES_DIR, f"arm-{timestamp}")
        os.makedirs(folder_path, exist_ok=True)

        # Project each point to image coordinates
        points = {"a": coords["a"], "b": coords["b"], "c": coords["c"]}
        images = {}

        for name, world_point in points.items():
            image_point = env.world_to_image(world_point)
            x, y = np.round(image_point).astype(int)

            img_array = np.zeros((height, width), dtype=np.uint8)
            if 0 <= x < width and 0 <= y < height:
                img_array[y, x] = 255

            images[name] = img_array
            Image.fromarray(img_array).save(os.path.join(folder_path, f"{name}.png"))

        # Create sum image
        sum_array = np.clip(images["a"] + images["b"] + images["c"], 0, 255).astype(np.uint8)
        Image.fromarray(sum_array).save(os.path.join(folder_path, "sum.png"))

        # Save arm and camera parameters as JSON
        params = {
            "a_pos": arm.a_pos.tolist(),
            "a_b_length": arm.a_b_length,
            "a_b_polar": list(arm.a_b_polar),
            "b_c_length": arm.b_c_length,
            "b_c_polar": list(arm.b_c_polar),
            "camera_position": env.camera_position.tolist(),
            "camera_rotation": env.camera_rotation.tolist(),
        }
        with open(os.path.join(folder_path, "data.json"), "w") as f:
            json.dump(params, f, indent=2)

        print(f"Saved {i+1}/{args.n}: {folder_path}")


if __name__ == "__main__":
    main()
