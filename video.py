import glob
import json
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from model.arm import Arm
from model.camera import Camera

# Heatmap constants
HEATMAP_POINT_STD = 3.0  # pixels (standard deviation for Gaussian blobs)
LOCALIZATION_NOISE_STD = 1  # pixels (random offset to shift heatmap peak)


class Video:
    def __init__(
        self,
        camera_position: np.ndarray,
        camera_rotation: np.ndarray,
        focal_length: tuple[float, float],
        principal_point: tuple[float, float],
        image_size: tuple[int, int],
        a_b_length: float,
        b_c_length: float,
        output_dir: str = "images",
    ):
        # Create Camera object
        self._camera = Camera(
            position=camera_position,
            rotation=camera_rotation,
            focal_length=focal_length,
            principal_point=principal_point,
            image_size=image_size,
        )
        self.a_b_length = a_b_length
        self.b_c_length = b_c_length

        self.frame_count = 0

        # Create timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.video_dir = os.path.join(output_dir, f"arm-video-{timestamp}")
        self.frames_dir = os.path.join(self.video_dir, "frames")
        self.frames_a_dir = os.path.join(self.video_dir, "frames-a")
        self.frames_b_dir = os.path.join(self.video_dir, "frames-b")
        self.frames_c_dir = os.path.join(self.video_dir, "frames-c")
        self.frames_data_dir = os.path.join(self.video_dir, "frames-data")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.frames_a_dir, exist_ok=True)
        os.makedirs(self.frames_b_dir, exist_ok=True)
        os.makedirs(self.frames_c_dir, exist_ok=True)
        os.makedirs(self.frames_data_dir, exist_ok=True)

        # Save simulation configuration (using Camera.to_dict for camera params)
        config = self._camera.to_dict()
        config["a_b_length"] = a_b_length
        config["b_c_length"] = b_c_length
        with open(
            os.path.join(self.video_dir, "simulation-configuration.json"), "w"
        ) as f:
            json.dump(config, f, indent=2)

    @property
    def camera(self) -> Camera:
        """Return the Camera object."""
        return self._camera

    @property
    def camera_position(self) -> np.ndarray:
        """Backward-compatible access to camera position."""
        return self._camera.position

    @property
    def camera_rotation(self) -> np.ndarray:
        """Backward-compatible access to camera rotation."""
        return self._camera.rotation

    @property
    def focal_length(self) -> tuple[float, float]:
        """Backward-compatible access to focal length."""
        return self._camera.focal_length

    @property
    def principal_point(self) -> tuple[float, float]:
        """Backward-compatible access to principal point."""
        return self._camera.principal_point

    @property
    def image_size(self) -> tuple[int, int]:
        """Backward-compatible access to image size."""
        return self._camera.image_size

    @property
    def K(self) -> np.ndarray:
        """Backward-compatible access to intrinsic matrix."""
        return self._camera.K

    @classmethod
    def load(cls, folder_path: str) -> "Video":
        """Load a saved video from disk."""
        # Load simulation configuration
        config_path = os.path.join(folder_path, "simulation-configuration.json")
        with open(config_path) as f:
            config = json.load(f)

        # Create a new instance without initializing directories
        video = object.__new__(cls)

        # Create Camera from config
        video._camera = Camera.from_dict(config)
        video.a_b_length = config["a_b_length"]
        video.b_c_length = config["b_c_length"]

        # Set directory paths
        video.video_dir = folder_path
        video.frames_dir = os.path.join(folder_path, "frames")
        video.frames_a_dir = os.path.join(folder_path, "frames-a")
        video.frames_b_dir = os.path.join(folder_path, "frames-b")
        video.frames_c_dir = os.path.join(folder_path, "frames-c")
        video.frames_data_dir = os.path.join(folder_path, "frames-data")

        # Count existing frames
        video.frame_count = video.get_frame_count()

        return video

    def get_frame_count(self) -> int:
        """Count frames in the frames directory."""
        return len(glob.glob(os.path.join(self.frames_dir, "frame-*.png")))

    def get_frame_heatmaps(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Load heatmap images for a specific frame."""
        frame_name = f"frame-{frame_idx:02d}"
        return {
            "a": np.array(Image.open(os.path.join(self.frames_a_dir, f"{frame_name}.png"))),
            "b": np.array(Image.open(os.path.join(self.frames_b_dir, f"{frame_name}.png"))),
            "c": np.array(Image.open(os.path.join(self.frames_c_dir, f"{frame_name}.png"))),
        }

    def get_frame_arm(self, frame_idx: int) -> Arm:
        """Load arm configuration from frame JSON."""
        frame_name = f"frame-{frame_idx:02d}"
        with open(os.path.join(self.frames_data_dir, f"{frame_name}.json")) as f:
            data = json.load(f)

        return Arm(
            a_pos=np.array(data["a_pos"]),
            a_b_length=self.a_b_length,
            a_b_polar=tuple(data["a_b_polar"]),
            b_c_length=self.b_c_length,
            b_c_theta=data["b_c_theta"],
        )

    def world_to_image(self, point: np.ndarray | torch.Tensor) -> np.ndarray:
        """Delegate to Camera.world_to_image."""
        return self._camera.world_to_image(point)

    def _generate_heatmap(
        self, x: float, y: float, height: int, width: int
    ) -> np.ndarray:
        """Generate a 2D Gaussian heatmap centered at (x, y)."""
        # Create coordinate grids
        yy, xx = np.mgrid[0:height, 0:width]

        # Compute Gaussian
        gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * HEATMAP_POINT_STD**2))

        # Normalize to [0, 255]
        return (gaussian * 255).astype(np.uint8)

    def update(
        self,
        coords: dict,
        a_pos: np.ndarray,
        a_b_polar: tuple,
        b_c_theta: float,
    ):
        height, width = self.image_size

        # Project each point and create heatmap images
        images = {}
        for name in ["a", "b", "c"]:
            world_point = coords[name]
            image_point = self.world_to_image(world_point)
            x, y = image_point

            # Add localization noise to shift peak
            x += np.random.normal(0, LOCALIZATION_NOISE_STD)
            y += np.random.normal(0, LOCALIZATION_NOISE_STD)

            images[name] = self._generate_heatmap(x, y, height, width)

        # Create sum image (clip to prevent overflow)
        sum_array = np.clip(
            images["a"].astype(np.float32)
            + images["b"].astype(np.float32)
            + images["c"].astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        # Save frame images
        frame_name = f"frame-{self.frame_count:02d}"
        Image.fromarray(sum_array).save(
            os.path.join(self.frames_dir, f"{frame_name}.png")
        )
        Image.fromarray(images["a"]).save(
            os.path.join(self.frames_a_dir, f"{frame_name}.png")
        )
        Image.fromarray(images["b"]).save(
            os.path.join(self.frames_b_dir, f"{frame_name}.png")
        )
        Image.fromarray(images["c"]).save(
            os.path.join(self.frames_c_dir, f"{frame_name}.png")
        )

        # Save frame JSON
        frame_data = {
            "a_pos": a_pos.tolist() if isinstance(a_pos, np.ndarray) else list(a_pos),
            "a_b_polar": list(a_b_polar),
            "b_c_theta": b_c_theta,
        }
        with open(os.path.join(self.frames_data_dir, f"{frame_name}.json"), "w") as f:
            json.dump(frame_data, f, indent=2)

        self.frame_count += 1
