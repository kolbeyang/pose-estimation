"""Pinhole camera model with numpy and torch projection."""

from typing import Any

import numpy as np
import torch


class Camera:
    """Perspective camera with known intrinsics.

    Projects 3D camera-space points to 2D image coordinates:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_size: tuple[int, int],
    ) -> None:
        """Initialize camera.

        Args:
            fx: Focal length in pixels (x).
            fy: Focal length in pixels (y).
            cx: Principal point x.
            cy: Principal point y.
            image_size: (height, width) of the image.
        """
        self.fx: float = fx
        self.fy: float = fy
        self.cx: float = cx
        self.cy: float = cy
        self.image_size: tuple[int, int] = image_size
        self._fx_t: torch.Tensor = torch.tensor(fx, dtype=torch.float32)
        self._fy_t: torch.Tensor = torch.tensor(fy, dtype=torch.float32)
        self._cx_t: torch.Tensor = torch.tensor(cx, dtype=torch.float32)
        self._cy_t: torch.Tensor = torch.tensor(cy, dtype=torch.float32)

    def world_to_image(self, point: np.ndarray) -> np.ndarray:
        """Project 3D point(s) to 2D image coordinates (numpy).

        Args:
            point: (3,) or (N, 3) in camera coordinates (Z > 0 is forward).

        Returns:
            (2,) or (N, 2) pixel coordinates.
        """
        if isinstance(point, torch.Tensor):
            point = point.detach().numpy()
        point = np.asarray(point, dtype=np.float64)
        single: bool = point.ndim == 1

        if single:
            point = point.reshape(1, 3)

        X: np.ndarray = point[:, 0]
        Y: np.ndarray = point[:, 1]
        Z: np.ndarray = np.maximum(point[:, 2], 0.01)
        u: np.ndarray = self.fx * X / Z + self.cx
        v: np.ndarray = self.fy * Y / Z + self.cy
        result: np.ndarray = np.stack([u, v], axis=-1)

        if single:
            return result[0]
        return result

    def world_to_image_torch(self, points: torch.Tensor) -> torch.Tensor:
        """Differentiable 3D->2D projection.

        Args:
            points: (3,) single point or (N, 3) batch of points.

        Returns:
            (2,) or (N, 2) pixel coordinates.
        """
        if points.dim() == 1:
            X: torch.Tensor = points[0]
            Y: torch.Tensor = points[1]
            Z: torch.Tensor = torch.clamp(points[2], min=0.01)
            u: torch.Tensor = self._fx_t * X / Z + self._cx_t
            v: torch.Tensor = self._fy_t * Y / Z + self._cy_t
            return torch.stack([u, v])

        X = points[:, 0]
        Y = points[:, 1]
        Z = torch.clamp(points[:, 2], min=0.01)
        u = self._fx_t * X / Z + self._cx_t
        v = self._fy_t * Y / Z + self._cy_t
        return torch.stack([u, v], dim=-1)

    def to_dict(self) -> dict[str, float | list[int]]:
        """Serialize camera parameters."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Camera":
        """Deserialize camera parameters."""
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            image_size=tuple(data["image_size"]),
        )

    @classmethod
    def from_panoptic_calibration(
        cls,
        K: np.ndarray,
        resolution: tuple[int, int],
    ) -> "Camera":
        """Create camera from Panoptic calibration data.

        Args:
            K: (3, 3) intrinsic matrix.
            resolution: (width, height) from Panoptic calibration.

        Returns:
            Camera instance.
        """
        fx: float = float(K[0, 0])
        fy: float = float(K[1, 1])
        cx: float = float(K[0, 2])
        cy: float = float(K[1, 2])
        # Panoptic resolution is (width, height), image_size is (height, width)
        image_size: tuple[int, int] = (int(resolution[1]), int(resolution[0]))
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, image_size=image_size)
