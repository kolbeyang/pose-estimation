"""Unified pinhole camera model with numpy and torch projection.

Stores both intrinsics (K) and optional extrinsics (R, t) so a single Camera
object can handle world_to_camera, camera_to_world, camera_to_image, and
is_in_frame checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class Camera:
    """Perspective camera with known intrinsics and optional extrinsics.

    Intrinsic projection (camera space -> image):
        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    Extrinsic transform (world -> camera):
        p_cam = R @ p_world + t
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_size: tuple[int, int],
        R: np.ndarray | None = None,
        t: np.ndarray | None = None,
    ) -> None:
        """Initialize camera.

        Args:
            fx: Focal length in pixels (x).
            fy: Focal length in pixels (y).
            cx: Principal point x.
            cy: Principal point y.
            image_size: (height, width) of the image.
            R: (3, 3) rotation matrix (world -> camera). Optional.
            t: (3, 1) translation vector. Optional.
        """
        self.fx: float = fx
        self.fy: float = fy
        self.cx: float = cx
        self.cy: float = cy
        self.image_size: tuple[int, int] = image_size
        self.R: np.ndarray | None = R
        self.t: np.ndarray | None = t

        # Cached torch tensors for differentiable projection
        self._fx_t: torch.Tensor = torch.tensor(fx, dtype=torch.float32)
        self._fy_t: torch.Tensor = torch.tensor(fy, dtype=torch.float32)
        self._cx_t: torch.Tensor = torch.tensor(cx, dtype=torch.float32)
        self._cy_t: torch.Tensor = torch.tensor(cy, dtype=torch.float32)

    # ----- Extrinsic transforms -----

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """Transform world coordinates to camera coordinates.

        In the pipeline, used on [3D:SKELETON_16] and [3D:COCO19] data,
        but the function is generic for any (..., 3) points.

        Args:
            points_world: (..., 3) positions in world frame.

        Returns:
            (..., 3) positions in camera frame.
        """
        if self.R is None or self.t is None:
            raise ValueError("Camera extrinsics (R, t) not set.")
        shape = points_world.shape
        pts = points_world.reshape(-1, 3)
        pts_cam = (self.R @ pts.T + self.t).T  # (N, 3)
        return pts_cam.reshape(shape)

    def camera_to_world(self, points_cam: np.ndarray) -> np.ndarray:
        """Transform camera coordinates to world coordinates.

        Inverse of world_to_camera: p_world = R^T @ (p_cam - t)

        Args:
            points_cam: (..., 3) positions in camera frame.

        Returns:
            (..., 3) positions in world frame.
        """
        if self.R is None or self.t is None:
            raise ValueError("Camera extrinsics (R, t) not set.")
        shape = points_cam.shape
        pts = points_cam.reshape(-1, 3)
        pts_world = (self.R.T @ (pts.T - self.t)).T  # (N, 3)
        return pts_world.reshape(shape)

    # ----- Intrinsic projection (camera space -> image) -----

    def camera_to_image(self, points_cam: np.ndarray) -> np.ndarray:
        """Project 3D camera-space point(s) to 2D image coordinates (numpy).

        In the pipeline, projects [3D:SKELETON_16] -> [2D:SKELETON_16],
        but the function is generic for any (..., 3) points.

        Args:
            points_cam: (3,) or (..., 3) in camera coordinates (Z > 0 forward).

        Returns:
            Same leading dims + (2,) pixel coordinates.
        """
        if isinstance(points_cam, torch.Tensor):
            points_cam = points_cam.detach().numpy()
        points_cam = np.asarray(points_cam, dtype=np.float64)
        single = points_cam.ndim == 1

        if single:
            points_cam = points_cam.reshape(1, 3)

        shape = points_cam.shape
        pts = points_cam.reshape(-1, 3)
        X = pts[:, 0]
        Y = pts[:, 1]
        Z = np.maximum(pts[:, 2], 0.01)
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
        result = np.stack([u, v], axis=-1)
        result = result.reshape(shape[:-1] + (2,))

        if single:
            return result[0]
        return result

    def camera_to_image_torch(self, points_cam: torch.Tensor) -> torch.Tensor:
        """Differentiable 3D camera-space -> 2D projection.

        In the optimization loop, projects [3D:SKELETON_16] -> [2D:SKELETON_16],
        but the function is generic for any (..., 3) tensors.

        Args:
            points_cam: (3,) or (..., 3) batch of points.

        Returns:
            Same leading dims + (2,) pixel coordinates.
        """
        if points_cam.dim() == 1:
            X = points_cam[0]
            Y = points_cam[1]
            Z = torch.clamp(points_cam[2], min=0.01)
            u = self._fx_t * X / Z + self._cx_t
            v = self._fy_t * Y / Z + self._cy_t
            return torch.stack([u, v])

        shape = points_cam.shape
        pts = points_cam.reshape(-1, 3)
        X = pts[:, 0]
        Y = pts[:, 1]
        Z = torch.clamp(pts[:, 2], min=0.01)
        u = self._fx_t * X / Z + self._cx_t
        v = self._fy_t * Y / Z + self._cy_t
        result = torch.stack([u, v], dim=-1)
        return result.reshape(shape[:-1] + (2,))

    # ----- Frame boundary checks -----

    def is_in_frame(self, points_2d: np.ndarray) -> np.ndarray:
        """Check which 2D points are within image bounds.

        Args:
            points_2d: (..., 2) pixel coordinates.

        Returns:
            (...) boolean array. True if point is inside [0, width) x [0, height).
        """
        points_2d = np.asarray(points_2d, dtype=np.float64)
        h, w = self.image_size
        u = points_2d[..., 0]
        v = points_2d[..., 1]
        return (u >= 0) & (u < w) & (v >= 0) & (v < h)

    # ----- Serialization -----

    def to_dict(self) -> dict[str, Any]:
        """Serialize camera parameters."""
        d: dict[str, Any] = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "image_size": list(self.image_size),
        }
        if self.R is not None:
            d["R"] = self.R.tolist()
        if self.t is not None:
            d["t"] = self.t.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Camera:
        """Deserialize camera parameters."""
        R = np.array(data["R"], dtype=np.float64) if "R" in data else None
        t = np.array(data["t"], dtype=np.float64) if "t" in data else None
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            image_size=tuple(data["image_size"]),
            R=R,
            t=t,
        )

    @classmethod
    def from_panoptic_calibration(
        cls,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        resolution: tuple[int, int],
    ) -> Camera:
        """Create camera from Panoptic calibration data.

        Args:
            K: (3, 3) intrinsic matrix.
            R: (3, 3) rotation matrix (world -> camera).
            t: (3, 1) translation vector.
            resolution: (width, height) from Panoptic calibration.

        Returns:
            Camera instance with both intrinsics and extrinsics.
        """
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        # Panoptic resolution is (width, height), image_size is (height, width)
        image_size = (int(resolution[1]), int(resolution[0]))
        return cls(
            fx=fx, fy=fy, cx=cx, cy=cy,
            image_size=image_size,
            R=np.array(R, dtype=np.float64).reshape(3, 3),
            t=np.array(t, dtype=np.float64).reshape(3, 1),
        )
