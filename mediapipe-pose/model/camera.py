import numpy as np
import torch


class Camera:
    """Perspective camera with known intrinsics.

    Projects 3D camera-space points to 2D image coordinates:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    Z gradient flows through the division, enabling depth optimization.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_size: tuple[int, int],
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.image_size = image_size
        self._fx_t = torch.tensor(fx, dtype=torch.float32)
        self._fy_t = torch.tensor(fy, dtype=torch.float32)
        self._cx_t = torch.tensor(cx, dtype=torch.float32)
        self._cy_t = torch.tensor(cy, dtype=torch.float32)

    def world_to_image(self, point: np.ndarray | torch.Tensor) -> np.ndarray:
        """Project 3D point(s) to 2D image coordinates.

        Args:
            point: (3,) or (N, 3) in camera coordinates (Z > 0 is forward)
        """
        if isinstance(point, torch.Tensor):
            point = point.detach().numpy()
        point = np.asarray(point, dtype=np.float64)
        single = point.ndim == 1

        if single:
            point = point.reshape(1, 3)

        X, Y, Z = point[:, 0], point[:, 1], point[:, 2]
        Z = np.maximum(Z, 0.01)  # avoid division by zero
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
        result = np.stack([u, v], axis=-1)

        if single:
            return result[0]
        return result

    def world_to_image_torch(self, points: torch.Tensor) -> torch.Tensor:
        """Differentiable 3D→2D projection.

        Args:
            points: (3,) single point or (N, 3) batch of points.

        Returns:
            (2,) or (N, 2) pixel coordinates.
        """
        if points.dim() == 1:
            X, Y, Z = points[0], points[1], points[2]
            Z = torch.clamp(Z, min=0.01)
            u = self._fx_t * X / Z + self._cx_t
            v = self._fy_t * Y / Z + self._cy_t
            return torch.stack([u, v])

        X = points[:, 0]
        Y = points[:, 1]
        Z = torch.clamp(points[:, 2], min=0.01)
        u = self._fx_t * X / Z + self._cx_t
        v = self._fy_t * Y / Z + self._cy_t
        return torch.stack([u, v], dim=-1)

    def reprojection_error(self, pts_2d: np.ndarray, pts_3d: np.ndarray) -> float:
        """Mean reprojection error in pixels."""
        reproj = self.world_to_image(pts_3d)
        return float(np.mean(np.linalg.norm(reproj - pts_2d, axis=1)))

    @classmethod
    def from_focal_length(
        cls,
        focal_length: float,
        image_size: tuple[int, int],
    ) -> "Camera":
        """Create camera with known focal length, principal point at image center.

        Args:
            focal_length: focal length in pixels (fx = fy)
            image_size: (height, width)
        """
        h, w = image_size
        return cls(
            fx=focal_length,
            fy=focal_length,
            cx=w / 2.0,
            cy=h / 2.0,
            image_size=image_size,
        )

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            image_size=tuple(data["image_size"]),
        )
