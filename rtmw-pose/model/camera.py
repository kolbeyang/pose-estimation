import cv2
import numpy as np
import torch


class Camera:
    def __init__(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        focal_length: tuple[float, float],
        principal_point: tuple[float, float],
        image_size: tuple[int, int],
    ):
        """
        Camera model with intrinsic and extrinsic parameters.

        Args:
            position: Camera position in world coordinates (3,)
            rotation: Camera rotation matrix (3, 3)
            focal_length: (fx, fy) focal lengths in pixels
            principal_point: (cx, cy) principal point in pixels
            image_size: (height, width) of the image
        """
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation = np.asarray(rotation, dtype=np.float64)
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size
        self.K = self._build_intrinsic_matrix()

    def _build_intrinsic_matrix(self) -> np.ndarray:
        """Build the 3x3 camera intrinsic matrix."""
        fx, fy = self.focal_length
        cx, cy = self.principal_point
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def world_to_image_torch(self, point: torch.Tensor) -> torch.Tensor:
        """
        Project 3D world point to 2D image coordinates (differentiable).

        Args:
            point: Single point (3,) as torch.Tensor

        Returns:
            Image coordinates (2,) as torch.Tensor
        """
        position = torch.tensor(self.position, dtype=torch.float32)
        rotation = torch.tensor(self.rotation, dtype=torch.float32)
        K = torch.tensor(self.K, dtype=torch.float32)

        point_camera = rotation @ (point - position)
        point_homogeneous = K @ point_camera
        return point_homogeneous[:2] / point_homogeneous[2]

    def world_to_image(self, point: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Project 3D world point(s) to 2D image coordinates.

        Args:
            point: Single point (3,) or batch of points (N, 3)

        Returns:
            Image coordinates (2,) or (N, 2)
        """
        if isinstance(point, torch.Tensor):
            point = point.detach().numpy()
        point = np.asarray(point, dtype=np.float64)
        single_point = point.ndim == 1

        if single_point:
            point = point.reshape(1, 3)

        points_camera = (self.rotation @ (point - self.position).T).T
        points_homogeneous = (self.K @ points_camera.T).T
        image_points = points_homogeneous[:, :2] / points_homogeneous[:, 2:3]

        if single_point:
            return image_points[0]
        return image_points

    def to_dict(self) -> dict:
        """Serialize camera parameters for JSON storage."""
        return {
            "camera_position": self.position.tolist(),
            "camera_rotation": self.rotation.tolist(),
            "focal_length": list(self.focal_length),
            "principal_point": list(self.principal_point),
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        """Load camera from JSON dict."""
        return cls(
            position=np.array(data["camera_position"], dtype=np.float64),
            rotation=np.array(data["camera_rotation"], dtype=np.float64),
            focal_length=tuple(data.get("focal_length", (50.0, 50.0))),
            principal_point=tuple(data.get("principal_point", (50.0, 50.0))),
            image_size=tuple(data.get("image_size", (100, 100))),
        )

    @classmethod
    def fit_from_correspondences(
        cls,
        pts_2d_px: np.ndarray,
        pts_3d: np.ndarray,
        image_size: tuple[int, int],
    ) -> "Camera":
        """
        Fit pinhole camera from 2D-3D point correspondences using cv2.solvePnP.

        Args:
            pts_2d_px: 2D pixel coordinates (N, 2)
            pts_3d: 3D world coordinates (N, 3)
            image_size: (height, width) of the image

        Returns:
            Fitted Camera instance
        """
        h, w = image_size
        f = float(max(h, w))
        cx, cy = w / 2.0, h / 2.0

        camera_matrix = np.array(
            [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        dist_coeffs = np.zeros(4, dtype=np.float64)

        pts_3d = np.asarray(pts_3d, dtype=np.float64)
        pts_2d_px = np.asarray(pts_2d_px, dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d_px, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            raise RuntimeError("solvePnP failed to find a solution")

        R, _ = cv2.Rodrigues(rvec)
        # Camera position in world coords: C = -R^T @ t
        position = (-R.T @ tvec).flatten()

        return cls(
            position=position,
            rotation=R,
            focal_length=(f, f),
            principal_point=(cx, cy),
            image_size=image_size,
        )
