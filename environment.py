import numpy as np
import cv2


class Environment:
    def __init__(
        self,
        cube_size: float = 10.0,
        camera_position: np.ndarray = None,
        camera_rotation: np.ndarray = None,
        focal_length: tuple[float, float] = (800.0, 800.0),
        principal_point: tuple[float, float] = (320.0, 240.0),
        image_size: tuple[int, int] = (640, 480),
    ):
        """
        Initialize the simulation environment.

        Args:
            cube_size: Size of the cubic environment (default 10x10x10)
            camera_position: 3D position of camera in world coordinates
            camera_rotation: 3x3 rotation matrix for camera orientation
            focal_length: (fx, fy) focal length in pixels
            principal_point: (cx, cy) principal point in pixels
            image_size: (width, height) of the image sensor
        """
        self.cube_size = cube_size
        self.half_size = cube_size / 2.0

        # Camera extrinsics
        if camera_position is None:
            camera_position = np.array([0.0, 0.0, 5.0])
        self.camera_position = camera_position.astype(np.float64)

        if camera_rotation is None:
            camera_rotation = np.eye(3)
        self.camera_rotation = camera_rotation.astype(np.float64)

        # Camera intrinsics
        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size

        # Build intrinsic matrix K
        fx, fy = focal_length
        cx, cy = principal_point
        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def world_to_image(self, point: np.ndarray) -> np.ndarray:
        """
        Project a 3D world coordinate to 2D image coordinates.

        Args:
            point: 3D point in world coordinates (shape: (3,) or (N, 3))

        Returns:
            2D pixel coordinates (shape: (2,) or (N, 2))
        """
        point = np.asarray(point, dtype=np.float64)
        single_point = point.ndim == 1

        if single_point:
            point = point.reshape(1, 3)

        # Convert rotation matrix to Rodrigues vector
        rvec, _ = cv2.Rodrigues(self.camera_rotation)

        # Translation vector (camera position in world frame -> need to transform)
        # tvec = -R @ camera_position (transforms world origin to camera frame)
        tvec = -self.camera_rotation @ self.camera_position

        # No distortion coefficients
        dist_coeffs = np.zeros(5, dtype=np.float64)

        # Project points
        image_points, _ = cv2.projectPoints(
            point.reshape(-1, 1, 3),
            rvec,
            tvec,
            self.K,
            dist_coeffs
        )

        image_points = image_points.reshape(-1, 2)

        if single_point:
            return image_points[0]
        return image_points

    def get_cube_corners(self) -> np.ndarray:
        """
        Get the 8 corners of the environment cube.

        Returns:
            Array of shape (8, 3) with corner coordinates
        """
        h = self.half_size
        corners = np.array([
            [-h, -h, -h],
            [+h, -h, -h],
            [+h, +h, -h],
            [-h, +h, -h],
            [-h, -h, +h],
            [+h, -h, +h],
            [+h, +h, +h],
            [-h, +h, +h],
        ], dtype=np.float64)
        return corners
