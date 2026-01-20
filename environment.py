import numpy as np


class Environment:
    def __init__(
        self,
        cube_size: float = 10.0,
        camera_position: np.ndarray = None,
        camera_rotation: np.ndarray = None,
        focal_length: tuple[float, float] = (800.0, 800.0),
        principal_point: tuple[float, float] = (200, 200),
        image_size: tuple[int, int] = (400, 400),
    ):
        self.cube_size = cube_size
        self.half_size = cube_size / 2.0

        if camera_position is None:
            camera_position = np.array([0.0, 0.0, 0.0])
        self.camera_position = camera_position.astype(np.float64)

        if camera_rotation is None:
            camera_rotation = np.eye(3)
        self.camera_rotation = camera_rotation.astype(np.float64)

        self.focal_length = focal_length
        self.principal_point = principal_point
        self.image_size = image_size

        fx, fy = focal_length
        cx, cy = principal_point
        self.K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def world_to_image(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64)
        single_point = point.ndim == 1

        if single_point:
            point = point.reshape(1, 3)

        points_camera = (self.camera_rotation @ (point - self.camera_position).T).T

        points_homogeneous = (self.K @ points_camera.T).T
        image_points = points_homogeneous[:, :2] / points_homogeneous[:, 2:3]

        if single_point:
            return image_points[0]
        return image_points

    def get_cube_corners(self) -> np.ndarray:
        h = self.half_size
        corners = np.array(
            [
                [-h, -h, -h],
                [+h, -h, -h],
                [+h, +h, -h],
                [-h, +h, -h],
                [-h, -h, +h],
                [+h, -h, +h],
                [+h, +h, +h],
                [-h, +h, +h],
            ],
            dtype=np.float64,
        )
        return corners

    def snap_a_photo(self, points: np.ndarray) -> np.ndarray:
        height, width = self.image_size
        image = np.zeros((height, width), dtype=np.uint8)

        image_points = self.world_to_image(points)

        # Round to nearest pixel and filter points within image bounds
        pixel_coords = np.round(image_points).astype(int)

        for x, y in pixel_coords:
            if 0 <= x < width and 0 <= y < height:
                image[y, x] = 255

        return image
