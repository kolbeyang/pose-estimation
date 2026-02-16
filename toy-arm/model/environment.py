import numpy as np


class Environment:
    def __init__(self, cube_size: float = 10.0):
        self.cube_size = cube_size
        self.half_size = cube_size / 2.0

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
