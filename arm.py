import numpy as np


class Arm:
    def __init__(
        self,
        a_pos: np.ndarray,
        a_b_length: float,
        a_b_polar: tuple[float, float],
        b_c_length: float,
        b_c_polar: tuple[float, float],
    ):
        """
        Initialize an articulated arm with three points A, B, C.

        Args:
            a_pos: 3D position of point A
            a_b_length: distance from A to B
            a_b_polar: (azimuth, elevation) angles for A->B segment in radians
            b_c_length: distance from B to C
            b_c_polar: (azimuth, elevation) angles for B->C segment in radians
        """
        self.a_pos = np.asarray(a_pos, dtype=np.float64)
        self.a_b_length = a_b_length
        self.a_b_polar = a_b_polar
        self.b_c_length = b_c_length
        self.b_c_polar = b_c_polar

    def _polar_to_offset(
        self, length: float, azimuth: float, elevation: float
    ) -> np.ndarray:
        """
        Convert polar coordinates (length, azimuth, elevation) to a 3D offset.

        Args:
            length: distance
            azimuth: angle in XY plane from +X axis (radians)
            elevation: angle above XY plane (radians)

        Returns:
            3D offset vector as numpy array
        """
        x = length * np.cos(elevation) * np.cos(azimuth)
        y = length * np.cos(elevation) * np.sin(azimuth)
        z = length * np.sin(elevation)
        return np.array([x, y, z], dtype=np.float64)

    def get_coordinates(self) -> dict[str, np.ndarray]:
        """
        Calculate and return the 3D positions of points A, B, and C.

        Returns:
            Dictionary with keys "a", "b", "c" mapping to their 3D positions
        """
        a_pos = self.a_pos

        # Calculate B position from A
        a_b_offset = self._polar_to_offset(
            self.a_b_length, self.a_b_polar[0], self.a_b_polar[1]
        )
        b_pos = a_pos + a_b_offset

        # Calculate C position from B
        b_c_offset = self._polar_to_offset(
            self.b_c_length, self.b_c_polar[0], self.b_c_polar[1]
        )
        c_pos = b_pos + b_c_offset

        return {"a": a_pos, "b": b_pos, "c": c_pos}
