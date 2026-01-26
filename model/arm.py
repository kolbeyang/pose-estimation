import numpy as np


class Arm:
    def __init__(
        self,
        a_pos: np.ndarray,
        a_b_length: float,
        a_b_polar: tuple[float, float, float],
        b_c_length: float,
        b_c_theta: float,
    ):
        """
        Args:
            a_pos: Position of point A (x, y, z)
            a_b_length: Length of segment AB
            a_b_polar: (azimuth, elevation, roll) in world coordinates
            b_c_length: Length of segment BC
            b_c_theta: Bend angle relative to AB's local frame
                       theta=0 -> BC perpendicular to AB (90° bend)
                       theta=π/2 -> BC aligned with AB (straight arm)
                       theta=-π/2 -> BC folds back opposite to AB
        """
        self.a_pos = np.asarray(a_pos, dtype=np.float64)
        self.a_b_length = a_b_length
        self.a_b_polar = a_b_polar
        self.b_c_length = b_c_length
        self.b_c_theta = b_c_theta

    def _polar_to_offset(
        self, length: float, azimuth: float, elevation: float
    ) -> np.ndarray:
        x = length * np.cos(elevation) * np.cos(azimuth)
        y = length * np.cos(elevation) * np.sin(azimuth)
        z = length * np.sin(elevation)
        return np.array([x, y, z], dtype=np.float64)

    def _build_local_frame(
        self, azimuth: float, elevation: float, roll: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build orthonormal local coordinate frame for AB segment.

        Returns:
            forward: Unit vector along AB direction
            up: Unit vector perpendicular to AB (after roll)
            right: Unit vector perpendicular to both (after roll)
        """
        # Compute AB direction (forward)
        forward = np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation)
        ], dtype=np.float64)

        # Reference up (handle gimbal lock when AB is near vertical)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(elevation) > np.pi / 2 - 0.01:
            world_up = np.array([0.0, 1.0, 0.0])

        # Build orthonormal frame
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Apply roll rotation around forward axis
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        up_rolled = cos_r * up + sin_r * right
        right_rolled = cos_r * right - sin_r * up

        return forward, up_rolled, right_rolled

    def get_coordinates(self) -> dict[str, np.ndarray]:
        a_pos = self.a_pos

        # Calculate B position from A
        a_b_offset = self._polar_to_offset(
            self.a_b_length, self.a_b_polar[0], self.a_b_polar[1]
        )
        b_pos = a_pos + a_b_offset

        # Calculate C position using local frame
        # theta=0 -> BC perpendicular to AB (up direction)
        # theta=π/2 -> BC aligned with AB (forward direction)
        forward, up, right = self._build_local_frame(*self.a_b_polar)
        bc_direction = np.sin(self.b_c_theta) * forward + np.cos(self.b_c_theta) * up
        c_pos = b_pos + self.b_c_length * bc_direction

        return {"a": a_pos, "b": b_pos, "c": c_pos}

    @classmethod
    def random(
        cls,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
        a_b_length: float | None = None,
        b_c_length: float | None = None,
    ) -> "Arm":
        a_pos = np.random.uniform(min_bounds, max_bounds)
        ab_len = a_b_length if a_b_length is not None else np.random.uniform(0.5, 3.0)
        bc_len = b_c_length if b_c_length is not None else np.random.uniform(0.5, 3.0)
        a_b_polar = (
            np.random.uniform(-np.pi, np.pi),       # azimuth
            np.random.uniform(-np.pi / 2, np.pi / 2),  # elevation
            np.random.uniform(-np.pi, np.pi),       # roll
        )
        b_c_theta = np.random.uniform(-np.pi / 2, np.pi / 2)

        return cls(
            a_pos=a_pos,
            a_b_length=ab_len,
            a_b_polar=a_b_polar,
            b_c_length=bc_len,
            b_c_theta=b_c_theta,
        )
