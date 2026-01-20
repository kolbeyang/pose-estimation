import numpy as np

MAX_RANDOM_TRIES = 10


class Arm:
    def __init__(
        self,
        a_pos: np.ndarray,
        a_b_length: float,
        a_b_polar: tuple[float, float],
        b_c_length: float,
        b_c_polar: tuple[float, float],
    ):
        self.a_pos = np.asarray(a_pos, dtype=np.float64)
        self.a_b_length = a_b_length
        self.a_b_polar = a_b_polar
        self.b_c_length = b_c_length
        self.b_c_polar = b_c_polar

    def _polar_to_offset(
        self, length: float, azimuth: float, elevation: float
    ) -> np.ndarray:
        x = length * np.cos(elevation) * np.cos(azimuth)
        y = length * np.cos(elevation) * np.sin(azimuth)
        z = length * np.sin(elevation)
        return np.array([x, y, z], dtype=np.float64)

    def get_coordinates(self) -> dict[str, np.ndarray]:
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

    @classmethod
    def random(
        cls,
        min_bounds: np.ndarray,
        max_bounds: np.ndarray,
        a_b_length: float | None = None,
        b_c_length: float | None = None,
    ) -> "Arm":
        min_bounds = np.asarray(min_bounds, dtype=np.float64)
        max_bounds = np.asarray(max_bounds, dtype=np.float64)

        for _ in range(MAX_RANDOM_TRIES):
            # Random position for point A within bounds
            a_pos = np.random.uniform(min_bounds, max_bounds)

            # Use provided lengths or generate randomly
            ab_len = a_b_length if a_b_length is not None else np.random.uniform(0.5, 3.0)
            bc_len = b_c_length if b_c_length is not None else np.random.uniform(0.5, 3.0)
            a_b_polar = (np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi / 2, np.pi / 2))
            b_c_polar = (np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi / 2, np.pi / 2))

            arm = cls(
                a_pos=a_pos,
                a_b_length=ab_len,
                a_b_polar=a_b_polar,
                b_c_length=bc_len,
                b_c_polar=b_c_polar,
            )

            # Check if all points are within bounds
            coords = arm.get_coordinates()
            all_in_bounds = True
            for point in coords.values():
                if not (np.all(point >= min_bounds) and np.all(point <= max_bounds)):
                    all_in_bounds = False
                    break

            if all_in_bounds:
                return arm

        raise RuntimeError(f"Failed to generate valid arm within {MAX_RANDOM_TRIES} tries")
