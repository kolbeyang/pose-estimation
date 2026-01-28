import numpy as np
import torch


class Arm:
    def __init__(
        self,
        a_pos: np.ndarray | torch.Tensor,
        a_b_length: float,
        a_b_polar: tuple[float, float, float] | torch.Tensor,
        b_c_length: float,
        b_c_theta: float | torch.Tensor,
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
        # Determine dtype from input tensors (default to float32 for optimization compatibility)
        if isinstance(a_pos, torch.Tensor):
            dtype = a_pos.dtype
        elif isinstance(a_b_polar, torch.Tensor):
            dtype = a_b_polar.dtype
        elif isinstance(b_c_theta, torch.Tensor):
            dtype = b_c_theta.dtype
        else:
            dtype = torch.float32

        # Convert a_pos to torch tensor
        if isinstance(a_pos, np.ndarray):
            self.a_pos = torch.tensor(a_pos, dtype=dtype)
        elif isinstance(a_pos, torch.Tensor):
            self.a_pos = a_pos if a_pos.dtype == dtype else a_pos.to(dtype=dtype)
        else:
            self.a_pos = torch.tensor(a_pos, dtype=dtype)

        self.a_b_length = float(a_b_length)
        self.b_c_length = float(b_c_length)
        self._dtype = dtype

        # Store a_b_polar as tensor (supports gradients for optimization)
        if isinstance(a_b_polar, torch.Tensor):
            self.a_b_polar = a_b_polar if a_b_polar.dtype == dtype else a_b_polar.to(dtype=dtype)
        else:
            self.a_b_polar = torch.tensor(list(a_b_polar), dtype=dtype)

        # Store b_c_theta as tensor (supports gradients for optimization)
        if isinstance(b_c_theta, torch.Tensor):
            self.b_c_theta = b_c_theta if b_c_theta.dtype == dtype else b_c_theta.to(dtype=dtype)
        else:
            self.b_c_theta = torch.tensor(b_c_theta, dtype=dtype)

    def _polar_to_offset(
        self, length: float, azimuth: torch.Tensor, elevation: torch.Tensor
    ) -> torch.Tensor:
        x = length * torch.cos(elevation) * torch.cos(azimuth)
        y = length * torch.cos(elevation) * torch.sin(azimuth)
        z = length * torch.sin(elevation)
        return torch.stack([x, y, z]).to(dtype=self._dtype)

    def _build_local_frame(
        self, azimuth: torch.Tensor, elevation: torch.Tensor, roll: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build orthonormal local coordinate frame for AB segment.

        Computes columns of rotation matrix R = Rz(azimuth) @ Ry(elevation) @ Rx(roll).
        Returns what local (x, y, z) axes map to in world coordinates.

        Returns:
            local_x: Unit vector along AB direction
            local_y: Unit vector perpendicular to AB (after roll)
            local_z: Unit vector perpendicular to both (after roll)
        """
        # Compute local axes using polar_to_offset with unit length
        local_x = self._polar_to_offset(1.0, azimuth, elevation)
        local_y_no_roll = self._polar_to_offset(1.0, azimuth + torch.pi / 2, elevation * 0)
        local_z_no_roll = self._polar_to_offset(1.0, azimuth + torch.pi, torch.pi / 2 - elevation)

        # Apply roll rotation around local_x axis
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        local_y = cos_r * local_y_no_roll + sin_r * local_z_no_roll
        local_z = cos_r * local_z_no_roll - sin_r * local_y_no_roll

        return local_x, local_y, local_z

    def get_coordinates(self) -> dict[str, torch.Tensor]:
        """Return coordinates as torch tensors."""
        a_pos = self.a_pos
        azimuth, elevation, roll = self.a_b_polar[0], self.a_b_polar[1], self.a_b_polar[2]

        # Calculate B position from A
        a_b_offset = self._polar_to_offset(self.a_b_length, azimuth, elevation)
        b_pos = a_pos + a_b_offset

        # Calculate C position using local frame
        # theta=0 -> BC perpendicular to AB (local_y direction)
        # theta=π/2 -> BC aligned with AB (local_x direction)
        local_x, local_y, local_z = self._build_local_frame(azimuth, elevation, roll)
        bc_direction = torch.sin(self.b_c_theta) * local_x + torch.cos(self.b_c_theta) * local_y
        c_pos = b_pos + self.b_c_length * bc_direction

        return {"a": a_pos, "b": b_pos, "c": c_pos}

    def get_coordinates_numpy(self) -> dict[str, np.ndarray]:
        """Return coordinates as numpy arrays (convenience for visualization)."""
        coords = self.get_coordinates()
        return {k: v.detach().numpy() for k, v in coords.items()}

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
