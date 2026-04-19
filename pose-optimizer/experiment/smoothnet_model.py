"""SmoothNet temporal smoothing model.

Reimplemented from: https://github.com/cure-lab/SmoothNet
Paper: "SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos"

The model is a simple 1D temporal network with residual fully-connected blocks.
It operates on flattened pose sequences: input (B, C, T) -> output (B, C, T)
where C = num_joints * 3 (flattened 3D coordinates).
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    # Matches official SmoothNet architecture (no LayerNorm).
    def __init__(self, in_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, in_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.relu(self.linear2(out))
        out = self.dropout(out)
        return x + out


class SmoothNet(nn.Module):
    """SmoothNet: temporal pose refinement via residual FC blocks.

    Input:  (B, C, T) where C = joints*3, T = window_size
    Output: (B, C, T)
    """

    def __init__(
        self,
        window_size: int,
        output_size: int,
        hidden_size: int = 512,
        res_hidden_size: int = 128,
        num_blocks: int = 5,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Linear(window_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(hidden_size, res_hidden_size, dropout))
        self.res_blocks = nn.Sequential(*blocks)

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.encoder(x)  # (B, C, hidden_size)
        out = self.res_blocks(out)  # (B, C, hidden_size)
        out = self.decoder(out)  # (B, C, output_size)
        return out


def smooth_poses(
    poses_3d: list,
    model: SmoothNet,
    device: torch.device,
) -> list:
    """Apply SmoothNet to a sequence of 3D poses with sliding window.

    Args:
        poses_3d: List of (J, 3) numpy arrays in camera space.
        model: Trained SmoothNet model.
        device: Torch device.

    Returns:
        List of (J, 3) numpy arrays, same length as input.
    """
    import numpy as np

    n_frames = len(poses_3d)
    n_joints = poses_3d[0].shape[0]
    window_size = model.window_size

    # Stack into (F, J, 3) then flatten to (F, J*3)
    poses_arr = np.stack(poses_3d, axis=0)  # (F, J, 3)
    poses_flat = poses_arr.reshape(n_frames, -1)  # (F, J*3)

    if n_frames <= window_size:
        # Pad sequence to window_size
        padded = np.zeros((window_size, poses_flat.shape[1]), dtype=np.float32)
        padded[:n_frames] = poses_flat
        x = torch.from_numpy(padded).float().unsqueeze(0).permute(0, 2, 1).to(device)
        with torch.no_grad():
            out = model(x)  # (1, C, output_size)
        out = out.permute(0, 2, 1).squeeze(0).cpu().numpy()  # (output_size, C)
        smoothed_flat = out[:n_frames]
    else:
        # Sliding window with averaging for overlapping regions
        smoothed_flat = np.zeros_like(poses_flat, dtype=np.float32)
        counts = np.zeros(n_frames, dtype=np.float32)

        stride = max(1, window_size // 2)
        starts = list(range(0, n_frames - window_size + 1, stride))
        if starts[-1] + window_size < n_frames:
            starts.append(n_frames - window_size)

        for start in starts:
            window = poses_flat[start:start + window_size]
            x = torch.from_numpy(window).float().unsqueeze(0).permute(0, 2, 1).to(device)
            with torch.no_grad():
                out = model(x)
            out = out.permute(0, 2, 1).squeeze(0).cpu().numpy()  # (window_size, C)
            smoothed_flat[start:start + window_size] += out
            counts[start:start + window_size] += 1.0

        counts = np.maximum(counts, 1.0)
        smoothed_flat /= counts[:, None]

    # Reshape back to list of (J, 3)
    smoothed_arr = smoothed_flat.reshape(n_frames, n_joints, 3)
    return [smoothed_arr[i] for i in range(n_frames)]
