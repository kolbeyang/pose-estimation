"""Generate Gaussian heatmaps from 2D landmarks."""

import numpy as np

from record import FrameData, ARM_INDICES


def generate_heatmap(x: float, y: float, height: int, width: int, sigma: float = 3.0) -> np.ndarray:
    """
    Generate a 2D Gaussian heatmap centered at (x, y).

    Args:
        x, y: Center pixel coordinates
        height, width: Image dimensions
        sigma: Gaussian standard deviation in pixels

    Returns:
        uint8 heatmap array [0, 255]
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return (gaussian * 255).astype(np.uint8)


def generate_frame_heatmaps(
    frame: FrameData,
    arm_indices: dict[str, int],
    image_size: tuple[int, int],
    sigma: float = 3.0,
) -> dict[str, np.ndarray]:
    """
    Generate heatmaps for arm joints from MediaPipe 2D landmarks.

    Args:
        frame: FrameData with normalized 2D landmarks
        arm_indices: Mapping from joint name to MediaPipe index
        image_size: (height, width) of the image
        sigma: Gaussian standard deviation in pixels

    Returns:
        Dict mapping joint name ("a", "b", "c") to uint8 heatmap arrays
    """
    h, w = image_size
    heatmaps = {}
    for name, idx in arm_indices.items():
        # Convert normalized [0,1] coords to pixel coords
        px = frame.landmarks_2d[idx, 0] * w
        py = frame.landmarks_2d[idx, 1] * h
        heatmaps[name] = generate_heatmap(px, py, h, w, sigma=sigma)
    return heatmaps
