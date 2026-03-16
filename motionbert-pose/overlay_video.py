"""Generate overlay video with heatmaps, 2D keypoints, and projected 3D skeletons.

Overlays on original video frames:
  1. Actual Stacked Hourglass (16, 64, 64) heatmaps (HOT colormap, additive blend)
  2. Yellow dots at raw Stacked Hourglass 2D detection positions (MPII 16 joints)
  3. Green skeleton: MotionBERT raw 3D projected to 2D via camera intrinsics
  4. Red skeleton: Optimized 3D projected to 2D via camera intrinsics
  5. Blue skeleton: Ground truth 3D projected to 2D (if available)
"""

import os

import cv2
import numpy as np

from skeleton import BONES, NUM_JOINTS


def _resize_heatmap_to_frame(
    heatmap_64: np.ndarray,
    affine: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Resize a 64x64 heatmap to the full video frame using the affine transform.

    The heatmap is in 64x64 space (quarter of the 256x256 crop).
    affine maps 256-crop coords -> original pixel coords:
        x_orig = sx * x_256 + tx
        y_orig = sy * y_256 + ty

    Steps:
    1. Resize 64x64 -> crop region size using bilinear interpolation
    2. Place into full frame at the correct offset

    Args:
        heatmap_64: (64, 64) single-channel heatmap.
        affine: (2, 3) affine transform (256-crop -> original pixels).
        frame_h: Full frame height.
        frame_w: Full frame width.

    Returns:
        (frame_h, frame_w) float32 heatmap in original pixel space.
    """
    sx: float = float(affine[0, 0])
    sy: float = float(affine[1, 1])
    tx: float = float(affine[0, 2])
    ty: float = float(affine[1, 2])

    # The 64x64 heatmap corresponds to a 256x256 crop.
    # In original pixel space, the crop spans (256 * sx) x (256 * sy) at offset (tx, ty).
    crop_w: int = max(1, int(round(256 * sx)))
    crop_h: int = max(1, int(round(256 * sy)))

    # Resize heatmap from 64x64 to crop region size
    resized: np.ndarray = cv2.resize(
        heatmap_64, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
    )

    # Place into full frame
    full_heatmap: np.ndarray = np.zeros((frame_h, frame_w), dtype=np.float32)

    # Compute paste region with clipping
    dst_x0: int = int(round(tx))
    dst_y0: int = int(round(ty))
    dst_x1: int = dst_x0 + crop_w
    dst_y1: int = dst_y0 + crop_h

    # Source region (within resized heatmap)
    src_x0: int = max(0, -dst_x0)
    src_y0: int = max(0, -dst_y0)
    src_x1: int = crop_w - max(0, dst_x1 - frame_w)
    src_y1: int = crop_h - max(0, dst_y1 - frame_h)

    # Destination region (within full frame)
    paste_x0: int = max(0, dst_x0)
    paste_y0: int = max(0, dst_y0)
    paste_x1: int = min(frame_w, dst_x1)
    paste_y1: int = min(frame_h, dst_y1)

    if paste_x1 > paste_x0 and paste_y1 > paste_y0 and src_x1 > src_x0 and src_y1 > src_y0:
        full_heatmap[paste_y0:paste_y1, paste_x0:paste_x1] = resized[src_y0:src_y1, src_x0:src_x1]

    return full_heatmap


def _blend_heatmap_additive(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    intensity: float = 200.0,
) -> np.ndarray:
    """Additively blend heatmap onto frame using OpenCV's HOT colormap.

    The heatmap glows on top of the video.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR frame.
        heatmap: (H, W) float32 heatmap.
        intensity: Additive blend strength.

    Returns:
        (H, W, 3) uint8 BGR frame with heatmap overlay.
    """
    maxval: float = float(heatmap.max())
    if maxval < 1e-6:
        return frame_bgr
    norm: np.ndarray = np.clip(heatmap / maxval, 0, 1)
    hm_uint8: np.ndarray = (norm * 255).astype(np.uint8)
    hm_color: np.ndarray = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)

    # Additive blend: frame + heatmap * intensity, weighted by heatmap strength
    glow: np.ndarray = hm_color.astype(np.float32) * (norm[:, :, np.newaxis] * intensity / 255.0)
    result: np.ndarray = np.clip(frame_bgr.astype(np.float32) + glow, 0, 255)
    return result.astype(np.uint8)


def _project_3d_to_2d(
    pts_3d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Perspective projection: (N, 3) camera-space -> (N, 2) pixels.

    Args:
        pts_3d: (N, 3) 3D points in camera space.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (N, 2) pixel coordinates.
    """
    pts: np.ndarray = np.asarray(pts_3d, dtype=np.float64)
    Z: np.ndarray = np.maximum(pts[:, 2], 0.01)
    u: np.ndarray = fx * pts[:, 0] / Z + cx
    v: np.ndarray = fy * pts[:, 1] / Z + cy
    return np.stack([u, v], axis=-1)


def _draw_skeleton_2d(
    frame: np.ndarray,
    pts_2d: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw skeleton bones + joint circles on frame.

    Args:
        frame: (H, W, 3) BGR image (modified in-place).
        pts_2d: (17, 2) pixel coordinates.
        color: BGR color tuple.
        thickness: Line thickness.
    """
    h: int
    w: int
    h, w = frame.shape[:2]
    for parent, child in BONES:
        p1: tuple[int, int] = (int(pts_2d[parent, 0]), int(pts_2d[parent, 1]))
        p2: tuple[int, int] = (int(pts_2d[child, 0]), int(pts_2d[child, 1]))
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    for j in range(NUM_JOINTS):
        pt: tuple[int, int] = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


def generate_overlay_video(
    output_path: str,
    frames_rgb: list[np.ndarray],
    heatmaps: list[np.ndarray],
    mpii_keypoints_2d: list[np.ndarray],
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    camera_fx: float,
    camera_fy: float,
    camera_cx: float,
    camera_cy: float,
    affine: np.ndarray,
    frame_indices: list[int] | None = None,
    gt_3d: list[np.ndarray | None] | None = None,
    intensity: float = 200.0,
) -> None:
    """Generate overlay video from in-memory pipeline data.

    Per-frame layers:
    1. Heatmap overlay from actual Stacked Hourglass (16, 64, 64) heatmaps
    2. Yellow dots at raw MPII 2D detection positions
    3. Green skeleton from MotionBERT raw 3D projected to 2D
    4. Red skeleton from optimized 3D projected to 2D
    5. Blue skeleton from ground truth 3D projected to 2D (if available)

    Args:
        output_path: Path for output .mp4 video.
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        heatmaps: List of (16, 64, 64) raw Stacked Hourglass heatmaps per frame.
        mpii_keypoints_2d: List of (16, 3) MPII keypoints (x, y, conf) in original pixel coords.
        detector_3d: List of (17, 3) MotionBERT camera-space 3D positions.
        optimized_3d: List of (17, 3) optimized camera-space 3D positions.
        camera_fx, camera_fy, camera_cx, camera_cy: Camera intrinsics.
        affine: (2, 3) affine from 256-crop coords to original pixel coords.
        frame_indices: Optional list of frame indices for labeling.
        gt_3d: Optional list of (17, 3) ground truth 3D positions (None for missing frames).
        intensity: Heatmap additive blend strength.
    """
    if not frames_rgb:
        print("  WARNING: No frames for overlay video.")
        return

    n_frames: int = len(frames_rgb)
    h: int
    w: int
    h, w = frames_rgb[0].shape[:2]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, 5.0, (w, h))

    for i in range(n_frames):
        frame_bgr: np.ndarray = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)

        # 1. Heatmap overlay: sum all 16 joint channels, resize to frame, blend
        if i < len(heatmaps):
            hm_combined: np.ndarray = np.clip(heatmaps[i], 0, None).sum(axis=0)  # (64, 64)
            hm_full: np.ndarray = _resize_heatmap_to_frame(hm_combined, affine, h, w)
            frame_bgr = _blend_heatmap_additive(frame_bgr, hm_full, intensity)

        # 2. Yellow dots for raw MPII 2D keypoints
        if i < len(mpii_keypoints_2d):
            kp_mpii: np.ndarray = mpii_keypoints_2d[i]
            for j in range(kp_mpii.shape[0]):
                if kp_mpii[j, 2] < 0.01:
                    continue
                pt: tuple[int, int] = (int(kp_mpii[j, 0]), int(kp_mpii[j, 1]))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(frame_bgr, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

        # 3. Green skeleton: MotionBERT raw 3D projected to 2D
        if i < len(detector_3d):
            det_proj: np.ndarray = _project_3d_to_2d(
                detector_3d[i], camera_fx, camera_fy, camera_cx, camera_cy
            )
            _draw_skeleton_2d(frame_bgr, det_proj, color=(0, 255, 0), thickness=2)

        # 4. Red skeleton: Optimized 3D projected to 2D
        if i < len(optimized_3d):
            opt_proj: np.ndarray = _project_3d_to_2d(
                optimized_3d[i], camera_fx, camera_fy, camera_cx, camera_cy
            )
            _draw_skeleton_2d(frame_bgr, opt_proj, color=(0, 0, 255), thickness=2)

        # 5. Blue skeleton: Ground truth 3D projected to 2D
        if gt_3d is not None and i < len(gt_3d) and gt_3d[i] is not None:
            gt_proj: np.ndarray = _project_3d_to_2d(
                gt_3d[i], camera_fx, camera_fy, camera_cx, camera_cy
            )
            _draw_skeleton_2d(frame_bgr, gt_proj, color=(255, 100, 0), thickness=2)

        # Frame label
        frame_label: str = f"Frame {frame_indices[i]}" if frame_indices and i < len(frame_indices) else f"Frame {i}"
        cv2.putText(
            frame_bgr, frame_label,
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        # Legend
        legend_y: int = h - 20
        cv2.putText(
            frame_bgr,
            "Yellow=SH 2D  Green=MotionBERT  Red=Optimized  Blue=GT",
            (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        writer.write(frame_bgr)

    writer.release()
    print(f"  Saved overlay video: {output_path} ({n_frames} frames)")
