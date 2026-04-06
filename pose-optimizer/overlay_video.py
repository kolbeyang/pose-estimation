"""Unified overlay video generation for both MotionBert and MediaPipe pipelines.

Overlays on original video frames:
  1. Heatmap overlay (Stacked Hourglass, HOT colormap)
  2. Yellow dots at raw 2D detection positions
  3. Green skeleton: raw 3D projected to 2D
  4. Red skeleton: optimized 3D projected to 2D
  5. Blue skeleton: GT 3D projected to 2D (if available)
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from skeleton import BONES, NUM_JOINTS


def _resize_heatmap_to_frame(
    heatmap_64: np.ndarray,
    affine: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Resize a single-channel 64x64 SH heatmap to the full video frame using the affine.

    Args:
        heatmap_64: (64, 64) single-channel heatmap from Stacked Hourglass.
        affine: (2, 3) affine transform (crop coords -> original pixel coords).
        frame_h: Full frame height.
        frame_w: Full frame width.

    Returns:
        (frame_h, frame_w) float32 heatmap in original pixel space.
    """
    sx = float(affine[0, 0])
    sy = float(affine[1, 1])
    tx = float(affine[0, 2])
    ty = float(affine[1, 2])

    # The affine maps [0..255] crop coords to original pixel coords.
    # The heatmap is 64x64, covering the full 256x256 crop at 1/4 resolution.
    # Compute crop region size in pixels and resize heatmap to fill it.
    crop_w = max(1, int(round(256 * sx)))
    crop_h = max(1, int(round(256 * sy)))

    resized = cv2.resize(
        heatmap_64, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR
    )

    full_heatmap = np.zeros((frame_h, frame_w), dtype=np.float32)

    dst_x0 = int(round(tx))
    dst_y0 = int(round(ty))
    dst_x1 = dst_x0 + crop_w
    dst_y1 = dst_y0 + crop_h

    src_x0 = max(0, -dst_x0)
    src_y0 = max(0, -dst_y0)
    src_x1 = crop_w - max(0, dst_x1 - frame_w)
    src_y1 = crop_h - max(0, dst_y1 - frame_h)

    paste_x0 = max(0, dst_x0)
    paste_y0 = max(0, dst_y0)
    paste_x1 = min(frame_w, dst_x1)
    paste_y1 = min(frame_h, dst_y1)

    if paste_x1 > paste_x0 and paste_y1 > paste_y0 and src_x1 > src_x0 and src_y1 > src_y0:
        full_heatmap[paste_y0:paste_y1, paste_x0:paste_x1] = resized[src_y0:src_y1, src_x0:src_x1]

    return full_heatmap


def _blend_heatmap_additive(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    intensity: float = 200.0,
) -> np.ndarray:
    """Additively blend heatmap onto frame using HOT colormap.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR frame.
        heatmap: (H, W) float32 heatmap (0 to 1 range).
        intensity: Blend intensity multiplier.

    Returns:
        (H, W, 3) uint8 BGR frame with heatmap overlay.
    """
    maxval = float(heatmap.max())
    if maxval < 1e-6:
        return frame_bgr
    norm = np.clip(heatmap / maxval, 0, 1)
    hm_uint8 = (norm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)
    glow = hm_color.astype(np.float32) * (norm[:, :, np.newaxis] * intensity / 255.0)
    result = np.clip(frame_bgr.astype(np.float32) + glow, 0, 255)
    return result.astype(np.uint8)


def _project_3d_to_2d(
    pts_3d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Perspective projection: (N, 3) camera-space -> (N, 2) pixels.

    Args:
        pts_3d: (N, 3) points in camera coordinates.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (N, 2) pixel coordinates.
    """
    pts = np.asarray(pts_3d, dtype=np.float64)
    Z = np.maximum(pts[:, 2], 0.01)
    u = fx * pts[:, 0] / Z + cx
    v = fy * pts[:, 1] / Z + cy
    return np.stack([u, v], axis=-1)


def _draw_skeleton_2d(
    frame: np.ndarray,
    pts_2d: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    visible_mask: np.ndarray | None = None,
) -> None:
    """Draw skeleton bones and joint circles on frame (in-place).

    Args:
        frame: (H, W, 3) uint8 BGR frame (modified in-place).
        pts_2d: (16, 2) pixel coordinates. [2D:SKELETON_16]
        color: BGR color tuple.
        thickness: Line thickness.
        visible_mask: (16,) boolean mask; skip invisible joints if provided.
    """
    h, w = frame.shape[:2]
    for parent, child in BONES:
        if visible_mask is not None and (not visible_mask[parent] or not visible_mask[child]):
            continue
        p1 = (int(pts_2d[parent, 0]), int(pts_2d[parent, 1]))
        p2 = (int(pts_2d[child, 0]), int(pts_2d[child, 1]))
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    for j in range(min(pts_2d.shape[0], NUM_JOINTS)):
        if visible_mask is not None and not visible_mask[j]:
            continue
        pt = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


def generate_overlay_video(
    output_path: str,
    frames_rgb: list[np.ndarray],
    heatmaps: list[np.ndarray],
    detector_2d: list[np.ndarray],
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    camera_fx: float,
    camera_fy: float,
    camera_cx: float,
    camera_cy: float,
    affine: np.ndarray,
    frame_indices: list[int] | None = None,
    gt_3d: list[np.ndarray | None] | None = None,
    visibility: list[np.ndarray] | None = None,
    visibility_threshold: float = 0.3,
    intensity: float = 200.0,
    pipeline_name: str = "Detector",
) -> None:
    """Generate overlay video from in-memory pipeline data.

    Args:
        output_path: Path for output .mp4 video.
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        heatmaps: List of (C, H_hm, W_hm) heatmaps per frame.
            [HEATMAP:MPII_16] for MotionBERT, [HEATMAP:SKELETON_16] for MediaPipe.
        detector_2d: List of (16, 2) or (16, 3) 2D detections in pixel coords.
            [2D:MPII_16] (for MotionBERT) or [2D:SKELETON_16] with vis (for MediaPipe).
        detector_3d: List of (16, 3) raw 3D positions in camera space.
            [3D:SKELETON_16]
        optimized_3d: List of (16, 3) optimized 3D positions in camera space.
            [3D:SKELETON_16]
        camera_fx, camera_fy, camera_cx, camera_cy: Camera intrinsics.
        affine: (2, 3) affine from crop/heatmap coords to original pixel coords.
        frame_indices: Optional frame indices for labeling.
        gt_3d: Optional list of (16, 3) ground truth 3D or None.
            [3D:SKELETON_16]
        visibility: Optional per-frame (16,) arrays. [VIS:SKELETON_16]
        visibility_threshold: Threshold for drawing joints.
        intensity: Heatmap blend intensity.
        pipeline_name: Name for legend (e.g. "MotionBERT", "MediaPipe").
    """
    if not frames_rgb:
        logger.warning("No frames for overlay video.")
        return

    n_frames = len(frames_rgb)
    h, w = frames_rgb[0].shape[:2]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 5.0, (w, h))

    for i in range(n_frames):
        frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)

        # 1. Heatmap overlay
        if i < len(heatmaps):
            hm_combined = np.clip(heatmaps[i], 0, None).sum(axis=0)
            hm_full = _resize_heatmap_to_frame(hm_combined, affine, h, w)
            frame_bgr = _blend_heatmap_additive(frame_bgr, hm_full, intensity)

        # Visibility mask
        vis_mask = None
        if visibility is not None and i < len(visibility):
            vis_mask = visibility[i] >= visibility_threshold

        # 2. Yellow dots for raw 2D detections
        if i < len(detector_2d):
            kp = detector_2d[i]
            for j in range(kp.shape[0]):
                # Check if confidence column exists and meets threshold
                if kp.shape[1] > 2 and kp[j, 2] < visibility_threshold:
                    continue
                pt = (int(kp[j, 0]), int(kp[j, 1]))
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(frame_bgr, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

        # 3. Green skeleton: raw 3D projected
        if i < len(detector_3d):
            det_proj = _project_3d_to_2d(
                detector_3d[i], camera_fx, camera_fy, camera_cx, camera_cy
            )
            _draw_skeleton_2d(frame_bgr, det_proj, color=(0, 255, 0), thickness=2,
                              visible_mask=vis_mask)

        # 4. Red skeleton: optimized 3D projected
        if i < len(optimized_3d):
            opt_proj = _project_3d_to_2d(
                optimized_3d[i], camera_fx, camera_fy, camera_cx, camera_cy
            )
            _draw_skeleton_2d(frame_bgr, opt_proj, color=(0, 0, 255), thickness=2,
                              visible_mask=vis_mask)

        # 5. Blue skeleton: GT 3D projected
        if gt_3d is not None and i < len(gt_3d) and gt_3d[i] is not None:
            gt_proj = _project_3d_to_2d(
                gt_3d[i], camera_fx, camera_fy, camera_cx, camera_cy
            )
            _draw_skeleton_2d(frame_bgr, gt_proj, color=(255, 100, 0), thickness=2)

        # Frame label
        frame_label = (
            f"Frame {frame_indices[i]}" if frame_indices and i < len(frame_indices)
            else f"Frame {i}"
        )
        cv2.putText(frame_bgr, frame_label,
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Legend
        legend_y = h - 20
        cv2.putText(
            frame_bgr,
            f"Yellow=2D Det  Green={pipeline_name}  Red=Optimized  Blue=GT",
            (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        writer.write(frame_bgr)

    writer.release()
    logger.info("Saved overlay video: %s (%d frames)", output_path, n_frames)
