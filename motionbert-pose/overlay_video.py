"""Generate overlay video from a predictions JSON.

Projects skeletons onto the original video frames, with optional
real Stacked Hourglass heatmap overlays.

Usage:
    python overlay_video.py <predictions.json> [--output overlay.mp4]

Shows:
  - Real Stacked Hourglass heatmaps (summed across joints, HOT colormap)
  - White dots at 2D detection centers
  - Green skeleton: Detector 3D projected to 2D via camera intrinsics
  - Red skeleton: Optimised 3D projected to 2D via camera intrinsics
  - Blue skeleton: Ground truth projected to 2D (if available)
"""

import argparse
import json
import os

import cv2
import numpy as np

from panoptic import extract_video_frames, get_video_path
from skeleton import BONES, NUM_JOINTS
import config as cfg


def _project_3d_to_2d(pts_3d, fx, fy, cx, cy):
    """Perspective projection: (N, 3) camera-space -> (N, 2) pixels."""
    pts = np.asarray(pts_3d, dtype=np.float64)
    Z = np.maximum(pts[:, 2], 0.01)
    u = fx * pts[:, 0] / Z + cx
    v = fy * pts[:, 1] / Z + cy
    return np.stack([u, v], axis=-1)


def _render_heatmap_from_detections(h, w, pts_2d, visibility, sigma):
    """Render Gaussian heatmap from 2D detection points.

    Fallback when real heatmaps are not available.
    """
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    radius = int(3 * sigma)
    heatmap = np.zeros((h, w), dtype=np.float32)

    for j in range(pts_2d.shape[0]):
        if visibility[j] < 0.01:
            continue
        cx_j = pts_2d[j, 0]
        cy_j = pts_2d[j, 1]
        x0 = max(0, int(cx_j) - radius)
        x1 = min(w, int(cx_j) + radius + 1)
        y0 = max(0, int(cy_j) - radius)
        y1 = min(h, int(cy_j) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        local_xs = xs[x0:x1]
        local_ys = ys[y0:y1]
        dx = local_xs[np.newaxis, :] - cx_j
        dy = local_ys[:, np.newaxis] - cy_j
        sq_dist = dx ** 2 + dy ** 2
        gauss = np.exp(-sq_dist / (2.0 * sigma ** 2))
        heatmap[y0:y1, x0:x1] += visibility[j] * gauss

    return heatmap


def _render_real_heatmaps(h, w, heatmaps_64, affine_256_to_pixel):
    """Render real Stacked Hourglass heatmaps at video resolution.

    Args:
        h, w: Video frame dimensions.
        heatmaps_64: (16, 64, 64) raw heatmaps for one frame.
        affine_256_to_pixel: (2, 3) affine from 256-crop to pixel coords.

    Returns:
        (h, w) float32 heatmap image.
    """
    # Sum all 16 joint heatmaps
    hm_sum = heatmaps_64.sum(axis=0)  # (64, 64)
    # Normalize to [0, 1]
    hm_max = hm_sum.max()
    if hm_max > 1e-6:
        hm_sum = hm_sum / hm_max

    # Upscale 64 -> 256 crop space
    hm_256 = cv2.resize(hm_sum, (256, 256), interpolation=cv2.INTER_LINEAR)

    # Apply affine to map 256-crop -> pixel coords
    # affine_256_to_pixel: pixel = A * crop + b
    #   [[scale_x, 0, offset_x],
    #    [0, scale_y, offset_y]]
    M = affine_256_to_pixel[:2, :].astype(np.float64)
    hm_full = cv2.warpAffine(
        hm_256, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return hm_full.astype(np.float32)


def _blend_heatmap_additive(frame_bgr, heatmap, intensity=200.0):
    """Additively blend heatmap onto frame using HOT colormap."""
    maxval = heatmap.max()
    if maxval < 1e-6:
        return frame_bgr
    norm = np.clip(heatmap / maxval, 0, 1)
    hm_uint8 = (norm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)

    glow = hm_color.astype(np.float32) * (norm[:, :, np.newaxis] * intensity / 255.0)
    result = np.clip(frame_bgr.astype(np.float32) + glow, 0, 255)
    return result.astype(np.uint8)


def _draw_skeleton_2d(frame, pts_2d, color, thickness=2):
    """Draw skeleton bones + joints on frame."""
    h, w = frame.shape[:2]
    for parent, child in BONES:
        p1 = (int(pts_2d[parent, 0]), int(pts_2d[parent, 1]))
        p2 = (int(pts_2d[child, 0]), int(pts_2d[child, 1]))
        if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    for j in range(NUM_JOINTS):
        pt = (int(pts_2d[j, 0]), int(pts_2d[j, 1]))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


def generate_overlay_video(
    json_path: str,
    output_path: str,
    heatmaps_list: list[np.ndarray] | None = None,
    affine_256_to_pixel: np.ndarray | None = None,
    sigma_fallback: float = 10.0,
):
    """Generate overlay video from predictions JSON.

    Args:
        json_path: Path to predictions JSON file.
        output_path: Output MP4 path.
        heatmaps_list: Optional per-frame (16, 64, 64) real heatmaps.
            If None, falls back to rendering Gaussians from 2D detections.
        affine_256_to_pixel: (2, 3) affine for mapping heatmaps to pixel space.
            Required if heatmaps_list is provided.
        sigma_fallback: Gaussian sigma for fallback heatmap rendering.
    """
    with open(json_path) as f:
        data = json.load(f)

    seq_name = data["sequence"]
    camera_name = data["camera"]
    frame_indices = data["frame_indices"]
    intrinsics = data["camera_intrinsics"]
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    frames_data = data["frames"]

    use_real_heatmaps = heatmaps_list is not None and affine_256_to_pixel is not None

    # Load original video frames
    video_path = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)
    print(f"  Loading {len(frame_indices)} frames from {video_path}...")
    frames_rgb = extract_video_frames(video_path, frame_indices)

    if len(frames_rgb) == 0:
        print("  ERROR: No frames loaded for overlay video.")
        return

    h, w = frames_rgb[0].shape[:2]
    hm_mode = "real Stacked Hourglass" if use_real_heatmaps else f"Gaussian (sigma={sigma_fallback})"
    print(f"  Frame size: {w}x{h}, {len(frames_rgb)} frames, heatmaps: {hm_mode}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, 5.0, (w, h))

    for i, frame_rgb in enumerate(frames_rgb):
        if i >= len(frames_data):
            break

        fd = frames_data[i]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        mp_2d = np.array(fd["mediapipe_2d"])
        mp_3d = np.array(fd["mediapipe_3d"])
        opt_3d = np.array(fd["optimized_3d"])

        # Visibility
        if "visibility" in fd:
            vis = np.array(fd["visibility"], dtype=np.float32)
        else:
            vis = (np.linalg.norm(mp_2d, axis=1) > 1.0).astype(np.float32)

        # 1. Render heatmaps
        if use_real_heatmaps and i < len(heatmaps_list):
            heatmap = _render_real_heatmaps(h, w, heatmaps_list[i], affine_256_to_pixel)
        else:
            heatmap = _render_heatmap_from_detections(h, w, mp_2d, vis, sigma_fallback)
        frame_bgr = _blend_heatmap_additive(frame_bgr, heatmap)

        # Project 3D to 2D
        mp_proj = _project_3d_to_2d(mp_3d, fx, fy, cx, cy)
        opt_proj = _project_3d_to_2d(opt_3d, fx, fy, cx, cy)

        # GT skeleton
        gt_3d_data = fd.get("ground_truth_3d")
        gt_proj = None
        if gt_3d_data is not None:
            gt_proj = _project_3d_to_2d(np.array(gt_3d_data), fx, fy, cx, cy)

        # 2. White dots at 2D detection centers
        for j in range(NUM_JOINTS):
            pt = (int(mp_2d[j, 0]), int(mp_2d[j, 1]))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(frame_bgr, pt, 5, (255, 255, 255), -1, cv2.LINE_AA)

        # 3. Green skeleton: Detector 3D projected to 2D
        _draw_skeleton_2d(frame_bgr, mp_proj, color=(0, 255, 0), thickness=2)

        # 4. Red skeleton: Optimised 3D projected to 2D
        _draw_skeleton_2d(frame_bgr, opt_proj, color=(0, 0, 255), thickness=2)

        # 5. Blue skeleton: Ground truth projected to 2D
        if gt_proj is not None:
            _draw_skeleton_2d(frame_bgr, gt_proj, color=(255, 100, 0), thickness=2)

        # Frame label with sequence info for debugging
        example_name = f"{seq_name}_{frame_indices[0]}" if frame_indices else seq_name
        cv2.putText(
            frame_bgr, f"{example_name}  Frame {frame_indices[i]} ({i+1}/{len(frames_data)})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        # Legend
        legend_y = h - 20
        cv2.putText(frame_bgr, "White=2D det  Green=Detector 3D  Red=Optimised  Blue=GT",
                    (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        writer.write(frame_bgr)

    writer.release()
    print(f"  Saved overlay video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate overlay video")
    parser.add_argument("json_path", help="Path to predictions JSON file")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output video path (default: same dir as JSON, <name>_overlay.mp4)",
    )
    parser.add_argument(
        "--sigma", type=float, default=10.0,
        help="Gaussian sigma in pixels for fallback heatmap rendering (default: 10)",
    )
    args = parser.parse_args()

    if args.output is None:
        json_dir = os.path.dirname(args.json_path)
        base = os.path.splitext(os.path.basename(args.json_path))[0]
        args.output = os.path.join(json_dir, f"{base}_overlay.mp4")

    generate_overlay_video(args.json_path, args.output, sigma_fallback=args.sigma)


if __name__ == "__main__":
    main()
