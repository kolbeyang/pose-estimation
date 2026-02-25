"""Overlay merged heatmaps on original video frames and save as video."""

import argparse
import glob
import os
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Original video path")
    parser.add_argument("--heatmaps", required=True, help="Heatmap directory (e.g. images/mediapipe-heatmaps-...)")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS (must match what main.py used)")
    parser.add_argument("--output", default="heatmap_overlay.mp4", help="Output video path")
    parser.add_argument("--opacity", type=float, default=0.8, help="Video opacity (0=heatmap only, 1=video only)")
    args = parser.parse_args()

    # Discover frames from heatmap dir
    a_files = sorted(glob.glob(os.path.join(args.heatmaps, "frame_*_a.png")))
    n_frames = len(a_files)
    if n_frames == 0:
        print(f"No heatmaps found in {args.heatmaps}")
        return

    print(f"Found {n_frames} frames of heatmaps")

    # Open video
    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, round(video_fps / args.fps))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Video: {w}x{h} @ {video_fps:.1f} fps, skipping every {frame_skip} frames")

    # Set up writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

    frame_idx = 0
    heatmap_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or heatmap_idx >= n_frames:
            break

        if frame_idx % frame_skip == 0:
            # Load and merge 3 heatmaps with max (keep all white)
            merged = np.zeros((h, w), dtype=np.uint8)
            for name in ["a", "b", "c"]:
                path = os.path.join(args.heatmaps, f"frame_{heatmap_idx:04d}_{name}.png")
                hm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if hm is not None:
                    # Resize if heatmap size differs from video
                    if hm.shape[:2] != (h, w):
                        hm = cv2.resize(hm, (w, h))
                    merged = np.maximum(merged, hm)

            # Convert merged heatmap to color (hot colormap)
            heatmap_color = cv2.applyColorMap(merged, cv2.COLORMAP_HOT)

            # Blend: video * opacity + heatmap * (1 - opacity)
            blended = cv2.addWeighted(frame, args.opacity, heatmap_color, 1.0 - args.opacity, 0)
            out.write(blended)
            heatmap_idx += 1

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Wrote {heatmap_idx} frames to {args.output}")


if __name__ == "__main__":
    main()
