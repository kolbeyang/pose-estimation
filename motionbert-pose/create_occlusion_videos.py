"""Create augmented test videos with occlusion patterns.

For each selected example, extracts video frames and creates new MP4 videos
where certain frames are replaced with black (all-zeros) images according
to predefined patterns.

Patterns:
  1. every_other_black:  [visible, black, visible, black, ...]
  2. every_4th_visible:  [visible, black, black, black, ...]
  3. every_4th_black:    [visible, visible, visible, black, ...]
  4. 16on_16off:         [visible*16, black*16, ...]
"""

import json
import os

import cv2
import numpy as np

import config as cfg
from panoptic import extract_video_frames, get_video_path


# Output directory for augmented videos
OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "occlusion_test_videos")

# Patterns: name -> function(frame_index) -> bool (True = visible, False = black)
PATTERNS: dict[str, callable] = {
    "every_other_black": lambda i: i % 2 == 0,
    "every_4th_visible": lambda i: i % 4 == 0,
    "every_4th_black": lambda i: i % 4 != 3,
    "16on_16off": lambda i: (i % 32) < 16,
}

# Select 2 examples (indices 0 and 4) for variety
SELECTED_EXAMPLE_INDICES: list[int] = [0, 4]


def create_occlusion_video(
    frames_rgb: list[np.ndarray],
    pattern_fn: callable,
    output_path: str,
    fps: float = 5.0,
) -> int:
    """Create an augmented video with occlusion pattern applied.

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        pattern_fn: Function(frame_idx) -> bool. True = keep, False = black.
        output_path: Path for output .mp4 file.
        fps: Output video FPS.

    Returns:
        Number of blacked-out frames.
    """
    if not frames_rgb:
        return 0

    h: int
    w: int
    h, w = frames_rgb[0].shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    n_blacked: int = 0
    for i, frame in enumerate(frames_rgb):
        if pattern_fn(i):
            frame_bgr: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)
            n_blacked += 1
        writer.write(frame_bgr)

    writer.release()
    return n_blacked


def main() -> None:
    """Create occlusion test videos for selected examples."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest: list[dict[str, str | int]] = []

    for ex_idx in SELECTED_EXAMPLE_INDICES:
        seq_name: str
        camera_name: str
        start_frame: int
        num_frames: int
        person_idx: int
        seq_name, camera_name, start_frame, num_frames, person_idx = cfg.EXAMPLES[ex_idx]

        print(f"\nProcessing: {seq_name}_{start_frame} (example {ex_idx})")

        video_path: str = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)

        # Extract frames
        video_fps: float = 30.0
        frame_step: int = max(1, int(round(video_fps / cfg.TARGET_FPS)))
        frame_indices: list[int] = list(range(start_frame, start_frame + num_frames, frame_step))
        frames_rgb: list[np.ndarray] = extract_video_frames(video_path, frame_indices)
        frame_indices = frame_indices[:len(frames_rgb)]
        print(f"  Extracted {len(frames_rgb)} frames")

        if len(frames_rgb) < 2:
            print(f"  ERROR: Not enough frames, skipping.")
            continue

        for pattern_name, pattern_fn in PATTERNS.items():
            video_name: str = f"{seq_name}_{start_frame}_{pattern_name}.mp4"
            out_path: str = os.path.join(OUTPUT_DIR, video_name)

            n_blacked: int = create_occlusion_video(
                frames_rgb, pattern_fn, out_path, fps=cfg.OVERLAY_FPS,
            )
            n_total: int = len(frames_rgb)
            print(f"  {pattern_name}: {n_blacked}/{n_total} frames blacked -> {out_path}")

            manifest.append({
                "video_path": out_path,
                "video_name": video_name,
                "source_example_idx": ex_idx,
                "sequence": seq_name,
                "camera": camera_name,
                "start_frame": start_frame,
                "num_frames": n_total,
                "pattern": pattern_name,
                "n_blacked": n_blacked,
                "n_visible": n_total - n_blacked,
            })

    # Save manifest
    manifest_path: str = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved: {manifest_path}")
    print(f"Total videos created: {len(manifest)}")


if __name__ == "__main__":
    main()
