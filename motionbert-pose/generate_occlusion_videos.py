"""Generate occluded test videos from CMU Panoptic source videos.

For each source example, extracts the relevant frame range, applies a cyclic
occlusion pattern (16 visible, 16 black), and writes a new video file.

The output videos can be used as new EXAMPLES entries to test occlusion
invariance — ground truth is unchanged since it comes from separate body JSON files.

Usage:
    uv run python generate_occlusion_videos.py
"""

import os

import cv2
import numpy as np

import config as cfg

# (sequence_name, camera, start_frame, num_frames, person_idx)
SOURCE_EXAMPLES: list[tuple[str, str, int, int, int]] = [
    ("171204_pose1_sample", "00_00", 0, 100, 0),
    ("171204_pose2", "00_00", 15000, 150, 0),
    ("171204_pose3", "00_00", 4000, 150, 0),
]

VISIBLE_FRAMES: int = 16
BLACK_FRAMES: int = 16
CYCLE: int = VISIBLE_FRAMES + BLACK_FRAMES

OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "occlusion_test_videos")


def generate_occluded_video(
    seq_name: str,
    camera_name: str,
    start_frame: int,
    num_frames: int,
) -> str:
    """Extract frame range from source video and write occluded version.

    Returns:
        Path to the output video file.
    """
    src_path: str = os.path.join(
        cfg.PANOPTIC_ROOT, seq_name, "hdVideos", f"hd_{camera_name}.mp4",
    )
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source video not found: {src_path}")

    tag: str = f"{seq_name}_{start_frame}_occ{VISIBLE_FRAMES}v{BLACK_FRAMES}b"
    out_path: str = os.path.join(OUTPUT_DIR, f"{tag}.mp4")

    cap: cv2.VideoCapture = cv2.VideoCapture(src_path)
    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
    writer: cv2.VideoWriter = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    black_frame: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
    frames_written: int = 0

    for i in range(num_frames):
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: Could only read {i}/{num_frames} frames from {seq_name}")
            break

        pos_in_cycle: int = i % CYCLE
        if pos_in_cycle >= VISIBLE_FRAMES:
            writer.write(black_frame)
        else:
            writer.write(frame)
        frames_written += 1

    cap.release()
    writer.release()

    n_black: int = sum(
        1 for i in range(frames_written) if (i % CYCLE) >= VISIBLE_FRAMES
    )
    print(f"  {tag}: {frames_written} frames ({frames_written - n_black} visible, {n_black} black)")
    return out_path


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Occlusion pattern: {VISIBLE_FRAMES} visible, {BLACK_FRAMES} black\n")

    for seq_name, camera, start, nframes, _person in SOURCE_EXAMPLES:
        print(f"Processing {seq_name} (frames {start}-{start + nframes - 1})...")
        out: str = generate_occluded_video(seq_name, camera, start, nframes)
        print(f"  -> {out}\n")

    print("Done. Add these as new EXAMPLES entries pointing to the occluded videos.")
    print("Ground truth is unchanged — it comes from the body JSON files, not the video.")


if __name__ == "__main__":
    main()
