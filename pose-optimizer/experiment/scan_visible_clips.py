"""Scan sequences for contiguous clips where the person is visible in frame.

For each sequence in the source config, loads COCO19 ground truth keypoints,
projects them through the camera, and identifies contiguous runs of frames
where >= MIN_KEYPOINTS_IN_FRAME keypoints are visible. Outputs a new config
with these clips as examples.

Usage:
    cd pose-optimizer
    uv run python experiment/scan_visible_clips.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from camera import Camera
from cmu_data import get_sequence_dir, load_calibration, load_ground_truth_frame

# ----- Configuration -----
SOURCE_CONFIG = "configs/server-single-person-gt-2026-04-11.json"
OUTPUT_CONFIG = "configs/server-visible-clips-2026-04-14.json"
MIN_KEYPOINTS_IN_FRAME = 5
MIN_CLIP_LENGTH = 100


def count_visible_keypoints(
    gt_coco19: np.ndarray,
    camera: Camera,
) -> int:
    """Count how many of the 19 COCO19 keypoints project into the image frame.

    Also requires Z > 0 in camera coordinates (point is in front of camera).
    """
    gt_cam = camera.world_to_camera(gt_coco19)  # (19, 3)
    in_front = gt_cam[:, 2] > 0  # (19,) bool
    gt_2d = camera.camera_to_image(gt_cam)  # (19, 2)
    in_bounds = camera.is_in_frame(gt_2d)  # (19,) bool
    return int((in_front & in_bounds).sum())


def find_contiguous_clips(
    valid_mask: list[bool],
    start_frame: int,
    min_clip_length: int,
) -> list[tuple[int, int]]:
    """Find contiguous runs of True values in valid_mask.

    Returns list of (start_frame_idx, num_frames) tuples for runs >= min_clip_length.
    """
    clips = []
    run_start = None
    for i, v in enumerate(valid_mask):
        if v:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len >= min_clip_length:
                    clips.append((start_frame + run_start, run_len))
                run_start = None
    # Handle run extending to end
    if run_start is not None:
        run_len = len(valid_mask) - run_start
        if run_len >= min_clip_length:
            clips.append((start_frame + run_start, run_len))
    return clips


def main():
    with open(SOURCE_CONFIG) as f:
        base_config = json.load(f)

    data_root = base_config["data_root"]
    all_clips = []
    total_frames_scanned = 0
    total_frames_in_clips = 0
    total_valid_frames = 0
    per_sequence_stats = []

    for ex in base_config["examples"]:
        seq_name = ex["sequence"]
        camera_name = ex["camera"]
        start_frame = ex["start_frame"]
        num_frames = ex["num_frames"]
        person_idx = ex["person_idx"]

        print(f"\n{'='*70}")
        print(f"Scanning: {seq_name}")
        print(f"  Frames {start_frame} to {start_frame + num_frames - 1} ({num_frames} total)")
        print(f"{'='*70}")

        seq_dir = get_sequence_dir(data_root, seq_name)

        # Load calibration
        calib = load_calibration(seq_dir)
        if camera_name not in calib:
            print(f"  WARNING: Camera {camera_name} not in calibration. Skipping.")
            continue
        cam_data = calib[camera_name]
        camera = Camera.from_panoptic_calibration(
            K=cam_data["K"],
            R=cam_data["R"],
            t=cam_data["t"],
            resolution=cam_data["resolution"],
        )
        h, w = camera.image_size
        print(f"  Camera resolution: {w}x{h}")

        # Scan frames
        valid_mask = []
        no_gt_count = 0
        kp_counts = []
        t0 = time.time()

        for i in range(num_frames):
            frame_idx = start_frame + i
            gt = load_ground_truth_frame(seq_dir, frame_idx, person_idx)

            if gt is None:
                valid_mask.append(False)
                no_gt_count += 1
                kp_counts.append(0)
                continue

            n_visible = count_visible_keypoints(gt, camera)
            kp_counts.append(n_visible)
            valid_mask.append(n_visible >= MIN_KEYPOINTS_IN_FRAME)

            # Progress every 5000 frames
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed
                remaining = (num_frames - i - 1) / fps
                print(f"  ... {i+1}/{num_frames} ({fps:.0f} frames/s, ~{remaining:.0f}s left)")

        elapsed = time.time() - t0
        n_valid = sum(valid_mask)
        total_frames_scanned += num_frames
        total_valid_frames += n_valid

        print(f"  Scanned {num_frames} frames in {elapsed:.1f}s ({num_frames/max(elapsed,0.01):.0f} frames/s)")
        print(f"  No GT data: {no_gt_count} frames")
        print(f"  Valid (>= {MIN_KEYPOINTS_IN_FRAME} kp in frame): {n_valid}/{num_frames} ({100*n_valid/max(num_frames,1):.1f}%)")

        if kp_counts:
            arr = np.array(kp_counts)
            print(f"  Keypoint count stats: mean={arr.mean():.1f}, median={np.median(arr):.0f}, min={arr.min()}, max={arr.max()}")

        # Find clips
        clips = find_contiguous_clips(valid_mask, start_frame, MIN_CLIP_LENGTH)
        clip_frames = sum(c[1] for c in clips)
        total_frames_in_clips += clip_frames

        print(f"  Clips (>= {MIN_CLIP_LENGTH} contiguous): {len(clips)}, total {clip_frames} frames ({100*clip_frames/max(num_frames,1):.1f}%)")

        seq_stats = {
            "sequence": seq_name,
            "total_frames": num_frames,
            "no_gt_frames": no_gt_count,
            "valid_frames": n_valid,
            "num_clips": len(clips),
            "clip_frames": clip_frames,
        }
        per_sequence_stats.append(seq_stats)

        for clip_start, clip_len in clips:
            if clip_len > 100:
                print(f"    Clip: start={clip_start}, length={clip_len}")
            all_clips.append({
                "sequence": seq_name,
                "camera": camera_name,
                "start_frame": clip_start,
                "num_frames": clip_len,
                "person_idx": person_idx,
            })

    # Build new config (preserve all settings, replace examples)
    new_config = dict(base_config)
    new_config["examples"] = all_clips

    os.makedirs(os.path.dirname(OUTPUT_CONFIG) or ".", exist_ok=True)
    with open(OUTPUT_CONFIG, "w") as f:
        json.dump(new_config, f, indent=4)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Source sequences: {len(base_config['examples'])}")
    print(f"Output clips: {len(all_clips)}")
    print(f"Total frames scanned: {total_frames_scanned}")
    print(f"Total valid frames (>= {MIN_KEYPOINTS_IN_FRAME} kp): {total_valid_frames} ({100*total_valid_frames/max(total_frames_scanned,1):.1f}%)")
    print(f"Total frames in clips (>= {MIN_CLIP_LENGTH} contiguous): {total_frames_in_clips} ({100*total_frames_in_clips/max(total_frames_scanned,1):.1f}%)")
    print(f"")
    print(f"Per-sequence breakdown:")
    print(f"{'Sequence':<25} {'Total':>7} {'NoGT':>7} {'Valid':>7} {'Valid%':>7} {'Clips':>6} {'ClipF':>7} {'Clip%':>7}")
    print("-" * 90)
    for s in per_sequence_stats:
        pct_valid = 100 * s["valid_frames"] / max(s["total_frames"], 1)
        pct_clip = 100 * s["clip_frames"] / max(s["total_frames"], 1)
        print(f"{s['sequence']:<25} {s['total_frames']:>7} {s['no_gt_frames']:>7} {s['valid_frames']:>7} {pct_valid:>6.1f}% {s['num_clips']:>6} {s['clip_frames']:>7} {pct_clip:>6.1f}%")

    print(f"\nConfig written to: {OUTPUT_CONFIG}")


if __name__ == "__main__":
    main()
