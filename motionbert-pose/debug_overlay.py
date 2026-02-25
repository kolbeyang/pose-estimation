"""Debug script: overlay raw 2D detections on video frames.

Draws Stacked Hourglass 2D keypoints directly on frames (no 3D, no camera, no optimization).
This helps verify the 2D detection pipeline is correct before debugging 3D issues.
"""

import argparse
import os
from datetime import datetime

import cv2
import numpy as np

from detect import (
    detect_persons,
    run_hourglass,
    MPII_JOINTS,
    MPII_LEFT_ARM,
    crop_and_resize,
)


# MPII skeleton connections for visualization
MPII_BONES = [
    (0, 1), (1, 2), (2, 6),      # right leg
    (3, 4), (4, 5), (3, 6),      # left leg
    (6, 7), (7, 8), (8, 9),      # spine -> head
    (10, 11), (11, 12), (12, 8), # right arm
    (13, 14), (14, 15), (13, 8), # left arm
]


def main():
    parser = argparse.ArgumentParser(description="Debug: overlay raw 2D detections")
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("training_runs", f"debug-2d-{timestamp}")

    # 1. Detect persons
    print("=== Person Detection ===")
    frames_rgb, bboxes, image_size = detect_persons(args.video, target_fps=args.fps)
    h, w = image_size
    print(f"  {len(frames_rgb)} frames, {w}x{h}")

    # 2. Run Hourglass
    print("\n=== 2D Pose (Stacked Hourglass) ===")
    all_heatmaps, all_keypoints_2d = run_hourglass(frames_rgb, bboxes, image_size)

    # 3. Save overlays
    print(f"\n=== Saving debug overlays ===")

    # 3a. Full frame overlays with all MPII joints
    full_dir = os.path.join(output_dir, "full_frame")
    os.makedirs(full_dir, exist_ok=True)

    for i in range(len(frames_rgb)):
        frame_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        kp = all_keypoints_2d[i]  # (16, 3) x,y,conf in original pixel coords

        # Draw YOLO bbox
        bbox = bboxes[i].astype(int)
        cv2.rectangle(frame_bgr, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 1)

        # Draw bones
        for j1, j2 in MPII_BONES:
            if kp[j1, 2] > 0.1 and kp[j2, 2] > 0.1:
                p1 = tuple(kp[j1, :2].astype(int))
                p2 = tuple(kp[j2, :2].astype(int))
                cv2.line(frame_bgr, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw joints
        for j in range(16):
            x, y, conf = kp[j]
            if conf < 0.1:
                continue
            # Left arm joints in green, others in blue
            if j in (MPII_LEFT_ARM["a"], MPII_LEFT_ARM["b"], MPII_LEFT_ARM["c"]):
                color = (0, 255, 0)
            else:
                color = (255, 100, 100)
            cv2.circle(frame_bgr, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"{j}", (int(x)+5, int(y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        cv2.imwrite(os.path.join(full_dir, f"frame_{i:04d}.png"), frame_bgr)

    print(f"  Saved {len(frames_rgb)} full-frame overlays to {full_dir}")

    # 3b. Save the actual 256x256 model input crops (what Hourglass sees)
    crop_dir = os.path.join(output_dir, "model_crops")
    os.makedirs(crop_dir, exist_ok=True)

    for i in range(len(frames_rgb)):
        cropped_256, affine = crop_and_resize(frames_rgb[i], bboxes[i], target_size=256)
        crop_bgr = cv2.cvtColor(cropped_256, cv2.COLOR_RGB2BGR)

        # Draw keypoints in crop space (reverse the affine)
        kp = all_keypoints_2d[i]
        for j in range(16):
            x_orig, y_orig, conf = kp[j]
            if conf < 0.1:
                continue
            # Reverse affine: x_256 = (x_orig - offset_x) / scale_x
            x_256 = (x_orig - affine[0, 2]) / affine[0, 0]
            y_256 = (y_orig - affine[1, 2]) / affine[1, 1]
            if j in (MPII_LEFT_ARM["a"], MPII_LEFT_ARM["b"], MPII_LEFT_ARM["c"]):
                color = (0, 255, 0)
            else:
                color = (255, 100, 100)
            cv2.circle(crop_bgr, (int(x_256), int(y_256)), 3, color, -1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(crop_dir, f"frame_{i:04d}.png"), crop_bgr)

    print(f"  Saved {len(frames_rgb)} model input crops to {crop_dir}")

    # 3c. Print some diagnostics
    print(f"\n=== Diagnostics ===")
    for i in range(min(3, len(frames_rgb))):
        kp = all_keypoints_2d[i]
        bbox = bboxes[i]
        print(f"\n  Frame {i}:")
        print(f"    YOLO bbox: ({bbox[0]:.0f}, {bbox[1]:.0f}) -> ({bbox[2]:.0f}, {bbox[3]:.0f})  "
              f"size: {bbox[2]-bbox[0]:.0f}x{bbox[3]-bbox[1]:.0f}")

        _, affine = crop_and_resize(frames_rgb[i], bboxes[i])
        print(f"    Affine scale: ({affine[0,0]:.3f}, {affine[1,1]:.3f})  "
              f"offset: ({affine[0,2]:.1f}, {affine[1,2]:.1f})")

        for name, idx in [("L.Shoulder", 13), ("L.Elbow", 14), ("L.Wrist", 15)]:
            print(f"    {name}: ({kp[idx,0]:.1f}, {kp[idx,1]:.1f}) conf={kp[idx,2]:.3f}")

    print(f"\nDone. Check {output_dir}/")


if __name__ == "__main__":
    main()
