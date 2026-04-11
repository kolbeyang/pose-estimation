"""Extract combined heatmap from a single video frame using YOLO + Stacked Hourglass."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "run_motionbert"))

import cv2
import numpy as np
from run_motionbert.detect import load_yolo_sh_models, detect_2d_poses
from overlay_video import _resize_heatmap_to_frame
from scoring import SKELETON_TO_MPII_HEATMAP
from skeleton import EVAL_JOINTS

# MPII heatmap channels corresponding to the 15 eval joints
_EVAL_MPII_CHANNELS = sorted(set(
    SKELETON_TO_MPII_HEATMAP[j] for j in EVAL_JOINTS
))

IMAGE_PATH = os.path.expanduser("~/Downloads/frame_378.jpg")
OUTPUT_PATH = os.path.expanduser("~/Downloads/heatmap_combined.png")


def main():
    # Load image as RGB
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w = rgb.shape[:2]
    print(f"Loaded image: {frame_w}x{frame_h}")

    # Load models and run detection
    models = load_yolo_sh_models()
    _, _, heatmaps, _, affine = detect_2d_poses([rgb], models)

    # heatmaps[0] is (16, 64, 64) — take max across 15 eval keypoints (exclude Spine/MPII[7])
    hm = heatmaps[0]  # (16, 64, 64)
    combined_64 = np.max(hm[_EVAL_MPII_CHANNELS], axis=0)  # (64, 64)
    print(f"Heatmap range: [{combined_64.min():.4f}, {combined_64.max():.4f}]")

    # Map back to original frame size using affine
    full = _resize_heatmap_to_frame(combined_64, affine, frame_h, frame_w)
    full = full / (full.max() + 1e-8)
    full_uint8 = (full * 255).astype(np.uint8)

    cv2.imwrite(OUTPUT_PATH, full_uint8)
    print(f"Saved {frame_w}x{frame_h} combined heatmap to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
