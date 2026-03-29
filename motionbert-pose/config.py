"""Configuration for MotionBERT pose estimation pipeline."""

import os

import numpy as np

# --- Paths ---
_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
PANOPTIC_ROOT: str = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "data", "panoptic-toolbox")
)
TRAINING_RUNS_DIR: str = os.path.join(_THIS_DIR, "training_runs")

# --- CMU Panoptic examples ---
# Each tuple: (sequence_name, hd_camera, start_frame, num_frames, person_idx,
#              [video_override], [name_suffix])
# Optional 6th element: path to an alternative video file (e.g. occluded version).
#   When set, frames are read from this file starting at index 0,
#   but ground truth still uses start_frame from the original sequence.
# Optional 7th element: suffix appended to the example name for display/output.
_OCC_DIR: str = os.path.join(_THIS_DIR, "occlusion_test_videos")
EXAMPLES: list[tuple] = [
    ("171204_pose1_sample", "00_00", 0, 100, 0),
    ("171204_pose2", "00_00", 200, 150, 0),
    ("171204_pose2", "00_00", 5000, 150, 0),
    ("171204_pose2", "00_00", 15000, 150, 0),
    ("171204_pose3", "00_00", 200, 150, 0),
    ("171204_pose3", "00_00", 4000, 150, 0),
    ("160422_ultimatum1", "00_00", 200, 150, 0),
    ("160422_ultimatum1", "00_00", 10000, 150, 0),
    ("171204_pose2", "00_00", 10000, 150, 0),
    ("171204_pose2", "00_00", 25000, 150, 0),
    # --- New sequences ---
    ("171204_pose1", "00_00", 5000, 150, 0),
    ("171204_pose1", "00_00", 14000, 150, 0),
    ("171204_pose1", "00_00", 22000, 150, 0),
    ("160226_haggling1", "00_00", 3000, 150, 0),
    ("160226_haggling1", "00_00", 7000, 150, 0),
    ("160906_pizza1", "00_00", 2000, 150, 0),
    ("160906_pizza1", "00_00", 4500, 150, 0),
    ("female_example_01", "00_00", 350, 150, 0),
    # --- Occlusion tests (16 visible, 16 black) ---
    (
        "171204_pose1_sample",
        "00_00",
        0,
        100,
        0,
        os.path.join(_OCC_DIR, "171204_pose1_sample_0_occ16v16b.mp4"),
        "_occ16v16b",
    ),
    (
        "171204_pose2",
        "00_00",
        15000,
        150,
        0,
        os.path.join(_OCC_DIR, "171204_pose2_15000_occ16v16b.mp4"),
        "_occ16v16b",
    ),
    (
        "171204_pose3",
        "00_00",
        4000,
        150,
        0,
        os.path.join(_OCC_DIR, "171204_pose3_4000_occ16v16b.mp4"),
        "_occ16v16b",
    ),
]

# --- Video processing ---
TARGET_FPS: float = 30.0  # Use native video rate for maximum temporal context

# --- Optimization (Phase 2) ---
NUM_STEPS: int = 50
LEARNING_RATE: float = 0.002
BONE_LENGTH_LR: float = 0.0001

# Gaussian sigma for heatmap scoring (pixels)
SIGMA: float = 50.0

# Gaussian blur sigma applied to Stacked Hourglass heatmaps before optimization.
# Applied in 64x64 heatmap space. sigma=2 at 64x64 ~ 8px at 256x256 crop.
# 0 = no blur. Widens gradient basin for optimization.
HEATMAP_BLUR_SIGMA: float = 4.0

# Motion penalty weights
POSITION_PENALTY_WEIGHT: float = 500.0

# Per-joint rotation penalty weights (16 joints)
# Trunk joints penalized more to prevent wild torso swings.
# Extremities penalized less so they can track fast motion.
ROTATION_PENALTY_SCALAR: float = 10.0
ROTATION_PENALTY_PER_JOINT: np.ndarray = np.array(
    [
        ROTATION_PENALTY_SCALAR * 3.0,  # 0: Hip (root rotation)
        ROTATION_PENALTY_SCALAR * 1.0,  # 1: RHip
        ROTATION_PENALTY_SCALAR * 0.5,  # 2: RKnee
        ROTATION_PENALTY_SCALAR * 0.2,  # 3: RAnkle
        ROTATION_PENALTY_SCALAR * 1.0,  # 4: LHip
        ROTATION_PENALTY_SCALAR * 0.5,  # 5: LKnee
        ROTATION_PENALTY_SCALAR * 0.2,  # 6: LAnkle
        ROTATION_PENALTY_SCALAR * 1.0,  # 7: Spine
        ROTATION_PENALTY_SCALAR * 1.0,  # 8: Thorax
        ROTATION_PENALTY_SCALAR * 0.5,  # 9: Neck
        ROTATION_PENALTY_SCALAR * 0.5,  # 10: LShoulder
        ROTATION_PENALTY_SCALAR * 0.3,  # 11: LElbow
        ROTATION_PENALTY_SCALAR * 0.1,  # 12: LWrist
        ROTATION_PENALTY_SCALAR * 0.5,  # 13: RShoulder
        ROTATION_PENALTY_SCALAR * 0.3,  # 14: RElbow
        ROTATION_PENALTY_SCALAR * 0.1,  # 15: RWrist
    ],
    dtype=np.float64,
)

# Confidence epsilon for occlusion-aware scoring.
# When a joint has low confidence, the score degrades to log(confidence_epsilon),
# providing no gradient signal. Higher values = less penalty for occluded joints.
CONFIDENCE_EPSILON: float = 1e-4

# Visibility threshold for overlay video and 3D visualization only.
# Joints below this threshold are not drawn (but still scored).
OVERLAY_VISIBILITY_THRESHOLD: float = 0.3

# Confidence threshold for 2D keypoints fed to MotionBERT.
# Joints below this threshold have their coordinates zeroed out,
# telling MotionBERT to treat them as missing and infer from context.
MOTIONBERT_CONF_THRESHOLD: float = 0.0

# Weight for initialization anchor penalty.
# Prevents optimizer from drifting away from MotionBERT predictions.
INIT_ANCHOR_WEIGHT: float = 0.0

# Stacked Hourglass batch size for 2D pose inference.
# Higher = faster but more memory. 256x256x3 float32 = 768KB per frame.
# Batch of 32: ~25MB GPU memory for input tensor alone.
# Batch of 150 (max frames): ~115MB. Safe for most GPUs.
SH_BATCH_SIZE: int = 32

# --- Overlay video ---
OVERLAY_HEATMAP_INTENSITY: float = 200.0
OVERLAY_FPS: float = 5.0
