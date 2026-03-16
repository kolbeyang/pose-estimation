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
# Each tuple: (sequence_name, hd_camera, start_frame, num_frames, person_idx)
EXAMPLES: list[tuple[str, str, int, int, int]] = [
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
]

# --- Video processing ---
TARGET_FPS: float = 30.0  # Use native video rate for maximum temporal context

# --- Optimization (Phase 2) ---
NUM_STEPS: int = 20
LEARNING_RATE: float = 0.001
BONE_LENGTH_LR: float = 0.0001

# Gaussian sigma for heatmap scoring (pixels)
SIGMA: float = 50.0

# Coarse-to-fine sigma schedule: (fraction_of_steps, sigma)
# Phase 1: wide basin for coarse alignment
# Phase 2: medium for refinement
# Phase 3: narrower for precision (but not too aggressive)
SIGMA_SCHEDULE: list[tuple[float, float]] = [
    (1.0, 80.0),   # Constant coarse sigma -- prevents overfitting to noisy 2D targets
]

# Motion penalty weights
POSITION_PENALTY_WEIGHT: float = 50.0

# Per-joint rotation penalty weights (17 joints)
# Trunk joints penalized more to prevent wild torso swings.
# Extremities penalized less so they can track fast motion.
ROTATION_PENALTY_SCALAR: float = 10.0
ROTATION_PENALTY_PER_JOINT: np.ndarray = np.array([
    ROTATION_PENALTY_SCALAR * 3.0,   # 0: Hip (root rotation)
    ROTATION_PENALTY_SCALAR * 1.0,   # 1: RHip
    ROTATION_PENALTY_SCALAR * 0.5,   # 2: RKnee
    ROTATION_PENALTY_SCALAR * 0.2,   # 3: RAnkle
    ROTATION_PENALTY_SCALAR * 1.0,   # 4: LHip
    ROTATION_PENALTY_SCALAR * 0.5,   # 5: LKnee
    ROTATION_PENALTY_SCALAR * 0.2,   # 6: LAnkle
    ROTATION_PENALTY_SCALAR * 1.0,   # 7: Spine
    ROTATION_PENALTY_SCALAR * 1.0,   # 8: Thorax
    ROTATION_PENALTY_SCALAR * 0.5,   # 9: Neck
    ROTATION_PENALTY_SCALAR * 0.5,   # 10: Head
    ROTATION_PENALTY_SCALAR * 0.5,   # 11: LShoulder
    ROTATION_PENALTY_SCALAR * 0.3,   # 12: LElbow
    ROTATION_PENALTY_SCALAR * 0.1,   # 13: LWrist
    ROTATION_PENALTY_SCALAR * 0.5,   # 14: RShoulder
    ROTATION_PENALTY_SCALAR * 0.3,   # 15: RElbow
    ROTATION_PENALTY_SCALAR * 0.1,   # 16: RWrist
], dtype=np.float64)

# Visibility threshold: joints below this are ignored in scoring
VISIBILITY_THRESHOLD: float = 0.5

# Confidence threshold for 2D keypoints fed to MotionBERT.
# Joints below this threshold have their coordinates zeroed out,
# telling MotionBERT to treat them as missing and infer from context.
MOTIONBERT_CONF_THRESHOLD: float = 0.0

# Confidence threshold for replacing FK optimization 2D targets.
# For joints below this threshold, use MotionBERT's projected 2D
# instead of (potentially garbage) Stacked Hourglass detections.
# This does NOT affect MotionBERT's input (MOTIONBERT_CONF_THRESHOLD controls that).
FK_TARGET_CONF_THRESHOLD: float = 0.1

# Weight for initialization anchor penalty.
# Prevents optimizer from drifting away from MotionBERT predictions.
INIT_ANCHOR_WEIGHT: float = 5.0

# Weight for all-joint temporal smoothing penalty.
ALL_JOINTS_SMOOTH_WEIGHT: float = 0.0
