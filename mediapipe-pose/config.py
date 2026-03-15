"""Configuration for full-body FK pose optimization pipeline."""

import os

import numpy as np

# --- Paths ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PANOPTIC_ROOT = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "data", "panoptic-toolbox")
)
TRAINING_RUNS_DIR = os.path.join(_THIS_DIR, "training_runs")

# --- CMU Panoptic examples ---
# Each tuple: (sequence_name, hd_camera, start_frame, num_frames, person_idx)
# - sequence_name: CMU Panoptic recording name
# - hd_camera: HD camera view (e.g. "00_00")
# - start_frame: starting HD video frame index (~30 fps)
# - num_frames: how many HD frames to process (subsampled to TARGET_FPS)
# - person_idx: which person to track (0 = first detected body in GT)
EXAMPLES = [
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
TARGET_FPS = 10.0  # Subsample HD video (native ~30 fps) to this rate

# --- Optimization ---
NUM_STEPS = 300
LEARNING_RATE = 0.001

# Gaussian sigma for heatmap scoring (pixels)
SIGMA = 50.0

# Motion penalty weights
POSITION_PENALTY_WEIGHT = 50.0
# Per-joint rotation penalty weights: inner joints (trunk) penalised more,
# outer joints (extremities) penalised less so they can move freely.
ROTATION_PENALTY_SCALAR = 10.0
ROTATION_PENALTY_PER_JOINT = np.array(
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
        ROTATION_PENALTY_SCALAR * 0.5,  # 9: Nose
        ROTATION_PENALTY_SCALAR * 0.5,  # 10: LShoulder
        ROTATION_PENALTY_SCALAR * 0.3,  # 11: LElbow
        ROTATION_PENALTY_SCALAR * 0.1,  # 12: LWrist
        ROTATION_PENALTY_SCALAR * 0.5,  # 13: RShoulder
        ROTATION_PENALTY_SCALAR * 0.3,  # 14: RElbow
        ROTATION_PENALTY_SCALAR * 0.1,  # 15: RWrist
    ],
    dtype=np.float64,
)

# Bone length learning rate (separate from main LR)
BONE_LENGTH_LR = 0.0001

# --- Heatmap / scoring ---
# Visibility threshold: joints with MediaPipe visibility below this are ignored
VISIBILITY_THRESHOLD = 0.5
