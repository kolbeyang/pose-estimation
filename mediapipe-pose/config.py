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
# (sequence_name, hd_camera, start_frame, num_frames, person_idx)
# Frames are in HD video frame indices (~30 fps).
# We subsample to TARGET_FPS during processing.
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
SIGMA = 30.0

# Motion penalty weights
POSITION_PENALTY_WEIGHT = 5.0
ROTATION_PENALTY_WEIGHT = 1.0

# Per-joint rotation penalty weights: inner joints (trunk) penalised more,
# outer joints (extremities) penalised less so they can move freely.
ROTATION_PENALTY_PER_JOINT = np.array([
    1.0,   # 0: Hip (root rotation)
    1.0,   # 1: RHip
    0.5,   # 2: RKnee
    0.2,   # 3: RAnkle
    1.0,   # 4: LHip
    0.5,   # 5: LKnee
    0.2,   # 6: LAnkle
    1.0,   # 7: Spine
    1.0,   # 8: Thorax
    0.5,   # 9: Neck
    0.5,   # 10: LShoulder
    0.3,   # 11: LElbow
    0.1,   # 12: LWrist
    0.5,   # 13: RShoulder
    0.3,   # 14: RElbow
    0.1,   # 15: RWrist
], dtype=np.float64)

# Bone length regularisation: penalise deviation from initial MediaPipe-derived lengths
BONE_LENGTH_REG_WEIGHT = 10.0

# --- Heatmap / scoring ---
# Visibility threshold: joints with MediaPipe visibility below this are ignored
VISIBILITY_THRESHOLD = 0.5
