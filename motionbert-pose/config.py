"""Configuration for full-body FK pose optimization pipeline."""

import os

# --- Paths ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PANOPTIC_ROOT = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "panoptic-toolbox")
)
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")

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
    ("160422_ultimatum1", "00_00", 20000, 150, 0),
    ("171204_pose2", "00_00", 25000, 150, 0),
]

# --- Video processing ---
TARGET_FPS = 10.0  # Subsample HD video (native ~30 fps) to this rate

# --- Optimization ---
NUM_STEPS = 150
LEARNING_RATE = 0.0008
BONE_LENGTH_LR = 0.002
GRAD_CLIP_NORM = 5.0

# Analytical Gaussian sigma schedule (coarse→fine): (fraction_of_steps, sigma_pixels)
BLUR_PHASES = [
    (0.40, 60.0),
    (0.70, 25.0),
    (1.00, 10.0),
]

# Motion penalty weights
POSITION_PENALTY_WEIGHT = 8.0
ROTATION_PENALTY_WEIGHT = 5.0

# Bone length regularisation: penalise deviation from initial MediaPipe-derived lengths
BONE_LENGTH_REG_WEIGHT = 50.0

# --- Heatmap / scoring ---
# Visibility threshold: joints with MediaPipe visibility below this are ignored
VISIBILITY_THRESHOLD = 0.5
