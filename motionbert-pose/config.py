"""Configuration for MotionBERT pose estimation pipeline."""

import os

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
TARGET_FPS: float = 10.0  # Subsample HD video (native ~30 fps) to this rate
