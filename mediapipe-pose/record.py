"""Webcam recording with MediaPipe pose landmark extraction."""

import os
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# MediaPipe landmark indices for left arm
SHOULDER_IDX = 11
ELBOW_IDX = 13
WRIST_IDX = 15
ARM_INDICES = {"a": SHOULDER_IDX, "b": ELBOW_IDX, "c": WRIST_IDX}

# Skeleton connections for visualization
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
]

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


@dataclass
class FrameData:
    landmarks_2d: np.ndarray      # (33, 2) normalized [0,1]
    landmarks_3d: np.ndarray      # (33, 3) world coords in meters
    visibility: np.ndarray         # (33,) scores
    timestamp_ms: int


@dataclass
class Recording:
    frames: list[FrameData]
    image_size: tuple[int, int]   # (height, width)


def record_webcam(target_fps: float = 10.0) -> Recording:
    """
    Record webcam with MediaPipe pose detection.

    Shows live preview with skeleton overlay. Press 'q' to stop.

    Args:
        target_fps: Target capture rate (default 10fps)

    Returns:
        Recording with all frame data
    """
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    frames: list[FrameData] = []
    frame_interval = 1.0 / target_fps
    last_capture_time = 0.0
    timestamp_ms = 0
    image_size = None

    print(f"Recording at ~{target_fps} fps. Press 'q' to stop.")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            if image_size is None:
                image_size = (h, w)

            current_time = time.time()

            # Pace captures at target fps
            if current_time - last_capture_time >= frame_interval:
                last_capture_time = current_time
                timestamp_ms += int(frame_interval * 1000)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and result.pose_world_landmarks:
                    lm_2d = result.pose_landmarks[0]
                    lm_3d = result.pose_world_landmarks[0]

                    landmarks_2d = np.array([[l.x, l.y] for l in lm_2d], dtype=np.float32)
                    landmarks_3d = np.array([[l.x, l.y, l.z] for l in lm_3d], dtype=np.float32)
                    visibility = np.array([l.visibility for l in lm_2d], dtype=np.float32)

                    frames.append(FrameData(
                        landmarks_2d=landmarks_2d,
                        landmarks_3d=landmarks_3d,
                        visibility=visibility,
                        timestamp_ms=timestamp_ms,
                    ))

                    # Draw skeleton on preview
                    pts = [(int(l.x * w), int(l.y * h)) for l in lm_2d]
                    for a, b in CONNECTIONS:
                        if lm_2d[a].visibility > 0.5 and lm_2d[b].visibility > 0.5:
                            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                    for i, (px, py) in enumerate(pts):
                        if lm_2d[i].visibility > 0.5:
                            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

            # Show frame count
            cv2.putText(frame, f"Frames: {len(frames)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No frames captured")

    print(f"Recorded {len(frames)} frames at {image_size[1]}x{image_size[0]}")
    return Recording(frames=frames, image_size=image_size)
