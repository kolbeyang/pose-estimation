"""Quick test of MediaPipe Pose Landmarker on a single image or webcam frame."""

import os
import sys
import cv2
import mediapipe as mp

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_lite.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

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

# Connections for drawing skeleton
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # shoulders/arms
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    (27, 29), (29, 31), (28, 30), (30, 32),  # feet
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
]


def detect_from_image(image_path: str):
    """Run pose detection on a single image file."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image.create_from_file(image_path)
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        print("No pose detected.")
        return

    # Print landmarks
    print(f"Detected {len(result.pose_landmarks)} pose(s)\n")
    for i, lm in enumerate(result.pose_landmarks[0]):
        name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"landmark_{i}"
        print(f"  {i:2d} {name:20s}  x={lm.x:.3f}  y={lm.y:.3f}  z={lm.z:.3f}  vis={lm.visibility:.2f}")

    if result.pose_world_landmarks:
        print("\nWorld landmarks (meters, hip-centered):")
        for i, lm in enumerate(result.pose_world_landmarks[0]):
            name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"landmark_{i}"
            print(f"  {i:2d} {name:20s}  x={lm.x:.4f}  y={lm.y:.4f}  z={lm.z:.4f}")

    # Draw on image
    frame = mp_image.numpy_view().copy()
    if frame.shape[2] == 4:  # RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    landmarks = result.pose_landmarks[0]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in CONNECTIONS:
        if landmarks[a].visibility > 0.5 and landmarks[b].visibility > 0.5:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    for i, (px, py) in enumerate(pts):
        if landmarks[i].visibility > 0.5:
            cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

    out_path = image_path.rsplit(".", 1)[0] + "_pose.jpg"
    cv2.imwrite(out_path, frame)
    print(f"\nSaved annotated image to: {out_path}")


def detect_from_webcam():
    """Run pose detection on webcam feed. Press 'q' to quit."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Webcam opened. Press 'q' to quit.")
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms <= 0:
                timestamp_ms = frame_idx * 33  # ~30fps fallback

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            h, w = frame.shape[:2]

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

                for a, b in CONNECTIONS:
                    if landmarks[a].visibility > 0.5 and landmarks[b].visibility > 0.5:
                        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for i, (px, py) in enumerate(pts):
                    if landmarks[i].visibility > 0.5:
                        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

            cv2.imshow("MediaPipe Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_from_image(sys.argv[1])
    else:
        detect_from_webcam()
