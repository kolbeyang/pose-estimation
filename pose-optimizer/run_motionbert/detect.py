"""YOLO + Stacked Hourglass + MotionBERT detection pipeline.

Adapted for the unified pose-optimizer.
"""

import copy
import logging
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Add parent dir to path for skeleton imports
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from skeleton import mpii_to_skeleton, mpii_to_motionbert_17, strip_nose_joint, NUM_JOINTS

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR: str = os.path.join(SCRIPT_DIR, "external")
CHECKPOINTS_DIR: str = os.path.join(SCRIPT_DIR, "checkpoints")

# MPII flip pairs for horizontal flip augmentation
MPII_FLIP_PAIRS: list[tuple[int, int]] = [
    (0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13),
]

# RGB channel means for Stacked Hourglass normalization
_SH_RGB_MEAN: np.ndarray = np.array([0.4404, 0.4440, 0.4327], dtype=np.float32)


@dataclass
class Detection2DModels:
    """Pre-loaded models for YOLO + Stacked Hourglass 2D detection only.

    Used by the shared heatmap generation step (both pipelines).
    """
    yolo: Any  # ultralytics.YOLO
    hourglass: torch.nn.Module
    device: torch.device


@dataclass
class MotionBertModels:
    """Pre-loaded models for the MotionBERT detection pipeline.

    Holds YOLO, Stacked Hourglass, and MotionBERT models so they can be
    loaded once and reused across multiple examples.
    """
    yolo: Any  # ultralytics.YOLO
    hourglass: torch.nn.Module
    motionbert: torch.nn.Module
    device: torch.device


def load_yolo_sh_models() -> Detection2DModels:
    """Load YOLO and Stacked Hourglass models for 2D heatmap generation.

    Returns:
        Detection2DModels container with YOLO + SH on the best device.
    """
    from ultralytics import YOLO
    from stacked_hourglass import hg8

    device = _get_device()

    # YOLO
    yolo = YOLO("yolov8n.pt")
    logger.info("Loaded YOLOv8n")

    # Stacked Hourglass
    try:
        hourglass = hg8(pretrained=True)
    except RuntimeError:
        hourglass = hg8(pretrained=False)
        cached_path = os.path.join(
            torch.hub.get_dir(), "checkpoints", "bearpaw_hg8-90e5d470.pth",
        )
        if os.path.exists(cached_path):
            state_dict = torch.load(cached_path, map_location="cpu", weights_only=False)
            hourglass.load_state_dict(state_dict)
        else:
            raise RuntimeError("Stacked Hourglass weights not found.")
    hourglass = hourglass.to(device)
    hourglass.eval()
    logger.info("Loaded Stacked Hourglass (8-stack, pretrained) on %s", device)

    return Detection2DModels(yolo=yolo, hourglass=hourglass, device=device)


def load_all_models() -> MotionBertModels:
    """Load YOLO, Stacked Hourglass, and MotionBERT models once.

    Returns:
        MotionBertModels container with all three models on the best
        available device.
    """
    from ultralytics import YOLO
    from stacked_hourglass import hg8

    device = _get_device()

    # YOLO
    yolo = YOLO("yolov8n.pt")
    logger.info("Loaded YOLOv8n")

    # Stacked Hourglass
    try:
        hourglass = hg8(pretrained=True)
    except RuntimeError:
        hourglass = hg8(pretrained=False)
        cached_path = os.path.join(
            torch.hub.get_dir(), "checkpoints", "bearpaw_hg8-90e5d470.pth",
        )
        if os.path.exists(cached_path):
            state_dict = torch.load(cached_path, map_location="cpu", weights_only=False)
            hourglass.load_state_dict(state_dict)
        else:
            raise RuntimeError("Stacked Hourglass weights not found.")
    hourglass = hourglass.to(device)
    hourglass.eval()
    logger.info("Loaded Stacked Hourglass (8-stack, pretrained) on %s", device)

    # MotionBERT
    motionbert = load_motionbert_model()
    motionbert = motionbert.to(device)

    logger.info("All models loaded on %s", device)
    return MotionBertModels(
        yolo=yolo, hourglass=hourglass, motionbert=motionbert, device=device,
    )


def _normalize_for_sh(cropped: np.ndarray) -> np.ndarray:
    """Normalize a cropped HWC uint8 image to CHW float32 for Stacked Hourglass.

    Converts (256, 256, 3) uint8 RGB to (3, 256, 256) float32 with mean subtracted.

    Args:
        cropped: (H, W, 3) uint8 RGB image.

    Returns:
        (3, H, W) float32 normalized image.
    """
    img = cropped.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img[0] -= _SH_RGB_MEAN[0]
    img[1] -= _SH_RGB_MEAN[1]
    img[2] -= _SH_RGB_MEAN[2]
    return img


def _get_device() -> torch.device:
    """Select best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# YOLO Person Detection
# ---------------------------------------------------------------------------

def detect_person_bbox(
    frames_rgb: list[np.ndarray],
    yolo: Any = None,
) -> np.ndarray:
    """Detect persons with YOLOv8, return union bounding box across all frames.

    Runs YOLOv8 person detection on each frame, selects the largest detection,
    and returns the union bounding box covering all frames.

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        yolo: Pre-loaded YOLO model.

    Returns:
        (4,) float32 array [x1, y1, x2, y2] in pixel coordinates.
    """

    h, w = frames_rgb[0].shape[:2]
    bboxes: list[np.ndarray] = []

    logger.info("Detecting persons in %d frames (%dx%d)...", len(frames_rgb), w, h)
    for frame in frames_rgb:
        results = yolo(frame, classes=[0], verbose=False)
        detections = results[0].boxes
        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
                detections.xyxy[:, 3] - detections.xyxy[:, 1]
            )
            best_idx = areas.argmax().item()
            bbox = detections.xyxy[best_idx].cpu().numpy().astype(np.float32)
            bboxes.append(bbox)
        else:
            bboxes.append(np.array([0, 0, w, h], dtype=np.float32))

    all_bboxes = np.array(bboxes)
    union_bbox = np.array([
        all_bboxes[:, 0].min(), all_bboxes[:, 1].min(),
        all_bboxes[:, 2].max(), all_bboxes[:, 3].max(),
    ], dtype=np.float32)

    logger.info(
        "Union bbox: (%.0f, %.0f) - (%.0f, %.0f)",
        union_bbox[0], union_bbox[1], union_bbox[2], union_bbox[3],
    )
    return union_bbox


def detect_person_bboxes_per_frame(
    frames_rgb: list[np.ndarray],
    yolo: Any = None,
) -> list[np.ndarray]:
    """Detect persons with YOLOv8, return per-frame bounding boxes.

    Runs YOLOv8 person detection on each frame, selects the largest detection,
    and returns individual bounding boxes (one per frame). If no detection is
    found for a frame, falls back to the full frame.

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        yolo: Pre-loaded YOLO model.

    Returns:
        List of (4,) float32 arrays [x1, y1, x2, y2] in pixel coordinates,
        one per frame.
    """
    h, w = frames_rgb[0].shape[:2]
    bboxes: list[np.ndarray] = []

    logger.info("Detecting per-frame bboxes in %d frames (%dx%d)...", len(frames_rgb), w, h)
    for frame in frames_rgb:
        results = yolo(frame, classes=[0], verbose=False)
        detections = results[0].boxes
        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
                detections.xyxy[:, 3] - detections.xyxy[:, 1]
            )
            best_idx = areas.argmax().item()
            bbox = detections.xyxy[best_idx].cpu().numpy().astype(np.float32)
            bboxes.append(bbox)
        else:
            bboxes.append(np.array([0, 0, w, h], dtype=np.float32))

    # Log bbox size statistics
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    logger.info(
        "Per-frame bboxes: min_area=%.0f, max_area=%.0f, median_area=%.0f",
        min(areas), max(areas), float(np.median(areas)),
    )
    return bboxes


# ---------------------------------------------------------------------------
# Crop and Resize
# ---------------------------------------------------------------------------

def crop_and_resize(
    frame: np.ndarray, bbox: np.ndarray, target_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop frame to bounding box with 20% padding and resize to square.

    Args:
        frame: (H, W, 3) uint8 RGB frame.
        bbox: (4,) float32 [x1, y1, x2, y2] bounding box in pixel coords.
        target_size: Output square size in pixels (default 256).

    Returns:
        Tuple of:
            resized: (target_size, target_size, 3) uint8 cropped+resized image.
            affine: (2, 3) float32 affine mapping crop coords (0..target_size-1)
                to original pixel coords.
    """
    x1, y1, x2, y2 = bbox
    cx_f = (x1 + x2) / 2
    cy_f = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.2
    half = side / 2

    crop_x1 = int(cx_f - half)
    crop_y1 = int(cy_f - half)
    crop_x2 = int(cx_f + half)
    crop_y2 = int(cy_f + half)

    h, w = frame.shape[:2]
    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)

    orig_crop_x1 = crop_x1
    orig_crop_y1 = crop_y1

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT
        )
        crop_x1 += pad_left
        crop_y1 += pad_top
        crop_x2 += pad_left
        crop_y2 += pad_top

    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    resized = cv2.resize(cropped, (target_size, target_size))

    scale_x = (crop_x2 - crop_x1) / target_size
    scale_y = (crop_y2 - crop_y1) / target_size
    affine = np.array([
        scale_x, 0, orig_crop_x1,
        0, scale_y, orig_crop_y1,
    ], dtype=np.float32).reshape(2, 3)

    return resized, affine


# ---------------------------------------------------------------------------
# Stacked Hourglass 2D Pose
# ---------------------------------------------------------------------------

def _flip_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Horizontally flip heatmaps and swap symmetric MPII joints.

    Args:
        heatmaps: (16, H, W) heatmaps in MPII order. [HEATMAP:MPII_16]

    Returns:
        (16, H, W) flipped heatmaps with L/R joints swapped. [HEATMAP:MPII_16]
    """
    flipped = heatmaps[:, :, ::-1].copy()
    for left, right in MPII_FLIP_PAIRS:
        flipped[left], flipped[right] = flipped[right].copy(), flipped[left].copy()
    return flipped


def _parse_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Extract 2D keypoint locations and confidence from heatmaps.

    Args:
        heatmaps: (16, H, W) heatmaps in MPII order. [HEATMAP:MPII_16]

    Returns:
        (16, 3) array of (x, y, confidence) per joint. [2D:MPII_16]
    """
    n_joints = heatmaps.shape[0]
    keypoints = np.zeros((n_joints, 3), dtype=np.float32)
    for j in range(n_joints):
        hm = heatmaps[j]
        idx = int(np.argmax(hm))
        y, x = np.unravel_index(idx, hm.shape)
        keypoints[j] = [x, y, float(hm[y, x])]
    return keypoints


def run_hourglass(
    frames_rgb: list[np.ndarray],
    bbox: np.ndarray | list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray | list[np.ndarray]]:
    """Run Stacked Hourglass (HG8) on cropped frames with flip augmentation.

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        bbox: Either a single (4,) float32 union bounding box [x1, y1, x2, y2]
            (shared across all frames), or a list of (4,) per-frame bboxes.
        model: Pre-loaded Stacked Hourglass model.
        device: Torch device for inference.
        batch_size: Inference batch size.

    Returns:
        Tuple of:
            all_keypoints_2d: List of (16, 3) per-frame. [2D:MPII_16]
                (x, y in original pixel coords, confidence).
            all_heatmaps: List of (16, 64, 64) per-frame. [HEATMAP:MPII_16]
            affine: If bbox was a single array, returns (2, 3) shared affine.
                If bbox was a list, returns list of (2, 3) per-frame affines.
    """
    per_frame = isinstance(bbox, list)

    # Preprocess all frames
    preprocessed: list[np.ndarray] = []
    preprocessed_flip: list[np.ndarray] = []
    all_affines: list[np.ndarray] = []

    for i, frame in enumerate(frames_rgb):
        frame_bbox = bbox[i] if per_frame else bbox
        cropped, affine = crop_and_resize(frame, frame_bbox, target_size=256)
        all_affines.append(affine)
        preprocessed.append(_normalize_for_sh(cropped))
        cropped_flip = cropped[:, ::-1].copy()
        preprocessed_flip.append(_normalize_for_sh(cropped_flip))

    n_frames = len(frames_rgb)

    # Batched inference
    all_heatmaps: list[np.ndarray] = []
    logger.info("Running Stacked Hourglass on %d frames (batch_size=%d)...", n_frames, batch_size)

    with torch.no_grad():
        for start in tqdm(range(0, n_frames, batch_size), desc="  2D Pose"):
            end = min(start + batch_size, n_frames)
            batch_imgs = preprocessed[start:end]
            batch_flips = preprocessed_flip[start:end]

            try:
                inp = torch.from_numpy(np.stack(batch_imgs)).to(device)
                output = model(inp)
                heatmaps_batch = output[-1].cpu().numpy()

                inp_flip = torch.from_numpy(np.stack(batch_flips)).to(device)
                output_flip = model(inp_flip)
                heatmaps_flip_batch = output_flip[-1].cpu().numpy()
            except RuntimeError as e:
                if device.type != "cpu":
                    logger.warning("%s failed (%s), falling back to CPU", device, e)
                    model = model.to("cpu")
                    device = torch.device("cpu")
                    inp = torch.from_numpy(np.stack(batch_imgs))
                    output = model(inp)
                    heatmaps_batch = output[-1].numpy()
                    inp_flip = torch.from_numpy(np.stack(batch_flips))
                    output_flip = model(inp_flip)
                    heatmaps_flip_batch = output_flip[-1].numpy()
                else:
                    raise

            for i in range(end - start):
                hm_flip = _flip_heatmaps(heatmaps_flip_batch[i])
                hm_avg = (heatmaps_batch[i] + hm_flip) / 2.0
                all_heatmaps.append(hm_avg)

    # Parse keypoints — each frame uses its own affine
    all_keypoints_2d: list[np.ndarray] = []
    for frame_idx, heatmaps in enumerate(all_heatmaps):
        keypoints_64 = _parse_heatmaps(heatmaps)
        keypoints_orig = keypoints_64.copy()
        keypoints_orig[:, :2] *= 4  # 64 -> 256
        frame_affine = all_affines[frame_idx]
        for j in range(16):
            x_256 = keypoints_orig[j, 0]
            y_256 = keypoints_orig[j, 1]
            keypoints_orig[j, 0] = frame_affine[0, 0] * x_256 + frame_affine[0, 2]
            keypoints_orig[j, 1] = frame_affine[1, 1] * y_256 + frame_affine[1, 2]
        all_keypoints_2d.append(keypoints_orig)

    # Return format matches input: single affine for single bbox, list for per-frame
    if per_frame:
        return all_keypoints_2d, all_heatmaps, all_affines
    else:
        return all_keypoints_2d, all_heatmaps, all_affines[0]


# ---------------------------------------------------------------------------
# MotionBERT 3D Lifting
# ---------------------------------------------------------------------------

def _find_motionbert_path() -> str:
    """Find MotionBERT installation directory under external/."""
    local_path = os.path.join(EXTERNAL_DIR, "MotionBERT")
    if os.path.exists(local_path):
        return local_path
    raise RuntimeError(
        f"MotionBERT not found at {local_path}. "
        "Run setup_models.py first."
    )


def _find_motionbert_checkpoint() -> str:
    """Find MotionBERT-Lite checkpoint file under checkpoints/."""
    local_ckpt = os.path.join(CHECKPOINTS_DIR, "motionbert_lite_h36m.bin")
    if os.path.exists(local_ckpt):
        return local_ckpt
    raise RuntimeError(
        f"MotionBERT checkpoint not found at {local_ckpt}. "
        "Run setup_models.py first."
    )


def load_motionbert_model() -> torch.nn.Module:
    """Load pretrained MotionBERT-Lite DSTformer model for 2D->3D lifting.

    Returns:
        MotionBERT-Lite model in eval mode (expects 17-joint input).
    """
    motionbert_path = _find_motionbert_path()

    lib_path = os.path.join(motionbert_path, "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    if motionbert_path not in sys.path:
        sys.path.insert(0, motionbert_path)

    from lib.model.DSTformer import DSTformer  # noqa: E402

    lite_ckpt = _find_motionbert_checkpoint()

    model = DSTformer(
        dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    checkpoint = torch.load(lite_ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get(
        "model_pos", checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logger.info("Loaded MotionBERT-Lite (global)")
    return model


def crop_scale(motion: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize 2D keypoints to [-1, 1] for MotionBERT input.

    Args:
        motion: (N, 17, 3) 2D keypoints with confidence. [2D:MPII_17]

    Returns:
        Tuple of:
            normalized: (N, 17, 3) keypoints in [-1, 1] range.
            params: Dict with 'xs', 'ys', 'scale' for denormalization.
    """
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape, dtype=motion.dtype), {
            "xs": 0.0, "ys": 0.0, "scale": 1.0,
        }
    xmin, xmax = float(valid_coords[:, 0].min()), float(valid_coords[:, 0].max())
    ymin, ymax = float(valid_coords[:, 1].min()), float(valid_coords[:, 1].max())
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape, dtype=motion.dtype), {
            "xs": 0.0, "ys": 0.0, "scale": 1.0,
        }
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result, {"xs": xs, "ys": ys, "scale": scale}


def flip_data(data):
    """Flip 17-joint data: negate X, swap L/R joints.

    Operates on [2D:MPII_17] or [3D:MOTIONBERT_17] 17-joint data for
    horizontal flip augmentation.

    Args:
        data: (..., 17, D) array of 17-joint keypoints.

    Returns:
        (..., 17, D) horizontally flipped data with L/R swapped.
    """
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[
        ..., right_joints + left_joints, :
    ]
    return flipped_data


def run_motionbert(
    keypoints_2d_list: list[np.ndarray],
    model: torch.nn.Module,
    device: torch.device,
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """Lift 2D keypoints to 3D using MotionBERT.

    Converts [2D:MPII_16] inputs to 17-joint format via mpii_to_skeleton(),
    runs MotionBERT 3D lifting, then strips head joint to produce
    [3D:SKELETON_16] output.

    Args:
        keypoints_2d_list: List of (16, 3) MPII keypoints per frame. [2D:MPII_16]
        model: Pre-loaded MotionBERT model.
        device: Torch device for inference.
        conf_threshold: Zero out joints below this confidence.

    Returns:
        (N, 16, 3) normalized MotionBERT output. [3D:SKELETON_16]
    """
    logger.info("MotionBERT on %s", device)

    n_frames = len(keypoints_2d_list)

    # Convert MPII 16-joint to 17-joint MotionBERT input format
    keypoints_17 = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for i, kp_mpii in enumerate(keypoints_2d_list):
        keypoints_17[i] = mpii_to_motionbert_17(kp_mpii)

    # Zero out low-confidence joints
    n_zeroed = 0
    if conf_threshold > 0.0:
        for i in range(n_frames):
            for j in range(17):
                if keypoints_17[i, j, 2] < conf_threshold:
                    keypoints_17[i, j, :] = 0.0
                    n_zeroed += 1
    n_total = n_frames * 17
    logger.info(
        "Confidence threshold=%.2f: zeroed %d/%d (%.1f%%)",
        conf_threshold, n_zeroed, n_total, 100 * n_zeroed / n_total,
    )

    keypoints_norm, cs_params = crop_scale(keypoints_17)
    logger.info(
        "crop_scale: scale=%.1f offset=(%.1f, %.1f)",
        cs_params['scale'], cs_params['xs'], cs_params['ys'],
    )

    clip_len = 243

    if n_frames <= clip_len:
        # Single pass — fits in one window
        input_tensor = torch.from_numpy(keypoints_norm).unsqueeze(0).to(device)
        logger.info("Running MotionBERT on %d frames...", n_frames)

        with torch.no_grad():
            try:
                output_3d = model(input_tensor)
            except RuntimeError as e:
                if device.type != "cpu":
                    logger.warning("%s failed (%s), falling back to CPU", device, e)
                    model = model.to("cpu")
                    input_tensor = input_tensor.to("cpu")
                    output_3d = model(input_tensor)
                else:
                    raise

        positions_3d = output_3d.cpu().numpy()[0]  # (N, 17, 3)
    else:
        # Sliding window — process clip_len frames at a time, stride clip_len
        # (no overlap for simplicity; last window is right-aligned)
        chunks: list[np.ndarray] = []
        start = 0
        while start < n_frames:
            end = min(start + clip_len, n_frames)
            # If the remaining chunk is smaller than clip_len, right-align the window
            if end - start < clip_len:
                window_start = max(0, end - clip_len)
                window = keypoints_norm[window_start:end]
                keep_from = start - window_start  # how many frames to discard from front
            else:
                window = keypoints_norm[start:end]
                keep_from = 0

            input_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            with torch.no_grad():
                try:
                    chunk_3d = model(input_tensor)
                except RuntimeError as e:
                    if device.type != "cpu":
                        logger.warning("%s failed (%s), falling back to CPU", device, e)
                        model = model.to("cpu")
                        input_tensor = input_tensor.to("cpu")
                        chunk_3d = model(input_tensor)
                    else:
                        raise

            chunk_np = chunk_3d.cpu().numpy()[0]  # (clip_len, 17, 3)
            chunks.append(chunk_np[keep_from:])
            start = end

        n_windows = len(chunks)
        logger.info("Running MotionBERT on %d frames (%d windows of %d)...",
                     n_frames, n_windows, clip_len)
        positions_3d = np.concatenate(chunks, axis=0)  # (n_frames, 17, 3)

    positions_3d[0, 0, 2] = 0

    logger.info("MotionBERT raw output: %s", positions_3d.shape)
    logger.info(
        "Root Z range (norm): %.4f to %.4f",
        positions_3d[:, 0, 2].min(), positions_3d[:, 0, 2].max(),
    )

    # Strip Nose joint (index 9) -> 16 joints, keeping HeadTop (index 10 becomes 9)
    positions_3d = strip_nose_joint(positions_3d)  # [3D:SKELETON_16]
    return positions_3d


# ---------------------------------------------------------------------------
# Denormalization to Camera-Space Meters
# ---------------------------------------------------------------------------

def _iqr_filtered_median(values: np.ndarray, k: float = 1.5) -> float:
    """Compute median after removing IQR outliers.

    Args:
        values: 1D array of numeric values.
        k: IQR multiplier for outlier bounds (default 1.5).

    Returns:
        Filtered median as a float.
    """
    if len(values) < 4:
        return float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    inliers = values[(values >= lower) & (values <= upper)]
    if len(inliers) < 2:
        return float(np.median(values))
    return float(np.median(inliers))


_RELIABLE_BONES_FOR_SCALE: set[int] = {10, 11, 12, 13, 14, 15}


def motionbert_to_camera_space(
    positions_3d_norm: np.ndarray,
    kp_2d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Convert MotionBERT normalized output to camera-space meters.

    Two-step process:
    1. Bone-length scale estimation: MotionBERT outputs arbitrary normalized
       units. Compares detected bone lengths to DEFAULT_BONE_LENGTHS to recover
       a scale factor that converts to meters.
    2. Depth estimation: For each joint pair, uses the pinhole camera equation
       tz = fy * dy_3d / dv_2d to estimate root depth from Y-axis correspondences.
       Takes the IQR-filtered median of all such estimates, then back-projects
       root XY from the 2D hip pixel position.

    Args:
        positions_3d_norm: (16, 3) MotionBERT-normalized positions. [3D:SKELETON_16]
        kp_2d: (16, 2) 2D pixel coordinates. [2D:SKELETON_16]
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (16, 3) positions in camera coordinates (meters). [3D:SKELETON_16]
    """
    from skeleton import PARENTS, DEFAULT_BONE_LENGTHS

    # Step 1: Bone-length-based scale estimation (MotionBERT output is not in
    # real-world units, so we recover meters from bone length ratios).
    root_relative = positions_3d_norm - positions_3d_norm[0:1]
    reliable_ratios: list[float] = []
    all_ratios: list[float] = []
    for j in range(1, NUM_JOINTS):
        p = int(PARENTS[j])
        det_bl = float(np.linalg.norm(root_relative[j] - root_relative[p]))
        ref_bl = float(DEFAULT_BONE_LENGTHS[j])
        if det_bl > 1e-4 and ref_bl > 1e-4:
            ratio = ref_bl / det_bl
            all_ratios.append(ratio)
            if j in _RELIABLE_BONES_FOR_SCALE:
                reliable_ratios.append(ratio)

    if len(reliable_ratios) >= 4:
        bone_scale = _iqr_filtered_median(np.array(reliable_ratios))
    elif all_ratios:
        bone_scale = _iqr_filtered_median(np.array(all_ratios))
    else:
        bone_scale = 1.0

    root_relative_m = root_relative * bone_scale

    # Step 2: Estimate root depth from pairwise Y-axis correspondences.
    # For each joint pair (i, j), tz = fy * |dy_3d| / |dv_2d| by the
    # pinhole camera model. Take IQR-filtered median of all estimates.
    tz_candidates: list[float] = []
    for i in range(NUM_JOINTS):
        for j in range(NUM_JOINTS):
            if i == j:
                continue
            if np.linalg.norm(kp_2d[i]) <= 1.0 or np.linalg.norm(kp_2d[j]) <= 1.0:
                continue
            dy_3d = abs(float(root_relative_m[i, 1]) - float(root_relative_m[j, 1]))
            if dy_3d < 0.001:
                continue
            dv_2d = abs(float(kp_2d[i, 1]) - float(kp_2d[j, 1]))
            if dv_2d > 5.0:
                tz_candidates.append(fy * dy_3d / dv_2d)

    if len(tz_candidates) >= 2:
        tz = _iqr_filtered_median(np.array(tz_candidates))
        tz = float(np.clip(tz, 1.0, 8.0))
    else:
        tz = 3.0

    # Back-project root XY from 2D hip pixel position
    u_root = float(kp_2d[0, 0])
    v_root = float(kp_2d[0, 1])
    tx = (u_root - cx) * tz / fx
    ty = (v_root - cy) * tz / fy

    cam_3d = root_relative_m.copy()
    cam_3d[:, 0] += tx
    cam_3d[:, 1] += ty
    cam_3d[:, 2] += tz
    return cam_3d.astype(np.float64)


# ---------------------------------------------------------------------------
# Pipeline Wrapper
# ---------------------------------------------------------------------------

def detect_2d_poses(
    frames_rgb: list[np.ndarray],
    models: Detection2DModels,
    sh_batch_size: int = 32,
    per_frame_bbox: bool = False,
) -> tuple[
    list[np.ndarray],                # kp_2d (16, 2) pixel coords [2D:SKELETON_16]
    list[np.ndarray],                # visibility (16,) [VIS:SKELETON_16]
    list[np.ndarray],                # heatmaps (16, 64, 64) [HEATMAP:MPII_16]
    list[np.ndarray],                # mpii_kp_2d (16, 3) raw MPII [2D:MPII_16]
    np.ndarray | list[np.ndarray],   # affine: (2,3) or list of (2,3)
]:
    """Run YOLO + Stacked Hourglass for 2D heatmap generation.

    Shared between both MotionBERT and MediaPipe pipelines.

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        models: Pre-loaded Detection2DModels from load_yolo_sh_models().
        sh_batch_size: Batch size for Stacked Hourglass inference.
        per_frame_bbox: If True, use per-frame bounding boxes instead of a
            single union bbox. This gives better SH resolution when the person
            moves significantly across frames.

    Returns:
        Tuple of:
            kp_2d: List of (16, 2) pixel coordinates. [2D:SKELETON_16]
            visibility: List of (16,) confidence scores. [VIS:SKELETON_16]
            heatmaps: List of (16, 64, 64) SH heatmaps. [HEATMAP:MPII_16]
            mpii_kp_2d: List of (16, 3) MPII 2D keypoints with confidence. [2D:MPII_16]
            affine: (2, 3) shared affine when per_frame_bbox=False,
                or list of (2, 3) per-frame affines when per_frame_bbox=True.
    """
    # 1. YOLO person detection
    if per_frame_bbox:
        bboxes = detect_person_bboxes_per_frame(frames_rgb, yolo=models.yolo)
    else:
        bboxes = detect_person_bbox(frames_rgb, yolo=models.yolo)

    # 2. Stacked Hourglass
    all_keypoints_2d, all_heatmaps, affine = run_hourglass(
        frames_rgb, bboxes,
        model=models.hourglass,
        device=models.device,
        batch_size=sh_batch_size,
    )

    # 3. Convert MPII 2D to skeleton 2D
    kp_2d_list: list[np.ndarray] = []  # list of [2D:SKELETON_16] (16, 2)
    visibility_list: list[np.ndarray] = []  # list of [VIS:SKELETON_16] (16,)
    for kp_mpii in all_keypoints_2d:
        kp_16 = mpii_to_skeleton(kp_mpii)  # [2D:SKELETON_16]
        kp_2d_list.append(kp_16[:, :2])
        visibility_list.append(kp_16[:, 2])

    return (
        kp_2d_list,
        visibility_list,
        all_heatmaps,
        all_keypoints_2d,
        affine,
    )


def detect_poses(
    frames_rgb: list[np.ndarray],
    models: MotionBertModels,
    sh_batch_size: int = 32,
    conf_threshold: float = 0.0,
) -> tuple[
    list[np.ndarray],  # kp_2d (16, 2) pixel coords [2D:SKELETON_16]
    list[np.ndarray],  # visibility (16,) [VIS:SKELETON_16]
    list[np.ndarray],  # heatmaps (16, 64, 64) [HEATMAP:MPII_16]
    list[np.ndarray],  # mpii_kp_2d (16, 3) raw MPII [2D:MPII_16]
    np.ndarray,        # affine (2, 3)
    np.ndarray,        # positions_3d_norm (N, 16, 3) [3D:SKELETON_16]
]:
    """Full detection pipeline: YOLO -> Stacked Hourglass -> MotionBERT.

    Runs the complete MotionBERT detection pipeline and converts all outputs
    from MPII format to the optimizer's 16-joint skeleton format.

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        models: Pre-loaded MotionBertModels container from load_all_models().
        sh_batch_size: Batch size for Stacked Hourglass inference.
        conf_threshold: MotionBERT confidence threshold.

    Returns:
        Tuple of:
            kp_2d: List of (16, 2) pixel coordinates. [2D:SKELETON_16]
            visibility: List of (16,) confidence scores. [VIS:SKELETON_16]
            heatmaps: List of (16, 64, 64) heatmaps. [HEATMAP:MPII_16]
            mpii_kp_2d: List of (16, 3) raw MPII keypoints. [2D:MPII_16]
            affine: (2, 3) affine mapping crop to pixel coords.
            positions_3d_norm: (N, 16, 3) MotionBERT normalized. [3D:SKELETON_16]
    """
    # 1+2. YOLO + Stacked Hourglass (shared 2D detection)
    d2d_models = Detection2DModels(
        yolo=models.yolo, hourglass=models.hourglass, device=models.device,
    )
    kp_2d_list, visibility_list, all_heatmaps, all_keypoints_2d, affine = detect_2d_poses(
        frames_rgb, d2d_models, sh_batch_size=sh_batch_size,
    )

    # 3. MotionBERT
    positions_3d_norm = run_motionbert(
        all_keypoints_2d,
        model=models.motionbert,
        device=models.device,
        conf_threshold=conf_threshold,
    )

    return (
        kp_2d_list,
        visibility_list,
        all_heatmaps,
        all_keypoints_2d,
        affine,
        positions_3d_norm,
    )
