"""2D pose estimation (Stacked Hourglass) and 3D lifting (MotionBERT).

Handles person detection (YOLOv8), 2D heatmap extraction, and 2D->3D lifting.
"""

import copy
import os
import sys
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config as cfg
from skeleton import mpii_to_h36m

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR: str = os.path.join(SCRIPT_DIR, "external")
CHECKPOINTS_DIR: str = os.path.join(SCRIPT_DIR, "checkpoints")

# --- MPII flip pairs for horizontal flip augmentation ---
MPII_FLIP_PAIRS: list[tuple[int, int]] = [
    (0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)
]


# ---------------------------------------------------------------------------
# YOLO Person Detection
# ---------------------------------------------------------------------------

def detect_person_bbox(frames_rgb: list[np.ndarray]) -> np.ndarray:
    """Detect persons with YOLOv8, return union bounding box.

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.

    Returns:
        (4,) array [x1, y1, x2, y2] -- union of largest-person bbox across frames.
    """
    from ultralytics import YOLO

    h: int
    w: int
    h, w = frames_rgb[0].shape[:2]
    yolo: YOLO = YOLO("yolov8n.pt")
    bboxes: list[np.ndarray] = []

    print(f"  Detecting persons in {len(frames_rgb)} frames ({w}x{h})...")

    for frame in frames_rgb:
        results = yolo(frame, classes=[0], verbose=False)
        detections = results[0].boxes

        if len(detections) > 0:
            areas: torch.Tensor = (
                (detections.xyxy[:, 2] - detections.xyxy[:, 0])
                * (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            )
            best_idx: int = areas.argmax().item()
            bbox: np.ndarray = detections.xyxy[best_idx].cpu().numpy().astype(np.float32)
            bboxes.append(bbox)
        else:
            bboxes.append(np.array([0, 0, w, h], dtype=np.float32))

    # Union (max) bounding box across all frames
    all_bboxes: np.ndarray = np.array(bboxes)
    union_bbox: np.ndarray = np.array([
        all_bboxes[:, 0].min(),
        all_bboxes[:, 1].min(),
        all_bboxes[:, 2].max(),
        all_bboxes[:, 3].max(),
    ], dtype=np.float32)

    print(f"  Union bbox: ({union_bbox[0]:.0f}, {union_bbox[1]:.0f}) - "
          f"({union_bbox[2]:.0f}, {union_bbox[3]:.0f})")

    return union_bbox


# ---------------------------------------------------------------------------
# Crop and Resize
# ---------------------------------------------------------------------------

def crop_and_resize(
    frame: np.ndarray,
    bbox: np.ndarray,
    target_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop frame to bounding box with 20% padding and resize to square.

    Args:
        frame: (H, W, 3) RGB image.
        bbox: (4,) array [x1, y1, x2, y2].
        target_size: Output size (square).

    Returns:
        (cropped_resized, affine_transform) where affine maps
        target_size coords -> original pixel coords. Shape (2, 3).
    """
    x1: float
    y1: float
    x2: float
    y2: float
    x1, y1, x2, y2 = bbox
    cx_f: float = (x1 + x2) / 2
    cy_f: float = (y1 + y2) / 2
    side: float = max(x2 - x1, y2 - y1) * 1.2  # 20% padding
    half: float = side / 2

    crop_x1: int = int(cx_f - half)
    crop_y1: int = int(cy_f - half)
    crop_x2: int = int(cx_f + half)
    crop_y2: int = int(cy_f + half)

    h: int
    w: int
    h, w = frame.shape[:2]

    pad_left: int = max(0, -crop_x1)
    pad_top: int = max(0, -crop_y1)
    pad_right: int = max(0, crop_x2 - w)
    pad_bottom: int = max(0, crop_y2 - h)

    orig_crop_x1: int = crop_x1
    orig_crop_y1: int = crop_y1

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        frame = cv2.copyMakeBorder(
            frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT
        )
        crop_x1 += pad_left
        crop_y1 += pad_top
        crop_x2 += pad_left
        crop_y2 += pad_top

    cropped: np.ndarray = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    resized: np.ndarray = cv2.resize(cropped, (target_size, target_size))

    # Affine: maps from target_size coords to original frame coords
    scale_x: float = (crop_x2 - crop_x1) / target_size
    scale_y: float = (crop_y2 - crop_y1) / target_size
    affine: np.ndarray = np.array([
        scale_x, 0, orig_crop_x1,
        0, scale_y, orig_crop_y1,
    ], dtype=np.float32).reshape(2, 3)

    return resized, affine


# ---------------------------------------------------------------------------
# Stacked Hourglass 2D Pose
# ---------------------------------------------------------------------------

def _flip_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Horizontally flip heatmaps and swap symmetric joints.

    Args:
        heatmaps: (16, 64, 64) heatmap array.

    Returns:
        (16, 64, 64) flipped heatmaps.
    """
    flipped: np.ndarray = heatmaps[:, :, ::-1].copy()
    for left, right in MPII_FLIP_PAIRS:
        flipped[left], flipped[right] = flipped[right].copy(), flipped[left].copy()
    return flipped


def _parse_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Extract 2D keypoint locations from heatmaps.

    Args:
        heatmaps: (16, 64, 64) heatmap array.

    Returns:
        (16, 3) array of (x, y, confidence) in 64x64 heatmap space.
    """
    n_joints: int = heatmaps.shape[0]
    keypoints: np.ndarray = np.zeros((n_joints, 3), dtype=np.float32)

    for j in range(n_joints):
        hm: np.ndarray = heatmaps[j]
        idx: int = int(np.argmax(hm))
        y: int
        x: int
        y, x = np.unravel_index(idx, hm.shape)
        confidence: float = float(hm[y, x])
        keypoints[j] = [x, y, confidence]

    return keypoints


def run_hourglass(
    frames_rgb: list[np.ndarray],
    bbox: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Run HG8 on cropped frames with flip augmentation.

    Args:
        frames_rgb: RGB frames (H, W, 3).
        bbox: (4,) union bounding box.

    Returns:
        keypoints_2d: List of (16, 3) in original pixel coords (x, y, confidence).
        heatmaps: List of (16, 64, 64) raw heatmaps.
        affine: (2, 3) affine from 256-crop -> original pixel coords.
    """
    from stacked_hourglass import hg8

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model - handle CPU-only machines
    try:
        model: torch.nn.Module = hg8(pretrained=True)
    except RuntimeError:
        model = hg8(pretrained=False)
        cached_path: str = os.path.join(
            torch.hub.get_dir(), "checkpoints", "bearpaw_hg8-90e5d470.pth"
        )
        if os.path.exists(cached_path):
            state_dict: dict = torch.load(
                cached_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(
                "Stacked Hourglass weights not found. Try downloading manually."
            )

    model = model.to(device)
    model.eval()
    print("  Loaded Stacked Hourglass (8-stack, pretrained)")

    all_keypoints_2d: list[np.ndarray] = []
    all_heatmaps: list[np.ndarray] = []
    shared_affine: np.ndarray | None = None

    print(f"  Running Stacked Hourglass on {len(frames_rgb)} frames...")

    with torch.no_grad():
        for frame in tqdm(frames_rgb, desc="  2D Pose"):
            # Crop and resize to 256x256
            cropped: np.ndarray
            affine: np.ndarray
            cropped, affine = crop_and_resize(frame, bbox, target_size=256)
            if shared_affine is None:
                shared_affine = affine

            # Preprocess: normalize to [0, 1], subtract RGB mean, HWC -> CHW
            # RGB channel means match the official HumanPosePredictor preprocessing
            img: np.ndarray = cropped.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img[0] -= 0.4404  # R mean
            img[1] -= 0.4440  # G mean
            img[2] -= 0.4327  # B mean
            inp: torch.Tensor = torch.from_numpy(img).unsqueeze(0).to(device)

            # Forward pass - model returns list of heatmaps per stack
            output: list[torch.Tensor] = model(inp)
            heatmaps: np.ndarray = output[-1].cpu().numpy()[0]  # (16, 64, 64)

            # Flip augmentation
            cropped_flip: np.ndarray = cropped[:, ::-1].copy()
            img_flip: np.ndarray = cropped_flip.astype(np.float32) / 255.0
            img_flip = np.transpose(img_flip, (2, 0, 1))  # HWC -> CHW
            img_flip[0] -= 0.4404  # R mean
            img_flip[1] -= 0.4440  # G mean
            img_flip[2] -= 0.4327  # B mean
            inp_flip: torch.Tensor = torch.from_numpy(img_flip).unsqueeze(0).to(device)
            output_flip: list[torch.Tensor] = model(inp_flip)
            heatmaps_flip: np.ndarray = output_flip[-1].cpu().numpy()[0]
            heatmaps_flip = _flip_heatmaps(heatmaps_flip)

            # Average original + flipped
            heatmaps = (heatmaps + heatmaps_flip) / 2.0
            all_heatmaps.append(heatmaps)

            # Parse keypoints in 64x64 space
            keypoints_64: np.ndarray = _parse_heatmaps(heatmaps)  # (16, 3)

            # Scale keypoints: 64 -> 256 -> original image coords
            keypoints_orig: np.ndarray = keypoints_64.copy()
            keypoints_orig[:, :2] *= 4  # 64 -> 256
            for j in range(16):
                x_256: float = keypoints_orig[j, 0]
                y_256: float = keypoints_orig[j, 1]
                keypoints_orig[j, 0] = shared_affine[0, 0] * x_256 + shared_affine[0, 2]
                keypoints_orig[j, 1] = shared_affine[1, 1] * y_256 + shared_affine[1, 2]

            all_keypoints_2d.append(keypoints_orig)

    assert shared_affine is not None
    return all_keypoints_2d, all_heatmaps, shared_affine


# ---------------------------------------------------------------------------
# MotionBERT 3D Lifting
# ---------------------------------------------------------------------------

def load_motionbert_model() -> torch.nn.Module:
    """Load pretrained MotionBERT-Lite model for 3D pose estimation.

    Returns:
        DSTformer model in eval mode on CPU.
    """
    motionbert_path: str = os.path.join(EXTERNAL_DIR, "MotionBERT")
    if not os.path.exists(motionbert_path):
        raise RuntimeError(
            f"MotionBERT not found at {motionbert_path}. Run setup_models.py first."
        )

    # Add to path for imports
    lib_path: str = os.path.join(motionbert_path, "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    if motionbert_path not in sys.path:
        sys.path.insert(0, motionbert_path)

    from lib.model.DSTformer import DSTformer  # noqa: E402

    lite_ckpt: str = os.path.join(CHECKPOINTS_DIR, "motionbert_lite_h36m.bin")

    model: torch.nn.Module = DSTformer(
        dim_in=3, dim_out=3,
        dim_feat=256, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=4,
        num_joints=17, maxlen=243,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if not os.path.exists(lite_ckpt):
        raise RuntimeError(
            f"MotionBERT checkpoint not found at {lite_ckpt}. Run setup_models.py first."
        )

    checkpoint: dict = torch.load(lite_ckpt, map_location="cpu", weights_only=False)
    state_dict: dict = checkpoint.get(
        "model_pos", checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("  Loaded MotionBERT-Lite (global, H3.6M)")
    return model


def crop_scale(motion: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize 2D keypoints to [-1, 1] based on bounding box.

    Matches the official MotionBERT crop_scale() from lib/utils/utils_data.py.

    Args:
        motion: (N, 17, 3) array of keypoints with confidence in last dim.

    Returns:
        (normalized_motion, params) where params has keys xs, ys, scale.
    """
    result: np.ndarray = copy.deepcopy(motion)
    valid_coords: np.ndarray = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape, dtype=motion.dtype), {
            "xs": 0.0, "ys": 0.0, "scale": 1.0
        }
    xmin: float = float(valid_coords[:, 0].min())
    xmax: float = float(valid_coords[:, 0].max())
    ymin: float = float(valid_coords[:, 1].min())
    ymax: float = float(valid_coords[:, 1].max())
    scale: float = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape, dtype=motion.dtype), {
            "xs": 0.0, "ys": 0.0, "scale": 1.0
        }
    xs: float = (xmin + xmax - scale) / 2
    ys: float = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result, {"xs": xs, "ys": ys, "scale": scale}


def flip_data(data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Flip H36M 17-joint data: negate X, swap L/R joints.

    Args:
        data: (..., 17, D) array or tensor.

    Returns:
        Flipped data of same type and shape.
    """
    left_joints: list[int] = [4, 5, 6, 11, 12, 13]
    right_joints: list[int] = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = (
        flipped_data[..., right_joints + left_joints, :]
    )
    return flipped_data


def run_motionbert(
    keypoints_2d_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Lift 2D keypoints to 3D using MotionBERT.

    Args:
        keypoints_2d_list: List of (16, 3) MPII keypoints per frame (x, y, conf).
        image_size: (height, width) of original frames.

    Returns:
        Tuple of:
            positions_3d_pixel: (N, 17, 3) pixel-aligned 3D joint positions (H36M).
            positions_3d_norm: (N, 17, 3) normalized 3D output (before denorm).
            cs_params: crop_scale parameters dict with keys xs, ys, scale.
    """
    model: torch.nn.Module = load_motionbert_model()
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    n_frames: int = len(keypoints_2d_list)

    # Convert MPII 16-joint to H36M 17-joint
    keypoints_h36m: np.ndarray = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for i, kp_mpii in enumerate(keypoints_2d_list):
        keypoints_h36m[i] = mpii_to_h36m(kp_mpii)

    # Zero out low-confidence joints so MotionBERT treats them as missing.
    # This prevents garbage 2D detections (e.g. ankles with conf < 0.01)
    # from corrupting MotionBERT's input.
    # We zero all 3 channels (x, y, conf) so crop_scale excludes them from
    # the bounding box and MotionBERT sees them as truly absent.
    n_zeroed: int = 0
    if cfg.MOTIONBERT_CONF_THRESHOLD > 0.0:
        for i in range(n_frames):
            for j in range(17):
                if keypoints_h36m[i, j, 2] < cfg.MOTIONBERT_CONF_THRESHOLD:
                    keypoints_h36m[i, j, :] = 0.0
                    n_zeroed += 1
    n_total: int = n_frames * 17
    print(f"  Confidence threshold={cfg.MOTIONBERT_CONF_THRESHOLD}: "
          f"zeroed {n_zeroed}/{n_total} joint-frames "
          f"({100*n_zeroed/n_total:.1f}%)")

    # Official MotionBERT preprocessing: crop_scale normalization
    keypoints_norm: np.ndarray
    cs_params: dict[str, float]
    keypoints_norm, cs_params = crop_scale(keypoints_h36m)

    print(f"  crop_scale: scale={cs_params['scale']:.1f} "
          f"offset=({cs_params['xs']:.1f}, {cs_params['ys']:.1f})")

    # MotionBERT handles variable-length input natively
    clip_len: int = 243
    if n_frames > clip_len:
        keypoints_norm = keypoints_norm[:clip_len]
        n_frames = clip_len

    input_tensor: torch.Tensor = (
        torch.from_numpy(keypoints_norm).unsqueeze(0).to(device)
    )

    print(f"  Running MotionBERT on {n_frames} frames (with flip augmentation)...")

    with torch.no_grad():
        predicted_3d_pos_1: torch.Tensor = model(input_tensor)
        input_flip: torch.Tensor = flip_data(input_tensor)
        predicted_3d_pos_flip: torch.Tensor = model(input_flip)
        predicted_3d_pos_2: torch.Tensor = flip_data(predicted_3d_pos_flip)
        output_3d: torch.Tensor = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0

    positions_3d: np.ndarray = output_3d.cpu().numpy()[0]  # (N, 17, 3)

    # Global variant: zero first frame root Z
    positions_3d[0, 0, 2] = 0

    print(f"  MotionBERT raw output: {positions_3d.shape}")
    print(f"  Root Z range (norm): {positions_3d[:, 0, 2].min():.4f} to "
          f"{positions_3d[:, 0, 2].max():.4f}")

    # Save normalized positions BEFORE denormalization
    positions_3d_norm: np.ndarray = positions_3d.copy()

    # Denormalize from crop_scale to pixel-aligned coordinates
    scale: float = cs_params["scale"]
    xs: float = cs_params["xs"]
    ys: float = cs_params["ys"]

    positions_3d *= (scale / 2.0)
    positions_3d[:, :, 0] += xs + scale / 2.0
    positions_3d[:, :, 1] += ys + scale / 2.0

    return positions_3d, positions_3d_norm, cs_params


# ---------------------------------------------------------------------------
# Denormalization to Camera-Space Meters
# ---------------------------------------------------------------------------

def pixel_aligned_to_camera_space(
    kp_3d: np.ndarray,
    kp_2d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Convert pixel-aligned 3D to camera-space meters using torso-height heuristic.

    DEPRECATED: Use motionbert_to_camera_space instead. Kept for reference.

    Args:
        kp_3d: (17, 3) pixel-aligned 3D from MotionBERT.
        kp_2d: (17, 2) pixel coordinates.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (17, 3) camera-space meters.
    """
    # Compute pixel torso height: thorax(8) to ankle midpoint (3+6)/2
    thorax_2d: np.ndarray = kp_2d[8]
    ankle_mid_2d: np.ndarray = (kp_2d[3] + kp_2d[6]) / 2.0
    pixel_height: float = abs(float(thorax_2d[1] - ankle_mid_2d[1]))

    # Thorax-to-ankle anatomical height ~1.38m
    assumed_height_m: float = 1.38
    if pixel_height > 20:
        root_depth: float = fx * assumed_height_m / pixel_height
    else:
        root_depth = 3.0
    root_depth = float(np.clip(root_depth, 1.0, 8.0))

    root_z_px: float = float(kp_3d[0, 2])
    cam_3d: np.ndarray = np.zeros((17, 3), dtype=np.float64)

    for j in range(17):
        u: float = float(kp_3d[j, 0])
        v: float = float(kp_3d[j, 1])
        z_px: float = float(kp_3d[j, 2])
        z_cam: float = root_depth + (z_px - root_z_px) * root_depth / fx
        x_cam: float = (u - cx) * z_cam / fx
        y_cam: float = (v - cy) * z_cam / fy
        cam_3d[j] = [x_cam, y_cam, z_cam]

    return cam_3d


def _iqr_filtered_median(values: np.ndarray, k: float = 1.5) -> float:
    """Compute median after removing IQR outliers.

    Args:
        values: 1D array of values.
        k: IQR multiplier for outlier detection (1.5 = standard).

    Returns:
        Median of inlier values, or median of all values if too few remain.
    """
    if len(values) < 4:
        return float(np.median(values))
    q1: float = float(np.percentile(values, 25))
    q3: float = float(np.percentile(values, 75))
    iqr: float = q3 - q1
    lower: float = q1 - k * iqr
    upper: float = q3 + k * iqr
    inliers: np.ndarray = values[(values >= lower) & (values <= upper)]
    if len(inliers) < 2:
        return float(np.median(values))
    return float(np.median(inliers))


# Bones that are most reliable for scale estimation.
# Arm bones are the most consistently well-estimated by MotionBERT --
# they have clear 2D separation and minimal depth ambiguity.
# Spine/neck/legs are excluded because MotionBERT often distorts them.
_RELIABLE_BONES_FOR_SCALE: set[int] = {
    11,  # Thorax -> LShoulder
    12,  # LShoulder -> LElbow
    13,  # LElbow -> LWrist
    14,  # Thorax -> RShoulder
    15,  # RShoulder -> RElbow
    16,  # RElbow -> RWrist
}


def _enforce_bone_lengths(
    positions: np.ndarray,
    parents: np.ndarray,
    default_lengths: np.ndarray,
    max_ratio: float = 1.5,
) -> np.ndarray:
    """Enforce bone-length constraints by clamping extreme bones.

    For each bone in kinematic chain order, if the bone length deviates by
    more than max_ratio from the default, the child joint is projected to
    be at the default distance from its parent, preserving the bone direction.

    Args:
        positions: (17, 3) root-relative joint positions in meters.
        parents: (17,) parent index array.
        default_lengths: (17,) default bone lengths in meters.
        max_ratio: Maximum allowed ratio (detected / default). Bones outside
            [1/max_ratio, max_ratio] are corrected.

    Returns:
        (17, 3) corrected positions.
    """
    result: np.ndarray = positions.copy()
    for j in range(1, 17):
        p: int = int(parents[j])
        bone_vec: np.ndarray = result[j] - result[p]
        bone_len: float = float(np.linalg.norm(bone_vec))
        ref_len: float = float(default_lengths[j])
        if bone_len < 1e-6 or ref_len < 1e-6:
            continue
        ratio: float = bone_len / ref_len
        if ratio > max_ratio or ratio < 1.0 / max_ratio:
            # Clamp to default length while preserving direction
            direction: np.ndarray = bone_vec / bone_len
            result[j] = result[p] + direction * ref_len
    return result


def _enforce_bone_lengths_with_2d(
    positions: np.ndarray,
    parents: np.ndarray,
    default_lengths: np.ndarray,
    kp_2d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tz: float,
    tx: float = 0.0,
    ty: float = 0.0,
    max_ratio: float = 1.5,
) -> np.ndarray:
    """Enforce bone-length constraints using 2D keypoints to solve for depth.

    For extreme bones, uses the 2D keypoint and camera intrinsics to
    reconstruct the child joint position. Given parent camera-space position,
    child 2D projection, and known bone length, solves for child depth (Z)
    analytically.

    For child joint at camera-space (X_c, Y_c, Z_c):
        u = fx * X_c / Z_c + cx  =>  X_c = (u - cx) * Z_c / fx
        v = fy * Y_c / Z_c + cy  =>  Y_c = (v - cy) * Z_c / fy

    Bone length constraint:
        (X_c - X_p)^2 + (Y_c - Y_p)^2 + (Z_c - Z_p)^2 = L^2

    Substituting X_c, Y_c as functions of Z_c gives a quadratic in Z_c.

    Args:
        positions: (17, 3) root-relative joint positions in meters.
        parents: (17,) parent index array.
        default_lengths: (17,) default bone lengths in meters.
        kp_2d: (17, 2) 2D pixel coordinates.
        fx, fy, cx, cy: Camera intrinsics.
        tz: Estimated root depth (z translation to camera space).
        tx: Estimated x translation to camera space.
        ty: Estimated y translation to camera space.
        max_ratio: Maximum allowed bone length ratio for triggering correction.

    Returns:
        (17, 3) corrected root-relative positions.
    """
    result: np.ndarray = positions.copy()
    for j in range(1, 17):
        p: int = int(parents[j])
        bone_vec: np.ndarray = result[j] - result[p]
        bone_len: float = float(np.linalg.norm(bone_vec))
        ref_len: float = float(default_lengths[j])
        if bone_len < 1e-6 or ref_len < 1e-6:
            continue
        ratio: float = bone_len / ref_len
        if ratio > max_ratio or ratio < 1.0 / max_ratio:
            u_child: float = float(kp_2d[j, 0])
            v_child: float = float(kp_2d[j, 1])
            if abs(u_child) < 1.0 and abs(v_child) < 1.0:
                # No valid 2D -- fall back to direction clamping
                direction: np.ndarray = bone_vec / bone_len
                result[j] = result[p] + direction * ref_len
                continue

            # Parent position in camera space
            x_p_cam: float = float(result[p, 0]) + tx
            y_p_cam: float = float(result[p, 1]) + ty
            z_p_cam: float = float(result[p, 2]) + tz

            # Child 2D -> normalized image coordinates
            nx: float = (u_child - cx) / fx
            ny: float = (v_child - cy) / fy

            # Solve quadratic for Z_c (child camera-space Z):
            # (nx*Z_c - x_p)^2 + (ny*Z_c - y_p)^2 + (Z_c - z_p)^2 = L^2
            a: float = nx * nx + ny * ny + 1.0
            b: float = -2.0 * (nx * x_p_cam + ny * y_p_cam + z_p_cam)
            c: float = x_p_cam**2 + y_p_cam**2 + z_p_cam**2 - ref_len**2

            discriminant: float = b * b - 4.0 * a * c
            if discriminant < 0:
                # No real solution -- bone is too short to reach the 2D point.
                # Fall back to direction clamping.
                direction = bone_vec / bone_len
                result[j] = result[p] + direction * ref_len
                continue

            sqrt_disc: float = float(np.sqrt(discriminant))
            z_c1: float = (-b + sqrt_disc) / (2.0 * a)
            z_c2: float = (-b - sqrt_disc) / (2.0 * a)

            # Pick the solution closest to parent Z (most likely correct)
            if abs(z_c1 - z_p_cam) < abs(z_c2 - z_p_cam):
                z_c: float = z_c1
            else:
                z_c = z_c2

            # Ensure positive depth
            if z_c < 0.1:
                z_c = z_p_cam  # Fall back to parent depth

            x_c: float = nx * z_c
            y_c: float = ny * z_c

            # Convert back to root-relative
            result[j, 0] = x_c - tx
            result[j, 1] = y_c - ty
            result[j, 2] = z_c - tz

    return result


def _reconstruct_from_2d(
    kp_2d: np.ndarray,
    parents: np.ndarray,
    default_lengths: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    root_depth: float,
    visibility: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct 3D skeleton from 2D keypoints using bone-length constraints.

    Places root at root_depth, then for each joint in kinematic chain order,
    uses the 2D keypoint to define a ray and solves for the depth that gives
    the correct bone length from parent.

    Args:
        kp_2d: (17, 2) 2D pixel coordinates.
        parents: (17,) parent index array.
        default_lengths: (17,) default bone lengths in meters.
        fx, fy, cx, cy: Camera intrinsics.
        root_depth: Estimated root joint depth (Z) in camera space.
        visibility: Optional (17,) confidence scores.

    Returns:
        (17, 3) camera-space meters.
    """
    result: np.ndarray = np.zeros((17, 3), dtype=np.float64)

    # Place root
    u_root: float = float(kp_2d[0, 0])
    v_root: float = float(kp_2d[0, 1])
    result[0, 0] = (u_root - cx) * root_depth / fx
    result[0, 1] = (v_root - cy) * root_depth / fy
    result[0, 2] = root_depth

    for j in range(1, 17):
        p: int = int(parents[j])
        ref_len: float = float(default_lengths[j])
        u_child: float = float(kp_2d[j, 0])
        v_child: float = float(kp_2d[j, 1])

        # Check if 2D is valid
        is_valid: bool = abs(u_child) > 1.0 or abs(v_child) > 1.0
        if visibility is not None and visibility[j] < 0.1:
            is_valid = False

        if not is_valid or ref_len < 1e-4:
            # No valid 2D -- place at parent position (zero bone)
            result[j] = result[p].copy()
            continue

        # Normalized image coordinates for child
        nx: float = (u_child - cx) / fx
        ny: float = (v_child - cy) / fy

        # Parent camera position
        x_p: float = float(result[p, 0])
        y_p: float = float(result[p, 1])
        z_p: float = float(result[p, 2])

        # Solve quadratic: (nx*z - x_p)^2 + (ny*z - y_p)^2 + (z - z_p)^2 = L^2
        a: float = nx * nx + ny * ny + 1.0
        b: float = -2.0 * (nx * x_p + ny * y_p + z_p)
        c: float = x_p**2 + y_p**2 + z_p**2 - ref_len**2

        discriminant: float = b * b - 4.0 * a * c
        if discriminant < 0:
            # Bone too short to reach ray -- place at closest point on ray to parent
            # z_closest = (nx*x_p + ny*y_p + z_p) / (nx^2 + ny^2 + 1)
            z_closest: float = (nx * x_p + ny * y_p + z_p) / a
            if z_closest < 0.1:
                z_closest = z_p
            result[j, 0] = nx * z_closest
            result[j, 1] = ny * z_closest
            result[j, 2] = z_closest
            continue

        sqrt_disc: float = float(np.sqrt(discriminant))
        z_c1: float = (-b + sqrt_disc) / (2.0 * a)
        z_c2: float = (-b - sqrt_disc) / (2.0 * a)

        # Pick solution closest to parent Z
        if abs(z_c1 - z_p) < abs(z_c2 - z_p):
            z_c: float = z_c1
        else:
            z_c = z_c2

        if z_c < 0.1:
            z_c = z_p

        result[j, 0] = nx * z_c
        result[j, 1] = ny * z_c
        result[j, 2] = z_c

    return result


def motionbert_to_camera_space(
    positions_3d_norm: np.ndarray,
    kp_2d: np.ndarray,
    scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    dist_coeffs: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
) -> np.ndarray:
    """Convert MotionBERT normalized output to camera-space meters.

    Uses a two-step approach:
    1. Scale the root-relative 3D structure using bone length matching against
       known anatomical reference lengths, with IQR outlier filtering and
       preference for reliable upper-body bones.
    2. Estimate depth (tz) from pairwise joint separation ratios with IQR
       outlier filtering, then solve for tx, ty from 2D projections.

    Falls back to a torso-height heuristic if too few valid joints or if
    the bone-length scale produces unreasonable results.

    Args:
        positions_3d_norm: (17, 3) normalized MotionBERT output (before pixel denorm).
        kp_2d: (17, 2) 2D detections in pixel coordinates.
        scale: crop_scale scale parameter.
        fx, fy, cx, cy: Camera intrinsics.
        dist_coeffs: Optional distortion coefficients from camera calibration.
            If None, zero distortion is assumed.
        visibility: Optional (17,) confidence scores for each joint. Joints with
            confidence below 0.1 are excluded from depth estimation.

    Returns:
        (17, 3) camera-space meters.
    """
    from skeleton import PARENTS, DEFAULT_BONE_LENGTHS

    # Step 1: Get root-relative structure in meters via bone-length matching
    root_relative: np.ndarray = positions_3d_norm - positions_3d_norm[0:1]

    # Compute scale factor using reliable arm bones with IQR filtering.
    # If arm bones are inconsistent (high coefficient of variation),
    # fall back to using all bones with IQR filtering.
    reliable_ratios: list[float] = []
    all_ratios: list[float] = []
    for j in range(1, 17):
        p: int = int(PARENTS[j])
        det_bl: float = float(np.linalg.norm(root_relative[j] - root_relative[p]))
        ref_bl: float = float(DEFAULT_BONE_LENGTHS[j])
        if det_bl > 1e-4 and ref_bl > 1e-4:
            ratio: float = ref_bl / det_bl
            all_ratios.append(ratio)
            if j in _RELIABLE_BONES_FOR_SCALE:
                reliable_ratios.append(ratio)

    if len(reliable_ratios) >= 4:
        bone_scale: float = _iqr_filtered_median(np.array(reliable_ratios))
    elif all_ratios:
        bone_scale = _iqr_filtered_median(np.array(all_ratios))
    else:
        bone_scale = 1.0

    root_relative_m: np.ndarray = root_relative * bone_scale

    # Step 2: Estimate optimal translation (tx, ty, tz) to place skeleton in
    # camera space. MotionBERT's coordinate frame is already approximately aligned
    # with the camera (trained on H3.6M camera-space data), so rotation is not
    # needed. We estimate depth (tz) from the median of pairwise joint separation
    # ratios (3D distance / 2D distance) with IQR outlier filtering, then compute
    # tx, ty from the 2D projections at the estimated depth.

    # Joints to use for depth estimation: prefer upper-body + hips which are
    # more reliable than knees/ankles in MotionBERT's output.
    # Knees (2,5) and ankles (3,6) are excluded from tz estimation because
    # MotionBERT often places them at wrong depths.
    _DEPTH_RELIABLE_JOINTS: set[int] = {
        0, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    }

    # Filter joints: require nonzero 2D position AND sufficient confidence
    valid: np.ndarray = np.linalg.norm(kp_2d, axis=1) > 1.0
    if visibility is not None:
        valid = valid & (visibility > 0.1)

    # For tz estimation, additionally filter to reliable joints only
    valid_for_tz: np.ndarray = valid.copy()
    for j in range(17):
        if j not in _DEPTH_RELIABLE_JOINTS:
            valid_for_tz[j] = False

    n_valid: int = int(valid.sum())
    n_valid_tz: int = int(valid_for_tz.sum())

    if n_valid_tz >= 4:
        pts_3d_tz: np.ndarray = root_relative_m[valid_for_tz].astype(np.float64)
        pts_2d_tz: np.ndarray = kp_2d[valid_for_tz].astype(np.float64)

        # Estimate tz from pairwise joint separations with IQR filtering.
        # Only uses reliable joints (no knees/ankles).
        tz_estimates: list[float] = []
        n_pts: int = len(pts_3d_tz)
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                dy_3d: float = float(pts_3d_tz[i, 1] - pts_3d_tz[j, 1])
                dv_2d: float = float(pts_2d_tz[i, 1] - pts_2d_tz[j, 1])
                if abs(dv_2d) > 10 and abs(dy_3d) > 0.05:
                    tz_est: float = fy * dy_3d / dv_2d
                    if 0.5 < tz_est < 15.0:
                        tz_estimates.append(tz_est)

                dx_3d: float = float(pts_3d_tz[i, 0] - pts_3d_tz[j, 0])
                du_2d: float = float(pts_2d_tz[i, 0] - pts_2d_tz[j, 0])
                if abs(du_2d) > 10 and abs(dx_3d) > 0.05:
                    tz_est = fx * dx_3d / du_2d
                    if 0.5 < tz_est < 15.0:
                        tz_estimates.append(tz_est)

        if tz_estimates:
            # Use IQR-filtered median for robust tz estimation
            tz_arr: np.ndarray = np.array(tz_estimates)
            tz: float = _iqr_filtered_median(tz_arr)

            # Cross-check: torso-height heuristic for tz.
            # Uses a fixed anatomical reference (0.55m shoulder-mid to hip-mid)
            # which is independent of MotionBERT's predictions.
            lshoulder_2d: np.ndarray = kp_2d[11]
            rshoulder_2d: np.ndarray = kp_2d[14]
            lhip_2d: np.ndarray = kp_2d[4]
            rhip_2d: np.ndarray = kp_2d[1]
            shoulder_mid_2d: np.ndarray = (lshoulder_2d + rshoulder_2d) / 2.0
            hip_mid_2d: np.ndarray = (lhip_2d + rhip_2d) / 2.0
            torso_pixel_height: float = abs(float(shoulder_mid_2d[1] - hip_mid_2d[1]))
            if torso_pixel_height > 15:
                # Fixed anatomical reference: shoulder-mid to hip-mid ~0.55m
                # for a typical adult (measured from H3.6M/Panoptic GT).
                torso_3d_height: float = 0.55
                tz_heuristic: float = fy * torso_3d_height / torso_pixel_height
                tz_heuristic = float(np.clip(tz_heuristic, 1.0, 8.0))
                # Always blend pairwise tz with heuristic for robustness.
                # The heuristic uses fixed anatomical reference and doesn't
                # depend on MotionBERT's scale, so it anchors the depth.
                # Weight the heuristic more heavily (60%) since the pairwise
                # estimate inherits errors from MotionBERT's bone_scale.
                tz = 0.4 * tz + 0.6 * tz_heuristic

            # First compute tx, ty using reliable (uncorrected) joints
            pts_3d_tx: np.ndarray = root_relative_m[valid_for_tz].astype(np.float64)
            pts_2d_tx: np.ndarray = kp_2d[valid_for_tz].astype(np.float64)
            tx_estimates: np.ndarray = (
                (pts_2d_tx[:, 0] - cx) * (pts_3d_tx[:, 2] + tz) / fx - pts_3d_tx[:, 0]
            )
            ty_estimates: np.ndarray = (
                (pts_2d_tx[:, 1] - cy) * (pts_3d_tx[:, 2] + tz) / fy - pts_3d_tx[:, 1]
            )
            tx: float = float(np.median(tx_estimates))
            ty: float = float(np.median(ty_estimates))

            # Enforce bone-length constraints: clamp extreme bones to default
            # length while preserving direction. Uses max_ratio=1.3 to catch
            # moderate distortions (MotionBERT commonly produces 1.3-1.5x
            # ratios for knees/ankles).
            root_relative_corrected: np.ndarray = _enforce_bone_lengths(
                root_relative_m, PARENTS, DEFAULT_BONE_LENGTHS,
                max_ratio=1.3,
            )

            cam_3d: np.ndarray = root_relative_corrected.copy()
            cam_3d[:, 0] += tx
            cam_3d[:, 1] += ty
            cam_3d[:, 2] += tz
            return cam_3d.astype(np.float64)

    # Fallback: heuristic depth estimation using shoulder-to-hip span
    lshoulder_2d_fb: np.ndarray = kp_2d[11]
    rshoulder_2d_fb: np.ndarray = kp_2d[14]
    lhip_2d_fb: np.ndarray = kp_2d[4]
    rhip_2d_fb: np.ndarray = kp_2d[1]
    shoulder_mid_2d_fb: np.ndarray = (lshoulder_2d_fb + rshoulder_2d_fb) / 2.0
    hip_mid_2d_fb: np.ndarray = (lhip_2d_fb + rhip_2d_fb) / 2.0
    pixel_height: float = abs(float(shoulder_mid_2d_fb[1] - hip_mid_2d_fb[1]))
    # Fixed anatomical reference: shoulder-mid to hip-mid ~0.55m
    assumed_height_m: float = 0.55
    if pixel_height > 20:
        root_depth: float = fy * assumed_height_m / pixel_height
    else:
        root_depth = 3.0
    root_depth = float(np.clip(root_depth, 1.0, 8.0))

    u_root: float = float(kp_2d[0, 0])
    v_root: float = float(kp_2d[0, 1])
    x_root: float = (u_root - cx) * root_depth / fx
    y_root: float = (v_root - cy) * root_depth / fy
    z_root: float = root_depth

    # Enforce bone-length constraints before translation
    root_relative_corrected = _enforce_bone_lengths(
        root_relative_m, PARENTS, DEFAULT_BONE_LENGTHS,
        max_ratio=1.3,
    )

    cam_3d = root_relative_corrected.copy()
    cam_3d[:, 0] += x_root
    cam_3d[:, 1] += y_root
    cam_3d[:, 2] += z_root

    return cam_3d


# ---------------------------------------------------------------------------
# Pipeline Wrapper
# ---------------------------------------------------------------------------

def detect_poses(
    frames_rgb: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
    """Full detection pipeline.

    Pipeline:
      1. YOLOv8 person detection -> union bounding box
      2. Stacked Hourglass -> MPII (16,3) 2D keypoints + raw heatmaps per frame
      3. MotionBERT -> H36M (N,17,3) pixel-aligned 3D + normalized output
      4. Convert MPII 2D to H36M 2D

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.

    Returns:
        keypoints_2d: List of (17, 2) pixel coordinates (H36M).
        keypoints_3d: List of (17, 3) pixel-aligned 3D (H36M).
        confidence: List of (17,) confidence scores.
        heatmaps: List of (16, 64, 64) raw MPII heatmaps per frame.
        affine: (2, 3) affine from 256-crop coords to original pixel coords.
        positions_3d_norm: (N, 17, 3) normalized MotionBERT output (before denorm).
        cs_params: crop_scale parameters dict with keys xs, ys, scale.
    """
    h: int
    w: int
    h, w = frames_rgb[0].shape[:2]
    image_size: tuple[int, int] = (h, w)

    # 1. YOLOv8 person detection -> union bounding box
    union_bbox: np.ndarray = detect_person_bbox(frames_rgb)

    # 2. Stacked Hourglass -> MPII 2D keypoints + raw heatmaps
    all_keypoints_2d: list[np.ndarray]
    all_heatmaps: list[np.ndarray]
    affine: np.ndarray
    all_keypoints_2d, all_heatmaps, affine = run_hourglass(frames_rgb, union_bbox)

    # 3. MotionBERT -> H36M pixel-aligned 3D + normalized output
    kp_3d_array: np.ndarray
    positions_3d_norm: np.ndarray
    cs_params: dict[str, float]
    kp_3d_array, positions_3d_norm, cs_params = run_motionbert(all_keypoints_2d, image_size)

    # 4. Convert MPII 2D to H36M 2D + extract visibility
    kp_2d_list: list[np.ndarray] = []
    visibility_list: list[np.ndarray] = []
    for kp_mpii in all_keypoints_2d:
        kp_h36m: np.ndarray = mpii_to_h36m(kp_mpii)  # (17, 3) with confidence
        kp_2d_list.append(kp_h36m[:, :2])  # (17, 2)
        visibility_list.append(kp_h36m[:, 2])  # (17,)

    kp_3d_list: list[np.ndarray] = [
        kp_3d_array[i] for i in range(kp_3d_array.shape[0])
    ]

    return kp_2d_list, kp_3d_list, visibility_list, all_heatmaps, affine, positions_3d_norm, cs_params
