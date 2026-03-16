"""2D pose estimation (Stacked Hourglass) and 3D lifting (MotionBERT).

Handles person detection (YOLOv8), 2D heatmap extraction, and 2D->3D lifting.
"""

import copy
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

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

            # Preprocess: normalize to [0, 1], HWC -> CHW
            img: np.ndarray = cropped.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            inp: torch.Tensor = torch.from_numpy(img).unsqueeze(0).to(device)

            # Forward pass - model returns list of heatmaps per stack
            output: list[torch.Tensor] = model(inp)
            heatmaps: np.ndarray = output[-1].cpu().numpy()[0]  # (16, 64, 64)

            # Flip augmentation
            cropped_flip: np.ndarray = cropped[:, ::-1].copy()
            img_flip: np.ndarray = cropped_flip.astype(np.float32) / 255.0
            img_flip = np.transpose(img_flip, (2, 0, 1))
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


def motionbert_to_camera_space(
    positions_3d_norm: np.ndarray,
    kp_2d: np.ndarray,
    scale: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Convert MotionBERT normalized output to camera-space meters.

    Uses a two-step approach:
    1. Scale the root-relative 3D structure using bone length matching against
       known anatomical reference lengths. This is more robust than relying on
       2D torso height measurements which can be inaccurate.
    2. Place the root in camera space using 2D detection + depth estimation.

    Args:
        positions_3d_norm: (17, 3) normalized MotionBERT output (before pixel denorm).
        kp_2d: (17, 2) 2D detections in pixel coordinates.
        scale: crop_scale scale parameter.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (17, 3) camera-space meters.
    """
    from skeleton import PARENTS, DEFAULT_BONE_LENGTHS

    # Root-relative in normalized space
    root_relative: np.ndarray = positions_3d_norm - positions_3d_norm[0:1]

    # Compute bone lengths in normalized space
    detected_bone_lengths: list[float] = []
    reference_bone_lengths: list[float] = []
    for j in range(1, 17):
        p: int = int(PARENTS[j])
        det_bl: float = float(np.linalg.norm(root_relative[j] - root_relative[p]))
        ref_bl: float = float(DEFAULT_BONE_LENGTHS[j])
        if det_bl > 1e-4 and ref_bl > 1e-4:
            detected_bone_lengths.append(det_bl)
            reference_bone_lengths.append(ref_bl)

    # Compute scale factor via median bone length ratio
    if detected_bone_lengths:
        ratios: np.ndarray = np.array(reference_bone_lengths) / np.array(detected_bone_lengths)
        bone_scale: float = float(np.median(ratios))
    else:
        bone_scale = 1.0

    # Scale root-relative structure to meters
    root_relative_m: np.ndarray = root_relative * bone_scale

    # Estimate root depth from 2D torso height for absolute positioning
    thorax_2d: np.ndarray = kp_2d[8]
    ankle_mid_2d: np.ndarray = (kp_2d[3] + kp_2d[6]) / 2.0
    pixel_height: float = abs(float(thorax_2d[1] - ankle_mid_2d[1]))
    assumed_height_m: float = 1.38
    if pixel_height > 20:
        root_depth: float = fy * assumed_height_m / pixel_height  # Use fy (vertical)
    else:
        root_depth = 3.0
    root_depth = float(np.clip(root_depth, 1.0, 8.0))

    # Place root in camera space using 2D detection (not MotionBERT output)
    u_root: float = float(kp_2d[0, 0])
    v_root: float = float(kp_2d[0, 1])
    x_root: float = (u_root - cx) * root_depth / fx
    y_root: float = (v_root - cy) * root_depth / fy
    z_root: float = root_depth

    # Assemble absolute camera-space positions
    cam_3d: np.ndarray = root_relative_m.copy()
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
