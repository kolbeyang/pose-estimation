"""2D pose estimation (Stacked Hourglass) and 3D lifting (MotionBERT).

Handles person detection (YOLOv8), 2D heatmap extraction, and 2D->3D lifting.
"""

import copy
import os
import sys
import time
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config as cfg
from skeleton import mpii_to_h36m, h36m_17_to_16, NUM_JOINTS

SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR: str = os.path.join(SCRIPT_DIR, "external")
CHECKPOINTS_DIR: str = os.path.join(SCRIPT_DIR, "checkpoints")

# --- MPII flip pairs for horizontal flip augmentation ---
MPII_FLIP_PAIRS: list[tuple[int, int]] = [
    (0, 5),
    (1, 4),
    (2, 3),
    (10, 15),
    (11, 14),
    (12, 13),
]

# RGB channel means for Stacked Hourglass normalization (from MPII training set).
_SH_RGB_MEAN: np.ndarray = np.array([0.4404, 0.4440, 0.4327], dtype=np.float32)


def _normalize_for_sh(cropped: np.ndarray) -> np.ndarray:
    """Normalize a cropped HWC uint8 image for Stacked Hourglass inference.

    Converts to float32, scales to [0,1], transposes to CHW, and subtracts
    the RGB channel means used during MPII training.

    Args:
        cropped: (H, W, 3) uint8 image.

    Returns:
        (3, H, W) float32 normalized image.
    """
    img: np.ndarray = cropped.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img[0] -= _SH_RGB_MEAN[0]
    img[1] -= _SH_RGB_MEAN[1]
    img[2] -= _SH_RGB_MEAN[2]
    return img


def _get_device() -> torch.device:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
            areas: torch.Tensor = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * (
                detections.xyxy[:, 3] - detections.xyxy[:, 1]
            )
            best_idx: int = areas.argmax().item()
            bbox: np.ndarray = (
                detections.xyxy[best_idx].cpu().numpy().astype(np.float32)
            )
            bboxes.append(bbox)
        else:
            bboxes.append(np.array([0, 0, w, h], dtype=np.float32))

    # Union (max) bounding box across all frames
    all_bboxes: np.ndarray = np.array(bboxes)
    union_bbox: np.ndarray = np.array(
        [
            all_bboxes[:, 0].min(),
            all_bboxes[:, 1].min(),
            all_bboxes[:, 2].max(),
            all_bboxes[:, 3].max(),
        ],
        dtype=np.float32,
    )

    print(
        f"  Union bbox: ({union_bbox[0]:.0f}, {union_bbox[1]:.0f}) - "
        f"({union_bbox[2]:.0f}, {union_bbox[3]:.0f})"
    )

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
    affine: np.ndarray = np.array(
        [
            scale_x,
            0,
            orig_crop_x1,
            0,
            scale_y,
            orig_crop_y1,
        ],
        dtype=np.float32,
    ).reshape(2, 3)

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
    """Run HG8 on cropped frames with flip augmentation (batched).

    Args:
        frames_rgb: RGB frames (H, W, 3).
        bbox: (4,) union bounding box.

    Returns:
        keypoints_2d: List of (16, 3) in original pixel coords (x, y, confidence).
        heatmaps: List of (16, 64, 64) raw heatmaps.
        affine: (2, 3) affine from 256-crop -> original pixel coords.
    """
    from stacked_hourglass import hg8

    device: torch.device = _get_device()

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
    print(f"  Loaded Stacked Hourglass (8-stack, pretrained) on {device}")

    # --- Phase 1: Preprocess all frames ---
    preprocessed: list[np.ndarray] = []       # normalized CHW arrays
    preprocessed_flip: list[np.ndarray] = []  # flipped normalized CHW arrays
    shared_affine: np.ndarray | None = None

    for frame in frames_rgb:
        cropped: np.ndarray
        affine: np.ndarray
        cropped, affine = crop_and_resize(frame, bbox, target_size=256)
        if shared_affine is None:
            shared_affine = affine

        # Normalize: [0,1], HWC->CHW, subtract RGB means
        preprocessed.append(_normalize_for_sh(cropped))

        # Flipped version
        cropped_flip: np.ndarray = cropped[:, ::-1].copy()
        preprocessed_flip.append(_normalize_for_sh(cropped_flip))

    assert shared_affine is not None
    n_frames: int = len(frames_rgb)
    batch_size: int = cfg.SH_BATCH_SIZE

    # --- Phase 2: Batched inference ---
    all_heatmaps: list[np.ndarray] = []

    print(f"  Running Stacked Hourglass on {n_frames} frames (batch_size={batch_size})...")

    with torch.no_grad():
        for start in tqdm(range(0, n_frames, batch_size), desc="  2D Pose"):
            end: int = min(start + batch_size, n_frames)
            batch_imgs: list[np.ndarray] = preprocessed[start:end]
            batch_flips: list[np.ndarray] = preprocessed_flip[start:end]

            # Stack into tensor and run forward pass
            try:
                inp: torch.Tensor = torch.from_numpy(np.stack(batch_imgs)).to(device)
                output: list[torch.Tensor] = model(inp)
                heatmaps_batch: np.ndarray = output[-1].cpu().numpy()  # (B, 16, 64, 64)

                # Flipped forward pass
                inp_flip: torch.Tensor = torch.from_numpy(np.stack(batch_flips)).to(device)
                output_flip: list[torch.Tensor] = model(inp_flip)
                heatmaps_flip_batch: np.ndarray = output_flip[-1].cpu().numpy()  # (B, 16, 64, 64)
            except RuntimeError as e:
                # MPS fallback: if GPU fails, retry on CPU
                if device.type != "cpu":
                    print(f"  WARNING: {device} failed ({e}), falling back to CPU")
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

            # Flip heatmaps and swap joints, then average
            for i in range(end - start):
                hm_flip: np.ndarray = _flip_heatmaps(heatmaps_flip_batch[i])
                hm_avg: np.ndarray = (heatmaps_batch[i] + hm_flip) / 2.0
                all_heatmaps.append(hm_avg)

    # --- Phase 3: Parse keypoints ---
    all_keypoints_2d: list[np.ndarray] = []
    for heatmaps in all_heatmaps:
        keypoints_64: np.ndarray = _parse_heatmaps(heatmaps)
        keypoints_orig: np.ndarray = keypoints_64.copy()
        keypoints_orig[:, :2] *= 4  # 64 -> 256
        for j in range(16):
            x_256: float = keypoints_orig[j, 0]
            y_256: float = keypoints_orig[j, 1]
            keypoints_orig[j, 0] = shared_affine[0, 0] * x_256 + shared_affine[0, 2]
            keypoints_orig[j, 1] = shared_affine[1, 1] * y_256 + shared_affine[1, 2]
        all_keypoints_2d.append(keypoints_orig)

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
        dim_in=3,
        dim_out=3,
        dim_feat=256,
        dim_rep=512,
        depth=5,
        num_heads=8,
        mlp_ratio=4,
        num_joints=17,
        maxlen=243,
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
            "xs": 0.0,
            "ys": 0.0,
            "scale": 1.0,
        }
    xmin: float = float(valid_coords[:, 0].min())
    xmax: float = float(valid_coords[:, 0].max())
    ymin: float = float(valid_coords[:, 1].min())
    ymax: float = float(valid_coords[:, 1].max())
    scale: float = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape, dtype=motion.dtype), {
            "xs": 0.0,
            "ys": 0.0,
            "scale": 1.0,
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
    flipped_data[..., left_joints + right_joints, :] = flipped_data[
        ..., right_joints + left_joints, :
    ]
    return flipped_data


def run_motionbert(
    keypoints_2d_list: list[np.ndarray],
) -> np.ndarray:
    """Lift 2D keypoints to 3D using MotionBERT.

    Args:
        keypoints_2d_list: List of (16, 3) MPII keypoints per frame (x, y, conf).

    Returns:
        positions_3d_norm: (N, 16, 3) normalized MotionBERT output (Head removed).
    """
    model: torch.nn.Module = load_motionbert_model()
    device: torch.device = _get_device()
    model = model.to(device)
    print(f"  MotionBERT on {device}")

    n_frames: int = len(keypoints_2d_list)

    # Convert MPII 16-joint to H36M 17-joint for MotionBERT input
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
    print(
        f"  Confidence threshold={cfg.MOTIONBERT_CONF_THRESHOLD}: "
        f"zeroed {n_zeroed}/{n_total} joint-frames "
        f"({100*n_zeroed/n_total:.1f}%)"
    )

    # Official MotionBERT preprocessing: crop_scale normalization
    keypoints_norm: np.ndarray
    cs_params: dict[str, float]
    keypoints_norm, cs_params = crop_scale(keypoints_h36m)

    print(
        f"  crop_scale: scale={cs_params['scale']:.1f} "
        f"offset=({cs_params['xs']:.1f}, {cs_params['ys']:.1f})"
    )

    # MotionBERT handles variable-length input natively
    clip_len: int = 243
    if n_frames > clip_len:
        keypoints_norm = keypoints_norm[:clip_len]
        n_frames = clip_len

    input_tensor: torch.Tensor = (
        torch.from_numpy(keypoints_norm).unsqueeze(0).to(device)
    )

    print(f"  Running MotionBERT on {n_frames} frames...")

    with torch.no_grad():
        try:
            output_3d: torch.Tensor = model(input_tensor)
        except RuntimeError as e:
            # MPS fallback: if GPU fails, retry on CPU
            if device.type != "cpu":
                print(f"  WARNING: {device} failed ({e}), falling back to CPU")
                model = model.to("cpu")
                input_tensor = input_tensor.to("cpu")
                output_3d = model(input_tensor)
            else:
                raise

    positions_3d: np.ndarray = output_3d.cpu().numpy()[0]  # (N, 17, 3)

    # Global variant: zero first frame root Z
    positions_3d[0, 0, 2] = 0

    print(f"  MotionBERT raw output: {positions_3d.shape}")
    print(
        f"  Root Z range (norm): {positions_3d[:, 0, 2].min():.4f} to "
        f"{positions_3d[:, 0, 2].max():.4f}"
    )

    # Strip Head joint (index 10) from 17-joint H36M output -> 16 joints
    # MotionBERT outputs H36M-17; Stacked Hourglass outputs MPII-16 (no head_top).
    # To keep joint sets consistent throughout the pipeline we drop the head.
    positions_3d = h36m_17_to_16(positions_3d)

    return positions_3d


# ---------------------------------------------------------------------------
# Denormalization to Camera-Space Meters
# ---------------------------------------------------------------------------


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
    10,  # Thorax -> LShoulder
    11,  # LShoulder -> LElbow
    12,  # LElbow -> LWrist
    13,  # Thorax -> RShoulder
    14,  # RShoulder -> RElbow
    15,  # RElbow -> RWrist
}


def motionbert_to_camera_space(
    positions_3d_norm: np.ndarray,
    kp_2d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Convert MotionBERT normalized output to camera-space meters.

    Two-step approach:
    1. Scale root-relative 3D structure via bone-length matching against
       anatomical reference lengths (IQR-filtered, arm bones preferred).
    2. Estimate depth (tz) from pairwise vertical joint separation ratios:
       tz = fy * dy_3d / dv_2d for each joint pair with >5px vertical
       separation; take IQR-filtered median.

    Args:
        positions_3d_norm: (16, 3) normalized MotionBERT output.
        kp_2d: (16, 2) 2D detections in pixel coordinates.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (16, 3) camera-space meters.
    """
    from skeleton import PARENTS, DEFAULT_BONE_LENGTHS

    # Step 1: Figure out scale. MotionBERT outputs 3D positions in arbitrary
    # units. Compute ratio (reference_bone_length / detected_bone_length) for
    # each bone, prefer arm bones, IQR-filter outliers, and multiply the whole
    # skeleton by the median ratio to get meters.
    root_relative: np.ndarray = positions_3d_norm - positions_3d_norm[0:1]
    reliable_ratios: list[float] = []
    all_ratios: list[float] = []
    for j in range(1, NUM_JOINTS):
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

    # Step 2: Estimate depth (tz) via similar triangles. For each joint pair
    # with enough vertical separation: tz = fy * dy_3d / dv_2d. IQR-filter
    # the candidates and clamp to [1m, 8m].
    tz_candidates: list[float] = []
    for i in range(NUM_JOINTS):
        for j in range(NUM_JOINTS):
            if i == j:
                continue
            # Require valid 2D detections (nonzero pixel distance from origin)
            if np.linalg.norm(kp_2d[i]) <= 1.0 or np.linalg.norm(kp_2d[j]) <= 1.0:
                continue
            # Require sufficient 3D vertical separation (>1 mm)
            dy_3d: float = abs(
                float(root_relative_m[i, 1]) - float(root_relative_m[j, 1])
            )
            if dy_3d < 0.001:
                continue
            # Require sufficient projected vertical pixel separation (>5 px)
            dv_2d: float = abs(float(kp_2d[i, 1]) - float(kp_2d[j, 1]))
            if dv_2d > 5.0:
                tz_candidates.append(fy * dy_3d / dv_2d)

    if len(tz_candidates) >= 2:
        tz: float = _iqr_filtered_median(np.array(tz_candidates))
        tz = float(np.clip(tz, 1.0, 8.0))
    else:
        tz = 3.0

    # Step 3: Back-project root joint's 2D pixel location to get tx, ty in
    # camera space using the known camera intrinsics (fx, fy, cx, cy) and
    # the estimated depth tz via standard pinhole projection inversion.
    u_root: float = float(kp_2d[0, 0])
    v_root: float = float(kp_2d[0, 1])
    tx: float = (u_root - cx) * tz / fx
    ty: float = (v_root - cy) * tz / fy

    # Step 4: Assemble camera-space position by adding (tx, ty, tz) to the
    # root-relative skeleton.
    cam_3d: np.ndarray = root_relative_m.copy()
    cam_3d[:, 0] += tx
    cam_3d[:, 1] += ty
    cam_3d[:, 2] += tz
    return cam_3d.astype(np.float64)


# ---------------------------------------------------------------------------
# Pipeline Wrapper
# ---------------------------------------------------------------------------


def detect_poses(
    frames_rgb: list[np.ndarray],
    return_timing: bool = False,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, float] | None,
]:
    """Full detection pipeline.

    Pipeline:
      1. YOLOv8 person detection -> union bounding box
      2. Stacked Hourglass -> MPII (16,3) 2D keypoints + raw heatmaps per frame
      3. MotionBERT -> H36M (N,16,3) normalized 3D output
      4. Convert MPII 2D to H36M 2D

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.
        return_timing: If True, append a timing dict to the return tuple.

    Returns:
        keypoints_2d: List of (16, 2) pixel coordinates (H36M, Head removed).
        confidence: List of (16,) confidence scores.
        heatmaps: List of (16, 64, 64) raw MPII heatmaps per frame.
        mpii_keypoints_2d: List of (16, 3) raw MPII keypoints (x, y, conf) in pixel coords.
        affine: (2, 3) affine from 256-crop coords to original pixel coords.
        positions_3d_norm: (N, 16, 3) normalized MotionBERT output (Head removed).
        timing: Dict with sub-stage timings when return_timing=True, else None.
    """
    # 1. YOLOv8 person detection -> union bounding box
    t0: float = time.perf_counter()
    union_bbox: np.ndarray = detect_person_bbox(frames_rgb)
    t1: float = time.perf_counter()

    # 2. Stacked Hourglass -> MPII 2D keypoints + raw heatmaps
    all_keypoints_2d: list[np.ndarray]
    all_heatmaps: list[np.ndarray]
    affine: np.ndarray
    all_keypoints_2d, all_heatmaps, affine = run_hourglass(frames_rgb, union_bbox)
    t2: float = time.perf_counter()

    # 3. MotionBERT -> H36M normalized 3D output
    positions_3d_norm: np.ndarray = run_motionbert(all_keypoints_2d)
    t3: float = time.perf_counter()

    # 4. Convert MPII 2D to H36M 2D + extract visibility
    # MPII uses a different 16-joint ordering than H36M. mpii_to_h36m() remaps
    # to H36M-17; we then strip Head (index 10) to get the 16 joints we use.
    kp_2d_list: list[np.ndarray] = []
    visibility_list: list[np.ndarray] = []
    for kp_mpii in all_keypoints_2d:
        kp_h36m: np.ndarray = mpii_to_h36m(kp_mpii)  # (17, 3) with confidence
        kp_h36m_16: np.ndarray = h36m_17_to_16(kp_h36m)  # (16, 3)
        kp_2d_list.append(kp_h36m_16[:, :2])  # (16, 2)
        visibility_list.append(kp_h36m_16[:, 2])  # (16,)

    t4: float = time.perf_counter()

    timing: dict[str, float] | None = None
    if return_timing:
        timing = {
            "yolo_s": round(t1 - t0, 3),
            "stacked_hourglass_s": round(t2 - t1, 3),
            "motionbert_s": round(t3 - t2, 3),
            "postprocess_s": round(t4 - t3, 3),
        }
        print(f"  Detection timing: YOLO={timing['yolo_s']:.1f}s, "
              f"SH={timing['stacked_hourglass_s']:.1f}s, "
              f"MB={timing['motionbert_s']:.1f}s, "
              f"post={timing['postprocess_s']:.3f}s")

    return (
        kp_2d_list,
        visibility_list,
        all_heatmaps,
        all_keypoints_2d,
        affine,
        positions_3d_norm,
        timing,
    )
