"""2D pose estimation (Stacked Hourglass) and 3D lifting (MotionBERT).

Handles person detection (YOLOv8), 2D heatmap extraction, and 2D→3D lifting.
"""

import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = os.path.join(SCRIPT_DIR, "external")
CHECKPOINTS_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

# --- Joint index mappings ---

# MPII 16-joint format (Stacked Hourglass output)
MPII_JOINTS = {
    "rank": 0, "rkne": 1, "rhip": 2, "lhip": 3, "lkne": 4, "lank": 5,
    "pelv": 6, "thrx": 7, "neck": 8, "head": 9,
    "rwri": 10, "relb": 11, "rsho": 12, "lsho": 13, "lelb": 14, "lwri": 15,
}

# H36M 17-joint format (MotionBERT output)
H36M_JOINTS = {
    "hip": 0, "rhip": 1, "rkne": 2, "rank": 3, "lhip": 4, "lkne": 5, "lank": 6,
    "spine": 7, "neck": 8, "head": 9, "headtop": 10,
    "lsho": 11, "lelb": 12, "lwri": 13, "rsho": 14, "relb": 15, "rwri": 16,
}

# MPII to H36M mapping (indices)
MPII_TO_H36M = {
    0: [2, 3],     # H36M hip = midpoint(MPII rhip, lhip)
    1: 2,          # H36M rhip = MPII rhip
    2: 1,          # H36M rkne = MPII rkne
    3: 0,          # H36M rank = MPII rank
    4: 3,          # H36M lhip = MPII lhip
    5: 4,          # H36M lkne = MPII lkne
    6: 5,          # H36M lank = MPII lank
    7: [6, 7],     # H36M spine = midpoint(MPII pelv, thrx)
    8: 8,          # H36M neck = MPII neck
    9: 9,          # H36M head = MPII head
    10: 9,         # H36M headtop = MPII head (no separate joint)
    11: 13,        # H36M lsho = MPII lsho
    12: 14,        # H36M lelb = MPII lelb
    13: 15,        # H36M lwri = MPII lwri
    14: 12,        # H36M rsho = MPII rsho
    15: 11,        # H36M relb = MPII relb
    16: 10,        # H36M rwri = MPII rwri
}

# Left arm indices
MPII_LEFT_ARM = {
    "a": MPII_JOINTS["lsho"],  # 13
    "b": MPII_JOINTS["lelb"],  # 14
    "c": MPII_JOINTS["lwri"],  # 15
}
H36M_LEFT_ARM = {
    "a": H36M_JOINTS["lsho"],  # 11
    "b": H36M_JOINTS["lelb"],  # 12
    "c": H36M_JOINTS["lwri"],  # 13
}


def mpii_to_h36m(keypoints_mpii: np.ndarray) -> np.ndarray:
    """Convert MPII 16-joint keypoints to H36M 17-joint format.

    Args:
        keypoints_mpii: (16, 2) or (16, 3) array of MPII keypoints

    Returns:
        (17, D) array of H36M keypoints
    """
    ndim = keypoints_mpii.shape[-1]
    h36m = np.zeros((17, ndim), dtype=keypoints_mpii.dtype)

    for h36m_idx, mpii_idx in MPII_TO_H36M.items():
        if isinstance(mpii_idx, list):
            h36m[h36m_idx] = np.mean(keypoints_mpii[mpii_idx], axis=0)
        else:
            h36m[h36m_idx] = keypoints_mpii[mpii_idx]

    return h36m


# --- Bounding Box Utilities ---

def crop_and_resize(frame: np.ndarray, bbox: np.ndarray, target_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Crop frame to bounding box and resize to square target.

    Returns:
        (cropped_resized, affine_transform) where affine_transform maps
        from target_size coordinates back to original frame coordinates.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.2  # 20% padding
    half = side / 2

    crop_x1, crop_y1 = int(cx - half), int(cy - half)
    crop_x2, crop_y2 = int(cx + half), int(cy + half)

    h, w = frame.shape[:2]

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)

    orig_crop_x1 = crop_x1
    orig_crop_y1 = crop_y1

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
        crop_x1 += pad_left
        crop_y1 += pad_top
        crop_x2 += pad_left
        crop_y2 += pad_top

    cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    resized = cv2.resize(cropped, (target_size, target_size))

    # Affine: maps from target_size coords to original frame coords
    scale_x = (crop_x2 - crop_x1) / target_size
    scale_y = (crop_y2 - crop_y1) / target_size
    affine = np.array([
        scale_x, 0, orig_crop_x1,
        0, scale_y, orig_crop_y1,
    ], dtype=np.float32).reshape(2, 3)

    return resized, affine


# --- Stacked Hourglass 2D Pose ---

# MPII joint pairs for horizontal flip augmentation
MPII_FLIP_PAIRS = [(0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)]


def _flip_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Horizontally flip heatmaps and swap symmetric joints."""
    flipped = heatmaps[:, :, ::-1].copy()
    for left, right in MPII_FLIP_PAIRS:
        flipped[left], flipped[right] = flipped[right].copy(), flipped[left].copy()
    return flipped


def _parse_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Extract 2D keypoint locations from heatmaps.

    Args:
        heatmaps: (16, 64, 64) heatmap array

    Returns:
        (16, 3) array of (x, y, confidence) in 64x64 heatmap space
    """
    n_joints = heatmaps.shape[0]
    keypoints = np.zeros((n_joints, 3), dtype=np.float32)

    for j in range(n_joints):
        hm = heatmaps[j]
        idx = np.argmax(hm)
        y, x = np.unravel_index(idx, hm.shape)
        confidence = hm[y, x]
        keypoints[j] = [x, y, confidence]

    return keypoints


def run_hourglass(
    frames_rgb: list[np.ndarray],
    bboxes: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Run Stacked Hourglass on video frames.

    Uses the pip-installable pytorch-stacked-hourglass package (anibali fork).
    Calls model directly to get raw heatmaps (not HumanPosePredictor).

    Args:
        frames_rgb: RGB frames (H, W, 3)
        bboxes: Per-frame bounding boxes (x1, y1, x2, y2)
        image_size: (height, width) of original frames

    Returns:
        all_keypoints_2d: List of (16, 3) keypoints (x, y, conf) in original pixel coords
        all_heatmaps: List of (16, 64, 64) raw heatmaps per frame
        affine: (2, 3) affine transform from 256-crop coords to original pixel coords
    """
    from stacked_hourglass import hg8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model - manually handle map_location for CPU-only machines
    try:
        model = hg8(pretrained=True)
    except RuntimeError:
        # Checkpoint was saved on CUDA; load manually with map_location
        model = hg8(pretrained=False)
        cached_path = os.path.join(
            torch.hub.get_dir(), "checkpoints", "bearpaw_hg8-90e5d470.pth"
        )
        if os.path.exists(cached_path):
            state_dict = torch.load(cached_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(
                "Stacked Hourglass weights not found. Try downloading manually from "
                "https://github.com/anibali/pytorch-stacked-hourglass/releases"
            )

    model = model.to(device)
    model.eval()

    print("  Loaded Stacked Hourglass (8-stack, pretrained)")

    all_keypoints_2d = []
    all_heatmaps = []
    shared_affine = None

    print(f"  Running Stacked Hourglass on {len(frames_rgb)} frames...")

    with torch.no_grad():
        for frame, bbox in tqdm(zip(frames_rgb, bboxes), total=len(frames_rgb), desc="  2D Pose"):
            # Crop and resize to 256x256
            cropped, affine = crop_and_resize(frame, bbox, target_size=256)
            if shared_affine is None:
                shared_affine = affine  # same bbox for all frames (union bbox)

            # Preprocess: normalize to [0, 1], HWC -> CHW
            img = cropped.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            inp = torch.from_numpy(img).unsqueeze(0).to(device)

            # Forward pass - model returns list of heatmaps per stack
            output = model(inp)
            # Use last stack output: (1, 16, 64, 64)
            heatmaps = output[-1].cpu().numpy()[0]  # (16, 64, 64)

            # Flip augmentation
            cropped_flip = cropped[:, ::-1].copy()
            img_flip = cropped_flip.astype(np.float32) / 255.0
            img_flip = np.transpose(img_flip, (2, 0, 1))
            inp_flip = torch.from_numpy(img_flip).unsqueeze(0).to(device)
            output_flip = model(inp_flip)
            heatmaps_flip = output_flip[-1].cpu().numpy()[0]
            heatmaps_flip = _flip_heatmaps(heatmaps_flip)

            # Average original + flipped
            heatmaps = (heatmaps + heatmaps_flip) / 2.0

            all_heatmaps.append(heatmaps)

            # Parse keypoints in 64x64 space
            keypoints_64 = _parse_heatmaps(heatmaps)  # (16, 3) x,y,conf

            # Scale keypoints: 64 -> 256 -> original image coords
            keypoints_orig = keypoints_64.copy()
            keypoints_orig[:, :2] *= 4  # 64 -> 256
            for j in range(16):
                x_256, y_256 = keypoints_orig[j, 0], keypoints_orig[j, 1]
                keypoints_orig[j, 0] = affine[0, 0] * x_256 + affine[0, 2]
                keypoints_orig[j, 1] = affine[1, 1] * y_256 + affine[1, 2]

            all_keypoints_2d.append(keypoints_orig)

    return all_keypoints_2d, all_heatmaps, shared_affine


# --- MotionBERT 3D Lifting ---

def _load_motionbert_model():
    """Load pretrained MotionBERT-Lite model for 3D pose estimation."""
    motionbert_path = os.path.join(EXTERNAL_DIR, "MotionBERT")
    if not os.path.exists(motionbert_path):
        raise RuntimeError(
            f"MotionBERT not found at {motionbert_path}. "
            "Run setup_models.py first."
        )

    # Add to path for imports
    lib_path = os.path.join(motionbert_path, "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    if motionbert_path not in sys.path:
        sys.path.insert(0, motionbert_path)

    from lib.model.DSTformer import DSTformer  # noqa: E402

    # Use the Lite global checkpoint (confirmed global variant from HuggingFace)
    # The full checkpoint may be rootrel (not global), giving near-zero root Z.
    lite_ckpt = os.path.join(CHECKPOINTS_DIR, "motionbert_lite_h36m.bin")

    model = DSTformer(
        dim_in=3, dim_out=3,
        dim_feat=256, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=4,
        num_joints=17, maxlen=243,
    )
    ckpt_path = lite_ckpt
    model_name = "MotionBERT-Lite (global, H3.6M)"
    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"MotionBERT checkpoint not found at {ckpt_path}. "
            "Run setup_models.py first."
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Checkpoint has 'model_pos' key with 'module.' prefixed keys
    state_dict = checkpoint.get("model_pos", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"  Loaded {model_name}")
    return model


def _crop_scale(motion: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize 2D keypoints to [-1, 1] based on bounding box across all frames.

    Matches the official MotionBERT `crop_scale()` from lib/utils/utils_data.py.

    Returns:
        (normalized_motion, params) where params has keys xs, ys, scale for inversion.
        To invert X,Y: pixel = (norm + 1) / 2 * scale + [xs, ys]
        To invert Z:   pixel_z = norm_z * scale / 2
    """
    import copy
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape, dtype=motion.dtype), {"xs": 0, "ys": 0, "scale": 1}
    xmin = valid_coords[:, 0].min()
    xmax = valid_coords[:, 0].max()
    ymin = valid_coords[:, 1].min()
    ymax = valid_coords[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape, dtype=motion.dtype), {"xs": 0, "ys": 0, "scale": 1}
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result, {"xs": float(xs), "ys": float(ys), "scale": float(scale)}


def _flip_data(data):
    """Horizontal flip for H36M 17-joint data. Matches official flip_data()."""
    import copy
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


def run_motionbert(keypoints_2d_list: list[np.ndarray], image_size: tuple[int, int]) -> np.ndarray:
    """Lift 2D keypoints to 3D using MotionBERT.

    Uses the official preprocessing: crop_scale normalization + flip augmentation.

    Args:
        keypoints_2d_list: List of (16, 3) MPII keypoints per frame (x, y, conf) in pixel coords
        image_size: (height, width) of original frames

    Returns:
        (N, 17, 3) array of 3D joint positions in H36M format
    """
    model = _load_motionbert_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    h, w = image_size
    n_frames = len(keypoints_2d_list)

    # Convert MPII 16-joint to H36M 17-joint
    keypoints_h36m = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for i, kp_mpii in enumerate(keypoints_2d_list):
        keypoints_h36m[i] = mpii_to_h36m(kp_mpii)

    # Official MotionBERT preprocessing: crop_scale normalization
    keypoints_norm, cs_params = _crop_scale(keypoints_h36m)

    print(f"  crop_scale: scale={cs_params['scale']:.1f} offset=({cs_params['xs']:.1f}, {cs_params['ys']:.1f})")

    # MotionBERT handles variable-length input natively via temp_embed[:,:F,:,:]
    # Do NOT pad to 243 — padding biases temporal attention with repeated frames.
    clip_len = 243
    if n_frames > clip_len:
        keypoints_norm = keypoints_norm[:clip_len]
        n_frames = clip_len

    input_tensor = torch.from_numpy(keypoints_norm).unsqueeze(0).to(device)

    print(f"  Running MotionBERT on {n_frames} frames (with flip augmentation)...")

    with torch.no_grad():
        # Forward pass
        predicted_3d_pos_1 = model(input_tensor)
        # Flip augmentation (matching official flip: True config)
        input_flip = _flip_data(input_tensor)
        predicted_3d_pos_flip = model(input_flip)
        predicted_3d_pos_2 = _flip_data(predicted_3d_pos_flip)
        # Average
        output_3d = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0

    positions_3d = output_3d.cpu().numpy()[0]  # (N, 17, 3)

    # rootrel=False: zero only first frame root Z (preserves global root movement)
    positions_3d[0, 0, 2] = 0

    print(f"  MotionBERT raw output: {positions_3d.shape}")
    print(f"  Root Z range (norm): {positions_3d[:, 0, 2].min():.4f} to {positions_3d[:, 0, 2].max():.4f}")

    # Convert from crop_scale normalized space to pixel-aligned coordinates.
    # Analogous to the --pixel conversion in official infer_wild.py:
    #   results_all = results_all * (min(vid_size) / 2.0)
    #   results_all[:,:,:2] += np.array(vid_size) / 2.0
    # But with crop_scale, the scale factor is scale/2 and center is [xs+scale/2, ys+scale/2].
    scale = cs_params['scale']
    xs = cs_params['xs']
    ys = cs_params['ys']

    positions_3d *= (scale / 2.0)
    positions_3d[:, :, 0] += xs + scale / 2.0
    positions_3d[:, :, 1] += ys + scale / 2.0
    # Z stays as pixel-proportional (no spatial offset, just scaled)

    # Diagnostics in pixel-aligned space
    lsho = positions_3d[:, 11]
    lelb = positions_3d[:, 12]
    lwri = positions_3d[:, 13]
    arm_z = lwri[:, 2] - lsho[:, 2]
    arm_len_ab = np.linalg.norm(lelb - lsho, axis=1)
    arm_len_bc = np.linalg.norm(lwri - lelb, axis=1)
    print(f"  Pixel-aligned — Upper arm: {np.median(arm_len_ab):.1f}px (range {arm_len_ab.min():.1f}-{arm_len_ab.max():.1f})")
    print(f"  Pixel-aligned — Forearm:   {np.median(arm_len_bc):.1f}px (range {arm_len_bc.min():.1f}-{arm_len_bc.max():.1f})")
    print(f"  Pixel-aligned — Arm Z:     min={arm_z.min():.1f} max={arm_z.max():.1f} range={arm_z.max()-arm_z.min():.1f}px")

    return positions_3d


# --- Pipeline Wrapper ---

def detect_poses(
    frames_rgb: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Run full MotionBERT pipeline on pre-extracted RGB frames.

    Pipeline:
      1. YOLOv8 person detection → union bounding box
      2. Stacked Hourglass → MPII (16,3) 2D keypoints + raw heatmaps per frame
      3. MotionBERT → H36M (N,17,3) pixel-aligned 3D
      4. Convert MPII 2D to H36M 2D

    Args:
        frames_rgb: List of (H, W, 3) uint8 RGB frames.

    Returns:
        keypoints_2d: List of (17, 2) pixel coordinates (H36M).
        keypoints_3d: List of (17, 3) pixel-aligned 3D (H36M).
        visibility: List of (17,) confidence scores.
        heatmaps: List of (16, 64, 64) raw MPII heatmaps per frame.
        affine: (2, 3) affine from 256-crop coords to original pixel coords.
    """
    from ultralytics import YOLO

    h, w = frames_rgb[0].shape[:2]
    image_size = (h, w)

    # 1. YOLOv8 person detection → union bounding box
    yolo = YOLO("yolov8n.pt")
    bboxes = []
    print(f"  Detecting persons in {len(frames_rgb)} frames ({w}x{h})...")

    for frame in frames_rgb:
        results = yolo(frame, classes=[0], verbose=False)
        detections = results[0].boxes

        if len(detections) > 0:
            areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                    (detections.xyxy[:, 3] - detections.xyxy[:, 1])
            best_idx = areas.argmax().item()
            bbox = detections.xyxy[best_idx].cpu().numpy().astype(np.float32)
            bboxes.append(bbox)
        else:
            bboxes.append(np.array([0, 0, w, h], dtype=np.float32))

    # Union (max) bounding box across all frames
    all_bboxes = np.array(bboxes)
    max_bbox = np.array([
        all_bboxes[:, 0].min(),
        all_bboxes[:, 1].min(),
        all_bboxes[:, 2].max(),
        all_bboxes[:, 3].max(),
    ], dtype=np.float32)
    bboxes = [max_bbox.copy() for _ in bboxes]
    print(f"  Max bbox: ({max_bbox[0]:.0f}, {max_bbox[1]:.0f}) - ({max_bbox[2]:.0f}, {max_bbox[3]:.0f})")
    print(f"  Detected persons in {len(frames_rgb)} frames")

    # 2. Stacked Hourglass → MPII (16,3) 2D keypoints + raw heatmaps per frame
    all_keypoints_2d, all_heatmaps, affine = run_hourglass(frames_rgb, bboxes, image_size)

    # 3. MotionBERT → H36M (N,17,3) pixel-aligned 3D
    kp_3d_array = run_motionbert(all_keypoints_2d, image_size)  # (N, 17, 3)

    # 4. Convert MPII 2D to H36M 2D + extract visibility
    kp_2d_list = []
    visibility_list = []
    for kp_mpii in all_keypoints_2d:
        kp_h36m = mpii_to_h36m(kp_mpii)  # (17, 3) with confidence
        kp_2d_list.append(kp_h36m[:, :2])  # (17, 2)
        visibility_list.append(kp_h36m[:, 2])  # (17,)

    kp_3d_list = [kp_3d_array[i] for i in range(kp_3d_array.shape[0])]

    return kp_2d_list, kp_3d_list, visibility_list, all_heatmaps, affine


def _build_metric_skeleton(kp_3d: np.ndarray) -> np.ndarray:
    """Reconstruct metric skeleton from MotionBERT directions + DEFAULT_BONE_LENGTHS.

    Takes MotionBERT's pixel-aligned 3D output, extracts bone directions,
    and reconstructs a skeleton with physically-grounded bone lengths.

    Args:
        kp_3d: (17, 3) pixel-aligned 3D from MotionBERT.

    Returns:
        (17, 3) metric skeleton in a local frame (hip at origin).
    """
    from skeleton import DEFAULT_BONE_LENGTHS, PARENTS

    metric = np.zeros((17, 3), dtype=np.float64)
    # Joint 0 (hip) stays at origin

    for j in range(1, 17):
        parent = int(PARENTS[j])
        bone_vec = kp_3d[j] - kp_3d[parent]
        bone_len_px = np.linalg.norm(bone_vec)
        if bone_len_px > 1e-6:
            direction = bone_vec / bone_len_px
        else:
            direction = np.array([0.0, 0.0, 0.0])
        metric[j] = metric[parent] + DEFAULT_BONE_LENGTHS[j] * direction

    return metric


def detector_3d_to_camera(
    kp_3d: np.ndarray,
    kp_2d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Convert MotionBERT pixel-aligned 3D to camera-space meters via solvePnP.

    Reconstructs a metric skeleton (using DEFAULT_BONE_LENGTHS + MotionBERT
    bone directions), then uses cv2.solvePnP to find the rigid transform
    that best aligns the metric 3D points with the 2D detections.

    Falls back to the torso-height heuristic if PnP fails.

    Args:
        kp_3d: (17, 3) pixel-aligned 3D from MotionBERT.
        kp_2d: (17, 2) pixel coordinates.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        (17, 3) positions in camera coordinates (meters).
    """
    # Torso-height heuristic (consistent across all frames)
    thorax_2d = kp_2d[8]
    ankle_mid_2d = (kp_2d[3] + kp_2d[6]) / 2.0
    pixel_height = abs(thorax_2d[1] - ankle_mid_2d[1])

    # Thorax-to-ankle anatomical height from DEFAULT_BONE_LENGTHS:
    # Hip→Spine(0.22) + Spine→Thorax(0.22) + Hip→*Hip(0.12) + *Hip→Knee(0.42) + Knee→Ankle(0.40)
    # = 0.44m (hip to thorax) + 0.94m (hip to ankle) = 1.38m
    assumed_height_m = 1.38
    if pixel_height > 20:
        root_depth = fx * assumed_height_m / pixel_height
    else:
        root_depth = 3.0
    root_depth = float(np.clip(root_depth, 1.0, 8.0))

    root_z_px = kp_3d[0, 2]
    cam_3d = np.zeros((17, 3), dtype=np.float64)
    for j in range(17):
        u = kp_3d[j, 0]
        v = kp_3d[j, 1]
        z_px = kp_3d[j, 2]
        z_cam = root_depth + (z_px - root_z_px) * root_depth / fx
        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy
        cam_3d[j] = [x_cam, y_cam, z_cam]

    return cam_3d
