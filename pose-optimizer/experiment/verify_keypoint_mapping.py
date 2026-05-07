"""Verify keypoint mapping across GT, MotionBERT, and MediaPipe.

Runs both pipelines on a short clip and renders the last frame with all 5
skeleton sources (GT, MB raw, MB opt, MP raw, MP opt) side by side.

Each joint is labeled with its name and whether it is direct or synthesized.
Raw source keypoints that are NOT mapped into our 16-joint skeleton are shown
as gray dots (e.g. COCO19 eyes/ears, MPII HeadTop, MediaPipe face/hands/feet).

For MotionBERT, the FULL 17-joint output is shown (before strip_head_joint),
so HeadTop is visible alongside the synthesized Nose.

Usage:
    cd pose-optimizer
    uv run python experiment/verify_keypoint_mapping.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import Camera
from cmu_data import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
)
from config import OptimizationConfig
from optimize import optimize
from skeleton import BONES, JOINT_NAMES, NUM_JOINTS, PARENTS, mpii_to_skeleton

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQUENCE = os.environ.get("VKM_SEQUENCE", "171204_pose1")
CAMERA = os.environ.get("VKM_CAMERA", "00_00")
START_FRAME = int(os.environ.get("VKM_START_FRAME", "360"))
NUM_FRAMES = int(os.environ.get("VKM_NUM_FRAMES", "10"))
DATA_ROOT = "data/panoptic-toolbox"
OUTPUT_PATH = os.path.expanduser("~/Downloads/verify_keypoint_mapping.png")

MB_OPT = OptimizationConfig(
    num_steps=50, learning_rate=0.001,
    heatmap_blur_sigma_start=16.0, heatmap_blur_sigma_end=4.0, anchor_weight=0.0,
)
MP_OPT = OptimizationConfig(
    num_steps=50, learning_rate=0.0005,
    heatmap_blur_sigma=4.0, anchor_weight=200.0,
)

# ---------------------------------------------------------------------------
# H36M 17-joint names (MotionBERT's native output format)
# ---------------------------------------------------------------------------
H36M_17_NAMES = [
    "Hip",          # 0
    "RHip",         # 1
    "RKnee",        # 2
    "RAnkle",       # 3
    "LHip",         # 4
    "LKnee",        # 5
    "LAnkle",       # 6
    "Spine",        # 7  (Thorax)
    "Neck",         # 8
    "Nose",         # 9  (H36M "Neck/Nose" — joint 14 in H36M 32-joint)
    "HeadTop",      # 10 (H36M "Head" — this gets stripped in our pipeline)
    "LShoulder",    # 11
    "LElbow",       # 12
    "LWrist",       # 13
    "RShoulder",    # 14
    "RElbow",       # 15
    "RWrist",       # 16
]

# H36M 17-joint bone connections for visualization
H36M_17_BONES = [
    (0, 1), (1, 2), (2, 3),      # right leg
    (0, 4), (4, 5), (5, 6),      # left leg
    (0, 7), (7, 8), (8, 9),      # spine → neck → nose
    (9, 10),                       # nose → head top
    (8, 11), (11, 12), (12, 13),  # left arm
    (8, 14), (14, 15), (15, 16),  # right arm
]

# Source-joint metadata (same as before for non-MB panels)
COCO19_NAMES = [
    "Neck", "Nose", "BodyCenter", "LShoulder", "LElbow", "LWrist",
    "LHip", "LKnee", "LAnkle", "RShoulder", "RElbow", "RWrist",
    "RHip", "RKnee", "RAnkle", "LEye", "LEar", "REye", "REar",
]
COCO19_TO_SKEL = {
    2: (0, "direct"), 12: (1, "direct"), 13: (2, "direct"), 14: (3, "direct"),
    6: (4, "direct"), 7: (5, "direct"), 8: (6, "direct"), 0: (8, "direct"),
    1: (9, "direct"), 3: (10, "direct"), 4: (11, "direct"), 5: (12, "direct"),
    9: (13, "direct"), 10: (14, "direct"), 11: (15, "direct"),
}

# MPII 16-joint names (Stacked Hourglass output — NO Nose joint)
MPII_NAMES = [
    "RAnkle", "RKnee", "RHip", "LHip", "LKnee", "LAnkle",
    "Pelvis", "Thorax", "UpperNeck", "HeadTop", "RWrist", "RElbow",
    "RShoulder", "LShoulder", "LElbow", "LWrist",
]

MP_NAMES = [
    "Nose", "LEyeIn", "LEye", "LEyeOut", "REyeIn", "REye",
    "REyeOut", "LEar", "REar", "MouthL", "MouthR",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist",
    "RWrist", "LPinky", "RPinky", "LIndex", "RIndex",
    "LThumb", "RThumb", "LHip", "RHip", "LKnee", "RKnee",
    "LAnkle", "RAnkle", "LHeel", "RHeel", "LFootIdx", "RFootIdx",
]
MP_TO_SKEL = {
    0: (9, "direct"), 11: (10, "direct"), 12: (13, "direct"),
    13: (11, "direct"), 14: (14, "direct"), 15: (12, "direct"),
    16: (15, "direct"), 23: (4, "direct"), 24: (1, "direct"),
    25: (5, "direct"), 26: (2, "direct"), 27: (6, "direct"), 28: (3, "direct"),
}

SYNTH_NOTES = {
    "GT": {7: "synth: mid(Pelvis,Neck)"},
    "MB Raw 17j": {},  # will label per-joint in draw_panel_17j
    "MB Opt": {9: "synth: 30% UpperNeck(MPII[8])→HeadTop(MPII[9])"},
    "MP Raw": {
        0: "synth: mid(LHip,RHip)", 7: "synth: mid(Pelvis,Neck)",
        8: "synth: mid(LSho,RSho)",
    },
    "MP Opt": {
        0: "synth: mid(LHip,RHip)", 7: "synth: mid(Pelvis,Neck)",
        8: "synth: mid(LSho,RSho)",
    },
}

COLORS = {
    "GT":           (30, 144, 255),
    "SH Heatmaps":  (255, 200, 0),
    "MB Raw 17j":   (50, 205, 50),
    "MB Opt":       (0, 128, 0),
    "MP Raw":       (255, 165, 0),
    "MP Opt":       (220, 20, 60),
}
UNMAPPED_COLOR = (160, 160, 160)


def project_3d_to_2d(pts_3d, fx, fy, cx, cy):
    pts = np.asarray(pts_3d, dtype=np.float64)
    Z = np.maximum(pts[:, 2], 0.01)
    u = fx * pts[:, 0] / Z + cx
    v = fy * pts[:, 1] / Z + cy
    return np.stack([u, v], axis=-1)


def _c01(rgb):
    return tuple(c / 255.0 for c in rgb)


def draw_panel_16j(ax, frame_rgb, skel_2d, color_rgb, panel_name,
                   extra_unmapped=None, hide_synth=False):
    """Draw a 16-joint skeleton panel with synth/direct labels.

    Args:
        hide_synth: If True, omit synthesized joints and any bones connecting
            to them. Use this for the GT panel so we don't display fabricated
            ground-truth points (e.g. Spine, which COCO19 does not annotate).
    """
    ax.imshow(frame_rgb)
    c = _c01(color_rgb)
    uc = _c01(UNMAPPED_COLOR)
    synth = SYNTH_NOTES.get(panel_name, {})
    skip = set(synth.keys()) if hide_synth else set()

    # When hiding synthesized joints, bridge the kinematic chain by
    # connecting their parent directly to their children. Both endpoints
    # of a bridged bone are real (non-skipped) joints.
    bones_to_draw = list(BONES)
    if skip:
        for j in skip:
            up = int(PARENTS[j])
            # walk up if parent is also skipped
            while up in skip and up != 0:
                up = int(PARENTS[up])
            if up in skip:
                continue
            for i in range(NUM_JOINTS):
                if i in skip or i == j:
                    continue
                if int(PARENTS[i]) == j:
                    bones_to_draw.append((up, i))

    for parent, child in bones_to_draw:
        if parent in skip or child in skip:
            continue
        ax.plot(
            [skel_2d[parent, 0], skel_2d[child, 0]],
            [skel_2d[parent, 1], skel_2d[child, 1]],
            color=c, linewidth=1.5, alpha=0.9,
        )
    for j in range(NUM_JOINTS):
        if j in skip:
            continue
        x, y = skel_2d[j]
        note = synth.get(j, "direct")
        label = f"{j}:{JOINT_NAMES[j]} [{note}]"
        is_synth = j in synth
        marker = "D" if is_synth else "o"
        ax.plot(x, y, marker, color=c, markersize=5 if is_synth else 4)
        ax.annotate(
            label, (x, y), textcoords="offset points", xytext=(5, 5),
            fontsize=4.5, color=c, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )
    if extra_unmapped:
        for x, y, label in extra_unmapped:
            ax.plot(x, y, "x", color=uc, markersize=4, alpha=0.6)
            ax.annotate(
                label, (x, y), textcoords="offset points", xytext=(5, -8),
                fontsize=3.5, color=uc, alpha=0.6,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.4),
            )
    ax.set_title(panel_name, fontsize=14, fontweight="bold")
    ax.axis("off")


def draw_panel_17j(ax, frame_rgb, pts_2d_17, color_rgb, panel_name):
    """Draw MotionBERT's full 17-joint output with H36M joint names.

    Joint 9 (Nose/Neck) and joint 10 (HeadTop) are both shown — joint 10
    is the one that normally gets stripped by strip_head_joint().
    """
    ax.imshow(frame_rgb)
    c = _c01(color_rgb)

    # Draw H36M bones
    for parent, child in H36M_17_BONES:
        ax.plot(
            [pts_2d_17[parent, 0], pts_2d_17[child, 0]],
            [pts_2d_17[parent, 1], pts_2d_17[child, 1]],
            color=c, linewidth=1.5, alpha=0.9,
        )

    # Draw all 17 joints with labels
    for j in range(17):
        x, y = pts_2d_17[j]
        name = H36M_17_NAMES[j]

        # Annotate special joints
        if j == 9:
            note = "H36M 'Neck/Nose' — 2D input was synth 30% UpperNeck(MPII[8])→HeadTop(MPII[9])"
        elif j == 10:
            note = "HeadTop — STRIPPED in our 16j skeleton"
        else:
            note = ""

        label = f"{j}:{name}"
        if note:
            label += f"\n  [{note}]"

        # Highlight the interesting joints
        if j == 9:
            marker, ms = "*", 10  # star for the controversial joint
        elif j == 10:
            marker, ms = "s", 6   # square for stripped joint
        else:
            marker, ms = "o", 4

        ax.plot(x, y, marker, color=c, markersize=ms)
        ax.annotate(
            label, (x, y), textcoords="offset points", xytext=(5, 5),
            fontsize=4.5, color=c, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7),
        )

    ax.set_title(panel_name, fontsize=14, fontweight="bold")
    ax.axis("off")


def draw_panel_sh_heatmaps(ax, frame_rgb, heatmaps_16, affine_mat):
    """Draw Stacked Hourglass heatmap overlay with per-channel labels.

    Shows all 16 MPII heatmap channels combined, with each channel's argmax
    labeled with its MPII joint name.

    Args:
        heatmaps_16: (16, 64, 64) Stacked Hourglass heatmaps.
        affine_mat: (2, 3) affine mapping crop coords to pixel coords.
    """
    import cv2

    h, w = frame_rgb.shape[:2]

    # Build combined heatmap overlay (same as overlay_video.py)
    hm_combined = np.clip(heatmaps_16, 0, None).sum(axis=0)  # (64, 64)

    # Resize to frame using affine
    sx, sy = float(affine_mat[0, 0]), float(affine_mat[1, 1])
    tx, ty = float(affine_mat[0, 2]), float(affine_mat[1, 2])
    crop_w = max(1, int(round(256 * sx)))
    crop_h = max(1, int(round(256 * sy)))
    resized = cv2.resize(hm_combined, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    full_hm = np.zeros((h, w), dtype=np.float32)
    dx0, dy0 = int(round(tx)), int(round(ty))
    dx1, dy1 = dx0 + crop_w, dy0 + crop_h
    sx0 = max(0, -dx0)
    sy0 = max(0, -dy0)
    sx1 = crop_w - max(0, dx1 - w)
    sy1 = crop_h - max(0, dy1 - h)
    px0, py0 = max(0, dx0), max(0, dy0)
    px1, py1 = min(w, dx1), min(h, dy1)
    if px1 > px0 and py1 > py0 and sx1 > sx0 and sy1 > sy0:
        full_hm[py0:py1, px0:px1] = resized[sy0:sy1, sx0:sx1]

    # Blend onto frame using HOT colormap
    maxval = float(full_hm.max())
    if maxval > 1e-6:
        norm = np.clip(full_hm / maxval, 0, 1)
        hm_uint8 = (norm * 255).astype(np.uint8)
        hm_color_bgr = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_HOT)
        hm_color_rgb = hm_color_bgr[:, :, ::-1]  # BGR → RGB
        intensity = 200.0
        glow = hm_color_rgb.astype(np.float32) * (norm[:, :, np.newaxis] * intensity / 255.0)
        blended = np.clip(frame_rgb.astype(np.float32) + glow, 0, 255).astype(np.uint8)
    else:
        blended = frame_rgb

    ax.imshow(blended)

    # For each of the 16 MPII channels, find argmax and label it
    for ch in range(16):
        hm_ch = heatmaps_16[ch]  # (64, 64)
        peak_val = float(hm_ch.max())
        if peak_val < 0.01:
            continue
        # Argmax in 64x64 space
        flat_idx = int(np.argmax(hm_ch))
        y64, x64 = divmod(flat_idx, 64)
        # Map from 64x64 heatmap to 256x256 crop space (4x scale)
        x256 = x64 * 4.0 + 2.0
        y256 = y64 * 4.0 + 2.0
        # Apply affine to get pixel coords
        px = sx * x256 + tx
        py = sy * y256 + ty

        label = f"MPII[{ch}]:{MPII_NAMES[ch]}"

        ax.plot(px, py, "o", color="white", markersize=5,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(
            label, (px, py), textcoords="offset points", xytext=(5, 5),
            fontsize=4.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.7),
        )

    ax.set_title("SH Heatmaps (16 MPII)", fontsize=14, fontweight="bold")
    ax.axis("off")


def run_motionbert_17j(mpii_kp_2d_list, model, device):
    """Run MotionBERT and return the FULL 17-joint output (no stripping).

    This duplicates the core inference from run_motionbert/detect.py but
    skips strip_head_joint() so we can see all 17 joints.
    """
    from run_motionbert.detect import crop_scale

    n_frames = len(mpii_kp_2d_list)
    keypoints_17 = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for i, kp_mpii in enumerate(mpii_kp_2d_list):
        keypoints_17[i] = mpii_to_skeleton(kp_mpii)

    keypoints_norm, cs_params = crop_scale(keypoints_17)

    input_tensor = torch.from_numpy(keypoints_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        output_3d = model(input_tensor)

    positions_3d_17 = output_3d.cpu().numpy()[0]  # (N, 17, 3)
    positions_3d_17[0, 0, 2] = 0  # match pipeline behavior

    return positions_3d_17  # (N, 17, 3) — NOT stripped


def motionbert_17j_to_camera_space(positions_3d_norm_17, kp_2d_16, fx, fy, cx, cy):
    """Convert 17-joint MotionBERT output to camera-space meters.

    Same logic as motionbert_to_camera_space but operates on 17 joints.
    Uses PARENTS from the 16-joint skeleton for bone length estimation
    (on joints 0-15, ignoring joint 10/HeadTop for scale).
    """
    from skeleton import PARENTS, DEFAULT_BONE_LENGTHS
    from run_motionbert.detect import _iqr_filtered_median

    # Strip to 16 for scale estimation (same as pipeline)
    pos16 = np.delete(positions_3d_norm_17, 10, axis=0)  # (16, 3)
    root_relative_16 = pos16 - pos16[0:1]

    _RELIABLE_BONES = {10, 11, 12, 13, 14, 15}
    reliable_ratios = []
    all_ratios = []
    for j in range(1, 16):
        p = int(PARENTS[j])
        det_bl = float(np.linalg.norm(root_relative_16[j] - root_relative_16[p]))
        ref_bl = float(DEFAULT_BONE_LENGTHS[j])
        if det_bl > 1e-4 and ref_bl > 1e-4:
            ratio = ref_bl / det_bl
            all_ratios.append(ratio)
            if j in _RELIABLE_BONES:
                reliable_ratios.append(ratio)

    if reliable_ratios:
        scale = float(np.median(reliable_ratios))
    elif all_ratios:
        scale = float(np.median(all_ratios))
    else:
        scale = 1.0

    # Scale the full 17-joint output
    root_relative_17 = positions_3d_norm_17 - positions_3d_norm_17[0:1]
    scaled_17 = root_relative_17 * scale  # (17, 3) in meters, root-relative

    # Estimate root depth from Y-axis correspondences (same as pipeline)
    tz_estimates = []
    for j in range(1, 16):  # use 16-joint indices for 2D
        p = int(PARENTS[j])
        # Map 17j index to 16j index for 2D lookup
        j16 = j if j < 10 else j - 1  # skip HeadTop
        p16 = p if p < 10 else p - 1
        dy_3d = scaled_17[j if j < 10 else j + 1, 1] - scaled_17[p if p < 10 else p + 1, 1]
        dv_2d = kp_2d_16[j16, 1] - kp_2d_16[p16, 1]
        if abs(dv_2d) > 2.0 and abs(dy_3d) > 0.001:
            tz_estimates.append(fy * dy_3d / dv_2d)

    if tz_estimates:
        root_tz = _iqr_filtered_median(np.array(tz_estimates))
    else:
        root_tz = 5.0

    root_tx = (kp_2d_16[0, 0] - cx) * root_tz / fx
    root_ty = (kp_2d_16[0, 1] - cy) * root_tz / fy

    cam_17 = scaled_17.copy()
    cam_17[:, 0] += root_tx
    cam_17[:, 1] += root_ty
    cam_17[:, 2] += root_tz

    return cam_17  # (17, 3) in camera space meters


def main():
    seq_dir = get_sequence_dir(DATA_ROOT, SEQUENCE)
    video_path = get_video_path(DATA_ROOT, SEQUENCE, CAMERA)

    # --- Camera ---
    cameras = load_calibration(seq_dir)
    cam_calib = cameras[CAMERA]
    K, R, t = cam_calib["K"], cam_calib["R"], cam_calib["t"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    resolution = cam_calib["resolution"]
    camera = Camera.from_panoptic_calibration(K, R, t, resolution)

    # --- Frames ---
    frame_indices = list(range(START_FRAME, START_FRAME + NUM_FRAMES))
    print(f"Extracting {len(frame_indices)} frames: {frame_indices[0]}-{frame_indices[-1]}")
    frames_rgb = extract_video_frames(video_path, frame_indices)

    # --- Ground truth (16-joint + raw COCO19) ---
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx=0)
    gt_cam = []
    for gt in gt_world:
        if gt is not None:
            gt_cam.append(camera.world_to_camera(gt) * 0.01)
        else:
            gt_cam.append(None)

    idx = len(frames_rgb) - 1

    # Load raw COCO19 for the last frame
    gt_json_path = os.path.join(
        seq_dir, "hdPose3d_stage1_coco19",
        f"body3DScene_{frame_indices[idx]:08d}.json",
    )
    coco19_2d = None
    if os.path.exists(gt_json_path):
        with open(gt_json_path) as f:
            gt_data = json.load(f)
        bodies = gt_data.get("bodies", [])
        if bodies:
            j19 = np.array(bodies[0]["joints19"]).reshape(19, 4)[:, :3]
            j19_cam = camera.world_to_camera(j19) * 0.01
            coco19_2d = project_3d_to_2d(j19_cam, fx, fy, cx, cy)

    # --- Shared 2D detection ---
    print("Running YOLO + Stacked Hourglass...")
    from run_motionbert.detect import (
        load_yolo_sh_models, load_motionbert_model, detect_2d_poses,
        motionbert_to_camera_space,
    )
    yolo_sh = load_yolo_sh_models()
    kp_2d, visibility, heatmaps, mpii_kp_2d, affine = detect_2d_poses(
        frames_rgb, yolo_sh, sh_batch_size=32,
    )

    # --- MotionBERT: get FULL 17-joint output ---
    print("Running MotionBERT (17-joint, no stripping)...")
    mb_model = load_motionbert_model()
    mb_device = yolo_sh.device
    mb_model = mb_model.to(mb_device)

    # Full 17-joint output
    positions_3d_17 = run_motionbert_17j(mpii_kp_2d, mb_model, mb_device)

    # Log all 17 joints for the last frame
    print(f"\n{'='*60}")
    print(f"MotionBERT RAW 17-joint output (frame {frame_indices[idx]}):")
    print(f"{'='*60}")
    for j in range(17):
        p = positions_3d_17[idx, j]
        print(f"  [{j:2d}] {H36M_17_NAMES[j]:12s}  x={p[0]:+8.4f}  y={p[1]:+8.4f}  z={p[2]:+8.4f}")
    print(f"{'='*60}\n")

    # Convert 17-joint to camera space for the last frame
    mb_cam_17 = [
        motionbert_17j_to_camera_space(positions_3d_17[i], kp_2d[i], fx, fy, cx, cy)
        for i in range(len(frames_rgb))
    ]

    # Also get the standard 16-joint pipeline output for optimization
    from skeleton import strip_head_joint
    positions_3d_16 = strip_head_joint(positions_3d_17)
    det_cam_mb = [
        motionbert_to_camera_space(positions_3d_16[i], kp_2d[i], fx, fy, cx, cy)
        for i in range(len(frames_rgb))
    ]

    print("  Optimizing MotionBERT...")
    opt_mb, _, _ = optimize(
        raw_3d=det_cam_mb, camera=camera, config=MB_OPT,
        heatmaps=heatmaps, affine=affine, visibility=visibility, verbose=False,
    )

    # --- MediaPipe ---
    print("Running MediaPipe...")
    from run_mediapipe.detect import load_landmarker, detect_poses as mp_detect_poses, mediapipe_3d_to_camera
    import mediapipe as mediapipe_lib

    landmarker = load_landmarker()
    mp_kp_2d, mp_kp_3d, _ = mp_detect_poses(frames_rgb, landmarker=landmarker)
    det_cam_mp = [
        mediapipe_3d_to_camera(mp_kp_3d[i], mp_kp_2d[i], fx, fy, cx, cy)
        for i in range(len(frames_rgb))
    ]
    print("  Optimizing MediaPipe...")
    opt_mp, _, _ = optimize(
        raw_3d=det_cam_mp, camera=camera, config=MP_OPT,
        heatmaps=heatmaps, affine=affine, visibility=visibility, verbose=False,
    )

    # --- Get raw MediaPipe 33 landmarks for the last frame ---
    last_frame = frames_rgb[idx]
    h, w = last_frame.shape[:2]
    mp_image = mediapipe_lib.Image(
        image_format=mediapipe_lib.ImageFormat.SRGB, data=last_frame,
    )
    mp_result = landmarker.detect(mp_image)
    mp33_2d = None
    if mp_result.pose_landmarks and len(mp_result.pose_landmarks) > 0:
        pose_lm = mp_result.pose_landmarks[0]
        mp33_2d = np.array(
            [[lm.x * w, lm.y * h] for lm in pose_lm], dtype=np.float64,
        )

    # --- Build per-panel data ---
    frame_rgb = frames_rgb[idx]

    # Project all skeleton sources to 2D
    gt_2d = project_3d_to_2d(gt_cam[idx], fx, fy, cx, cy) if gt_cam[idx] is not None else None
    mb_17_2d = project_3d_to_2d(mb_cam_17[idx], fx, fy, cx, cy)  # full 17 joints!
    mb_opt_2d = project_3d_to_2d(opt_mb[idx], fx, fy, cx, cy)
    mp_raw_2d = project_3d_to_2d(det_cam_mp[idx], fx, fy, cx, cy)
    mp_opt_2d = project_3d_to_2d(opt_mp[idx], fx, fy, cx, cy)

    # Log the 17-joint 2D positions
    print(f"\nMotionBERT 17-joint projected to 2D (frame {frame_indices[idx]}):")
    for j in range(17):
        print(f"  [{j:2d}] {H36M_17_NAMES[j]:12s}  u={mb_17_2d[j,0]:7.1f}  v={mb_17_2d[j,1]:7.1f}")

    # --- Compute crop region from all keypoints ---
    all_pts_list = [mb_17_2d, mb_opt_2d, mp_raw_2d, mp_opt_2d]
    if gt_2d is not None:
        all_pts_list.append(gt_2d)
    all_pts = np.concatenate(all_pts_list, axis=0)
    margin = 100
    x_min = max(0, int(all_pts[:, 0].min()) - margin)
    x_max = min(frame_rgb.shape[1], int(all_pts[:, 0].max()) + margin)
    y_min = max(0, int(all_pts[:, 1].min()) - margin)
    y_max = min(frame_rgb.shape[0], int(all_pts[:, 1].max()) + margin)

    # --- Render 6 panels ---
    # Order: GT | SH Heatmaps | MB Raw 17j | MB Opt | MP Raw | MP Opt
    panel_count = 6 if gt_2d is not None else 5
    fig, axes = plt.subplots(1, panel_count, figsize=(7 * panel_count, 10))
    ax_idx = 0

    # Panel 1: GT
    if gt_2d is not None:
        gt_unmapped = []
        if coco19_2d is not None:
            for ci in range(19):
                if ci not in COCO19_TO_SKEL:
                    gt_unmapped.append((
                        coco19_2d[ci, 0], coco19_2d[ci, 1],
                        f"COCO[{ci}]:{COCO19_NAMES[ci]} [unmapped]",
                    ))
        draw_panel_16j(axes[ax_idx], frame_rgb, gt_2d, COLORS["GT"], "GT",
                       extra_unmapped=gt_unmapped, hide_synth=True)
        axes[ax_idx].set_xlim(x_min, x_max)
        axes[ax_idx].set_ylim(y_max, y_min)
        ax_idx += 1

    # Panel 2: Stacked Hourglass heatmaps
    draw_panel_sh_heatmaps(axes[ax_idx], frame_rgb, heatmaps[idx], affine)
    axes[ax_idx].set_xlim(x_min, x_max)
    axes[ax_idx].set_ylim(y_max, y_min)
    ax_idx += 1

    # Panel 3: MB Raw — FULL 17 joints
    draw_panel_17j(axes[ax_idx], frame_rgb, mb_17_2d, COLORS["MB Raw 17j"], "MB Raw 17j")
    axes[ax_idx].set_xlim(x_min, x_max)
    axes[ax_idx].set_ylim(y_max, y_min)
    ax_idx += 1

    # Panel 4: MB Opt — 16 joints
    draw_panel_16j(axes[ax_idx], frame_rgb, mb_opt_2d, COLORS["MB Opt"], "MB Opt")
    axes[ax_idx].set_xlim(x_min, x_max)
    axes[ax_idx].set_ylim(y_max, y_min)
    ax_idx += 1

    # Panel 5: MP Raw — 16 joints + unmapped landmarks
    mp_unmapped = []
    if mp33_2d is not None:
        for mi in range(33):
            if mi not in MP_TO_SKEL:
                mp_unmapped.append((
                    mp33_2d[mi, 0], mp33_2d[mi, 1],
                    f"MP[{mi}]:{MP_NAMES[mi]}",
                ))
    draw_panel_16j(axes[ax_idx], frame_rgb, mp_raw_2d, COLORS["MP Raw"], "MP Raw",
                   extra_unmapped=mp_unmapped)
    axes[ax_idx].set_xlim(x_min, x_max)
    axes[ax_idx].set_ylim(y_max, y_min)
    ax_idx += 1

    # Panel 6: MP Opt — 16 joints
    draw_panel_16j(axes[ax_idx], frame_rgb, mp_opt_2d, COLORS["MP Opt"], "MP Opt")
    axes[ax_idx].set_xlim(x_min, x_max)
    axes[ax_idx].set_ylim(y_max, y_min)

    fig.suptitle(
        f"Keypoint Mapping Verification — {SEQUENCE} frame {frame_indices[idx]}",
        fontsize=18, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
