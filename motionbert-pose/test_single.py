"""Quick single-example test for optimization tuning.

Runs one example, saves a mid-frame overlay image and diagnostic JSON.
"""

import json
import os
import sys

import cv2
import numpy as np

import config as cfg
from detect import detect_poses, detector_3d_to_camera
from evaluate import compute_comparison
from model.camera import Camera
from optimize import run_optimization
from panoptic import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
    world_to_camera,
)
from skeleton import JOINT_NAMES, BONES


def draw_skeleton_on_frame(frame, kp_2d, color, label, offset_y=0):
    """Draw skeleton overlay on frame."""
    for parent, child in BONES:
        pt1 = (int(kp_2d[parent, 0]), int(kp_2d[parent, 1]))
        pt2 = (int(kp_2d[child, 0]), int(kp_2d[child, 1]))
        cv2.line(frame, pt1, pt2, color, 2)
    for j in range(17):
        pt = (int(kp_2d[j, 0]), int(kp_2d[j, 1]))
        cv2.circle(frame, pt, 3, color, -1)
    cv2.putText(frame, label, (10, 25 + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    # Use first example by default, or pass index as arg
    example_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seq_name, camera_name, start_frame, num_frames, person_idx = cfg.EXAMPLES[example_idx]

    out_dir = os.path.join(cfg.TRAINING_RUNS_DIR, "test_single")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Testing: {seq_name}_{start_frame} (example {example_idx})")

    seq_dir = get_sequence_dir(cfg.PANOPTIC_ROOT, seq_name)
    video_path = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)

    # 1. Camera
    cameras = load_calibration(seq_dir)
    cam_calib = None
    for cname in cameras:
        if cname == camera_name or cname == f"00_{camera_name.split('_')[1]}":
            cam_calib = cameras[cname]
            break
    if cam_calib is None:
        for cname, cal in cameras.items():
            if cname.startswith("00_00"):
                cam_calib = cal
                break
    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    resolution = cam_calib["resolution"]
    camera = Camera(fx=fx, fy=fy, cx=cx, cy=cy, image_size=(resolution[1], resolution[0]))

    # 2. Extract frames
    video_fps = 30.0
    frame_step = max(1, int(round(video_fps / cfg.TARGET_FPS)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    frames_rgb = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[:len(frames_rgb)]

    # 3. Detect
    kp_2d, kp_3d, visibility, heatmaps, affine = detect_poses(frames_rgb)

    # 4. Camera-space conversion
    mp_cam_positions = []
    for i in range(len(frames_rgb)):
        pos_cam = detector_3d_to_camera(kp_3d[i], kp_2d[i], fx, fy, cx, cy)
        mp_cam_positions.append(pos_cam)

    # 5. Ground truth
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam = []
    for gt in gt_world:
        if gt is not None:
            gt_cam_cm = world_to_camera(gt, R, t)
            gt_cam.append(gt_cam_cm * 0.01)
        else:
            gt_cam.append(None)

    # 5b. Verify heatmap coordinate transform
    import torch
    from scoring import prepare_heatmaps, heatmap_score, _JOINTS_WITH_HEATMAP
    from optimize import _compute_affine_pixel_to_hm

    affine_pixel_to_hm = _compute_affine_pixel_to_hm(affine)
    log_hm_raw = prepare_heatmaps(heatmaps, blur_sigma=0.0)

    # Score the detected 2D points against the heatmaps (should be near-zero = peak)
    mid = len(frames_rgb) // 2
    det_2d_t = torch.tensor(kp_2d[mid], dtype=torch.float32)
    vis_t = torch.tensor(visibility[mid], dtype=torch.float32)
    det_score = heatmap_score(det_2d_t, log_hm_raw[mid], vis_t, affine_pixel_to_hm)
    print(f"\n=== HEATMAP VERIFICATION (frame {mid}) ===")
    print(f"  Score for detected 2D points: {det_score.item():.2f}")
    print(f"  (should be close to 0 if transform is correct)")
    n_active = sum(1 for h, _ in _JOINTS_WITH_HEATMAP if vis_t[h] >= 0.01)
    print(f"  Active joints: {n_active}, per-joint avg: {det_score.item()/max(n_active,1):.2f}")

    # Also score with blur=8
    log_hm_blur8 = prepare_heatmaps(heatmaps, blur_sigma=8.0)
    det_score_blur = heatmap_score(det_2d_t, log_hm_blur8[mid], vis_t, affine_pixel_to_hm)
    print(f"  Score for detected 2D (blur=8): {det_score_blur.item():.2f}")
    print(f"  Per-joint avg (blur=8): {det_score_blur.item()/max(n_active,1):.2f}")

    # Score the initial projected 2D
    init_2d_check = camera.world_to_image_torch(
        torch.tensor(mp_cam_positions[mid], dtype=torch.float32)
    )
    init_score = heatmap_score(init_2d_check, log_hm_raw[mid], vis_t, affine_pixel_to_hm)
    print(f"  Score for init projected 2D: {init_score.item():.2f}")
    print(f"  Per-joint avg: {init_score.item()/max(n_active,1):.2f}")

    # 6. Optimize
    result = run_optimization(
        initial_positions_cam=mp_cam_positions,
        heatmaps_raw=heatmaps,
        affine_256_to_pixel=affine,
        visibility=visibility,
        camera=camera,
    )

    # 7. Evaluate
    metrics = compute_comparison(result.mediapipe_3d, result.optimized_3d, gt_cam)

    print(f"\n=== RESULTS ===")
    if "mp_mpjpe" in metrics:
        print(f"  MP  MPJPE:   {metrics['mp_mpjpe']*100:.2f} cm")
        print(f"  Opt MPJPE:   {metrics['opt_mpjpe']*100:.2f} cm")
        print(f"  MP  P-MPJPE: {metrics['mp_p_mpjpe']*100:.2f} cm")
        print(f"  Opt P-MPJPE: {metrics['opt_p_mpjpe']*100:.2f} cm")

    # 8. Save mid-frame overlay
    mid = len(frames_rgb) // 2
    frame_bgr = cv2.cvtColor(frames_rgb[mid], cv2.COLOR_RGB2BGR)

    # Project GT to 2D
    if gt_cam[mid] is not None:
        import torch
        gt_2d = camera.world_to_image_torch(
            torch.tensor(gt_cam[mid], dtype=torch.float32)
        ).detach().numpy()
        draw_skeleton_on_frame(frame_bgr, gt_2d, (0, 255, 0), "GT", offset_y=0)

    # Project initial (detector) to 2D
    import torch
    init_2d = camera.world_to_image_torch(
        torch.tensor(result.mediapipe_3d[mid], dtype=torch.float32)
    ).detach().numpy()
    draw_skeleton_on_frame(frame_bgr, init_2d, (0, 0, 255), "Init", offset_y=25)

    # Project optimized to 2D
    opt_2d = camera.world_to_image_torch(
        torch.tensor(result.optimized_3d[mid], dtype=torch.float32)
    ).detach().numpy()
    draw_skeleton_on_frame(frame_bgr, opt_2d, (255, 0, 0), "Opt", offset_y=50)

    # Also draw raw 2D detections
    det_2d = kp_2d[mid]
    for j in range(17):
        pt = (int(det_2d[j, 0]), int(det_2d[j, 1]))
        cv2.circle(frame_bgr, pt, 5, (0, 255, 255), 1)  # yellow circles

    overlay_path = os.path.join(out_dir, "mid_frame_overlay.png")
    cv2.imwrite(overlay_path, frame_bgr)
    print(f"  Saved overlay: {overlay_path}")

    # 9. Save diagnostic JSON
    diag = {
        "example": f"{seq_name}_{start_frame}",
        "n_frames": len(frames_rgb),
        "config": {
            "NUM_STEPS": cfg.NUM_STEPS,
            "LEARNING_RATE": cfg.LEARNING_RATE,
            "BONE_LENGTH_LR": cfg.BONE_LENGTH_LR,
            "GRAD_CLIP_NORM": cfg.GRAD_CLIP_NORM,
            "BLUR_PHASES": cfg.BLUR_PHASES,
            "POSITION_PENALTY_WEIGHT": cfg.POSITION_PENALTY_WEIGHT,
            "ROTATION_PENALTY_WEIGHT": cfg.ROTATION_PENALTY_WEIGHT,
            "BONE_LENGTH_REG_WEIGHT": cfg.BONE_LENGTH_REG_WEIGHT,
        },
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        "loss_history": result.loss_history,
        "score_details_history": result.score_details_history,
        "bone_lengths_final": result.bone_lengths_final.tolist(),
        "mid_frame_idx": mid,
    }
    # Per-frame data for the mid frame
    diag["mid_frame"] = {
        "init_3d": result.mediapipe_3d[mid].tolist(),
        "opt_3d": result.optimized_3d[mid].tolist(),
        "gt_3d": gt_cam[mid].tolist() if gt_cam[mid] is not None else None,
        "kp_2d_detected": kp_2d[mid].tolist(),
        "init_2d_projected": init_2d.tolist(),
        "opt_2d_projected": opt_2d.tolist(),
        "gt_2d_projected": gt_2d.tolist() if gt_cam[mid] is not None else None,
    }

    # Per-joint errors if available
    if "mp_per_joint" in metrics:
        diag["mp_per_joint"] = metrics["mp_per_joint"].tolist() if hasattr(metrics["mp_per_joint"], "tolist") else metrics["mp_per_joint"]
        diag["opt_per_joint"] = metrics["opt_per_joint"].tolist() if hasattr(metrics["opt_per_joint"], "tolist") else metrics["opt_per_joint"]
    if "mp_per_frame_mpjpe" in metrics:
        diag["mp_per_frame_mpjpe"] = metrics["mp_per_frame_mpjpe"].tolist() if hasattr(metrics["mp_per_frame_mpjpe"], "tolist") else metrics["mp_per_frame_mpjpe"]
        diag["opt_per_frame_mpjpe"] = metrics["opt_per_frame_mpjpe"].tolist() if hasattr(metrics["opt_per_frame_mpjpe"], "tolist") else metrics["opt_per_frame_mpjpe"]

    diag_path = os.path.join(out_dir, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"  Saved diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
