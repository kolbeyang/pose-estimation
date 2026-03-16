"""Quick single-example test for the MotionBERT pipeline.

Runs one example, prints detailed diagnostics.
"""

import json
import os
import sys
from typing import Any

import numpy as np

import config as cfg
from camera import Camera
from detect import detect_poses, motionbert_to_camera_space
from overlay_video import generate_overlay_video
from evaluate import compute_comparison, compute_comparison_with_optimization, EVAL_JOINT_NAMES_NO_ANKLES
from optimize import run_optimization
from panoptic import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
    world_to_camera,
)
from skeleton import (
    JOINT_NAMES,
    EVAL_JOINTS,
    EVAL_JOINT_NAMES,
    BONES,
    PARENTS,
    DEFAULT_BONE_LENGTHS,
)


def main() -> None:
    """Run single example and print diagnostics."""
    # Use first example by default, or pass index as arg
    example_idx: int = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seq_name: str
    camera_name: str
    start_frame: int
    num_frames: int
    person_idx: int
    seq_name, camera_name, start_frame, num_frames, person_idx = cfg.EXAMPLES[example_idx]

    out_dir: str = os.path.join(cfg.TRAINING_RUNS_DIR, "test_single")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Testing: {seq_name}_{start_frame} (example {example_idx})")

    seq_dir: str = get_sequence_dir(cfg.PANOPTIC_ROOT, seq_name)
    video_path: str = get_video_path(cfg.PANOPTIC_ROOT, seq_name, camera_name)

    # 1. Camera calibration
    cameras: dict[str, dict[str, np.ndarray]] = load_calibration(seq_dir)
    cam_calib: dict[str, np.ndarray] | None = None
    for cname in cameras:
        if cname == camera_name or cname == f"00_{camera_name.split('_')[1]}":
            cam_calib = cameras[cname]
            break
    if cam_calib is None:
        for cname, cal in cameras.items():
            if cname.startswith("00_00"):
                cam_calib = cal
                break
    assert cam_calib is not None, f"Camera {camera_name} not found"

    K: np.ndarray = cam_calib["K"]
    R: np.ndarray = cam_calib["R"]
    t: np.ndarray = cam_calib["t"]
    fx: float = float(K[0, 0])
    fy: float = float(K[1, 1])
    cx: float = float(K[0, 2])
    cy: float = float(K[1, 2])
    resolution: tuple[int, int] = cam_calib["resolution"]
    camera: Camera = Camera.from_panoptic_calibration(K, resolution)

    print(f"Camera: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f} res={resolution}")

    # 2. Extract frames
    video_fps: float = 30.0
    frame_step: int = max(1, int(round(video_fps / cfg.TARGET_FPS)))
    frame_indices: list[int] = list(range(start_frame, start_frame + num_frames, frame_step))
    frames_rgb: list[np.ndarray] = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[:len(frames_rgb)]
    print(f"Extracted {len(frames_rgb)} frames")

    # 3. Detect
    kp_2d: list[np.ndarray]
    kp_3d: list[np.ndarray]
    visibility: list[np.ndarray]
    heatmaps: list[np.ndarray]
    affine: np.ndarray
    positions_3d_norm: np.ndarray
    cs_params: dict[str, float]
    mpii_kp_2d: list[np.ndarray]
    kp_2d, kp_3d, visibility, heatmaps, mpii_kp_2d, affine, positions_3d_norm, cs_params = detect_poses(frames_rgb)

    # 4. Camera-space conversion (new method)
    scale: float = cs_params["scale"]
    det_cam_positions: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        dist_coeffs: np.ndarray | None = cam_calib.get("distCoef")
        pos_cam: np.ndarray = motionbert_to_camera_space(
            positions_3d_norm[i], kp_2d[i], scale, fx, fy, cx, cy,
            dist_coeffs=dist_coeffs,
            visibility=visibility[i],
        )
        det_cam_positions.append(pos_cam)

    # 5. Ground truth
    gt_world: list[np.ndarray | None] = load_ground_truth_sequence(
        seq_dir, frame_indices, person_idx
    )
    gt_cam: list[np.ndarray | None] = []
    for gt in gt_world:
        if gt is not None:
            gt_cam_cm: np.ndarray = world_to_camera(gt, R, t)
            gt_cam.append(gt_cam_cm * 0.01)
        else:
            gt_cam.append(None)

    # 6. Optimize (with improved 2D targets for low-confidence joints)
    print(f"\n=== OPTIMIZATION ===")

    # Replace garbage 2D targets with MotionBERT's projected 2D for low-conf joints
    improved_target_2d: list[np.ndarray] = []
    for i in range(len(frames_rgb)):
        target: np.ndarray = kp_2d[i].copy()
        mb_projected: np.ndarray = camera.world_to_image(det_cam_positions[i])
        for j in range(17):
            if visibility[i][j] < cfg.FK_TARGET_CONF_THRESHOLD:
                target[j] = mb_projected[j]
        improved_target_2d.append(target)

    optimized_3d: list[np.ndarray]
    bone_lengths_final: np.ndarray
    loss_history: list[float]
    optimized_3d, bone_lengths_final, loss_history = run_optimization(
        initial_positions_cam=det_cam_positions,
        target_2d=improved_target_2d,
        visibility=visibility,
        camera=camera,
        heatmaps=heatmaps,
        affine=affine,
    )

    # 7. Evaluate
    metrics: dict[str, Any] = compute_comparison_with_optimization(
        det_cam_positions, optimized_3d, gt_cam,
        camera=camera,
    )

    print(f"\n=== RESULTS ===")
    if "det_mpjpe" in metrics:
        print(f"  Det MPJPE:   {metrics['det_mpjpe']*100:.2f} cm")
        print(f"  Det P-MPJPE: {metrics['det_p_mpjpe']*100:.2f} cm")
    if "det_mpjpe_no_ankles" in metrics:
        print(f"  Det MPJPE (no ankles):   {metrics['det_mpjpe_no_ankles']*100:.2f} cm")
        print(f"  Det P-MPJPE (no ankles): {metrics['det_p_mpjpe_no_ankles']*100:.2f} cm")
    if "opt_mpjpe" in metrics:
        print(f"  Opt MPJPE:   {metrics['opt_mpjpe']*100:.2f} cm")
        print(f"  Opt P-MPJPE: {metrics['opt_p_mpjpe']*100:.2f} cm")
    if "opt_mpjpe_no_ankles" in metrics:
        print(f"  Opt MPJPE (no ankles):   {metrics['opt_mpjpe_no_ankles']*100:.2f} cm")
        print(f"  Opt P-MPJPE (no ankles): {metrics['opt_p_mpjpe_no_ankles']*100:.2f} cm")
    if "improvement" in metrics:
        print(f"  Improvement: {metrics['improvement']*100:+.2f} cm")
    if "det_mpjpe" in metrics:
        print(f"  Frames with GT: {metrics['n_frames_with_gt']}/{metrics['n_frames']}")

    # --- Detailed diagnostics ---
    print(f"\n=== COORDINATE RANGES ===")
    det_arr: np.ndarray = np.array(det_cam_positions)
    print(f"  Detector X: {det_arr[:,:,0].min():.3f} to {det_arr[:,:,0].max():.3f} m")
    print(f"  Detector Y: {det_arr[:,:,1].min():.3f} to {det_arr[:,:,1].max():.3f} m")
    print(f"  Detector Z: {det_arr[:,:,2].min():.3f} to {det_arr[:,:,2].max():.3f} m")

    gt_frames: list[np.ndarray] = [g for g in gt_cam if g is not None]
    if gt_frames:
        gt_arr: np.ndarray = np.array(gt_frames)
        print(f"  GT X: {gt_arr[:,:,0].min():.3f} to {gt_arr[:,:,0].max():.3f} m")
        print(f"  GT Y: {gt_arr[:,:,1].min():.3f} to {gt_arr[:,:,1].max():.3f} m")
        print(f"  GT Z: {gt_arr[:,:,2].min():.3f} to {gt_arr[:,:,2].max():.3f} m")

    # Per-joint errors
    if "det_per_joint" in metrics:
        print(f"\n=== PER-JOINT MPJPE (12 eval joints) ===")
        for i, jname in enumerate(EVAL_JOINT_NAMES):
            print(f"  {jname:<15} {metrics['det_per_joint'][i]*100:.2f} cm")

    # Bone lengths from detector
    print(f"\n=== BONE LENGTHS (detector, frame 0 vs defaults) ===")
    det_f0: np.ndarray = det_cam_positions[0]
    for j in range(1, 17):
        parent: int = int(PARENTS[j])
        det_bl: float = float(np.linalg.norm(det_f0[j] - det_f0[parent]))
        default_bl: float = float(DEFAULT_BONE_LENGTHS[j])
        print(f"  {JOINT_NAMES[j]:<15} det={det_bl:.3f}m  default={default_bl:.3f}m  "
              f"ratio={det_bl/default_bl:.2f}" if default_bl > 0 else
              f"  {JOINT_NAMES[j]:<15} det={det_bl:.3f}m  default={default_bl:.3f}m")

    # 2D reprojection sanity check
    print(f"\n=== 2D REPROJECTION CHECK (frame 0) ===")
    det_2d_reproj: np.ndarray = camera.world_to_image(det_cam_positions[0])
    det_2d_orig: np.ndarray = kp_2d[0]
    reproj_errors: np.ndarray = np.linalg.norm(det_2d_reproj - det_2d_orig, axis=-1)
    print(f"  Mean reprojection error: {reproj_errors.mean():.1f} px")
    print(f"  Max reprojection error:  {reproj_errors.max():.1f} px")

    # Save diagnostics JSON
    diag: dict[str, Any] = {
        "example": f"{seq_name}_{start_frame}",
        "n_frames": len(frames_rgb),
        "n_frames_with_gt": metrics.get("n_frames_with_gt", 0),
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        "detector_coord_ranges": {
            "x": [float(det_arr[:,:,0].min()), float(det_arr[:,:,0].max())],
            "y": [float(det_arr[:,:,1].min()), float(det_arr[:,:,1].max())],
            "z": [float(det_arr[:,:,2].min()), float(det_arr[:,:,2].max())],
        },
    }
    if gt_frames:
        diag["gt_coord_ranges"] = {
            "x": [float(gt_arr[:,:,0].min()), float(gt_arr[:,:,0].max())],
            "y": [float(gt_arr[:,:,1].min()), float(gt_arr[:,:,1].max())],
            "z": [float(gt_arr[:,:,2].min()), float(gt_arr[:,:,2].max())],
        }

    diag_path: str = os.path.join(out_dir, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n  Saved diagnostics: {diag_path}")

    # Generate overlay video
    overlay_path: str = os.path.join(out_dir, f"overlay_{seq_name}_{start_frame}.mp4")
    generate_overlay_video(
        output_path=overlay_path,
        frames_rgb=frames_rgb,
        heatmaps=heatmaps,
        mpii_keypoints_2d=mpii_kp_2d,
        detector_3d=det_cam_positions,
        optimized_3d=optimized_3d,
        camera_fx=fx,
        camera_fy=fy,
        camera_cx=cx,
        camera_cy=cy,
        affine=affine,
        frame_indices=frame_indices,
        gt_3d=gt_cam,
    )
    print(f"  Saved overlay video: {overlay_path}")


if __name__ == "__main__":
    main()
