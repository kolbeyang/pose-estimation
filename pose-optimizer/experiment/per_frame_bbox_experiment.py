"""Per-frame bbox vs union bbox experiment.

Compares Stacked Hourglass with a single union bounding box (baseline)
against per-frame bounding boxes on sequences where the person moves
significantly, as well as a control sequence.

Usage:
    cd pose-optimizer
    uv run python experiment/per_frame_bbox_experiment.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from camera import Camera
from cmu_data import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
)
from config import OptimizationConfig
from evaluate import evaluate
from optimize import optimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Example definitions
# ---------------------------------------------------------------------------

EXAMPLES = [
    {
        "name": "pose2 (moving person)",
        "seq": "171026_pose2",
        "camera": "00_00",
        "start_frame": 109,
        "num_frames": 500,
        "person_idx": 0,
    },
    {
        "name": "pose1 (control - stationary)",
        "seq": "171204_pose1",
        "camera": "00_00",
        "start_frame": 360,
        "num_frames": 360,
        "person_idx": 0,
    },
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class ExampleData:
    """Pre-loaded data for one example with both bbox modes."""
    name: str
    frames_rgb: list
    frame_indices: list[int]

    # Union bbox results
    heatmaps_union: list
    kp_2d_union: list
    visibility_union: list
    affine_union: np.ndarray
    mpii_kp_2d_union: list

    # Per-frame bbox results
    heatmaps_perframe: list
    kp_2d_perframe: list
    visibility_perframe: list
    affine_perframe: list[np.ndarray]
    mpii_kp_2d_perframe: list

    # Shared
    camera: Camera
    fx: float
    fy: float
    cx: float
    cy: float
    gt_cam: list[Any]


def load_example(
    example: dict,
    yolo_sh_models: Any,
    target_fps: float = 10.0,
    data_root: str = "data/panoptic-toolbox",
) -> ExampleData:
    """Load one example and run detection with both bbox modes."""
    from run_motionbert.detect import detect_2d_poses

    seq = example["seq"]
    camera_name = example["camera"]
    start_frame = example["start_frame"]
    num_frames = example["num_frames"]
    person_idx = example["person_idx"]

    seq_dir = get_sequence_dir(data_root, seq)
    video_path = get_video_path(data_root, seq, camera_name)

    # Camera calibration
    cameras = load_calibration(seq_dir)
    cam_calib = cameras[camera_name]
    K = cam_calib["K"]
    R = cam_calib["R"]
    t = cam_calib["t"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    resolution = cam_calib["resolution"]
    camera = Camera.from_panoptic_calibration(K, R, t, resolution)

    # Extract frames
    video_fps = 30.0
    frame_step = max(1, int(round(video_fps / target_fps)))
    frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
    frames_rgb = extract_video_frames(video_path, frame_indices)
    frame_indices = frame_indices[: len(frames_rgb)]

    print(f"  Extracted {len(frames_rgb)} frames")

    # Run detection with UNION bbox (baseline)
    print("  Running detection with UNION bbox...")
    t0 = time.time()
    kp_2d_u, vis_u, hm_u, mpii_u, affine_u = detect_2d_poses(
        frames_rgb, yolo_sh_models, sh_batch_size=32, per_frame_bbox=False,
    )
    print(f"    Done in {time.time() - t0:.1f}s")

    # Run detection with PER-FRAME bbox
    print("  Running detection with PER-FRAME bbox...")
    t0 = time.time()
    kp_2d_pf, vis_pf, hm_pf, mpii_pf, affine_pf = detect_2d_poses(
        frames_rgb, yolo_sh_models, sh_batch_size=32, per_frame_bbox=True,
    )
    print(f"    Done in {time.time() - t0:.1f}s")

    # Ground truth
    gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
    gt_cam: list[Any] = []
    for gt in gt_world:
        if gt is not None:
            gt_cam.append(camera.world_to_camera(gt) * 0.01)
        else:
            gt_cam.append(None)
    n_gt = sum(1 for g in gt_cam if g is not None)
    print(f"  Ground truth: {n_gt}/{len(frame_indices)} frames")

    return ExampleData(
        name=example["name"],
        frames_rgb=frames_rgb,
        frame_indices=frame_indices,
        heatmaps_union=hm_u,
        kp_2d_union=kp_2d_u,
        visibility_union=vis_u,
        affine_union=affine_u,
        mpii_kp_2d_union=mpii_u,
        heatmaps_perframe=hm_pf,
        kp_2d_perframe=kp_2d_pf,
        visibility_perframe=vis_pf,
        affine_perframe=affine_pf,
        mpii_kp_2d_perframe=mpii_pf,
        camera=camera,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        gt_cam=gt_cam,
    )


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def run_motionbert_pipeline(
    data: ExampleData,
    heatmaps: list,
    kp_2d: list,
    visibility: list,
    affine: Any,
    mpii_kp_2d: list,
    opt_config: OptimizationConfig,
    mb_model: Any,
    mb_device: Any,
) -> dict:
    """Run MotionBERT 3D lifting + optimization and evaluate."""
    from run_motionbert.detect import run_motionbert as run_mb_3d, motionbert_to_camera_space

    # 3D lifting
    positions_3d_norm = run_mb_3d(
        mpii_kp_2d, model=mb_model, device=mb_device, conf_threshold=0.0,
    )

    # Convert to camera space
    det_cam: list[np.ndarray] = []
    for i in range(len(data.frames_rgb)):
        pos_cam = motionbert_to_camera_space(
            positions_3d_norm[i], kp_2d[i], data.fx, data.fy, data.cx, data.cy,
        )
        det_cam.append(pos_cam)

    # Optimize
    opt_3d, _, _ = optimize(
        raw_3d=det_cam,
        camera=data.camera,
        config=opt_config,
        heatmaps=heatmaps,
        affine=affine,
        visibility=visibility,
        verbose=False,
    )

    # Evaluate
    gt_indices = [i for i, g in enumerate(data.gt_cam) if g is not None]
    if len(gt_indices) < 2:
        return {"det_vw_si_mpjpe_cm": float("nan"), "opt_vw_si_mpjpe_cm": float("nan")}

    gt_arr = np.array([data.gt_cam[i] for i in gt_indices])
    det_arr = np.array([det_cam[i] for i in gt_indices])
    opt_arr = np.array([opt_3d[i] for i in gt_indices])

    det_metrics = evaluate(det_arr, gt_arr, data.camera)
    opt_metrics = evaluate(opt_arr, gt_arr, data.camera)

    return {
        "det_vw_si_mpjpe_cm": det_metrics["vw_si_mpjpe"] * 100,
        "opt_vw_si_mpjpe_cm": opt_metrics["vw_si_mpjpe"] * 100,
    }


def run_mediapipe_pipeline(
    data: ExampleData,
    heatmaps: list,
    visibility: list,
    affine: Any,
    opt_config: OptimizationConfig,
) -> dict:
    """Run MediaPipe detection + optimization and evaluate."""
    from run_mediapipe.detect import (
        load_landmarker,
        detect_poses as mp_detect_poses,
        mediapipe_3d_to_camera,
    )

    # MediaPipe detection
    mp_landmarker = load_landmarker()
    mp_kp_2d, mp_kp_3d, mp_visibility = mp_detect_poses(
        data.frames_rgb, landmarker=mp_landmarker,
    )
    mp_landmarker.close()

    n_detected = sum(1 for v in mp_visibility if v.mean() > 0.3)
    print(f"    MediaPipe detected poses in {n_detected}/{len(data.frames_rgb)} frames")

    # Convert to camera space
    det_cam: list[np.ndarray] = []
    for i in range(len(data.frames_rgb)):
        pos_cam = mediapipe_3d_to_camera(
            mp_kp_3d[i], mp_kp_2d[i], data.fx, data.fy, data.cx, data.cy,
        )
        det_cam.append(pos_cam)

    # Optimize
    opt_3d, _, _ = optimize(
        raw_3d=det_cam,
        camera=data.camera,
        config=opt_config,
        heatmaps=heatmaps,
        affine=affine,
        visibility=visibility,
        verbose=False,
    )

    # Evaluate
    gt_indices = [i for i, g in enumerate(data.gt_cam) if g is not None]
    if len(gt_indices) < 2:
        return {"det_vw_si_mpjpe_cm": float("nan"), "opt_vw_si_mpjpe_cm": float("nan")}

    gt_arr = np.array([data.gt_cam[i] for i in gt_indices])
    det_arr = np.array([det_cam[i] for i in gt_indices])
    opt_arr = np.array([opt_3d[i] for i in gt_indices])

    det_metrics = evaluate(det_arr, gt_arr, data.camera)
    opt_metrics = evaluate(opt_arr, gt_arr, data.camera)

    return {
        "det_vw_si_mpjpe_cm": det_metrics["vw_si_mpjpe"] * 100,
        "opt_vw_si_mpjpe_cm": opt_metrics["vw_si_mpjpe"] * 100,
        "n_detected": n_detected,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from run_motionbert.detect import load_yolo_sh_models, load_motionbert_model

    print("=" * 72)
    print("Per-Frame BBox vs Union BBox Experiment")
    print("=" * 72)

    # Optimization configs (matching local-debug-pose2.json)
    mb_opt = OptimizationConfig(
        num_steps=100,
        learning_rate=0.001,
        bone_length_lr=0.005,
        heatmap_blur_sigma_start=16.0,
        heatmap_blur_sigma_end=4.0,
        anchor_weight=0.0,
    )
    mp_opt = OptimizationConfig(
        num_steps=100,
        heatmap_blur_sigma=4.0,
        learning_rate=0.0005,
        anchor_weight=50.0,
    )

    # Load shared models once
    print("\nLoading models...")
    yolo_sh_models = load_yolo_sh_models()
    mb_model = load_motionbert_model()
    mb_device = yolo_sh_models.device
    mb_model = mb_model.to(mb_device)
    print("Models loaded.\n")

    all_results: list[dict] = []

    for example in EXAMPLES:
        print("=" * 72)
        print(f"Example: {example['name']}")
        print(f"  Sequence: {example['seq']}, frames {example['start_frame']}..+{example['num_frames']}")
        print("=" * 72)

        data = load_example(example, yolo_sh_models)

        # --- MotionBERT: Union bbox ---
        print("\n  [MB] Union bbox...")
        mb_union = run_motionbert_pipeline(
            data, data.heatmaps_union, data.kp_2d_union, data.visibility_union,
            data.affine_union, data.mpii_kp_2d_union, mb_opt, mb_model, mb_device,
        )
        print(f"    Det: {mb_union['det_vw_si_mpjpe_cm']:.2f} cm  "
              f"Opt: {mb_union['opt_vw_si_mpjpe_cm']:.2f} cm")

        # --- MotionBERT: Per-frame bbox ---
        print("  [MB] Per-frame bbox...")
        mb_perframe = run_motionbert_pipeline(
            data, data.heatmaps_perframe, data.kp_2d_perframe, data.visibility_perframe,
            data.affine_perframe, data.mpii_kp_2d_perframe, mb_opt, mb_model, mb_device,
        )
        print(f"    Det: {mb_perframe['det_vw_si_mpjpe_cm']:.2f} cm  "
              f"Opt: {mb_perframe['opt_vw_si_mpjpe_cm']:.2f} cm")

        # --- MediaPipe: Union bbox ---
        print("  [MP] Union bbox...")
        mp_union = run_mediapipe_pipeline(
            data, data.heatmaps_union, data.visibility_union, data.affine_union, mp_opt,
        )
        print(f"    Det: {mp_union['det_vw_si_mpjpe_cm']:.2f} cm  "
              f"Opt: {mp_union['opt_vw_si_mpjpe_cm']:.2f} cm  "
              f"Detected: {mp_union.get('n_detected', 'N/A')}")

        # --- MediaPipe: Per-frame bbox ---
        print("  [MP] Per-frame bbox...")
        mp_perframe = run_mediapipe_pipeline(
            data, data.heatmaps_perframe, data.visibility_perframe,
            data.affine_perframe, mp_opt,
        )
        print(f"    Det: {mp_perframe['det_vw_si_mpjpe_cm']:.2f} cm  "
              f"Opt: {mp_perframe['opt_vw_si_mpjpe_cm']:.2f} cm  "
              f"Detected: {mp_perframe.get('n_detected', 'N/A')}")

        result = {
            "example": example["name"],
            "seq": example["seq"],
            "mb_union": mb_union,
            "mb_perframe": mb_perframe,
            "mp_union": mp_union,
            "mp_perframe": mp_perframe,
        }
        all_results.append(result)

    # --- Summary ---
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n{'Example':<30} {'Pipeline':<6} {'Mode':<12} {'Det':>8} {'Opt':>8} {'Delta':>8}")
    print("-" * 72)

    for r in all_results:
        name = r["example"]
        for pipeline, union_key, pf_key in [
            ("MB", "mb_union", "mb_perframe"),
            ("MP", "mp_union", "mp_perframe"),
        ]:
            u = r[union_key]
            pf = r[pf_key]
            print(f"{name:<30} {pipeline:<6} {'union':<12} "
                  f"{u['det_vw_si_mpjpe_cm']:>7.2f}  {u['opt_vw_si_mpjpe_cm']:>7.2f}")
            delta_det = pf['det_vw_si_mpjpe_cm'] - u['det_vw_si_mpjpe_cm']
            delta_opt = pf['opt_vw_si_mpjpe_cm'] - u['opt_vw_si_mpjpe_cm']
            print(f"{'':<30} {'':<6} {'per-frame':<12} "
                  f"{pf['det_vw_si_mpjpe_cm']:>7.2f}  {pf['opt_vw_si_mpjpe_cm']:>7.2f}  "
                  f"({delta_opt:+.2f})")
            print()

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "per_frame_bbox_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
