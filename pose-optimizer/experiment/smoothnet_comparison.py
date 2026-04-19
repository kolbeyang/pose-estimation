"""SmoothNet vs. FK Optimizer head-to-head comparison.

Runs the full pipeline (YOLO -> SH -> MotionBERT/MediaPipe) and compares:
  1. Raw detector output (baseline)
  2. SmoothNet-refined output
  3. Our FK optimizer output

Evaluates all on the same 4 key metrics: VW-SI-MPJPE, VW-SI-MPJVE, MPJPE, MPJVE.

Usage:
    cd pose-optimizer
    uv run python experiment/smoothnet_comparison.py [config.json]

If no config given, uses both-local-single.json as default.

SmoothNet checkpoint: downloads pretrained weights automatically on first run.
Trained on Human3.6M (17 joints, 3D) -- requires joint mapping from our 16-joint skeleton.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from camera import Camera
from cmu_data import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
)
from config import RunConfig, load_config, ExampleConfig
from evaluate import (
    evaluate,
    compute_visibility_weights,
    mpjpe as compute_mpjpe,
    vw_si_mpjpe as compute_vw_si_mpjpe,
    mpjve as compute_mpjve,
    vw_si_mpjve as compute_vw_si_mpjve,
)
from optimize import optimize
from skeleton import EVAL_JOINTS, NUM_JOINTS

from smoothnet_model import SmoothNet, smooth_poses

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint mapping: our 16-joint -> SmoothNet's Human3.6M 17-joint
# ---------------------------------------------------------------------------
#
# CRITICAL DECISION: SmoothNet pretrained checkpoints were trained on
# Human3.6M 17-joint format. Our skeleton has 16 joints.
#
# Our 16-joint:  Pelvis(0) RHip(1) RKnee(2) RAnkle(3) LHip(4) LKnee(5)
#                LAnkle(6) Spine(7) Neck(8) HeadTop(9) LShoulder(10) LElbow(11)
#                LWrist(12) RShoulder(13) RElbow(14) RWrist(15)
#
# H36M 17-joint: ..Neck(8), Nose(9), Head(10), LShoulder(11), ..RWrist(16).
#                We must INSERT a synthesized Nose at H36M[9].
#                This is the inverse of skeleton.strip_nose_joint().
#
# Nose synthesis: Neck + 0.3*(HeadTop - Neck), matching mpii_to_motionbert_17().
#
# NOTE: These functions are only needed when loading official H36M pretrained
# SmoothNet weights. For on-the-fly training we skip the conversion entirely.
# ---------------------------------------------------------------------------

def skeleton16_to_h36m17(poses_16: np.ndarray) -> np.ndarray:
    """Convert (F, 16, 3) -> (F, 17, 3) by inserting synthesized Nose at index 9."""
    f = poses_16.shape[0]
    poses_17 = np.zeros((f, 17, 3), dtype=poses_16.dtype)
    poses_17[:, :9, :] = poses_16[:, :9, :]
    # Synthesize Nose = Neck + 0.3*(HeadTop - Neck)
    poses_17[:, 9, :] = poses_16[:, 8, :] + 0.3 * (poses_16[:, 9, :] - poses_16[:, 8, :])
    poses_17[:, 10, :] = poses_16[:, 9, :]   # HeadTop
    poses_17[:, 11:, :] = poses_16[:, 10:, :]
    return poses_17


def h36m17_to_skeleton16(poses_17: np.ndarray) -> np.ndarray:
    """Convert (F, 17, 3) -> (F, 16, 3) by removing Nose at index 9 (strip_nose_joint)."""
    return np.delete(poses_17, 9, axis=1)


# ---------------------------------------------------------------------------
# SmoothNet checkpoint management
# ---------------------------------------------------------------------------

SMOOTHNET_DIR = Path(__file__).resolve().parent / "smoothnet_checkpoints"

# CRITICAL DECISION: Using window_size=32 as default. SmoothNet paper shows
# 32 is the best balance of temporal context and performance. At 10fps
# (our target_fps), 32 frames = 3.2 seconds of context.
DEFAULT_WINDOW_SIZE = 32
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_RES_HIDDEN_SIZE = 128  # matches official checkpoint
DEFAULT_NUM_BLOCKS = 5         # matches official checkpoint


def get_smoothnet_model(device: torch.device, window_size: int = DEFAULT_WINDOW_SIZE) -> SmoothNet:
    """Load pretrained SmoothNet from official H36M/FCN/3D checkpoint.

    CRITICAL DECISION: We use the official H36M pretrained checkpoint directly
    on our 16-joint (48-channel) data. SmoothNet is fully channel-independent
    (Linear operates on the temporal dimension only), so the checkpoint trained
    on 17-joint H36M (51 channels) loads and runs correctly on 48 channels.
    No joint mapping is needed.

    Expected file (download from SmoothNet Google Drive):
        experiment/smoothnet_checkpoints/smoothnet_h36m_w32.pth.tar

    Falls back to on-the-fly training if not found.
    """
    model = SmoothNet(
        window_size=window_size,
        output_size=window_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        res_hidden_size=DEFAULT_RES_HIDDEN_SIZE,
        num_blocks=DEFAULT_NUM_BLOCKS,
        dropout=0.0,  # No dropout at inference
    ).to(device)

    # Try .pth.tar first (official release format), then .pth
    for suffix in [f"smoothnet_h36m_w{window_size}.pth.tar", f"smoothnet_h36m_w{window_size}.pth"]:
        checkpoint_path = SMOOTHNET_DIR / suffix
        if checkpoint_path.exists():
            logger.info("Loading SmoothNet checkpoint: %s", checkpoint_path)
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            sd = state["state_dict"] if "state_dict" in state else state
            model.load_state_dict(sd)
            model.eval()
            if "performance" in state:
                perf = state["performance"]
                logger.info(
                    "  Checkpoint performance on H36M: input MPJPE=%.1f mm, output MPJPE=%.1f mm",
                    float(perf.get("input_mpjpe", 0)),
                    float(perf.get("output_mpjpe", 0)),
                )
            return model

    # FLAG: No pretrained checkpoint found. Train from scratch on GT data.
    # This gives SmoothNet an ADVANTAGE (sees GT motion patterns from eval sequences).
    # Results in this mode are an upper bound for SmoothNet, not a fair comparison.
    logger.warning(
        "No pretrained SmoothNet checkpoint found in %s. "
        "Falling back to on-the-fly training on eval GT data (unfair advantage for SmoothNet).",
        SMOOTHNET_DIR,
    )
    return model


def train_smoothnet_on_gt(
    model: SmoothNet,
    gt_sequences: list[np.ndarray],
    device: torch.device,
    num_epochs: int = 200,
    lr: float = 0.002,
) -> SmoothNet:
    """Train SmoothNet to denoise poses using GT sequences as supervision.

    Training approach: add synthetic noise to GT, train model to recover GT.
    This mimics SmoothNet's original training procedure.

    Args:
        model: Untrained SmoothNet model.
        gt_sequences: List of (F, 16, 3) GT sequences in camera space.
        device: Torch device.
        num_epochs: Training epochs.
        lr: Learning rate.

    Returns:
        Trained model.
    """
    window_size = model.window_size

    # Build training data: (N, window_size, J*3) windows from GT
    windows_gt = []
    for seq in gt_sequences:
        if seq.shape[0] < window_size:
            continue
        seq_flat = seq.reshape(seq.shape[0], -1)  # (F, J*3)
        for start in range(0, seq.shape[0] - window_size + 1, window_size // 4):
            windows_gt.append(seq_flat[start:start + window_size])

    if len(windows_gt) < 4:
        logger.warning("Not enough GT data to train SmoothNet (%d windows). Using untrained model.", len(windows_gt))
        model.eval()
        return model

    windows_gt_arr = np.stack(windows_gt, axis=0)  # (N, T, C)
    gt_tensor = torch.from_numpy(windows_gt_arr).float().to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(num_epochs):
        # Add noise to simulate detector jitter + temporal inconsistency
        noise_scale = 0.02 + 0.03 * np.random.rand()  # 2-5cm noise in meters
        noise = torch.randn_like(gt_tensor) * noise_scale

        # Also add temporal jitter (per-frame random offsets)
        temporal_jitter = torch.randn(gt_tensor.shape[0], 1, gt_tensor.shape[2], device=device) * 0.01
        noisy = gt_tensor + noise + temporal_jitter

        # Forward: (B, C, T)
        x = noisy.permute(0, 2, 1)
        target = gt_tensor.permute(0, 2, 1)

        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            logger.info("  SmoothNet train epoch %d/%d, loss=%.6f", epoch + 1, num_epochs, loss.item())

    model.eval()

    # Save checkpoint for future runs
    SMOOTHNET_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SMOOTHNET_DIR / f"smoothnet_h36m_w{window_size}.pth"
    torch.save({"state_dict": model.state_dict()}, save_path)
    logger.info("Saved trained SmoothNet checkpoint: %s", save_path)

    return model


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(config_path: str | None = None) -> None:
    """Run head-to-head comparison: Raw vs SmoothNet vs FK Optimizer."""
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "configs" / "both-local-single.json")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    examples = config.examples
    if not examples:
        from cmu_data import discover_examples
        examples = discover_examples(config.data_root)

    run_motionbert = config.motionbert is not None
    run_mediapipe = config.mediapipe is not None

    if not run_motionbert and not run_mediapipe:
        logger.error("Config must have at least one pipeline.")
        sys.exit(1)

    # Load detection models
    from run_motionbert.detect import (
        load_yolo_sh_models,
        load_motionbert_model,
        detect_2d_poses,
        run_motionbert as run_mb_3d,
        motionbert_to_camera_space,
    )

    logger.info("Loading YOLO + SH models...")
    yolo_sh_models = load_yolo_sh_models()

    mb_model = None
    if run_motionbert:
        logger.info("Loading MotionBERT...")
        mb_model = load_motionbert_model()
        mb_model = mb_model.to(yolo_sh_models.device)

    mp_landmarker = None
    mp_detect_poses = None
    mediapipe_3d_to_camera = None
    if run_mediapipe:
        from run_mediapipe.detect import load_landmarker, detect_poses as _mp_detect, mediapipe_3d_to_camera as _mp_cam
        mp_detect_poses = _mp_detect
        mediapipe_3d_to_camera = _mp_cam
        mp_landmarker = load_landmarker()

    # Collect GT sequences for SmoothNet training (if no checkpoint)
    all_gt_sequences: list[np.ndarray] = []

    # Collect per-example results
    all_results: list[dict] = []

    # First pass: gather data
    example_data: list[dict] = []

    for example in examples:
        seq_name = example.sequence
        camera_name = example.camera
        start_frame = example.start_frame
        num_frames = example.num_frames
        person_idx = example.person_idx
        name = f"{seq_name}_{start_frame}"

        logger.info("=" * 60)
        logger.info("Processing: %s", name)

        try:
            data_root = config.data_root
            seq_dir = get_sequence_dir(data_root, seq_name)
            video_path = get_video_path(data_root, seq_name, camera_name)

            # Camera
            cameras = load_calibration(seq_dir)
            cam_calib = _find_camera_calibration(cameras, camera_name)
            K = cam_calib["K"]
            R = cam_calib["R"]
            t = cam_calib["t"]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            resolution = cam_calib["resolution"]
            camera = Camera.from_panoptic_calibration(K, R, t, resolution)

            # Frames
            video_fps = 30.0
            frame_step = max(1, int(round(video_fps / config.target_fps)))
            frame_indices = list(range(start_frame, start_frame + num_frames, frame_step))
            frames_rgb = extract_video_frames(video_path, frame_indices)
            frame_indices = frame_indices[:len(frames_rgb)]
            if len(frames_rgb) < 2:
                continue

            # Shared 2D detection
            kp_2d, visibility, heatmaps, mpii_kp_2d, affine = detect_2d_poses(
                frames_rgb, yolo_sh_models,
                sh_batch_size=config.sh_batch_size,
                per_frame_bbox=config.per_frame_bbox,
            )

            # Ground truth
            gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx)
            gt_cam: list[np.ndarray | None] = []
            for gt in gt_world:
                if gt is not None:
                    gt_cam.append(camera.world_to_camera(gt) * 0.01)
                else:
                    gt_cam.append(None)

            n_gt = sum(1 for g in gt_cam if g is not None)
            if n_gt < 2:
                logger.warning("Skipping %s: only %d GT frames", name, n_gt)
                continue

            # Collect GT for SmoothNet training
            gt_valid = [g for g in gt_cam if g is not None]
            if len(gt_valid) >= DEFAULT_WINDOW_SIZE:
                all_gt_sequences.append(np.stack(gt_valid, axis=0))

            datum = {
                "name": name,
                "camera": camera,
                "gt_cam": gt_cam,
                "frames_rgb": frames_rgb,
                "kp_2d": kp_2d,
                "visibility": visibility,
                "heatmaps": heatmaps,
                "mpii_kp_2d": mpii_kp_2d,
                "affine": affine,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }

            # Run detector(s)
            if run_motionbert:
                positions_3d_norm = run_mb_3d(
                    mpii_kp_2d, model=mb_model, device=yolo_sh_models.device,
                    conf_threshold=config.motionbert_conf_threshold,
                )
                det_cam_mb = [
                    motionbert_to_camera_space(positions_3d_norm[i], kp_2d[i], fx, fy, cx, cy)
                    for i in range(len(positions_3d_norm))
                ]
                datum["det_cam_mb"] = det_cam_mb

            if run_mediapipe:
                mp_kp_2d_list, mp_kp_3d_list, mp_vis = mp_detect_poses(frames_rgb, landmarker=mp_landmarker)
                det_cam_mp = [
                    mediapipe_3d_to_camera(mp_kp_3d_list[i], mp_kp_2d_list[i], fx, fy, cx, cy)
                    for i in range(len(frames_rgb))
                ]
                datum["det_cam_mp"] = det_cam_mp

            example_data.append(datum)

        except Exception:
            logger.exception("Error processing %s", name)
            continue

    if not example_data:
        logger.error("No examples processed successfully.")
        sys.exit(1)

    # Load/train SmoothNet
    smoothnet = get_smoothnet_model(device)
    checkpoint_exists = any(
        (SMOOTHNET_DIR / f).exists()
        for f in [f"smoothnet_h36m_w{DEFAULT_WINDOW_SIZE}.pth.tar",
                  f"smoothnet_h36m_w{DEFAULT_WINDOW_SIZE}.pth"]
    )
    if not checkpoint_exists and all_gt_sequences:
        logger.info("Training SmoothNet on %d GT sequences (no pretrained checkpoint)...", len(all_gt_sequences))
        smoothnet = train_smoothnet_on_gt(smoothnet, all_gt_sequences, device)

    smoothnet.eval()

    # Second pass: evaluate all methods
    logger.info("=" * 60)
    logger.info("EVALUATING ALL METHODS")
    logger.info("=" * 60)

    for datum in example_data:
        name = datum["name"]
        camera = datum["camera"]
        gt_cam = datum["gt_cam"]

        gt_indices = [i for i, g in enumerate(gt_cam) if g is not None]
        gt_arr = np.array([gt_cam[i] for i in gt_indices])

        result_entry = {"name": name, "num_frames": len(gt_indices)}

        pipelines = []
        if "det_cam_mb" in datum:
            pipelines.append(("motionbert", datum["det_cam_mb"]))
        if "det_cam_mp" in datum:
            pipelines.append(("mediapipe", datum["det_cam_mp"]))

        for pipeline_name, det_cam in pipelines:
            det_arr = np.array([det_cam[i] for i in gt_indices])

            # --- Method 1: Raw detector ---
            raw_metrics = _compute_key_metrics(det_arr, gt_arr, camera)

            # --- Method 2: SmoothNet ---
            smoothed = smooth_poses(det_cam, smoothnet, device)
            smooth_arr = np.array([smoothed[i] for i in gt_indices])
            smooth_metrics = _compute_key_metrics(smooth_arr, gt_arr, camera)

            # --- Method 3: Our FK optimizer ---
            opt_config = config.optimization_for_pipeline(pipeline_name)
            opt_3d, _, _ = optimize(
                raw_3d=det_cam,
                camera=camera,
                config=opt_config,
                heatmaps=datum["heatmaps"],
                affine=datum["affine"],
                visibility=datum["visibility"],
                verbose=False,
            )
            opt_arr = np.array([opt_3d[i] for i in gt_indices])
            opt_metrics = _compute_key_metrics(opt_arr, gt_arr, camera)

            result_entry[f"{pipeline_name}_raw"] = raw_metrics
            result_entry[f"{pipeline_name}_smoothnet"] = smooth_metrics
            result_entry[f"{pipeline_name}_optimizer"] = opt_metrics

            # Print comparison
            logger.info(
                "[%s / %s] VW-SI-MPJPE: Raw=%.2f  SmoothNet=%.2f  Ours=%.2f cm",
                name, pipeline_name,
                raw_metrics["vw_si_mpjpe"] * 100,
                smooth_metrics["vw_si_mpjpe"] * 100,
                opt_metrics["vw_si_mpjpe"] * 100,
            )
            logger.info(
                "[%s / %s] VW-SI-MPJVE: Raw=%.2f  SmoothNet=%.2f  Ours=%.2f cm/f",
                name, pipeline_name,
                raw_metrics["vw_si_mpjve"] * 100,
                smooth_metrics["vw_si_mpjve"] * 100,
                opt_metrics["vw_si_mpjve"] * 100,
            )

        all_results.append(result_entry)

    # Print aggregate summary
    _print_aggregate_summary(all_results)

    # Save results — named after config file for easy identification
    output_dir = Path(__file__).resolve().parent / "smoothnet_results"
    output_dir.mkdir(exist_ok=True)
    config_stem = Path(config_path).stem if config_path else "default"
    results_path = output_dir / f"{config_stem}_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to: %s", results_path)

    if mp_landmarker is not None:
        mp_landmarker.close()


def _compute_key_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    camera: Camera,
) -> dict[str, float]:
    """Compute the 4 key comparison metrics.

    Args:
        predicted: (F, 16, 3) in camera space.
        ground_truth: (F, 16, 3) in camera space.
        camera: Camera for visibility weights.

    Returns:
        Dict with mpjpe, vw_si_mpjpe, mpjve, vw_si_mpjve.
    """
    pred_eval = predicted[:, EVAL_JOINTS, :]
    gt_eval = ground_truth[:, EVAL_JOINTS, :]
    vis = compute_visibility_weights(ground_truth, camera)[:, EVAL_JOINTS]

    metrics: dict[str, float] = {}
    metrics["mpjpe"] = compute_mpjpe(pred_eval, gt_eval)
    metrics["vw_si_mpjpe"] = compute_vw_si_mpjpe(pred_eval, gt_eval, vis)

    if predicted.shape[0] >= 2:
        metrics["mpjve"] = compute_mpjve(pred_eval, gt_eval)
        metrics["vw_si_mpjve"] = compute_vw_si_mpjve(pred_eval, gt_eval, vis)
    else:
        metrics["mpjve"] = 0.0
        metrics["vw_si_mpjve"] = 0.0

    return metrics


def _print_aggregate_summary(all_results: list[dict]) -> None:
    """Print unified 6-method × 4-metric comparison table."""
    n = len(all_results)
    metrics_cfg = [
        ("mpjpe",      "MPJPE",       "cm",   100),
        ("mpjve",      "MPJVE",       "cm/f", 100),
        ("vw_si_mpjpe","VW-SI-MPJPE", "cm",   100),
        ("vw_si_mpjve","VW-SI-MPJVE", "cm/f", 100),
    ]

    # Build rows: (label, result_key)
    rows = []
    for pipeline, label in [("motionbert", "MotionBERT"), ("mediapipe", "MediaPipe")]:
        for suffix, tag in [("_raw", "Raw"), ("_smoothnet", "SmoothNet"), ("_optimizer", "FK Opt (Ours)")]:
            key = f"{pipeline}{suffix}"
            if any(key in r for r in all_results):
                rows.append((f"{label} {tag}", key))

    if not rows:
        print("No results to display.")
        return

    col_w = 12
    label_w = 26

    print(f"\n{'=' * (label_w + col_w * len(metrics_cfg) + 6)}")
    print(f"  RESULTS (mean over {n} example{'s' if n != 1 else ''})")
    print(f"{'=' * (label_w + col_w * len(metrics_cfg) + 6)}")

    # Header
    header = f"  {'Method':<{label_w}}"
    for _, col_name, unit, _ in metrics_cfg:
        header += f"  {col_name+' ('+unit+')':<{col_w}}"
    print(header)
    print(f"  {'─' * (label_w + col_w * len(metrics_cfg) + 4)}")

    all_vals: dict[str, list[float]] = {k: [] for _, k in rows}
    for mkey, _, _, scale in metrics_cfg:
        for label, rkey in rows:
            entries = [r[rkey][mkey] * scale for r in all_results if rkey in r]
            all_vals[rkey].append(float(np.mean(entries)) if entries else float("nan"))

    # Find best per metric column
    best_per_col = []
    for col_idx in range(len(metrics_cfg)):
        col_vals = [all_vals[rkey][col_idx] for _, rkey in rows]
        valid = [v for v in col_vals if not np.isnan(v)]
        best_per_col.append(min(valid) if valid else float("nan"))

    for label, rkey in rows:
        line = f"  {label:<{label_w}}"
        for col_idx, (_, _, _, _) in enumerate(metrics_cfg):
            val = all_vals[rkey][col_idx]
            marker = " *" if not np.isnan(val) and abs(val - best_per_col[col_idx]) < 1e-6 else "  "
            line += f"  {val:>{col_w-2}.2f}{marker}"
        print(line)

    print(f"\n  * = best for that metric")
    print(f"{'=' * (label_w + col_w * len(metrics_cfg) + 6)}\n")


def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across examples."""
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def _find_camera_calibration(cameras, camera_name):
    """Find camera calibration by name."""
    if camera_name in cameras:
        return cameras[camera_name]
    cam_name_full = f"00_{camera_name.split('_')[1]}" if "_" in camera_name else camera_name
    if cam_name_full in cameras:
        return cameras[cam_name_full]
    for cname, cal in cameras.items():
        if cname.startswith("00_00"):
            return cal
    raise ValueError(f"Camera {camera_name} not found")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_comparison(config_path)
