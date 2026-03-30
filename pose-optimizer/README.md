# pose-optimizer

Unified pipeline for 3D human pose estimation and optimization. Supports two base estimators (MediaPipe, MotionBert) with a shared differentiable optimizer and evaluation suite.

## Quick Start

```bash
cd pose-optimizer
uv sync

# Single example
uv run python run_motionbert.py run_motionbert/run_single_config.json
uv run python run_mediapipe.py run_mediapipe/run_single_config.json

# Full 25-example evaluation
uv run python run_motionbert.py run_motionbert/run_full_config.json
uv run python run_mediapipe.py run_mediapipe/run_full_config.json

# Full 25-example evaluation
# Results are written to timestamped output directories
```

## Data

Uses CMU Panoptic Studio sequences. Data should be at `data/panoptic-toolbox/` (symlinked to shared data directory).

```bash
# Download helpers
uv run python -c "from cmu_data import download_sequence; download_sequence('171204_pose1_sample')"
```

## Pipeline

```
Video Frames ─┬─ MediaPipe PoseLandmarker ──── 2D/3D landmarks ── synthetic Gaussian heatmaps ─┐
              └─ YOLO → Stacked Hourglass → MotionBert ── 2D heatmaps + 3D predictions ────────┤
                                                                                                 ▼
                                                                              Shared Optimizer (FK + Adam)
                                                                                                 │
                                                                                    Improved 3D predictions
                                                                                                 │
                                                                              Evaluation (9 metrics) + Output
```

Both paths produce raw 3D predictions and 2D heatmaps. The shared optimizer refines the 3D predictions using differentiable forward kinematics and heatmap-based scoring.

## File Structure

```
run_motionbert.py          MotionBert entrypoint
run_mediapipe.py           MediaPipe entrypoint

run_motionbert/
    __init__.py            Pipeline: YOLO → SH → MotionBert → Optimizer → Evaluate → Output
    detect.py              YOLO person detection, Stacked Hourglass 2D, MotionBert 3D lifting
    setup_models.py        Model checkpoint downloads
    run_single_config.json
    run_full_config.json

run_mediapipe/
    __init__.py            Pipeline: MediaPipe → solvePnP → Optimizer → Evaluate → Output
    detect.py              MediaPipe PoseLandmarker detection + camera-space conversion
    run_single_config.json
    run_full_config.json

optimize/
    __init__.py            Shared optimizer: heatmaps + raw 3D + Camera → improved 3D

skeleton.py                16-joint skeleton definition and joint mappings
camera.py                  Camera model (intrinsics, extrinsics, projection, visibility)
fk.py                      Differentiable forward kinematics (single + batch)
evaluate.py                9 evaluation metrics (MPJPE, P-MPJPE, SI, VW, velocity variants)
scoring.py                 Heatmap scoring via grid_sample + motion/rotation penalties
cmu_data.py                CMU Panoptic data loading, GT parsing, download helpers
config.py                  Pydantic config models with JSON serialization
graphs.py                  Trajectory, per-joint, per-frame, loss, bone length graphs
overlay_video.py           Overlay video with heatmaps + skeleton overlays
visualize.py               3D animated skeleton viewer from trajectories.json

experiment/                Test scripts
data/                      Symlink to CMU Panoptic data (gitignored)
output/                    Run outputs (gitignored)
```

## Configuration

Configs are JSON. Key fields:

```jsonc
{
    "examples": [
        {"sequence": "171204_pose1_sample", "camera": "00_00", "start_frame": 0, "num_frames": 100}
    ],
    "data_root": "data/panoptic-toolbox",
    "output_dir": null,             // null = auto timestamped directory
    "target_fps": 10.0,
    "optimization": {
        "num_steps": 50,
        "learning_rate": 0.002,
        "heatmap_blur_sigma": 4.0,  // Gaussian blur on input heatmaps
        "heatmap_sigma": 50.0,      // Sigma for synthetic Gaussian heatmaps (MediaPipe)
        "position_penalty_weight": 500.0,
        "rotation_penalty_scalar": 10.0
    }
}
```

## Output

Each run creates a timestamped directory:

```
output/motionbert_2026_03_29_21_24/
    171204_pose1_sample_0/
        results.json         All 9 metrics (raw + optimized) + config
        trajectories.json    GT, raw, optimized trajectories for visualization
        overlay_video.mp4    Video with heatmap + skeleton overlays
        summary.png          Multi-panel summary image
        graphs/              Individual graphs (per-joint error, per-frame MPJPE, etc.)
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MPJPE | Mean Per-Joint Position Error |
| P-MPJPE | Procrustes-aligned MPJPE |
| SI-MPJPE | Scale-Invariant MPJPE (single global scalar) |
| VW-MPJPE | Visibility-Weighted MPJPE (frame-boundary based) |
| **VW-SI-MPJPE** | **Primary metric** — visibility-weighted, scale-invariant |
| MPJVE | Mean Per-Joint Velocity Error |
| SI-MPJVE | Scale-Invariant MPJVE |
| VW-MPJVE | Visibility-Weighted MPJVE |
| VW-SI-MPJVE | Visibility-weighted, scale-invariant velocity error |

## Skeleton

16-joint skeleton. No extrapolated joints.

```
0=Pelvis  1=RHip  2=RKnee  3=RAnkle  4=LHip  5=LKnee  6=LAnkle
7=Spine  8=Neck  9=Head  10=LShoulder  11=LElbow  12=LWrist
13=RShoulder  14=RElbow  15=RWrist
```

Evaluation uses 14 joints (excludes midpoints: Spine and Head).

## Tests

```bash
uv run python -m pytest experiment/ -v
```
