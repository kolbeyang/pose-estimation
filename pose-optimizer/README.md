# pose-optimizer

Unified pipeline for 3D human pose estimation and optimization. Supports two base estimators (MediaPipe, MotionBERT) with shared Stacked Hourglass heatmaps, a differentiable optimizer, and evaluation suite.

## Quick Start

```bash
cd pose-optimizer
uv sync

# Single example (one pipeline)
uv run python main.py configs/motionbert-single.json
uv run python main.py configs/mediapipe-single.json

# Both pipelines together
uv run python main.py configs/both-local-single.json

# Full 25-example evaluation
uv run python main.py configs/motionbert-local-25-examples.json
```

## Data

Uses CMU Panoptic Studio sequences. Data should be at `data/panoptic-toolbox/` (symlinked to shared data directory).

```bash
uv run python -c "from cmu_data import download_sequence; download_sequence('171204_pose1_sample')"
```

## Pipeline

```
Video Frames ── YOLO + Stacked Hourglass ── 2D heatmaps (shared) ──┐
              │                                                      │
              ├─ MotionBERT 3D lifting ── camera space ─────────────┤
              └─ MediaPipe 3D + solvePnP ── camera space ──────────┤
                                                                     ▼
                                                   Shared Optimizer (FK + Adam)
                                                                     │
                                                        Improved 3D predictions
                                                                     │
                                                   Evaluation (10 metrics) + Output
```

Both paths share the same YOLO + Stacked Hourglass heatmaps. The optimizer refines 3D predictions using differentiable forward kinematics and heatmap-based scoring.

## File Structure

```
main.py                    Unified entrypoint
configs/                   Reusable config files

run_motionbert/
    detect.py              YOLO detection, Stacked Hourglass 2D, MotionBERT 3D lifting
    setup_models.py        Model checkpoint downloads

run_mediapipe/
    detect.py              MediaPipe PoseLandmarker detection + solvePnP camera conversion

optimize/
    __init__.py            Shared optimizer: heatmaps + raw 3D + Camera -> improved 3D

skeleton.py                16-joint skeleton definition and joint mappings (14 eval joints)
camera.py                  Camera model (intrinsics, extrinsics, projection, visibility)
fk.py                      Differentiable forward kinematics (single + batch)
evaluate.py                10 evaluation metrics (MPJPE, P-MPJPE, SI, VW, velocity, 2D reprojection)
scoring.py                 Heatmap scoring via grid_sample + motion/rotation penalties
cmu_data.py                CMU Panoptic data loading, GT parsing, download helpers
config.py                  Pydantic config models with JSON serialization
graphs.py                  Per-example, aggregate, and cross-pipeline comparison graphs
overlay_video.py           Overlay video with heatmaps + skeleton overlays
visualize.py               3D animated skeleton viewer from trajectories.json

experiment/                Analysis scripts (elbow smoothness, bone length variation, etc.)
data/                      Symlink to CMU Panoptic data (gitignored)
output/                    Run outputs (gitignored)
```

## Configuration

Configs are JSON with optional `mediapipe`/`motionbert`/`graphs` sections:

```jsonc
{
    "examples": [
        {"sequence": "171204_pose1_sample", "camera": "00_00", "start_frame": 0, "num_frames": 100}
    ],
    "data_root": "data/panoptic-toolbox",
    "motionbert": { "is_generate_heatmap_videos": true },
    "mediapipe": { "is_generate_heatmap_videos": true },
    "graphs": { "per_example_mediapipe_vs_motionbert": true },
    "optimization": {
        "num_steps": 50,
        "learning_rate": 0.002,
        "heatmap_blur_sigma": 4.0,
        "position_penalty_weight": 500.0,
        "rotation_penalty_scalar": 10.0
    }
}
```

## Output

Each run creates a timestamped directory:

```
output/run_2026_04_04_12_00/
    171204_pose1_sample_0/
        motionbert/
            results.json         All 10 metrics (raw + optimized) + config
            trajectories.json    GT, raw, optimized trajectories
            heatmap_overlay_video.mp4
            graphs/
        mediapipe/
            results.json
            trajectories.json
            heatmap_overlay_video.mp4
            graphs/
    results.json                 Run-level aggregate results
    cross_pipeline/              Cross-pipeline comparison graphs (if both pipelines ran)
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MPJPE | Mean Per-Joint Position Error |
| P-MPJPE | Procrustes-aligned MPJPE |
| SI-MPJPE | Scale-Invariant MPJPE |
| VW-MPJPE | Visibility-Weighted MPJPE |
| **VW-SI-MPJPE** | **Primary metric** -- visibility-weighted, scale-invariant |
| MPJVE | Mean Per-Joint Velocity Error |
| SI-MPJVE | Scale-Invariant MPJVE |
| VW-MPJVE | Visibility-Weighted MPJVE |
| VW-SI-MPJVE | Visibility-weighted, scale-invariant velocity error |
| 2D-MPJPE | 2D reprojected MPJPE (pixels) |

All metrics are computed in camera coordinates (not root-relative).

## Skeleton

16-joint skeleton. 14 evaluation joints (excludes Spine, used FK-internally; and HeadTop, optimization-only without reliable GT).

```
0=Pelvis  1=RHip  2=RKnee  3=RAnkle  4=LHip  5=LKnee  6=LAnkle
7=Spine  8=Neck  9=HeadTop  10=LShoulder  11=LElbow  12=LWrist
13=RShoulder  14=RElbow  15=RWrist
```
