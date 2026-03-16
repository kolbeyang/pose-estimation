# Spec: MotionBERT Pose Estimation Pipeline

## Overview

Build a clean, standalone MotionBERT-based 3D pose estimation pipeline in `motionbert-pose/`. The pipeline uses Stacked Hourglass for 2D detection, MotionBERT for 3D lifting, and then feeds those predictions into our FK optimization framework to improve accuracy.

This is a two-phase effort. Phase 1 gets MotionBERT itself producing correct 3D predictions. Phase 2 layers our optimization on top.

## Important Constraints

- **All code lives in `motionbert-pose/`.** This is a new, clean directory. Do not import from `mediapipe-pose/` or any other sibling directory. You may reference `motionbert-pose-old/` for understanding, but do not copy code from it — rewrite from scratch.
- **Use `uv` for all Python execution.** `uv run python <script>`, `uv pip install`, etc. Never use pip directly.
- **Type hints on every function, every parameter, every return value.** No exceptions.
- **Pydantic models for all data structures** that pass between pipeline stages: config, detection results, optimization results, evaluation results, camera parameters. Use Pydantic's runtime validation to catch bad data early. Agents should see clear validation errors, not silent corruption.
- **Progressive testing.** Until we are confident the system works, limit optimization to 10 steps at a time. Scale up only after confirming correctness. Do not run 300-step optimizations as a first attempt.
- **Same 10 CMU Panoptic test videos** used in mediapipe-pose (see `mediapipe-pose/config.py` for the EXAMPLES list with sequence names, cameras, frame ranges, and person indices).

## Reference Material

Agents should consult these references during implementation, testing, and debugging. Read them when you need to understand how something is supposed to work.

### Papers
- **MotionBERT paper:** `/Users/kolbeyang/Documents/School/spring_2026/capstone/papers/motion-bert.pdf` — The original paper describing the MotionBERT architecture, training procedure, and evaluation. Consult this for understanding the model's input/output format, coordinate conventions, and expected performance benchmarks.
- **Stacked Hourglass paper:** `/Users/kolbeyang/Documents/School/spring_2026/capstone/papers/stacked-hourglass.pdf` — The original paper for the 2D pose estimator we use. Consult this for understanding heatmap output format, MPII joint definitions, and the multi-stack architecture.

### Existing Code
- **`mediapipe-pose/`** — The MediaPipe version of this pipeline. This is the gold standard reference for how our evaluation, camera projection, CMU Panoptic data loading, FK optimization, scoring, and graph generation work. Same 10 test videos, same metrics, same output format. Study this to understand what the final product should look like. **Do not import from it** — reimplement what you need.
- **`motionbert-pose-old/`** — Prior attempt at this pipeline. Contains working Stacked Hourglass + MotionBERT detection code, FK optimization, and a detailed `OPTIMIZATION_ATTEMPTS.md` log with 7 iterations of tuning. Achieved ~19% average MPJPE improvement over MotionBERT baseline. Study this to understand what worked and what didn't. **Do not copy code from it** — rewrite from scratch with proper types and Pydantic.
- **`motionbert-pose-old/external/MotionBERT/`** — Cloned MotionBERT repo with official inference code (`infer_wild.py`) and model configs. This is the authoritative reference for how MotionBERT expects input and produces output.
- **`DOCS.md`** — Full project documentation including skeleton conventions, FK details, and known pitfalls.

---

## Phase 1: MotionBERT Baseline

### Goal

Get Stacked Hourglass + MotionBERT producing 3D pose predictions on CMU Panoptic video, evaluated against ground truth, with correct coordinate handling and realistic MPJPE numbers.

If the MPJPE numbers are unrealistic (e.g., >1m average error), something is wrong with the pipeline and must be fixed before moving on. The old implementation achieved 20-60 cm MPJPE across examples — Phase 1 should be in that ballpark.

### What to Build

#### 1. Model Setup (`setup_models.py`)
- Clone MotionBERT repo to `external/MotionBERT/` (if not present)
- Download MotionBERT-Lite H36M checkpoint from HuggingFace: `https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin` to `checkpoints/motionbert_lite_h36m.bin`
- Stacked Hourglass weights auto-download via `pytorch-stacked-hourglass` package (HG8 model)
- Script should be idempotent (skip if already downloaded)

#### 2. Detection Pipeline (`detect.py`)
Responsible for: video frames in, 3D pose predictions out.

**2D Detection (Stacked Hourglass):**
- Input: RGB video frames
- Person detection: YOLOv8 for bounding boxes (class 0 = person), union box across all frames for temporal consistency
- Crop and resize to 256x256 with affine transform tracking
- Run HG8 (8-stack Stacked Hourglass) to get MPII 16-joint keypoints + (16, 64, 64) heatmaps
- Flip augmentation: average predictions from original + horizontally flipped input
- Output: 2D keypoints in original image coordinates, heatmaps, confidence scores

**MPII to H36M Conversion:**
- Map MPII 16-joint format to H36M 17-joint format
- Hip (H36M 0) = midpoint of MPII right hip and left hip
- Spine (H36M 7) = midpoint of MPII pelvis and thorax
- Other joints: direct index mapping

**3D Lifting (MotionBERT):**
- Normalize 2D keypoints to [-1, 1] using MotionBERT's official `crop_scale` preprocessing
- Run MotionBERT-Lite forward pass (supports variable-length input up to 243 frames)
- Flip augmentation: average original + flipped predictions
- Output: (N, 17, 3) in MotionBERT's normalized coordinate space

**Denormalization to Camera-Space Meters:**
- Follow MotionBERT's own approach: convert normalized coords back to pixel-aligned 3D
- Then convert pixel-aligned 3D to camera-space meters using depth estimation
- Use the torso-height heuristic for depth: `root_depth = fx * assumed_height / pixel_torso_height` (assumed height ~1.38m for thorax-to-ankle)
- Per-joint camera-space conversion: `z_cam = root_depth + (z_px - root_z_px) * root_depth / fx`, then `x_cam = (u - cx) * z_cam / fx`, `y_cam = (v - cy) * z_cam / fy`
- This is where MotionBERT struggles most (Z-axis accuracy). If results are poor, investigate the denormalization math carefully before blaming the model.

#### 3. CMU Panoptic Data Loading (`panoptic.py`)
- Load camera calibration (K, R, t) from Panoptic JSON files
- Load ground truth 3D poses from `hdPose3d_stage1_coco19/` JSON files
- Convert COCO19 ground truth to H36M format
- Extract video frames from HD videos with FPS subsampling
- World-to-camera coordinate transform: `p_cam = R @ p_world + t`

#### 4. Camera Model (`camera.py`)
- Pinhole camera with intrinsics (fx, fy, cx, cy) and image size
- `world_to_image()`: 3D to 2D projection (numpy, for evaluation)
- `world_to_image_torch()`: differentiable version (for Phase 2 optimization)
- Load intrinsics from CMU Panoptic calibration

#### 5. Skeleton Definition (`skeleton.py`)
- H36M 17-joint hierarchy with parent indices
- Joint names and indices
- MPII-to-H36M and COCO19-to-H36M mapping functions
- REST_DIRECTIONS in camera convention (Y-down)
- Default bone lengths (meters)

#### 6. Evaluation (`evaluate.py`)
- MPJPE: root-relative, averaged over joints and frames
- P-MPJPE: Procrustes-aligned MPJPE (per-frame rigid alignment)
- Per-joint MPJPE breakdown
- Per-frame MPJPE breakdown
- Comparison function: MotionBERT predictions vs ground truth

#### 7. Visualization (`graphs.py`)
- Per-joint MPJPE bar chart
- Per-frame MPJPE line plot
- Summary statistics printout

#### 8. Main Entry Point (`main.py`)
- Process all 10 CMU Panoptic examples
- For each: load calibration, extract frames, run detection, evaluate against GT
- Save results JSON per example (predictions + metrics)
- Save graphs per example
- Print summary table of all examples

#### 9. Quick Test Script (`test_single.py`)
- Process only the first example
- Print detailed diagnostics: per-joint errors, coordinate ranges, bone lengths
- Useful for rapid iteration during development

### Pydantic Models (suggested, not exhaustive)

```python
class CameraParams(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int

class DetectionResult(BaseModel):
    keypoints_2d: ...     # (N, 17, 2) pixel coordinates
    heatmaps: ...         # (N, 16, 64, 64) Stacked Hourglass output
    confidence: ...       # (N, 17) per-joint confidence
    positions_3d: ...     # (N, 17, 3) camera-space meters
    bbox: ...             # bounding box used

class EvaluationResult(BaseModel):
    mpjpe: float
    p_mpjpe: float
    per_joint_mpjpe: list[float]
    per_frame_mpjpe: list[float]
    num_frames: int
    num_eval_joints: int
```

Note: tensors/arrays don't go directly in Pydantic models. Use Pydantic for metadata, configs, and result summaries. For large tensor data, store as numpy/torch and validate shapes separately, or use custom validators.

### Phase 1 Definition of Done

- `uv run python main.py` runs to completion on all 10 CMU Panoptic examples without errors
- MPJPE and P-MPJPE are reported for each example
- MPJPE values are in a realistic range (roughly 20-60 cm per example, consistent with known MotionBERT performance on in-the-wild video). If numbers are far outside this range, the pipeline has a bug.
- Results JSON files and graphs are saved per example
- All code has type hints and Pydantic validation on pipeline data structures
- No imports from `mediapipe-pose/` or any sibling directory

---

## Phase 2: FK Optimization

### Goal

Layer our differentiable FK optimization on top of MotionBERT's predictions to improve MPJPE. The optimization uses the 2D heatmaps from Stacked Hourglass as the objective signal and enforces rigid skeleton constraints.

### What to Build

#### 1. Forward Kinematics (`fk.py`)
- Axis-angle rotation representation (Rodrigues formula)
- `forward_kinematics(root_pos, root_rot, local_rots, bone_lengths)` -> (J, 3) positions
- `positions_to_fk_params(positions)` -> inverse FK for initialization
- All operations in PyTorch, fully differentiable
- See DOCS.md "Known Pitfalls" for critical edge cases (180-degree rotations, coordinate conventions)

#### 2. Scoring Function (`scoring.py`)
- Heatmap reprojection score: project 3D skeleton to 2D, evaluate against Stacked Hourglass heatmaps
- Temporal smoothness penalty: penalize large frame-to-frame angle changes
- Bone length regularization: anchor to detector-estimated median bone lengths
- Combined differentiable objective

#### 3. Optimization Loop (`optimize.py`)
- Initialize FK parameters from MotionBERT's 3D predictions via inverse FK
- Per-frame learnable parameters: root position, root rotation, local joint rotations (all axis-angle)
- Shared learnable parameters: bone lengths
- Adam optimizer with separate param groups (pose params vs bone lengths)
- **Start with 10 optimization steps** for initial testing. Only increase after confirming the loss decreases and metrics improve.
- Coarse-to-fine blur schedule on heatmaps (Phase 2 can start with a simple single-phase approach and add blur scheduling in later iterations)

#### 4. Updated Evaluation (`evaluate.py`)
- Compare three things: MotionBERT baseline vs optimized vs ground truth
- Report improvement/regression per example

#### 5. Updated Main Entry Point
- After detection, run optimization, then evaluate both baseline and optimized
- Save both sets of metrics in results JSON

### Phase 2 Definition of Done

- Optimization runs without errors on all 10 examples
- MPJPE improves (decreases) on a majority of examples compared to Phase 1 baseline
- No example regresses by more than a trivial amount (< 2 cm)
- Mean MPJPE across all examples is lower after optimization than before
- Results are saved with both baseline and optimized metrics for comparison

---

## File Structure

```
motionbert-pose/
├── main.py               # Full pipeline: all 10 examples
├── test_single.py         # Quick test: first example only
├── setup_models.py        # Download MotionBERT + Stacked Hourglass
├── detect.py              # Stacked Hourglass + MotionBERT pipeline
├── skeleton.py            # H36M 17-joint definition + mappings
├── fk.py                  # Forward/inverse kinematics (Phase 2)
├── scoring.py             # Heatmap scoring + penalties (Phase 2)
├── optimize.py            # FK optimization loop (Phase 2)
├── evaluate.py            # MPJPE, P-MPJPE, comparisons
├── graphs.py              # Visualization
├── panoptic.py            # CMU Panoptic data loading
├── models.py              # Pydantic model definitions
├── config.py              # All hyperparameters and example definitions
├── camera.py              # Pinhole camera model
├── pyproject.toml         # Dependencies (uv-managed)
├── external/              # Cloned MotionBERT repo
│   └── MotionBERT/
├── checkpoints/           # Model weights
└── training_runs/         # Output directory
    └── {run_name}/
        ├── predictions/   # Per-example JSON
        └── graphs/        # Per-example visualizations
```

## Dependencies

```
torch
numpy
opencv-python
matplotlib
scipy
tqdm
pydantic
ultralytics          # YOLOv8
pytorch-stacked-hourglass  # Stacked Hourglass HG8
```

MotionBERT is loaded from the cloned repo in `external/`, not installed as a package.

The file structure and other non-primary constraints can be subject to change if you see fit. I don't want the constraints here to stop you from achieving your goal.
