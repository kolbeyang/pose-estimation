# Architect Plan: Phase 1, Iteration 0

## Goal Summary

Build a clean, standalone MotionBERT-based 3D pose estimation pipeline in `motionbert-pose/`. The pipeline uses Stacked Hourglass for 2D detection, MotionBERT-Lite for 3D lifting, and evaluates against CMU Panoptic ground truth. All old code in `motionbert-pose/` is replaced from scratch with full type hints and Pydantic models.

**Success criteria:** `uv run python main.py` processes all 10 CMU Panoptic examples, reports MPJPE in the 20-60 cm range per example, and saves results JSON + graphs.

## Key Decisions from Research

### H36M 17-Joint Skeleton (not 16)
The old motionbert-pose code uses **17 joints** (with Head and Neck/HeadTop), unlike mediapipe-pose which uses 16 (no Head). MotionBERT outputs 17 joints natively, so we keep 17.

Joint ordering (H36M 17):
```
0: Hip, 1: RHip, 2: RKnee, 3: RAnkle, 4: LHip, 5: LKnee, 6: LAnkle,
7: Spine, 8: Thorax, 9: Neck, 10: Head/HeadTop,
11: LShoulder, 12: LElbow, 13: LWrist, 14: RShoulder, 15: RElbow, 16: RWrist
```

### Eval Joints (12 of 17)
The old code excludes 5 joints from MPJPE evaluation: Hip(0), Spine(7), Thorax(8), Neck(9), Head(10) because their definitions differ between detector, GT, and FK. We use the same 12 eval joints: `[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]`.

### MotionBERT-Lite Global Checkpoint
Use the global (not rootrel) variant: `FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`. DSTformer params: dim_in=3, dim_out=3, dim_feat=256, dim_rep=512, depth=5, num_heads=8, mlp_ratio=4, num_joints=17, maxlen=243.

### Denormalization: Torso-Height Heuristic (not solvePnP)
The old code tried solvePnP but fell back to the torso-height heuristic. The spec recommends the torso-height approach directly. Use: `root_depth = fx * 1.38 / pixel_torso_height`, then per-joint: `z_cam = root_depth + (z_px - root_z_px) * root_depth / fx`.

### Ground Truth Units
CMU Panoptic GT is in **centimeters**. We convert to meters after the world-to-camera transform: `gt_cam_m = world_to_camera(gt, R, t) * 0.01`.

### No Optimization in Phase 1
Phase 1 is detection + evaluation only. No FK optimization. The `compute_comparison` function will compare MotionBERT baseline vs GT (no "optimized" column yet).

---

## Files to Create (all in `motionbert-pose/`)

| File | Purpose |
|------|---------|
| `pyproject.toml` | uv project config with dependencies |
| `models.py` | Pydantic model definitions |
| `config.py` | Hyperparameters, paths, example definitions |
| `skeleton.py` | H36M 17-joint skeleton definition + mappings |
| `camera.py` | Pinhole camera model (numpy + torch) |
| `panoptic.py` | CMU Panoptic data loading |
| `detect.py` | Stacked Hourglass + MotionBERT pipeline |
| `evaluate.py` | MPJPE, P-MPJPE, comparison metrics |
| `graphs.py` | Matplotlib visualizations |
| `setup_models.py` | Download MotionBERT checkpoint |
| `main.py` | Full pipeline: all 10 examples |
| `test_single.py` | Quick test: first example only |

## Files NOT to Create in Phase 1

`fk.py`, `scoring.py`, `optimize.py` -- these are Phase 2.

## Existing Files to Preserve

- `motionbert-pose/external/MotionBERT/` -- already cloned, do not touch
- `motionbert-pose/checkpoints/` -- may already contain the checkpoint

---

## Step-by-Step Instructions

### Step 1: `pyproject.toml`

```toml
[project]
name = "motionbert-pose"
version = "1.0.0"
description = "MotionBERT 3D pose estimation on CMU Panoptic"
requires-python = ">=3.12"
dependencies = [
    "opencv-python",
    "torch",
    "torchvision",
    "numpy",
    "matplotlib",
    "scipy",
    "tqdm",
    "ultralytics",
    "easydict",
    "pytorch-stacked-hourglass",
    "pydantic>=2.0",
]

[tool.setuptools]
py-modules = ["main"]
```

After creating, run `uv sync` to install dependencies.

### Step 2: `models.py` -- Pydantic Models

```python
from pydantic import BaseModel, field_validator

class CameraParams(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int

class ExampleConfig(BaseModel):
    sequence_name: str
    camera_name: str
    start_frame: int
    num_frames: int
    person_idx: int

class DetectionResult(BaseModel):
    """Metadata about a detection run. Actual tensor data stored separately."""
    num_frames: int
    num_joints: int  # always 17
    image_height: int
    image_width: int
    bbox: list[float]  # [x1, y1, x2, y2] union bbox

class EvaluationResult(BaseModel):
    mpjpe: float           # meters
    p_mpjpe: float         # meters
    per_joint_mpjpe: list[float]   # 12 eval joints
    per_frame_mpjpe: list[float]
    num_frames: int
    num_eval_joints: int

class ExampleResult(BaseModel):
    name: str
    sequence: str
    camera: str
    start_frame: int
    num_frames: int
    camera_params: CameraParams
    mpjpe: float | None = None
    p_mpjpe: float | None = None
    mpjpe_cm: float | None = None
    p_mpjpe_cm: float | None = None
```

### Step 3: `config.py`

Copy the 10 EXAMPLES from `mediapipe-pose/config.py` exactly (note: the old motionbert config has a slightly different example list -- example 9 differs). Use the mediapipe-pose list as authoritative since the spec says "same 10 examples".

```python
EXAMPLES: list[tuple[str, str, int, int, int]] = [
    ("171204_pose1_sample", "00_00", 0, 100, 0),
    ("171204_pose2", "00_00", 200, 150, 0),
    ("171204_pose2", "00_00", 5000, 150, 0),
    ("171204_pose2", "00_00", 15000, 150, 0),
    ("171204_pose3", "00_00", 200, 150, 0),
    ("171204_pose3", "00_00", 4000, 150, 0),
    ("160422_ultimatum1", "00_00", 200, 150, 0),
    ("160422_ultimatum1", "00_00", 10000, 150, 0),
    ("171204_pose2", "00_00", 10000, 150, 0),
    ("171204_pose2", "00_00", 25000, 150, 0),
]

TARGET_FPS: float = 10.0
PANOPTIC_ROOT: str  # = os.path.normpath(os.path.join(_THIS_DIR, "..", "data", "panoptic-toolbox"))
TRAINING_RUNS_DIR: str  # = os.path.join(_THIS_DIR, "training_runs")
```

**Important:** The mediapipe-pose example list has `("171204_pose2", "00_00", 10000, 150, 0)` at index 8, while the old motionbert config has `("160422_ultimatum1", "00_00", 20000, 150, 0)`. Use the mediapipe-pose version.

### Step 4: `skeleton.py`

Define the H36M 17-joint skeleton. Key elements:

```python
NUM_JOINTS: int = 17

JOINT_NAMES: list[str] = [
    "Hip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck", "Head",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]

EVAL_JOINTS: list[int] = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
EVAL_JOINT_NAMES: list[str] = [JOINT_NAMES[j] for j in EVAL_JOINTS]
NUM_EVAL_JOINTS: int = len(EVAL_JOINTS)

PARENTS: np.ndarray  # [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
BONES: list[tuple[int, int]]  # [(PARENTS[i], i) for i in range(1, NUM_JOINTS)]

# Default bone lengths in meters (17 entries, index 0 = 0.0)
DEFAULT_BONE_LENGTHS: np.ndarray  # same as old skeleton.py

# Body groups for coloring
BODY_GROUPS: dict[str, list[int]]
GROUP_COLORS_RGB: dict[str, tuple[float, float, float]]
```

Functions:
```python
def mpii_to_h36m(keypoints_mpii: np.ndarray) -> np.ndarray:
    """(16, D) MPII -> (17, D) H36M. Hip = midpoint(rhip, lhip). Spine = midpoint(pelv, thrx). Head/HeadTop = head."""

def coco19_to_h36m(joints19: np.ndarray) -> np.ndarray:
    """(19, 3) COCO19 -> (17, 3) H36M. Head extrapolated from neck->nose direction."""
```

The MPII-to-H36M mapping (from old detect.py):
```
H36M 0 (Hip)      = midpoint(MPII 2=rhip, MPII 3=lhip)
H36M 1 (RHip)     = MPII 2
H36M 2 (RKnee)    = MPII 1
H36M 3 (RAnkle)   = MPII 0
H36M 4 (LHip)     = MPII 3
H36M 5 (LKnee)    = MPII 4
H36M 6 (LAnkle)   = MPII 5
H36M 7 (Spine)    = midpoint(MPII 6=pelv, MPII 7=thrx)
H36M 8 (Thorax)   = MPII 8 (neck)
H36M 9 (Neck)     = MPII 9 (head)
H36M 10 (Head)    = MPII 9 (head, same as neck -- no separate joint)
H36M 11 (LShoulder) = MPII 13
H36M 12 (LElbow)  = MPII 14
H36M 13 (LWrist)  = MPII 15
H36M 14 (RShoulder) = MPII 12
H36M 15 (RElbow)  = MPII 11
H36M 16 (RWrist)  = MPII 10
```

COCO19-to-H36M mapping (from old skeleton.py):
```
H36M 0 (Hip)       = COCO19 2 (BodyCenter)
H36M 1 (RHip)      = COCO19 12
H36M 2 (RKnee)     = COCO19 13
H36M 3 (RAnkle)    = COCO19 14
H36M 4 (LHip)      = COCO19 6
H36M 5 (LKnee)     = COCO19 7
H36M 6 (LAnkle)    = COCO19 8
H36M 7 (Spine)     = midpoint(COCO19 2, COCO19 0)
H36M 8 (Thorax)    = COCO19 0 (Neck)
H36M 9 (Neck)      = COCO19 1 (Nose)  -- note: naming mismatch is known
H36M 10 (Head)     = extrapolate nose + (nose - neck)
H36M 11-16         = COCO19 3-5, 9-11 (limbs)
```

### Step 5: `camera.py`

Rewrite the Camera class with type hints. Same interface as old code but standalone:

```python
class Camera:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, image_size: tuple[int, int]) -> None: ...
    def world_to_image(self, point: np.ndarray) -> np.ndarray: ...
    def world_to_image_torch(self, points: torch.Tensor) -> torch.Tensor: ...
    def to_dict(self) -> dict[str, float | list[int]]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Camera": ...
    @classmethod
    def from_panoptic_calibration(cls, K: np.ndarray, resolution: tuple[int, int]) -> "Camera": ...
```

The `from_panoptic_calibration` factory: `fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2], image_size=(resolution[1], resolution[0])` -- note resolution is (width, height) in Panoptic calibration, but image_size convention should be (height, width) for consistency.

### Step 6: `panoptic.py`

Rewrite from mediapipe-pose reference. Same functions, same logic:

```python
def load_calibration(sequence_dir: str) -> dict[str, dict[str, np.ndarray]]: ...
def load_ground_truth_frame(sequence_dir: str, frame_idx: int, person_idx: int = 0) -> np.ndarray | None: ...
def load_ground_truth_sequence(sequence_dir: str, frame_indices: list[int], person_idx: int = 0) -> list[np.ndarray | None]: ...
def world_to_camera(points_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray: ...
def extract_video_frames(video_path: str, frame_indices: list[int]) -> list[np.ndarray]: ...
def get_video_path(panoptic_root: str, sequence_name: str, camera_name: str) -> str: ...
def get_sequence_dir(panoptic_root: str, sequence_name: str) -> str: ...
```

**Critical detail**: `load_ground_truth_sequence` should call `coco19_to_h36m` to convert to 17-joint H36M format. GT is in centimeters (world coords).

### Step 7: `detect.py`

This is the largest and most critical file. Rewrite the detection pipeline.

#### 7a. YOLO Person Detection

```python
def detect_person_bbox(frames_rgb: list[np.ndarray]) -> np.ndarray:
    """Detect persons with YOLOv8, return union bounding box.

    Returns:
        (4,) array [x1, y1, x2, y2] -- union of largest-person bbox across all frames.
    """
```

Use `YOLO("yolov8n.pt")`, class 0, pick largest-area detection per frame, compute union bbox.

#### 7b. Crop and Resize

```python
def crop_and_resize(frame: np.ndarray, bbox: np.ndarray, target_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Crop to bbox with 20% padding, resize to target_size x target_size.

    Returns:
        (cropped_resized, affine_transform) where affine maps target_size coords -> original pixel coords.
    """
```

Same logic as old code. Affine is (2, 3) matrix.

#### 7c. Stacked Hourglass

```python
def run_hourglass(
    frames_rgb: list[np.ndarray],
    bbox: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Run HG8 on cropped frames with flip augmentation.

    Returns:
        keypoints_2d: list of (16, 3) in original pixel coords (x, y, confidence)
        heatmaps: list of (16, 64, 64) raw heatmaps
        affine: (2, 3) affine from 256-crop -> original pixel coords
    """
```

Key details:
- Load `hg8(pretrained=True)` with CPU fallback for map_location
- Per frame: crop, normalize to [0,1], CHW, forward pass, take last stack output
- Flip augmentation: horizontally flip input, forward, flip heatmaps back (swap left/right pairs), average
- Parse heatmaps: argmax per joint in 64x64 space, scale to 256 (*4), apply affine to get pixel coords
- MPII flip pairs: `[(0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)]`

#### 7d. MotionBERT Lifting

```python
def load_motionbert_model() -> torch.nn.Module:
    """Load MotionBERT-Lite global model."""

def run_motionbert(
    keypoints_2d_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Lift 2D to 3D using MotionBERT.

    Returns:
        (N, 17, 3) pixel-aligned 3D positions.
    """
```

Processing steps:
1. Convert MPII 16-joint to H36M 17-joint (via `mpii_to_h36m`)
2. `crop_scale` normalization: compute bounding box of all valid keypoints, normalize to [-1, 1]
3. Forward pass (no padding to 243 -- model handles variable length)
4. Flip augmentation: flip input (negate X, swap left/right joints), forward, flip back, average
5. Post-process: `positions_3d[0, 0, 2] = 0` (zero first frame root Z for global variant)
6. Denormalize from crop_scale to pixel-aligned coords:
   - `positions_3d *= scale / 2.0`
   - `positions_3d[:, :, 0] += xs + scale / 2.0`
   - `positions_3d[:, :, 1] += ys + scale / 2.0`

**crop_scale function** (match official MotionBERT `lib/utils/utils_data.py`):
```python
def crop_scale(motion: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize 2D keypoints to [-1, 1].
    Returns (normalized, {"xs": float, "ys": float, "scale": float})"""
```

**flip_data function** (match official):
```python
def flip_data(data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Flip H36M 17-joint data: negate X, swap L/R joints."""
    # left_joints = [4, 5, 6, 11, 12, 13]
    # right_joints = [1, 2, 3, 14, 15, 16]
```

#### 7e. Denormalization to Camera-Space Meters

```python
def pixel_aligned_to_camera_space(
    kp_3d: np.ndarray,
    kp_2d: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Convert pixel-aligned 3D to camera-space meters using torso-height heuristic.

    Args:
        kp_3d: (17, 3) pixel-aligned 3D from MotionBERT
        kp_2d: (17, 2) pixel coordinates
        fx, fy, cx, cy: camera intrinsics

    Returns:
        (17, 3) camera-space meters
    """
```

Logic:
1. Compute pixel torso height: `abs(thorax_2d[1] - ankle_mid_2d[1])` where thorax=joint 8, ankle_mid=(joint 3 + joint 6)/2
2. `root_depth = fx * 1.38 / pixel_height` (clamped to [1.0, 8.0])
3. For each joint j: `z_cam = root_depth + (z_px[j] - z_px[root]) * root_depth / fx`, `x_cam = (u - cx) * z_cam / fx`, `y_cam = (v - cy) * z_cam / fy`

#### 7f. Pipeline Wrapper

```python
def detect_poses(
    frames_rgb: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Full detection pipeline.

    Returns:
        keypoints_2d: list of (17, 2) pixel coords
        keypoints_3d: list of (17, 3) pixel-aligned 3D
        confidence: list of (17,) confidence scores
        heatmaps: list of (16, 64, 64) raw heatmaps
        affine: (2, 3) affine transform
    """
```

### Step 8: `evaluate.py`

```python
def mpjpe(predicted: np.ndarray, target: np.ndarray) -> float: ...
def mpjpe_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray: ...
def root_relative(positions: np.ndarray) -> np.ndarray: ...
def procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray: ...
def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float: ...

def compute_comparison(
    detector_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
) -> dict[str, Any]:
    """Compare detector predictions vs ground truth.

    Phase 1 version: only detector vs GT (no optimized).
    Uses root-relative comparison on 12 eval joints.

    Returns dict with keys: n_frames, n_frames_with_gt, mpjpe, p_mpjpe,
    per_joint_mpjpe, per_frame_mpjpe, per_joint_p_mpjpe
    """
```

**Important difference from old code**: Phase 1 has no "optimized" column. The comparison is just detector vs GT. The function signature should be simpler. However, to make Phase 2 easier, name the keys generically: `det_mpjpe`, `det_p_mpjpe`, etc.

### Step 9: `graphs.py`

```python
def generate_per_joint_error_bar(per_joint: list[float], output_dir: str) -> None: ...
def generate_per_frame_mpjpe(per_frame: list[float], output_dir: str) -> None: ...
def generate_aggregate_summary(all_metrics: list[dict], output_dir: str) -> None: ...
```

Phase 1: simpler graphs since there's only one prediction (no detector vs optimized comparison). The aggregate summary prints a table and generates a bar chart of MPJPE across examples.

Use `matplotlib.use("Agg")` to avoid display issues.

### Step 10: `setup_models.py`

Same as old code, rewritten with type hints:

```python
def clone_repo(url: str, target_dir: str) -> None: ...
def download_file(url: str, target_path: str) -> None: ...
def main() -> None: ...
```

Download MotionBERT-Lite H36M checkpoint from:
`https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin`

to `checkpoints/motionbert_lite_h36m.bin`.

### Step 11: `main.py`

```python
def process_example(
    seq_name: str,
    camera_name: str,
    start_frame: int,
    num_frames: int,
    person_idx: int,
    run_dir: str,
) -> dict[str, Any]:
    """Process one CMU Panoptic example end-to-end."""
```

Steps per example:
1. Load camera calibration, create Camera object
2. Extract video frames (subsample 30fps -> 10fps)
3. Run `detect_poses()` to get 2D keypoints, 3D pixel-aligned, confidence, heatmaps
4. Convert pixel-aligned 3D to camera-space meters (`pixel_aligned_to_camera_space`)
5. Load GT, convert to camera coords (centimeters), then to meters (*0.01)
6. Evaluate: `compute_comparison(detector_3d_cam, gt_cam)`
7. Save predictions JSON + graphs

```python
def main() -> None:
    """Process all 10 examples, save aggregate results."""
```

### Step 12: `test_single.py`

Minimal version of main.py that processes only the first example (or user-specified index), prints detailed diagnostics:
- Per-joint errors
- Coordinate ranges of detector predictions
- Bone lengths from detector vs GT
- 2D reprojection sanity check

---

## Integration Points

### MotionBERT Model Loading
The MotionBERT code lives in `external/MotionBERT/`. We need to add it to sys.path to import `DSTformer`:
```python
sys.path.insert(0, os.path.join(SCRIPT_DIR, "external", "MotionBERT"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "external", "MotionBERT", "lib"))
from lib.model.DSTformer import DSTformer
```

The checkpoint loads with `checkpoint['model_pos']` and keys are prefixed with `module.` (from DataParallel). Strip prefix: `{k.replace("module.", ""): v for k, v in state_dict.items()}`.

### Camera Calibration Format
Panoptic calibration JSON has cameras as a list. Camera names use the convention `"00_XX"` for HD cameras. The old code searches for matching camera name. Keep this logic but make it more robust:
- Try exact match first
- Then try `"00_{node}"` format
- Log warning if falling back

### Data Path Convention
`PANOPTIC_ROOT` points to `../data/panoptic-toolbox/` relative to `motionbert-pose/`. Video at `{root}/{sequence}/hdVideos/hd_{camera}.mp4`. GT at `{root}/{sequence}/hdPose3d_stage1_coco19/body3DScene_{frame:08d}.json`.

---

## Risks and Edge Cases

### Risk 1: MPJPE Outside Expected Range
**Symptom**: MPJPE > 100cm or < 5cm.
**Likely cause**: Unit mismatch (meters vs centimeters) or denormalization bug.
**Mitigation**: In `test_single.py`, print coordinate ranges of GT and predictions. GT should be roughly 0-5m in camera Z. Predictions from torso-height heuristic should be similar.

### Risk 2: MotionBERT Output Quality
**Symptom**: Garbled 3D predictions.
**Likely cause**: Wrong input normalization or model architecture mismatch.
**Mitigation**: Compare crop_scale output range against official code. Verify DSTformer constructor params match the YAML config exactly.

### Risk 3: Stacked Hourglass CPU Loading
**Symptom**: RuntimeError on pretrained weight loading.
**Cause**: Checkpoint saved with CUDA, loaded on CPU-only machine.
**Mitigation**: Catch RuntimeError, manually load with `map_location="cpu"` (as done in old code).

### Risk 4: Missing Ground Truth Frames
**Symptom**: GT files not found for some frame indices.
**Cause**: Not all frames have GT annotations in Panoptic.
**Mitigation**: `load_ground_truth_frame` returns None, and all downstream code handles None entries gracefully.

### Risk 5: Variable-Length MotionBERT Input
**Symptom**: Model crashes on sequences != 243 frames.
**Cause**: MotionBERT's temporal embedding supports variable length.
**Mitigation**: Do NOT pad to 243 frames. Clip if > 243. The old code confirmed this works.

### Risk 6: flip_data Left/Right Swap
**Symptom**: Flipped predictions are mirrored wrong.
**Cause**: Wrong left/right joint indices.
**Mitigation**: H36M left joints = [4,5,6,11,12,13], right = [1,2,3,14,15,16]. Negate X coordinate. This matches the official `flip_data` from MotionBERT.

### Risk 7: Heatmap Flip Pairs
**Symptom**: Averaged heatmaps are misaligned.
**Cause**: Wrong MPII symmetric joint pairs.
**Mitigation**: MPII pairs: [(0,5), (1,4), (2,3), (10,15), (11,14), (12,13)]. After flipping heatmaps horizontally, swap these pairs.

---

## Validation Checklist (for Developer)

Before declaring Phase 1 done:

- [ ] `uv run python setup_models.py` downloads checkpoint successfully
- [ ] `uv run python test_single.py` completes without errors
- [ ] test_single.py prints MPJPE in range 20-60 cm
- [ ] GT coordinate Z values are positive and in range 1-6m
- [ ] Detector coordinate Z values are positive and in range 1-6m
- [ ] `uv run python main.py` processes all 10 examples
- [ ] All examples have MPJPE in roughly 20-60 cm range
- [ ] Results JSON files saved in `training_runs/{run_name}/predictions/`
- [ ] Graph PNGs saved in `training_runs/{run_name}/graphs/`
- [ ] No imports from `mediapipe-pose/` or other sibling directories
- [ ] All functions have type hints
- [ ] Pydantic models used for config and result summaries

---

## Build Order (Recommended)

1. `pyproject.toml` + `uv sync`
2. `models.py` (Pydantic definitions)
3. `config.py` (constants)
4. `skeleton.py` (joint definitions + mappings)
5. `camera.py` (pinhole model)
6. `panoptic.py` (data loading)
7. `setup_models.py` (checkpoint download)
8. `detect.py` (full detection pipeline)
9. `evaluate.py` (metrics)
10. `graphs.py` (visualization)
11. `test_single.py` (quick validation)
12. `main.py` (full pipeline)

Build and test incrementally: after steps 1-7, you can verify data loading works. After step 8, verify detection produces reasonable outputs. After step 9-10, run the full evaluation.
