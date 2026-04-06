# Project Documentation

## Title

**Improved Single Camera 3D Pose Estimation with Reprojection and Filtering**


## Problem Statement

Modern deep learning models for 3D human pose estimation produce unconstrained outputs: nothing prevents them from predicting skeletons with impossible bone lengths, joints that bend in invalid directions, or limbs that teleport between frames. This project proposes an optimization-based approach that enforces hard physical constraints by construction.

The core challenge is **monocular depth ambiguity** - a single 2D image corresponds to infinitely many valid 3D configurations. Current methods (MotionBERT, VideoPose3D) handle this with learned priors but output raw 3D coordinates with no structural guarantees. Post-hoc fixes like Kalman filters or SmoothNet treat symptoms, not causes.

## Our Approach

Instead of training a network to regress 3D coordinates, we formulate pose estimation as a **constrained optimization problem** over a **rigid skeleton**.

### Pipeline

```
2D Heatmaps (input)
    -> Rigid skeleton parameterization (joint angles + root position)
    -> Differentiable forward kinematics (angles -> 3D positions)
    -> 2D reprojection via pinhole camera model
    -> Heatmap scoring (how well projected joints match observed heatmaps)
    -> Gradient update (Adam optimizer)
    -> Repeat with coarse-to-fine blur schedule
```

### Why This Works

- **Bone lengths are shared parameters across all frames** - the optimizer literally cannot produce a skeleton where an arm changes length between frames
- **Joint angles are the optimization variables** - every output is anatomically plausible by construction
- **Heatmaps preserve spatial uncertainty** - richer signal than extracted keypoint coordinates alone
- **Temporal smoothness penalties** - penalize large frame-to-frame angle changes, preventing teleportation
- **No training data required** - this is optimization-based, not learned

### Key Metrics

- **MPJPE** (Mean Per Joint Position Error): Euclidean distance between predicted and ground truth 3D joint positions, averaged over all joints and frames. This is the standard metric in 3D pose estimation.
- **P-MPJPE** (Procrustes-aligned MPJPE): MPJPE after optimal rigid alignment per frame, isolating shape accuracy from global positioning.
- **MPJVE** (Mean Per Joint Velocity Error): Measures temporal smoothness of predicted trajectories.

## Repository Structure

```
pose-estimation/
├── claude-team/              # Agent workflow (specs, docs, personas)
│   ├── DOCS.md               # This file
│   ├── PLAN.md               # Current work plan
│   ├── TODO.md               # Task tracking
│   ├── specs/                # Specification documents
│   └── personas/             # Review personas
│
├── mediapipe-pose/           # PRIMARY IMPLEMENTATION (see detailed section below)
│   ├── main.py               # End-to-end pipeline for all examples
│   ├── experiment.py         # Single-example quick runner
│   ├── config.py             # Hyperparameters & example definitions
│   ├── detect.py             # MediaPipe 2D detection + solvePnP
│   ├── fk.py                 # Forward/inverse kinematics (axis-angle)
│   ├── skeleton.py           # H36M 16-joint skeleton + joint mappings
│   ├── optimize.py           # Adam optimization loop
│   ├── scoring.py            # Heatmap score + motion penalties
│   ├── evaluate.py           # MPJPE / P-MPJPE / MPJVE computation
│   ├── graphs.py             # Matplotlib visualizations
│   ├── overlay_video.py      # Heatmap + skeleton overlay video
│   ├── panoptic.py           # CMU Panoptic dataset loading
│   ├── visualize.py          # VPython skeleton viewer
│   ├── model/camera.py       # Pinhole camera (numpy + torch)
│   ├── pose_landmarker_lite.task  # MediaPipe model weights
│   ├── training_runs/        # All experiment outputs
│   └── OPTIMIZATION_ATTEMPTS.md   # FK debugging log
│
├── toy-arm/                  # Early prototype: 2-segment arm optimization
├── motionbert-pose/          # MotionBERT baseline comparison
├── rtmw-pose/                # RTMPose/RTMW baseline comparison
├── comparison/               # Cross-method comparison tooling
├── model/                    # Original toy arm classes (Arm, Environment)
├── data/                     # Shared data directory
├── videos/                   # Source video files
├── images/                   # Rendered images
└── training_runs/            # Legacy training outputs (toy arm era)
```

## mediapipe-pose/ - Detailed Architecture

This is where the bulk of the capstone work lives. It implements the full optimization pipeline against real video data from the CMU Panoptic dataset.

### Skeleton Model (`skeleton.py`)

Uses **Human3.6M (H36M) 16-joint format** (head joint removed from the standard 17):

```
Hip (root) -> RHip -> RKnee -> RAnkle
           -> LHip -> LKnee -> LAnkle
           -> Spine -> Thorax -> Nose
                             -> LShoulder -> LElbow -> LWrist
                             -> RShoulder -> RElbow -> RWrist
```

Key details:
- **PARENTS array** defines the kinematic tree (each joint's parent index)
- **REST_DIRECTIONS** are bone unit vectors in **camera convention (Y-down)**, not world Y-up. Getting this wrong was a major bug (see pitfalls below).
- **Joint mapping functions** convert between formats:
  - `mediapipe_to_h36m()`: MediaPipe 33 landmarks -> H36M 16 joints (some joints are averages of multiple MediaPipe landmarks)
  - `coco19_to_h36m()`: CMU Panoptic COCO19 -> H36M 16 joints (for ground truth)
  - `mediapipe_visibility_to_h36m()`: Aggregates visibility scores for averaged joints

### Forward Kinematics (`fk.py`)

All rotations use **axis-angle representation** (3D vector where direction = axis, magnitude = angle). This avoids gimbal lock.

**`forward_kinematics(root_pos, root_rot, local_rots, bone_lengths)`**
- Converts axis-angle to rotation matrices via Rodrigues formula
- Walks the kinematic tree: each joint's world rotation = parent_world_rot @ local_rot
- Each joint's position = parent_pos + world_rot @ (rest_direction * bone_length)
- Returns (J, 3) tensor of 3D joint positions

**`positions_to_fk_params(positions)`** (inverse FK)
- Extracts bone lengths from inter-joint distances
- Computes root rotation from spine/hip directions
- Recovers per-joint local rotations via `_rotation_between_vectors()`
- Returns (root_pos, root_rot, local_rots, bone_lengths) for initialization

### Detection Pipeline (`detect.py`)

**`detect_poses(frames_rgb)`**
1. Runs MediaPipe PoseLandmarker (Lite model, 33 landmarks)
2. Returns 2D pixel coords, 3D hip-relative coords, and visibility scores
3. Maps from MediaPipe 33 landmarks to H36M 16 joints

**`mediapipe_3d_to_camera()`**
- Converts MediaPipe's hip-relative 3D to camera-space 3D
- Uses **OpenCV solvePnP** (SQPNP) with visible joints as 2D-3D correspondences
- Fallback: depth heuristic (person height ~ 1.2m) if fewer than 4 visible joints

### Scoring Function (`scoring.py`)

The optimization maximizes: `heatmap_score - position_penalty - rotation_penalty`

**`heatmap_score()`**: Analytical Gaussian log-likelihood (closed-form, O(J) not O(J*H*W))
```python
score = sum(visibility[j] * exp(-(dist_j^2) / (2 * sigma^2)))
```
where dist_j is the pixel distance between the projected joint and the detected 2D keypoint.

**`motion_penalty_position()`**: Penalizes root position jumps between consecutive frames.

**`motion_penalty_rotation()`**: Penalizes large angle changes using chord distance (wraps correctly at +/-pi).

### Optimization Loop (`optimize.py`)

**Per-frame learnable parameters:**
- Root position: (3,) per frame
- Root rotation: (3,) axis-angle per frame
- Local joint rotations: (J, 3) axis-angle per frame

**Shared across all frames:**
- Bone lengths: (J,) initialized as median across frames

**Adam optimizer with two param groups:**
- Pose parameters: LR = 0.001
- Bone lengths: LR = 0.0001 (10x slower - these should be stable)

**Per-joint rotation penalty weights:** Trunk joints (3.0) weighted higher than extremities (0.1-0.2), so the optimizer penalizes wild torso swings more than wrist adjustments.

**Current hyperparameters (from config.py):**
- SIGMA = 50.0 (Gaussian heatmap width in pixels)
- NUM_STEPS = 300
- POSITION_PENALTY_WEIGHT = 50.0
- BONE_LENGTH_LR = 0.0001

### Camera Model (`model/camera.py`)

Standard pinhole camera:
- `world_to_image()`: NumPy version for evaluation
- `world_to_image_torch()`: Differentiable PyTorch version for optimization (clamps Z >= 0.01)
- Intrinsics loaded from CMU Panoptic calibration files

### Evaluation (`evaluate.py`)

**`compute_comparison()`**: Compares MediaPipe baseline vs optimized vs ground truth
- Root-relative MPJPE (subtract hip position before comparing)
- P-MPJPE (Procrustes alignment per frame)
- MPJVE (velocity error for temporal smoothness)
- Per-joint and per-frame breakdowns

### CMU Panoptic Data Loading (`panoptic.py`)

- `load_calibration()`: Camera K, R, t from Panoptic JSON
- `load_ground_truth_frame()`: COCO19 3D poses from `hdPose3d_stage1_coco19/` files
- `extract_video_frames()`: HD video frame extraction at configurable FPS
- `world_to_camera()`: Rigid transform p_cam = R @ p_world + t

### Visualization

**`graphs.py`**: Per-joint trajectory plots, loss curves, per-joint error bars, per-frame MPJPE, bone length comparisons, summary grids, velocity error plots.

**`overlay_video.py`**: Renders heatmaps on original video frames with projected skeletons (green = MediaPipe, red = optimized, blue = ground truth).

### Execution Flow

```
For each of 10 CMU Panoptic examples:
  1. Load camera calibration (K, R, t)
  2. Extract HD video frames (30fps -> 10fps subsampled)
  3. Run MediaPipe detection (33 landmarks -> 16 H36M joints)
  4. solvePnP to get camera-space 3D initialization
  5. Inverse FK to get initial angle parameters
  6. Adam optimization (300 steps) against 2D heatmap targets
  7. Evaluate MPJPE / P-MPJPE / MPJVE vs ground truth
  8. Save prediction JSON + graphs + overlay video
```

### Current Results

- **Baseline (MediaPipe):** ~18.89 cm mean MPJPE across 9 examples
- **After optimization:** ~17.35 cm mean MPJPE (~1.5 cm improvement)
- 7/9 examples improved, 2/9 had slight regression (<1 cm)
- ~12-13 cm appears to be the error floor (limited by depth ambiguity and detection noise)

### Output Format

Each example produces a JSON file in `training_runs/` with:
```json
{
  "sequence": "171204_pose1_sample",
  "camera": "00_00",
  "joint_names": ["Hip", "RHip", ...],
  "camera_intrinsics": {"fx": ..., "fy": ..., "cx": ..., "cy": ...},
  "metrics": {
    "mp_mpjpe": 0.1458,
    "opt_mpjpe": 0.1735,
    "mp_p_mpjpe": ...,
    "opt_p_mpjpe": ...,
    "mp_per_joint": [...],
    "opt_per_joint": [...]
  },
  "frames": [
    {
      "frame_idx": 0,
      "mediapipe_3d": [[x, y, z], ...],
      "optimized_3d": [[x, y, z], ...],
      "mediapipe_2d": [[u, v], ...],
      "visibility": [...],
      "ground_truth_3d": [[x, y, z], ...]
    }
  ]
}
```

## Historical Development

### Phase 1: Toy Arm (`toy-arm/`, root `model/`)
A simplified 2-segment, 3-joint arm model used to validate the core optimization framework:
- Reprojection scoring against synthetic heatmaps
- Coarse-to-fine Gaussian blur schedule
- Temporal smoothness constraints
- BVH mocap data for realistic motion sequences
- Confirmed that the approach converges and the math works before scaling up

### Phase 2: Full Skeleton with BVH Data (root `model/`, `training_runs/`)
Scaled to a 38-joint CMU BVH skeleton (21 tracked joints, 96 DOF per frame):
- Differentiable FK in pure PyTorch
- 3-phase coarse-to-fine optimization (blur=20 -> blur=8 -> blur=0)
- Adam optimizer with separate param groups (root_pos gets 5x LR)
- Preloaded heatmaps into memory (lazy loading was too slow)
- Separable Gaussian blur for efficiency

### Phase 3: Real Video with MediaPipe (`mediapipe-pose/`)
Applied the framework to real video from CMU Panoptic:
- MediaPipe for 2D detection (replacing synthetic heatmaps)
- solvePnP for initial 3D estimation
- Axis-angle rotation representation (replacing Euler angles to avoid gimbal lock)
- H36M 16-joint skeleton (simplified from 38-joint BVH)
- Evaluation against Panoptic 3D ground truth
- Significant FK debugging (see pitfalls)

## Known Pitfalls and Lessons Learned

1. **REST_DIRECTIONS must match the coordinate convention.** Camera coords are Y-down; the original rest directions assumed Y-up. This caused the inverse FK to produce 180-degree rotations everywhere, leading to 48-87 cm roundtrip error. Fixed by flipping to camera convention.

2. **Axis-angle 180-degree edge case.** The antisymmetric part of a rotation matrix vanishes at exactly 180 degrees, causing `rotation_matrix_to_axis_angle()` to return a zero vector. Fixed by extracting the axis from the symmetric part (R+I)/2.

3. **Anti-parallel vector rotation.** `rotation_between_vectors()` for opposite-direction vectors returned a reflection (-I) instead of a proper rotation. Fixed with R = -I + 2nn^T.

4. **Lazy heatmap loading kills performance.** Reading heatmaps from disk per optimization step is orders of magnitude slower than preloading into memory.

5. **SGD converges very slowly** for this problem. Adam is dramatically better.

6. **Grad clipping at 1.0 is too aggressive** for a 96-DOF skeleton. Use 10.0 or higher.

7. **VPython requires `float()` wrapping** for numpy float32 values.

## Baseline Comparisons (Planned)

- **MotionBERT**: Learning-based temporal model (transformer architecture)
- **PoseFix**: Model-agnostic pose refinement network
- **Ablations**: Our method without temporal smoothness; without coarse-to-fine scheduling

## Tech Stack

- Python 3.12, managed with `uv`
- PyTorch (differentiable FK + optimization)
- MediaPipe (2D pose detection)
- OpenCV (video I/O, solvePnP)
- NumPy / SciPy (linear algebra, Procrustes alignment)
- Matplotlib (visualization)
- VPython (3D skeleton viewer)
- CMU Panoptic Dataset (evaluation data)
