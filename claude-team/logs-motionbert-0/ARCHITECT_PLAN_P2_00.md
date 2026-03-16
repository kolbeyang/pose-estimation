# Architect Plan: Phase 2, Iteration 0

## Goal Summary

Layer differentiable FK optimization on top of MotionBERT's Phase 1 predictions to reduce MPJPE. The optimizer takes MotionBERT's camera-space 3D predictions, converts them to FK parameters (root position, root rotation, per-joint local rotations, shared bone lengths), then optimizes those parameters against the 2D heatmaps from Stacked Hourglass. The rigid skeleton parameterization prevents impossible bone lengths and enforces physical constraints by construction.

**Target**: Mean MPJPE improvement across 10 examples. Majority of examples should improve. No example regresses by more than 2 cm. Start with 10 optimization steps, scale up after confirming correctness.

## Key Lessons from Prior Work

From `mediapipe-pose/OPTIMIZATION_ATTEMPTS.md`:
1. **The FK roundtrip must be near-perfect.** The #1 failure mode was REST_DIRECTIONS using Y-up in Y-down camera space, causing 180-degree rotations everywhere and 48-87 cm roundtrip error. Once fixed, roundtrip error was 0.0000 cm.
2. **Three bugs to handle**: (a) REST_DIRECTIONS must match camera convention (Y-down), (b) `_rotation_matrix_to_axis_angle` must handle the 180-degree edge case via symmetric part extraction, (c) `_rotation_between_vectors` must return proper 180-degree rotation (R = -I + 2nn^T), not a reflection (-I).
3. **Best hyperparameters** from mediapipe-pose: SIGMA=50, POS_W=50, LR=0.001, 300 steps, per-joint rotation weights (trunk high, extremities low), BONE_LENGTH_LR=0.0001.
4. **Optimizer works across wide range of settings** once FK is correct.

## Key Differences from mediapipe-pose

- **17 joints** (not 16): motionbert-pose includes Head (joint 10), with parent chain: Thorax(8) -> Neck(9) -> Head(10). The mediapipe-pose version has Thorax -> Nose directly.
- **PARENTS array**: `[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]` (17 joints)
- **Phase 1 MPJPE is much higher** (44.89 cm mean vs 18.89 cm for mediapipe-pose) due to MotionBERT's depth estimation challenges. More room for improvement, but also harder initialization.
- **Heatmaps are MPII 16-joint format** (from Stacked Hourglass), not H36M 17-joint. Need MPII-to-H36M mapping for 2D targets, and some H36M joints (Hip, Spine, Head) won't have direct heatmap targets.
- **2D targets come from parsed heatmap peaks**, not from MediaPipe landmarks. The `detect_poses` function already returns H36M-format 2D keypoints and confidence.

## Files to Create

### 1. `motionbert-pose/fk.py` (NEW)
Forward and inverse kinematics, axis-angle rotation utilities.

### 2. `motionbert-pose/scoring.py` (NEW)
Heatmap scoring and temporal penalties.

### 3. `motionbert-pose/optimize.py` (NEW)
FK optimization loop.

## Files to Modify

### 4. `motionbert-pose/skeleton.py` (MODIFY)
Add REST_DIRECTIONS constant (17 joints, Y-down camera convention).

### 5. `motionbert-pose/config.py` (MODIFY)
Add optimization hyperparameters.

### 6. `motionbert-pose/evaluate.py` (MODIFY)
Add `compute_comparison_with_optimization()` that compares detector baseline vs optimized vs GT.

### 7. `motionbert-pose/graphs.py` (MODIFY)
Update graphs to show both detector and optimized results side-by-side.

### 8. `motionbert-pose/models.py` (MODIFY)
Add `OptimizationConfig` and `OptimizationResult` Pydantic models.

### 9. `motionbert-pose/main.py` (MODIFY)
After detection, run optimization, then evaluate both baseline and optimized. Save both sets of metrics.

---

## Step-by-Step Instructions

### Step 1: Add REST_DIRECTIONS to skeleton.py

Add after `DEFAULT_BONE_LENGTHS`:

```python
# Rest-pose bone directions (unit vectors when all local rotations are zero).
# Camera convention: Y-down, person facing camera.
# Person's right = camera left (-X), head direction = -Y, feet = +Y.
REST_DIRECTIONS: np.ndarray = np.array([
    [0, 0, 0],       # 0: Hip (root, unused)
    [-1, 0, 0],      # 1: Hip -> RHip  (camera-left = person's right)
    [0, 1, 0],       # 2: RHip -> RKnee  (down)
    [0, 1, 0],       # 3: RKnee -> RAnkle  (down)
    [1, 0, 0],       # 4: Hip -> LHip  (camera-right = person's left)
    [0, 1, 0],       # 5: LHip -> LKnee  (down)
    [0, 1, 0],       # 6: LKnee -> LAnkle  (down)
    [0, -1, 0],      # 7: Hip -> Spine  (up = -Y in camera)
    [0, -1, 0],      # 8: Spine -> Thorax  (up)
    [0, -1, 0],      # 9: Thorax -> Neck  (up)
    [0, -1, 0],      # 10: Neck -> Head  (up)
    [1, 0, 0],       # 11: Thorax -> LShoulder  (camera-right = person's left)
    [0, 1, 0],       # 12: LShoulder -> LElbow  (down)
    [0, 1, 0],       # 13: LElbow -> LWrist  (down)
    [-1, 0, 0],      # 14: Thorax -> RShoulder  (camera-left = person's right)
    [0, 1, 0],       # 15: RShoulder -> RElbow  (down)
    [0, 1, 0],       # 16: RElbow -> RWrist  (down)
], dtype=np.float64)
```

Note the 17-joint ordering matches the PARENTS/JOINT_NAMES arrays exactly. Joints 9 (Neck) and 10 (Head) are new vs the 16-joint mediapipe-pose version.

### Step 2: Add optimization hyperparameters to config.py

Add at end of `config.py`:

```python
import numpy as np

# --- Optimization (Phase 2) ---
NUM_STEPS: int = 10  # Start small for testing, increase after confirming correctness
LEARNING_RATE: float = 0.001
BONE_LENGTH_LR: float = 0.0001

# Gaussian sigma for heatmap scoring (pixels)
SIGMA: float = 50.0

# Motion penalty weights
POSITION_PENALTY_WEIGHT: float = 50.0

# Per-joint rotation penalty weights (17 joints)
# Trunk joints penalized more to prevent wild torso swings.
# Extremities penalized less so they can track fast motion.
ROTATION_PENALTY_SCALAR: float = 10.0
ROTATION_PENALTY_PER_JOINT: np.ndarray = np.array([
    ROTATION_PENALTY_SCALAR * 3.0,   # 0: Hip (root rotation)
    ROTATION_PENALTY_SCALAR * 1.0,   # 1: RHip
    ROTATION_PENALTY_SCALAR * 0.5,   # 2: RKnee
    ROTATION_PENALTY_SCALAR * 0.2,   # 3: RAnkle
    ROTATION_PENALTY_SCALAR * 1.0,   # 4: LHip
    ROTATION_PENALTY_SCALAR * 0.5,   # 5: LKnee
    ROTATION_PENALTY_SCALAR * 0.2,   # 6: LAnkle
    ROTATION_PENALTY_SCALAR * 1.0,   # 7: Spine
    ROTATION_PENALTY_SCALAR * 1.0,   # 8: Thorax
    ROTATION_PENALTY_SCALAR * 0.5,   # 9: Neck
    ROTATION_PENALTY_SCALAR * 0.5,   # 10: Head
    ROTATION_PENALTY_SCALAR * 0.5,   # 11: LShoulder
    ROTATION_PENALTY_SCALAR * 0.3,   # 12: LElbow
    ROTATION_PENALTY_SCALAR * 0.1,   # 13: LWrist
    ROTATION_PENALTY_SCALAR * 0.5,   # 14: RShoulder
    ROTATION_PENALTY_SCALAR * 0.3,   # 15: RElbow
    ROTATION_PENALTY_SCALAR * 0.1,   # 16: RWrist
], dtype=np.float64)

# Visibility threshold: joints below this are ignored in scoring
VISIBILITY_THRESHOLD: float = 0.5
```

### Step 3: Create fk.py

This is the most critical file. Rewrite from scratch following mediapipe-pose/fk.py as reference but adapted for 17 joints.

**Functions to implement:**

```python
def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (3,) to 3x3 rotation matrix via Rodrigues' formula.

    Handle small-angle case (angle < 1e-8) by returning identity.
    """

def forward_kinematics(
    root_pos: torch.Tensor,      # (3,)
    root_rot: torch.Tensor,      # (3,) axis-angle
    local_rots: torch.Tensor,    # (17, 3) axis-angle per joint
    bone_lengths: torch.Tensor,  # (17,)
) -> torch.Tensor:
    """Compute 3D joint positions from FK parameters.

    Returns: (17, 3) world positions.

    Algorithm:
    1. positions[0] = root_pos
    2. rotations[0] = axis_angle_to_matrix(root_rot)
    3. For j = 1..16:
       a. R_local = axis_angle_to_matrix(local_rots[j])
       b. R_world = rotations[parent[j]] @ R_local
       c. direction = R_world @ rest_dirs[j]
       d. positions[j] = positions[parent[j]] + bone_lengths[j] * direction
    """

def _rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix mapping unit vector v_from -> v_to.

    CRITICAL: Handle anti-parallel case (dot ~= -1) with proper 180-degree rotation:
    R = -I + 2*nn^T where n is any perpendicular axis.
    Do NOT return -I (that's a reflection, not a rotation).
    """

def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to axis-angle (3,).

    CRITICAL: Handle near-180-degree case where antisymmetric part vanishes.
    Use (R + I)/2 = nn^T to extract axis from the column with largest norm.
    """

def positions_to_fk_params(
    positions: np.ndarray,  # (17, 3)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert joint positions to FK parameters (inverse FK).

    Returns: (root_pos, root_rot, local_rots, bone_lengths)

    Algorithm:
    1. root_pos = positions[0]
    2. bone_lengths[j] = ||positions[j] - positions[parent[j]]|| for j > 0
    3. Root rotation from spine/hip directions:
       - spine_dir = positions[8] - positions[0]  (Thorax - Hip)
       - hip_axis = positions[1] - positions[4]   (RHip - LHip)
       - Build orthonormal frame, mapping rest axes to body axes
       - Rest convention: right=(-1,0,0), up=(0,-1,0), forward=(0,0,-1)
       - R_root = column_stack([-right, -up, -forward])
    4. For each child joint j > 0:
       - actual_dir = (positions[j] - positions[parent[j]]) / bone_length[j]
       - R_world = rotation_between_vectors(rest_dirs[j], actual_dir)
       - R_local = R_parent^T @ R_world
       - local_rots[j] = rotation_matrix_to_axis_angle(R_local)
    """
```

**CRITICAL VALIDATION**: After implementing, verify that `forward_kinematics(positions_to_fk_params(positions))` roundtrips with near-zero error (< 0.01 cm). If roundtrip error is large, there is a bug in FK that will make optimization diverge. Print roundtrip error during development.

### Step 4: Create scoring.py

```python
def heatmap_score(
    projected_2d: torch.Tensor,    # (J, 2)
    target_2d: torch.Tensor,       # (J, 2)
    visibility: torch.Tensor,      # (J,)
    sigma: float,
) -> torch.Tensor:
    """Analytical Gaussian log-likelihood.

    score = sum(visibility[j] * exp(-(dist_j^2) / (2 * sigma^2)))

    This is O(J) -- no heatmap image needed, just pixel distance.
    """

def motion_penalty_position(
    positions_prev: torch.Tensor,  # (17, 3)
    positions_curr: torch.Tensor,  # (17, 3)
) -> torch.Tensor:
    """Penalize root position jumps between consecutive frames.

    Returns: squared L2 distance of root (joint 0).
    """

def motion_penalty_rotation(
    local_rots_prev: torch.Tensor,     # (17, 3) axis-angle
    local_rots_curr: torch.Tensor,     # (17, 3) axis-angle
    per_joint_weights: torch.Tensor,   # (17,)
) -> torch.Tensor:
    """Penalize rotation jumps using chord distance.

    Chord distance wraps correctly at +/-pi, unlike raw axis-angle diff.
    per_joint = sum_over_components( (cos(curr) - cos(prev))^2 + (sin(curr) - sin(prev))^2 )
    return sum(per_joint * weights)
    """

def compute_total_score(
    all_positions: list[torch.Tensor],
    all_projected_2d: list[torch.Tensor],
    all_local_rots: list[torch.Tensor],
    target_2d_list: list[torch.Tensor],
    visibility_list: list[torch.Tensor],
    sigma: float,
    position_penalty_weight: float,
    rotation_per_joint_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined score across all frames.

    total = sum(heatmap_scores) - pos_w * sum(pos_penalties) - sum(rot_penalties)

    Returns (total_score, details_dict).
    Score is positive = good. Optimizer minimizes loss = -score.
    """
```

### Step 5: Create optimize.py

```python
from pydantic import BaseModel

class OptimizationResult(BaseModel):
    """Result of FK optimization. Pydantic for validation."""
    class Config:
        arbitrary_types_allowed = True

    # These store numpy arrays but Pydantic validates the metadata
    num_frames: int
    num_steps: int
    final_loss: float
    loss_history: list[float]

def run_optimization(
    initial_positions_cam: list[np.ndarray],  # Per-frame (17, 3) from MotionBERT
    target_2d: list[np.ndarray],              # Per-frame (17, 2) pixel targets
    visibility: list[np.ndarray],             # Per-frame (17,) visibility weights
    camera: Camera,
    num_steps: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    """Run FK optimization.

    Returns:
        optimized_3d: list of (17, 3) optimized camera-space positions per frame
        bone_lengths_final: (17,) final shared bone lengths
        loss_history: list of loss values per step

    Algorithm:
    1. For each frame, call positions_to_fk_params() to get initial FK params
    2. Print roundtrip error (CRITICAL: must be < 0.1 cm or FK is broken)
    3. Create learnable parameters:
       - Per-frame: root_pos (3,), root_rot (3,), local_rots (17, 3)
       - Shared: bone_lengths (17,) = median across frames
    4. Convert targets to tensors
    5. Set up Adam optimizer with two param groups:
       - Pose params: LR = cfg.LEARNING_RATE
       - Bone lengths: LR = cfg.BONE_LENGTH_LR
    6. For each step:
       a. Zero grad
       b. For each frame: FK -> 3D positions -> project to 2D
       c. Compute total score (heatmap + penalties)
       d. loss = -score; loss.backward(); optimizer.step()
       e. Clamp bone lengths >= 0.01
       f. Log loss and score details
    7. Extract final positions via FK with optimized params (no_grad)
    """
```

**Key implementation details:**
- The initial FK params come from `positions_to_fk_params()` on the MotionBERT camera-space predictions.
- Bone lengths are initialized as median across frames (robust to per-frame noise).
- The optimizer minimizes `loss = -total_score`.
- Print progress every step initially (only 10 steps), including loss, heatmap score, penalties.
- Clamp bone lengths to `min=0.01` after each step.

### Step 6: Add Pydantic models to models.py

Add to `models.py`:

```python
class OptimizationConfig(BaseModel):
    """Optimization hyperparameters."""
    num_steps: int
    learning_rate: float
    bone_length_lr: float
    sigma: float
    position_penalty_weight: float
    visibility_threshold: float

class ComparisonResult(BaseModel):
    """Comparison of detector vs optimized vs ground truth."""
    det_mpjpe: float | None = None
    det_p_mpjpe: float | None = None
    opt_mpjpe: float | None = None
    opt_p_mpjpe: float | None = None
    det_mpjpe_cm: float | None = None
    opt_mpjpe_cm: float | None = None
    improvement_cm: float | None = None  # positive = improved
```

### Step 7: Update evaluate.py

Add a new function (keep the existing `compute_comparison` for backward compatibility):

```python
def compute_comparison_with_optimization(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
) -> dict[str, Any]:
    """Compare detector baseline, optimized, and ground truth.

    Same as compute_comparison but also computes opt_mpjpe, opt_p_mpjpe,
    opt_per_joint, opt_per_frame_mpjpe, opt_p_per_joint.

    All positions should be in camera-space meters.
    Uses root-relative comparison on 12 eval joints.
    """
```

This function should:
1. Call the existing logic to compute `det_*` metrics
2. Apply the same root-relative + eval-joint slicing to `optimized_3d`
3. Compute `opt_mpjpe`, `opt_p_mpjpe`, `opt_per_joint`, `opt_per_frame_mpjpe`, `opt_p_per_joint`
4. Compute `improvement = det_mpjpe - opt_mpjpe` (positive = optimized is better)

### Step 8: Update graphs.py

Modify the three graph functions to show both detector and optimized results:

1. **`generate_per_joint_error_bar`**: Accept both `det_per_joint` and `opt_per_joint`. Show side-by-side bars (blue = detector, green = optimized).

2. **`generate_per_frame_mpjpe`**: Accept both `det_per_frame` and `opt_per_frame`. Show two line plots (blue = detector, green = optimized).

3. **`generate_aggregate_summary`**: Show grouped bars for each example (detector vs optimized MPJPE). Add improvement column to the printed table.

Keep backward compatibility: if `opt_*` keys are missing, show detector-only (Phase 1 behavior).

### Step 9: Update main.py

In `process_example()`, after step 4 (converting to camera coordinates):

```python
# --- 5. Optimize (Phase 2) ---
from optimize import run_optimization

optimized_3d, bone_lengths_final, loss_history = run_optimization(
    initial_positions_cam=det_cam_positions,
    target_2d=kp_2d,          # H36M (17, 2) from detect_poses
    visibility=visibility,     # H36M (17,) from detect_poses
    camera=camera,
)

# --- 6. Evaluate ---
from evaluate import compute_comparison_with_optimization

metrics = compute_comparison_with_optimization(
    detector_3d=det_cam_positions,
    optimized_3d=optimized_3d,
    gt_3d=gt_cam,
)
```

Update the results JSON to include:
- Both `det_*` and `opt_*` metrics
- Per-frame `optimized_3d` in addition to `detector_3d`
- `bone_lengths_final`
- `loss_history`

Update the graph calls to pass both detector and optimized metrics.

Update the aggregate summary to print both columns and improvement.

Update ExampleResult to include `opt_mpjpe`, `opt_p_mpjpe`, `opt_mpjpe_cm`, `opt_p_mpjpe_cm`, `improvement_cm`.

---

## Integration Points

1. **detect.py returns heatmaps and 2D keypoints** -- Phase 2 uses the H36M-format 2D keypoints as optimization targets (not the raw MPII heatmaps). The `detect_poses()` function already returns `kp_2d` as list of (17, 2) and `visibility` as list of (17,).

2. **Camera object** -- The existing `Camera` class already has `world_to_image_torch()` for differentiable projection. Phase 2 uses this directly.

3. **Skeleton constants** -- Phase 2 adds REST_DIRECTIONS to skeleton.py. The existing PARENTS, NUM_JOINTS, JOINT_NAMES are used as-is.

4. **Evaluation** -- Phase 2 adds `opt_*` metrics alongside existing `det_*` metrics. The eval joint selection (12 joints) stays the same.

## Risks and Edge Cases

### Risk 1: MotionBERT depth estimates are much noisier than MediaPipe
Phase 1 MPJPE is 44.89 cm (vs 18.89 cm for mediapipe-pose). The depth spread is wider (2-8m detector Z vs 2-3m GT). This means:
- FK initialization will have larger bone length variance across frames
- Median bone lengths may be less reliable
- The optimizer needs to correct larger depth errors

**Mitigation**: Start with 10 steps. Check roundtrip error first. If initialization looks reasonable, gradually increase steps. The bone length median should help smooth out per-frame noise.

### Risk 2: FK roundtrip error
The #1 lesson from prior work: if positions_to_fk_params -> forward_kinematics roundtrip error is large, optimization will diverge.

**Mitigation**: Print roundtrip error at the start of every optimization run. If mean error > 0.1 cm, the developer should stop and debug FK before proceeding. Target: < 0.001 cm roundtrip error.

### Risk 3: 17-joint vs 16-joint skeleton
The mediapipe-pose reference code uses 16 joints (no Head). The motionbert-pose skeleton has 17 joints including Head (joint 10) with parent Neck (joint 9). The REST_DIRECTIONS, PARENTS array, and per-joint rotation weights all need to correctly handle this extra joint.

**Mitigation**: Double-check PARENTS indexing. Joint 10 (Head) parent is 9 (Neck). Joint 11 (LShoulder) parent is 8 (Thorax). This matches the existing PARENTS array.

### Risk 4: Visibility/confidence for computed joints
Some H36M joints are computed (not directly detected): Hip (midpoint of RHip/LHip), Spine (midpoint of Pelvis/Thorax), Head (extrapolation). Their visibility/confidence may be lower or zero.

**Mitigation**: The `mpii_to_h36m` function sets confidence for computed joints. The optimizer will weight these joints lower in the heatmap score via the visibility weights. This is fine -- the optimizer focuses on well-detected joints.

### Risk 5: Some examples may regress
The spec allows up to 2 cm regression per example. With only 10 optimization steps, the risk is low -- the optimizer won't move far from initialization.

**Mitigation**: 10 steps is very conservative. If any example regresses by more than 2 cm, reduce step count or increase regularization.

## Validation Checklist

The developer should verify:
1. [ ] FK roundtrip error < 0.001 cm on all 10 examples
2. [ ] Optimization loss decreases monotonically over 10 steps
3. [ ] Optimized MPJPE improves on majority of examples
4. [ ] No example regresses by more than 2 cm
5. [ ] Mean MPJPE is lower after optimization
6. [ ] Results JSON includes both `det_*` and `opt_*` metrics
7. [ ] Graphs show both detector and optimized results
8. [ ] All new functions have type hints
9. [ ] Pydantic models validate optimization config and results
10. [ ] `uv run python main.py` completes without errors on all 10 examples

## After Initial 10-Step Validation

Once the 10-step run is confirmed working:
1. Increase NUM_STEPS to 50, then 100, then 300
2. Verify MPJPE continues to improve (not over-fit)
3. Adjust sigma, penalty weights if needed
4. The old code's best config (SIGMA=50, 300 steps) is a good target

## File Creation/Modification Order

1. `skeleton.py` -- add REST_DIRECTIONS (no dependencies)
2. `config.py` -- add optimization hyperparameters (no dependencies)
3. `models.py` -- add Pydantic models (no dependencies)
4. `fk.py` -- create (depends on skeleton.py)
5. `scoring.py` -- create (standalone torch functions)
6. `optimize.py` -- create (depends on fk.py, scoring.py, camera.py, config.py)
7. `evaluate.py` -- update (add optimization comparison)
8. `graphs.py` -- update (show both detector and optimized)
9. `main.py` -- update (integrate optimization into pipeline)
