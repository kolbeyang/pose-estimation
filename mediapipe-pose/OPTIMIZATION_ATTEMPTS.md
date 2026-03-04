# Optimization Attempts Log

## Problem Statement
The FK optimization pipeline starts with excellent MediaPipe+solvePnP initialization (~14-19 cm MPJPE)
but the optimizer consistently makes results **worse** (up to 59 cm). The loss converges monotonically
but the skeleton diverges from ground truth.

## Baseline (before any tuning)
- Config: SIGMA=75, POS_WEIGHT=8.0, ROT_WEIGHT=5.0, BL_REG=50.0, GRAD_CLIP=5.0, LR=0.0008, separate bone LR=0.002
- Rotation penalty: uniform across all joints
- Result on 171204_pose1_sample: MP MPJPE=14.58 cm, Opt MPJPE=59.38 cm (4x worse)
- Full 9-example mean: MP=18.89 cm, Opt=48.28 cm (optimizer destroys results)

---

## Attempt 1: Per-joint rotation weights + remove grad clip + single LR + sigma 50

**Changes:** SIGMA 75→50, removed gradient clip, single LR=0.0008, per-joint rotation weights (trunk=1.0, mid=0.3-0.5, extremities=0.1-0.2)

**Result:** MP=14.58 cm → Opt=53.57 cm. Still catastrophically worse.

**Analysis:** The heatmap improved massively (-21167 → -7934) but 3D got 4x worse. This is depth ambiguity exploitation — the optimizer finds configs that project well to 2D but are wrong in 3D.

---

## Attempt 2: Very conservative settings

**Changes:** LR=0.00005, POS_W=50, ROT_W=50, STEPS=30 (barely move anything)

**Result:** MP=14.58 cm → Opt=64.91 cm. Even with 30 steps and tiny LR, still 4x worse!

**Key insight:** Heatmap barely changed (1.5%), rotations barely changed (0.2%), yet MPJPE went from 14.58 → 64.91 cm. **This proved the optimizer wasn't the problem — the FK initialization itself was broken.**

---

## ROOT CAUSE FOUND: FK Decomposition Bug

Traced the issue to `positions_to_fk_params()` → `forward_kinematics()` roundtrip.
**Roundtrip error: 48 cm mean, 87 cm max!** The FK decomposition was destroying the input.

### The Bug (3 parts):

1. **REST_DIRECTIONS used Y-up convention, camera coords are Y-down.** All bone directions
   were anti-parallel to their rest directions (dot products ≈ -1.0), forcing 180° root rotation.

2. **`_rotation_matrix_to_axis_angle` was degenerate at 180°.** Near angle=π, the antisymmetric
   part of R (used to extract the axis) vanishes. The function returned zero-vector, losing the
   rotation entirely.

3. **`_rotation_between_vectors` returned `-I` (det=-1, a REFLECTION) for anti-parallel vectors.**
   Not a proper rotation matrix.

### The Fix:

- **Flipped REST_DIRECTIONS** to camera convention: Y-down, person facing camera.
  RHip=(-1,0,0), Spine=(0,-1,0), legs=(0,1,0), etc.

- **Updated root rotation construction** in `positions_to_fk_params`:
  `R_root = column_stack([-right, -up, -forward])` (was `[right, up, forward]`)
  to correctly map rest axes to camera-space body axes.

- **Fixed `_rotation_between_vectors`** for anti-parallel case: proper 180° rotation
  (R = -I + 2nn^T) instead of reflection (-I).

- **Fixed `_rotation_matrix_to_axis_angle`** for near-180°: extract axis from symmetric
  part (R+I)/2 when antisymmetric part vanishes.

### Result: Perfect roundtrip (0.0000 cm error), root rotation now ~11° (was 180°), max local rotation ~63° (was 180°).

---

## Attempt 3: FK-fixed + conservative settings

**Config:** SIGMA=50, POS_W=50, ROT_W=50, LR=0.00005, STEPS=30

**Result:** MP=14.58 cm → Opt=14.31 cm. **First time optimizer doesn't destroy results!**

---

## Attempt 4: FK-fixed + normal settings

**Config:** SIGMA=50, POS_W=8, ROT_W=5, LR=0.0005, STEPS=150

**Result:** MP=14.58 cm → Opt=13.64 cm. **Optimizer improving!**

---

## Attempt 5: More aggressive

**Config:** SIGMA=50, POS_W=8, ROT_W=2, LR=0.001, STEPS=300

**Result:** MP=14.58 cm → Opt=12.81 cm. **~2 cm improvement.**

---

## Attempt 6: Low rotation penalty

**Config:** SIGMA=50, POS_W=5, ROT_W=0.5, LR=0.001, STEPS=500

**Result:** MP=14.58 cm → Opt=12.93 cm. Marginal — diminishing returns.

---

## Attempt 7: Sigma 30 + loose bone lengths

**Config:** SIGMA=30, POS_W=5, ROT_W=1, BL_REG=10, LR=0.001, STEPS=300

**Result:** MP=14.58 cm → Opt=12.76 cm. Slightly better.

---

## Attempt 8: Minimal constraints

**Config:** SIGMA=30, POS_W=3, ROT_W=0.2, BL_REG=10, LR=0.001, STEPS=500

**Result:** MP=14.58 cm → Opt=12.58 cm. Best single-example result.

---

## Full Pipeline Run (Attempt 7 settings)

**Config:** SIGMA=30, POS_W=5, ROT_W=1, BL_REG=10, LR=0.001, STEPS=300

| Example                    | MP MPJPE (cm) | Opt MPJPE (cm) | Improvement |
|----------------------------|---------------|----------------|-------------|
| 171204_pose1_sample_0      | 14.58         | 12.76          | +1.82       |
| 171204_pose2_200           | 18.33         | 15.61          | +2.71       |
| 171204_pose2_5000          | 14.53         | 15.53          | -1.00       |
| 171204_pose2_15000         | 13.73         | 13.11          | +0.62       |
| 171204_pose3_200           | 19.11         | 15.72          | +3.39       |
| 171204_pose3_4000          | 15.88         | 14.76          | +1.12       |
| 160422_ultimatum1_200      | 34.14         | 30.80          | +3.33       |
| 160422_ultimatum1_10000    | 25.94         | 23.75          | +2.18       |
| 171204_pose2_25000         | 13.76         | 14.10          | -0.34       |
| **MEAN**                   | **18.89**     | **17.35**      | **+1.54**   |

**7/9 examples improved, 2/9 slightly regressed (by <1 cm).**

### Before vs After FK Fix:
- Before: MP=18.89 cm → Opt=48.28 cm (**29 cm worse**, optimizer destroyed results)
- After:  MP=18.89 cm → Opt=17.35 cm (**1.5 cm better**, optimizer actually helps)

---

## Summary of Lessons

1. **The real bug was NOT hyperparameters** — it was the FK decomposition using a Y-up rest pose
   in Y-down camera coordinates, causing 180° rotations that are numerically degenerate.

2. **Once FK roundtrip works**, the optimizer does what it should: improve 2D alignment while
   maintaining kinematic constraints, resulting in ~1-3 cm improvement.

3. **Remaining limits** (~12-13 cm floor on best examples) are likely from:
   - Depth ambiguity (2D heatmaps can't constrain Z)
   - MediaPipe detection noise
   - Joint mapping differences (MediaPipe→H36M vs CMU COCO19→H36M)

4. **Hyperparameter sensitivity** is now low — the optimizer works across a wide range of
   settings (ROT_W from 0.2 to 5, SIGMA from 30 to 50, etc.) without catastrophic failure.

## Current Best Config
```
SIGMA = 30.0
LEARNING_RATE = 0.001
NUM_STEPS = 300
POSITION_PENALTY_WEIGHT = 5.0
ROTATION_PENALTY_WEIGHT = 1.0
BONE_LENGTH_REG_WEIGHT = 10.0
Per-joint rotation weights: trunk=1.0, mid=0.3-0.5, extremities=0.1-0.2
```

Run: `training_runs/mediapipe-run-20260303-165456/`
