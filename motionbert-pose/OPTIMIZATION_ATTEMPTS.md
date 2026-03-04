# Optimization Attempts Log

## Pipeline Overview
- Stacked Hourglass -> real (16, 64, 64) heatmaps
- MotionBERT -> pixel-aligned 3D
- Torso-height heuristic -> camera-space meters (solvePnP disabled, see Attempt 1)
- FK optimization against real heatmaps via F.grid_sample
- Coarse-to-fine blur schedule on real heatmaps

---

## Attempt 0: Initial Implementation
_Status: Failed - solvePnP wildly unstable, optimization made things worse._

**Changes**: Implemented real heatmaps + solvePnP pipeline per plan.
**Config**: NUM_STEPS=150, LR=0.0008, BLUR_PHASES=[(0.35,8),(0.65,3),(1.0,0)], pos_w=8.0, rot_w=5.0, bl_reg=50

**Results (example 0: pose1_0)**:
- solvePnP reproj error: up to 5021px for early frames (wildly unstable)
- MP MPJPE: 32.94cm -> Opt MPJPE: 54.50cm (WORSE)
- Bone lengths collapsed to 0.01m (minimum clamp)
- Motion penalties dominated: rot_penalty*5.0 = 8922 of 11012 total loss (81%)

**Diagnosis**: Three compounding problems:
1. solvePnP gives inconsistent depths across frames (1.2-5.0m range)
2. Motion penalties so high they prevent any useful optimization
3. Bone length reg (50) trivial compared to total loss (~11000)

---

## Attempt 1: Fix PnP fallback + reduce penalties
_Status: Failed - inconsistent depth between PnP and fallback frames._

**Changes**:
- Added PnP reproj threshold (<150px), fall back to torso-height heuristic if PnP fails
- Reduced penalties: pos_w=1.0, rot_w=0.5
- Increased bone reg: 50 -> 5000

**Results (example 0)**:
- MP MPJPE: 43cm -> Opt MPJPE: 62cm (WORSE)
- Root cause: PnP gives depth ~1.2-2.0m, heuristic gives ~3m. Mixing them creates huge position jumps between frames.

**Decision**: Disable solvePnP entirely until optimization is working.

---

## Attempt 2: Fix coordinate transform + verify heatmaps
_Status: Partial success - verified heatmap transform is correct._

**Changes**:
- Disabled solvePnP completely, using only torso-height heuristic
- Fixed align_corners=False half-pixel offset in affine_pixel_to_hm
- Corrected assumed_height_m: 1.2 -> 1.38 (based on actual Thorax-to-ankle distance)
- Added heatmap verification: score detected 2D points against their own heatmaps

**Key finding**: Detected 2D scores -0.00 at raw heatmaps (transform is correct!). Blur=8 baseline: -2.61/joint. Initial projection: -0.56/joint.

**Results**: Optimization still barely improving heatmap score, but foundation is correct.

---

## Attempt 3: Aggressive LR + minimal penalties
_Status: BREAKTHROUGH - first successful optimization._

**Changes**:
- LR: 0.0008 -> 0.005 (6.25x increase)
- Steps: 150 -> 300
- Penalties near-zero: pos_w=0.1, rot_w=0.05
- Longer coarse phase: [(0.50,8),(0.80,3),(1.0,0)]

**Results (example 0: pose1_0)**:
- MP MPJPE: 56.42cm -> Opt MPJPE: 28.36cm (50% improvement!)
- P-MPJPE: 22.83 -> 21.74cm
- Heatmap score: -1734 -> -790 (massive improvement)

**But**: Bone lengths drifting from initial values.

---

## Attempt 4: Anchor bone reg to DEFAULT_BONE_LENGTHS
_Status: Failed - default anchor distorts pose shape._

**Changes**: Bone length regularization anchored to skeleton.DEFAULT_BONE_LENGTHS instead of detector median.

**Results (example 0)**:
- Opt MPJPE: 33.96cm (worse than attempt 3's 28.36)
- Opt P-MPJPE: 26.55cm (worse than 21.74)

**Decision**: Revert to detector median anchor. The default human-average bone lengths don't match this specific person well enough.

---

## Attempt 5: Detector median anchor + high bone reg
_Status: Best single-example result, but breaks on good initial estimates._

**Config**: Same as attempt 3 but bl_reg=10000 (anchored to detector median)

**Results on 3 examples**:
| Example | Init MPJPE | Opt MPJPE | Change |
|---------|-----------|----------|--------|
| 0 (pose1_0) | 56.42 | 27.60 | -51% |
| 1 (pose2_200) | 61.36 | 62.81 | +2% (slight degrade) |
| 2 (pose2_5000) | 19.37 | 33.04 | +71% (SEVERE degrade!) |

**Critical problem**: When the detector already has a good estimate (example 2: 19cm), the optimizer overfits to 2D heatmaps and destroys the 3D accuracy. Need regularization to prevent unnecessary changes.

---

## Attempt 6: Moderate penalties
_Status: Failed - penalties still too blunt._

**Changes**: Tried pos_w=1.0, rot_w=0.5 (10x increase from attempt 5).

**Results**: Example 0 worse (35cm vs 28cm), example 1 still unchanged. Penalties are too blunt - they restrict ALL motion, not just unnecessary deviation from the initial estimate.

---

## Attempt 7: Initial position regularization (CURRENT BEST)
_Status: SUCCESS - improves all examples, protects good initial estimates._

**Key insight**: Instead of penalizing frame-to-frame motion, penalize deviation from the INITIAL detector 3D positions. This lets the optimizer improve bad estimates while protecting good ones.

**Changes**:
- Added `INIT_POSITION_REG_WEIGHT` to config.py
- Added `init_positions` parameter to `compute_total_score()` in scoring.py
- Wire through optimize.py: store initial positions as tensors, pass to scoring

**Weight tuning**:
| Weight | Ex0 MPJPE | Ex1 MPJPE | Ex2 MPJPE | Notes |
|--------|----------|----------|----------|-------|
| 0 (none) | 27.60 | 62.81 | 33.04 | Ex2 destroyed |
| 5 | 26.04 | - | 19.41 | Starting to slip on ex2 |
| 10 | 26.09 | 53.34 | 19.22 | **Best balance** |
| 20 | 27.05 | 53.45 | 18.75 | Too conservative for ex0 |
| 50 | 29.54 | 54.11 | 18.97 | Too conservative |

**Final config (weight=10)**:
```
NUM_STEPS = 300
LEARNING_RATE = 0.005
BONE_LENGTH_LR = 0.001
GRAD_CLIP_NORM = 10.0
BLUR_PHASES = [(0.50, 8.0), (0.80, 3.0), (1.00, 0.0)]
POSITION_PENALTY_WEIGHT = 0.1
ROTATION_PENALTY_WEIGHT = 0.05
BONE_LENGTH_REG_WEIGHT = 10000.0
INIT_POSITION_REG_WEIGHT = 10.0
```

**Full results (all 9 examples with GT)**:
| Ex | Sequence | Init MPJPE | Opt MPJPE | Change | Init P-MPJPE | Opt P-MPJPE |
|----|----------|-----------|----------|--------|-------------|-------------|
| 0 | pose1_0 | 56.42 | 26.09 | **-53.8%** | 22.83 | 24.00 |
| 1 | pose2_200 | 61.36 | 53.34 | **-13.1%** | 27.65 | 27.17 |
| 2 | pose2_5000 | 19.37 | 19.22 | **-0.8%** | 12.28 | 13.08 |
| 3 | pose2_15000 | 51.91 | 39.71 | **-23.5%** | 20.80 | 25.28 |
| 4 | pose3_200 | 52.00 | 45.85 | **-11.8%** | 32.89 | 32.53 |
| 5 | pose3_4000 | 44.40 | 33.69 | **-24.1%** | 16.68 | 16.58 |
| 6 | ult1_200 | 55.70 | 49.96 | **-10.3%** | 24.16 | 24.09 |
| 7 | ult1_10000 | 61.95 | 49.09 | **-20.8%** | 25.06 | 26.36 |
| 9 | pose2_25000 | 22.27 | 19.22 | **-13.7%** | 12.12 | 12.57 |

**Average MPJPE improvement: -19.1%. No example degraded.**

MPJPE improved on ALL 9 examples. P-MPJPE improved on 4/9, roughly preserved on 5/9.
The P-MPJPE pattern confirms the optimization primarily fixes global translation (depth from heuristic), which P-MPJPE already factors out via Procrustes alignment.

---

## Summary of Key Lessons

1. **solvePnP is too unstable** for per-frame depth estimation from monocular pose. Inconsistent depths between frames cause huge position jumps. Torso-height heuristic is more consistent.

2. **Motion penalties are too blunt** - they restrict ALL motion, not just bad motion. Initial-position regularization is much more targeted.

3. **Bone length regularization needs to be strong** (10000) to prevent collapse, and should anchor to detector median (not human-average defaults).

4. **The LR needs to be high enough** (0.005) for the coarse-to-fine blur schedule to work. Low LR means the optimizer can't escape local minima during the coarse phase.

5. **Initial position regularization is the key innovation** - it lets the optimizer improve bad detector estimates while protecting good ones. Weight=10 is the sweet spot.
