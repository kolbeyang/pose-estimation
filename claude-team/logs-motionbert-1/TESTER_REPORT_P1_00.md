# TESTER_REPORT_P1_00: Verify Spirit of the Project

## Tests Run

### Test 1: Smoke Test -- `test_single.py` runs to completion

**Command:** `cd motionbert-pose && uv run python test_single.py`

**Result: PASS**

The pipeline ran to completion on the first CMU Panoptic example (171204_pose1_sample, 100 frames). Key output:
- Det MPJPE: 30.98 cm, Opt MPJPE: 30.44 cm (improvement: +0.54 cm)
- Det P-MPJPE: 28.57 cm, Opt P-MPJPE: 28.42 cm
- No ankles: Det 14.14 cm, Opt 13.85 cm
- 100/100 frames with ground truth
- No errors or tracebacks

### Test 2: Code Review -- Real heatmaps are used in scoring

**Result: PASS**

Complete data flow traced and verified:

1. **`config.py`** line 100: `USE_REAL_HEATMAPS: bool = True` -- flag is enabled.

2. **`detect.py`** `run_hourglass()`: Heatmaps come from `output[-1].cpu().numpy()[0]` (line 263) -- the final stack of the Stacked Hourglass network. Flip augmentation is applied and averaged (line 278). These are genuine neural network outputs, shape (16, 64, 64).

3. **`detect.py`** `detect_poses()`: Returns `all_heatmaps` and `affine` as elements of the 7-tuple return value (line 1103).

4. **`test_single.py`** line 139-146: Passes `heatmaps=heatmaps` and `affine=affine` to `run_optimization()`.

5. **`optimize.py`** `run_optimization()` line 134: Checks `cfg.USE_REAL_HEATMAPS` and converts heatmaps to torch tensors. Line 140: Prints "Using real Stacked Hourglass heatmaps for scoring". Line 209: Passes `use_real_heatmaps=cfg.USE_REAL_HEATMAPS` to `compute_total_score()`.

6. **`scoring.py`** `compute_total_score()` line 274-278: Computes `_use_real = use_real_heatmaps and heatmaps_list is not None and affine is not None`. When True, line 282 calls `real_heatmap_score()` instead of `heatmap_score()`.

7. **`scoring.py`** `real_heatmap_score()` lines 64-141: Uses `F.grid_sample(hm, grid, mode="bilinear", ...)` to sample the actual Stacked Hourglass heatmap at projected 2D locations. Takes `torch.log(torch.clamp(value, min=eps))` of the sampled value. Falls back to analytical Gaussian only for Hip (0) and Spine (7) which have no dedicated MPII heatmap.

**Conclusion:** When `USE_REAL_HEATMAPS=True` (which is the default), the optimizer genuinely samples from real Stacked Hourglass heatmaps for 15 of 17 joints. The 2 fallback joints (Hip, Spine) are synthetic midpoints that have no corresponding heatmap channel, so analytical Gaussian is the correct approach for those.

### Test 3: Code Review -- Heatmaps come from Stacked Hourglass (not generated)

**Result: PASS**

In `detect.py`, `run_hourglass()`:
- Line 262: `output = model(inp)` -- forward pass through the HG8 model.
- Line 263: `heatmaps = output[-1].cpu().numpy()[0]` -- takes the final stack output directly.
- Lines 266-278: Flip augmentation: flips the input image, runs the model again, flips the output heatmaps, averages with the original. This is a standard augmentation technique that preserves the real heatmap character.
- Line 279: `all_heatmaps.append(heatmaps)` -- raw network output is stored.

No Gaussian generation functions are applied to the heatmaps. They are genuine Stacked Hourglass outputs.

### Test 4: Code Review -- H36M-to-MPII heatmap mapping

**Result: PASS (with minor note)**

Cross-referenced `H36M_TO_MPII_HEATMAP` in `scoring.py` against `mpii_to_h36m()` in `skeleton.py`:

| H36M idx | H36M name    | MPII heatmap idx | MPII name      | Correct? |
|----------|-------------|------------------|----------------|----------|
| 0        | Hip         | None             | midpoint       | Yes      |
| 1        | RHip        | 2                | RHip           | Yes      |
| 2        | RKnee       | 1                | RKnee          | Yes      |
| 3        | RAnkle      | 0                | RAnkle         | Yes      |
| 4        | LHip        | 3                | LHip           | Yes      |
| 5        | LKnee       | 4                | LKnee          | Yes      |
| 6        | LAnkle      | 5                | LAnkle         | Yes      |
| 7        | Spine       | None             | midpoint       | Yes      |
| 8        | Thorax      | 7                | Thorax         | Yes*     |
| 9        | Neck        | 8                | Neck           | Yes*     |
| 10       | Head        | 9                | Head           | Yes*     |
| 11       | LShoulder   | 13               | LShoulder      | Yes      |
| 12       | LElbow      | 14               | LElbow         | Yes      |
| 13       | LWrist      | 15               | LWrist         | Yes      |
| 14       | RShoulder   | 12               | RShoulder      | Yes      |
| 15       | RElbow      | 11               | RElbow         | Yes      |
| 16       | RWrist      | 10               | RWrist         | Yes      |

*Note: There is a naming inconsistency in `mpii_to_h36m()` where `h36m[8] = keypoints_mpii[8]` (comment: "Thorax = MPII Neck") but `H36M_TO_MPII_HEATMAP[8] = 7` (MPII Thorax). The heatmap mapping follows the standard academic convention (H36M Thorax = MPII Thorax = index 7), which is correct for sampling the appropriate heatmap. The `mpii_to_h36m()` 2D keypoint mapping is a separate concern used only for MotionBERT input, not for heatmap sampling. This does not cause a bug in practice because for H36M joint 8, the heatmap path is taken (not the fallback), and the correct heatmap channel (MPII 7, Thorax) is sampled.

### Test 5: Code Review -- Other capstone report constraints

**Result: PASS**

All core claims from the capstone report are respected:

1. **Rigid skeleton parameterization (joint angles, not raw coordinates):** `optimize.py` optimizes `param_root_pos`, `param_root_rot`, and `param_local_rots` (all axis-angle), with `forward_kinematics()` converting these to 3D positions. Joint angles are the optimization variables.

2. **Bone lengths shared across frames:** `optimize.py` line 111: `median_bone_lengths = np.median(np.array(all_bone_lengths), axis=0)` -- a single `param_bone_lengths` tensor is shared across all frames (line 112-116).

3. **Forward kinematics is differentiable:** `fk.py` uses PyTorch operations (Rodrigues formula, matrix multiplication). `positions_3d` is computed via `forward_kinematics()` with gradient tracking.

4. **Temporal smoothness penalties:** `scoring.py` includes `motion_penalty_position()` (root position jumps), `motion_penalty_rotation()` (rotation jumps using chord distance), and `motion_penalty_all_joints()`. These are applied in `compute_total_score()` for consecutive frame pairs.

5. **Coarse-to-fine schedule:** `config.py` line 44: `SIGMA_SCHEDULE` is defined (currently just a single phase at sigma=80). `optimize.py` line 17-23: `_get_sigma()` implements the schedule lookup. The infrastructure exists; it's currently set to a single coarse phase.

6. **Evaluation uses MPJPE and P-MPJPE:** Confirmed in the test output: both `Det MPJPE` and `Det P-MPJPE` are reported, along with per-joint breakdowns.

7. **No imports from sibling directories:** Grep confirmed zero matches for imports from `mediapipe-pose/`, `rtmw-pose/`, `toy-arm/`, or `comparison/`.

### Test 6: Runtime verification -- log output confirms real heatmaps

**Result: PASS**

From the test output:
```
    Using real Stacked Hourglass heatmaps for scoring
```

This line is printed when `heatmaps is not None and cfg.USE_REAL_HEATMAPS` is True (optimize.py line 134). Confirms that real heatmaps are active at runtime.

## Bugs Found

None. No blocking bugs were found in this phase.

## Code Review Findings

### Finding 1: MPII naming inconsistency (MINOR, non-blocking)

**Description:** The `mpii_to_h36m()` function maps `h36m[8] = keypoints_mpii[8]` with the comment "Thorax = MPII Neck", while `H36M_TO_MPII_HEATMAP[8] = 7` (MPII Thorax). These refer to different MPII joints for the same H36M joint. This does not cause a runtime bug because the heatmap sampling and 2D keypoint paths serve different purposes, but the naming inconsistency could confuse future developers.

**Severity:** Minor. Does not affect correctness.

**Suggested fix:** Add a clarifying comment in `scoring.py` noting that the heatmap mapping follows the standard academic MPII-H36M correspondence, which differs from the `mpii_to_h36m()` keypoint mapping used for MotionBERT input.

### Finding 2: Single-phase sigma schedule (NOTED, not a bug)

**Description:** `SIGMA_SCHEDULE` is `[(1.0, 80.0)]` -- a constant coarse sigma for all steps. The infrastructure for multi-phase coarse-to-fine exists but is not being used. The capstone report mentions coarse-to-fine scheduling as a feature. This is acceptable at the current 20-step optimization level but should be revisited when step count increases.

**Severity:** Not a bug. The schedule infrastructure is present; the current single-phase setting is a conscious configuration choice.

### Finding 3: High ankle MPJPE (NOTED, known issue)

**Description:** RAnkle and LAnkle MPJPE are ~115 cm, dominating the overall 30.98 cm MPJPE. Without ankles, MPJPE drops to ~14 cm. This is a known depth ambiguity issue documented in the developer report.

**Severity:** Not a bug in this phase's scope. The "no ankles" metric provides a better picture of upper-body accuracy.

## Verdict

**YES** -- the current implementation meets Phase 1's definition of done ("Verify that we are respecting the spirit of the project").

Specifically:
1. Real Stacked Hourglass heatmaps ARE being used in the FK optimization scoring function via differentiable bilinear sampling (`F.grid_sample`). The data flow from Stacked Hourglass network output through to the scoring function is complete and verified.
2. All other capstone report constraints (rigid skeleton, shared bone lengths, differentiable FK, temporal smoothness, MPJPE evaluation) are satisfied.
3. The pipeline runs to completion on a real CMU Panoptic example with no errors.
4. No imports from sibling directories.
