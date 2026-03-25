# Architect Plan: Phase 2b, Iteration 0 -- Scale-Z-Invariant MPJPE

## Goal Summary

Create a new evaluation metric called **Scale-Z-Invariant MPJPE (SZI-MPJPE)**. The current P-MPJPE gives each frame invariance to scale, rotation, and translation (via Procrustes alignment), which is too generous. The new metric only corrects for the single degree of freedom that a monocular camera truly cannot determine: overall scale/depth. We find a single optimal scale factor `s` across ALL keypoints and ALL frames that minimizes the total squared error, apply it to the predictions, then compute standard MPJPE. We also need unit tests and a 3D skeleton visualization comparing before-scaling, after-scaling, and ground truth from a non-camera viewpoint.

## Math Verification

We minimize: `L(s) = sum_over_all_frames_and_joints ||s * pred - gt||^2`

Expanding: `L(s) = s^2 * sum(pred . pred) - 2s * sum(pred . gt) + sum(gt . gt)`

Setting `dL/ds = 0`: `2s * sum(pred . pred) - 2 * sum(pred . gt) = 0`

Therefore: **`s = sum(pred . gt) / sum(pred . pred)`**

Note: The spec document has a typo with `sum(gt . gt)` in the denominator. The correct denominator is `sum(pred . pred)`.

Here, `pred` and `gt` are root-relative 3D positions (hip subtracted), and the dot products are element-wise multiply then sum across all coordinates, joints, and frames.

## Files to Modify

### 1. `motionbert-pose/evaluate.py`
- Add `scale_z_invariant_mpjpe()` function.
- Add `scale_z_invariant_mpjpe_per_joint()` function.
- Integrate into `compute_comparison()` and `compute_comparison_with_optimization()`.

### 2. `motionbert-pose/main.py`
- Print the new SZI-MPJPE metric alongside existing metrics.

### 3. `motionbert-pose/graphs.py`
- Add the SZI-MPJPE to the aggregate summary bar chart (new chart, same pattern as P-MPJPE chart).

## Files to Create

### 1. `motionbert-pose/test_szi_mpjpe.py`
- Unit tests for the scale computation and SZI-MPJPE metric.
- 3D skeleton visualization (before scaling, after scaling, ground truth).

## Step-by-Step Instructions

### Step 1: Add `scale_z_invariant_mpjpe` to `evaluate.py`

Add the following function after the existing `p_mpjpe` function (after line 191):

```python
def optimal_scale(predicted: np.ndarray, target: np.ndarray) -> float:
    """Find the optimal scale s that minimizes ||s * predicted - target||^2.

    Computed across ALL keypoints and ALL frames simultaneously.

    Args:
        predicted: (F, J, 3) root-relative predicted positions.
        target: (F, J, 3) root-relative ground truth positions.

    Returns:
        Optimal scale factor s.
    """
    numerator: float = float(np.sum(predicted * target))
    denominator: float = float(np.sum(predicted * predicted))
    if denominator < 1e-12:
        return 1.0
    return numerator / denominator


def szi_mpjpe(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Scale-Z-Invariant MPJPE.

    Finds optimal global scale, applies it, then computes MPJPE.

    Args:
        predicted: (F, J, 3) root-relative positions.
        target: (F, J, 3) root-relative positions.

    Returns:
        Tuple of (szi_mpjpe_value, optimal_scale_factor).
    """
    s: float = optimal_scale(predicted, target)
    scaled: np.ndarray = s * predicted
    return mpjpe(scaled, target), s


def szi_mpjpe_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-joint SZI-MPJPE (scale determined globally, error computed per joint).

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        (J,) mean error per joint after optimal scaling.
    """
    s: float = optimal_scale(predicted, target)
    scaled: np.ndarray = s * predicted
    return mpjpe_per_joint(scaled, target)
```

Key design decisions:
- `optimal_scale` is a standalone function so it can be tested independently.
- `szi_mpjpe` returns both the metric value AND the scale factor (useful for visualization and debugging).
- The scale is computed over ALL eval joints and ALL frames simultaneously -- a single scalar for the entire trajectory.
- The function operates on root-relative positions (hip already subtracted before calling).

### Step 2: Integrate SZI-MPJPE into `compute_comparison()`

In `compute_comparison()` (around line 235, after the `det_mpjpe` and `det_p_mpjpe` lines), add:

```python
det_szi_val, det_szi_scale = szi_mpjpe(det_eval, gt_eval)
results["det_szi_mpjpe"] = det_szi_val
results["det_szi_scale"] = det_szi_scale
results["det_szi_per_joint"] = szi_mpjpe_per_joint(det_eval, gt_eval).tolist()
```

### Step 3: Integrate SZI-MPJPE into `compute_comparison_with_optimization()`

After the `opt_mpjpe` and `opt_p_mpjpe` lines (around line 315), add:

```python
opt_szi_val, opt_szi_scale = szi_mpjpe(opt_eval, gt_eval)
results["opt_szi_mpjpe"] = opt_szi_val
results["opt_szi_scale"] = opt_szi_scale
results["opt_szi_per_joint"] = szi_mpjpe_per_joint(opt_eval, gt_eval).tolist()
```

Also add no-ankles variants after the existing no-ankles block (around line 339):

```python
results["det_szi_mpjpe_no_ankles"] = szi_mpjpe(det_eval_na, gt_eval_na)[0]
results["opt_szi_mpjpe_no_ankles"] = szi_mpjpe(opt_eval_na, gt_eval_na)[0]
```

### Step 4: Print SZI-MPJPE in `main.py`

In `process_example()`, after the P-MPJPE print lines (around line 232), add:

```python
if "det_szi_mpjpe" in metrics:
    print(f"    Det SZI-MPJPE: {metrics['det_szi_mpjpe']*100:.2f} cm (scale={metrics['det_szi_scale']:.4f})")
if "opt_szi_mpjpe" in metrics:
    print(f"    Opt SZI-MPJPE: {metrics['opt_szi_mpjpe']*100:.2f} cm (scale={metrics['opt_szi_scale']:.4f})")
```

### Step 5: Add SZI-MPJPE to aggregate summary in `graphs.py`

In `generate_aggregate_summary()`, after the P-MPJPE bar chart block (around line 703), add a new SZI-MPJPE bar chart following the exact same pattern as the P-MPJPE chart:

1. Gather `det_szi_mpjpe_cm` and `opt_szi_mpjpe_cm` from `with_gt` metrics.
2. Check that `"det_szi_mpjpe"` exists in all metrics before plotting.
3. Create a bar chart with title "Scale-Z-Invariant MPJPE Across Examples".
4. Save to `aggregate_szi_mpjpe.png`.

The code should follow the identical structure of the P-MPJPE chart (lines 686-703), just replacing the data source keys and labels.

### Step 6: Create `test_szi_mpjpe.py` with unit tests and 3D visualization

Create `motionbert-pose/test_szi_mpjpe.py`. This is a standalone script (not pytest) that the tester can run with `uv run python test_szi_mpjpe.py`.

**Unit tests to include:**

1. **Identity test**: `pred == gt` -> scale should be 1.0, SZI-MPJPE should be 0.0.

2. **Known scale test**: `pred = 2.0 * gt` -> scale should be 0.5, SZI-MPJPE should be 0.0 (perfect recovery).

3. **Known scale + noise test**: `pred = 2.0 * gt + small_noise` -> scale should be approximately 0.5, SZI-MPJPE should be small but nonzero.

4. **Asymmetric scale test**: `pred = 0.5 * gt` -> scale should be 2.0, SZI-MPJPE should be 0.0.

5. **SZI-MPJPE <= MPJPE test**: For random data, SZI-MPJPE should always be <= MPJPE (since optimal scaling can only help or be neutral).

6. **SZI-MPJPE >= P-MPJPE test**: SZI-MPJPE should be >= P-MPJPE (since Procrustes has more degrees of freedom).

For each test, print PASS/FAIL and a description. Use assertions with tolerances (e.g., `np.isclose` with `atol=1e-6`).

**3D Visualization:**

After the unit tests, generate a 3D skeleton visualization PNG. The approach:

1. Create synthetic data: take a realistic skeleton shape (hardcode a single frame from the H36M skeleton -- e.g., T-pose-ish with some variation), then create `gt` from it, and `pred = 1.5 * gt` (scaled version).
2. Compute the optimal scale and create `scaled_pred = s * pred`.
3. Create a matplotlib figure with a single 3D subplot (use `mpl_toolkits.mplot3d`).
4. Draw three skeletons side by side (offset them on the X axis so they don't overlap):
   - Green: `pred` (before scaling), offset X by -1.0
   - Red: `scaled_pred` (after scaling), offset X by 0.0
   - Blue: `gt` (ground truth), offset X by +1.0
5. Use `BONES` from `skeleton.py` to draw the bone connections.
6. Set the view angle to something NOT along the camera axis (e.g., `ax.view_init(elev=20, azim=45)`).
7. Add a legend and title: "SZI-MPJPE: Before Scaling / After Scaling / Ground Truth".
8. Save to `motionbert-pose/test_output/szi_mpjpe_visualization.png`.

The developer should also generate a version with a **real example** (not synthetic):
1. Load one prediction JSON from a previous run (if available in `training_runs/`).
2. Extract root-relative eval-joint positions for pred and gt.
3. Compute optimal scale, apply it.
4. Pick one representative frame (e.g., frame 0 or the middle frame).
5. Render the same 3-skeleton plot (pred, scaled_pred, gt) with side-by-side offset.
6. Save to `motionbert-pose/test_output/szi_mpjpe_real_example.png`.

If no previous run data exists, skip this step (the synthetic test is sufficient).

## Integration Points

- `evaluate.py` is the only file where computation happens. All other files just consume the results dict.
- The new keys (`det_szi_mpjpe`, `opt_szi_mpjpe`, `det_szi_scale`, `opt_szi_scale`, `*_szi_per_joint`, `*_szi_mpjpe_no_ankles`) follow the existing naming pattern.
- `main.py` prints the metrics. `graphs.py` charts them. Neither computes them.
- The test script is standalone and does not affect the pipeline.

## Risks and Edge Cases

1. **Zero predictions**: If all predicted positions are zero (degenerate case), `sum(pred . pred)` = 0 and division by zero occurs. The `optimal_scale` function guards against this with `if denominator < 1e-12: return 1.0`.

2. **Negative scale**: If predictions are systematically in the wrong direction (e.g., mirrored), the optimal scale could be negative. This is unlikely for root-relative poses (which are centered at the hip) but possible. A negative scale would flip the skeleton, which is not physically meaningful. However, mathematically it minimizes the error, so we should allow it. If the tester observes negative scales, flag it as a concern.

3. **Scale near zero**: If predictions are very large relative to GT, the optimal scale will be near zero, collapsing the skeleton to a point. This would give an MPJPE approximately equal to the mean GT distance from root. This is correct behavior -- it means the predictions were bad.

4. **Root-relative is important**: The scale must be applied to root-relative positions (hip subtracted). If applied to absolute camera-space positions, the scale would also affect the distance from the camera, which conflates translation with scale. The current code already makes positions root-relative before computing metrics, so this is handled.

5. **Eval joints vs all joints**: The scale is computed over the 12 eval joints only (same set used for MPJPE and P-MPJPE). This is consistent with existing metrics.
