# Architect Plan P1-00: Parameter Tuning

## Goal Summary

The FK optimization currently provides only +0.34 cm mean improvement over raw MotionBERT (32.99 -> 32.65 cm MPJPE). The hypothesis is that penalty terms (position, rotation, anchor) dominate the gradient signal and drown out the heatmap score. This plan covers: (1) adding a 2D detection-projected MPJPE metric that measures how well predictions match the 2D Stacked Hourglass detections (not GT projections), (2) creating a parameter sweep script, (3) defining specific experiments for Phase 1.1 coarse-to-fine parameter tuning, and (4) preparing for Phase 1.2 heatmap blur.

## Key Analysis

**Why optimization barely helps:** The `real_heatmap_score` returns `log(heatmap_value) * visibility` per joint. Heatmap values are 0-1, so log values are negative (e.g., log(0.5) = -0.69, log(0.01) = -4.6). With ~15 visible joints across ~150 frames, total heatmap score might be around -500 to -2000. Meanwhile, `POSITION_PENALTY_WEIGHT=50.0` and `ROTATION_PENALTY_SCALAR=10.0` with per-joint weights up to 30.0 create large penalty gradients. The `INIT_ANCHOR_WEIGHT=5.0` further anchors positions to MotionBERT, preventing the optimizer from moving joints toward heatmap peaks.

**Current 2D reprojection metric bug:** The existing `det_2d_mpjpe` and `opt_2d_mpjpe` in `evaluate.py` measure reprojection error against *GT 2D projections* (`camera.world_to_image(gt_arr[i])`). The spec asks for error against *2D SH detections* -- this is a different and more informative metric for understanding optimization behavior.

## Files to Modify

1. **`motionbert-pose/evaluate.py`** -- Add `det_2d_det_mpjpe` and `opt_2d_det_mpjpe` metrics that measure 2D error against SH detections (not GT projections).

2. **`motionbert-pose/config.py`** -- No changes for now. The sweep script will override parameters.

## Files to Create

1. **`motionbert-pose/sweep.py`** -- Parameter sweep script that runs a single example with different configs and reports all metrics in a table.

## Step-by-Step Instructions

### Step 1: Add 2D-detection-projected MPJPE to evaluate.py

Add a new function `reprojection_error_vs_detections` and integrate it into `compute_comparison_with_optimization`.

**Add function** after the existing `root_relative` function:

```python
def reprojection_error_vs_detections(
    positions_3d: list[np.ndarray],
    detections_2d: list[np.ndarray],
    visibility: list[np.ndarray],
    camera: "Camera",
    visibility_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute 2D reprojection error against 2D SH detections.

    Projects 3D predictions to 2D via the camera model, then measures
    pixel distance to the 2D Stacked Hourglass keypoints (NOT GT projections).

    Args:
        positions_3d: Per-frame (17, 3) camera-space positions.
        detections_2d: Per-frame (17, 2) SH 2D keypoints in pixels.
        visibility: Per-frame (17,) visibility scores.
        camera: Camera for 3D->2D projection.
        visibility_threshold: Only count joints above this threshold.

    Returns:
        Dict with 'mean_px' (mean pixel error across visible joints and frames),
        'per_frame_px' (list of per-frame mean pixel errors).
    """
    per_frame_errors: list[float] = []
    for i in range(len(positions_3d)):
        proj_2d = camera.world_to_image(positions_3d[i])  # (17, 2)
        diffs = np.linalg.norm(proj_2d - detections_2d[i], axis=-1)  # (17,)
        mask = visibility[i] >= visibility_threshold
        if mask.sum() > 0:
            per_frame_errors.append(float(diffs[mask].mean()))
        else:
            per_frame_errors.append(0.0)
    return {
        "mean_px": float(np.mean(per_frame_errors)) if per_frame_errors else 0.0,
        "per_frame_px": per_frame_errors,
    }
```

**Modify `compute_comparison_with_optimization` signature** to accept additional optional parameters:

```python
def compute_comparison_with_optimization(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    camera: Camera | None = None,
    detections_2d: list[np.ndarray] | None = None,
    visibility: list[np.ndarray] | None = None,
) -> dict[str, Any]:
```

**Add at the end of `compute_comparison_with_optimization`**, after the existing 2D reprojection block:

```python
    # --- 2D reprojection error vs 2D DETECTIONS (not GT) ---
    if camera is not None and detections_2d is not None and visibility is not None:
        det_vs_det = reprojection_error_vs_detections(
            detector_3d, detections_2d, visibility, camera,
        )
        opt_vs_det = reprojection_error_vs_detections(
            optimized_3d, detections_2d, visibility, camera,
        )
        results["det_2d_det_mpjpe_px"] = det_vs_det["mean_px"]
        results["opt_2d_det_mpjpe_px"] = opt_vs_det["mean_px"]
        results["det_2d_det_per_frame_px"] = det_vs_det["per_frame_px"]
        results["opt_2d_det_per_frame_px"] = opt_vs_det["per_frame_px"]
```

### Step 2: Thread 2D detections and visibility through callers

**In `main.py`**, update the call to `compute_comparison_with_optimization` (around line 234):

```python
    metrics: dict[str, Any] = compute_comparison_with_optimization(
        detector_3d=det_cam_positions,
        optimized_3d=optimized_3d,
        gt_3d=gt_cam,
        camera=camera,
        detections_2d=improved_target_2d,
        visibility=visibility,
    )
```

Also add printing for the new metric after the existing 2D MPJPE print (around line 260):

```python
    if "det_2d_det_mpjpe_px" in metrics:
        print(f"    Det 2D-vs-Det: {metrics['det_2d_det_mpjpe_px']:.1f} px")
        print(f"    Opt 2D-vs-Det: {metrics['opt_2d_det_mpjpe_px']:.1f} px")
```

**In `test_single.py`**, similarly update the call (around line 151):

```python
    metrics: dict[str, Any] = compute_comparison_with_optimization(
        det_cam_positions, optimized_3d, gt_cam,
        camera=camera,
        detections_2d=improved_target_2d,
        visibility=visibility,
    )
```

And add printing in the RESULTS section:

```python
    if "det_2d_det_mpjpe_px" in metrics:
        print(f"  Det 2D-vs-Det: {metrics['det_2d_det_mpjpe_px']:.1f} px")
        print(f"  Opt 2D-vs-Det: {metrics['opt_2d_det_mpjpe_px']:.1f} px")
```

### Step 3: Create sweep.py

Create `motionbert-pose/sweep.py`. This script:

1. Runs detection once (expensive) and caches results
2. Re-runs optimization with different parameter configs
3. Reports all metrics in a table

The script structure:

```python
"""Parameter sweep for FK optimization hyperparameters.

Runs detection once, then re-optimizes with different parameter configs.
Reports MPJPE, P-MPJPE, MPJVE, and 2D-vs-detection reprojection error.
"""
import json
import os
import sys
import time
from typing import Any

import numpy as np

import config as cfg
from camera import Camera
from detect import detect_poses, motionbert_to_camera_space
from evaluate import compute_comparison_with_optimization
from optimize import run_optimization
from panoptic import (
    extract_video_frames,
    get_sequence_dir,
    get_video_path,
    load_calibration,
    load_ground_truth_sequence,
    world_to_camera,
)
```

**Core data class for a parameter config:**

```python
from dataclasses import dataclass, field

@dataclass
class SweepConfig:
    """One parameter configuration to test."""
    name: str
    num_steps: int = 20
    position_penalty_weight: float = 50.0
    rotation_penalty_scalar: float = 10.0
    init_anchor_weight: float = 5.0
    all_joints_smooth_weight: float = 0.0
    sigma_schedule: list[tuple[float, float]] = field(default_factory=lambda: [(1.0, 80.0)])
    heatmap_blur_sigma: float = 0.0  # Additional Gaussian blur on SH heatmaps (Phase 1.2)
```

**Detection caching function** -- run detection and camera-space conversion once, return all needed data:

```python
def load_example(example_idx: int = 0) -> dict[str, Any]:
    """Load and detect poses for one example. Returns cached data dict."""
    seq_name, camera_name, start_frame, num_frames, person_idx = cfg.EXAMPLES[example_idx]
    # ... (same as test_single.py steps 1-5: calibration, frame extraction,
    #      detection, camera-space conversion, GT loading)
    # Return dict with keys:
    #   camera, det_cam_positions, kp_2d, visibility, heatmaps, affine,
    #   improved_target_2d, gt_cam, frames_rgb, frame_indices, name,
    #   fx, fy, cx, cy
```

**Sweep function** -- takes cached data and a SweepConfig, runs optimization, returns metrics:

```python
def run_sweep_config(data: dict[str, Any], config: SweepConfig) -> dict[str, Any]:
    """Run optimization with one parameter config, return metrics."""
    # Temporarily override cfg values
    orig_pos_weight = cfg.POSITION_PENALTY_WEIGHT
    orig_rot_scalar = cfg.ROTATION_PENALTY_SCALAR
    orig_anchor = cfg.INIT_ANCHOR_WEIGHT
    orig_smooth = cfg.ALL_JOINTS_SMOOTH_WEIGHT
    orig_sigma_schedule = cfg.SIGMA_SCHEDULE

    try:
        cfg.POSITION_PENALTY_WEIGHT = config.position_penalty_weight
        cfg.INIT_ANCHOR_WEIGHT = config.init_anchor_weight
        cfg.ALL_JOINTS_SMOOTH_WEIGHT = config.all_joints_smooth_weight
        cfg.SIGMA_SCHEDULE = config.sigma_schedule

        # Rebuild per-joint rotation weights with new scalar
        cfg.ROTATION_PENALTY_SCALAR = config.rotation_penalty_scalar
        cfg.ROTATION_PENALTY_PER_JOINT = np.array([
            config.rotation_penalty_scalar * m for m in
            [3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1]
        ], dtype=np.float64)

        # Optionally blur heatmaps
        heatmaps = data["heatmaps"]
        if config.heatmap_blur_sigma > 0:
            import scipy.ndimage
            heatmaps = [
                np.stack([
                    scipy.ndimage.gaussian_filter(hm[c], sigma=config.heatmap_blur_sigma)
                    for c in range(hm.shape[0])
                ]) for hm in heatmaps
            ]

        optimized_3d, bone_lengths_final, loss_history = run_optimization(
            initial_positions_cam=data["det_cam_positions"],
            target_2d=data["improved_target_2d"],
            visibility=data["visibility"],
            camera=data["camera"],
            num_steps=config.num_steps,
            heatmaps=heatmaps,
            affine=data["affine"],
        )

        metrics = compute_comparison_with_optimization(
            detector_3d=data["det_cam_positions"],
            optimized_3d=optimized_3d,
            gt_3d=data["gt_cam"],
            camera=data["camera"],
            detections_2d=data["improved_target_2d"],
            visibility=data["visibility"],
        )
        metrics["loss_history"] = loss_history
        return metrics

    finally:
        # Restore original config
        cfg.POSITION_PENALTY_WEIGHT = orig_pos_weight
        cfg.ROTATION_PENALTY_SCALAR = orig_rot_scalar
        cfg.INIT_ANCHOR_WEIGHT = orig_anchor
        cfg.ALL_JOINTS_SMOOTH_WEIGHT = orig_smooth
        cfg.SIGMA_SCHEDULE = orig_sigma_schedule
        cfg.ROTATION_PENALTY_PER_JOINT = np.array([
            orig_rot_scalar * m for m in
            [3.0, 1.0, 0.5, 0.2, 1.0, 0.5, 0.2, 1.0, 1.0, 0.5, 0.5, 0.5, 0.3, 0.1, 0.5, 0.3, 0.1]
        ], dtype=np.float64)
```

**Main function** with the sweep configs defined:

```python
def main() -> None:
    example_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    print("Loading example data (detection + GT)...")
    data = load_example(example_idx)
    print(f"Example: {data['name']}, {len(data['det_cam_positions'])} frames\n")

    # Define sweep configurations -- Phase 1.1
    configs: list[SweepConfig] = get_phase1_1_configs()

    results: list[tuple[str, dict[str, Any]]] = []
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(configs)}] {config.name}")
        print(f"  pos_w={config.position_penalty_weight}, rot_s={config.rotation_penalty_scalar}, "
              f"anchor={config.init_anchor_weight}, smooth={config.all_joints_smooth_weight}")
        print(f"{'='*60}")

        t0 = time.time()
        metrics = run_sweep_config(data, config)
        elapsed = time.time() - t0

        results.append((config.name, metrics))
        print(f"  Time: {elapsed:.1f}s")

    # Print summary table
    print(f"\n\n{'='*100}")
    print(f"  SWEEP RESULTS -- {data['name']}")
    print(f"{'='*100}")
    header = f"{'Config':<40} {'Det MPJPE':>10} {'Opt MPJPE':>10} {'Improv':>8} {'Opt P-MPJPE':>12} {'Det 2D-Det':>10} {'Opt 2D-Det':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results:
        det_mpjpe = m.get('det_mpjpe', 0) * 100
        opt_mpjpe = m.get('opt_mpjpe', 0) * 100
        improv = m.get('improvement', 0) * 100
        opt_p = m.get('opt_p_mpjpe', 0) * 100
        det_2d = m.get('det_2d_det_mpjpe_px', 0)
        opt_2d = m.get('opt_2d_det_mpjpe_px', 0)
        print(f"{name:<40} {det_mpjpe:>10.2f} {opt_mpjpe:>10.2f} {improv:>+8.2f} {opt_p:>12.2f} {det_2d:>10.1f} {opt_2d:>10.1f}")

    # Save results to JSON
    out_dir = os.path.join(cfg.TRAINING_RUNS_DIR, "sweep_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"sweep_{data['name']}.json")
    save_data = []
    for name, m in results:
        save_data.append({
            "config": name,
            "det_mpjpe_cm": m.get('det_mpjpe', 0) * 100,
            "opt_mpjpe_cm": m.get('opt_mpjpe', 0) * 100,
            "improvement_cm": m.get('improvement', 0) * 100,
            "opt_p_mpjpe_cm": m.get('opt_p_mpjpe', 0) * 100,
            "det_2d_det_mpjpe_px": m.get('det_2d_det_mpjpe_px', 0),
            "opt_2d_det_mpjpe_px": m.get('opt_2d_det_mpjpe_px', 0),
            "opt_mpjve_cm": m.get('opt_mpjve', 0) * 100 if 'opt_mpjve' in m else None,
        })
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {out_path}")
```

### Step 4: Define Phase 1.1 Sweep Configs

Create a `get_phase1_1_configs()` function in `sweep.py` that returns the configs. The strategy is to test each parameter independently at orders of magnitude up/down, then test combinations of the best settings.

**Baseline** (current params):
```python
SweepConfig(name="baseline",
    position_penalty_weight=50.0, rotation_penalty_scalar=10.0,
    init_anchor_weight=5.0)
```

**Position penalty weight sweep** (currently 50.0):
```python
SweepConfig(name="pos_w=0", position_penalty_weight=0.0, rotation_penalty_scalar=10.0, init_anchor_weight=5.0)
SweepConfig(name="pos_w=0.5", position_penalty_weight=0.5, ...)
SweepConfig(name="pos_w=5", position_penalty_weight=5.0, ...)
SweepConfig(name="pos_w=50 (baseline)", position_penalty_weight=50.0, ...)
SweepConfig(name="pos_w=500", position_penalty_weight=500.0, ...)
SweepConfig(name="pos_w=5000", position_penalty_weight=5000.0, ...)
```

**Rotation penalty scalar sweep** (currently 10.0):
```python
SweepConfig(name="rot_s=0", rotation_penalty_scalar=0.0, ...)
SweepConfig(name="rot_s=0.1", rotation_penalty_scalar=0.1, ...)
SweepConfig(name="rot_s=1", rotation_penalty_scalar=1.0, ...)
SweepConfig(name="rot_s=10 (baseline)", rotation_penalty_scalar=10.0, ...)
SweepConfig(name="rot_s=100", rotation_penalty_scalar=100.0, ...)
SweepConfig(name="rot_s=1000", rotation_penalty_scalar=1000.0, ...)
```

**Init anchor weight sweep** (currently 5.0):
```python
SweepConfig(name="anchor=0", init_anchor_weight=0.0, ...)
SweepConfig(name="anchor=0.05", init_anchor_weight=0.05, ...)
SweepConfig(name="anchor=0.5", init_anchor_weight=0.5, ...)
SweepConfig(name="anchor=5 (baseline)", init_anchor_weight=5.0, ...)
SweepConfig(name="anchor=50", init_anchor_weight=50.0, ...)
SweepConfig(name="anchor=500", init_anchor_weight=500.0, ...)
```

**All penalties off** (heatmap only):
```python
SweepConfig(name="heatmap_only", position_penalty_weight=0.0, rotation_penalty_scalar=0.0, init_anchor_weight=0.0)
```

**All penalties very low** (let heatmap dominate):
```python
SweepConfig(name="all_low", position_penalty_weight=0.5, rotation_penalty_scalar=0.1, init_anchor_weight=0.05)
```

**More steps to see if convergence matters:**
```python
SweepConfig(name="baseline_100steps", num_steps=100, position_penalty_weight=50.0, rotation_penalty_scalar=10.0, init_anchor_weight=5.0)
SweepConfig(name="all_low_100steps", num_steps=100, position_penalty_weight=0.5, rotation_penalty_scalar=0.1, init_anchor_weight=0.05)
```

Total: ~22 configs. At 20 steps each, this should run in a few minutes per example. The 100-step configs will take proportionally longer.

### Step 5: Phase 1.2 Preparation (Heatmap Blur)

After Phase 1.1 identifies good parameter ranges, add blur configs to the sweep:

```python
SweepConfig(name="best_blur1", ..., heatmap_blur_sigma=1.0)
SweepConfig(name="best_blur2", ..., heatmap_blur_sigma=2.0)
SweepConfig(name="best_blur4", ..., heatmap_blur_sigma=4.0)
SweepConfig(name="best_blur8", ..., heatmap_blur_sigma=8.0)
```

The `heatmap_blur_sigma` is in 64x64 heatmap coordinates. A sigma of 2.0 at 64x64 corresponds to ~8px at 256x256 crop resolution. This should widen the gradient basin without destroying spatial precision.

The blur is applied via `scipy.ndimage.gaussian_filter` on each channel of the (16, 64, 64) heatmap tensor before passing to optimization. This is done once per config, not per optimization step.

**Important:** The blur is NOT applied to the sigma parameter of the analytical Gaussian fallback (for Hip and Spine joints). Those joints already use the sigma schedule.

## Integration Points

- `sweep.py` imports from the same modules as `test_single.py` and `main.py` -- no new dependencies except `scipy.ndimage` for Phase 1.2 blur (scipy is already a project dependency)
- The new `detections_2d` and `visibility` parameters to `compute_comparison_with_optimization` are optional with default `None`, so existing callers continue to work without changes (they just won't get the new metric until updated)
- Config overrides in `sweep.py` use a try/finally block to restore original values, so running the sweep doesn't corrupt the module-level config state

## Risks and Edge Cases

1. **Heatmap-only mode might explode**: With all penalties at 0, the optimizer has no regularization. Bone lengths could go to the 0.01 clamp, rotations could diverge. The FK roundtrip may produce wildly wrong poses. This is expected -- we want to see the failure mode to understand the heatmap signal strength.

2. **Config mutation is not thread-safe**: The `cfg` module is mutated globally. The sweep must run configs sequentially, not in parallel. This is fine since each optimization run uses all CPU/GPU resources anyway.

3. **Memory**: With 150 frames of (16, 64, 64) heatmaps, each example uses ~9.4 MB for heatmaps. Blurred copies add the same. This is negligible.

4. **The 2D-vs-detection metric will be identical for detector baseline across all configs** since `det_cam_positions` doesn't change. Only `opt_2d_det_mpjpe_px` will vary. This is correct and expected -- the baseline column serves as a reference.

5. **Sigma schedule vs penalty weights**: The sigma schedule (currently constant 80.0) affects the analytical Gaussian fallback for Hip and Spine joints, as well as the gradient basin width. Phase 1.1 does not sweep sigma -- it sweeps penalty weights only. Phase 1.2 addresses the gradient basin via heatmap blur instead.
