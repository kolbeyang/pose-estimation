# ARCHITECT_PLAN_P2_00: More In-Depth Outputs

## Goal Summary

The motionbert-pose pipeline currently produces only two graphs per example (per-joint error bar, per-frame MPJPE) and has no summary graph, no 3D visualization, no trajectory plots, no velocity error graphs, no loss curve, and no bone length comparison. The mediapipe-pose pipeline produces all of these. This iteration brings motionbert-pose to feature-parity with mediapipe-pose for graphs and visualization, adapting naming conventions (Detector/Optimized instead of MediaPipe/Optimised) and the 12-eval-joint subset that motionbert uses.

## Gap Analysis

### Graphs in mediapipe-pose but MISSING in motionbert-pose:
1. **Trajectory graphs** (`generate_trajectory_graphs`) -- per-joint, per-coordinate (X/Y/Z) plots showing Detector vs Optimized vs GT over time
2. **Loss curve** (`generate_loss_curve`) -- optimization loss over steps
3. **Per-joint MPJVE bar chart** (`generate_per_joint_mpjve_bar`) -- velocity error per joint
4. **Per-frame MPJVE** (`generate_per_frame_mpjve`) -- velocity error over time
5. **Bone lengths graph** (`generate_bone_lengths_graph`) -- GT vs Detector vs Optimized bone lengths
6. **Summary grid** (`generate_summary`) -- 4x4 grid combining key plots + metrics in title
7. **Aggregate MPJVE** -- the aggregate summary is missing MPJVE bar chart

### Evaluation metrics MISSING in motionbert-pose:
1. MPJVE (velocity error) -- `mpjve`, `mpjve_per_joint`, `mpjve_per_frame`
2. 2D reprojection error (needs camera passed to `compute_comparison_with_optimization`)
3. Bone length extraction from GT/Detector in evaluate.py

### Visualization MISSING:
1. `visualize.py` -- 3D matplotlib skeleton viewer (loads prediction JSON, animates)

### Data MISSING from prediction JSON:
1. `bone_lengths_final` -- already saved (good)
2. `loss_history` -- already saved (good)
3. `detector_3d` key -- already present as `detector_3d` (maps to mediapipe's `mediapipe_3d`)

## Files to Modify

1. **`motionbert-pose/evaluate.py`** -- Add `mpjve`, `mpjve_per_joint`, `mpjve_per_frame` functions. Add bone length extraction. Add 2D reprojection error. Add camera parameter to `compute_comparison_with_optimization`. Compute all the missing metrics.

2. **`motionbert-pose/graphs.py`** -- Add `generate_trajectory_graphs`, `generate_loss_curve`, `generate_per_joint_mpjve_bar`, `generate_per_frame_mpjve`, `generate_bone_lengths_graph`, `generate_summary`. Update `generate_aggregate_summary` with MPJVE. All graphs should use 17 joints for trajectory/bone plots and 12 eval joints for error bars (matching current convention).

3. **`motionbert-pose/main.py`** -- Pass camera to evaluation. Call all new graph functions. Call summary. Pass bone_lengths to evaluate.

## Files to Create

1. **`motionbert-pose/visualize.py`** -- 3D matplotlib skeleton viewer, nearly identical to mediapipe-pose/visualize.py but loading motionbert JSON format (key `detector_3d` instead of `mediapipe_3d`).

## Step-by-Step Instructions

### Step 1: Add velocity error functions to `evaluate.py`

Add these functions (copy from mediapipe-pose/evaluate.py, adapt types):

```python
def mpjve(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean Per-Joint Velocity Error.
    Args: predicted (F, J, 3), target (F, J, 3). Returns scalar."""
    pred_vel = np.diff(predicted, axis=0)
    tgt_vel = np.diff(target, axis=0)
    return float(np.mean(np.linalg.norm(pred_vel - tgt_vel, axis=-1)))

def mpjve_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Returns (J,) velocity error per joint."""
    pred_vel = np.diff(predicted, axis=0)
    tgt_vel = np.diff(target, axis=0)
    return np.mean(np.linalg.norm(pred_vel - tgt_vel, axis=-1), axis=0)

def mpjve_per_frame(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    """Returns F-1 per-frame velocity errors."""
    pred_vel = np.diff(predicted, axis=0)
    tgt_vel = np.diff(target, axis=0)
    return [float(np.mean(np.linalg.norm(pred_vel[i] - tgt_vel[i], axis=-1)))
            for i in range(pred_vel.shape[0])]
```

### Step 2: Expand `compute_comparison_with_optimization` in `evaluate.py`

Add an optional `camera: Camera | None = None` parameter. Add the following computations after the existing metrics:

1. **Bone lengths** -- For each joint j with parent p, compute mean bone length across GT frames and detector frames. Store as `gt_bone_lengths` (17,) and `det_bone_lengths` (17,) in results.

2. **MPJVE** -- If >= 2 GT frames exist, compute `det_mpjve`, `opt_mpjve`, `det_mpjve_per_joint`, `opt_mpjve_per_joint`, `det_per_frame_mpjve`, `opt_per_frame_mpjve`. These should all operate on the eval-joint-sliced arrays (consistent with existing MPJPE).

3. **2D reprojection error** -- If camera is not None, compute `det_per_frame_2d_mpjpe`, `opt_per_frame_2d_mpjpe`, `det_2d_mpjpe`, `opt_2d_mpjpe`. Project all 17 joints (not just eval joints) to 2D, compute mean pixel error per frame.

Also add the PARENTS import from skeleton.py (already imported: `from skeleton import NUM_JOINTS, JOINT_NAMES, EVAL_JOINTS, NUM_EVAL_JOINTS, EVAL_JOINT_NAMES`). Add `PARENTS` to that import.

### Step 3: Add all missing graph functions to `graphs.py`

Add these imports at the top:
```python
from matplotlib.lines import Line2D
from skeleton import JOINT_NAMES, NUM_JOINTS, EVAL_JOINT_NAMES, NUM_EVAL_JOINTS, PARENTS, JOINT_GROUP, GROUP_COLORS_RGB
```

Add functions (each following the mediapipe-pose pattern but using "Detector"/"Optimized" labels and green/red colors):

#### 3a. `generate_trajectory_graphs(detector_3d, optimized_3d, gt_3d, output_dir)`
- Args: lists of (17,3) arrays + list of (17,3)|None
- One subplot per joint per coordinate (X/Y/Z). 17 joints x 3 coords = 51 PNG files.
- Lines: Detector=green solid, Optimized=red solid, GT=blue dashed
- Save as `{JOINT_NAMES[j]}_{cname}.png`

#### 3b. `generate_loss_curve(loss_history, output_dir)`
- Single blue line plot. Save as `loss_curve.png`.

#### 3c. `generate_per_joint_mpjve_bar(det_per_joint, opt_per_joint, output_dir)`
- Same pattern as existing `generate_per_joint_error_bar` but for velocity error.
- Uses EVAL_JOINT_NAMES (12 joints), values in cm/frame.
- Save as `per_joint_mpjve.png`.

#### 3d. `generate_per_frame_mpjve(det_per_frame, opt_per_frame, output_dir)`
- Line plot, two series. Save as `per_frame_mpjve.png`.

#### 3e. `generate_bone_lengths_graph(bone_lengths, output_dir, gt_bone_lengths=None, det_bone_lengths=None)`
- Bar chart of 16 bones (joints 1-16). Triple bars if GT+Detector provided.
- Uses JOINT_NAMES for labels, values in meters.
- Save as `bone_lengths.png`.

#### 3f. `generate_summary(detector_3d, optimized_3d, gt_3d, loss_history, metrics, bone_lengths, output_dir, title="")`
- 4x4 grid (same layout as mediapipe-pose):
  - Row 0-1, Cols 0-2: 6 key joint trajectories (Hip, Thorax, Neck/Nose, Head, LWrist, RAnkle)
    - Each subplot: 3 coordinates (X=red, Y=green, Z=blue), 3 line styles (solid=Detector, dashed=Optimized, dotted=GT)
  - Row 0, Col 3: Legend
  - Row 1, Col 3: Per-frame MPJPE (if GT)
  - Row 2, Col 0: Loss curve
  - Row 2, Col 1: Per-joint MPJPE bar (12 eval joints)
  - Row 2, Col 2: Per-joint P-MPJPE bar (12 eval joints)
  - Row 2, Col 3: Bone lengths (16 bones)
  - Row 3, Col 0: Per-frame MPJVE
  - Row 3, Col 1: Per-joint MPJVE bar (12 eval joints)
  - Row 3, Col 2: Per-frame 2D MPJPE (if available)
  - Row 3, Col 3: empty (axis off)
- Title includes metric summary: Det MPJPE, Opt MPJPE, Det P-MPJPE, Opt P-MPJPE, velocity errors
- Save as `summary.png`.

**Critical adaptation:** The mediapipe summary uses `NUM_JOINTS` (16 in mediapipe) for bar charts. Here we use `NUM_EVAL_JOINTS` (12) for error bars and `NUM_JOINTS` (17) for trajectories/bone lengths. The MPJPE/P-MPJPE bar charts in the summary should use eval joint names. Metric keys in motionbert use `det_` prefix (not `mp_`).

#### 3g. Update `generate_aggregate_summary`
- Add MPJVE bar chart (same pattern as mediapipe's aggregate).
- Add MPJVE columns to the printed table.

### Step 4: Update `main.py` to use new graphs and metrics

In `process_example()`:

1. Pass `camera` to `compute_comparison_with_optimization`:
```python
metrics = compute_comparison_with_optimization(
    detector_3d=det_cam_positions,
    optimized_3d=optimized_3d,
    gt_3d=gt_cam,
    camera=camera,  # NEW
)
```

2. Add calls to new graph functions after existing graph calls:
```python
generate_trajectory_graphs(
    det_cam_positions, optimized_3d, gt_cam, example_graph_dir,
)
generate_loss_curve(loss_history, example_graph_dir)
generate_bone_lengths_graph(
    bone_lengths_final, example_graph_dir,
    gt_bone_lengths=metrics.get("gt_bone_lengths"),
    det_bone_lengths=metrics.get("det_bone_lengths"),
)
if "det_mpjve_per_joint" in metrics:
    generate_per_joint_mpjve_bar(
        metrics["det_mpjve_per_joint"], metrics.get("opt_mpjve_per_joint"),
        example_graph_dir,
    )
if "det_per_frame_mpjve" in metrics:
    generate_per_frame_mpjve(
        metrics["det_per_frame_mpjve"], metrics.get("opt_per_frame_mpjve"),
        example_graph_dir,
    )
generate_summary(
    det_cam_positions, optimized_3d, gt_cam,
    loss_history, metrics, bone_lengths_final,
    example_graph_dir, title=name,
)
```

3. Add new imports:
```python
from graphs import (
    generate_aggregate_summary,
    generate_bone_lengths_graph,
    generate_loss_curve,
    generate_per_frame_mpjpe,
    generate_per_frame_mpjve,
    generate_per_joint_error_bar,
    generate_per_joint_mpjve_bar,
    generate_summary,
    generate_trajectory_graphs,
)
```

4. Print MPJVE metrics:
```python
if "det_mpjve" in metrics:
    print(f"    Det MPJVE:   {metrics['det_mpjve']*100:.2f} cm/f")
    print(f"    Opt MPJVE:   {metrics['opt_mpjve']*100:.2f} cm/f")
if "det_2d_mpjpe" in metrics:
    print(f"    Det 2D MPJPE: {metrics['det_2d_mpjpe']:.1f} px")
    print(f"    Opt 2D MPJPE: {metrics['opt_2d_mpjpe']:.1f} px")
```

### Step 5: Create `visualize.py`

Create `motionbert-pose/visualize.py` closely following mediapipe-pose/visualize.py. Key differences:

- The JSON key is `detector_3d` (not `mediapipe_3d`).
- Labels: "Detector" (green), "Optimized" (red), "Ground Truth" (blue).
- Import `BONES` from `skeleton` (same module name).
- Same interactive features: scroll to zoom, drag to rotate, auto-cycling frames.
- Usage: `python visualize.py path/to/predictions.json [--fps 5]`

```python
"""Standalone 3D visualisation: load predictions JSON and animate.

Usage:
    python visualize.py training_runs/.../predictions/example.json

Shows Detector (green) vs Optimized (red) skeletons,
with optional Ground Truth (blue).
"""
import argparse, json
import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from skeleton import BONES

def _draw_camera(ax, radius=0.15): ...  # identical to mediapipe
def _draw_skeleton(ax, positions, color, label, alpha=0.8): ...  # identical
def visualize_prediction_file(json_path, fps=5.0):
    # Load JSON, iterate frames, draw skeletons
    # Key difference: frame["detector_3d"] instead of frame["mediapipe_3d"]
    ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(...)
    ...
```

### Step 6: Ensure prediction JSON has all needed data

The current JSON already saves `bone_lengths_final`, `loss_history`, `detector_3d`, `optimized_3d`, `ground_truth_3d`, `visibility`, `detector_2d`. Verify the JSON also stores these new metric keys by confirming the `metrics` dict is saved to JSON (it is -- line 265 of main.py saves `{k: v for k, v in metrics.items() if k != "name"}`). The new MPJVE and 2D error metrics will flow through automatically.

No changes needed to the JSON format.

## Integration Points

- `evaluate.py` gains a `camera` parameter on `compute_comparison_with_optimization`. This is backward-compatible (defaults to None). Both `main.py` and `test_single.py` should pass camera.
- `graphs.py` gains 6 new functions. `main.py` calls them all. The aggregate summary function gains MPJVE support.
- `visualize.py` is standalone (run from command line with a JSON path).
- All new metric keys (`det_mpjve`, `opt_mpjve`, `gt_bone_lengths`, etc.) are generated by evaluate.py and consumed by graphs.py. They also flow into the JSON through the existing metrics serialization.

## Risks and Edge Cases

1. **Eval joints vs all joints:** Trajectory graphs use all 17 joints. Error bars use 12 eval joints. Bone lengths use 16 bones (joints 1-16). The summary must be careful about which joint set each subplot uses.

2. **MPJVE with < 2 frames:** Already handled -- MPJVE functions need `np.diff` which needs >= 2 frames. The conditional in evaluate.py guards this.

3. **Camera not passed:** If camera is None, 2D reprojection metrics are skipped. The summary handles this gracefully (empty subplot).

4. **bone_lengths_final as np.ndarray:** When loaded from JSON, this becomes a list. The bone lengths graph should handle both `list[float]` and `np.ndarray`. Use `np.asarray()` to be safe.

5. **matplotlib backend:** `visualize.py` needs `macosx` backend for interactive display. `graphs.py` uses `Agg` for headless rendering. These are separate files so no conflict.

6. **Per-joint MPJVE uses eval joints (12):** The mediapipe version uses all 16 joints. We should use eval joints (12) for consistency with the rest of motionbert-pose's evaluation. The `mpjve_per_joint` and `mpjve_per_frame` in evaluate.py should operate on the eval-joint-sliced arrays.

## Testing Plan

1. Run `test_single.py` (or `main.py` with one example) and verify all graphs are generated.
2. Check that the following files exist in the output graph directory: `summary.png`, `loss_curve.png`, `bone_lengths.png`, `per_joint_mpjve.png`, `per_frame_mpjve.png`, plus trajectory PNGs.
3. Run `visualize.py` on the prediction JSON. It should open a window (tester can kill it after confirming no crash).
