# Architect Plan: Phase 1, Iteration 0

## Goal Summary

Create a line graph showing 4 limb lengths (R shoulder->elbow, R elbow->wrist, L shoulder->elbow, L elbow->wrist) over time for 3 sources (ground truth, MotionBERT raw 3D, optimized 3D), totaling 12 lines. Ground truth in blue, MotionBERT raw in green, optimized in red. The graph must be generated per-example and included in the summary view. Sanity check the output.

## Joint Index Reference

From `skeleton.py`:
- LShoulder = 10, LElbow = 11, LWrist = 12
- RShoulder = 13, RElbow = 14, RWrist = 15

The 4 limb segments to track:
1. R Shoulder->Elbow: `norm(pos[14] - pos[13])`
2. R Elbow->Wrist: `norm(pos[15] - pos[14])`
3. L Shoulder->Elbow: `norm(pos[11] - pos[10])`
4. L Elbow->Wrist: `norm(pos[12] - pos[11])`

## Data Flow

In `main.py::process_example()`:
- **Ground truth 3D**: `gt_cam` -- list of `(16, 3)` numpy arrays (camera-space meters), or `None` for missing frames.
- **MotionBERT raw 3D**: `det_cam_positions` -- list of `(16, 3)` numpy arrays (camera-space meters). This is the raw MotionBERT output converted to camera space.
- **Optimized 3D**: `optimized_3d` -- list of `(16, 3)` numpy arrays (camera-space meters). This is the FK-optimized result.

All three are already available at the point where graphs are generated (after step 6, evaluate).

## Files to Modify

### 1. `motionbert-pose/graphs.py`

Add one new function: `generate_limb_length_graph`.

### 2. `motionbert-pose/main.py`

Add a call to `generate_limb_length_graph` in the graph generation section of `process_example()`.

## Files to Create

None.

## Step-by-Step Instructions

### Step 1: Add `generate_limb_length_graph` to `graphs.py`

Add the following function after the existing `generate_bone_lengths_graph` function (around line 270):

**Function signature:**
```python
def generate_limb_length_graph(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    output_dir: str,
) -> None:
```

**Implementation logic:**

1. Define the 4 limb segments as a list of tuples:
   ```python
   LIMB_SEGMENTS = [
       (13, 14, "R Shoulder-Elbow"),
       (14, 15, "R Elbow-Wrist"),
       (10, 11, "L Shoulder-Elbow"),
       (11, 12, "L Elbow-Wrist"),
   ]
   ```

2. For each source (detector, optimized, gt), compute the Euclidean distance between the two joints for each frame. For GT, use `np.nan` when `gt_3d[i]` is `None`.

   Helper pattern:
   ```python
   def _limb_length(positions: np.ndarray, j1: int, j2: int) -> float:
       return float(np.linalg.norm(positions[j2] - positions[j1]))
   ```

3. Create a single figure with `figsize=(12, 6)`.

4. Plot 12 lines total:
   - Ground truth lines: **blue**, solid, one per limb segment. Label: `"GT: {limb_name}"`.
   - Detector lines: **green**, solid, one per limb segment. Label: `"Det: {limb_name}"`.
   - Optimized lines: **red**, solid, one per limb segment. Label: `"Opt: {limb_name}"`.

5. Use different line styles (solid, dashed, dotted, dashdot) per limb segment within each color to distinguish the 4 limbs. Specifically:
   - R Shoulder-Elbow: solid (`-`)
   - R Elbow-Wrist: dashed (`--`)
   - L Shoulder-Elbow: dotted (`:`)
   - L Elbow-Wrist: dashdot (`-.`)

   This way, color encodes source and linestyle encodes which limb.

6. X-axis: frame index. Y-axis: limb length in meters. Title: `"Arm Limb Lengths Over Time"`.

7. Place legend outside the plot (right side) or use `fontsize=7` to keep it readable with 12 entries. Use `bbox_to_anchor=(1.05, 1)` and `loc='upper left'` to place legend outside.

8. Save to `os.path.join(output_dir, "limb_lengths.png")` using the existing `_save()` helper.

### Step 2: Import and call from `main.py`

1. In the import block at the top of `main.py`, add `generate_limb_length_graph` to the imports from `graphs`.

2. In `process_example()`, after the existing graph generation calls (around line 325, after `generate_per_frame_mpjve`), add:
   ```python
   generate_limb_length_graph(
       det_cam_positions, optimized_3d, gt_cam, example_graph_dir,
   )
   ```

   Note: use the **absolute** 3D positions, NOT root-relative. Limb lengths are the same regardless of root subtraction, but using absolute positions is simpler and avoids any confusion.

### Step 3: Add to the summary grid (optional but recommended)

In the `generate_summary` function, the slot at `axes[3, 3]` is currently unused (`axes[3, 3].axis("off")`). Replace it with a miniature version of the limb length graph:

1. On `axes[3, 3]`, plot the same 12 lines but with `linewidth=0.8` and `fontsize=5` for readability in the small subplot.
2. Set title to `"Arm Limb Lengths (m)"` with `fontsize=9`.
3. Add a compact legend with `fontsize=5`.

## Integration Points

- The new function follows the exact same pattern as `generate_trajectory_graphs` and `generate_bone_lengths_graph` -- same input types, same `_save()` helper, same `output_dir` convention.
- No new dependencies. Only uses `numpy` and `matplotlib` which are already imported.
- The data (`det_cam_positions`, `optimized_3d`, `gt_cam`) is already available at the call site in `main.py`.

## Sanity Checks for the Tester

The tester should verify:

1. **GT limb lengths should be nearly constant** across frames -- real human arms do not change length. If GT lines are noisy, something is wrong with GT loading.
2. **Optimized limb lengths should be more constant than detector limb lengths** -- the FK optimization enforces shared bone lengths across frames, so the red lines should be flatter than the green lines.
3. **All limb lengths should be in a reasonable range**: upper arm ~0.25-0.35m, forearm ~0.22-0.30m for an adult. If values are outside 0.1-0.6m, something is likely wrong.
4. **12 lines should be visible** -- verify the legend shows all 12 entries.
5. **Colors match spec**: blue=GT, green=detector, red=optimized.

## Risks and Edge Cases

- **Missing GT frames**: When `gt_3d[i]` is `None`, use `np.nan` for that frame's limb length. Matplotlib will automatically create gaps in the line at NaN values.
- **Single-frame examples**: The graph will just show dots. This is fine; unlikely to occur with the current config (all examples have 100-150 frames).
- **Legend crowding**: With 12 lines, the legend could be large. Using `bbox_to_anchor` to place it outside the plot area, combined with `bbox_inches="tight"` in `_save()`, handles this cleanly.
