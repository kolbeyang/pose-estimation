# Architect Plan: Phase 2, Iteration 0 -- Occlusion Handling

## Goal Summary

Improve how the pipeline handles occluded/low-confidence keypoints. Currently, `real_heatmap_score` skips joints with visibility < 1e-6 entirely, and `optimize.py` zeroes out visibility below a hard threshold (0.5). The spec asks us to replace this with a cleaner confidence-weighted scoring formula that gracefully degrades rather than hard-cutting. We also need to: (1) remove the old synthetic `heatmap_score` function, (2) add a confidence score graph, (3) pick a visibility threshold for overlay video visualization, and (4) create augmented test videos with occlusion patterns.

## Data Flow Summary (Current State)

1. **Stacked Hourglass** produces `(16, 64, 64)` heatmaps per frame. The peak value of each joint's heatmap is its confidence score.
2. `detect.py::detect_poses()` returns `visibility_list` -- a list of `(16,)` arrays where each value is the SH confidence (peak heatmap value) for that joint.
3. `optimize.py` converts visibility to torch tensors, then **hard-thresholds** them: any joint with confidence < `VISIBILITY_THRESHOLD` (0.5) gets zeroed to 0.0 (line 176-181).
4. `scoring.py::real_heatmap_score()` receives the (already thresholded) visibility, and **skips** any joint with visibility < 1e-6 (line 103). For non-skipped joints, it computes `log(clamp(sampled_value, eps)) * visibility[j]`.
5. The old `heatmap_score()` (analytical Gaussian) is still present but only used as fallback when no heatmaps are provided.

## Files to Modify

### 1. `motionbert-pose/scoring.py`
- Delete the `heatmap_score` function (lines 36-60).
- Rename `real_heatmap_score` to `heatmap_score`.
- Modify the renamed function to use confidence-weighted formula.
- Remove the `visibility[j] < 1e-6: continue` skip logic.
- Update `compute_total_score` to call the renamed function.

### 2. `motionbert-pose/optimize.py`
- Remove the hard visibility thresholding block (lines 176-181) that zeroes out low-confidence joints.

### 3. `motionbert-pose/config.py`
- Remove `VISIBILITY_THRESHOLD` (no longer needed for scoring).
- Add `CONFIDENCE_EPSILON` parameter.
- Add `OVERLAY_VISIBILITY_THRESHOLD` for visualization-only thresholding.

### 4. `motionbert-pose/graphs.py`
- Add `generate_confidence_score_graph` function.

### 5. `motionbert-pose/main.py`
- Add call to `generate_confidence_score_graph`.
- Pass visibility data to overlay video for thresholding.

### 6. `motionbert-pose/overlay_video.py`
- Accept visibility data and threshold parameter.
- Skip drawing keypoints/bones for joints below the visibility threshold.

## Files to Create

### 1. `motionbert-pose/create_occlusion_videos.py`
- Script to generate augmented test videos with occlusion patterns.

## Step-by-Step Instructions

### Step 1: Modify `scoring.py` -- Remove old `heatmap_score`, rename and update `real_heatmap_score`

1. **Delete** the `heatmap_score` function (the analytical Gaussian one, lines 36-60).

2. **Rename** `real_heatmap_score` to `heatmap_score`. Keep the same signature but add a `confidence_epsilon` parameter:

```python
def heatmap_score(
    projected_2d: torch.Tensor,
    heatmaps: torch.Tensor,
    affine: torch.Tensor,
    visibility: torch.Tensor,
    target_2d: torch.Tensor,
    sigma: float,
    confidence_epsilon: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
```

3. **Remove** the early-skip logic `if visibility[j] < 1e-6: continue` (line 103). Every joint should now contribute to the score.

4. **Replace** the log-likelihood computation. Currently:
```python
log_val: torch.Tensor = torch.log(torch.clamp(value, min=eps))
total_score = total_score + log_val * visibility[j]
```

Change to:
```python
conf = visibility[j]
log_val = torch.log(value * conf + confidence_epsilon * (1.0 - conf))
total_score = total_score + log_val
```

Key points about this formula:
- When `conf` is high (near 1.0): `log_val ~= log(value)` -- full heatmap signal.
- When `conf` is low (near 0.0): `log_val ~= log(confidence_epsilon)` -- a constant negative number, contributing a flat penalty regardless of the projected position. The optimizer gets no gradient from this joint.
- The `value * conf` term is already non-negative (heatmap values are >= 0, conf >= 0). The `confidence_epsilon * (1 - conf)` term provides a floor that prevents log(0). So the `min=eps` clamp on `value` is no longer needed, but to be safe, keep an outer clamp: `log_val = torch.log(torch.clamp(value * conf + confidence_epsilon * (1.0 - conf), min=eps))`.
- Note: `visibility[j]` is **no longer multiplied** as a weight outside the log. The confidence is now *inside* the log, which is the spec's formula.

5. **Apply the same formula to the fallback** (Hip/Spine analytical Gaussian path). For the fallback joints where `mpii_idx is None`:

Currently:
```python
diff = projected_2d[j] - target_2d[j]
sq_dist = (diff ** 2).sum()
log_likelihood = -sq_dist / (2.0 * sigma ** 2)
total_score = total_score + log_likelihood * visibility[j]
```

Change to: convert the analytical Gaussian to a value in [0, 1], then apply the same confidence formula:
```python
diff = projected_2d[j] - target_2d[j]
sq_dist = (diff ** 2).sum()
value = torch.exp(-sq_dist / (2.0 * sigma ** 2))
conf = visibility[j]
log_val = torch.log(torch.clamp(value * conf + confidence_epsilon * (1.0 - conf), min=eps))
total_score = total_score + log_val
```

6. **Update `compute_total_score`**:
   - Remove the `else` branch that calls the old analytical `heatmap_score`. Since we always have real heatmaps in the MotionBERT pipeline, this branch is dead code. But if we want to keep a fallback, we can leave it -- just note that the old function no longer exists. For safety, raise an error if heatmaps are not provided:
   ```python
   if not _use_real:
       raise ValueError("Heatmaps and affine are required for scoring.")
   ```
   - Pass `confidence_epsilon` through. Add it as a parameter to `compute_total_score` with default `1e-4`.
   - Update the call to `heatmap_score` (was `real_heatmap_score`) to pass `confidence_epsilon`.

### Step 2: Modify `optimize.py` -- Remove hard visibility thresholding

Remove lines 176-181 (the block that zeroes out visibility below `VISIBILITY_THRESHOLD`):

```python
# DELETE THIS BLOCK:
# Apply visibility threshold -- zero out low-confidence joints
for i in range(len(visibility_t)):
    visibility_t[i] = torch.where(
        visibility_t[i] >= cfg.VISIBILITY_THRESHOLD,
        visibility_t[i],
        torch.zeros_like(visibility_t[i]),
    )
```

The confidence is now handled inside the scoring formula; no need to pre-threshold.

### Step 3: Modify `config.py`

1. **Remove** `VISIBILITY_THRESHOLD = 0.5` (line 83-84).
2. **Add** after the existing scoring config:
```python
# Confidence epsilon for occlusion-aware scoring.
# When a joint has low confidence, the score degrades to log(confidence_epsilon),
# providing no gradient signal. Higher values = less penalty for occluded joints.
CONFIDENCE_EPSILON: float = 1e-4

# Visibility threshold for overlay video and 3D visualization only.
# Joints below this threshold are not drawn (but still scored).
OVERLAY_VISIBILITY_THRESHOLD: float = 0.3
```

Note: The initial value of `OVERLAY_VISIBILITY_THRESHOLD` (0.3) is a placeholder. The developer will determine the actual value in Step 6.

### Step 4: Add `generate_confidence_score_graph` to `graphs.py`

Add the following function after `generate_limb_length_graph`:

**Function signature:**
```python
def generate_confidence_score_graph(
    visibility: list[np.ndarray],
    output_dir: str,
) -> None:
```

**Implementation:**
1. Create a figure with `figsize=(14, 6)`.
2. `visibility` is a list of `(16,)` arrays, one per frame.
3. Stack into `(N, 16)` array.
4. For each of the 16 joints, plot a line: x = frame index, y = confidence score.
5. Use `JOINT_NAMES` for labels.
6. Use a distinct color per joint. Use `plt.cm.tab20` colormap: `colors = plt.cm.tab20(np.linspace(0, 1, 16))`.
7. X-axis: "Frame". Y-axis: "Confidence Score". Title: "Per-Keypoint Confidence Scores".
8. Place legend outside the plot (`bbox_to_anchor=(1.05, 1)`, `loc="upper left"`, `fontsize=7`).
9. Save to `os.path.join(output_dir, "confidence_scores.png")`.

Also add a miniature version to the summary grid. Replace the legend panel at `axes[0, 3]` -- actually, that's already occupied. Instead, the developer should look for a less critical subplot slot. The best option: add to the aggregate summary as a new graph. Actually, keeping it simple -- just generate it as a standalone per-example graph. Do NOT modify the summary grid for this.

### Step 5: Integrate confidence graph in `main.py`

1. Add `generate_confidence_score_graph` to the imports from `graphs`.
2. In `process_example()`, after the limb length graph call (line 327-329), add:
```python
generate_confidence_score_graph(visibility, example_graph_dir)
```

### Step 6: Determine the overlay visibility threshold

The developer must:
1. Run one example (e.g., `EXAMPLES[0]`: `171204_pose1_sample`).
2. Generate the confidence score graph.
3. Print min/median/max confidence per joint across all frames.
4. Save out one video frame and identify which keypoints are clearly not in the frame.
5. Cross-reference those joints' confidence scores to identify a reasonable threshold.

Expected behavior: joints that are clearly visible will have confidence scores > 0.5 (often > 2.0 since SH heatmap peaks can exceed 1.0). Joints that are off-screen or heavily occluded will have very low values (< 0.1).

**Important note**: SH heatmap peak values are NOT necessarily in [0, 1]. They can exceed 1.0. The developer should check the actual range before picking a threshold. A threshold around 0.1-0.5 is likely reasonable.

Once determined, update `OVERLAY_VISIBILITY_THRESHOLD` in `config.py`.

### Step 7: Modify `overlay_video.py` -- Visibility-based filtering

1. Add `visibility` and `visibility_threshold` parameters to `generate_overlay_video`:
```python
def generate_overlay_video(
    ...
    visibility: list[np.ndarray] | None = None,
    visibility_threshold: float = 0.3,
) -> None:
```

2. Modify `_draw_skeleton_2d` to accept an optional visibility mask:
```python
def _draw_skeleton_2d(
    frame: np.ndarray,
    pts_2d: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    visible_mask: np.ndarray | None = None,
) -> None:
```

When `visible_mask` is provided, skip drawing bones where either endpoint is not visible, and skip drawing joint circles for invisible joints.

3. In the main loop, when drawing the green (detector) and red (optimized) skeletons, pass the visibility mask:
```python
vis_mask = None
if visibility is not None and i < len(visibility):
    vis_mask = visibility[i] >= visibility_threshold
```

Pass `vis_mask` to `_draw_skeleton_2d` for the detector and optimized skeletons. Do NOT apply it to the ground truth skeleton (blue), which should always be fully drawn.

4. Also apply the threshold to the yellow SH 2D keypoint dots: skip dots where `kp_mpii[j, 2] < visibility_threshold` (replace the current `< 0.01` check).

### Step 8: Update `main.py` to pass visibility to overlay video

In the `generate_overlay_video` call (around line 339-353), add:
```python
visibility=visibility,
visibility_threshold=cfg.OVERLAY_VISIBILITY_THRESHOLD,
```

### Step 9: Create `create_occlusion_videos.py`

This script creates augmented test videos by blacking out frames according to patterns.

**Approach:**
- Select 2 examples from `cfg.EXAMPLES`. Use indices 0 and 4 (different sequences for variety): `171204_pose1_sample` and `171204_pose3`.
- For each example, extract video frames using the existing `panoptic.py::extract_video_frames()`.
- For each of 4 occlusion patterns, create a new set of frames where "black" frames are replaced with all-zeros (black image of same resolution).
- Save the augmented frames as MP4 videos in a new directory: `motionbert-pose/occlusion_test_videos/`.
- Name convention: `{seq_name}_{start_frame}_{pattern_name}.mp4`

**Patterns:**
1. `every_other_black`: `[visible, black, visible, black, ...]`
2. `every_4th_visible`: `[visible, black, black, black, visible, black, black, black, ...]`
3. `every_4th_black`: `[visible, visible, visible, black, visible, visible, visible, black, ...]`
4. `16on_16off`: `[visible*16, black*16, visible*16, black*16, ...]`

The script should also save a JSON manifest listing the created videos and their patterns, for the tester to use.

**Important**: These are just the raw augmented *video files*. The spec says "create augmented test videos" -- meaning the developer creates the videos and the tester runs them through the full pipeline to evaluate how the system handles occlusion.

## Integration Points

- `scoring.py` changes affect `compute_total_score` which is called by `optimize.py`. The function signature change (rename + new parameter) needs to be consistent.
- `optimize.py` removing the threshold block is safe because the scoring formula now handles low-confidence joints gracefully.
- `config.py` removing `VISIBILITY_THRESHOLD` -- grep for any other uses. It's also referenced in `evaluate.py` (line 109, 121, 131) for 2D reprojection error evaluation. Leave that one alone -- it's a separate concern (evaluation filtering, not optimization scoring). If `evaluate.py` imports `VISIBILITY_THRESHOLD` from config, the developer should check and either keep that constant with a different name or inline the value.
- `overlay_video.py` changes are purely visual -- they don't affect scoring or optimization.

## Risks and Edge Cases

1. **`VISIBILITY_THRESHOLD` removal may break `evaluate.py`**: Check if `evaluate.py` imports `cfg.VISIBILITY_THRESHOLD`. If so, either keep the constant under a new name (e.g., `EVAL_VISIBILITY_THRESHOLD`) or inline the value `0.5` in `evaluate.py`. The developer should grep for all uses.

2. **Confidence epsilon value**: `1e-4` is a reasonable starting point. If it's too small, log(1e-4) = -9.2, which is a significant penalty per occluded joint per frame. If there are many occluded joints, this could dominate the loss. If too large (e.g., 0.1), the penalty for occluded joints is small (log(0.1) = -2.3) and the optimizer might not care enough about visible joints. The developer should verify that optimization still converges sensibly after this change.

3. **SH confidence range**: Stacked Hourglass heatmap peaks may exceed 1.0 (they're raw network outputs, not probabilities). If `visibility[j] > 1.0`, then `value * conf` could be larger than `value`, making the log larger than `log(value)`. This is actually fine -- it means high-confidence joints contribute more. But the developer should verify the range.

4. **Fallback joints (Hip, Spine)**: These use analytical Gaussian and have `visibility` derived from averaging neighboring joints' confidence. Their confidence may be systematically different from heatmap-based joints. Monitor these in the confidence graph.

5. **Occlusion test videos**: Black frames will cause SH to produce garbage heatmaps (low confidence, random positions). The system should handle this via the confidence-weighted scoring -- the low confidence means these frames contribute minimal gradient. MotionBERT should also handle missing/garbage input since we zero out low-confidence joints before feeding to it (via `MOTIONBERT_CONF_THRESHOLD`).

## Sanity Checks for the Tester

1. Run one normal example and verify MPJPE is not significantly worse than before the changes (small changes are expected due to the new scoring formula).
2. Verify the confidence score graph shows reasonable patterns: visible joints should have higher confidence than occluded joints.
3. Verify the overlay video correctly omits low-confidence joints.
4. Run the pipeline on each of the 4 occlusion-augmented videos and verify:
   - The system doesn't crash on black frames.
   - Confidence scores drop appropriately on black frames.
   - The system recovers on visible frames.
   - MPJPE is degraded but not catastrophically so (especially for patterns with mostly visible frames).
5. Verify `evaluate.py` still works (not broken by `VISIBILITY_THRESHOLD` removal).
