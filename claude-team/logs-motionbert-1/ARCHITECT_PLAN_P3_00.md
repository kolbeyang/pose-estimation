# ARCHITECT_PLAN_P3_00: Video with Overlays

## Goal Summary

Create an overlay video for each CMU Panoptic example that augments the original video frames with four visual layers: (1) a heatmap overlay rendered from the actual Stacked Hourglass (16, 64, 64) intermediate heatmaps (not synthetic Gaussians), (2) small yellow dots at the raw Stacked Hourglass 2D detection positions, (3) a green skeleton from MotionBERT's raw 3D prediction projected back to 2D via camera intrinsics, and (4) a red skeleton from the optimized 3D prediction projected the same way. The overlay video should be generated per-example and saved alongside the existing predictions JSON and graphs.

## Key Data Flow Analysis

The pipeline currently produces all the data needed for overlays:

1. **Stacked Hourglass heatmaps**: `detect_poses()` already returns `all_heatmaps` as a list of `(16, 64, 64)` numpy arrays in MPII joint order. These are the actual SH outputs, averaged over original + flipped inference.

2. **Affine transform**: `detect_poses()` returns `affine` -- a `(2, 3)` matrix mapping from 256-crop coordinates to original pixel coordinates. The heatmaps are 64x64 (quarter of the 256x256 crop), so the mapping from heatmap pixel `(hx, hy)` to original image pixel is: `x_orig = affine[0,0] * (hx * 4) + affine[0,2]`, `y_orig = affine[1,1] * (hy * 4) + affine[1,2]`.

3. **2D keypoints**: `kp_2d` is a list of `(17, 2)` in H36M format in original pixel coordinates. But we also want the raw MPII 2D points. The original MPII keypoints come from `run_hourglass()` as `all_keypoints_2d` -- a list of `(16, 3)` arrays `(x, y, confidence)` in original pixel coords. Currently only the H36M-converted 2D is passed through `detect_poses()`. We need access to the raw MPII 2D as well.

4. **3D positions**: `det_cam_positions` (MotionBERT camera-space) and `optimized_3d` are lists of `(17, 3)` arrays. These project to 2D via the Camera model: `camera.world_to_image(pos_3d)`.

5. **Camera intrinsics**: Available as `Camera` object and `CameraParams`.

6. **Video frames**: Need to reload from disk (same as mediapipe-pose approach), or pass through from the pipeline. Reloading is simpler and keeps memory usage bounded.

## Critical Design Decisions

### Heatmap Rendering Strategy

The mediapipe-pose overlay generates synthetic Gaussian blobs at detected 2D positions (`_render_heatmap` in `mediapipe-pose/overlay_video.py`). The spec explicitly says **not** to do this. Instead, we must use the actual Stacked Hourglass heatmaps.

The SH heatmaps are `(16, 64, 64)` per frame. To overlay on the video:
1. Sum across all 16 joint channels to get a single `(64, 64)` combined heatmap
2. Resize from 64x64 to the crop region size using bilinear interpolation
3. Place into the full frame using the affine transform (the crop may be offset and padded)
4. Apply the HOT colormap additive blend (same as mediapipe-pose)

### Raw 2D Points

The spec says "raw Stacked Hourglass 2D estimations" as yellow dots. These are the MPII 16-joint detections (not the H36M 17-joint conversion). Currently `detect_poses()` converts to H36M before returning 2D. We need to also pass through the raw MPII 2D keypoints.

**Two options:**
- (A) Return the raw MPII keypoints from `detect_poses()` in the return tuple
- (B) Store them in the predictions JSON

Option (A) is cleaner since we need them at video generation time during `process_example()`. We also need to store the heatmaps (or at least pass them through to the overlay function).

**Decision:** We will NOT store heatmaps in the JSON (they are large: 16x64x64 float32 = 256KB per frame). Instead, the overlay video will be generated during `process_example()` while heatmaps are still in memory.

### 3D-to-2D Projection

Both MotionBERT raw and optimized 3D are in camera space (meters). Projection to 2D uses: `u = fx * X / Z + cx`, `v = fy * Y / Z + cy`. This is exactly what `Camera.world_to_image()` does. No denormalization needed -- the 3D positions are already in camera meters.

## Files to Modify

### 1. `motionbert-pose/detect.py`
- Modify `detect_poses()` return type to also include the raw MPII 2D keypoints list
- Return `all_keypoints_2d` (the MPII 16-joint `(16, 3)` arrays with x, y, conf in original pixel coords) as an additional element

### 2. `motionbert-pose/main.py`
- Update `detect_poses()` call to capture the new MPII 2D keypoints return value
- After saving graphs, call the new overlay video generation function
- Pass: frames_rgb (or frame_indices + video_path for reload), heatmaps, MPII 2D keypoints, det_cam_positions, optimized_3d, camera, affine

### 3. `motionbert-pose/config.py`
- Add `OVERLAY_HEATMAP_INTENSITY: float = 200.0` (additive blend strength)
- Add `OVERLAY_SIGMA_SMOOTH: float = 0.0` (optional Gaussian smooth on heatmap before overlay, 0 = no smoothing)

## Files to Create

### 1. `motionbert-pose/overlay_video.py`
Purpose: Generate overlay video from in-memory data (heatmaps, keypoints, 3D positions, camera).

## Step-by-Step Instructions

### Step 1: Modify `detect.py` to return raw MPII 2D keypoints

In `detect_poses()`:

1. Change the return type annotation to add one more element:
```python
def detect_poses(
    frames_rgb: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, dict[str, float]]:
```

2. Return `all_keypoints_2d` (the raw MPII list from `run_hourglass()`) as a new element. Insert it after `all_heatmaps`:
```python
return kp_2d_list, kp_3d_list, visibility_list, all_heatmaps, all_keypoints_2d, affine, positions_3d_norm, cs_params
```
Where `all_keypoints_2d` is the list of `(16, 3)` MPII keypoints in original pixel coords.

3. Update the docstring to document the new return element:
```
mpii_keypoints_2d: List of (16, 3) raw MPII keypoints per frame (x, y, conf) in original pixel coords.
```

### Step 2: Create `motionbert-pose/overlay_video.py`

Create a new file with the following functions:

#### Function: `_resize_heatmap_to_frame(heatmap_64: np.ndarray, affine: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray`

Resizes and places a (64, 64) heatmap into the full frame coordinate system.

```python
def _resize_heatmap_to_frame(
    heatmap_64: np.ndarray,
    affine: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> np.ndarray:
    """Resize a 64x64 heatmap to the full video frame using the affine transform.

    The heatmap is in 64x64 space (quarter of the 256x256 crop).
    affine maps 256-crop coords -> original pixel coords:
        x_orig = sx * x_256 + tx
        y_orig = sy * y_256 + ty

    Steps:
    1. Resize 64x64 -> crop region size using bilinear interpolation
    2. Place into full frame at the correct offset

    Args:
        heatmap_64: (64, 64) single-channel heatmap.
        affine: (2, 3) affine transform (256-crop -> original pixels).
        frame_h: Full frame height.
        frame_w: Full frame width.

    Returns:
        (frame_h, frame_w) float32 heatmap in original pixel space.
    """
```

Implementation details:
- `sx = affine[0, 0]` and `sy = affine[1, 1]` are the scale factors from 256-crop to original
- `tx = affine[0, 2]` and `ty = affine[1, 2]` are the offsets
- The 64x64 heatmap corresponds to a 256x256 crop. The crop maps to an original region of size `(256 * sx, 256 * sy)` at offset `(tx, ty)`
- Use `cv2.resize(heatmap_64, (int(256 * sx), int(256 * sy)), interpolation=cv2.INTER_LINEAR)` to upscale
- Create a zeros array of shape `(frame_h, frame_w)` and paste the resized heatmap at the correct location, clipping to frame bounds

#### Function: `_blend_heatmap_additive(frame_bgr: np.ndarray, heatmap: np.ndarray, intensity: float = 200.0) -> np.ndarray`

Copy directly from mediapipe-pose `overlay_video.py` -- same HOT colormap additive blend logic. This is a well-tested function.

#### Function: `_project_3d_to_2d(pts_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray`

Same as mediapipe-pose -- perspective projection `(N, 3) -> (N, 2)`.

#### Function: `_draw_skeleton_2d(frame: np.ndarray, pts_2d: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None`

Draw skeleton bones + joint circles. Use `BONES` from `skeleton.py`. Same logic as mediapipe-pose.

#### Function: `generate_overlay_video(...)`

Main function signature:
```python
def generate_overlay_video(
    output_path: str,
    frames_rgb: list[np.ndarray],
    heatmaps: list[np.ndarray],
    mpii_keypoints_2d: list[np.ndarray],
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    camera_fx: float,
    camera_fy: float,
    camera_cx: float,
    camera_cy: float,
    affine: np.ndarray,
    frame_indices: list[int] | None = None,
    gt_3d: list[np.ndarray | None] | None = None,
    intensity: float = 200.0,
) -> None:
```

Per-frame logic:
1. Convert frame RGB -> BGR for OpenCV
2. **Heatmap overlay**: Sum the 16-channel heatmap to get a single `(64, 64)` combined map. Use `_resize_heatmap_to_frame()` to place into frame coords. Use `_blend_heatmap_additive()` to overlay with HOT colormap.
3. **Yellow dots for raw 2D**: For each of the 16 MPII joints in `mpii_keypoints_2d[i]`, draw a small filled circle (radius=4) in yellow `(0, 255, 255)` in BGR at `(x, y)`. Skip joints with confidence < 0.01.
4. **Green skeleton (MotionBERT raw)**: Project `detector_3d[i]` to 2D using `_project_3d_to_2d()`. Draw with `_draw_skeleton_2d()` in green `(0, 255, 0)`.
5. **Red skeleton (Optimized)**: Project `optimized_3d[i]` to 2D. Draw in red `(0, 0, 255)`.
6. **Blue skeleton (Ground Truth)**: If `gt_3d` is provided and not None for this frame, project and draw in blue `(255, 100, 0)` (same teal-blue as mediapipe-pose).
7. **Frame label + legend**: Add text overlay with frame index and legend explaining colors.

Write with `cv2.VideoWriter` at 5 fps (matching mediapipe-pose), codec `mp4v`.

### Step 3: Update `main.py`

In `process_example()`:

1. Update the `detect_poses()` call to unpack the new MPII 2D keypoints:
```python
kp_2d, kp_3d, visibility, heatmaps, mpii_kp_2d, affine, positions_3d_norm, cs_params = detect_poses(frames_rgb)
```

2. After the graph generation section (after `generate_summary(...)` call), add overlay video generation:
```python
# Overlay video
from overlay_video import generate_overlay_video
overlay_path: str = os.path.join(example_graph_dir, f"{name}_overlay.mp4")
generate_overlay_video(
    output_path=overlay_path,
    frames_rgb=frames_rgb,
    heatmaps=heatmaps,
    mpii_keypoints_2d=mpii_kp_2d,
    detector_3d=det_cam_positions,
    optimized_3d=optimized_3d,
    camera_fx=fx,
    camera_fy=fy,
    camera_cx=cx,
    camera_cy=cy,
    affine=affine,
    frame_indices=frame_indices[:len(frames_rgb)],
    gt_3d=gt_cam,
)
print(f"    Saved overlay video: {overlay_path}")
```

3. Move the `from overlay_video import generate_overlay_video` to the top-level imports.

### Step 4: Update `config.py` (optional)

Add overlay configuration constants if needed for tuning:
```python
# --- Overlay video ---
OVERLAY_HEATMAP_INTENSITY: float = 200.0
OVERLAY_FPS: float = 5.0
```

### Step 5: Update `test_single.py` (optional)

Update the `detect_poses()` return unpacking if `test_single.py` calls it directly. Check the file and add the new return element to the unpacking.

## Integration Points

1. **`detect_poses()` return change**: This is a breaking API change. Every caller must be updated. Known callers:
   - `main.py` -> `process_example()` -- updated in Step 3
   - `test_single.py` -- check and update

2. **Overlay video is generated inside `process_example()`** while `frames_rgb` and `heatmaps` are in memory. This avoids having to save/reload heatmaps.

3. **The overlay video goes in the per-example graph directory** alongside summary.png, etc. Path: `training_runs/{run}/graphs/{example_name}/{example_name}_overlay.mp4`.

## Risks and Edge Cases

1. **Memory**: Holding `frames_rgb` (100 frames at 1080p = ~600MB) and `heatmaps` (100 frames * 16 * 64 * 64 * 4 bytes = ~25MB) simultaneously is fine on modern machines. The frames are already loaded for detection.

2. **Affine edge cases**: If the crop extends outside the frame (negative offsets or exceeds frame size), the `_resize_heatmap_to_frame()` must clip correctly. The crop_and_resize function in detect.py handles padding, but the affine's offset (`tx`, `ty`) can be negative. The paste operation must handle this by computing intersection bounds.

3. **Heatmap value range**: The SH heatmaps may have negative values (they come from a neural network). Before blending, clamp to `max(0, ...)` or use only positive values. Check this -- if the heatmaps are pre-sigmoid, they may need sigmoid/softmax. Based on the `_parse_heatmaps()` function which uses `argmax` directly, they appear to be post-sigmoid (positive values). But verify during implementation by checking the actual value range.

4. **Empty frames**: If a frame has no detections or missing GT, handle gracefully (skip overlay elements that are unavailable).

5. **Coordinate convention**: The MPII 2D keypoints from `run_hourglass()` are already in original pixel coordinates (the affine transform is applied inside `run_hourglass()`). The heatmaps are in 64x64 crop space and need the affine to map to original coordinates.

6. **`test_single.py` breakage**: Since we change the `detect_poses()` return signature, `test_single.py` will break if it calls `detect_poses()` directly. Must be updated.

## Testing Requirements (from spec)

1. Run and create the overlay video for a single example
2. Select one frame from the middle, save as a temp image
3. Visually confirm all requirements are present on that frame:
   - Orange/yellow heatmap glow from actual SH heatmaps
   - Small yellow dots at raw SH 2D detection positions (16 MPII joints)
   - Green skeleton from MotionBERT 3D projected to 2D
   - Red skeleton from optimized 3D projected to 2D
4. Note anything glaringly incorrect
