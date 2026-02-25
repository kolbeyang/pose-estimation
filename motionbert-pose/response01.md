# Response 01: Fresh Start Architecture

## What I Understand

You want a clean, single-file pipeline that:
1. Takes a video as input
2. Runs a 2D pose model (with heatmap output) + a 3D lifting model (MotionBERT)
3. Initializes a rigid arm model from the 3D estimates
4. Runs iterative optimization against the heatmaps + motion smoothness
5. Outputs: VPython visualization, graphs, overlay frames, saved heatmaps — all in `training_runs/`

Left arm only for now. Single person. Minimize complexity.

---

## Model Pipeline Recommendation

### Current Pipeline (3 models)
```
YOLOv8 (detection) → Stacked Hourglass (2D heatmaps) → MotionBERT (3D lifting)
```

### Proposed Pipeline (2 models)
```
ViTPose (2D heatmaps, includes built-in detection) → MotionBERT (3D lifting)
```

**Wait — ViTPose does NOT include detection.** It's a top-down model that still needs bounding boxes. So you'd still need a detector. The options:

| Option | Models | Complexity | Notes |
|--------|--------|-----------|-------|
| A. YOLOv8-nano + ViTPose | 2 | Medium | Best heatmaps, well-supported in HuggingFace Transformers |
| B. YOLOv8-nano + Stacked Hourglass | 2 | Medium | What you had before, 64x64 heatmaps (low res) |
| C. YOLO11-pose (single model, no heatmaps) + MotionBERT | 1 for 2D | Low | BUT: no heatmaps — just coordinate regressions. Can't use for optimization. |
| D. RTDetr + ViTPose | 2 | Medium | RTDetr is a transformer detector, slightly more accurate than YOLO |

**My recommendation: Option A (YOLOv8-nano + ViTPose).** Reasons:
- ViTPose outputs proper spatial heatmaps (not just coordinates) — essential for your optimization
- Available directly from HuggingFace Transformers (`from transformers import VitPoseForPoseEstimation`)
- Higher resolution heatmaps than Stacked Hourglass (which gives 64x64)
- YOLOv8-nano is fast and you already have it working
- Single-person means detection is trivial — just take the biggest box

The only true one-step models (RTMO, YOLO-pose) either don't output heatmaps or use internal 1D representations that aren't useful for your optimization loss.

### ViTPose Details
- Pretrained: `usyd-community/vitpose-base-simple` (COCO 17 joints)
- Output: `(batch, 17, H, W)` heatmaps — directly usable
- Needs COCO→H36M joint mapping for MotionBERT input
- Left arm joints: shoulder=5, elbow=7, wrist=9 (COCO) → 11, 12, 13 (H36M)

### MotionBERT
- Keep using MotionBERT-Lite as before
- Input: H36M 17-joint 2D keypoints normalized to [-1, 1]
- Output: normalized 3D coordinates (not metric)

---

## Camera Calibration — Simplified Strategy

This was the most complex part of your old pipeline (per-frame affine camera fitting). Here's how to simplify:

**The core question:** You need to project 3D arm positions back to 2D pixels for the heatmap loss. This requires some camera model.

**Option 1: Weak perspective (simplest)**
- Assume camera at fixed position looking at the scene
- Fit a single scale + translation per frame (4 params: sx, sy, tx, ty)
- `u = sx * X + tx`, `v = sy * Y + ty`
- Pro: Dead simple, 4 params, closed-form solution
- Con: No perspective effects, but for a single arm this barely matters

**Option 2: Pinhole with assumed focal length (moderate)**
- Assume principal point = image center
- Estimate focal length from image width: `f = w * 1.2` (reasonable for most cameras)
- Fit extrinsics (R, t) from 2D-3D correspondences via solvePnP
- Pro: More physically accurate
- Con: solvePnP can be unstable with only 3 arm joints

**Option 3: Per-frame affine (what you had)**
- Fit a 2x4 affine matrix per frame
- Needs enough correspondences (you used 11 upper body joints)
- Pro: Most flexible
- Con: Most complex, and you only care about 3 arm joints

**My recommendation: Option 1 (weak perspective) or a hybrid.**

Since you have MotionBERT outputting 17 joints and ViTPose outputting 17 joint heatmaps, you can fit the camera using ALL visible joints (not just the arm), but then only optimize the arm parameters. This gives you robust camera estimation without complexity.

Concretely:
1. Extract 2D keypoints from ViTPose (17 joints)
2. Get 3D keypoints from MotionBERT (17 joints)
3. Fit a simple camera (weak perspective or pinhole) using all 17 joint correspondences
4. Fix the camera — don't optimize it
5. Only optimize the arm parameters against the arm heatmaps

This decouples camera estimation from arm optimization and is much simpler.

**Question for you:** Are you okay with fitting the camera from all 17 joints and then freezing it? Or do you want the camera to also be learnable during optimization?

---

## 3D → Arm Initialization (Simple Approach)

You said it doesn't have to be exact. Here's the plan:

From MotionBERT's 3D output for left arm (joints 11=shoulder, 12=elbow, 13=wrist):

```python
# Shoulder position = direct xyz
a_pos = joints_3d[11]  # shoulder

# Upper arm direction: shoulder → elbow
ab_vec = joints_3d[12] - joints_3d[11]
a_b_length = np.linalg.norm(ab_vec)

# Convert AB vector to polar (azimuth, elevation)
azimuth = np.arctan2(ab_vec[1], ab_vec[0])
elevation = np.arcsin(ab_vec[2] / a_b_length)
roll = 0.0  # default, not critical for initialization

# Forearm bend: angle between AB and BC vectors
bc_vec = joints_3d[13] - joints_3d[12]
b_c_length = np.linalg.norm(bc_vec)
# Compute angle in local frame → b_c_theta
cos_angle = np.dot(ab_vec, bc_vec) / (a_b_length * b_c_length)
b_c_theta = np.arccos(np.clip(cos_angle, -1, 1))
# Adjust for your theta convention (θ=π/2 is straight)
```

No inverse kinematics needed. Just raw geometry. The optimization will refine from there.

---

## Output Organization

Everything in `motionbert-pose/training_runs/<run-name>/`:

```
training_runs/run-2026-02-23-143000/
├── heatmaps/                  # saved heatmap arrays
│   ├── frame_0000_shoulder.npy
│   ├── frame_0000_elbow.npy
│   ├── frame_0000_wrist.npy
│   └── ...
├── overlays/                  # video frames with projected skeletons
│   ├── frame_0000.png         # MotionBERT (green) + optimized (red) overlaid
│   └── ...
├── graphs/
│   ├── shoulder_x.png
│   ├── shoulder_y.png
│   ├── shoulder_z.png
│   ├── elbow_x.png
│   ├── elbow_y.png
│   ├── elbow_z.png
│   ├── wrist_x.png
│   ├── wrist_y.png
│   ├── wrist_z.png
│   ├── bone_lengths.png
│   └── loss_curve.png
├── summary.png                # all graphs in one grid
├── config.json                # hyperparameters used
└── metrics.json               # final scores, reprojection errors, etc.
```

---

## VPython Visualization + Recording

**VPython has no built-in video export.** Options:

1. **scene.capture()** — saves individual frames as images. You'd call this each animation step, then stitch with ffmpeg. Doable but clunky.

2. **Use matplotlib 3D instead** — `matplotlib.animation.FuncAnimation` can save directly to MP4 via ffmpeg. Less interactive than VPython but trivial to record.

3. **Keep VPython for interactive viewing, add matplotlib for recording** — best of both worlds.

**My recommendation:** Use VPython for the interactive browser visualization (as you requested). For saving a video, render the same data with matplotlib's 3D plotting + `FuncAnimation.save('output.mp4')`. This gives you both an interactive view and a saved video without any screen-capture hacks.

**Question for you:** Is the matplotlib video approach acceptable, or do you specifically want to record the VPython window?

---

## Graphs

Keeping the same graphs you had, they're good:
- 9 coordinate trajectory plots (shoulder/elbow/wrist x xyz) showing MotionBERT vs optimized
- Bone length evolution during training
- Loss curve
- Summary grid combining everything

---

## Single-File Structure

You asked for a single file. Here's the rough structure of `main.py`:

```python
# main.py — complete pipeline

# === IMPORTS ===
# torch, cv2, numpy, matplotlib, vpython, transformers, ultralytics

# === CONFIG ===
@dataclass
class Config:
    video_path: str
    target_fps: int = 10
    num_steps: int = 100
    learning_rate: float = 0.0005
    # ... penalty weights, etc.

# === MODEL LOADING ===
def load_models():
    """Load YOLOv8, ViTPose, MotionBERT"""

# === DETECTION ===
def detect_2d(frames, yolo, vitpose):
    """Run YOLO + ViTPose → 2D keypoints + heatmaps"""

def lift_3d(keypoints_2d, motionbert):
    """Run MotionBERT → 3D keypoints"""

# === CAMERA ===
def fit_cameras(keypoints_2d, keypoints_3d):
    """Fit per-frame cameras from 2D-3D correspondences"""

# === INITIALIZATION ===
def init_arm_from_3d(joints_3d):
    """Convert 3D joint positions to Arm parameters"""

# === OPTIMIZATION ===
def optimize(arms, heatmaps, cameras, config):
    """Adam optimization against heatmaps + smoothness"""

# === EVALUATION ===
def generate_graphs(results, run_dir):
    """Save all trajectory + training plots"""

def generate_overlays(frames, results, cameras, run_dir):
    """Save frames with projected skeletons"""

def save_heatmaps(heatmaps, run_dir):
    """Save raw heatmaps as .npy"""

# === VISUALIZATION ===
def vpython_visualize(results):
    """Interactive VPython browser visualization"""

def save_video(results, run_dir):
    """Matplotlib 3D animation → MP4"""

# === MAIN ===
if __name__ == "__main__":
    config = Config(video_path=sys.argv[1])
    # ... run everything
```

---

## Open Questions for You

1. **Camera strategy:** Weak perspective (simplest, 4 params) vs pinhole with solvePnP (more accurate)? I recommend weak perspective for simplicity.

Ok sounds good

2. **Video recording:** matplotlib MP4 export acceptable, or must it be VPython?

No need for this

3. **ViTPose vs Stacked Hourglass:** I'm recommending ViTPose for better heatmaps. Any concerns about switching?

Alright let's do that.

4. **MotionBERT coordinate system:** The output is in normalized coordinates (not metric). For initialization, this means `a_pos` will be in a [-1, 1]-ish range, not meters. The optimization should still work since the camera projection maps these to pixel space. Just flagging this.

Oh shoot, is there an alternative to motionbert, what other state of the art options are there that fit our constraints.

5. **Dependencies:** ViTPose needs `transformers` library. MotionBERT needs its repo cloned. Want me to handle the setup script as well, or just focus on the main pipeline?

Yeah all that is okay.

6. **Existing code reuse:** You said start from scratch, but the `model/arm.py` and `scoring.py` logic is solid. Should I rewrite those or import from the existing codebase?

Let's do more research first.
