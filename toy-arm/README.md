# toy-arm

A 2-segment toy arm in 3D used as a reference / sandbox for the differentiable pose-optimization approach used in [`pose-optimizer/`](../pose-optimizer/). Strips the problem down to its essentials: rigid skeleton with fixed bone lengths, parameterized by joint angles, projected to 2D, recovered from synthetic heatmap-like observations via Adam.

## Quick Start

```bash
cd toy-arm
uv sync

# Headless: generate a synthetic video and recover the arm parameters
uv run python run_headless_eval.py

# Recover arm parameters from a saved video folder
uv run python optimization.py path/to/video_folder

# Interactive 3D viewer (VPython, opens in browser)
uv run python visualize.py path/to/video_folder

# Random-walk arm animation (VPython)
uv run python motion_simulation.py
```

## Arm Parameterization

The arm has two rigid segments AB and BC with fixed bone lengths:

- `a_pos` — 3D position of point A (the "shoulder"), 3 params
- `a_b_polar` — orientation of segment AB as `(azimuth, elevation, roll)`, 3 params
- `b_c_theta` — bend angle of forearm BC relative to AB's local frame, 1 param
  - `θ = π/2` → straight arm (BC aligned with AB)
  - `θ = 0` → 90° bend
  - `θ = -π/2` → fully folded back

7 parameters per frame total. Bone lengths (`a_b_length`, `b_c_length`) are fixed across a video.

## Pipeline

```
Random-walk arm motion ── render to 2D image (synthetic "heatmaps")
                                    │
                                    ▼
                         Differentiable forward kinematics
                                    │
                                    ▼
                          Adam optimizer (PyTorch)
                                    │
                                    ▼
                       Recovered arm parameters per frame
```

## File Structure

```
model/
    arm.py              Arm class: polar-coord parameterization, FK to (a, b, c)
    camera.py           Pinhole camera: world-to-image projection
    environment.py      3D scene + camera intrinsics/extrinsics

run_headless_eval.py    End-to-end: generate synthetic video, optimize, report metrics
optimization.py         Adam optimizer for recovering arm params from a saved video
tune_hyperparams.py     Grid-search hyperparameter sweep
visualize.py            VPython 3D viewer for ground-truth + recovered arm
motion_simulation.py    VPython random-walk arm animation
arm_demo.py             VPython interactive parameter sliders
glowscript_demo.py      Browser-only GlowScript variant of the demo
video.py                Synthetic video container (heatmaps + GT angles per frame)
utils.py                Loss terms, optimization config, evaluation helpers

convergence_data.json   Saved convergence curves from prior eval runs
```

## Conventions

- Polar angles in radians: azimuth `[-π, π]`, elevation `[-π/2, π/2]`, roll `[-π, π]`.
- Position bounded to a unit cube `[-1, 1]³`.
- Frame-to-frame motion is generated as a random walk over 7 velocity components clipped to per-axis bounds.
