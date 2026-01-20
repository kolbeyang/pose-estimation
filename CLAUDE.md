# Pose Estimation Project

## Quick Start

This is a uv-managed Python project. Run scripts with:

```bash
uv run python <script.py>
```

## Project Structure

```
model/
  arm.py          # Arm class - 2-segment arm with polar coordinates
  environment.py  # Environment class - camera, cube bounds, projection

motion_simulation.py  # VPython animation with random velocity-based arm movement
visualize.py          # Static VPython visualization from JSON arm config
gen_arm_images.py     # Generate arm images with JSON params
main.py               # Basic VPython demo
arm_demo.py           # Interactive arm parameter demo
```

## Key Classes

### Arm (model/arm.py)

Two-segment arm defined by:
- `a_pos`: np.ndarray (x, y, z) - position of point A
- `a_b_length`, `b_c_length`: segment lengths
- `a_b_polar`, `b_c_polar`: tuple (azimuth, elevation) in radians

Polar coordinate convention:
- Azimuth: rotation in XY plane, range [-π, π]
- Elevation: angle from XY plane, range [-π/2, π/2]

`get_coordinates()` returns dict with keys "a", "b", "c" as np.ndarray positions.

### Environment (model/environment.py)

Defines the 3D environment and camera:
- `cube_size`: bounding cube centered at origin
- `camera_position`, `camera_rotation`: camera extrinsics
- `focal_length`, `principal_point`, `image_size`: camera intrinsics

`world_to_image()` projects 3D points to 2D image coordinates.

## motion_simulation.py

Animates arm with random-walk velocities:
- 7 state variables: 3 position velocities + 4 angular velocities
- Each frame: sample random(-1, 0, 1) * step, add to velocity, clip to bounds
- Position clipped to [-1, 1], azimuth wraps, elevation reflects at ±π/2
