"""PyVista 3D visualization for arm pose estimation results.

Features:
  - Overlay video (with heatmaps + skeletons) on a 3D "screen" plane
  - Frame-by-frame stepping with keyboard (Left/Right) and slider
  - MediaPipe (green) vs Optimized (red) arm overlays in 3D
  - Camera frustum visualization

Usage:
  uv run python visualize_pyvista.py --video <path> [--run-dir <training_run>]

  Without --run-dir, only shows MediaPipe detections (raw video on screen).
  With --run-dir, also shows optimized results (overlay video on screen).
"""

import argparse
import json
import os

import cv2
import numpy as np
import pyvista as pv

from main import (
    ARM_INDICES,
    process_video,
    fit_cameras,
    estimate_segment_lengths,
    mediapipe_frame_to_arm,
)
from model.camera import Camera


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def make_arm_line(coords: dict[str, np.ndarray]) -> pv.PolyData:
    """Create straight line segments through shoulder -> elbow -> wrist."""
    pts = np.array([coords["a"], coords["b"], coords["c"]])
    lines = np.array([[2, 0, 1], [2, 1, 2]])
    return pv.PolyData(pts, lines=np.hstack(lines))


def video_frame_to_texture(frame_bgr: np.ndarray) -> pv.Texture:
    """Convert a BGR OpenCV frame to a PyVista texture, rotated 180 degrees about Z."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.flipud(frame_rgb).copy()
    return pv.numpy_to_texture(frame_rgb)


def make_video_plane(
    image_size: tuple[int, int],
    position: np.ndarray,
    normal: np.ndarray,
    scale: float = 0.5,
) -> pv.PolyData:
    """Create a textured plane sized to the video aspect ratio."""
    h, w = image_size
    aspect = w / h
    plane = pv.Plane(
        center=position,
        direction=normal,
        i_size=scale * aspect,
        j_size=scale,
    )
    plane.texture_map_to_plane(inplace=True)
    return plane


def make_camera_frustum(camera: Camera, depth: float = 0.15, scale: float = 1.0) -> pv.PolyData:
    """Create a wireframe camera frustum from a Camera object."""
    pos = camera.position
    R_inv = camera.rotation.T
    right = R_inv[:, 0]
    down = R_inv[:, 1]
    forward = R_inv[:, 2]

    h, w = camera.image_size
    fx, fy = camera.focal_length
    half_w = (w / 2) / fx * depth * scale
    half_h = (h / 2) / fy * depth * scale

    center_far = pos + forward * depth
    tl = center_far - right * half_w - down * half_h
    tr = center_far + right * half_w - down * half_h
    bl = center_far - right * half_w + down * half_h
    br = center_far + right * half_w + down * half_h

    points = np.array([pos, tl, tr, br, bl])
    lines = np.array([
        [2, 0, 1], [2, 0, 2], [2, 0, 3], [2, 0, 4],
        [2, 1, 2], [2, 2, 3], [2, 3, 4], [2, 4, 1],
    ])
    return pv.PolyData(points, lines=np.hstack(lines))


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------


class PoseVisualizer:
    def __init__(
        self,
        display_frames: list[np.ndarray],
        cameras: list[Camera],
        mp_coords: list[dict[str, np.ndarray]],
        opt_coords: list[dict[str, np.ndarray]] | None = None,
        cmu_gt_coords: list[dict[str, np.ndarray]] | None = None,
        image_size: tuple[int, int] = (480, 640),
    ):
        self.display_frames = display_frames
        self.cameras = cameras
        self.mp_coords = mp_coords
        self.opt_coords = opt_coords
        self.cmu_gt_coords = cmu_gt_coords
        self.image_size = image_size
        self.n_frames = len(display_frames)
        self.current_frame = 0
        self.playing = False

        # Compute scene center from median shoulder position
        shoulders = np.array([c["a"] for c in mp_coords])
        self.scene_center = np.median(shoulders, axis=0)

        # Place video screen behind the arm (offset along camera forward)
        cam0 = cameras[0]
        R_inv = cam0.rotation.T
        cam_forward = R_inv[:, 2]
        self.screen_pos = self.scene_center + cam_forward * 0.4
        self.screen_normal = cam_forward  # face toward viewer (back of camera)

        self._setup_plotter()

    def _setup_plotter(self):
        self.pl = pv.Plotter(title="Pose Estimation Viewer")
        self.pl.set_background("black")

        # --- Video screen ---
        self.video_plane = make_video_plane(
            self.image_size, self.screen_pos, self.screen_normal, scale=0.5
        )
        tex = video_frame_to_texture(self.display_frames[0])
        self.video_actor = self.pl.add_mesh(
            self.video_plane, texture=tex, name="video_screen", lighting=False
        )

        # --- MediaPipe arm (green) ---
        mp = self.mp_coords[0]
        self.pl.add_mesh(
            make_arm_line(mp), color="lime", line_width=4, name="mp_line"
        )
        for key in ("a", "b", "c"):
            self.pl.add_mesh(
                pv.Sphere(radius=0.008, center=mp[key]),
                color="lime",
                name=f"mp_joint_{key}",
            )

        # --- Optimized arm (red), if available ---
        if self.opt_coords is not None:
            opt = self.opt_coords[0]
            self.pl.add_mesh(
                make_arm_line(opt), color="red", line_width=4, name="opt_line"
            )
            for key in ("a", "b", "c"):
                self.pl.add_mesh(
                    pv.Sphere(radius=0.008, center=opt[key]),
                    color="red",
                    name=f"opt_joint_{key}",
                )

        # --- CMU GT arm (yellow), if available ---
        if self.cmu_gt_coords is not None:
            gt = self.cmu_gt_coords[0]
            self.pl.add_mesh(
                make_arm_line(gt), color="yellow", line_width=4, name="gt_line"
            )
            for key in ("a", "b", "c"):
                self.pl.add_mesh(
                    pv.Sphere(radius=0.008, center=gt[key]),
                    color="yellow",
                    name=f"gt_joint_{key}",
                )

        # --- Camera frustum ---
        frustum = make_camera_frustum(self.cameras[0])
        self.pl.add_mesh(frustum, color="yellow", line_width=2, name="frustum")
        self.pl.add_mesh(
            pv.Sphere(radius=0.005, center=self.cameras[0].position),
            color="yellow",
            name="cam_sphere",
        )

        # --- Controls ---
        self.pl.add_slider_widget(
            self._on_slider,
            rng=[0, self.n_frames - 1],
            value=0,
            title="Frame",
            fmt="%.0f",
            pointa=(0.1, 0.05),
            pointb=(0.9, 0.05),
            style="modern",
        )

        self.pl.add_key_event("Right", self._next_frame)
        self.pl.add_key_event("Left", self._prev_frame)
        self.pl.add_key_event("space", self._toggle_play)

        self.pl.add_text(
            "Left/Right: step | Space: play/pause",
            position="upper_left",
            font_size=10,
            color="white",
            name="help_text",
        )
        self.pl.add_text(
            f"Frame 0 / {self.n_frames - 1}",
            position="upper_right",
            font_size=10,
            color="white",
            name="frame_label",
        )

        # Set initial camera view
        self.pl.camera.position = (
            self.scene_center[0] - 0.5,
            self.scene_center[1] - 0.5,
            self.scene_center[2] + 0.3,
        )
        self.pl.camera.focal_point = tuple(self.scene_center)
        self.pl.camera.up = (0, 0, 1)

    def _update_frame(self, idx: int):
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        self.current_frame = idx

        # Update video texture
        self.video_actor.texture = video_frame_to_texture(self.display_frames[idx])

        # Update MediaPipe arm
        mp = self.mp_coords[idx]
        self.pl.add_mesh(
            make_arm_line(mp), color="lime", line_width=4, name="mp_line"
        )
        for key in ("a", "b", "c"):
            self.pl.add_mesh(
                pv.Sphere(radius=0.008, center=mp[key]),
                color="lime",
                name=f"mp_joint_{key}",
            )

        # Update optimized arm
        if self.opt_coords is not None:
            opt = self.opt_coords[idx]
            self.pl.add_mesh(
                make_arm_line(opt), color="red", line_width=4, name="opt_line"
            )
            for key in ("a", "b", "c"):
                self.pl.add_mesh(
                    pv.Sphere(radius=0.008, center=opt[key]),
                    color="red",
                    name=f"opt_joint_{key}",
                )

        # Update CMU GT arm
        if self.cmu_gt_coords is not None and idx < len(self.cmu_gt_coords):
            gt = self.cmu_gt_coords[idx]
            self.pl.add_mesh(
                make_arm_line(gt), color="yellow", line_width=4, name="gt_line"
            )
            for key in ("a", "b", "c"):
                self.pl.add_mesh(
                    pv.Sphere(radius=0.008, center=gt[key]),
                    color="yellow",
                    name=f"gt_joint_{key}",
                )

        # Update camera frustum
        cam = self.cameras[idx]
        self.pl.add_mesh(
            make_camera_frustum(cam), color="yellow", line_width=2, name="frustum"
        )
        self.pl.add_mesh(
            pv.Sphere(radius=0.005, center=cam.position),
            color="yellow", name="cam_sphere",
        )

        # Update label
        self.pl.add_text(
            f"Frame {idx} / {self.n_frames - 1}",
            position="upper_right",
            font_size=10,
            color="white",
            name="frame_label",
        )

        self.pl.render()

    def _on_slider(self, value):
        self._update_frame(int(value))

    def _next_frame(self):
        self._update_frame(self.current_frame + 1)

    def _prev_frame(self):
        self._update_frame(self.current_frame - 1)

    def _toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.pl.add_timer_event(
                max_steps=self.n_frames,
                duration=100,
                callback=self._auto_step,
            )

    def _auto_step(self, step):
        if self.playing and self.current_frame < self.n_frames - 1:
            self._update_frame(self.current_frame + 1)
        else:
            self.playing = False

    def show(self):
        self.pl.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="PyVista 3D visualization for pose estimation"
    )
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Target FPS for processing"
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Path to a training_runs/ directory with optimization results",
    )
    parser.add_argument(
        "--cmu-gt",
        default=None,
        help="Path to CMU Panoptic hdPose3d_stage1_coco19/ directory for ground truth overlay",
    )
    parser.add_argument(
        "--cmu-calib",
        default=None,
        help="Path to CMU Panoptic calibration JSON (required with --cmu-gt)",
    )
    parser.add_argument(
        "--cmu-camera",
        default="00_00",
        help="CMU camera name to align GT coordinates (default: 00_00)",
    )
    args = parser.parse_args()

    # --- Process video ---
    print("=== Processing Video ===")
    frames, raw_bgr_frames, image_size, frame_skip = process_video(args.video, target_fps=args.fps)
    if len(frames) < 2:
        print("Need at least 2 frames.")
        return

    # --- Fit cameras ---
    print("=== Fitting Cameras ===")
    cameras = fit_cameras(frames, image_size)

    # --- MediaPipe 3D coords ---
    from main import SHOULDER_IDX, ELBOW_IDX, WRIST_IDX

    mp_coords = []
    for frame in frames:
        mp_coords.append({
            "a": frame.landmarks_3d[SHOULDER_IDX].copy(),
            "b": frame.landmarks_3d[ELBOW_IDX].copy(),
            "c": frame.landmarks_3d[WRIST_IDX].copy(),
        })

    # --- Optionally run optimization and generate overlay frames ---
    opt_coords = None
    display_frames = raw_bgr_frames  # default: raw video

    if args.run_dir and os.path.isdir(args.run_dir):
        print("=== Running Optimization (for visualization) ===")
        from scoring import OptimizationConfig
        from optimize import run_optimization
        from main import generate_frame_heatmaps, generate_overlay_frames

        a_b_length, b_c_length = estimate_segment_lengths(frames)
        initial_arms = [mediapipe_frame_to_arm(f, a_b_length, b_c_length) for f in frames]

        config_path = os.path.join(args.run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = OptimizationConfig(**json.load(f))
        else:
            config = OptimizationConfig()

        all_heatmaps = [
            generate_frame_heatmaps(frame, ARM_INDICES, image_size, sigma=50.0)
            for frame in frames
        ]

        result = run_optimization(
            initial_arms, all_heatmaps, cameras, a_b_length, b_c_length, config
        )
        opt_coords = result.optimized_coords
        display_frames = generate_overlay_frames(
            raw_bgr_frames, frames, all_heatmaps, result, cameras, image_size
        )

    # --- Load CMU ground truth if provided ---
    cmu_gt_coords = None
    if args.cmu_gt:
        from main import load_cmu_gt, load_cmu_camera_rotation
        print("=== Loading CMU Ground Truth ===")
        cam_rot = None
        if args.cmu_calib:
            cam_rot = load_cmu_camera_rotation(args.cmu_calib, args.cmu_camera)
        cmu_gt_coords = load_cmu_gt(args.cmu_gt, len(frames), frame_skip, cam_rot)

    # --- Launch visualizer ---
    print(f"=== Launching PyVista Viewer ({len(frames)} frames) ===")
    vis = PoseVisualizer(
        display_frames=display_frames,
        cameras=cameras,
        mp_coords=mp_coords,
        opt_coords=opt_coords,
        cmu_gt_coords=cmu_gt_coords,
        image_size=image_size,
    )
    vis.show()


if __name__ == "__main__":
    main()
