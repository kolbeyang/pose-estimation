"""VPython visualization for MediaPipe arm poses."""

import numpy as np
import torch
from vpython import canvas, box, sphere, curve, vector, color, rate, checkbox

from model.camera import Camera
from model.environment import Environment


def to_vpython(arr: np.ndarray | torch.Tensor) -> vector:
    """Convert numpy array or torch tensor to vpython vector."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()
    return vector(float(arr[0]), float(arr[1]), float(arr[2]))


ARM_COLORS = [
    color.green,   # MediaPipe
    color.red,     # Optimized
    color.yellow,  # CMU GT
]

ARM_LABELS = [
    "MediaPipe",
    "Optimized",
    "CMU GT",
]


class Visualizer:
    def __init__(self, env: Environment, coords_list: list[dict], camera: Camera | None = None, fps: int = 10):
        """
        Args:
            env: Environment object defining the scene bounds
            coords_list: List of coordinate dicts, each with keys "a", "b", "c"
            camera: Optional Camera object to display as transparent red sphere
            fps: Frames per second for animation
        """
        self.fps = fps
        self.scene = canvas(
            title="MediaPipe (green) vs Optimized (red)",
            width=800,
            height=600,
            center=vector(0, 0, 0),
            background=color.gray(0.2),
        )
        self.scene.up = vector(0, 0, 1)
        self.scene.forward = vector(-1, -0.5, -1)
        self.scene.range = 1.0

        # Camera position
        if camera is not None:
            camera_pos = to_vpython(camera.position)
            sphere(pos=camera_pos, radius=0.02, color=color.red, opacity=0.2)

        # Environment cube
        box(
            pos=vector(0, 0, 0),
            size=vector(env.cube_size, env.cube_size, env.cube_size),
            color=color.white,
            opacity=0.1,
        )

        # Create visualization objects for each arm
        self.arms = []
        self.arm_visible = []

        for i, coords in enumerate(coords_list):
            arm_color = ARM_COLORS[i % len(ARM_COLORS)]

            sphere_a = sphere(
                pos=to_vpython(coords["a"]), radius=0.02, color=arm_color
            )
            sphere_b = sphere(
                pos=to_vpython(coords["b"]), radius=0.015, color=arm_color
            )
            sphere_c = sphere(
                pos=to_vpython(coords["c"]), radius=0.01, color=arm_color
            )
            arm_curve = curve(
                pos=[
                    to_vpython(coords["a"]),
                    to_vpython(coords["b"]),
                    to_vpython(coords["c"]),
                ],
                radius=0.003,
                color=arm_color,
            )
            self.arms.append((sphere_a, sphere_b, sphere_c, arm_curve))
            self.arm_visible.append(True)

        # Create toggle checkboxes
        self.scene.append_to_caption("\n\nToggle Arms:\n")
        for i in range(len(coords_list)):
            arm_label = ARM_LABELS[i] if i < len(ARM_LABELS) else f"Arm {i}"
            checkbox(bind=self._make_toggle_handler(i), text=arm_label, checked=True)
            self.scene.append_to_caption("  ")

    def _make_toggle_handler(self, arm_index: int):
        def handler(evt):
            self._set_arm_visible(arm_index, evt.checked)
        return handler

    def _set_arm_visible(self, arm_index: int, visible: bool):
        if arm_index >= len(self.arms):
            return
        self.arm_visible[arm_index] = visible
        sphere_a, sphere_b, sphere_c, arm_curve = self.arms[arm_index]
        sphere_a.visible = visible
        sphere_b.visible = visible
        sphere_c.visible = visible
        arm_curve.visible = visible

    def update(self, coords_list: list[dict]):
        """Update positions for all arms."""
        for i, coords in enumerate(coords_list):
            if i >= len(self.arms):
                break
            if not self.arm_visible[i]:
                continue

            sphere_a, sphere_b, sphere_c, arm_curve = self.arms[i]
            sphere_a.pos = to_vpython(coords["a"])
            sphere_b.pos = to_vpython(coords["b"])
            sphere_c.pos = to_vpython(coords["c"])
            arm_curve.clear()
            arm_curve.append(to_vpython(coords["a"]))
            arm_curve.append(to_vpython(coords["b"]))
            arm_curve.append(to_vpython(coords["c"]))
