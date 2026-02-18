"""VPython visualization for human body pose with interactive slider controls."""

import numpy as np
from vpython import (
    canvas, sphere, curve, box, vector, color,
    slider, winput, button,
)

from human_pose import (
    HumanPose, JOINT_NAMES, PARENTS, BODY_GROUPS, NUM_JOINTS,
)


def to_vpython(arr: np.ndarray) -> vector:
    """Convert numpy array to vpython vector."""
    return vector(float(arr[0]), float(arr[1]), float(arr[2]))


# Colors per body group
GROUP_COLORS = {
    "Root": color.white,
    "Spine": color.white,
    "Left Arm": vector(0.3, 0.5, 1.0),    # blue
    "Right Arm": vector(1.0, 0.3, 0.3),    # red
    "Left Leg": vector(0.3, 0.8, 0.8),     # cyan
    "Right Leg": vector(1.0, 0.6, 0.2),    # orange
}

# Map joint index to its group color
JOINT_COLORS = {}
for group_name, joint_indices in BODY_GROUPS.items():
    for ji in joint_indices:
        JOINT_COLORS[ji] = GROUP_COLORS[group_name]


class HumanVisualizer:
    def __init__(self, pose: HumanPose):
        self.pose = pose
        self._controls = {}  # param_key -> (slider, winput)

        # Create scene
        self.scene = canvas(
            title="Human Body Pose Visualizer",
            width=1000,
            height=700,
            center=vector(0, 0.85, 0),
            background=color.gray(0.15),
        )
        self.scene.up = vector(0, 1, 0)
        self.scene.forward = vector(-1, -0.3, -0.8)
        self.scene.range = 1.2

        # Ground plane
        box(
            pos=vector(0, -0.01, 0),
            size=vector(3, 0.02, 3),
            color=color.gray(0.3),
            opacity=0.5,
        )

        # Compute initial positions
        positions = self.pose.forward_kinematics()

        # Create joint spheres
        self.joint_spheres = []
        for i in range(NUM_JOINTS):
            c = JOINT_COLORS.get(i, color.white)
            s = sphere(
                pos=to_vpython(positions[i]),
                radius=0.025 if i == 0 else 0.018,
                color=c,
            )
            self.joint_spheres.append(s)

        # Create bone curves (index 0 is a placeholder)
        self.bone_curves: list[curve | None] = [None]
        for i in range(1, NUM_JOINTS):
            parent = PARENTS[i]
            c = JOINT_COLORS.get(i, color.white)
            crv = curve(
                pos=[to_vpython(positions[parent]), to_vpython(positions[i])],
                radius=0.006,
                color=c,
            )
            self.bone_curves.append(crv)

        # Build slider UI
        self._build_ui()

    def _build_ui(self):
        """Build all slider controls organized by body group."""
        sc = self.scene

        # Root position
        sc.append_to_caption("\n<b>Root Position</b>\n")
        for axis_i, axis_name in enumerate(["X", "Y", "Z"]):
            key = f"root_pos_{axis_i}"
            val = self.pose.root_position[axis_i]
            self._add_slider_row(sc, f"  {axis_name}:", key, -2.0, 2.0, val, 0.01)

        # Root rotation
        sc.append_to_caption("\n<b>Root Rotation (deg)</b>\n")
        for axis_i, axis_name in enumerate(["Rx", "Ry", "Rz"]):
            key = f"root_rot_{axis_i}"
            val_deg = np.degrees(self.pose.root_rotation[axis_i])
            self._add_slider_row(sc, f"  {axis_name}:", key, -180, 180, val_deg, 1.0)

        # Joint rotations by body group (skip Root group for rotations)
        for group_name in ["Spine", "Left Leg", "Right Leg", "Left Arm", "Right Arm"]:
            joint_indices = BODY_GROUPS[group_name]
            sc.append_to_caption(f"\n<b>{group_name} Rotations (deg)</b>\n")
            for ji in joint_indices:
                jname = JOINT_NAMES[ji]
                for axis_i, axis_name in enumerate(["Rx", "Ry", "Rz"]):
                    key = f"joint_{ji}_{axis_i}"
                    val_deg = np.degrees(self.pose.local_rotations[ji][axis_i])
                    self._add_slider_row(
                        sc, f"  {jname} {axis_name}:", key,
                        -180, 180, val_deg, 1.0,
                    )

        # Bone lengths
        sc.append_to_caption("\n<b>Bone Lengths (m)</b>\n")
        for ji in range(1, NUM_JOINTS):
            jname = JOINT_NAMES[ji]
            parent_name = JOINT_NAMES[PARENTS[ji]]
            key = f"bone_{ji}"
            val = self.pose.bone_lengths[ji]
            self._add_slider_row(
                sc, f"  {parent_name}->{jname}:", key,
                0.01, 1.0, val, 0.01,
            )

        # Reset button
        sc.append_to_caption("\n\n")
        button(bind=self._on_reset, text="Reset to Default")

    def _add_slider_row(self, sc, label, key, min_val, max_val, default, step):
        """Add a labeled slider + numeric input row."""
        sc.append_to_caption(f"{label:<25s}")
        sl = slider(
            min=min_val, max=max_val, value=default, step=step,
            length=250, bind=self._make_slider_handler(key),
        )
        sc.append_to_caption("  ")
        wi = winput(
            bind=self._make_winput_handler(key),
            type="numeric",
            text=f"{default:.2f}",
            width=70,
        )
        sc.append_to_caption("\n")
        self._controls[key] = (sl, wi)

    def _make_slider_handler(self, key):
        def handler(evt):
            sl, wi = self._controls[key]
            wi.text = f"{sl.value:.2f}"
            self._apply_param(key, sl.value)
            self._update_visuals()
        return handler

    def _make_winput_handler(self, key):
        def handler(evt):
            sl, wi = self._controls[key]
            try:
                val = float(evt.text)
            except (ValueError, TypeError):
                return
            val = max(sl.min, min(sl.max, val))
            sl.value = val
            wi.text = f"{val:.2f}"
            self._apply_param(key, val)
            self._update_visuals()
        return handler

    def _apply_param(self, key: str, value: float):
        """Write a slider value back to the pose parameters."""
        if key.startswith("root_pos_"):
            axis = int(key[-1])
            self.pose.root_position[axis] = value
        elif key.startswith("root_rot_"):
            axis = int(key[-1])
            self.pose.root_rotation[axis] = np.radians(value)
        elif key.startswith("joint_"):
            parts = key.split("_")
            ji = int(parts[1])
            axis = int(parts[2])
            self.pose.local_rotations[ji][axis] = np.radians(value)
        elif key.startswith("bone_"):
            ji = int(key.split("_")[1])
            self.pose.bone_lengths[ji] = value

    def _update_visuals(self):
        """Recompute FK and update all VPython objects."""
        positions = self.pose.forward_kinematics()
        for i in range(NUM_JOINTS):
            self.joint_spheres[i].pos = to_vpython(positions[i])
        for i in range(1, NUM_JOINTS):
            parent = PARENTS[i]
            crv = self.bone_curves[i]
            assert crv is not None
            crv.clear()
            crv.append(to_vpython(positions[parent]))
            crv.append(to_vpython(positions[i]))

    def _on_reset(self, evt):
        """Reset to default standing pose."""
        default = HumanPose.default_standing()
        self.pose.root_position[:] = default.root_position
        self.pose.root_rotation[:] = default.root_rotation
        self.pose.local_rotations[:] = default.local_rotations
        self.pose.bone_lengths[:] = default.bone_lengths

        # Update all slider/winput controls
        for key, (sl, wi) in self._controls.items():
            if key.startswith("root_pos_"):
                axis = int(key[-1])
                val = self.pose.root_position[axis]
            elif key.startswith("root_rot_"):
                axis = int(key[-1])
                val = np.degrees(self.pose.root_rotation[axis])
            elif key.startswith("joint_"):
                parts = key.split("_")
                ji, axis = int(parts[1]), int(parts[2])
                val = np.degrees(self.pose.local_rotations[ji][axis])
            elif key.startswith("bone_"):
                ji = int(key.split("_")[1])
                val = self.pose.bone_lengths[ji]
            else:
                continue
            sl.value = val
            wi.text = f"{val:.2f}"

        self._update_visuals()
