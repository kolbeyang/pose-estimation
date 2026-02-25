"""VPython visualization for Anatomy3D-style human pose with bone direction/length sliders."""

import numpy as np
from vpython import (
    canvas, sphere, curve, box, vector, color,
    slider, winput, button,
)

from anatomy_3d.human_pose_anatomy_3d import (
    HumanPoseAnatomy3D, JOINT_NAMES, PARENTS, BODY_GROUPS, NUM_JOINTS,
    NUM_BONES, BONE_INDICES, direction_to_spherical, spherical_to_direction,
)


def to_vpython(arr: np.ndarray) -> vector:
    """Convert numpy array to vpython vector."""
    return vector(float(arr[0]), float(arr[1]), float(arr[2]))


# Colors per body group
GROUP_COLORS = {
    "Root": color.white,
    "Spine": color.white,
    "Left Arm": vector(0.3, 0.5, 1.0),
    "Right Arm": vector(1.0, 0.3, 0.3),
    "Left Leg": vector(0.3, 0.8, 0.8),
    "Right Leg": vector(1.0, 0.6, 0.2),
}

JOINT_COLORS = {}
for group_name, joint_indices in BODY_GROUPS.items():
    for ji in joint_indices:
        JOINT_COLORS[ji] = GROUP_COLORS[group_name]


class Anatomy3DVisualizer:
    def __init__(self, pose: HumanPoseAnatomy3D):
        self.pose = pose
        self._controls = {}

        self.scene = canvas(
            title="Anatomy3D Pose Visualizer (Bone Directions + Lengths)",
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

        positions = self.pose.compute_positions()

        # Joint spheres
        self.joint_spheres = []
        for i in range(NUM_JOINTS):
            c = JOINT_COLORS.get(i, color.white)
            s = sphere(
                pos=to_vpython(positions[i]),
                radius=0.025 if i == 0 else 0.018,
                color=c,
            )
            self.joint_spheres.append(s)

        # Bone curves
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

        self._build_ui()

    def _build_ui(self):
        sc = self.scene

        # Root position
        sc.append_to_caption("\n<b>Root Position</b>\n")
        for axis_i, axis_name in enumerate(["X", "Y", "Z"]):
            key = f"root_pos_{axis_i}"
            val = self.pose.root_position[axis_i]
            self._add_slider_row(sc, f"  {axis_name}:", key, -2.0, 2.0, val, 0.01)

        # Bone directions (azimuth + elevation per bone)
        bone_groups = {
            "Right Leg": [0, 1, 2],     # RHip, RKnee, RAnkle
            "Left Leg": [3, 4, 5],      # LHip, LKnee, LAnkle
            "Spine": [6, 7, 8, 9],      # Spine, Thorax, Neck, Head
            "Left Arm": [10, 11, 12],   # LShoulder, LElbow, LWrist
            "Right Arm": [13, 14, 15],  # RShoulder, RElbow, RWrist
        }

        for group_name in ["Spine", "Left Leg", "Right Leg", "Left Arm", "Right Arm"]:
            bone_idxs = bone_groups[group_name]
            sc.append_to_caption(f"\n<b>{group_name} Bone Directions (deg)</b>\n")
            for bi in bone_idxs:
                parent_idx, child_idx = BONE_INDICES[bi]
                label = f"{JOINT_NAMES[parent_idx]}->{JOINT_NAMES[child_idx]}"
                az, el = direction_to_spherical(self.pose.bone_directions[bi])
                az_deg, el_deg = np.degrees(az), np.degrees(el)

                key_az = f"bone_dir_az_{bi}"
                key_el = f"bone_dir_el_{bi}"
                self._add_slider_row(sc, f"  {label} Az:", key_az, -180, 180, az_deg, 1.0)
                self._add_slider_row(sc, f"  {label} El:", key_el, -90, 90, el_deg, 1.0)

        # Bone lengths
        sc.append_to_caption("\n<b>Bone Lengths (m)</b>\n")
        for bi in range(NUM_BONES):
            parent_idx, child_idx = BONE_INDICES[bi]
            label = f"{JOINT_NAMES[parent_idx]}->{JOINT_NAMES[child_idx]}"
            key = f"bone_len_{bi}"
            val = self.pose.bone_lengths[bi]
            self._add_slider_row(sc, f"  {label}:", key, 0.01, 1.0, val, 0.01)

        sc.append_to_caption("\n\n")
        button(bind=self._on_reset, text="Reset to Default")

    def _add_slider_row(self, sc, label, key, min_val, max_val, default, step):
        sc.append_to_caption(f"{label:<30s}")
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
        if key.startswith("root_pos_"):
            axis = int(key[-1])
            self.pose.root_position[axis] = value
        elif key.startswith("bone_dir_az_") or key.startswith("bone_dir_el_"):
            parts = key.split("_")
            bi = int(parts[-1])
            # Get current azimuth and elevation
            az_key = f"bone_dir_az_{bi}"
            el_key = f"bone_dir_el_{bi}"
            az_sl, _ = self._controls[az_key]
            el_sl, _ = self._controls[el_key]
            az_rad = np.radians(az_sl.value)
            el_rad = np.radians(el_sl.value)
            self.pose.bone_directions[bi] = spherical_to_direction(az_rad, el_rad)
        elif key.startswith("bone_len_"):
            bi = int(key.split("_")[-1])
            self.pose.bone_lengths[bi] = value

    def _update_visuals(self):
        positions = self.pose.compute_positions()
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
        default = HumanPoseAnatomy3D.default_standing()
        self.pose.root_position[:] = default.root_position
        self.pose.bone_directions[:] = default.bone_directions
        self.pose.bone_lengths[:] = default.bone_lengths

        for key, (sl, wi) in self._controls.items():
            if key.startswith("root_pos_"):
                axis = int(key[-1])
                val = self.pose.root_position[axis]
            elif key.startswith("bone_dir_az_"):
                bi = int(key.split("_")[-1])
                az, _ = direction_to_spherical(self.pose.bone_directions[bi])
                val = np.degrees(az)
            elif key.startswith("bone_dir_el_"):
                bi = int(key.split("_")[-1])
                _, el = direction_to_spherical(self.pose.bone_directions[bi])
                val = np.degrees(el)
            elif key.startswith("bone_len_"):
                bi = int(key.split("_")[-1])
                val = self.pose.bone_lengths[bi]
            else:
                continue
            sl.value = val
            wi.text = f"{val:.2f}"

        self._update_visuals()
