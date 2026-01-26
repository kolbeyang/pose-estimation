GlowScript 3.2 VPython

# Scene setup
scene.background = color.gray(0.2)
scene.up = vec(0, 0, 1)
scene.forward = vec(-1, -0.5, -1)
scene.range = 12
scene.center = vec(0, 0, 0)

# Environment cube (size 10, white, transparent)
box(pos=vec(0, 0, 0), size=vec(10, 10, 10), color=color.white, opacity=0.1)

# Camera position (red sphere)
sphere(pos=vec(8, 0, 0), radius=0.3, color=color.red, opacity=0.2)

# Frame data as simple lists: [a, b, c] for each frame
# Ground truth (green)
gt_a = [vec(-0.04, -0.04, 0.0), vec(-0.04, -0.08, 0.04), vec(-0.08, -0.08, 0.04), vec(-0.16, -0.04, 0.0), vec(-0.2, 0.0, -0.04), vec(-0.28, 0.08, -0.04), vec(-0.36, 0.16, -0.04), vec(-0.44, 0.28, 0.0), vec(-0.48, 0.44, 0.04), vec(-0.52, 0.6, 0.12), vec(-0.56, 0.72, 0.16), vec(-0.56, 0.88, 0.24), vec(-0.6, 1.0, 0.36), vec(-0.6, 1.0, 0.48), vec(-0.64, 1.0, 0.64), vec(-0.68, 1.0, 0.8), vec(-0.68, 1.0, 1.0), vec(-0.68, 1.0, 1.0), vec(-0.68, 1.0, 1.0), vec(-0.68, 1.0, 1.0)]
gt_b = [vec(1.306, -1.269, 0.824), vec(1.278, -1.384, 0.790), vec(1.165, -1.526, 0.640), vec(0.926, -1.659, 0.446), vec(0.698, -1.764, 0.248), vec(0.407, -1.798, 0.009), vec(0.097, -1.784, -0.151), vec(-0.220, -1.699, -0.190), vec(-0.498, -1.529, -0.309), vec(-0.842, -1.286, -0.462), vec(-1.091, -1.063, -0.573), vec(-1.288, -0.799, -0.567), vec(-1.548, -0.482, -0.591), vec(-1.797, -0.236, -0.540), vec(-2.023, 0.050, -0.448), vec(-2.183, 0.362, -0.354), vec(-2.234, 0.683, -0.219), vec(-2.163, 0.998, -0.342), vec(-2.080, 1.282, -0.400), vec(-1.944, 1.532, -0.456)]
gt_c = [vec(0.857, -3.172, 1.246), vec(0.732, -3.294, 1.018), vec(0.448, -3.393, 0.631), vec(-0.012, -3.398, 0.132), vec(-0.404, -3.349, -0.274), vec(-0.819, -3.224, -0.671), vec(-1.255, -3.059, -0.891), vec(-1.690, -2.763, -1.031), vec(-1.985, -2.391, -1.333), vec(-2.284, -1.820, -1.742), vec(-2.405, -1.383, -2.046), vec(-2.457, -0.992, -2.179), vec(-2.574, -0.478, -2.308), vec(-2.642, -0.123, -2.349), vec(-2.687, 0.298, -2.319), vec(-2.619, 0.627, -2.288), vec(-2.417, 0.963, -2.191), vec(-2.031, 1.133, -2.333), vec(-1.789, 1.233, -2.378), vec(-1.621, 1.337, -2.420)]

# Predicted (red)
pred_a = [vec(0.108, -0.0004, -0.151), vec(0.200, -0.214, 0.068), vec(0.078, -0.116, -0.065), vec(0.196, -0.149, -0.083), vec(0.021, 0.139, -0.027), vec(-0.012, -0.164, -0.086), vec(-0.339, -0.007, 0.094), vec(-0.248, 0.236, -0.215), vec(-0.382, 0.340, -0.110), vec(-0.650, 0.523, -0.251), vec(-0.238, 0.495, 0.098), vec(-0.468, 0.673, 0.028), vec(-0.276, 0.847, 0.305), vec(-0.515, 0.943, 0.281), vec(-0.249, 0.996, 0.358), vec(-0.013, 0.801, 0.700), vec(0.197, 0.940, 0.868), vec(0.523, 0.712, 0.832), vec(0.771, 0.959, 0.947), vec(0.589, 0.878, 0.781)]
pred_b = [vec(1.630, -1.015, 0.658), vec(1.434, -1.437, 1.059), vec(1.144, -1.639, 0.674), vec(1.266, -1.773, 0.386), vec(1.304, -1.367, 0.264), vec(0.935, -1.924, -0.006), vec(1.129, -1.362, -0.001), vec(0.853, -1.431, -0.135), vec(0.996, -1.109, -0.141), vec(0.912, -0.717, -0.409), vec(0.958, -1.106, 0.024), vec(1.080, -0.560, -0.258), vec(1.274, -0.276, -0.277), vec(1.023, -0.293, -0.048), vec(1.309, 0.006, -0.411), vec(1.644, 0.060, -0.140), vec(1.800, 0.488, -0.238), vec(2.216, 0.579, -0.224), vec(2.369, 0.684, -0.224), vec(2.143, 1.060, -0.464)]
pred_c = [vec(1.804, -2.981, 0.981), vec(0.949, -3.376, 1.107), vec(0.232, -3.418, 0.622), vec(0.364, -3.524, 0.041), vec(0.499, -3.148, -0.162), vec(-0.501, -3.220, -0.513), vec(-0.036, -2.727, -0.884), vec(-0.748, -2.490, -0.697), vec(-0.350, -2.174, -1.167), vec(-0.701, -1.592, -1.202), vec(-0.471, -1.353, -1.353), vec(-0.302, -0.777, -1.688), vec(-0.168, -0.495, -1.645), vec(-0.339, -0.172, -1.507), vec(-0.243, 0.250, -1.649), vec(0.257, 0.362, -1.550), vec(0.094, 0.643, -1.270), vec(0.884, 0.722, -1.709), vec(0.912, 0.990, -1.559), vec(0.608, 0.890, -1.734)]

num_frames = 20

# Create ground truth arm (green) - sphere sizes: 0.4, 0.3, 0.2
gt_sphere_a = sphere(pos=gt_a[0], radius=0.4, color=color.green)
gt_sphere_b = sphere(pos=gt_b[0], radius=0.3, color=color.green)
gt_sphere_c = sphere(pos=gt_c[0], radius=0.2, color=color.green)
gt_curve = curve(color=color.green, radius=0.05)
gt_curve.append(gt_a[0])
gt_curve.append(gt_b[0])
gt_curve.append(gt_c[0])

# Create predicted arm (red) - sphere sizes: 0.4, 0.3, 0.2
pred_sphere_a = sphere(pos=pred_a[0], radius=0.4, color=color.red)
pred_sphere_b = sphere(pos=pred_b[0], radius=0.3, color=color.red)
pred_sphere_c = sphere(pos=pred_c[0], radius=0.2, color=color.red)
pred_curve = curve(color=color.red, radius=0.05)
pred_curve.append(pred_a[0])
pred_curve.append(pred_b[0])
pred_curve.append(pred_c[0])

# Legend
scene.append_to_caption("\n\nGreen = Ground Truth\nRed = Predicted\n")

# Animation loop
t = 0
dt = 1

while True:
    rate(6)

    frame_idx = int(t) % num_frames

    # Update ground truth positions
    gt_sphere_a.pos = gt_a[frame_idx]
    gt_sphere_b.pos = gt_b[frame_idx]
    gt_sphere_c.pos = gt_c[frame_idx]
    gt_curve.modify(0, pos=gt_a[frame_idx])
    gt_curve.modify(1, pos=gt_b[frame_idx])
    gt_curve.modify(2, pos=gt_c[frame_idx])

    # Update predicted positions
    pred_sphere_a.pos = pred_a[frame_idx]
    pred_sphere_b.pos = pred_b[frame_idx]
    pred_sphere_c.pos = pred_c[frame_idx]
    pred_curve.modify(0, pos=pred_a[frame_idx])
    pred_curve.modify(1, pos=pred_b[frame_idx])
    pred_curve.modify(2, pos=pred_c[frame_idx])

    t = t + dt
