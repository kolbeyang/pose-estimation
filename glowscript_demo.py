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

# Frame data: ground truth (green) and predicted (red)
gt_frames = [
    {"a": vec(-0.04, -0.04, 0.0), "b": vec(1.306, -1.269, 0.824), "c": vec(0.857, -3.172, 1.246)},
    {"a": vec(-0.04, -0.08, 0.04), "b": vec(1.278, -1.384, 0.790), "c": vec(0.732, -3.294, 1.018)},
    {"a": vec(-0.08, -0.08, 0.04), "b": vec(1.165, -1.526, 0.640), "c": vec(0.448, -3.393, 0.631)},
    {"a": vec(-0.16, -0.04, 0.0), "b": vec(0.926, -1.659, 0.446), "c": vec(-0.012, -3.398, 0.132)},
    {"a": vec(-0.2, 0.0, -0.04), "b": vec(0.698, -1.764, 0.248), "c": vec(-0.404, -3.349, -0.274)},
    {"a": vec(-0.28, 0.08, -0.04), "b": vec(0.407, -1.798, 0.009), "c": vec(-0.819, -3.224, -0.671)},
    {"a": vec(-0.36, 0.16, -0.04), "b": vec(0.097, -1.784, -0.151), "c": vec(-1.255, -3.059, -0.891)},
    {"a": vec(-0.44, 0.28, 0.0), "b": vec(-0.220, -1.699, -0.190), "c": vec(-1.690, -2.763, -1.031)},
    {"a": vec(-0.48, 0.44, 0.04), "b": vec(-0.498, -1.529, -0.309), "c": vec(-1.985, -2.391, -1.333)},
    {"a": vec(-0.52, 0.6, 0.12), "b": vec(-0.842, -1.286, -0.462), "c": vec(-2.284, -1.820, -1.742)},
    {"a": vec(-0.56, 0.72, 0.16), "b": vec(-1.091, -1.063, -0.573), "c": vec(-2.405, -1.383, -2.046)},
    {"a": vec(-0.56, 0.88, 0.24), "b": vec(-1.288, -0.799, -0.567), "c": vec(-2.457, -0.992, -2.179)},
    {"a": vec(-0.6, 1.0, 0.36), "b": vec(-1.548, -0.482, -0.591), "c": vec(-2.574, -0.478, -2.308)},
    {"a": vec(-0.6, 1.0, 0.48), "b": vec(-1.797, -0.236, -0.540), "c": vec(-2.642, -0.123, -2.349)},
    {"a": vec(-0.64, 1.0, 0.64), "b": vec(-2.023, 0.050, -0.448), "c": vec(-2.687, 0.298, -2.319)},
    {"a": vec(-0.68, 1.0, 0.8), "b": vec(-2.183, 0.362, -0.354), "c": vec(-2.619, 0.627, -2.288)},
    {"a": vec(-0.68, 1.0, 1.0), "b": vec(-2.234, 0.683, -0.219), "c": vec(-2.417, 0.963, -2.191)},
    {"a": vec(-0.68, 1.0, 1.0), "b": vec(-2.163, 0.998, -0.342), "c": vec(-2.031, 1.133, -2.333)},
    {"a": vec(-0.68, 1.0, 1.0), "b": vec(-2.080, 1.282, -0.400), "c": vec(-1.789, 1.233, -2.378)},
    {"a": vec(-0.68, 1.0, 1.0), "b": vec(-1.944, 1.532, -0.456), "c": vec(-1.621, 1.337, -2.420)},
]

pred_frames = [
    {"a": vec(0.108, -0.0004, -0.151), "b": vec(1.630, -1.015, 0.658), "c": vec(1.804, -2.981, 0.981)},
    {"a": vec(0.200, -0.214, 0.068), "b": vec(1.434, -1.437, 1.059), "c": vec(0.949, -3.376, 1.107)},
    {"a": vec(0.078, -0.116, -0.065), "b": vec(1.144, -1.639, 0.674), "c": vec(0.232, -3.418, 0.622)},
    {"a": vec(0.196, -0.149, -0.083), "b": vec(1.266, -1.773, 0.386), "c": vec(0.364, -3.524, 0.041)},
    {"a": vec(0.021, 0.139, -0.027), "b": vec(1.304, -1.367, 0.264), "c": vec(0.499, -3.148, -0.162)},
    {"a": vec(-0.012, -0.164, -0.086), "b": vec(0.935, -1.924, -0.006), "c": vec(-0.501, -3.220, -0.513)},
    {"a": vec(-0.339, -0.007, 0.094), "b": vec(1.129, -1.362, -0.001), "c": vec(-0.036, -2.727, -0.884)},
    {"a": vec(-0.248, 0.236, -0.215), "b": vec(0.853, -1.431, -0.135), "c": vec(-0.748, -2.490, -0.697)},
    {"a": vec(-0.382, 0.340, -0.110), "b": vec(0.996, -1.109, -0.141), "c": vec(-0.350, -2.174, -1.167)},
    {"a": vec(-0.650, 0.523, -0.251), "b": vec(0.912, -0.717, -0.409), "c": vec(-0.701, -1.592, -1.202)},
    {"a": vec(-0.238, 0.495, 0.098), "b": vec(0.958, -1.106, 0.024), "c": vec(-0.471, -1.353, -1.353)},
    {"a": vec(-0.468, 0.673, 0.028), "b": vec(1.080, -0.560, -0.258), "c": vec(-0.302, -0.777, -1.688)},
    {"a": vec(-0.276, 0.847, 0.305), "b": vec(1.274, -0.276, -0.277), "c": vec(-0.168, -0.495, -1.645)},
    {"a": vec(-0.515, 0.943, 0.281), "b": vec(1.023, -0.293, -0.048), "c": vec(-0.339, -0.172, -1.507)},
    {"a": vec(-0.249, 0.996, 0.358), "b": vec(1.309, 0.006, -0.411), "c": vec(-0.243, 0.250, -1.649)},
    {"a": vec(-0.013, 0.801, 0.700), "b": vec(1.644, 0.060, -0.140), "c": vec(0.257, 0.362, -1.550)},
    {"a": vec(0.197, 0.940, 0.868), "b": vec(1.800, 0.488, -0.238), "c": vec(0.094, 0.643, -1.270)},
    {"a": vec(0.523, 0.712, 0.832), "b": vec(2.216, 0.579, -0.224), "c": vec(0.884, 0.722, -1.709)},
    {"a": vec(0.771, 0.959, 0.947), "b": vec(2.369, 0.684, -0.224), "c": vec(0.912, 0.990, -1.559)},
    {"a": vec(0.589, 0.878, 0.781), "b": vec(2.143, 1.060, -0.464), "c": vec(0.608, 0.890, -1.734)},
]

# Create ground truth arm (green) - sphere sizes: 0.4, 0.3, 0.2
gt_a = sphere(pos=gt_frames[0]["a"], radius=0.4, color=color.green)
gt_b = sphere(pos=gt_frames[0]["b"], radius=0.3, color=color.green)
gt_c = sphere(pos=gt_frames[0]["c"], radius=0.2, color=color.green)
gt_curve = curve(color=color.green, radius=0.05)
gt_curve.append(gt_frames[0]["a"])
gt_curve.append(gt_frames[0]["b"])
gt_curve.append(gt_frames[0]["c"])

# Create predicted arm (red) - sphere sizes: 0.4, 0.3, 0.2
pred_a = sphere(pos=pred_frames[0]["a"], radius=0.4, color=color.red)
pred_b = sphere(pos=pred_frames[0]["b"], radius=0.3, color=color.red)
pred_c = sphere(pos=pred_frames[0]["c"], radius=0.2, color=color.red)
pred_curve = curve(color=color.red, radius=0.05)
pred_curve.append(pred_frames[0]["a"])
pred_curve.append(pred_frames[0]["b"])
pred_curve.append(pred_frames[0]["c"])

# Legend
scene.append_to_caption("\n\nGreen = Ground Truth\nRed = Predicted\n")

# Animation loop
frame_idx = 0
num_frames = 20

while True:
    rate(6)

    # Update ground truth positions
    gt_a.pos = gt_frames[frame_idx]["a"]
    gt_b.pos = gt_frames[frame_idx]["b"]
    gt_c.pos = gt_frames[frame_idx]["c"]

    # Update ground truth curve
    gt_curve.modify(0, pos=gt_frames[frame_idx]["a"])
    gt_curve.modify(1, pos=gt_frames[frame_idx]["b"])
    gt_curve.modify(2, pos=gt_frames[frame_idx]["c"])

    # Update predicted positions
    pred_a.pos = pred_frames[frame_idx]["a"]
    pred_b.pos = pred_frames[frame_idx]["b"]
    pred_c.pos = pred_frames[frame_idx]["c"]

    # Update predicted curve
    pred_curve.modify(0, pos=pred_frames[frame_idx]["a"])
    pred_curve.modify(1, pos=pred_frames[frame_idx]["b"])
    pred_curve.modify(2, pos=pred_frames[frame_idx]["c"])

    frame_idx = (frame_idx + 1) % num_frames
