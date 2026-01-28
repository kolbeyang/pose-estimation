import numpy as np
from vpython import box, sphere, curve, vector, color, rate, wtext, winput, slider, scene

from model.arm import Arm
from model.environment import Environment
from visualize import to_vpython, Visualizer


# Global variables for the arm visualization objects
arm_sphere_a = None
arm_sphere_b = None
arm_sphere_c = None
arm_curve = None

# Global variables for input values
input_x = 0.0
input_y = 0.0
input_z = 0.0
input_ab_azimuth = -np.pi / 6
input_ab_elevation = np.pi / 4
input_ab_roll = 0.0
input_ab_length = 2.0
input_bc_theta = 0.0  # theta=0 -> perpendicular (90° bend)
input_bc_length = 2.0


def update_arm():
    """Update the arm visualization based on current input values."""
    global arm_sphere_a, arm_sphere_b, arm_sphere_c, arm_curve

    arm = Arm(
        a_pos=np.array([input_x, input_y, input_z]),
        a_b_length=input_ab_length,
        a_b_polar=(input_ab_azimuth, input_ab_elevation, input_ab_roll),
        b_c_length=input_bc_length,
        b_c_theta=input_bc_theta,
    )

    coords = arm.get_coordinates_numpy()
    a_pos = to_vpython(coords["a"])
    b_pos = to_vpython(coords["b"])
    c_pos = to_vpython(coords["c"])

    # Update sphere positions
    arm_sphere_a.pos = a_pos
    arm_sphere_b.pos = b_pos
    arm_sphere_c.pos = c_pos

    # Update curve - need to recreate it
    arm_curve.clear()
    arm_curve.append(a_pos)
    arm_curve.append(b_pos)
    arm_curve.append(c_pos)


def on_x_change(evt):
    global input_x
    try:
        input_x = float(evt.text)
        update_arm()
    except ValueError:
        pass


def on_y_change(evt):
    global input_y
    try:
        input_y = float(evt.text)
        update_arm()
    except ValueError:
        pass


def on_z_change(evt):
    global input_z
    try:
        input_z = float(evt.text)
        update_arm()
    except ValueError:
        pass


def on_ab_azimuth_change(evt):
    global input_ab_azimuth
    input_ab_azimuth = evt.value
    update_arm()


def on_ab_elevation_change(evt):
    global input_ab_elevation
    input_ab_elevation = evt.value
    update_arm()


def on_ab_roll_change(evt):
    global input_ab_roll
    input_ab_roll = evt.value
    update_arm()


def on_ab_length_change(evt):
    global input_ab_length
    try:
        input_ab_length = float(evt.text)
        update_arm()
    except ValueError:
        pass


def on_bc_theta_change(evt):
    global input_bc_theta
    input_bc_theta = evt.value
    update_arm()


def on_bc_length_change(evt):
    global input_bc_length
    try:
        input_bc_length = float(evt.text)
        update_arm()
    except ValueError:
        pass


def main():
    global arm_sphere_a, arm_sphere_b, arm_sphere_c, arm_curve

    # Create the canvas
    scene.title = "Arm Demo - Interactive Controls"
    scene.width = 800
    scene.height = 600
    scene.center = vector(0, 0, 0)
    scene.background = color.gray(0.2)

    # Enable mouse controls
    scene.up = vector(0, 0, 1)
    scene.userspin = True
    scene.userzoom = True
    scene.userpan = True

    # Create the environment
    env = Environment(cube_size=10.0)
    box(
        pos=vector(0, 0, 0),
        size=vector(env.cube_size, env.cube_size, env.cube_size),
        color=color.white,
        opacity=0.1,
    )

    # Create a translucent XY plane at z=-5
    box(
        pos=vector(0, 0, -5),
        size=vector(env.cube_size, env.cube_size, 0.01),
        color=color.blue,
        opacity=0.1,
    )

    # Set the initial viewing angle
    scene.forward = vector(-1, -0.5, -1)
    scene.range = 12

    # Create initial arm
    arm = Arm(
        a_pos=np.array([input_x, input_y, input_z]),
        a_b_length=input_ab_length,
        a_b_polar=(input_ab_azimuth, input_ab_elevation, input_ab_roll),
        b_c_length=input_bc_length,
        b_c_theta=input_bc_theta,
    )

    coords = arm.get_coordinates_numpy()
    a_pos = to_vpython(coords["a"])
    b_pos = to_vpython(coords["b"])
    c_pos = to_vpython(coords["c"])

    # Create visualization objects
    arm_sphere_a = sphere(pos=a_pos, radius=0.4, color=color.green)
    arm_sphere_b = sphere(pos=b_pos, radius=0.3, color=color.yellow)
    arm_sphere_c = sphere(pos=c_pos, radius=0.2, color=color.cyan)
    arm_curve = curve(pos=[a_pos, b_pos, c_pos], radius=0.05, color=color.white)

    # Create input controls
    scene.append_to_caption("\n\n")

    # Position inputs (X, Y, Z)
    scene.append_to_caption("Position A:\n")
    wtext(text="X: ")
    winput(bind=on_x_change, text=str(input_x), width=80)
    wtext(text="  Y: ")
    winput(bind=on_y_change, text=str(input_y), width=80)
    wtext(text="  Z: ")
    winput(bind=on_z_change, text=str(input_z), width=80)

    scene.append_to_caption("\n\n")

    # AB segment inputs
    scene.append_to_caption("Segment A→B:\n")
    wtext(text="Azimuth: ")
    slider(bind=on_ab_azimuth_change, min=-np.pi, max=np.pi, value=input_ab_azimuth, length=200)
    scene.append_to_caption("\n")
    wtext(text="Elevation: ")
    slider(bind=on_ab_elevation_change, min=-np.pi/2, max=np.pi/2, value=input_ab_elevation, length=200)
    scene.append_to_caption("\n")
    wtext(text="Roll: ")
    slider(bind=on_ab_roll_change, min=-np.pi, max=np.pi, value=input_ab_roll, length=200)
    scene.append_to_caption("\n")
    wtext(text="Length: ")
    winput(bind=on_ab_length_change, text=str(input_ab_length), width=80)

    scene.append_to_caption("\n\n")

    # BC segment inputs
    scene.append_to_caption("Segment B→C:\n")
    wtext(text="Theta (bend): ")
    slider(bind=on_bc_theta_change, min=-np.pi/2, max=np.pi/2, value=input_bc_theta, length=200)
    scene.append_to_caption("  (0=90° bend, π/2=straight)\n")
    wtext(text="Length: ")
    winput(bind=on_bc_length_change, text=str(input_bc_length), width=80)

    # Keep the visualization running
    while True:
        rate(Visualizer.FPS)


if __name__ == "__main__":
    main()
