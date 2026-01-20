import numpy as np
from vpython import canvas, box, sphere, curve, vector, color, rate

from model.arm import Arm
from model.environment import Environment

# Position velocity constants
POS_VEL_MIN = -0.05
POS_VEL_MAX = 0.05
POS_VEL_STEP = 0.01

# Azimuth velocity constants
AZIMUTH_VEL_MIN = -0.05
AZIMUTH_VEL_MAX = 0.05
AZIMUTH_VEL_STEP = 0.01

# Elevation velocity constants
ELEV_VEL_MIN = -0.05
ELEV_VEL_MAX = 0.05
ELEV_VEL_STEP = 0.01


def numpy_to_vpython(arr: np.ndarray) -> vector:
    return vector(float(arr[0]), float(arr[1]), float(arr[2]))


def wrap(value: float, min_val: float, max_val: float) -> float:
    """Wrap value to range [min_val, max_val] using modulo."""
    range_size = max_val - min_val
    return ((value - min_val) % range_size) + min_val


def main():
    # Initialize arm state
    a_pos = np.array([0.0, 0.0, 0.0])
    a_b_azimuth = np.random.uniform(-np.pi, np.pi)
    a_b_elev = np.random.uniform(-np.pi / 2, np.pi / 2)
    b_c_azimuth = np.random.uniform(-np.pi, np.pi)
    b_c_elev = np.random.uniform(-np.pi / 2, np.pi / 2)

    # Initialize velocities
    pos_vel = np.array([0.0, 0.0, 0.0])
    a_b_azimuth_vel = 0.0
    a_b_elev_vel = 0.0
    b_c_azimuth_vel = 0.0
    b_c_elev_vel = 0.0

    # Arm segment lengths
    a_b_length = 2.0
    b_c_length = 2.0

    # Create environment
    camera_rotation = np.array(
        [
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0],
        ]
    )
    env = Environment(
        cube_size=10.0,
        camera_position=np.array([8.0, 0.0, 0.0]),
        camera_rotation=camera_rotation,
        focal_length=(50.0, 50.0),
        principal_point=(50.0, 50.0),
        image_size=(100, 100),
    )

    # Create initial arm and get coordinates
    arm = Arm(
        a_pos=a_pos,
        a_b_length=a_b_length,
        a_b_polar=(a_b_azimuth, a_b_elev),
        b_c_length=b_c_length,
        b_c_polar=(b_c_azimuth, b_c_elev),
    )
    coords = arm.get_coordinates()

    # Setup VPython scene
    scene = canvas(
        title="Arm Motion Simulation",
        width=800,
        height=600,
        center=vector(0, 0, 0),
        background=color.gray(0.2),
    )
    scene.up = vector(0, 0, 1)
    scene.forward = vector(-1, -0.5, -1)
    scene.range = 12

    # Environment cube
    box(
        pos=vector(0, 0, 0),
        size=vector(env.cube_size, env.cube_size, env.cube_size),
        color=color.white,
        opacity=0.1,
    )

    # Camera position
    camera_pos = numpy_to_vpython(env.camera_position)
    sphere(pos=camera_pos, radius=0.3, color=color.red)

    # Create arm visualization objects (will be updated in loop)
    sphere_a = sphere(pos=numpy_to_vpython(coords["a"]), radius=0.4, color=color.green)
    sphere_b = sphere(pos=numpy_to_vpython(coords["b"]), radius=0.3, color=color.yellow)
    sphere_c = sphere(pos=numpy_to_vpython(coords["c"]), radius=0.2, color=color.cyan)
    arm_curve = curve(
        pos=[
            numpy_to_vpython(coords["a"]),
            numpy_to_vpython(coords["b"]),
            numpy_to_vpython(coords["c"]),
        ],
        radius=0.05,
        color=color.white,
    )

    # Main animation loop
    while True:
        rate(30)

        # Update velocities with random walk
        # For position velocities (3 components)
        for i in range(3):
            delta = np.random.choice([-1, 0, 1]) * POS_VEL_STEP
            pos_vel[i] = np.clip(pos_vel[i] + delta, POS_VEL_MIN, POS_VEL_MAX)

        # For angular velocities
        a_b_azimuth_vel += np.random.choice([-1, 0, 1]) * AZIMUTH_VEL_STEP
        a_b_azimuth_vel = np.clip(a_b_azimuth_vel, AZIMUTH_VEL_MIN, AZIMUTH_VEL_MAX)

        a_b_elev_vel += np.random.choice([-1, 0, 1]) * ELEV_VEL_STEP
        a_b_elev_vel = np.clip(a_b_elev_vel, ELEV_VEL_MIN, ELEV_VEL_MAX)

        b_c_azimuth_vel += np.random.choice([-1, 0, 1]) * AZIMUTH_VEL_STEP
        b_c_azimuth_vel = np.clip(b_c_azimuth_vel, AZIMUTH_VEL_MIN, AZIMUTH_VEL_MAX)

        b_c_elev_vel += np.random.choice([-1, 0, 1]) * ELEV_VEL_STEP
        b_c_elev_vel = np.clip(b_c_elev_vel, ELEV_VEL_MIN, ELEV_VEL_MAX)

        # Update position (clip to [-1, 1])
        a_pos = a_pos + pos_vel
        a_pos = np.clip(a_pos, -1.0, 1.0)

        # Update azimuth angles (wrap to [-pi, pi])
        a_b_azimuth += a_b_azimuth_vel
        a_b_azimuth = wrap(a_b_azimuth, -np.pi, np.pi)

        b_c_azimuth += b_c_azimuth_vel
        b_c_azimuth = wrap(b_c_azimuth, -np.pi, np.pi)

        # Update elevation angles (wrap and adjust azimuth if out of bounds)
        a_b_elev += a_b_elev_vel
        if a_b_elev > np.pi / 2:
            a_b_elev = np.pi - a_b_elev
            a_b_azimuth = wrap(a_b_azimuth + np.pi, -np.pi, np.pi)
            a_b_elev_vel = -a_b_elev_vel
        elif a_b_elev < -np.pi / 2:
            a_b_elev = -np.pi - a_b_elev
            a_b_azimuth = wrap(a_b_azimuth + np.pi, -np.pi, np.pi)
            a_b_elev_vel = -a_b_elev_vel

        b_c_elev += b_c_elev_vel
        if b_c_elev > np.pi / 2:
            b_c_elev = np.pi - b_c_elev
            b_c_azimuth = wrap(b_c_azimuth + np.pi, -np.pi, np.pi)
            b_c_elev_vel = -b_c_elev_vel
        elif b_c_elev < -np.pi / 2:
            b_c_elev = -np.pi - b_c_elev
            b_c_azimuth = wrap(b_c_azimuth + np.pi, -np.pi, np.pi)
            b_c_elev_vel = -b_c_elev_vel

        # Create new arm with updated values
        arm = Arm(
            a_pos=a_pos,
            a_b_length=a_b_length,
            a_b_polar=(a_b_azimuth, a_b_elev),
            b_c_length=b_c_length,
            b_c_polar=(b_c_azimuth, b_c_elev),
        )
        coords = arm.get_coordinates()

        # Update VPython visualization
        sphere_a.pos = numpy_to_vpython(coords["a"])
        sphere_b.pos = numpy_to_vpython(coords["b"])
        sphere_c.pos = numpy_to_vpython(coords["c"])

        # Update curve by clearing and recreating points
        arm_curve.clear()
        arm_curve.append(numpy_to_vpython(coords["a"]))
        arm_curve.append(numpy_to_vpython(coords["b"]))
        arm_curve.append(numpy_to_vpython(coords["c"]))


if __name__ == "__main__":
    main()
