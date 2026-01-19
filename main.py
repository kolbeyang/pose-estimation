import numpy as np
from vpython import canvas, box, sphere, curve, vector, color, rate

from arm import Arm
from environment import Environment


def numpy_to_vpython(arr: np.ndarray) -> vector:
    """Convert a numpy array to a VPython vector."""
    return vector(float(arr[0]), float(arr[1]), float(arr[2]))


def create_visualization(env: Environment):
    """
    Create VPython visualization of the environment.

    Args:
        env: The Environment instance to visualize
    """
    # Create the canvas
    scene = canvas(
        title="Pose Estimation Simulation",
        width=800,
        height=600,
        center=vector(0, 0, 0),
        background=color.gray(0.2),
    )

    # Enable mouse controls for rotation, zoom, and pan
    scene.userspin = True  # Right-click drag to rotate
    scene.userzoom = True  # Scroll wheel to zoom
    scene.userpan = True  # Shift + drag to pan

    # Create the environment cube (wireframe-style using opacity)
    cube = box(
        pos=vector(0, 0, 0),
        size=vector(env.cube_size, env.cube_size, env.cube_size),
        color=color.white,
        opacity=0.1,
    )

    # Create a sphere to represent the camera position
    camera_pos = numpy_to_vpython(env.camera_position)

    camera_sphere = sphere(pos=camera_pos, radius=0.3, color=color.red)

    # Set the initial viewing angle
    scene.forward = vector(-1, -0.5, -1)
    scene.range = 12

    return scene, cube, camera_sphere


def main():
    # Create the environment with default settings
    env = Environment(
        cube_size=10.0,
        camera_position=np.array([8.0, 0.0, 0.0]),
        camera_rotation=np.eye(3),
        focal_length=(800.0, 800.0),
        principal_point=(200.0, 200.0),
        image_size=(400, 400),
    )

    # Create the visualization
    scene, cube, camera_sphere = create_visualization(env)

    # Create an arm
    arm = Arm(
        a_pos=np.array([0, 0, 0]),
        a_b_length=2.0,
        a_b_polar=(-np.pi / 6, np.pi / 4),  # 45 degrees elevation
        b_c_length=2,
        b_c_polar=(np.pi / 2, np.pi / 3),  # 90 degrees azimuth, -30 degrees elevation
    )

    # Get arm coordinates and visualize
    coords = arm.get_coordinates()
    a_pos = numpy_to_vpython(coords["a"])
    b_pos = numpy_to_vpython(coords["b"])
    c_pos = numpy_to_vpython(coords["c"])

    # Draw spheres for each point (different colors)
    sphere(pos=a_pos, radius=0.4, color=color.green)  # A - green
    sphere(pos=b_pos, radius=0.3, color=color.yellow)  # B - yellow
    sphere(pos=c_pos, radius=0.2, color=color.cyan)  # C - cyan

    # Draw line connecting all points
    curve(pos=[a_pos, b_pos, c_pos], radius=0.05, color=color.white)

    # Keep the visualization running
    while True:
        rate(30)


if __name__ == "__main__":
    main()
