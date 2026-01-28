import argparse

import numpy as np
from vpython import rate

from model.arm import Arm
from model.environment import Environment
from video import Video
from visualize import Visualizer

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200

# Camera intrinsics
FOCAL_LENGTH = (50.0, 50.0)
PRINCIPAL_POINT = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)

# Position velocity constants
POS_VEL_MIN = -0.20
POS_VEL_MAX = 0.20
POS_VEL_STEP = 0.04

# Azimuth velocity constants
AZIMUTH_VEL_MIN = -0.20
AZIMUTH_VEL_MAX = 0.20
AZIMUTH_VEL_STEP = 0.04

# Elevation velocity constants
ELEV_VEL_MIN = -0.20
ELEV_VEL_MAX = 0.20
ELEV_VEL_STEP = 0.04

# Roll velocity constants
ROLL_VEL_MIN = -0.20
ROLL_VEL_MAX = 0.20
ROLL_VEL_STEP = 0.04

# Theta velocity constants (for BC bend angle)
THETA_VEL_MIN = -0.20
THETA_VEL_MAX = 0.20
THETA_VEL_STEP = 0.04


def wrap(value: float, min_val: float, max_val: float) -> float:
    """Wrap value to range [min_val, max_val] using modulo."""
    range_size = max_val - min_val
    return ((value - min_val) % range_size) + min_val


def main():
    parser = argparse.ArgumentParser(description="Arm motion simulation")
    parser.add_argument("-r", "--record", action="store_true", help="Record frames")
    parser.add_argument(
        "-n", "--max-frames", type=int, default=None, help="Max frames to record"
    )
    args = parser.parse_args()

    # Initialize arm state
    a_pos = np.array([0.0, 0.0, 0.0])
    a_b_azimuth = np.random.uniform(-np.pi, np.pi)
    a_b_elev = np.random.uniform(-np.pi / 2, np.pi / 2)
    a_b_roll = np.random.uniform(-np.pi, np.pi)
    b_c_theta = np.random.uniform(-np.pi / 2, np.pi / 2)

    # Initialize velocities
    pos_vel = np.array([0.0, 0.0, 0.0])
    a_b_azimuth_vel = 0.0
    a_b_elev_vel = 0.0
    a_b_roll_vel = 0.0
    b_c_theta_vel = 0.0

    # Arm segment lengths
    a_b_length = 2.0
    b_c_length = 2.0

    # Create environment (just cube bounds for visualization)
    env = Environment(cube_size=10.0)

    # Camera parameters
    camera_position = np.array([8.0, 0.0, 0.0])
    camera_rotation = np.array(
        [
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0],
        ]
    )

    # Create initial arm and get coordinates
    arm = Arm(
        a_pos=a_pos,
        a_b_length=a_b_length,
        a_b_polar=(a_b_azimuth, a_b_elev, a_b_roll),
        b_c_length=b_c_length,
        b_c_theta=b_c_theta,
    )
    coords = arm.get_coordinates_numpy()

    # Create video (always, for visualization camera marker)
    video = (
        Video(
            camera_position=camera_position,
            camera_rotation=camera_rotation,
            focal_length=FOCAL_LENGTH,
            principal_point=PRINCIPAL_POINT,
            image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            a_b_length=a_b_length,
            b_c_length=b_c_length,
        )
        if args.record
        else None
    )

    visualizer = Visualizer(env, [coords], camera=video)

    if video:
        print(f"Recording to: {video.video_dir}")

    frame_count = 0

    # Main animation loop
    while True:
        rate(Visualizer.FPS)

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

        a_b_roll_vel += np.random.choice([-1, 0, 1]) * ROLL_VEL_STEP
        a_b_roll_vel = np.clip(a_b_roll_vel, ROLL_VEL_MIN, ROLL_VEL_MAX)

        b_c_theta_vel += np.random.choice([-1, 0, 1]) * THETA_VEL_STEP
        b_c_theta_vel = np.clip(b_c_theta_vel, THETA_VEL_MIN, THETA_VEL_MAX)

        # Update position (clip to [-1, 1])
        a_pos = a_pos + pos_vel
        a_pos = np.clip(a_pos, -1.0, 1.0)

        # Update azimuth and roll angles (wrap to [-pi, pi])
        a_b_azimuth += a_b_azimuth_vel
        a_b_azimuth = wrap(a_b_azimuth, -np.pi, np.pi)

        a_b_roll += a_b_roll_vel
        a_b_roll = wrap(a_b_roll, -np.pi, np.pi)

        # Update elevation angle (reflect at boundaries)
        a_b_elev += a_b_elev_vel
        if a_b_elev > np.pi / 2:
            a_b_elev = np.pi - a_b_elev
            a_b_azimuth = wrap(a_b_azimuth + np.pi, -np.pi, np.pi)
            a_b_elev_vel = -a_b_elev_vel
        elif a_b_elev < -np.pi / 2:
            a_b_elev = -np.pi - a_b_elev
            a_b_azimuth = wrap(a_b_azimuth + np.pi, -np.pi, np.pi)
            a_b_elev_vel = -a_b_elev_vel

        # Update theta angle (reflect at boundaries [-pi/2, pi/2])
        b_c_theta += b_c_theta_vel
        if b_c_theta > np.pi / 2:
            b_c_theta = np.pi - b_c_theta
            b_c_theta_vel = -b_c_theta_vel
        elif b_c_theta < -np.pi / 2:
            b_c_theta = -np.pi - b_c_theta
            b_c_theta_vel = -b_c_theta_vel

        # Create new arm with updated values
        arm = Arm(
            a_pos=a_pos,
            a_b_length=a_b_length,
            a_b_polar=(a_b_azimuth, a_b_elev, a_b_roll),
            b_c_length=b_c_length,
            b_c_theta=b_c_theta,
        )
        coords = arm.get_coordinates_numpy()

        visualizer.update([coords])

        if video:
            video.update(
                coords,
                a_pos,
                (a_b_azimuth, a_b_elev, a_b_roll),
                b_c_theta,
            )

        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            print(f"Recorded {frame_count} frames")
            break


if __name__ == "__main__":
    main()
