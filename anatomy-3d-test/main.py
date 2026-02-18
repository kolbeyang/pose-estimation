"""Human body pose visualizer with interactive parameter controls."""

from vpython import rate

from human_pose import HumanPose
from visualize import HumanVisualizer


def main():
    pose = HumanPose.default_standing()
    HumanVisualizer(pose)

    # VPython needs an event loop to process slider callbacks
    while True:
        rate(30)


if __name__ == "__main__":
    main()
