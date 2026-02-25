"""Human body pose visualizer — choose between FK and Anatomy3D parameterizations."""

import argparse

from vpython import rate


def main():
    parser = argparse.ArgumentParser(description="Human pose visualizer")
    parser.add_argument(
        "--mode", choices=["fk", "anatomy3d"], default="fk",
        help="Parameterization mode: 'fk' (parent-relative rotations) or 'anatomy3d' (world-space bone directions)",
    )
    args = parser.parse_args()

    if args.mode == "fk":
        from fk.human_pose_fk import HumanPose
        from fk.visualize import HumanVisualizer

        pose = HumanPose.default_standing()
        HumanVisualizer(pose)
    else:
        from anatomy_3d.human_pose_anatomy_3d import HumanPoseAnatomy3D
        from anatomy_3d.visualize import Anatomy3DVisualizer

        pose = HumanPoseAnatomy3D.default_standing()
        Anatomy3DVisualizer(pose)

    while True:
        rate(30)


if __name__ == "__main__":
    main()
