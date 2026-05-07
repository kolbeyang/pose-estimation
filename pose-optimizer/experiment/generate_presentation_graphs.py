"""Generate bone length and position-over-time graphs for 5 diverse samples.

Runs bone_length_variation.py and elbow_smoothness.py on 5 examples from
a completed run, using forearm/wrist/elbow joints.

Usage:
    cd pose-optimizer
    uv run python experiment/generate_presentation_graphs.py <run_dir> <output_dir>

Example:
    uv run python experiment/generate_presentation_graphs.py \
        /home/kky2806/pose-estimation-output/run_2026_04_14_14_48 \
        output/presentation-graphs
"""

import os
import sys
import subprocess

# 5 diverse samples: stationary, dance, office, pose, instrumental
SAMPLES = [
    "161029_piano3_54",
    "170307_dance5_3453",
    "170915_office1_180",
    "171026_pose1_232",
    "171204_pose2_4662",
]

# Joints/bones to graph
BONE_LENGTH_SIDES = ["R", "L"]
POSITION_JOINTS = ["RElbow", "RWrist", "LElbow", "LWrist"]


def main():
    if len(sys.argv) < 3:
        print("Usage: uv run python experiment/generate_presentation_graphs.py <run_dir> <output_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    for sample in SAMPLES:
        mb_traj = os.path.join(run_dir, sample, "motionbert", "trajectories.json")
        mp_traj = os.path.join(run_dir, sample, "mediapipe", "trajectories.json")

        if not os.path.exists(mb_traj):
            print(f"SKIP {sample}: {mb_traj} not found")
            continue
        if not os.path.exists(mp_traj):
            print(f"SKIP {sample}: {mp_traj} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Sample: {sample}")
        print(f"{'='*60}")

        # Bone length graphs (R and L forearm)
        for side in BONE_LENGTH_SIDES:
            print(f"  Bone length: {side} forearm")
            subprocess.run([
                sys.executable,
                os.path.join(script_dir, "bone_length_variation.py"),
                mb_traj, mp_traj,
                "--side", side,
                "--sample-name", sample,
                "--output", output_dir,
            ], check=True)

        # Position over time graphs
        for joint in POSITION_JOINTS:
            print(f"  Position: {joint}")
            subprocess.run([
                sys.executable,
                os.path.join(script_dir, "elbow_smoothness.py"),
                mb_traj, mp_traj,
                "--joint", joint,
                "--sample-name", sample,
                "--output", output_dir,
            ], check=True)

    graphs = [f for f in os.listdir(output_dir) if f.endswith(".png")]
    print(f"\nGenerated {len(graphs)} graphs in {output_dir}:")
    for g in sorted(graphs):
        print(f"  {g}")


if __name__ == "__main__":
    main()
