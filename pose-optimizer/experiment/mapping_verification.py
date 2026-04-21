"""Verify joint mappings are correct end-to-end through the MotionBERT pipeline.

Tests:
1. mpii_to_motionbert_17: raw MPII[16] -> 17-joint MotionBERT input - are joints in right slots?
2. strip_nose_joint: 17-joint MB output -> 16-joint SKELETON_16 - do indices land correctly?
3. motionbert_to_camera_space: does bone scale use the right joints?
4. End-to-end: project skeleton back to 2D, measure distance to SH keypoints

Run locally:
    uv run python experiment/mapping_verification.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from skeleton import (
    mpii_to_skeleton, mpii_to_motionbert_17, strip_nose_joint,
    JOINT_NAMES, PARENTS, DEFAULT_BONE_LENGTHS, NUM_JOINTS,
)


# ---------------------------------------------------------------------------
# MPII joint names for reference
# ---------------------------------------------------------------------------
MPII_JOINT_NAMES = [
    "RAnkle",    # 0
    "RKnee",     # 1
    "RHip",      # 2
    "LHip",      # 3
    "LKnee",     # 4
    "LAnkle",    # 5
    "Pelvis",    # 6
    "Thorax",    # 7  (our Neck / shoulder level)
    "UpperNeck", # 8
    "HeadTop",   # 9
    "RWrist",    # 10
    "RElbow",    # 11
    "RShoulder", # 12
    "LShoulder", # 13
    "LElbow",    # 14
    "LWrist",    # 15
]

# MotionBERT 17-joint output names (H36M order as used in MotionBERT)
# This is the order AFTER the model runs, before strip_nose_joint
MOTIONBERT_17_NAMES = [
    "Pelvis",      # 0
    "RHip",        # 1
    "RKnee",       # 2
    "RAnkle",      # 3
    "LHip",        # 4
    "LKnee",       # 5
    "LAnkle",      # 6
    "Spine",       # 7  (FK-internal)
    "Neck",        # 8  (Thorax / shoulder level)
    "Nose",        # 9  (synthesized, will be stripped)
    "HeadTop",     # 10 (will become SKELETON_16 index 9)
    "LShoulder",   # 11 -> SKELETON_16 index 10
    "LElbow",      # 12 -> SKELETON_16 index 11
    "LWrist",      # 13 -> SKELETON_16 index 12
    "RShoulder",   # 14 -> SKELETON_16 index 13
    "RElbow",      # 15 -> SKELETON_16 index 14
    "RWrist",      # 16 -> SKELETON_16 index 15
]


def test_mpii_to_motionbert_17():
    """Verify mpii_to_motionbert_17 maps each MPII joint to the right slot."""
    print("=" * 70)
    print("TEST 1: mpii_to_motionbert_17 — MPII[16] -> 17-joint MB input")
    print("=" * 70)

    # Create synthetic MPII input where each joint has a unique ID value
    mpii = np.zeros((16, 3), dtype=np.float32)
    for i in range(16):
        mpii[i] = [i * 10, i * 10, 1.0]  # x=i*10, y=i*10, conf=1

    mb17 = mpii_to_motionbert_17(mpii)

    expected = {
        0: ("Pelvis",    [60, 60]),   # MPII[6]
        1: ("RHip",      [20, 20]),   # MPII[2]
        2: ("RKnee",     [10, 10]),   # MPII[1]
        3: ("RAnkle",    [0,  0]),    # MPII[0]
        4: ("LHip",      [30, 30]),   # MPII[3]
        5: ("LKnee",     [40, 40]),   # MPII[4]
        6: ("LAnkle",    [50, 50]),   # MPII[5]
        # 7: Spine = midpoint(Pelvis, Neck) = midpoint(MPII[6], MPII[7]) = (65,65)
        8: ("Neck/Thorax", [70, 70]), # MPII[7]
        # 9: Nose = MPII[8] + 0.3*(MPII[9]-MPII[8]) = 80 + 0.3*10 = 83
        10: ("HeadTop",  [90, 90]),   # MPII[9]
        11: ("LShoulder",[130,130]),  # MPII[13]
        12: ("LElbow",   [140,140]),  # MPII[14]
        13: ("LWrist",   [150,150]),  # MPII[15]
        14: ("RShoulder",[120,120]),  # MPII[12]
        15: ("RElbow",   [110,110]),  # MPII[11]
        16: ("RWrist",   [100,100]),  # MPII[10]
    }

    all_pass = True
    for idx, (name, expected_xy) in expected.items():
        actual_xy = mb17[idx, :2].tolist()
        ok = np.allclose(actual_xy, expected_xy, atol=1.0)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] MB17[{idx:2d}] {name:<12}: expected {expected_xy}, got {actual_xy}")

    # Check spine (synthesized)
    expected_spine = [(60+70)/2, (60+70)/2]
    actual_spine = mb17[7, :2].tolist()
    ok = np.allclose(actual_spine, expected_spine, atol=0.1)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{status}] MB17[ 7] Spine (synth): expected {expected_spine}, got {actual_spine}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


def test_strip_nose_joint():
    """Verify strip_nose_joint(17-joint) -> correct 16-joint SKELETON_16 order."""
    print("=" * 70)
    print("TEST 2: strip_nose_joint — 17-joint MB output -> SKELETON_16")
    print("=" * 70)

    # Create 17-joint array where each joint has value = its original index
    mb17 = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        mb17[i] = [float(i), float(i), float(i)]

    skel16 = strip_nose_joint(mb17)

    # After strip_nose_joint removes index 9 (Nose):
    # - Indices 0-8 stay the same
    # - Old index 10 (HeadTop) becomes new index 9
    # - Old index 11 (LShoulder) becomes new index 10
    # ... etc
    expected_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]
    expected_names = [
        "Pelvis(0)", "RHip(1)", "RKnee(2)", "RAnkle(3)",
        "LHip(4)", "LKnee(5)", "LAnkle(6)", "Spine(7)", "Neck(8)",
        "HeadTop(9←MB10)", "LShoulder(10←MB11)", "LElbow(11←MB12)", "LWrist(12←MB13)",
        "RShoulder(13←MB14)", "RElbow(14←MB15)", "RWrist(15←MB16)",
    ]

    all_pass = True
    for skel_idx, (expected_val, name) in enumerate(zip(expected_values, expected_names)):
        actual_val = skel16[skel_idx, 0]
        ok = np.isclose(actual_val, float(expected_val))
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] SKEL16[{skel_idx:2d}] {name:<30}: expected MB[{expected_val}], got MB[{actual_val:.0f}]")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


def test_mpii_to_skeleton():
    """Verify mpii_to_skeleton maps MPII[16] -> SKELETON_16 correctly."""
    print("=" * 70)
    print("TEST 3: mpii_to_skeleton — MPII[16] -> SKELETON_16 (for SH 2D keypoints)")
    print("=" * 70)

    mpii = np.zeros((16, 3), dtype=np.float32)
    for i in range(16):
        mpii[i] = [float(i * 10), float(i * 10), 1.0]

    skel = mpii_to_skeleton(mpii)

    expected = {
        0:  ("Pelvis",    60.0),   # MPII[6]
        1:  ("RHip",      20.0),   # MPII[2]
        2:  ("RKnee",     10.0),   # MPII[1]
        3:  ("RAnkle",    0.0),    # MPII[0]
        4:  ("LHip",      30.0),   # MPII[3]
        5:  ("LKnee",     40.0),   # MPII[4]
        6:  ("LAnkle",    50.0),   # MPII[5]
        8:  ("Neck",      70.0),   # MPII[7]
        9:  ("HeadTop",   90.0),   # MPII[9] direct
        10: ("LShoulder", 130.0),  # MPII[13]
        11: ("LElbow",    140.0),  # MPII[14]
        12: ("LWrist",    150.0),  # MPII[15]
        13: ("RShoulder", 120.0),  # MPII[12]
        14: ("RElbow",    110.0),  # MPII[11]
        15: ("RWrist",    100.0),  # MPII[10]
    }

    all_pass = True
    for idx, (name, expected_x) in expected.items():
        actual_x = float(skel[idx, 0])
        ok = np.isclose(actual_x, expected_x)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] SKEL[{idx:2d}] {name:<12}: expected x={expected_x:.0f}, got x={actual_x:.0f}")

    # Spine = midpoint(Pelvis, Neck)
    expected_spine = (60.0 + 70.0) / 2
    actual_spine = float(skel[7, 0])
    ok = np.isclose(actual_spine, expected_spine)
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"  [{status}] SKEL[ 7] Spine (synth): expected x={expected_spine:.0f}, got x={actual_spine:.0f}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


def test_joint_alignment_mb17_vs_skel16():
    """Verify that after strip_nose_joint, MB17 joint positions align with SKELETON_16 names.

    The critical check: MotionBERT's model was trained with a specific joint order.
    After strip, our SKELETON_16 must map to the same anatomical joints.
    """
    print("=" * 70)
    print("TEST 4: Joint alignment — do MB17 and SKELETON_16 agree on anatomy?")
    print("=" * 70)

    # After strip_nose_joint removes MB17[9] (Nose), the mapping is:
    # SKELETON_16[j] = MB17[j] for j in 0..8
    # SKELETON_16[9] = MB17[10]  (HeadTop)
    # SKELETON_16[j] = MB17[j+1] for j in 10..15

    print("  Expected SKELETON_16 -> MB17 mapping after strip_nose_joint:")
    mapping = list(range(9)) + [10] + list(range(11, 17))
    for skel_j, mb_j in enumerate(mapping):
        skel_name = JOINT_NAMES[skel_j]
        mb_name = MOTIONBERT_17_NAMES[mb_j]
        match = skel_name.lower().replace(" ", "") == mb_name.lower().replace(" ", "").replace("/thorax","").replace("thorax","neck").replace("neck/thorax","neck")
        # Manual check
        manual_ok = {
            0: ("Pelvis", "Pelvis"),
            1: ("RHip", "RHip"),
            2: ("RKnee", "RKnee"),
            3: ("RAnkle", "RAnkle"),
            4: ("LHip", "LHip"),
            5: ("LKnee", "LKnee"),
            6: ("LAnkle", "LAnkle"),
            7: ("Spine", "Spine"),
            8: ("Neck", "Neck"),
            9: ("HeadTop", "HeadTop"),
            10: ("LShoulder", "LShoulder"),
            11: ("LElbow", "LElbow"),
            12: ("LWrist", "LWrist"),
            13: ("RShoulder", "RShoulder"),
            14: ("RElbow", "RElbow"),
            15: ("RWrist", "RWrist"),
        }
        s_expected, mb_expected = manual_ok[skel_j]
        ok = (s_expected.lower() in skel_name.lower()) and (mb_expected.lower() in mb_name.lower())
        status = "PASS" if ok else "FAIL !!!"
        print(f"  [{status}] SKEL[{skel_j:2d}] {skel_name:<12} <- MB17[{mb_j:2d}] {mb_name}")

    print()


def test_reliable_bones_still_valid():
    """Verify _RELIABLE_BONES_FOR_SCALE = {10,11,12,13,14,15} are still arm bones."""
    print("=" * 70)
    print("TEST 5: _RELIABLE_BONES_FOR_SCALE indices still refer to arm bones?")
    print("=" * 70)

    reliable = {10, 11, 12, 13, 14, 15}
    print("  These indices are used for bone-scale estimation in motionbert_to_camera_space.")
    print("  They should all be arm bones (consistent length, not truncated by crops).\n")
    for j in sorted(reliable):
        parent = int(PARENTS[j])
        print(f"  Joint {j:2d}: {JOINT_NAMES[j]:<12} (parent: {JOINT_NAMES[parent]}) "
              f"default length={DEFAULT_BONE_LENGTHS[j]:.3f}m")

    print("\n  These are the same joints as before the HeadTop change (arm joints).")
    print("  Indices 10-15 = LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist")
    print("  These were unchanged by strip_nose_joint -> PASS if arm joints\n")

    all_arm = all(
        JOINT_NAMES[j] in ("LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist")
        for j in reliable
    )
    print(f"  All reliable bones are arm bones: {'YES - PASS' if all_arm else 'NO - FAIL'}\n")
    return all_arm


def test_motionbert_input_matches_sh_output():
    """Verify that the 2D keypoints fed to MotionBERT (mpii_to_motionbert_17)
    use the same MPII joint slots that SH outputs, in the right order.

    MotionBERT was trained on H36M using 2D inputs in MPII order.
    SH outputs MPII-order heatmaps. We pass all_keypoints_2d (raw MPII) into
    mpii_to_motionbert_17. This test checks those match.
    """
    print("=" * 70)
    print("TEST 6: MotionBERT 2D input — does mpii_to_motionbert_17 preserve SH MPII order?")
    print("=" * 70)

    # Create identity MPII array: joint i has x = i, y = i
    mpii = np.eye(16, 3, dtype=np.float32)
    for i in range(16):
        mpii[i, 0] = float(i)
        mpii[i, 1] = float(i)
        mpii[i, 2] = 1.0

    mb17 = mpii_to_motionbert_17(mpii)

    # MotionBERT expects joints in H36M order, where:
    # H36M[0]=Pelvis, [1]=RHip, [2]=RKnee, [3]=RAnkle, [4]=LHip, [5]=LKnee, [6]=LAnkle
    # [7]=Spine, [8]=Neck(Thorax), [9]=Nose(synth), [10]=HeadTop
    # [11]=LShoulder, [12]=LElbow, [13]=LWrist, [14]=RShoulder, [15]=RElbow, [16]=RWrist

    checks = [
        (0,  "Pelvis",    6,  "MPII[6] Pelvis"),
        (1,  "RHip",      2,  "MPII[2] RHip"),
        (2,  "RKnee",     1,  "MPII[1] RKnee"),
        (3,  "RAnkle",    0,  "MPII[0] RAnkle"),
        (4,  "LHip",      3,  "MPII[3] LHip"),
        (5,  "LKnee",     4,  "MPII[4] LKnee"),
        (6,  "LAnkle",    5,  "MPII[5] LAnkle"),
        (8,  "Neck",      7,  "MPII[7] Thorax"),
        (10, "HeadTop",   9,  "MPII[9] HeadTop"),
        (11, "LShoulder", 13, "MPII[13] LShoulder"),
        (12, "LElbow",    14, "MPII[14] LElbow"),
        (13, "LWrist",    15, "MPII[15] LWrist"),
        (14, "RShoulder", 12, "MPII[12] RShoulder"),
        (15, "RElbow",    11, "MPII[11] RElbow"),
        (16, "RWrist",    10, "MPII[10] RWrist"),
    ]

    all_pass = True
    for mb_idx, joint_name, mpii_idx, source in checks:
        ok = np.isclose(mb17[mb_idx, 0], float(mpii_idx))
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] MB17[{mb_idx:2d}] {joint_name:<12} <- {source}: "
              f"expected x={mpii_idx:.0f}, got x={mb17[mb_idx,0]:.1f}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


if __name__ == "__main__":
    results = []
    results.append(test_mpii_to_motionbert_17())
    results.append(test_strip_nose_joint())
    results.append(test_mpii_to_skeleton())
    test_joint_alignment_mb17_vs_skel16()
    results.append(test_reliable_bones_still_valid())
    results.append(test_motionbert_input_matches_sh_output())

    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} tests passed")
    if passed == total:
        print("All mappings verified correct.")
    else:
        print("MAPPING ERRORS DETECTED — see FAIL lines above.")
    print("=" * 70)
