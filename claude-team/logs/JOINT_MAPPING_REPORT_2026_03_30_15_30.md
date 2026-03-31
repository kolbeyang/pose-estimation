# Joint Mapping Report

## All Joints Across All Representations

Every row is a distinct anatomical point. Nothing is merged unless it refers to the truly identical landmark.

| Anatomical Point | MotionBert (MPII 16) | MediaPipe (33 landmarks) | CMU GT (COCO19) | Optimizer (H36M 16) |
|---|---|---|---|---|
| **HEAD / FACE** | | | | |
| Top of Head | ✅ MPII[9] "Head" | | | |
| Nose | | ✅ MP[0] "Nose" | ✅ COCO[1] "Nose" | |
| Left Eye | | | ✅ COCO[15] "LEye" | |
| Right Eye | | | ✅ COCO[17] "REye" | |
| Left Ear | | | ✅ COCO[16] "LEar" | |
| Right Ear | | | ✅ COCO[18] "REar" | |
| **NECK / TORSO** | | | | |
| Base of Neck | ✅ MPII[8] "Neck" | | ✅ COCO[0] "Neck" | |
| Mid-Torso | ✅ MPII[7] "Thorax" | | | |
| Pelvis Center | ✅ MPII[6] "Pelvis" | | ✅ COCO[2] "BodyCenter" | |
| **COMPUTED JOINTS** | | | | |
| "Hip" (joint 0) | | | | ✅ From COCO: direct(BodyCenter[2]). From MP: midpoint(LHip,RHip). From MB: midpoint(RHip,LHip) |
| "Spine" (joint 7) | | | | ✅ From COCO: midpoint(BodyCenter,Neck). From MP: midpoint(Hip,Thorax). From MB: midpoint(Pelvis,Thorax) |
| "Thorax" (joint 8) | | | | ✅ From COCO: direct(Neck[0]). From MP: **midpoint(LShoulder,RShoulder)**. From MB: direct(MPII Neck[8]) |
| "Neck" (joint 9) | | | | ✅ From COCO: direct(**Nose**[1]). From MP: direct(**Nose**[0]). From MB: direct(**Head top**[9]) |
| **LEFT ARM** | | | | |
| L Shoulder | ✅ MPII[13] | ✅ MP[11] | ✅ COCO[3] | ✅ H36M[10] |
| L Elbow | ✅ MPII[14] | ✅ MP[13] | ✅ COCO[4] | ✅ H36M[11] |
| L Wrist | ✅ MPII[15] | ✅ MP[15] | ✅ COCO[5] | ✅ H36M[12] |
| **RIGHT ARM** | | | | |
| R Shoulder | ✅ MPII[12] | ✅ MP[12] | ✅ COCO[9] | ✅ H36M[13] |
| R Elbow | ✅ MPII[11] | ✅ MP[14] | ✅ COCO[10] | ✅ H36M[14] |
| R Wrist | ✅ MPII[10] | ✅ MP[16] | ✅ COCO[11] | ✅ H36M[15] |
| **LEFT LEG** | | | | |
| L Hip | ✅ MPII[3] | ✅ MP[23] | ✅ COCO[6] | ✅ H36M[4] |
| L Knee | ✅ MPII[4] | ✅ MP[25] | ✅ COCO[7] | ✅ H36M[5] |
| L Ankle | ✅ MPII[5] | ✅ MP[27] | ✅ COCO[8] | ✅ H36M[6] |
| **RIGHT LEG** | | | | |
| R Hip | ✅ MPII[2] | ✅ MP[24] | ✅ COCO[12] | ✅ H36M[1] |
| R Knee | ✅ MPII[1] | ✅ MP[26] | ✅ COCO[13] | ✅ H36M[2] |
| R Ankle | ✅ MPII[0] | ✅ MP[28] | ✅ COCO[14] | ✅ H36M[3] |

## Key Observations

### Clean (12 joints) — all systems agree
Arms and legs: L/R Shoulder, Elbow, Wrist, Hip, Knee, Ankle. No ambiguity.

### Problematic — optimizer "Neck" (joint 9)
- COCO19 GT and MediaPipe both map **Nose** to this joint
- MotionBert maps **Top of Head** to this joint
- These are ~10-15cm apart depending on head pose
- Currently excluded from evaluation (EVAL_JOINTS)

### Problematic — optimizer "Thorax" (joint 8)
- COCO19 GT maps **Base of Neck** (COCO[0]) directly
- MotionBert maps **Base of Neck** (MPII[8] "Neck") directly
- MediaPipe **computes midpoint(LShoulder, RShoulder)** — a different point
- Currently excluded from evaluation (EVAL_JOINTS)

### Problematic — optimizer "Hip" (joint 0)
- COCO19 GT has a **direct annotation** (BodyCenter[2])
- MediaPipe and MotionBert **compute** midpoint(LHip, RHip)
- Close but not guaranteed identical
- Currently excluded from evaluation (EVAL_JOINTS)

### Problematic — optimizer "Spine" (joint 7)
- All sources compute this as a midpoint, but of *different pairs*:
  - COCO19: midpoint(BodyCenter, Neck)
  - MediaPipe: midpoint(computed Hip, computed Thorax)
  - MotionBert: midpoint(Pelvis, Thorax)
- Currently excluded from evaluation (EVAL_JOINTS)

### Unused source joints
- COCO19: 4 face landmarks (LEye, REye, LEar, REar) — discarded
- MediaPipe: 20 landmarks (face mesh, hands, feet detail) — discarded
- MotionBert: none discarded from MPII 16 (all used)

## Source Code References
- Optimizer skeleton: `pose-optimizer/skeleton.py:11-28`
- COCO19 → H36M mapping: `pose-optimizer/skeleton.py:154-182`
- MPII → H36M mapping: `pose-optimizer/skeleton.py:113-147`
- MediaPipe → H36M mapping: `pose-optimizer/skeleton.py:201-233`
- Eval joints: `pose-optimizer/skeleton.py:30-32`
