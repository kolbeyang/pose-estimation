# Revise mapping

Here is the revised mapping.

I think the only change is the handling of the neck joint but please confirm this is true.

In addition let's make sure that the skeleton representations for MotionBert connects the shoulders to the new neck joint (which is called "spine").

The joint that is called "neck" for motionbert and the ground truth can be omitted from the pipeline because it does not exist in mediapipe.

### Category 1: Evaluation Joints

**USED IN**: Optimization, Evaluation, Visualization

| # | Joint | Stacked Hourglass (MPII 16) | MotionBert (MPII) | MediaPipe | CMU GT (COCO19) |
|---|-------|----------------------------|-------------------|-----------|-----------------|
| 1 | R Hip | SH[2] RHip | MPII[2] | MP[24] | COCO[12] |
| 2 | R Knee | SH[1] RKnee | MPII[1] | MP[26] | COCO[13] |
| 3 | R Ankle | SH[0] RAnkle | MPII[0] | MP[28] | COCO[14] |
| 4 | L Hip | SH[3] LHip | MPII[3] | MP[23] | COCO[6] |
| 5 | L Knee | SH[4] LKnee | MPII[4] | MP[25] | COCO[7] |
| 6 | L Ankle | SH[5] LAnkle | MPII[5] | MP[27] | COCO[8] |
| 7 | R Shoulder | SH[12] RShoulder | MPII[12] | MP[12] | COCO[9] |
| 8 | R Elbow | SH[11] RElbow | MPII[11] | MP[14] | COCO[10] |
| 9 | R Wrist | SH[10] RWrist | MPII[10] | MP[16] | COCO[11] |
| 10 | L Shoulder | SH[13] LShoulder | MPII[13] | MP[11] | COCO[3] |
| 11 | L Elbow | SH[14] LElbow | MPII[14] | MP[13] | COCO[4] |
| 12 | L Wrist | SH[15] LWrist | MPII[15] | MP[15] | COCO[5] |

### Category 2: Synthesized Evaluation Joints

Direct in MotionBert + CMU GT, synthesized for MediaPipe. 

**USED IN**: Optimization, Evaluation, Visualization

| # | Joint | Stacked Hourglass (MPII 16) | MotionBert (MPII) | MediaPipe | CMU GT (COCO19) |
|---|-------|----------------------------|-------------------|-----------|-----------------|
| 13 | Pelvis | SH[6] Pelvis | MPII[6] direct | midpoint(LHip, RHip) | COCO[2] direct |
| 14 | Neck | SH[7] Thorax | MB joint 7 "Spine" | midpoint(LShoulder, RShoulder) (synthetic) | COCO[0] Neck |

NOTE: neck is the most confusing joint here. What we are calling neck is called "spine" in Motionbert and "thorax" Stacked Hourglass. The joint called "neck" in Motionbert should be unused for the purposes of this pipeline. 

### Category 3: Synthesized Joint

Stacked Hourglass (MPII) does not have a nose keypoint. It has HeadTop (SH[9]) and UpperNeck (SH[8]) but nothing in between. MotionBert's training data synthesized the "Nose" 2D input as the midpoint of UpperNeck(SH[8]) and HeadTop(SH[9]). We currently synthesize it as 30% from UpperNeck toward HeadTop.

nose = upper_neck + (head_top - upper_neck) * 0.3

Note: MotionBert DOES output a 3D prediction for this joint (H36M joint 9, "Neck/Nose"), trained against the H36M mocap marker at that position.

**USED IN**: Optimization, Visualization, Evaluation

| # | Joint | Stacked Hourglass (MPII 16) | MotionBert (MPII) | MediaPipe | CMU GT (COCO19) |
|---|-------|----------------------------|-------------------|-----------|-----------------|
| 15 | Nose | **not available** (synthesized from SH[8] UpperNeck + SH[9] HeadTop) | MB joint 9 "Neck/Nose" (synthesized 2D input, trained 3D output) | MP[0] direct | COCO[1] |

### Additional Stacked Hourglass Joints (not in our skeleton)

| SH Index | Joint | Notes |
|----------|-------|-------|
| SH[8] | UpperNeck | Anatomical neck, above shoulder level. Not directly mapped to our skeleton. |
| SH[9] | HeadTop | Used to synthesize Nose, then stripped from MotionBert output |
