# Improved Single Camera 3D Pose Estimation with Reprojection and Filtering

Kolbe Yang — RIT Computer Science Capstone (20261)

## Abstract

This paper presents a post-processing process for improving single-camera 3D pose estimation. Rather than training a model, we reformulate pose estimation as a constrained optimization problem improving on the raw outputs of a base state-of-the-art 3D pose estimation model. Given heatmaps from a 2D pose-estimation model and raw 3D estimations from a base 3D pose-estimation model, we optimize a rigid skeleton initialized on the observations of our upstream models. Our rigid skeleton is parameterized by a position, angles, and bone lengths. This optimization enforces temporal smoothness, consistent bone lengths, and adherence to observed 2D heatmaps via reprojection. The entire pipeline is implemented in PyTorch and evaluated on 17 sequences from the CMU Panoptic dataset using two base 3D pose-estimators: MotionBERT and MediaPipe. Our method achieves consistent position error reduction of 7.4% for MotionBERT and 1.6% for MediaPipe as well as consistent velocity error reduction of 33.2% for MotionBERT and 15.9% for MediaPipe across all sequences, demonstrating that post-process optimization is a viable method for improving 3D pose estimations.

## Structure

```
pose-optimizer/   Main pipeline — estimators, optimizer, evaluation
toy-arm/          2-segment toy-arm reference implementation
```

See [`pose-optimizer/README.md`](pose-optimizer/README.md) for setup, configuration, and usage.

## Branches

- `main` — clean snapshot of the final pipeline.
- `dev` — full development history (~180 commits), including earlier MediaPipe-only and MotionBERT-only standalone implementations and the agent-driven experiment logs.
