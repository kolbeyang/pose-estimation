# Pose Estimation Capstone

Monocular 3D human pose estimation using CMU Panoptic Studio data. Compares MediaPipe and MotionBert base estimators, with a shared differentiable optimizer that refines predictions using 2D heatmap evidence.

## Project Structure

```
pose-optimizer/     Main codebase — unified pipeline for both estimators
mediapipe-pose/     Original MediaPipe standalone implementation (archived)
motionbert-pose/    Original MotionBert standalone implementation (archived)
claude-team/        Agent orchestration specs and logs
```

See [`pose-optimizer/README.md`](pose-optimizer/README.md) for usage and details.
