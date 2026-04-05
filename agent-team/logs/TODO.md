# TODO

- [ ] (low) `overlay_video.py` line 1 docstring still says "real SH or synthetic Gaussian" -- update to reflect that only SH heatmaps are used now
- [ ] (medium) `run_mediapipe/__init__.py` lines 376-407 still generates synthetic heatmaps for overlay video instead of using SH heatmaps+affine like `main.py` does -- this means the standalone MediaPipe pipeline produces misaligned overlay videos
- [ ] (low) `run_mediapipe/__init__.py` line 376 stale comment: "use synthetic heatmaps for visualization"
