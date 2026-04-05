# Report 03: TODO Items 4, 5, 6, 8 Fixes

Date: 2026-03-29

## Item 4: Multi-person sequences in full run configs

Replaced all 6 examples from `160422_ultimatum1` (3) and `160906_pizza1` (3) with additional windows from single-subject sequences:
- `171204_pose1` frames 10000, 18000
- `171204_pose2` frames 20000, 30000
- `171204_pose3` frames 2000, 6000

Updated in `motionbert/run_full_config.json`, `mediapipe/run_full_config.json`, and `cmu_data.py:EXAMPLES`. All 25 examples now use only single-subject sequences (`171204_pose1`, `171204_pose1_sample`, `171204_pose2`, `171204_pose3`, `female_example_01`).

## Item 5: generate_per_frame_mpjve imported but never called

Added per-frame velocity error (MPJVE) computation in both `motionbert/__init__.py` and `mediapipe/__init__.py`. The code computes frame-to-frame velocity differences for detector, optimizer, and ground truth, then stores them as `det_per_frame_mpjve` and `opt_per_frame_mpjve` in the metrics dict. Added a call to `generate_per_frame_mpjve()` after the existing `generate_per_frame_mpjpe()` call, gated on the metric being present. Requires at least 3 GT frames to produce meaningful velocity data.

## Item 6: data/ directory empty and misleading

Created a symlink `pose-optimizer/data/panoptic-toolbox -> ../../data/panoptic-toolbox`. Updated `cmu_data.py:DEFAULT_DATA_ROOT` to resolve to `<repo>/data/panoptic-toolbox` (was `../data/panoptic-toolbox`). Updated `data_root` in all 4 config files (`motionbert/run_full_config.json`, `motionbert/run_single_config.json`, `mediapipe/run_full_config.json`, `mediapipe/run_single_config.json`) and the default in `config.py:RunConfig` from `../data/panoptic-toolbox` to `data/panoptic-toolbox`.

## Item 8: Duplicate gaussian_sigma and heatmap_sigma

Removed `gaussian_sigma` from `config.py:RunConfig` (was a top-level field). Updated `mediapipe/__init__.py` line 178 to use `config.optimization.heatmap_sigma` instead of `config.gaussian_sigma`. Removed `gaussian_sigma` key from `mediapipe/run_full_config.json` and `mediapipe/run_single_config.json`. The single source of truth is now `optimization.heatmap_sigma` (default 50.0).

## Test Results

All 55 tests pass across all 3 test files (test_phase1: 28, test_phase2: 22, test_phase3: 5).
