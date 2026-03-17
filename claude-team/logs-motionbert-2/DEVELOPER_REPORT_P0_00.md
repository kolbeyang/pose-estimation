# Developer Report P0-00: Add CMU Panoptic Examples

## Summary

Downloaded 3 new CMU Panoptic sequences and added 8 new example clips to the config, bringing the total from 10 to 18 examples across 8 sequences.

## Downloaded Sequences

| Sequence | GT Frames | Frame Range | People | Video | Calibration |
|----------|-----------|-------------|--------|-------|-------------|
| `171204_pose1` | 27,561 | 118-27,678 | 1 | hd_00_00.mp4 (2.9GB) | Yes |
| `160226_haggling1` | 11,465 | 129-11,593 | 3 | hd_00_00.mp4 (1.3GB) | Yes |
| `160906_pizza1` | 6,582 | 112-6,693 | 6 | hd_00_00.mp4 (819MB) | Yes |

## female_example_01 Status

- **Valid**: 301 GT frames (300-600), 1 person, HD video present
- **Calibration fix**: Original file was named `calibration_171204_pose2.json` (wrong name for the loader). Copied it to `calibration_female_example_01.json` so `load_calibration()` can find it.

## New Examples Added to config.py

| # | Sequence | Start | Frames | Person | Notes |
|---|----------|-------|--------|--------|-------|
| 11 | `171204_pose1` | 5000 | 150 | 0 | Single person, early |
| 12 | `171204_pose1` | 14000 | 150 | 0 | Single person, middle |
| 13 | `171204_pose1` | 22000 | 150 | 0 | Single person, late |
| 14 | `160226_haggling1` | 3000 | 150 | 0 | 3-person haggling scene |
| 15 | `160226_haggling1` | 7000 | 150 | 0 | 3-person haggling scene |
| 16 | `160906_pizza1` | 2000 | 150 | 0 | 6-person pizza scene |
| 17 | `160906_pizza1` | 4500 | 150 | 0 | 6-person pizza scene |
| 18 | `female_example_01` | 350 | 150 | 0 | Single person |

All 8 new clips verified to have 150/150 GT frames present.

## Files Modified

- `motionbert-pose/config.py` - Added 8 new entries to EXAMPLES list

## Files Created (external, not in repo)

- `panoptic-toolbox/female_example_01/calibration_female_example_01.json` (copy of existing calibration)
- `panoptic-toolbox/171204_pose1/` (new download)
- `panoptic-toolbox/160226_haggling1/` (new download)
- `panoptic-toolbox/160906_pizza1/` (new download)

## Notes

- Tar archives were cleaned up after extraction to save disk space.
- `160226_haggling1` and `160906_pizza1` are multi-person scenes. The config uses `person_idx=0` which selects the first person detected. This should work but the person identity may not be consistent across frames in crowded scenes.
- `171204_pose1` is the full version of `171204_pose1_sample` (27k frames vs the sample's limited set).
