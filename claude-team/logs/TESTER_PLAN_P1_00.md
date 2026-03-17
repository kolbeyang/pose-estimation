# Tester Plan: Phase 1, Iteration 0 -- solvePnP + Remove ALL_JOINTS_SMOOTH_WEIGHT

**Date:** 2026-03-17
**Spec:** Phase 1 of `claude-team/specs/motion-bert-round-5.md`
**Developer Report:** `claude-team/logs/DEVELOPER_REPORT_P1_00.md`

## What to Test

### T1: ALL_JOINTS_SMOOTH_WEIGHT fully removed
- Verify no references in `config.py`, `scoring.py`, `optimize.py`
- Verify `motion_penalty_all_joints()` function removed from `scoring.py`
- Grep entire motionbert-pose/ for stale references

### T2: solvePnP implementation code review
- Code review `detect.py` solvePnP implementation
- Verify camera matrix K constructed correctly
- Verify visibility filtering logic
- Verify fallback path for <4 visible joints or solvePnP failure
- Verify safety check on root Z range

### T3: Smoke test -- run example 0 end-to-end
- Run `process_example()` for example 0 (171204_pose1_sample_0, 100 frames)
- Confirm no errors, predictions JSON produced

### T4: Z-value trajectory analysis
- Load prediction JSON for example 0
- Extract root Z values across frames for detection and optimized
- Check range (should be 1-10m), check smoothness (frame-to-frame std dev)
- Compare against GT root Z

### T5: MPJPE comparison vs round-4 baseline
- Cross-check developer's reported numbers against actual JSON output
- Round-4: Ex 0 Det 30.98, Opt 30.44; Ex 5 Det 15.85, Opt 15.44

### T6: MPJVE comparison -- critical spec requirement
- Spec requires MPJVE improvement for ALL test videos
- Round-4: Ex 0 Det MPJVE 0.94, Opt MPJVE 1.05; Ex 5 Det MPJVE 0.51, Opt MPJVE 0.48

### T7: Verify developer claim -- solvePnP produced WORSE results
- Independent verification by loading prediction JSONs and recomputing metrics

### T8: Unused function removal verification
- Verify `_enforce_bone_lengths_with_2d()` and `_reconstruct_from_2d()` removed from detect.py

## Regression Checks
- Pipeline still runs end-to-end without errors
- `scoring.py` functions work without the removed parameter
- No import errors from removed functions
