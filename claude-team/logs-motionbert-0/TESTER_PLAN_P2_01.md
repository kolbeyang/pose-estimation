# Test Plan: Phase 2, Iteration 1

## Objective
Verify that the MPJPE bug fix (bone-length matching) and FK optimization meet Phase 2 Definition of Done.

## Tests

### T1: Smoke Test
Run `uv run python main.py` from `motionbert-pose/` on all 10 examples. Must complete without errors.

### T2: Pyright Issues Check
Verify that the reported Pyright issues (tuple size mismatch, undefined function) are stale and the code is correct.

### T3: MPJPE Baseline Check
Verify mean MPJPE improved from 44.89 cm to ~29 cm with bone-length matching.

### T4: Optimization Improvement
- MPJPE improves (decreases) on a majority of examples after optimization
- No example regresses by more than 2 cm
- Mean MPJPE is lower after optimization than before (baseline)

### T5: Results Output Validation
- Results JSON saved per example with both det_* and opt_* metrics
- Graphs generated per example

### T6: Code Review
- Type hints on all functions
- No imports from sibling directories
- Pydantic validation present

### T7: MPJPE Realism Assessment
Assess whether 29 cm MPJPE is realistic given the evaluation setup (CMU Panoptic, single camera, Stacked Hourglass 2D).
