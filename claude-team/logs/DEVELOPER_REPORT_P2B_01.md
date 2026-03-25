# Developer Report: Phase 2b, Iteration 1 -- SZI-MPJPE Bug Fixes

## What was implemented

Three changes per the architect's plan:

1. **Clarifying docstring on `szi_mpjpe()` in evaluate.py** -- Added a note explaining that the optimal scale minimizes SSE (sum of squared errors), not MPJPE (mean of L2 norms), so SZI-MPJPE is NOT guaranteed to be <= MPJPE. This documents the behavior the tester flagged as Bug #1.

2. **Fixed unit test in test_szi_mpjpe.py** -- Replaced the 5 "SZI-MPJPE <= MPJPE" trials (which tested a property that does not hold in general) with 5 "SSE after scaling <= SSE before scaling" trials (which IS guaranteed by the math). Test count remains 18.

3. **Added SZI-MPJPE prints to test_single.py** -- Added print lines for `det_szi_mpjpe`, `opt_szi_mpjpe`, `det_szi_mpjpe_no_ankles`, and `opt_szi_mpjpe_no_ankles` in the RESULTS block, matching the pattern from main.py.

## Files changed

- `motionbert-pose/evaluate.py` -- Added docstring note to `szi_mpjpe()` (lines 217-220)
- `motionbert-pose/test_szi_mpjpe.py` -- Replaced "SZI-MPJPE <= MPJPE" test with "SSE after scaling <= SSE before" test (lines 124-134)
- `motionbert-pose/test_single.py` -- Added 4 SZI-MPJPE print statements in the RESULTS block (lines 149, 153-154, 158, 162-163)

## Commands run

1. `cd motionbert-pose && uv run python test_szi_mpjpe.py` -- **18/18 tests PASSED**. The new SSE property test passes on all 5 random trials. Synthetic and real visualizations generated successfully.

2. `cd motionbert-pose && uv run python test_single.py` -- **PASSED**. SZI-MPJPE values now appear in stdout:
   - Det SZI-MPJPE: 53.87 cm (scale=0.0812)
   - Opt SZI-MPJPE: 52.71 cm (scale=0.1402)
   - Det SZI-MPJPE (no ankles): 30.20 cm
   - Opt SZI-MPJPE (no ankles): 14.05 cm

## Decisions made

None. The plan was unambiguous and all three changes were straightforward.

## Concerns

None. These are documentation, test correction, and print-statement changes with no risk of regression.

## Deviations from plan

None. Implemented exactly as specified.
