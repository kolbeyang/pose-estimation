# Test Plan: Phase 1, Iteration 0

## What to test

1. **Code review: `graphs.py` - `generate_limb_length_graph` function**
   - Correct joint indices (13-14, 14-15, 10-11, 11-12 per skeleton.py)
   - Correct colors: blue=GT, green=Det, red=Opt
   - Correct linestyles per limb segment (solid, dashed, dotted, dashdot)
   - NaN handling for missing GT frames
   - Legend with 12 entries, placed outside plot
   - Saved to `limb_lengths.png`

2. **Code review: `graphs.py` - summary grid subplot at axes[3,3]**
   - Same 12 lines with compact labels
   - Correct indexing into `det` and `opt` numpy arrays

3. **Code review: `main.py` - integration**
   - `generate_limb_length_graph` imported and called with correct arguments
   - Called after other graph generation, before summary

4. **Smoke test: run a single example**
   - Run `uv run python main.py` (or a single-example variant) and verify it completes
   - Verify `limb_lengths.png` is produced in the example graph directory

5. **Output validation: sanity check the graph**
   - GT lines (blue) should be near-constant (real arms don't change length)
   - All limb lengths in range ~0.2-0.35m
   - 12 lines visible
   - Optimized (red) should be flatter than detector (green)

## How to test

- Code review: Read `graphs.py` lines 273-340 and 536-567, `main.py` lines 27 and 327-329
- Smoke test: `cd /Users/kolbeyang/Documents/School/spring_2026/capstone/pose-estimation/motionbert-pose && uv run python main.py` (limit to 1 example if possible)
- Output validation: Read the generated PNG image file

## Regression checks

- Verify `main.py` still runs end-to-end without errors
- Verify summary.png is still generated (axes[3,3] was previously unused)
