# SGLD Optimization Implementation Summary

## What Was Implemented

A systematic 4-phase experimental framework to optimize SGLD noise for MPJPE performance while preventing gradient escape from sparse heatmap gradients.

## Files Created

### 1. `sgld_diagnostics.py` (Phase 1)
**Purpose:** Validate gradient escape hypothesis

**Features:**
- Runs 3 diagnostic experiments with different noise levels (0.0, 1e-6, 0.01)
- Tracks gradient L2 norms every 10 steps
- Tracks penalty-free heatmap scores every 10 steps
- Generates comparison plots and diagnostic summary table
- Calculates expected pixel displacements vs gradient basin radius (~15px)

**Output:** Plots and JSON data in `experiments/sgld-diagnostics-{timestamp}/`

### 2. `sgld_grid_search.py` (Phase 2)
**Purpose:** Find optimal noise + penalty combination

**Features:**
- **Phase 2.1 - Core Noise Sweep:** 7 noise levels (1e-7 to 1e-4)
- **Phase 2.2 - Penalty Tuning:** Grid search over position/rotation penalties
- **Phase 2.3 - LR Fine-Tune:** Learning rate variants
- Automatically selects best configs for next phase
- Sortable summary tables by MPJPE
- Can run phases independently or all together

**Output:** Complete results in `experiments/sgld-grid-{timestamp}/summary.json`

### 3. `sgld_ensemble.py` (Phase 4)
**Purpose:** Leverage SGLD exploration for ensemble methods

**Features:**
- Runs N independent SGLD optimizations (default: 10)
- **Best-of-N strategy:** Select run with lowest MPJPE
- **Ensemble average:** Average joint predictions across runs
- Computes pairwise diversity metrics (std > 0.05 target)
- Compares against no-noise baseline (0.401 MPJPE)

**Output:** Ensemble results in `experiments/sgld-ensemble-{timestamp}/summary.json`

### 4. `SGLD_OPTIMIZATION_PLAN.md`
Comprehensive documentation of the entire plan including:
- Background and hypothesis
- Execution order for all phases
- Usage examples for each script
- Success criteria and validation metrics
- Implementation details

## Files Modified

### `utils.py`
**Changes:**
- Added `track_diagnostics: bool` to `OptimizationConfig` (default: False)
- Added `diagnostic_interval: int` to `OptimizationConfig` (default: 10)
- Extended `OptimizationResult` with optional diagnostic fields:
  - `gradient_norms: list[float] | None`
  - `heatmap_scores: list[float] | None`
  - `pixel_displacements: list[float] | None`

### `optimization.py`
**Changes:**
- Added diagnostic tracking in optimization loop (lines 125-190)
- Computes gradient L2 norm before clipping when `track_diagnostics=True`
- Computes penalty-free heatmap scores for all frames
- Stores diagnostic data in `OptimizationResult`
- Tracks diagnostics every `diagnostic_interval` steps

## How to Use

### Quick Start

```bash
# Phase 1: Validate gradient escape (30 min)
uv run python sgld_diagnostics.py /path/to/video

# Phase 2: Find optimal config (2-3 hours)
uv run python sgld_grid_search.py /path/to/video

# Phase 4: Test ensemble methods (2 hours, optional)
uv run python sgld_ensemble.py /path/to/video --num-runs 10 \
  --noise-temperature 1e-6  # Use best from Phase 2
```

### Advanced Usage

```bash
# Run only core noise sweep
uv run python sgld_grid_search.py /path/to/video --phase core_noise

# Run penalty tuning with specific noise levels
uv run python sgld_grid_search.py /path/to/video --phase penalty_tuning \
  --best-noise 1e-6 5e-7

# Run ensemble with custom config
uv run python sgld_ensemble.py /path/to/video \
  --num-runs 20 \
  --noise-temperature 5e-7 \
  --position-penalty 0.25 \
  --ab-rotation-penalty 0.6 \
  --bc-rotation-penalty 0.3 \
  --learning-rate 0.4
```

## Key Insights

### Gradient Basin Radius
- **Heatmap std:** 3.0 pixels (defined in `video.py`)
- **Effective gradient radius:** ~15 pixels (5σ)
- **Implication:** SGLD noise causing >15px displacement will escape gradients

### Noise Scale Formula
```python
noise_scale = sqrt(2 * noise_temperature * learning_rate)
```

For `noise_temperature=0.01`, `lr=0.5`:
- `noise_scale = sqrt(2 * 0.01 * 0.5) = 0.1`
- Typical parameter change: ~0.1 radians
- Pixel displacement: ~20-40 pixels (exceeds gradient basin!)

### Success Metrics

**Primary Target:** MPJPE ≤ 0.45 (within 10% of no-noise baseline)
**Stretch Target:** MPJPE ≤ 0.40 (beat no-noise baseline)
**Diversity Target:** Pairwise std > 0.05 (effective exploration)

## Validation Workflow

1. **Phase 1 confirms hypothesis:**
   - Plot shows gradient collapse with high noise
   - Expected displacement exceeds gradient radius

2. **Phase 2 finds optimal config:**
   - Ranked by MPJPE (primary), score (secondary)
   - Best config achieves target MPJPE ≤ 0.45

3. **Phase 4 leverages exploration:**
   - Multiple runs show diversity (std > 0.05)
   - Best-of-N or ensemble beats baseline

## Expected Timeline

- **Phase 1:** 30 minutes (3 experiments × 100 steps)
- **Phase 2.1:** 45 minutes (7 experiments × 100 steps)
- **Phase 2.2:** 1-2 hours (12-36 experiments × 100 steps)
- **Phase 2.3:** 20 minutes (3 experiments × 100 steps)
- **Phase 4:** 2 hours (10-20 runs × 100 steps)

**Total:** ~4-6 hours for complete systematic optimization

## Phase 3 (Not Implemented)

Adaptive noise schedules were designed but not implemented because:
- Plan prioritizes finding good constant noise first (Phase 2)
- Early stopping if Phase 2 achieves MPJPE ≤ 0.42
- Can be added later if constant noise insufficient

**To implement Phase 3:**
1. Modify `optimization.py` lines 150-156 to support noise schedule function
2. Create `sgld_schedules.py` with linear/exponential/two-stage schedules
3. Test 3 schedules × best noise_start from Phase 2

## Next Steps

1. Run Phase 1 to validate gradient escape hypothesis
2. Run Phase 2 to find optimal configuration
3. Check if MPJPE ≤ 0.42:
   - **Yes:** Skip Phase 3, proceed to Phase 4 if targeting < 0.40
   - **No:** Implement and run Phase 3
4. Run Phase 4 if stretch goal (MPJPE < 0.40) is desired
5. Update `OptimizationConfig` defaults with best parameters found

## Integration with Existing Code

The implementation is **non-invasive**:
- Diagnostic tracking is opt-in (`track_diagnostics=False` by default)
- No changes to core optimization algorithm
- All experiments use existing `run_optimization()` and `evaluate_results()`
- Results follow established `experiment.py` pattern
- Backward compatible with existing scripts

## References

- **Original baseline:** MPJPE=0.401, score=-13.55 (no noise)
- **Problematic config:** MPJPE~18.79, score=-18.79 (noise=0.01)
- **Heatmap gradient basin:** `video.py:14` defines `HEATMAP_POINT_STD=3.0`
- **Plan transcript:** `.claude/projects/.../0b454b06-5f71-4b57-b9f5-880c68ac16d6.jsonl`
