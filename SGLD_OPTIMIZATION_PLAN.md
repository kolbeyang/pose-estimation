# SGLD Noise Optimization Plan

## Goal
Optimize SGLD noise_temperature for MPJPE while preventing gradient escape from sparse heatmap gradients.

**Target:** MPJPE ≤ 0.45 with noise_temperature > 0
**Stretch:** MPJPE ≤ 0.40 (beat no-noise baseline of 0.401)

## Background

Current SGLD implementation with `noise_temperature=0.01` achieves poor MPJPE (~18.79) compared to no-noise baseline (0.401). Investigation revealed:

- **Sparse heatmaps:** 3-pixel Gaussian std → gradients only within ~15-pixel radius
- **Gradient escape:** SGLD noise at 0.01 causes ~20-40 pixel parameter jumps
- **Problem:** Optimization falls outside gradient zones and loses signal

## Execution Order

### Phase 1: Validate Gradient Escape Hypothesis ✓

**Script:** `sgld_diagnostics.py`

**Purpose:** Measure if SGLD noise causes "falling off" sparse heatmap gradients

**Usage:**
```bash
uv run python sgld_diagnostics.py /path/to/video
```

**Output:**
- `experiments/sgld-diagnostics-{timestamp}/`
  - `gradient_norms.png` - Gradient L2 norms over time
  - `heatmap_scores.png` - Penalty-free heatmap fit over time
  - `diagnostic_summary.png` - Summary table with collapse analysis
  - `*.json` - Raw diagnostic data

**Experiments:**
1. No noise (baseline): `noise_temperature=0.0`
2. Tiny noise: `noise_temperature=1e-6`
3. Problematic noise: `noise_temperature=0.01`

**Expected outcome:** High noise shows gradient collapse and score degradation

---

### Phase 2: Noise-Penalty Grid Search ✓

**Script:** `sgld_grid_search.py`

**Purpose:** Find optimal noise + penalty combination achieving MPJPE ≤ 0.45

**Usage:**
```bash
# Run all phases
uv run python sgld_grid_search.py /path/to/video

# Run individual phases
uv run python sgld_grid_search.py /path/to/video --phase core_noise
uv run python sgld_grid_search.py /path/to/video --phase penalty_tuning --best-noise 1e-6 5e-7
uv run python sgld_grid_search.py /path/to/video --phase lr_tuning
```

**Output:**
- `experiments/sgld-grid-{timestamp}/summary.json` - Best configs sorted by MPJPE

#### Phase 2.1: Core Noise Sweep (7 experiments)
- **Fixed:** pos=0.2, ab=0.5, bc=0.3, lr=0.5
- **Sweep:** noise_temperature in [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
- **Selects:** Best 2 noise levels for penalty tuning

#### Phase 2.2: Penalty Tuning (12-36 experiments)
- **Uses:** Best 1-2 noise levels from Phase 2.1
- **Grid:**
  - position_penalty_weight: [0.15, 0.2, 0.25, 0.3]
  - ab_rotation_penalty_weight: [0.4, 0.5, 0.6]
  - bc_rotation_penalty_weight: [0.2, 0.3, 0.4]

#### Phase 2.3: LR Fine-Tune (3 experiments)
- **Uses:** Best noise+penalty combo from Phase 2.2
- **Sweep:** learning_rate in [0.3, 0.4, 0.5]

**Early stopping:** If Phase 2 achieves MPJPE ≤ 0.42, skip Phase 3

---

### Phase 3: Adaptive Noise Schedules (Optional)

**Status:** Not yet implemented
**Skip if:** Phase 2 finds good constant noise (MPJPE ≤ 0.42)

**Purpose:** Test if decaying noise over time improves convergence

**Schedules to test:**
1. Linear decay: `noise(t) = noise_start * (1 - t/T) + noise_end * (t/T)`
2. Exponential decay: `noise(t) = noise_start * (noise_end/noise_start)^(t/T)`
3. Two-stage: High noise (40% steps) → low noise (60% steps)

**Implementation required:** Modify `optimization.py` lines 150-156 to support noise schedules

---

### Phase 4: Multi-Run Ensemble ✓

**Script:** `sgld_ensemble.py`

**Purpose:** Use SGLD exploration to generate diverse solutions, select/average for better MPJPE

**Usage:**
```bash
uv run python sgld_ensemble.py /path/to/video \
  --num-runs 10 \
  --noise-temperature 1e-6 \
  --position-penalty 0.2 \
  --ab-rotation-penalty 0.5 \
  --bc-rotation-penalty 0.3 \
  --learning-rate 0.5
```

**Strategies:**
1. **Best-of-N:** Run N times, select run with lowest MPJPE
2. **Ensemble average:** Average joint predictions across N runs

**Target:** MPJPE < 0.40 via ensemble (beat no-noise baseline)

**Output:**
- `experiments/sgld-ensemble-{timestamp}/summary.json`
  - Individual run MPJPEs
  - Best-of-N MPJPE
  - Ensemble average MPJPE
  - Diversity metrics (pairwise std)

**Validation:** Diversity std > 0.05 indicates effective exploration

---

## Implementation Details

### Modified Files

**`utils.py`** (lines 14-27):
- Added `track_diagnostics: bool = False`
- Added `diagnostic_interval: int = 10`
- Added diagnostic fields to `OptimizationResult`:
  - `gradient_norms: list[float] | None`
  - `heatmap_scores: list[float] | None`
  - `pixel_displacements: list[float] | None`

**`optimization.py`** (lines 125-190):
- Added diagnostic tracking in optimization loop
- Tracks gradient L2 norms every N steps
- Tracks penalty-free heatmap scores every N steps
- Returns diagnostic data in `OptimizationResult`

### New Files

1. **`sgld_diagnostics.py`** - Phase 1 diagnostic experiments
2. **`sgld_grid_search.py`** - Phase 2 systematic grid search
3. **`sgld_ensemble.py`** - Phase 4 ensemble strategies

---

## Success Criteria

### Primary
- MPJPE ≤ 0.45 with `noise_temperature > 0` (within 10% of no-noise baseline)

### Stretch
- MPJPE ≤ 0.40 with ensemble (beat no-noise baseline)

### Validation
- Runs show diversity (pairwise MPJPE std > 0.05 across runs)

---

## Key Parameters

### Heatmap Gradient Basin
- **Source:** `video.py` line 14: `HEATMAP_POINT_STD = 3.0`
- **Gradient radius:** ~15 pixels (5σ)
- **Implication:** Parameter changes causing >15px projection shifts escape gradients

### Baseline Performance (No Noise)
- **MPJPE:** 0.401
- **Score:** -13.55
- **Config:** lr=0.5, pos_penalty=0.2, 100 steps, noise_temp=0.0

---

## Next Steps

1. **Run Phase 1** to confirm gradient escape hypothesis
2. **Run Phase 2** to find optimal noise + penalty combination
3. **If MPJPE ≤ 0.42:** Skip Phase 3, proceed to Phase 4
4. **If MPJPE > 0.42:** Implement and run Phase 3 (noise schedules)
5. **Run Phase 4** if targeting MPJPE < 0.40 (beat baseline)

---

## References

- **Plan source:** `/Users/kolbeyang/.claude/projects/.../0b454b06-5f71-4b57-b9f5-880c68ac16d6.jsonl`
- **Experiment pattern:** `experiment.py`
- **Heatmap std definition:** `video.py:14`
