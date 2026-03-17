# Tester Plan: Phase 1, Iteration 2

**Date:** 2026-03-17
**Plan covers:** P1-02 -- Revert to Pairwise Depth, Remove Dead Config
**Based on:** ARCHITECT_PLAN_P1_02.md

---

## What to Test

### 1. Code cleanup verifications (config, detect, optimize, scoring)

| Item | Check |
|------|-------|
| `SIGMA_SCHEDULE` removed from `config.py` | `hasattr(cfg, 'SIGMA_SCHEDULE')` == False |
| `HEATMAP_BLUR_SCHEDULE` removed from `config.py` | `hasattr(cfg, 'HEATMAP_BLUR_SCHEDULE')` == False |
| `FK_TARGET_CONF_THRESHOLD` removed from `config.py` | `hasattr(cfg, 'FK_TARGET_CONF_THRESHOLD')` == False |
| `USE_REAL_HEATMAPS` removed from `config.py` | `hasattr(cfg, 'USE_REAL_HEATMAPS')` == False |
| `motionbert_to_camera_space_batch` removed from `detect.py` | `hasattr(detect, 'motionbert_to_camera_space_batch')` == False |
| `_get_sigma` removed from `optimize.py` | not in `dir(optimize)` |
| `_get_blur_sigma` kept in `optimize.py` (needed for sweep) | in `dir(optimize)` |
| `use_real_heatmaps` removed from `compute_total_score` signature | not in `inspect.signature` params |

### 2. sweep.py integrity

| Item | Check |
|------|-------|
| `sigma_schedule` field removed from `SweepConfig` | not in `dataclass.__dict__` |
| `all_joints_smooth_weight` still absent | not in `dataclass.__dict__` |
| `get_phase1_2_configs()` returns 17 configs | `len(configs) == 17` |
| sweep.py imports without crash | `from sweep import SweepConfig, get_phase1_2_configs` |

### 3. main.py regression

| Item | Check |
|------|-------|
| `motionbert_to_camera_space` (not batch) imported | grep/inspect |
| `improved_target_2d` removed | not in source |
| `FK_TARGET_CONF_THRESHOLD` not referenced | not in source |
| No `NUM_JOINTS` stale import | not in source |

### 4. Z-value analysis

Pull frame 0 and several representative frames from both examples.

- Root Z should be in the range [1, 8] m (physically plausible)
- No negative Z values (camera behind subject)
- GT root Z should be in range [2.5, 2.7] m for Ex0 (person ~2.6 m from camera)
- Frame-to-frame Z variation should be smooth (no 1+ m jumps)

### 5. MPJVE spec requirement

Spec: "MPJVE should be improved for ALL test videos vs round-4 baseline."

Round-4 baseline:
- Example 0: Det MPJVE 0.94 cm/f, Opt MPJVE 1.05 cm/f
- Example 5: Det MPJVE 0.51 cm/f, Opt MPJVE 0.48 cm/f

Test: run both examples, compare Det MPJVE and Opt MPJVE to baseline.

### 6. Overlay video frame analysis

Extract frame 0 from each example's overlay video. Visually verify:
- Heatmaps visible and in correct positions
- Skeleton projection overlaid on person
- No obviously wrong joint positions

### 7. Other metrics vs round-4

Confirm no catastrophic regression in:
- Det MPJPE (should match round-4 ± trivially)
- Det P-MPJPE (should match round-4 ± trivially)
- All metrics plausible and non-NaN

---

## How to Test

### Smoke test: run both examples
```bash
cd .../motionbert-pose && uv run python -c "
from main import process_example
import config as cfg, os
run_dir = os.path.join(cfg.TRAINING_RUNS_DIR, 'p1-pairwise-revert')
os.makedirs(run_dir, exist_ok=True)
seq, cam, start, nf, pidx = cfg.EXAMPLES[0]
process_example(seq, cam, start, nf, pidx, run_dir)
"
```
(repeat for index 5)

### Config verification
```bash
uv run python -c "import config as cfg; ..."
```

### sweep.py verification
```bash
uv run python -c "from sweep import SweepConfig, get_phase1_2_configs; ..."
```

### JSON analysis: extract metrics, Z-values from saved predictions

### Visual analysis: extract frames from overlay MP4 with cv2

---

## Regression Checks

- `from main import process_example` imports cleanly
- `from sweep import ...` imports cleanly
- All 4 removed config constants absent
- `motionbert_to_camera_space_batch` absent from detect
- `_get_blur_sigma` still present in optimize (needed for sweep schedules)
- Metrics comparable to round-4 baseline (Det MPJPE within 0.1 cm)
