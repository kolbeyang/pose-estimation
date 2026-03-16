# Architect Plan P1-01: Heatmap Blur Sweep (Phase 1.2)

## Goal Summary

Phase 1.1 showed that penalty weights are irrelevant at 20 steps and that at 100 steps, 2D alignment improves but 3D MPJPE worsens (the optimizer overfits to noisy 2D targets, sacrificing depth). The hypothesis for Phase 1.2 is that applying Gaussian blur to the 64x64 Stacked Hourglass heatmaps will widen the gradient basin, allowing the optimizer to converge faster with less overshoot. This plan adds heatmap blur capability, sweeps blur sigma values at multiple step counts, adds MPJVE to the console output (tester recommendation CR-2), and considers a coarse-to-fine blur schedule.

## Key Insight from Phase 1.1

- At 100 steps, baseline: MPJPE 31.10 cm (+0.66 worse), 2D-vs-Det 28.2 px (-5.6 better), MPJVE 1.36 cm/f (+0.31 worse)
- The optimizer can find better 2D alignment, but this destroys depth accuracy
- Blur should provide gentler gradients that guide joints to the right region without demanding pixel-perfect 2D alignment
- Testing at 50-100 steps is the interesting regime (20 steps is too few for anything to matter)

## What Already Exists

The `sweep.py` already has:
- `SweepConfig.heatmap_blur_sigma` field (default 0.0)
- Blur application logic in `run_sweep_config()` using `scipy.ndimage.gaussian_filter`
- This means **no changes are needed to the blur infrastructure itself**

What's missing:
- A `get_phase1_2_configs()` function with blur sweep configurations
- MPJVE in the console summary table
- A way to select phase 1.2 configs from the command line
- Coarse-to-fine blur schedule support (blur that decreases during optimization)

## Files to Modify

1. **`motionbert-pose/sweep.py`** -- Add `get_phase1_2_configs()`, add MPJVE to console table, add CLI argument to select config set, add coarse-to-fine blur schedule support.

2. **`motionbert-pose/optimize.py`** -- Add support for a blur schedule that applies progressively less blur as optimization progresses. This requires accepting a blur schedule parameter and re-blurring heatmaps at phase transitions.

3. **`motionbert-pose/config.py`** -- Add `HEATMAP_BLUR_SIGMA` parameter (default 0.0, no blur).

## Files to Create

None.

## Step-by-Step Instructions

### Step 1: Add MPJVE to sweep console output

In `sweep.py`, modify the summary table printing section. Currently the header is:

```python
header = (
    f"{'Config':<40} {'Det MPJPE':>10} {'Opt MPJPE':>10} {'Improv':>8} "
    f"{'Opt P-MPJPE':>12} {'Det 2D-Det':>10} {'Opt 2D-Det':>10}"
)
```

Change to:

```python
header = (
    f"{'Config':<40} {'Det MPJPE':>10} {'Opt MPJPE':>10} {'Improv':>8} "
    f"{'Opt P-MPJPE':>12} {'Opt MPJVE':>10} {'Det 2D-Det':>10} {'Opt 2D-Det':>10}"
)
```

And in the per-config row printing, add MPJVE:

```python
opt_mpjve = m.get("opt_mpjve", 0) * 100 if m.get("opt_mpjve") is not None else 0.0
```

Include `{opt_mpjve:>10.2f}` between P-MPJPE and Det 2D-Det in the format string.

### Step 2: Add `HEATMAP_BLUR_SIGMA` to config.py

Add after the `SIGMA_SCHEDULE` block (around line 55):

```python
# Gaussian blur sigma applied to Stacked Hourglass heatmaps before optimization.
# Applied in 64x64 heatmap space. sigma=2 at 64x64 ~ 8px at 256x256 crop.
# 0 = no blur. Widens gradient basin for optimization.
HEATMAP_BLUR_SIGMA: float = 0.0
```

### Step 3: Add coarse-to-fine blur schedule to optimize.py

The key idea: instead of applying a fixed blur to the heatmaps before optimization starts, we allow a **blur schedule** that decreases during optimization. This is analogous to the existing `SIGMA_SCHEDULE` for the analytical Gaussian.

**In `optimize.py`**, modify `run_optimization` to accept and apply a blur schedule:

Add a new parameter to `run_optimization`:

```python
def run_optimization(
    initial_positions_cam: list[np.ndarray],
    target_2d: list[np.ndarray],
    visibility: list[np.ndarray],
    camera: Camera,
    num_steps: int | None = None,
    heatmaps: list[np.ndarray] | None = None,
    affine: np.ndarray | None = None,
    heatmap_blur_schedule: list[tuple[float, float]] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
```

The `heatmap_blur_schedule` has the same format as `SIGMA_SCHEDULE`: a list of `(fraction_of_steps, blur_sigma)` tuples. For example:
- `[(1.0, 4.0)]` = constant blur of sigma 4.0 throughout
- `[(0.3, 4.0), (0.7, 2.0), (1.0, 0.0)]` = blur 4 for first 30%, blur 2 for next 40%, no blur for final 30%

Add a helper function `_get_blur_sigma(step, num_steps, schedule)` analogous to `_get_sigma`:

```python
def _get_blur_sigma(step: int, num_steps: int, schedule: list[tuple[float, float]]) -> float:
    """Get heatmap blur sigma for current step from schedule."""
    progress: float = step / max(num_steps - 1, 1)
    for frac, sigma in schedule:
        if progress <= frac:
            return sigma
    return schedule[-1][1]
```

**Inside the optimization loop**, add blur application logic. This requires:
1. Keeping the original (unblurred) heatmaps as `heatmaps_t_orig`
2. Before the scoring call, check if blur sigma changed since last step
3. If changed, re-blur from originals and update `heatmaps_t`

Add this block right after the `heatmaps_t` / `affine_t` conversion (around line 140):

```python
# Store originals for re-blurring during coarse-to-fine schedule
heatmaps_t_orig: list[torch.Tensor] | None = None
if heatmaps_t is not None and heatmap_blur_schedule is not None:
    heatmaps_t_orig = [hm.clone() for hm in heatmaps_t]
    # Apply initial blur
    initial_blur = _get_blur_sigma(0, num_steps, heatmap_blur_schedule)
    if initial_blur > 0:
        heatmaps_t = _apply_blur_torch(heatmaps_t_orig, initial_blur)
current_blur_sigma: float = (
    _get_blur_sigma(0, num_steps, heatmap_blur_schedule)
    if heatmap_blur_schedule else 0.0
)
```

Add a helper to apply Gaussian blur using PyTorch (to keep everything on tensors):

```python
def _apply_blur_torch(
    heatmaps: list[torch.Tensor], sigma: float
) -> list[torch.Tensor]:
    """Apply Gaussian blur to heatmaps using scipy (called rarely, not per-step)."""
    import scipy.ndimage
    blurred = []
    for hm in heatmaps:
        hm_np = hm.numpy()
        blurred_np = np.stack([
            scipy.ndimage.gaussian_filter(hm_np[c], sigma=sigma)
            for c in range(hm_np.shape[0])
        ])
        blurred.append(torch.tensor(blurred_np, dtype=torch.float32))
    return blurred
```

Inside the step loop, before the `compute_total_score` call, add:

```python
# Update blur if schedule changed
if heatmaps_t_orig is not None and heatmap_blur_schedule is not None:
    new_blur = _get_blur_sigma(step, num_steps, heatmap_blur_schedule)
    if abs(new_blur - current_blur_sigma) > 1e-6:
        if new_blur > 0:
            heatmaps_t = _apply_blur_torch(heatmaps_t_orig, new_blur)
        else:
            heatmaps_t = [hm.clone() for hm in heatmaps_t_orig]
        current_blur_sigma = new_blur
        print(f"    [Step {step}] Heatmap blur sigma changed to {new_blur:.1f}")
```

Also add `blur={current_blur_sigma:.1f}` to the step logging line.

### Step 4: Update sweep.py to pass blur schedule

Modify `SweepConfig` to support a blur schedule in addition to the existing fixed `heatmap_blur_sigma`:

```python
@dataclass
class SweepConfig:
    """One parameter configuration to test."""
    name: str
    num_steps: int = 20
    position_penalty_weight: float = 50.0
    rotation_penalty_scalar: float = 10.0
    init_anchor_weight: float = 5.0
    all_joints_smooth_weight: float = 0.0
    sigma_schedule: list[tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 80.0)]
    )
    heatmap_blur_sigma: float = 0.0  # Fixed blur (applied before optimization)
    heatmap_blur_schedule: list[tuple[float, float]] | None = None  # Coarse-to-fine blur
```

In `run_sweep_config`, pass the blur schedule to `run_optimization`:

```python
optimized_3d, bone_lengths_final, loss_history = run_optimization(
    initial_positions_cam=data["det_cam_positions"],
    target_2d=data["improved_target_2d"],
    visibility=data["visibility"],
    camera=data["camera"],
    num_steps=config.num_steps,
    heatmaps=heatmaps,
    affine=data["affine"],
    heatmap_blur_schedule=config.heatmap_blur_schedule,
)
```

Note: when `heatmap_blur_sigma > 0` AND `heatmap_blur_schedule` is None, the existing pre-blur logic in `run_sweep_config` applies the fixed blur before passing to `run_optimization`. When `heatmap_blur_schedule` is set, the fixed pre-blur should be skipped (set `heatmap_blur_sigma = 0` in the config). The developer should ensure these two mechanisms don't conflict -- if `heatmap_blur_schedule` is set, skip the `scipy` pre-blur in `run_sweep_config`.

Add this guard in `run_sweep_config` before the existing blur block:

```python
# Optionally blur heatmaps (fixed, pre-optimization)
heatmaps = data["heatmaps"]
if config.heatmap_blur_sigma > 0 and config.heatmap_blur_schedule is None:
    # Fixed blur only if no dynamic schedule
    import scipy.ndimage
    heatmaps = [...]  # existing code
```

### Step 5: Create `get_phase1_2_configs()` in sweep.py

Add this function. The sweep tests:
1. **Fixed blur at multiple step counts**: blur sigma 0, 1, 2, 4, 8 at 50 and 100 steps
2. **Coarse-to-fine blur schedules**: start wide, narrow down
3. Uses baseline penalty weights (from Phase 1.1: penalties don't matter much, so keep defaults)

```python
def get_phase1_2_configs() -> list[SweepConfig]:
    """Return Phase 1.2 heatmap blur sweep configurations."""
    configs: list[SweepConfig] = []

    # Common penalty settings (baseline -- Phase 1.1 showed these don't matter much)
    base = dict(
        position_penalty_weight=50.0,
        rotation_penalty_scalar=10.0,
        init_anchor_weight=5.0,
    )

    # --- Baseline (no blur) at multiple step counts ---
    for steps in [20, 50, 100]:
        configs.append(SweepConfig(
            name=f"no_blur_{steps}s",
            num_steps=steps,
            **base,
        ))

    # --- Fixed blur sweep at 50 steps ---
    for sigma in [1.0, 2.0, 4.0, 8.0]:
        configs.append(SweepConfig(
            name=f"blur{sigma:.0f}_{50}s",
            num_steps=50,
            heatmap_blur_sigma=sigma,
            **base,
        ))

    # --- Fixed blur sweep at 100 steps ---
    for sigma in [1.0, 2.0, 4.0, 8.0]:
        configs.append(SweepConfig(
            name=f"blur{sigma:.0f}_{100}s",
            num_steps=100,
            heatmap_blur_sigma=sigma,
            **base,
        ))

    # --- Coarse-to-fine blur schedules at 100 steps ---
    # Schedule: (fraction_of_steps, blur_sigma)
    configs.append(SweepConfig(
        name="c2f_8to0_100s",
        num_steps=100,
        heatmap_blur_schedule=[(0.3, 8.0), (0.7, 4.0), (1.0, 0.0)],
        **base,
    ))
    configs.append(SweepConfig(
        name="c2f_4to0_100s",
        num_steps=100,
        heatmap_blur_schedule=[(0.3, 4.0), (0.7, 2.0), (1.0, 0.0)],
        **base,
    ))
    configs.append(SweepConfig(
        name="c2f_4to1_100s",
        num_steps=100,
        heatmap_blur_schedule=[(0.3, 4.0), (0.7, 2.0), (1.0, 1.0)],
        **base,
    ))
    configs.append(SweepConfig(
        name="c2f_8to2_100s",
        num_steps=100,
        heatmap_blur_schedule=[(0.3, 8.0), (0.7, 4.0), (1.0, 2.0)],
        **base,
    ))

    # --- Coarse-to-fine blur at 50 steps ---
    configs.append(SweepConfig(
        name="c2f_4to0_50s",
        num_steps=50,
        heatmap_blur_schedule=[(0.4, 4.0), (0.8, 2.0), (1.0, 0.0)],
        **base,
    ))
    configs.append(SweepConfig(
        name="c2f_8to0_50s",
        num_steps=50,
        heatmap_blur_schedule=[(0.4, 8.0), (0.8, 4.0), (1.0, 0.0)],
        **base,
    ))

    return configs
```

Total: 3 baselines + 4 fixed-blur-50s + 4 fixed-blur-100s + 4 c2f-100s + 2 c2f-50s = **17 configs**.

### Step 6: Add CLI selection of config set

Modify `main()` in `sweep.py` to accept a `--phase` argument:

```python
def main() -> None:
    """Run the parameter sweep."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("example_idx", type=int, nargs="?", default=0)
    parser.add_argument("--phase", choices=["1.1", "1.2"], default="1.2",
                        help="Which config set to run")
    args = parser.parse_args()

    example_idx = args.example_idx

    print("Loading example data (detection + GT)...")
    data = load_example(example_idx)
    print(f"Example: {data['name']}, {len(data['det_cam_positions'])} frames\n")

    if args.phase == "1.1":
        configs = get_phase1_1_configs()
    else:
        configs = get_phase1_2_configs()

    # ... rest of sweep loop unchanged
```

### Step 7: Print blur info in sweep header

In the sweep loop, add blur sigma to the per-config printout:

```python
print(
    f"  pos_w={config.position_penalty_weight}, rot_s={config.rotation_penalty_scalar}, "
    f"anchor={config.init_anchor_weight}, smooth={config.all_joints_smooth_weight}, "
    f"blur={config.heatmap_blur_sigma}, blur_sched={config.heatmap_blur_schedule}"
)
```

### Step 8: Save blur config info to JSON

In the JSON save block, add blur parameters:

```python
save_data.append({
    "config": name,
    "det_mpjpe_cm": ...,
    "opt_mpjpe_cm": ...,
    "improvement_cm": ...,
    "opt_p_mpjpe_cm": ...,
    "det_2d_det_mpjpe_px": ...,
    "opt_2d_det_mpjpe_px": ...,
    "opt_mpjve_cm": ...,
    "num_steps": config.num_steps,  # ADD
    "heatmap_blur_sigma": config.heatmap_blur_sigma,  # ADD
    "heatmap_blur_schedule": config.heatmap_blur_schedule,  # ADD
})
```

This requires the sweep loop to store configs alongside results. Change the results list to store the config too:

```python
results: list[tuple[str, dict[str, Any], SweepConfig]] = []
# ...
results.append((config.name, metrics, config))
```

And update the JSON save section to access `config` from each tuple.

### Step 9: Run the sweep

```bash
cd motionbert-pose && uv run python sweep.py 0 --phase 1.2
```

This runs all 17 configs on example 0. Expected runtime: ~5-10 minutes (50-step configs take ~2.5x the 20-step ones, 100-step configs ~5x).

## Integration Points

- `optimize.py` gains a new optional parameter `heatmap_blur_schedule` (default None = no dynamic blur). All existing callers (`main.py`, `test_single.py`) are unaffected since they don't pass this parameter.
- `config.py` gains `HEATMAP_BLUR_SIGMA` (default 0.0). Not consumed by `optimize.py` directly -- only used by `sweep.py` or `main.py` for pre-optimization blur.
- `sweep.py` gains `get_phase1_2_configs()` and argparse for phase selection. The existing `get_phase1_1_configs()` remains untouched.
- The coarse-to-fine blur schedule re-uses the same `(fraction, value)` format as `SIGMA_SCHEDULE` for consistency.

## Risks and Edge Cases

1. **Re-blurring cost**: `scipy.ndimage.gaussian_filter` on 16 channels of 64x64 is cheap (~0.1ms per frame). With 150 frames, a blur phase transition costs ~15ms. There are at most 2-3 transitions per 100-step run. Negligible overhead.

2. **Blur + real heatmaps interaction**: Blur is applied to the raw SH heatmaps before `real_heatmap_score` samples them. This is correct -- it widens the heatmap peaks. However, very high blur (sigma=8 at 64x64 is huge -- nearly 1/8 of the image) will spread the signal too thin, potentially washing out heatmap peaks entirely. The log-likelihood will become log(~uniform) which is a nearly flat gradient. If sigma=8 performs poorly, this is why.

3. **Blur + analytical fallback**: Hip (0) and Spine (7) use analytical Gaussian fallback in `real_heatmap_score`. These are NOT affected by heatmap blur -- they use the `sigma` parameter from `SIGMA_SCHEDULE`. This is intentional and correct. These joints have no SH heatmap to blur.

4. **Dynamic blur and torch tensors**: The `_apply_blur_torch` helper converts torch -> numpy -> scipy blur -> torch. This breaks the computation graph, but that's fine -- heatmaps are data (not learnable), so no gradients flow through them. The `grid_sample` in `real_heatmap_score` provides the differentiable path.

5. **Fixed blur vs schedule conflict**: If a config has both `heatmap_blur_sigma > 0` and `heatmap_blur_schedule is not None`, the fixed pre-blur in `run_sweep_config` would double-blur with the dynamic schedule in `run_optimization`. Step 4 adds a guard to prevent this.
