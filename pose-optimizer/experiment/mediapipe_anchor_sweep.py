"""Sweep num_steps x anchor_weight for MediaPipe optimization.

Grid: num_steps=[25, 50, 100] x anchor_weight=[0, 50, 200] = 9 runs
Uses the same 2 examples as local-debug-baseline-mediapipe.json.

Usage:
    cd pose-optimizer
    uv run python experiment/mediapipe_anchor_sweep.py
"""

import json
import os
import sys
import copy
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config

BASE_CONFIG = "configs/local-debug-baseline-mediapipe.json"
STEPS_VALUES = [25, 50, 100]
ANCHOR_VALUES = [0, 50, 200]


def main():
    base = load_config(BASE_CONFIG)

    results = []

    for steps in STEPS_VALUES:
        for anchor in ANCHOR_VALUES:
            label = f"steps={steps}_anchor={anchor}"
            print(f"\n{'='*60}")
            print(f"Running: {label}")
            print(f"{'='*60}")

            # Create temp config
            cfg = copy.deepcopy(base)
            cfg.mediapipe.optimization = {
                "num_steps": steps,
                "heatmap_blur_sigma": 4.0,
                "learning_rate": 0.0005,
                "anchor_weight": anchor,
            }
            cfg.generate_video = False  # skip video gen for speed
            cfg.generate_graphs = False
            if cfg.mediapipe:
                cfg.mediapipe.is_generate_heatmap_videos = False

            out_dir = f"output/sweep_mp_anchor/{label}"
            cfg.output_dir = out_dir

            # Save temp config
            tmp_cfg_path = f"/tmp/sweep_{label}.json"
            with open(tmp_cfg_path, "w") as f:
                json.dump(cfg.model_dump(), f, indent=2)

            t0 = time.time()
            os.system(f"uv run python main.py {tmp_cfg_path} 2>&1 | grep -E '(MPJPE|MPJVE|VW-SI|Processing|Done)'")
            elapsed = time.time() - t0

            # Read results
            results_path = os.path.join(out_dir, "results.json")
            if os.path.exists(results_path):
                with open(results_path) as f:
                    run_results = json.load(f)

                for ex in run_results.get("mediapipe", []):
                    results.append({
                        "steps": steps,
                        "anchor": anchor,
                        "example": ex["name"],
                        "det_vw_si_mpjpe": ex["det_vw_si_mpjpe"],
                        "opt_vw_si_mpjpe": ex["opt_vw_si_mpjpe"],
                        "det_mpjve": ex["det_mpjve"],
                        "opt_mpjve": ex["opt_mpjve"],
                        "elapsed": elapsed,
                    })

    # Print summary table
    print(f"\n\n{'='*120}")
    print("SWEEP RESULTS: MediaPipe anchor_weight x num_steps")
    print(f"{'='*120}")

    # Group by example
    examples = sorted(set(r["example"] for r in results))

    for ex_name in examples:
        print(f"\n--- {ex_name} ---")
        print(f"{'steps':>5} {'anchor':>6} | {'Det VW-SI':>10} {'Opt VW-SI':>10} {'Δ cm':>8} {'Δ %':>8} | {'Det MPJVE':>10} {'Opt MPJVE':>10} {'Δ cm':>8} {'Δ %':>8} | {'time':>5}")
        print("-" * 110)

        ex_results = [r for r in results if r["example"] == ex_name]
        for r in sorted(ex_results, key=lambda x: (x["steps"], x["anchor"])):
            det_vw = r["det_vw_si_mpjpe"] * 100
            opt_vw = r["opt_vw_si_mpjpe"] * 100
            d_vw = opt_vw - det_vw
            pct_vw = (d_vw / det_vw * 100) if det_vw > 0 else 0

            det_ve = r["det_mpjve"] * 100
            opt_ve = r["opt_mpjve"] * 100
            d_ve = opt_ve - det_ve
            pct_ve = (d_ve / det_ve * 100) if det_ve > 0 else 0

            print(f"{r['steps']:>5} {r['anchor']:>6} | {det_vw:>9.2f}cm {opt_vw:>9.2f}cm {d_vw:>+7.2f} {pct_vw:>+7.1f}% | {det_ve:>9.2f}cm {opt_ve:>9.2f}cm {d_ve:>+7.2f} {pct_ve:>+7.1f}% | {r['elapsed']:>4.0f}s")

    # Average across examples
    print(f"\n--- AVERAGE ---")
    print(f"{'steps':>5} {'anchor':>6} | {'Avg Opt VW-SI':>13} {'Avg Δ %':>8} | {'Avg Opt MPJVE':>13} {'Avg Δ %':>8}")
    print("-" * 80)

    for steps in STEPS_VALUES:
        for anchor in ANCHOR_VALUES:
            subset = [r for r in results if r["steps"] == steps and r["anchor"] == anchor]
            if not subset:
                continue
            avg_opt_vw = sum(r["opt_vw_si_mpjpe"] for r in subset) / len(subset) * 100
            avg_det_vw = sum(r["det_vw_si_mpjpe"] for r in subset) / len(subset) * 100
            avg_pct_vw = ((avg_opt_vw - avg_det_vw) / avg_det_vw * 100) if avg_det_vw > 0 else 0

            avg_opt_ve = sum(r["opt_mpjve"] for r in subset) / len(subset) * 100
            avg_det_ve = sum(r["det_mpjve"] for r in subset) / len(subset) * 100
            avg_pct_ve = ((avg_opt_ve - avg_det_ve) / avg_det_ve * 100) if avg_det_ve > 0 else 0

            print(f"{steps:>5} {anchor:>6} | {avg_opt_vw:>12.2f}cm {avg_pct_vw:>+7.1f}% | {avg_opt_ve:>12.2f}cm {avg_pct_ve:>+7.1f}%")


if __name__ == "__main__":
    main()
