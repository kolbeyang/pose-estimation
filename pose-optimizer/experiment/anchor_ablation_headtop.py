"""Anchor penalty ablation after HeadTop fix.

Compares anchor_weight=50 vs anchor_weight=0 on MediaPipe pipeline
across increasing subsets of the 25 benchmark examples.
"""

import os, sys, json, logging, glob

logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RunConfig, PipelineConfig, ExampleConfig, save_config
from cmu_data import EXAMPLES
from main import main

SUBSETS = {
    1: [0],
    3: [0, 1, 6],
    5: [0, 1, 6, 9, 20],
    10: [0, 1, 2, 6, 7, 9, 10, 16, 20, 21],
    25: list(range(25)),
}


def make_examples(n):
    return [
        ExampleConfig(
            sequence=EXAMPLES[i][0],
            camera=EXAMPLES[i][1],
            start_frame=EXAMPLES[i][2],
            num_frames=EXAMPLES[i][3],
            person_idx=EXAMPLES[i][4],
        )
        for i in SUBSETS[n]
    ]


results = {}

for n in [1, 3, 5, 10, 25]:
    for aw in [50.0, 0.0]:
        out_dir = f"output/headtop_anchor_ablation/n{n}_aw{int(aw)}"
        cfg = RunConfig(
            mediapipe=PipelineConfig(
                is_generate_heatmap_videos=False,
                optimization={
                    "num_steps": 100,
                    "heatmap_blur_sigma": 4.0,
                    "learning_rate": 0.0005,
                    "anchor_weight": aw,
                },
            ),
            examples=make_examples(n),
            generate_graphs=False,
            generate_video=False,
            output_dir=out_dir,
        )
        os.makedirs(out_dir, exist_ok=True)
        config_path = os.path.join(out_dir, "config.json")
        save_config(cfg, config_path)
        main(config_path)

        # Load results
        run_dirs = sorted(glob.glob(os.path.join(out_dir, "run_*")))
        if run_dirs:
            rpath = os.path.join(run_dirs[-1], "results.json")
            with open(rpath) as f:
                data = json.load(f)
            mp_results = data.get("mediapipe", [])
            vw_vals = [
                r.get("opt_vw_si_mpjpe")
                for r in mp_results
                if r.get("opt_vw_si_mpjpe") is not None
            ]
            mean_vw = sum(vw_vals) / len(vw_vals) * 100 if vw_vals else None
            results[(n, aw)] = {
                "mean_vw_si_mpjpe": mean_vw,
                "per_example": mp_results,
                "count": len(vw_vals),
            }

    # Print intermediate
    b = results.get((n, 50.0), {}).get("mean_vw_si_mpjpe")
    t = results.get((n, 0.0), {}).get("mean_vw_si_mpjpe")
    if b and t:
        diff = t - b
        pct = (diff / b) * 100
        print(
            f"n={n:>2}: anchor=50 -> {b:.3f} cm | anchor=0 -> {t:.3f} cm | diff={diff:+.3f} cm ({pct:+.1f}%)"
        )

# Final summary
print("\n" + "=" * 70)
print("FINAL SUMMARY: Anchor Penalty Ablation (after HeadTop fix)")
print("=" * 70)
print(
    f"{'N':>3} | {'anchor=50 (cm)':>14} | {'anchor=0 (cm)':>13} | {'diff (cm)':>9} | {'% change':>8}"
)
print("-" * 60)
for n in [1, 3, 5, 10, 25]:
    b = results.get((n, 50.0), {}).get("mean_vw_si_mpjpe")
    t = results.get((n, 0.0), {}).get("mean_vw_si_mpjpe")
    if b and t:
        diff = t - b
        pct = (diff / b) * 100
        print(
            f"{n:>3} | {b:>14.3f} | {t:>13.3f} | {diff:>+9.3f} | {pct:>+7.1f}%"
        )

# Per-example breakdown at n=25
print("\nPer-example at n=25:")
print(f"{'Example':<35} {'anchor=50':>9} {'anchor=0':>9} {'diff':>8}")
print("-" * 65)
b_data = results.get((25, 50.0), {}).get("per_example", [])
t_data = results.get((25, 0.0), {}).get("per_example", [])
for be, te in zip(b_data, t_data):
    bv = be.get("opt_vw_si_mpjpe")
    tv = te.get("opt_vw_si_mpjpe")
    name = be.get("name", "?")
    if bv is not None and tv is not None:
        print(f"{name:<35} {bv*100:>8.3f}  {tv*100:>8.3f}  {(tv-bv)*100:>+7.3f}")
