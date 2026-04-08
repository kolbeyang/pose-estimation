"""Parse results.json and print a summary table."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/kky2806/pose-estimation-output/results.json"

with open(path) as f:
    data = json.load(f)

print(f"{'Example':<30} {'Pipeline':<12} {'Det VW-SI':>10} {'Opt VW-SI':>10} {'Δ':>8} {'Δ%':>8}")
print("-" * 80)

mb_det, mb_opt, mp_det, mp_opt = [], [], [], []
mb_improved, mp_improved = 0, 0

for ex in data["examples"]:
    name = ex["name"]
    for pipe in ["motionbert", "mediapipe"]:
        if pipe not in ex:
            continue
        p = ex[pipe]
        det = p.get("det_vw_si_mpjpe", p.get("det_si_mpjpe", 0))
        opt = p.get("opt_vw_si_mpjpe", p.get("opt_si_mpjpe", 0))
        delta = opt - det
        pct = (delta / det * 100) if det > 0 else 0
        label = "MB" if pipe == "motionbert" else "MP"
        marker = "✓" if delta < 0 else "✗"
        print(f"{name:<30} {label:<12} {det:>9.2f}cm {opt:>9.2f}cm {delta:>+7.2f} {pct:>+7.1f}% {marker}")
        if pipe == "motionbert":
            mb_det.append(det)
            mb_opt.append(opt)
            if delta < 0:
                mb_improved += 1
        else:
            mp_det.append(det)
            mp_opt.append(opt)
            if delta < 0:
                mp_improved += 1

print("\n" + "=" * 80)
print("AGGREGATE")
print("=" * 80)

if mb_det:
    avg_det = sum(mb_det) / len(mb_det)
    avg_opt = sum(mb_opt) / len(mb_opt)
    pct = (avg_opt - avg_det) / avg_det * 100
    print(f"MotionBERT:  Det avg={avg_det:.2f}cm  Opt avg={avg_opt:.2f}cm  Δ={pct:+.1f}%  Improved: {mb_improved}/{len(mb_det)}")

if mp_det:
    avg_det = sum(mp_det) / len(mp_det)
    avg_opt = sum(mp_opt) / len(mp_opt)
    pct = (avg_opt - avg_det) / avg_det * 100
    print(f"MediaPipe:   Det avg={avg_det:.2f}cm  Opt avg={avg_opt:.2f}cm  Δ={pct:+.1f}%  Improved: {mp_improved}/{len(mp_det)}")
