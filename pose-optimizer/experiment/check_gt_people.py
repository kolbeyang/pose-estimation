"""Check max number of people in GT across all sequences with GT data."""
import json
import glob
import os
import sys

data_root = sys.argv[1] if len(sys.argv) > 1 else "/data/datasets/cmu-panoptic"

results = []
for seq in sorted(os.listdir(data_root)):
    gt_dir = os.path.join(data_root, seq, "hdPose3d_stage1_coco19")
    files = sorted(glob.glob(os.path.join(gt_dir, "*.json")))
    if not files:
        continue
    # Sample every 5% of frames to keep it fast
    step = max(1, len(files) // 20)
    max_people = 0
    for f in files[::step]:
        with open(f) as fh:
            d = json.load(fh)
        max_people = max(max_people, len(d.get("bodies", [])))
    results.append((seq, max_people, len(files)))

print(f"\n{'Sequence':<30} {'Max People':>10} {'GT Frames':>10}")
print("-" * 52)
single = []
for seq, mp, nf in results:
    marker = " <--" if mp == 1 else ""
    print(f"{seq:<30} {mp:>10} {nf:>10}{marker}")
    if mp == 1:
        single.append((seq, nf))

print(f"\n--- Single-person sequences with GT: {len(single)} ---")
for seq, nf in single:
    print(f"  {seq}: {nf} GT frames")
