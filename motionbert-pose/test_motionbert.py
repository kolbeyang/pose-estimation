"""Quick test: run MotionBERT on the official alphapose.json sample
and compare to the official X3D.npy output to verify our pipeline."""

import json
import numpy as np
import torch
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "external", "MotionBERT", "lib"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "external", "MotionBERT"))

from lib.model.DSTformer import DSTformer


def halpe2h36m(x):
    """Official Halpe 26 -> H36M 17 conversion from MotionBERT."""
    T, V, C = x.shape
    y = np.zeros([T, 17, C])
    y[:, 0, :] = x[:, 19, :]   # hip
    y[:, 1, :] = x[:, 12, :]   # rhip
    y[:, 2, :] = x[:, 14, :]   # rkne
    y[:, 3, :] = x[:, 16, :]   # rank
    y[:, 4, :] = x[:, 11, :]   # lhip
    y[:, 5, :] = x[:, 13, :]   # lkne
    y[:, 6, :] = x[:, 15, :]   # lank
    y[:, 7, :] = (x[:, 18, :] + x[:, 19, :]) * 0.5  # spine
    y[:, 8, :] = x[:, 18, :]   # neck
    y[:, 9, :] = x[:, 0, :]    # nose/head
    y[:, 10, :] = x[:, 17, :]  # headtop
    y[:, 11, :] = x[:, 5, :]   # lsho
    y[:, 12, :] = x[:, 7, :]   # lelb
    y[:, 13, :] = x[:, 9, :]   # lwri
    y[:, 14, :] = x[:, 6, :]   # rsho
    y[:, 15, :] = x[:, 8, :]   # relb
    y[:, 16, :] = x[:, 10, :]  # rwri
    return y


def crop_scale(motion):
    """Official crop_scale from MotionBERT (scale_range=[1,1])."""
    import copy
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = valid_coords[:, 0].min()
    xmax = valid_coords[:, 0].max()
    ymin = valid_coords[:, 1].min()
    ymax = valid_coords[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


def flip_data(data):
    """Official flip_data from MotionBERT."""
    import copy
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


def main():
    # 1. Load official AlphaPose input
    with open("motionbert-raw/alphapose.json") as f:
        results = json.load(f)

    kpts_all = []
    for item in results:
        kpts = np.array(item['keypoints']).reshape(-1, 3)
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)  # (87, 26, 3)
    print(f"AlphaPose input: {kpts_all.shape}")

    # 2. Convert Halpe -> H36M
    kpts_h36m = halpe2h36m(kpts_all)  # (87, 17, 3)
    print(f"H36M keypoints: {kpts_h36m.shape}")

    # 3. Normalize with crop_scale (official method, scale_range=[1,1])
    kpts_norm = crop_scale(kpts_h36m)
    print(f"Normalized range: x=[{kpts_norm[:,:,0].min():.3f}, {kpts_norm[:,:,0].max():.3f}] "
          f"y=[{kpts_norm[:,:,1].min():.3f}, {kpts_norm[:,:,1].max():.3f}]")

    # 4. Load MotionBERT-Lite (same model as main pipeline)
    model = DSTformer(
        dim_in=3, dim_out=3,
        dim_feat=256, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=4,
        num_joints=17, maxlen=243,
    )
    ckpt = torch.load("checkpoints/motionbert_lite_h36m.bin", map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_pos"].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Loaded MotionBERT-Lite")

    # 5. Run inference with flip augmentation (matching official)
    input_tensor = torch.from_numpy(kpts_norm.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        pred1 = model(input_tensor)
        input_flip = flip_data(input_tensor)
        pred_flip = model(input_flip)
        pred2 = flip_data(pred_flip)
        output = (pred1 + pred2) / 2.0

    our_output = output.cpu().numpy()[0]  # (87, 17, 3)

    # Zero first frame root Z (matching official rootrel=False)
    our_output[0, 0, 2] = 0

    # 6. Load official output
    official = np.load("motionbert-raw/output_test/X3D.npy")
    print(f"\nOur output shape: {our_output.shape}")
    print(f"Official output shape: {official.shape}")

    # 7. Compare
    diff = our_output - official
    print(f"\n=== Comparison (our - official) ===")
    print(f"Max absolute diff: {np.abs(diff).max():.6f}")
    print(f"Mean absolute diff: {np.abs(diff).mean():.6f}")
    print(f"RMS diff: {np.sqrt((diff**2).mean()):.6f}")

    # Per-axis comparison
    for axis, name in enumerate(["X", "Y", "Z"]):
        axis_diff = diff[:, :, axis]
        print(f"  {name}: max={np.abs(axis_diff).max():.6f} mean={np.abs(axis_diff).mean():.6f}")

    # Z specifically for left arm
    print(f"\n=== Left Arm Z Comparison ===")
    for jname, jidx in [("Shoulder", 11), ("Elbow", 12), ("Wrist", 13)]:
        ours_z = our_output[:, jidx, 2]
        off_z = official[:, jidx, 2]
        print(f"  {jname} Z — Ours: [{ours_z.min():.4f}, {ours_z.max():.4f}] span={ours_z.max()-ours_z.min():.4f}")
        print(f"  {jname} Z — Official: [{off_z.min():.4f}, {off_z.max():.4f}] span={off_z.max()-off_z.min():.4f}")
        print(f"  {jname} Z — Diff: max={np.abs(ours_z-off_z).max():.6f}")


if __name__ == "__main__":
    main()
