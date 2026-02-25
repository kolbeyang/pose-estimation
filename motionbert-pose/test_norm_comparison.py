"""Compare MotionBERT output under three different 2D input normalization methods.

Uses H36M CPN 2D detections for S9 / Walking 1 / camera 0 / first 243 frames.

Method 1: crop_scale - bounding-box normalization (wild/in-the-wild mode)
Method 2: vid_size pixel normalization (--pixel mode: center at (w/2,h/2), scale by min(w,h)/2)
Method 3: H36M training normalization (from datareader_h36m.py: kpts / res_w * 2 - [1, res_h/res_w])
"""

import numpy as np
import torch
import sys
import os
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "external", "MotionBERT", "lib"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "external", "MotionBERT"))

from lib.model.DSTformer import DSTformer


# ---------------------------------------------------------------------------
# Normalization methods
# ---------------------------------------------------------------------------

def normalize_crop_scale(kpts_2d):
    """Method 1: crop_scale bounding-box normalization (wild mode).

    kpts_2d: (T, 17, 2) raw pixel coords
    Returns: (T, 17, 3) with confidence=1 as channel 2, normalized to ~[-1, 1]
    """
    T = kpts_2d.shape[0]
    motion = np.concatenate([kpts_2d, np.ones((T, 17, 1))], axis=2).astype(np.float32)

    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    xmin, xmax = valid_coords[:, 0].min(), valid_coords[:, 0].max()
    ymin, ymax = valid_coords[:, 1].min(), valid_coords[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape, dtype=np.float32)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result


def normalize_vid_size(kpts_2d, res_w=1000, res_h=1000):
    """Method 2: vid_size pixel normalization (official --pixel mode).

    Centers at (w/2, h/2), scales by min(w,h)/2.
    kpts_2d: (T, 17, 2) raw pixel coords
    Returns: (T, 17, 3) with confidence=1 as channel 2
    """
    T = kpts_2d.shape[0]
    kpts = np.concatenate([kpts_2d.copy(), np.ones((T, 17, 1))], axis=2).astype(np.float32)

    w, h = res_w, res_h
    scale = min(w, h) / 2.0
    kpts[:, :, :2] = (kpts[:, :, :2] - np.array([w, h]) / 2.0) / scale
    return kpts


def normalize_h36m_training(kpts_2d, res_w=1000, res_h=1000):
    """Method 3: H36M training normalization (from datareader_h36m.py).

    kpts[:,:,:2] = kpts[:,:,:2] / res_w * 2 - [1, res_h / res_w]
    kpts_2d: (T, 17, 2) raw pixel coords
    Returns: (T, 17, 3) with confidence=1 as channel 2
    """
    T = kpts_2d.shape[0]
    kpts = np.concatenate([kpts_2d.copy(), np.ones((T, 17, 1))], axis=2).astype(np.float32)

    kpts[:, :, :2] = kpts[:, :, :2] / res_w * 2 - [1, res_h / res_w]
    return kpts


# ---------------------------------------------------------------------------
# Flip augmentation
# ---------------------------------------------------------------------------

def flip_data(data):
    """Official flip_data from MotionBERT."""
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[
        ..., right_joints + left_joints, :
    ]
    return flipped_data


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, kpts_norm):
    """Run MotionBERT with flip augmentation.

    kpts_norm: (T, 17, 3) normalized keypoints with confidence channel
    Returns: (T, 17, 3) predicted 3D pose
    """
    input_tensor = torch.from_numpy(kpts_norm.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        pred1 = model(input_tensor)
        pred2 = flip_data(model(flip_data(input_tensor)))
        output = (pred1 + pred2) / 2.0

    result = output.cpu().numpy()[0]  # (T, 17, 3)

    # Zero first frame root Z (rootrel=False convention)
    result[0, 0, 2] = 0.0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load H36M CPN 2D detections
    print("Loading H36M CPN 2D detections...")
    d = np.load("/tmp/h36m_cpn_2d.npz", allow_pickle=True)
    pos2d = d["positions_2d"].item()

    # S9, Walking 1, camera 0, first 243 frames
    kpts_2d_raw = pos2d["S9"]["Walking 1"][0][:243].copy()  # (243, 17, 2)
    print(f"Input 2D keypoints: {kpts_2d_raw.shape}")
    print(f"  Pixel range X: [{kpts_2d_raw[:,:,0].min():.1f}, {kpts_2d_raw[:,:,0].max():.1f}]")
    print(f"  Pixel range Y: [{kpts_2d_raw[:,:,1].min():.1f}, {kpts_2d_raw[:,:,1].max():.1f}]")

    # Load MotionBERT-Lite
    print("\nLoading MotionBERT-Lite...")
    model = DSTformer(
        dim_in=3, dim_out=3,
        dim_feat=256, dim_rep=512,
        depth=5, num_heads=8, mlp_ratio=4,
        num_joints=17, maxlen=243,
    )
    ckpt_path = os.path.join(SCRIPT_DIR, "checkpoints", "motionbert_lite_h36m.bin")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt["model_pos"].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully.\n")

    # Run all three normalization methods
    methods = {
        "Method 1: crop_scale (wild mode)": normalize_crop_scale(kpts_2d_raw),
        "Method 2: vid_size pixel norm (--pixel mode)": normalize_vid_size(kpts_2d_raw, 1000, 1000),
        "Method 3: H36M training norm (datareader)": normalize_h36m_training(kpts_2d_raw, 1000, 1000),
    }

    results = {}

    print("=" * 80)
    print("INPUT NORMALIZATION COMPARISON")
    print("=" * 80)

    for name, kpts_norm in methods.items():
        print(f"\n--- {name} ---")
        print(f"  Normalized X range: [{kpts_norm[:,:,0].min():.4f}, {kpts_norm[:,:,0].max():.4f}]")
        print(f"  Normalized Y range: [{kpts_norm[:,:,1].min():.4f}, {kpts_norm[:,:,1].max():.4f}]")
        print(f"  Confidence channel: [{kpts_norm[:,:,2].min():.4f}, {kpts_norm[:,:,2].max():.4f}]")

        output_3d = run_inference(model, kpts_norm)
        results[name] = output_3d
        print(f"  Output 3D shape: {output_3d.shape}")

    # Compare results
    print("\n" + "=" * 80)
    print("3D OUTPUT COMPARISON")
    print("=" * 80)

    # Root (joint 0) trajectory analysis
    print("\n--- Root (joint 0) trajectory spans ---")
    print(f"{'Method':<50s} {'X span':>10s} {'Y span':>10s} {'Z span':>10s}")
    print("-" * 80)
    for name, output in results.items():
        root = output[:, 0, :]  # (243, 3)
        x_span = root[:, 0].max() - root[:, 0].min()
        y_span = root[:, 1].max() - root[:, 1].min()
        z_span = root[:, 2].max() - root[:, 2].min()
        print(f"{name:<50s} {x_span:10.4f} {y_span:10.4f} {z_span:10.4f}")

    # Left wrist (joint 13) trajectory analysis
    print("\n--- Left wrist (joint 13) trajectory spans ---")
    print(f"{'Method':<50s} {'X span':>10s} {'Y span':>10s} {'Z span':>10s}")
    print("-" * 80)
    for name, output in results.items():
        wrist = output[:, 13, :]  # (243, 3)
        x_span = wrist[:, 0].max() - wrist[:, 0].min()
        y_span = wrist[:, 1].max() - wrist[:, 1].min()
        z_span = wrist[:, 2].max() - wrist[:, 2].min()
        print(f"{name:<50s} {x_span:10.4f} {y_span:10.4f} {z_span:10.4f}")

    # Root Z range details
    print("\n--- Root Z detailed range ---")
    for name, output in results.items():
        root_z = output[:, 0, 2]
        print(f"  {name}")
        print(f"    Z min={root_z.min():.4f}  Z max={root_z.max():.4f}  "
              f"Z mean={root_z.mean():.4f}  Z std={root_z.std():.4f}")

    # Left wrist Z range details
    print("\n--- Left wrist (joint 13) Z detailed range ---")
    for name, output in results.items():
        wrist_z = output[:, 13, 2]
        print(f"  {name}")
        print(f"    Z min={wrist_z.min():.4f}  Z max={wrist_z.max():.4f}  "
              f"Z mean={wrist_z.mean():.4f}  Z std={wrist_z.std():.4f}")

    # Pairwise differences
    print("\n--- Pairwise output differences (mean absolute error across all joints/frames) ---")
    method_names = list(results.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            diff = np.abs(results[method_names[i]] - results[method_names[j]])
            print(f"  {method_names[i]}")
            print(f"    vs {method_names[j]}")
            print(f"    MAE: {diff.mean():.4f}  Max: {diff.max():.4f}")
            print(f"    Per-axis MAE: X={diff[:,:,0].mean():.4f}  Y={diff[:,:,1].mean():.4f}  Z={diff[:,:,2].mean():.4f}")

    # Overall scale comparison
    print("\n--- Overall output scale (RMS of all coordinates) ---")
    for name, output in results.items():
        rms = np.sqrt((output ** 2).mean())
        print(f"  {name}: RMS = {rms:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
