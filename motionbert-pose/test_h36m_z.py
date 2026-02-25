"""Test MotionBERT Z quality on H36M data.

Uses 4 camera views: what appears as X motion in a side camera
should appear as Z motion from MotionBERT in a front camera.
This lets us cross-validate Z without needing 3D ground truth.
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


def crop_scale(motion):
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin, xmax = valid_coords[:, 0].min(), valid_coords[:, 0].max()
    ymin, ymax = valid_coords[:, 1].min(), valid_coords[:, 1].max()
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
    left_joints = [4, 5, 6, 11, 12, 13]
    right_joints = [1, 2, 3, 14, 15, 16]
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


def run_motionbert(model, kpts_2d, clip_len=243):
    """Run MotionBERT on 2D keypoints. kpts_2d: (T, 17, 2)"""
    # Add confidence channel
    T = kpts_2d.shape[0]
    kpts_3ch = np.concatenate([kpts_2d, np.ones((T, 17, 1))], axis=2).astype(np.float32)

    # Normalize
    kpts_norm = crop_scale(kpts_3ch)

    # Process in clips
    all_outputs = []
    for start in range(0, T, clip_len):
        end = min(start + clip_len, T)
        clip = kpts_norm[start:end]
        inp = torch.from_numpy(clip).unsqueeze(0)

        with torch.no_grad():
            pred1 = model(inp)
            pred2 = flip_data(model(flip_data(inp)))
            output = (pred1 + pred2) / 2.0

        out = output.cpu().numpy()[0]
        # rootrel=False: zero first frame root Z
        if start == 0:
            out[0, 0, 2] = 0
        all_outputs.append(out)

    return np.concatenate(all_outputs, axis=0)


def main():
    # Load 2D CPN detections
    d2 = np.load("/tmp/h36m_cpn_2d.npz", allow_pickle=True)
    pos2d = d2["positions_2d"].item()

    # Load model
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
    print("Loaded MotionBERT-Lite\n")

    # Pick action with obvious depth motion
    for action in ["Walking 1", "Directions 1", "Greeting 1"]:
        views = pos2d["S9"][action]  # 4 camera views
        print(f"=== S9 / {action} ===")
        print(f"  4 views: {[v.shape for v in views]}")

        # Use first 243 frames for speed
        n = min(243, min(v.shape[0] for v in views))

        # Run MotionBERT on each view
        results = []
        for cam_idx in range(4):
            kpts = views[cam_idx][:n]  # (n, 17, 2)
            output = run_motionbert(model, kpts)
            results.append(output[:n])

        # Compare: for walking, the person walks in a line.
        # Root position tells us the motion direction.
        print(f"\n  Root trajectory per camera view (first {n} frames):")
        for cam_idx in range(4):
            root = results[cam_idx][:, 0, :]  # (n, 3) - root joint
            x_span = root[:, 0].max() - root[:, 0].min()
            y_span = root[:, 1].max() - root[:, 1].min()
            z_span = root[:, 2].max() - root[:, 2].min()
            print(f"    Cam {cam_idx}: X span={x_span:.4f}  Y span={y_span:.4f}  Z span={z_span:.4f}")

        # Left arm analysis
        print(f"\n  Left wrist trajectory per camera view:")
        for cam_idx in range(4):
            wrist = results[cam_idx][:, 13, :]
            x_span = wrist[:, 0].max() - wrist[:, 0].min()
            y_span = wrist[:, 1].max() - wrist[:, 1].min()
            z_span = wrist[:, 2].max() - wrist[:, 2].min()
            print(f"    Cam {cam_idx}: X span={x_span:.4f}  Y span={y_span:.4f}  Z span={z_span:.4f}")

        # Cross-view consistency check:
        # If cam 0 is frontal and cam 1 is side, then:
        #   cam0 Z variation should correlate with cam1 X variation
        print(f"\n  Cross-view Z consistency (root):")
        for i in range(4):
            for j in range(i + 1, 4):
                z_i = results[i][:, 0, 2]
                x_j = results[j][:, 0, 0]
                z_j = results[j][:, 0, 2]
                x_i = results[i][:, 0, 0]
                # Correlation between cam_i Z and cam_j X
                corr_zi_xj = np.corrcoef(z_i, x_j)[0, 1]
                corr_xi_zj = np.corrcoef(x_i, z_j)[0, 1]
                print(f"    Cam{i}Z vs Cam{j}X: r={corr_zi_xj:.3f}  |  Cam{i}X vs Cam{j}Z: r={corr_xi_zj:.3f}")
        print()


if __name__ == "__main__":
    main()
