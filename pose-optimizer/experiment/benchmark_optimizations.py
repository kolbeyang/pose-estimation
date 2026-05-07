"""Benchmark 5 proposed optimizations on real CMU Panoptic data.

Usage (run on server):
    CUDA_VISIBLE_DEVICES=1 uv run python experiment/benchmark_optimizations.py /data/datasets/cmu-panoptic

Uses 171204_pose3 (single person, ~8920 GT frames).
Tests each optimization in isolation and reports wall-clock time.
"""

import sys
import os
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmu_data import get_video_path, extract_video_frames
from run_motionbert.detect import (
    load_yolo_sh_models,
    detect_person_bbox,
    crop_and_resize,
    _normalize_for_sh,
    _flip_heatmaps,
    _parse_heatmaps,
)
from scoring import apply_blur


data_root = sys.argv[1] if len(sys.argv) > 1 else "data/panoptic-toolbox"
SEQ = "171204_pose3"
CAMERA = "00_00"
NUM_FRAMES = 3000  # use first 3000 video frames -> ~1000 at 10fps


def timed(label, fn):
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"  {label}: {elapsed:.2f}s")
    return result, elapsed


def main():
    video_path = get_video_path(data_root, SEQ, CAMERA)
    frame_step = 3  # 30fps -> 10fps
    frame_indices = list(range(0, NUM_FRAMES, frame_step))
    n_target = len(frame_indices)
    print(f"Sequence: {SEQ}, {NUM_FRAMES} video frames -> {n_target} at 10fps")
    print(f"Video: {video_path}")
    print()

    # =========================================================================
    # Optimization 1: YOLO sampling
    # =========================================================================
    print("=" * 60)
    print("OPT 1: YOLO detection — every frame vs sampled")
    print("=" * 60)

    # First extract frames (needed for all tests)
    print("  Extracting frames...")
    frames_rgb, t_extract = timed("Frame extraction (sequential read)", lambda: extract_video_frames(video_path, frame_indices))
    n_frames = len(frames_rgb)
    print(f"  Got {n_frames} frames")

    models = load_yolo_sh_models()

    # Baseline: YOLO on every frame
    _, t_yolo_all = timed(f"YOLO on all {n_frames} frames", lambda: detect_person_bbox(frames_rgb, yolo=models.yolo))

    # Sampled: YOLO on every 50th frame
    sample_step = 50
    sampled_frames = frames_rgb[::sample_step]
    bbox_sampled, t_yolo_sampled = timed(
        f"YOLO on {len(sampled_frames)} frames (every {sample_step}th)",
        lambda: detect_person_bbox(sampled_frames, yolo=models.yolo),
    )

    # Also check: YOLO on every 100th frame
    sampled_100 = frames_rgb[::100]
    bbox_100, t_yolo_100 = timed(
        f"YOLO on {len(sampled_100)} frames (every 100th)",
        lambda: detect_person_bbox(sampled_100, yolo=models.yolo),
    )

    print(f"  Speedup (50-sample): {t_yolo_all / t_yolo_sampled:.1f}x")
    print(f"  Speedup (100-sample): {t_yolo_all / t_yolo_100:.1f}x")
    print()

    # =========================================================================
    # Optimization 2: Video seek vs sequential read
    # =========================================================================
    print("=" * 60)
    print("OPT 2: Video frame extraction — sequential vs seek")
    print("=" * 60)

    # Already timed sequential above. Now try seek-based:
    def extract_seek():
        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    frames_seek, t_seek = timed(f"Seek-based extraction ({n_target} frames)", extract_seek)
    print(f"  Sequential: {t_extract:.2f}s, Seek: {t_seek:.2f}s")
    if t_seek < t_extract:
        print(f"  Speedup: {t_extract / t_seek:.1f}x")
    else:
        print(f"  Seek is SLOWER by {t_seek / t_extract:.1f}x (codec may not support fast seek)")

    # Verify frames are the same
    if len(frames_seek) == len(frames_rgb):
        diffs = [np.abs(a.astype(float) - b.astype(float)).mean() for a, b in zip(frames_rgb[:5], frames_seek[:5])]
        print(f"  Mean pixel diff (first 5 frames): {np.mean(diffs):.2f}")
    print()

    # =========================================================================
    # Optimization 3: SH batch size
    # =========================================================================
    print("=" * 60)
    print("OPT 3: Stacked Hourglass batch size")
    print("=" * 60)

    # Preprocess frames for SH
    bbox = detect_person_bbox(frames_rgb[::50], yolo=models.yolo)
    preprocessed = []
    for frame in frames_rgb:
        cropped, affine = crop_and_resize(frame, bbox, target_size=256)
        preprocessed.append(_normalize_for_sh(cropped))

    sh_model = models.hourglass
    device = models.device

    for bs in [32, 64, 128, 256]:
        def run_sh(batch_size=bs):
            results = []
            with torch.no_grad():
                for start in range(0, n_frames, batch_size):
                    end = min(start + batch_size, n_frames)
                    inp = torch.from_numpy(np.stack(preprocessed[start:end])).to(device)
                    out = sh_model(inp)
                    results.append(out[-1].cpu().numpy())
            return results

        _, t_sh = timed(f"SH batch_size={bs} ({n_frames} frames, no flip)", run_sh)
    print()

    # =========================================================================
    # Optimization 4: apply_blur — scipy loop vs torch conv2d
    # =========================================================================
    print("=" * 60)
    print("OPT 4: Heatmap blur — scipy loop vs torch conv2d")
    print("=" * 60)

    # Create dummy heatmaps matching real shape
    n_hm = min(500, n_frames)  # use 500 frames for blur benchmark
    dummy_heatmaps = torch.rand(n_hm, 16, 64, 64, dtype=torch.float32)
    sigma = 4.0

    # Baseline: scipy loop
    _, t_scipy = timed(f"scipy blur ({n_hm} frames x 16 channels, sigma={sigma})", lambda: apply_blur(dummy_heatmaps, sigma))

    # Torch conv2d version
    def apply_blur_torch(heatmaps, sigma):
        ks = int(6 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        pad = ks // 2
        x = torch.arange(ks, dtype=torch.float32) - pad
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        # Separable: blur rows then columns
        F_count, C, H, W = heatmaps.shape
        # Reshape to (F*C, 1, H, W) for depthwise conv
        flat = heatmaps.reshape(-1, 1, H, W)
        # Horizontal
        k_h = kernel_1d.reshape(1, 1, 1, ks)
        flat = TF.pad(flat, (pad, pad, 0, 0), mode='reflect')
        flat = TF.conv2d(flat, k_h)
        # Vertical
        k_v = kernel_1d.reshape(1, 1, ks, 1)
        flat = TF.pad(flat, (0, 0, pad, pad), mode='reflect')
        flat = TF.conv2d(flat, k_v)
        return flat.reshape(F_count, C, H, W)

    _, t_torch_cpu = timed(f"torch conv2d CPU ({n_hm} frames x 16 channels)", lambda: apply_blur_torch(dummy_heatmaps, sigma))

    # Torch on GPU
    dummy_gpu = dummy_heatmaps.to(device)
    def blur_gpu():
        ks = int(6 * sigma + 1)
        if ks % 2 == 0:
            ks += 1
        pad = ks // 2
        x = torch.arange(ks, dtype=torch.float32, device=device) - pad
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        F_count, C, H, W = dummy_gpu.shape
        flat = dummy_gpu.reshape(-1, 1, H, W)
        k_h = kernel_1d.reshape(1, 1, 1, ks)
        flat = TF.pad(flat, (pad, pad, 0, 0), mode='reflect')
        flat = TF.conv2d(flat, k_h)
        k_v = kernel_1d.reshape(1, 1, ks, 1)
        flat = TF.pad(flat, (0, 0, pad, pad), mode='reflect')
        flat = TF.conv2d(flat, k_v)
        torch.cuda.synchronize()
        return flat.reshape(F_count, C, H, W)

    _, t_torch_gpu = timed(f"torch conv2d GPU ({n_hm} frames x 16 channels)", blur_gpu)

    print(f"  Speedup (torch CPU): {t_scipy / t_torch_cpu:.1f}x")
    print(f"  Speedup (torch GPU): {t_scipy / t_torch_gpu:.1f}x")
    print()

    # =========================================================================
    # Optimization 5: Skip flip augmentation
    # =========================================================================
    print("=" * 60)
    print("OPT 5: SH flip augmentation — with vs without")
    print("=" * 60)

    preprocessed_flip = []
    for frame in frames_rgb:
        cropped, _ = crop_and_resize(frame, bbox, target_size=256)
        cropped_flip = cropped[:, ::-1].copy()
        preprocessed_flip.append(_normalize_for_sh(cropped_flip))

    bs = 128  # use the faster batch size

    def run_sh_with_flip():
        results = []
        with torch.no_grad():
            for start in range(0, n_frames, bs):
                end = min(start + bs, n_frames)
                inp = torch.from_numpy(np.stack(preprocessed[start:end])).to(device)
                out = sh_model(inp)
                hm = out[-1].cpu().numpy()
                inp_f = torch.from_numpy(np.stack(preprocessed_flip[start:end])).to(device)
                out_f = sh_model(inp_f)
                hm_f = out_f[-1].cpu().numpy()
                for i in range(end - start):
                    results.append((hm[i] + _flip_heatmaps(hm_f[i])) / 2.0)
        return results

    def run_sh_no_flip():
        results = []
        with torch.no_grad():
            for start in range(0, n_frames, bs):
                end = min(start + bs, n_frames)
                inp = torch.from_numpy(np.stack(preprocessed[start:end])).to(device)
                out = sh_model(inp)
                hm = out[-1].cpu().numpy()
                for i in range(end - start):
                    results.append(hm[i])
        return results

    _, t_flip = timed(f"SH with flip (bs={bs}, {n_frames} frames)", run_sh_with_flip)
    _, t_noflip = timed(f"SH without flip (bs={bs}, {n_frames} frames)", run_sh_no_flip)
    print(f"  Speedup: {t_flip / t_noflip:.1f}x")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Frames tested: {n_frames}")
    print()
    print(f"  1. YOLO sampling (every 50th):  {t_yolo_all:.1f}s -> {t_yolo_sampled:.1f}s  ({t_yolo_all/t_yolo_sampled:.0f}x)")
    print(f"  2. Video seek:                  {t_extract:.1f}s -> {t_seek:.1f}s  ({'%.0fx' % (t_extract/t_seek) if t_seek < t_extract else 'no improvement'})")
    print(f"  3. SH batch size 32->128:       see above timings")
    print(f"  4. Blur torch GPU vs scipy:     {t_scipy:.1f}s -> {t_torch_gpu:.1f}s  ({t_scipy/t_torch_gpu:.0f}x) [per {n_hm} frames, multiply by num_steps]")
    print(f"  5. Skip flip augmentation:      {t_flip:.1f}s -> {t_noflip:.1f}s  ({t_flip/t_noflip:.1f}x)")


if __name__ == "__main__":
    main()
