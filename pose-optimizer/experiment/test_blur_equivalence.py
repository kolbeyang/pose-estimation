"""Verify that torch conv2d blur matches scipy gaussian_filter output.

The torch implementation uses a slightly different boundary convention
(torch reflect does NOT repeat the edge pixel; scipy reflect DOES).
Interior pixels should match closely; boundary pixels may differ.

Usage:
    uv run python experiment/test_blur_equivalence.py
"""

import sys
import os
import numpy as np
import torch
import scipy.ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scoring import apply_blur


def apply_blur_scipy(heatmaps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Reference implementation using scipy (the old code)."""
    result = torch.zeros_like(heatmaps)
    for f in range(heatmaps.shape[0]):
        for c in range(heatmaps.shape[1]):
            hm_np = heatmaps[f, c].numpy()
            blurred = scipy.ndimage.gaussian_filter(hm_np, sigma=sigma)
            result[f, c] = torch.tensor(blurred, dtype=torch.float32)
    return result


def test_blur(name, heatmaps, sigma):
    """Compare new apply_blur against scipy reference."""
    heatmaps_cpu = heatmaps.cpu()
    ref = apply_blur_scipy(heatmaps_cpu, sigma)
    new = apply_blur(heatmaps_cpu, sigma)

    # Interior: skip boundary pixels affected by padding differences
    margin = max(1, int(4 * sigma + 1))
    H, W = heatmaps.shape[2], heatmaps.shape[3]
    margin = min(margin, H // 4)  # don't skip more than 25% of image
    interior_ref = ref[:, :, margin:-margin, margin:-margin]
    interior_new = new[:, :, margin:-margin, margin:-margin]

    int_diff = (interior_ref - interior_new).abs()
    int_max = int_diff.max().item()
    int_mean = int_diff.mean().item()
    ref_range = interior_ref.abs().max().item()
    int_rel = int_max / ref_range if ref_range > 0 else 0

    # Full image stats
    full_diff = (ref - new).abs()
    full_max = full_diff.max().item()
    full_rel = full_max / ref.abs().max().item() if ref.abs().max().item() > 0 else 0

    int_pass = int_rel < 0.01  # <1% relative error on interior
    status = "PASS" if int_pass else "FAIL"
    print(f"  [{status}] {name}:")
    print(f"         interior: max_rel={int_rel:.4%}, max_abs={int_max:.6f}, mean_abs={int_mean:.6f}")
    print(f"         full:     max_rel={full_rel:.4%}, max_abs={full_max:.6f}")
    return int_pass


def main():
    print("Blur equivalence test: torch conv2d vs scipy gaussian_filter")
    print("=" * 60)

    all_pass = True

    # Test 1-3: Various sigmas on random data
    hm = torch.rand(10, 16, 64, 64)
    for sigma in [0.5, 2.0, 4.0, 8.0, 16.0]:
        print(f"\nTest: Random heatmaps, sigma={sigma}")
        all_pass &= test_blur(f"sigma={sigma}", hm, sigma)

    # Test 4: Peaked heatmaps (realistic SH output)
    print("\nTest: Peaked heatmaps (realistic SH output), sigma=4.0")
    hm_peaked = torch.zeros(5, 16, 64, 64)
    for f in range(5):
        for c in range(16):
            cy, cx = np.random.randint(15, 49), np.random.randint(15, 49)
            hm_peaked[f, c, cy - 2:cy + 3, cx - 2:cx + 3] = torch.rand(5, 5)
            hm_peaked[f, c, cy, cx] = 1.0
    all_pass &= test_blur("peaked sigma=4.0", hm_peaked, 4.0)

    # Test 5: Single frame
    print("\nTest: Single frame, sigma=4.0")
    all_pass &= test_blur("single frame", torch.rand(1, 16, 64, 64), 4.0)

    # Test 6: sigma=0 should return input unchanged
    print("\nTest: sigma=0 (no-op)")
    hm_orig = torch.rand(3, 16, 64, 64)
    result = apply_blur(hm_orig, 0.0)
    is_same = torch.equal(hm_orig, result)
    status = "PASS" if is_same else "FAIL"
    print(f"  [{status}] sigma=0 returns input unchanged: {is_same}")
    all_pass &= is_same

    # Test 7: GPU if available
    if torch.cuda.is_available():
        print("\nTest: GPU execution, sigma=4.0")
        hm_gpu = torch.rand(10, 16, 64, 64, device="cuda")
        result_gpu = apply_blur(hm_gpu, 4.0)
        assert result_gpu.device.type == "cuda", f"Expected CUDA, got {result_gpu.device}"
        # Compare GPU to CPU torch (not scipy — both should match)
        result_cpu = apply_blur(hm_gpu.cpu(), 4.0)
        gpu_diff = (result_cpu - result_gpu.cpu()).abs().max().item()
        gpu_pass = gpu_diff < 1e-5
        status = "PASS" if gpu_pass else "FAIL"
        print(f"  [{status}] GPU vs CPU max diff: {gpu_diff:.8f}")
        all_pass &= gpu_pass

    # Test 8: Speed comparison
    print("\nSpeed comparison (500 frames, sigma=4.0):")
    import time
    hm_speed = torch.rand(500, 16, 64, 64)
    t0 = time.time()
    apply_blur_scipy(hm_speed, 4.0)
    t_scipy = time.time() - t0
    t0 = time.time()
    apply_blur(hm_speed, 4.0)
    t_torch = time.time() - t0
    print(f"  scipy: {t_scipy:.2f}s, torch: {t_torch:.2f}s, speedup: {t_scipy / t_torch:.1f}x")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
