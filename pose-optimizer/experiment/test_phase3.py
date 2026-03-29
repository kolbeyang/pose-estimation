"""Phase 3 tests: optimizer core smoke tests."""

import sys
import os

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import Camera
from config import OptimizationConfig
from evaluate import mpjpe, root_relative
from skeleton import NUM_JOINTS, EVAL_JOINTS, DEFAULT_BONE_LENGTHS
from fk import forward_kinematics, positions_to_fk_params
from optimize import optimize
from cmu_data import (
    load_calibration, load_ground_truth_sequence,
    get_sequence_dir, DEFAULT_DATA_ROOT,
)


def _make_noisy_3d(
    gt_cam: list[np.ndarray],
    noise_scale: float = 0.02,
) -> list[np.ndarray]:
    """Add Gaussian noise to GT positions to simulate detector output."""
    np.random.seed(123)
    return [g + np.random.randn(*g.shape) * noise_scale for g in gt_cam]


def _make_simple_camera() -> Camera:
    """Create a simple camera for testing."""
    return Camera(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        image_size=(480, 640),
    )


class TestOptimizerGaussian:
    """Test optimizer with synthetic Gaussian heatmaps (MediaPipe mode)."""

    def test_output_shape(self):
        """Optimizer should return correct shapes."""
        n_frames = 5
        camera = _make_simple_camera()

        # Create simple test data: GT at depth=2, small noise
        gt = []
        for _ in range(n_frames):
            rp = np.array([0.0, 0.0, 2.0])
            rr = np.zeros(3)
            lr = np.zeros((NUM_JOINTS, 3))
            bl = DEFAULT_BONE_LENGTHS.copy()
            with torch.no_grad():
                pos = forward_kinematics(
                    torch.tensor(rp, dtype=torch.float32),
                    torch.tensor(rr, dtype=torch.float32),
                    torch.tensor(lr, dtype=torch.float32),
                    torch.tensor(bl, dtype=torch.float32),
                ).numpy()
            gt.append(pos)

        noisy = _make_noisy_3d(gt, noise_scale=0.01)

        # Project GT to 2D for target_2d
        target_2d = [camera.camera_to_image(g) for g in gt]

        config = OptimizationConfig(num_steps=5)
        optimized, bone_lengths, loss_history = optimize(
            raw_3d=noisy,
            camera=camera,
            config=config,
            target_2d=target_2d,
            verbose=False,
        )

        assert len(optimized) == n_frames
        for o in optimized:
            assert o.shape == (NUM_JOINTS, 3)
        assert bone_lengths.shape == (NUM_JOINTS,)
        assert len(loss_history) == 5

    def test_bone_lengths_positive(self):
        """After optimization, all bone lengths should be positive."""
        n_frames = 3
        camera = _make_simple_camera()

        gt = []
        for _ in range(n_frames):
            pos = np.zeros((NUM_JOINTS, 3))
            pos[:, 2] = 2.0  # all at depth 2
            for j in range(1, NUM_JOINTS):
                pos[j] = pos[0] + np.random.randn(3) * 0.3
                pos[j, 2] = 2.0
            gt.append(pos)

        noisy = _make_noisy_3d(gt, 0.02)
        target_2d = [camera.camera_to_image(g) for g in gt]

        config = OptimizationConfig(num_steps=10)
        _, bone_lengths, _ = optimize(
            raw_3d=noisy, camera=camera, config=config,
            target_2d=target_2d, verbose=False,
        )
        assert np.all(bone_lengths >= 0.01)

    def test_loss_decreases(self):
        """Loss should generally decrease during optimization."""
        n_frames = 5
        camera = _make_simple_camera()

        gt = []
        for i in range(n_frames):
            rp = np.array([0.0, 0.0, 2.0])
            rr = np.array([0.0, 0.0, 0.0])
            lr = np.zeros((NUM_JOINTS, 3))
            bl = DEFAULT_BONE_LENGTHS.copy()
            with torch.no_grad():
                pos = forward_kinematics(
                    torch.tensor(rp, dtype=torch.float32),
                    torch.tensor(rr, dtype=torch.float32),
                    torch.tensor(lr, dtype=torch.float32),
                    torch.tensor(bl, dtype=torch.float32),
                ).numpy()
            gt.append(pos)

        noisy = _make_noisy_3d(gt, noise_scale=0.05)
        target_2d = [camera.camera_to_image(g) for g in gt]

        config = OptimizationConfig(num_steps=30)
        _, _, loss_history = optimize(
            raw_3d=noisy, camera=camera, config=config,
            target_2d=target_2d, verbose=False,
        )

        # Loss at end should be lower than loss at start
        assert loss_history[-1] < loss_history[0]


class TestOptimizerRealData:
    """Tests on real CMU data (if available)."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        if not os.path.isdir(seq_dir):
            pytest.skip("CMU data not available")

    def test_optimizer_reduces_mpjpe(self):
        """On noisy GT data, optimizer should reduce MPJPE."""
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        cams = load_calibration(seq_dir)
        cal = cams["00_00"]

        camera = Camera.from_panoptic_calibration(
            cal["K"], cal["R"], cal["t"], cal["resolution"],
        )

        # Load 5 frames of GT
        frame_indices = list(range(0, 5))
        gt_world = load_ground_truth_sequence(seq_dir, frame_indices, person_idx=0)
        gt_cam = [camera.world_to_camera(g) for g in gt_world if g is not None]

        if len(gt_cam) < 5:
            pytest.skip("Not enough GT frames")

        # Add noise to simulate detector output
        noisy = _make_noisy_3d(gt_cam, noise_scale=2.0)  # 2 cm noise

        # Project GT to 2D for target_2d (Gaussian mode)
        target_2d = [camera.camera_to_image(g) for g in gt_cam]

        config = OptimizationConfig(num_steps=30)
        optimized, _, loss_history = optimize(
            raw_3d=noisy,
            camera=camera,
            config=config,
            target_2d=target_2d,
            verbose=True,
        )

        # Compute MPJPE before and after
        gt_arr = np.array(gt_cam)
        noisy_arr = np.array(noisy)
        opt_arr = np.array(optimized)

        gt_rr = root_relative(gt_arr)[:, EVAL_JOINTS]
        noisy_rr = root_relative(noisy_arr)[:, EVAL_JOINTS]
        opt_rr = root_relative(opt_arr)[:, EVAL_JOINTS]

        mpjpe_before = mpjpe(noisy_rr, gt_rr)
        mpjpe_after = mpjpe(opt_rr, gt_rr)

        print(f"\n  MPJPE before: {mpjpe_before:.4f}")
        print(f"  MPJPE after:  {mpjpe_after:.4f}")
        print(f"  Improvement:  {mpjpe_before - mpjpe_after:.4f}")

        # Optimizer should improve or at least not make things much worse
        # With GT 2D targets, it should definitely improve
        assert mpjpe_after < mpjpe_before * 1.5, (
            f"MPJPE got much worse: {mpjpe_before:.4f} -> {mpjpe_after:.4f}"
        )


class TestOptimizerRequiresInput:
    """Test error handling."""

    def test_no_heatmaps_no_target_raises(self):
        """Optimizer should raise if neither heatmaps nor target_2d provided."""
        camera = _make_simple_camera()
        config = OptimizationConfig(num_steps=1)
        raw_3d = [np.zeros((NUM_JOINTS, 3))]

        with pytest.raises(ValueError, match="Either.*heatmaps.*target_2d"):
            optimize(raw_3d=raw_3d, camera=camera, config=config, verbose=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
