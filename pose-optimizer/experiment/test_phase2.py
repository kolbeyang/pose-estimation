"""Phase 2 tests: FK, evaluation metrics, scoring gradients."""

import sys
import os

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton import NUM_JOINTS, EVAL_JOINTS, DEFAULT_BONE_LENGTHS, REST_DIRECTIONS, PARENTS
from camera import Camera
from fk import forward_kinematics, forward_kinematics_batch, positions_to_fk_params
from evaluate import (
    mpjpe, p_mpjpe, si_mpjpe, vw_mpjpe, vw_si_mpjpe,
    mpjve, si_mpjve, vw_mpjve, vw_si_mpjve,
    compute_visibility_weights, evaluate, root_relative,
)
from scoring import (
    heatmap_score_batch, compute_total_score_batch,
    generate_synthetic_heatmaps, motion_penalty_position_batch,
    motion_penalty_rotation_batch,
)
from cmu_data import (
    load_calibration, load_ground_truth_sequence,
    get_sequence_dir, DEFAULT_DATA_ROOT,
)


# ---------------------------------------------------------------------------
# FK tests
# ---------------------------------------------------------------------------

class TestFK:
    def test_fk_identity_rotations(self):
        """With zero rotations, FK should produce positions along rest directions."""
        root_pos = torch.zeros(3)
        root_rot = torch.zeros(3)
        local_rots = torch.zeros(NUM_JOINTS, 3)
        bone_lengths = torch.tensor(DEFAULT_BONE_LENGTHS, dtype=torch.float32)

        positions = forward_kinematics(root_pos, root_rot, local_rots, bone_lengths)
        assert positions.shape == (NUM_JOINTS, 3)
        # Root should be at origin
        np.testing.assert_allclose(positions[0].numpy(), [0, 0, 0], atol=1e-6)

    def test_fk_roundtrip_synthetic(self):
        """positions -> params -> FK -> positions should match."""
        # Create a known skeleton
        root_pos_np = np.array([0.5, -0.3, 2.0])
        root_rot_np = np.array([0.1, -0.2, 0.3])
        local_rots_np = np.random.randn(NUM_JOINTS, 3) * 0.3
        bone_lengths_np = DEFAULT_BONE_LENGTHS.copy()

        with torch.no_grad():
            positions = forward_kinematics(
                torch.tensor(root_pos_np, dtype=torch.float32),
                torch.tensor(root_rot_np, dtype=torch.float32),
                torch.tensor(local_rots_np, dtype=torch.float32),
                torch.tensor(bone_lengths_np, dtype=torch.float32),
            ).numpy()

        # Inverse FK
        rp, rr, lr, bl = positions_to_fk_params(positions)

        # Forward again
        with torch.no_grad():
            reconstructed = forward_kinematics(
                torch.tensor(rp, dtype=torch.float32),
                torch.tensor(rr, dtype=torch.float32),
                torch.tensor(lr, dtype=torch.float32),
                torch.tensor(bl, dtype=torch.float32),
            ).numpy()

        error = np.mean(np.linalg.norm(reconstructed - positions, axis=-1))
        assert error < 0.0001, f"Roundtrip error {error*100:.4f} cm too large"

    def test_fk_batch_matches_single(self):
        """Batch FK should produce same results as sequential single FK."""
        n_frames = 5
        root_pos = torch.randn(n_frames, 3)
        root_rot = torch.randn(n_frames, 3) * 0.3
        local_rots = torch.randn(n_frames, NUM_JOINTS, 3) * 0.3
        bone_lengths = torch.tensor(DEFAULT_BONE_LENGTHS, dtype=torch.float32)

        batch_result = forward_kinematics_batch(root_pos, root_rot, local_rots, bone_lengths)

        for i in range(n_frames):
            single_result = forward_kinematics(
                root_pos[i], root_rot[i], local_rots[i], bone_lengths,
            )
            np.testing.assert_allclose(
                batch_result[i].detach().numpy(),
                single_result.detach().numpy(),
                atol=1e-5,
            )

    def test_fk_roundtrip_real_data(self):
        """FK roundtrip on real GT data should be < 0.01 cm."""
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        if not os.path.isdir(seq_dir):
            pytest.skip("CMU data not available")

        cams = load_calibration(seq_dir)
        cal = cams["00_00"]
        R, t = cal["R"], cal["t"]

        gt = load_ground_truth_sequence(seq_dir, [0], person_idx=0)
        if gt[0] is None:
            pytest.skip("GT frame 0 not available")

        # Convert to camera space
        pts_world = gt[0]
        pts_cam = (R @ pts_world.T + t).T

        rp, rr, lr, bl = positions_to_fk_params(pts_cam)
        with torch.no_grad():
            reconstructed = forward_kinematics(
                torch.tensor(rp, dtype=torch.float32),
                torch.tensor(rr, dtype=torch.float32),
                torch.tensor(lr, dtype=torch.float32),
                torch.tensor(bl, dtype=torch.float32),
            ).numpy()

        error = np.mean(np.linalg.norm(reconstructed - pts_cam, axis=-1))
        assert error < 0.01, f"Real data roundtrip error {error*100:.4f} cm"

    def test_fk_gradients(self):
        """FK output should have gradients wrt parameters."""
        root_pos = torch.randn(3, requires_grad=True)
        root_rot = torch.randn(3, requires_grad=True)
        local_rots = torch.randn(NUM_JOINTS, 3, requires_grad=True)
        bone_lengths = torch.tensor(DEFAULT_BONE_LENGTHS, dtype=torch.float32, requires_grad=True)

        positions = forward_kinematics(root_pos, root_rot, local_rots, bone_lengths)
        loss = positions.sum()
        loss.backward()

        assert root_pos.grad is not None
        assert root_rot.grad is not None
        assert local_rots.grad is not None
        assert bone_lengths.grad is not None


# ---------------------------------------------------------------------------
# Evaluation metrics tests
# ---------------------------------------------------------------------------

class TestEvaluationMetrics:
    def setup_method(self):
        np.random.seed(42)
        self.n_frames = 20
        self.n_joints = 16
        # Random GT
        self.gt = np.random.randn(self.n_frames, self.n_joints, 3)
        # Predicted = GT + noise
        self.pred = self.gt + np.random.randn(self.n_frames, self.n_joints, 3) * 0.1

        # Camera for VW
        self.camera = Camera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            image_size=(480, 640),
        )

    def test_mpjpe_positive(self):
        val = mpjpe(self.pred, self.gt)
        assert val > 0
        assert np.isfinite(val)

    def test_mpjpe_zero_for_identical(self):
        val = mpjpe(self.gt, self.gt)
        assert val < 1e-10

    def test_si_mpjpe_leq_mpjpe(self):
        """SI-MPJPE should be <= MPJPE since s=1 is always valid."""
        pred_rr = root_relative(self.pred)
        gt_rr = root_relative(self.gt)
        val_si = si_mpjpe(pred_rr, gt_rr)
        val_base = mpjpe(pred_rr, gt_rr)
        assert val_si <= val_base + 1e-10

    def test_p_mpjpe_leq_mpjpe(self):
        """P-MPJPE should be <= MPJPE since Procrustes can only help."""
        val_p = p_mpjpe(self.pred, self.gt)
        val_base = mpjpe(self.pred, self.gt)
        assert val_p <= val_base + 1e-10

    def test_vw_mpjpe_with_uniform_weights(self):
        """With all-ones weights, VW-MPJPE should equal MPJPE."""
        weights = np.ones((self.n_frames, self.n_joints))
        val_vw = vw_mpjpe(self.pred, self.gt, weights)
        val_base = mpjpe(self.pred, self.gt)
        np.testing.assert_allclose(val_vw, val_base, atol=1e-10)

    def test_vw_si_mpjpe_finite(self):
        weights = np.ones((self.n_frames, self.n_joints))
        pred_rr = root_relative(self.pred)
        gt_rr = root_relative(self.gt)
        val = vw_si_mpjpe(pred_rr, gt_rr, weights)
        assert np.isfinite(val)
        assert val > 0

    def test_mpjve_positive(self):
        val = mpjve(self.pred, self.gt)
        assert val > 0
        assert np.isfinite(val)

    def test_si_mpjve_finite(self):
        pred_rr = root_relative(self.pred)
        gt_rr = root_relative(self.gt)
        val = si_mpjve(pred_rr, gt_rr)
        assert np.isfinite(val)
        assert val > 0

    def test_vw_mpjve_finite(self):
        weights = np.ones((self.n_frames, self.n_joints))
        val = vw_mpjve(self.pred, self.gt, weights)
        assert np.isfinite(val)
        assert val > 0

    def test_vw_si_mpjve_finite(self):
        weights = np.ones((self.n_frames, self.n_joints))
        pred_rr = root_relative(self.pred)
        gt_rr = root_relative(self.gt)
        val = vw_si_mpjve(pred_rr, gt_rr, weights)
        assert np.isfinite(val)
        assert val > 0

    def test_all_nine_metrics_finite(self):
        """All 9 metrics should produce finite positive values."""
        pred_rr = root_relative(self.pred)
        gt_rr = root_relative(self.gt)
        weights = np.ones((self.n_frames, self.n_joints))

        metrics = {
            "mpjpe": mpjpe(pred_rr, gt_rr),
            "p_mpjpe": p_mpjpe(pred_rr, gt_rr),
            "si_mpjpe": si_mpjpe(pred_rr, gt_rr),
            "vw_mpjpe": vw_mpjpe(pred_rr, gt_rr, weights),
            "vw_si_mpjpe": vw_si_mpjpe(pred_rr, gt_rr, weights),
            "mpjve": mpjve(pred_rr, gt_rr),
            "si_mpjve": si_mpjve(pred_rr, gt_rr),
            "vw_mpjve": vw_mpjve(pred_rr, gt_rr, weights),
            "vw_si_mpjve": vw_si_mpjve(pred_rr, gt_rr, weights),
        }
        for name, val in metrics.items():
            assert np.isfinite(val), f"{name} is not finite: {val}"
            assert val >= 0, f"{name} is negative: {val}"

    def test_compute_visibility_weights(self):
        """Visibility weights should be binary and based on frame bounds."""
        # Points at various distances -- all with positive Z
        gt_cam = np.zeros((5, 16, 3))
        gt_cam[:, :, 2] = 2.0  # 2 meters depth

        # Joint 0: at image center -> in frame
        gt_cam[:, 0, 0] = 0.0
        gt_cam[:, 0, 1] = 0.0

        # Joint 1: way off to the right -> out of frame
        gt_cam[:, 1, 0] = 10.0  # This will project far right

        vis = compute_visibility_weights(gt_cam, self.camera)
        assert vis.shape == (5, 16)
        # Joint 0 should be in frame
        assert vis[0, 0] == 1.0
        # Joint 1 should be out of frame
        assert vis[0, 1] == 0.0

    def test_evaluate_function(self):
        """The evaluate() function should return all 9 metrics."""
        # Need GT in camera space for VW
        camera = Camera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            image_size=(480, 640),
        )
        gt_cam = np.random.randn(10, 16, 3) * 0.5
        gt_cam[:, :, 2] = 2.0 + np.abs(gt_cam[:, :, 2])  # positive Z
        pred_cam = gt_cam + np.random.randn(10, 16, 3) * 0.05

        results = evaluate(pred_cam, gt_cam, camera)
        expected_keys = [
            "mpjpe", "p_mpjpe", "si_mpjpe", "vw_mpjpe", "vw_si_mpjpe",
            "mpjve", "si_mpjve", "vw_mpjve", "vw_si_mpjve",
        ]
        for key in expected_keys:
            assert key in results, f"Missing metric: {key}"
            assert np.isfinite(results[key]), f"{key} not finite"


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    def test_synthetic_heatmap_generation(self):
        """Synthetic heatmaps should have correct shape and reasonable values."""
        target_2d = np.random.rand(5, 16, 2) * np.array([640, 480])
        heatmaps, affine = generate_synthetic_heatmaps(
            target_2d, image_size=(480, 640), heatmap_size=64, sigma=50.0,
        )
        assert heatmaps.shape == (5, 16, 64, 64)
        assert affine.shape == (2, 3)
        # Max of each heatmap should be near 1.0 (center of Gaussian)
        for f in range(5):
            for j in range(16):
                assert heatmaps[f, j].max() > 0.5

    def test_scoring_produces_gradients(self):
        """Scoring function should produce nonzero gradients."""
        n_frames = 3
        n_joints = 16

        # Create synthetic heatmaps
        target_2d = np.random.rand(n_frames, n_joints, 2) * np.array([640, 480])
        heatmaps_np, affine_np = generate_synthetic_heatmaps(
            target_2d, image_size=(480, 640), heatmap_size=64, sigma=50.0,
        )

        heatmaps_t = torch.tensor(heatmaps_np, dtype=torch.float32)
        affine_t = torch.tensor(affine_np, dtype=torch.float32)

        # Projected 2D with grad -- slightly offset from target
        projected_2d = torch.tensor(
            target_2d + np.random.randn(n_frames, n_joints, 2) * 10,
            dtype=torch.float32,
            requires_grad=True,
        )
        visibility = torch.ones(n_frames, n_joints)

        score = heatmap_score_batch(
            projected_2d, heatmaps_t, affine_t, visibility,
            use_mpii_mapping=False,
        )
        score.backward()

        assert projected_2d.grad is not None
        assert torch.any(projected_2d.grad != 0)

    def test_total_score_batch(self):
        """compute_total_score_batch should produce a scalar with gradients."""
        n_frames = 3
        n_joints = 16

        target_2d = np.random.rand(n_frames, n_joints, 2) * np.array([640, 480])
        heatmaps_np, affine_np = generate_synthetic_heatmaps(
            target_2d, (480, 640), 64, 50.0,
        )

        positions = torch.randn(n_frames, n_joints, 3, requires_grad=True)
        projected_2d = torch.tensor(
            target_2d + np.random.randn(n_frames, n_joints, 2) * 5,
            dtype=torch.float32,
            requires_grad=True,
        )
        local_rots = torch.randn(n_frames, n_joints, 3, requires_grad=True)
        visibility = torch.ones(n_frames, n_joints)
        rot_weights = torch.ones(n_joints)

        score, details = compute_total_score_batch(
            positions, projected_2d, local_rots, visibility,
            position_penalty_weight=500.0,
            rotation_per_joint_weights=rot_weights,
            heatmaps=torch.tensor(heatmaps_np),
            affine=torch.tensor(affine_np),
            use_mpii_mapping=False,
        )

        assert score.dim() == 0  # scalar
        loss = -score
        loss.backward()
        assert projected_2d.grad is not None

    def test_motion_penalties(self):
        """Motion penalties should be zero for identical consecutive frames."""
        positions = torch.zeros(5, 16, 3)
        assert motion_penalty_position_batch(positions).item() == 0.0

        local_rots = torch.zeros(5, 16, 3)
        weights = torch.ones(16)
        assert motion_penalty_rotation_batch(local_rots, weights).item() == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
