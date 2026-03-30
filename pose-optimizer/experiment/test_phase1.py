"""Phase 1 tests: skeleton, camera, config, data loading."""

import sys
import os
import json
import tempfile

import numpy as np
import pytest

# Add parent dir to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skeleton import (
    NUM_JOINTS, JOINT_NAMES, PARENTS, EVAL_JOINTS,
    coco19_to_skeleton, mediapipe_to_skeleton, mediapipe_visibility_to_skeleton,
    strip_head_joint, mpii_to_skeleton,
)
from camera import Camera
from config import RunConfig, ExampleConfig, OptimizationConfig, load_config, save_config
from cmu_data import (
    load_calibration, load_ground_truth_sequence,
    get_sequence_dir, DEFAULT_DATA_ROOT,
)


# ---------------------------------------------------------------------------
# Skeleton tests
# ---------------------------------------------------------------------------

class TestSkeleton:
    def test_joint_count(self):
        assert NUM_JOINTS == 16
        assert len(JOINT_NAMES) == 16
        assert len(PARENTS) == 16

    def test_eval_joints(self):
        excluded = {7, 9}  # Spine (FK-internal) and Head (mode-dependent)
        for j in EVAL_JOINTS:
            assert j not in excluded
        assert len(EVAL_JOINTS) == 14

    def test_coco19_to_skeleton_shape(self):
        coco19 = np.random.randn(19, 3)
        skel = coco19_to_skeleton(coco19)
        assert skel.shape == (16, 3)

    def test_coco19_to_skeleton_pelvis_is_body_center(self):
        coco19 = np.random.randn(19, 3)
        skel = coco19_to_skeleton(coco19)
        np.testing.assert_allclose(skel[0], coco19[2])  # BodyCenter

    def test_coco19_to_skeleton_spine_is_midpoint(self):
        coco19 = np.random.randn(19, 3)
        skel = coco19_to_skeleton(coco19)
        expected_spine = (coco19[2] + coco19[0]) / 2.0
        np.testing.assert_allclose(skel[7], expected_spine)

    def test_mediapipe_to_skeleton_shape(self):
        landmarks = np.random.randn(33, 3)
        skel = mediapipe_to_skeleton(landmarks)
        assert skel.shape == (16, 3)

    def test_mediapipe_visibility_to_skeleton_shape(self):
        vis = np.random.rand(33)
        skel_vis = mediapipe_visibility_to_skeleton(vis)
        assert skel_vis.shape == (16,)

    def test_strip_head_joint(self):
        arr = np.random.randn(17, 3)
        result = strip_head_joint(arr)
        assert result.shape == (16, 3)

    def test_strip_head_joint_batch(self):
        arr = np.random.randn(5, 17, 3)
        result = strip_head_joint(arr)
        assert result.shape == (5, 16, 3)

    def test_mpii_to_skeleton_shape(self):
        mpii = np.random.randn(16, 3)
        skel = mpii_to_skeleton(mpii)
        assert skel.shape == (17, 3)

    def test_mpii_to_skeleton_pelvis_direct(self):
        """Pelvis should map directly from MPII[6], not midpoint."""
        mpii = np.random.randn(16, 3)
        skel = mpii_to_skeleton(mpii)
        np.testing.assert_allclose(skel[0], mpii[6])

    def test_mpii_to_skeleton_spine_direct(self):
        """Spine should map directly from MPII[7], not midpoint."""
        mpii = np.random.randn(16, 3)
        skel = mpii_to_skeleton(mpii)
        np.testing.assert_allclose(skel[7], mpii[7])


# ---------------------------------------------------------------------------
# Camera tests
# ---------------------------------------------------------------------------

class TestCamera:
    def setup_method(self):
        self.camera = Camera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            image_size=(480, 640),
        )

    def test_camera_to_image_single_point(self):
        """A point at (0, 0, 1) should project to (cx, cy)."""
        pt = np.array([0.0, 0.0, 1.0])
        proj = self.camera.camera_to_image(pt)
        np.testing.assert_allclose(proj, [320.0, 240.0], atol=1e-6)

    def test_camera_to_image_offset(self):
        """A point at (0.1, 0.2, 1.0) should project to (cx + fx*0.1, cy + fy*0.2)."""
        pt = np.array([0.1, 0.2, 1.0])
        proj = self.camera.camera_to_image(pt)
        np.testing.assert_allclose(proj, [370.0, 340.0], atol=1e-6)

    def test_camera_to_image_batch(self):
        pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]])
        proj = self.camera.camera_to_image(pts)
        assert proj.shape == (2, 2)
        np.testing.assert_allclose(proj[0], [320.0, 240.0], atol=1e-6)

    def test_camera_to_image_torch(self):
        import torch
        pt = torch.tensor([0.0, 0.0, 1.0])
        proj = self.camera.camera_to_image_torch(pt)
        np.testing.assert_allclose(proj.detach().numpy(), [320.0, 240.0], atol=1e-6)

    def test_camera_to_image_torch_batch(self):
        import torch
        pts = torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.2, 1.0]])
        proj = self.camera.camera_to_image_torch(pts)
        assert proj.shape == (2, 2)

    def test_world_camera_roundtrip(self):
        """world_to_camera followed by camera_to_world should be identity."""
        R = np.eye(3)
        # Small rotation
        theta = 0.3
        R[0, 0] = np.cos(theta)
        R[0, 2] = np.sin(theta)
        R[2, 0] = -np.sin(theta)
        R[2, 2] = np.cos(theta)
        t = np.array([[1.0], [2.0], [3.0]])

        cam = Camera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            image_size=(480, 640), R=R, t=t,
        )

        pts_world = np.random.randn(10, 3)
        pts_cam = cam.world_to_camera(pts_world)
        pts_world_back = cam.camera_to_world(pts_cam)
        np.testing.assert_allclose(pts_world_back, pts_world, atol=1e-10)

    def test_world_camera_roundtrip_single(self):
        cam = Camera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            image_size=(480, 640),
            R=np.eye(3), t=np.zeros((3, 1)),
        )
        pt = np.array([1.0, 2.0, 3.0])
        pt_cam = cam.world_to_camera(pt)
        pt_back = cam.camera_to_world(pt_cam)
        np.testing.assert_allclose(pt_back, pt, atol=1e-10)

    def test_is_in_frame(self):
        pts = np.array([
            [320.0, 240.0],  # center -- in
            [-1.0, 240.0],   # left edge -- out
            [640.0, 240.0],  # right edge -- out (>= width)
            [0.0, 0.0],      # top-left corner -- in
            [639.9, 479.9],  # just inside -- in
        ])
        mask = self.camera.is_in_frame(pts)
        expected = [True, False, False, True, True]
        np.testing.assert_array_equal(mask, expected)

    def test_is_in_frame_multidim(self):
        """Test (F, K, 2) input."""
        pts = np.array([
            [[100.0, 100.0], [700.0, 100.0]],
            [[320.0, 240.0], [-10.0, 240.0]],
        ])  # (2, 2, 2)
        mask = self.camera.is_in_frame(pts)
        assert mask.shape == (2, 2)
        expected = [[True, False], [True, False]]
        np.testing.assert_array_equal(mask, expected)

    def test_no_extrinsics_raises(self):
        with pytest.raises(ValueError):
            self.camera.world_to_camera(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            self.camera.camera_to_world(np.array([1.0, 2.0, 3.0]))

    def test_serialization_roundtrip(self):
        cam = Camera(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            image_size=(480, 640),
            R=np.eye(3), t=np.zeros((3, 1)),
        )
        d = cam.to_dict()
        cam2 = Camera.from_dict(d)
        assert cam2.fx == cam.fx
        assert cam2.image_size == cam.image_size
        np.testing.assert_allclose(cam2.R, cam.R)

    def test_from_panoptic_calibration(self):
        K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
        R = np.eye(3)
        t = np.zeros((3, 1))
        cam = Camera.from_panoptic_calibration(K, R, t, (640, 480))
        assert cam.fx == 500.0
        assert cam.image_size == (480, 640)
        assert cam.R is not None


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        cfg = RunConfig()
        assert len(cfg.examples) == 1
        assert cfg.optimization.num_steps == 50

    def test_config_json_roundtrip(self):
        cfg = RunConfig(
            examples=[ExampleConfig(sequence="test_seq", num_frames=50)],
            target_fps=15.0,
            optimization=OptimizationConfig(num_steps=100, learning_rate=0.005),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            save_config(cfg, f.name)
            loaded = load_config(f.name)
        os.unlink(f.name)

        assert loaded.examples[0].sequence == "test_seq"
        assert loaded.optimization.num_steps == 100
        assert loaded.target_fps == 15.0

    def test_rotation_penalty_per_joint(self):
        opt = OptimizationConfig()
        rpj = opt.rotation_penalty_per_joint
        assert rpj.shape == (16,)
        assert rpj[0] == opt.rotation_penalty_scalar * 3.0
        assert rpj[15] == opt.rotation_penalty_scalar * 0.1


# ---------------------------------------------------------------------------
# Data loading tests (require actual CMU data)
# ---------------------------------------------------------------------------

class TestCMUData:
    """Tests that require actual CMU Panoptic data on disk."""

    @pytest.fixture(autouse=True)
    def _check_data(self):
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        if not os.path.isdir(seq_dir):
            pytest.skip("CMU Panoptic data not available")

    def test_load_calibration(self):
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        cams = load_calibration(seq_dir)
        assert "00_00" in cams
        K = cams["00_00"]["K"]
        assert K.shape == (3, 3)
        R = cams["00_00"]["R"]
        assert R.shape == (3, 3)

    def test_load_ground_truth(self):
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        gt = load_ground_truth_sequence(seq_dir, [0, 1, 2], person_idx=0)
        assert len(gt) == 3
        for g in gt:
            if g is not None:
                assert g.shape == (16, 3)
                # Values should be in centimeters -- roughly reasonable range
                assert np.max(np.abs(g)) < 1000  # less than 10 meters

    def test_camera_from_panoptic(self):
        """Load real calibration and create Camera with extrinsics."""
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        cams = load_calibration(seq_dir)
        cal = cams["00_00"]
        cam = Camera.from_panoptic_calibration(
            cal["K"], cal["R"], cal["t"], cal["resolution"],
        )
        assert cam.R is not None
        assert cam.t is not None
        # Project GT point to 2D -- should be within image bounds
        gt = load_ground_truth_sequence(seq_dir, [0], person_idx=0)
        if gt[0] is not None:
            gt_cam = cam.world_to_camera(gt[0])
            proj = cam.camera_to_image(gt_cam)
            assert proj.shape == (16, 2)
            # At least some joints should be in frame
            in_frame = cam.is_in_frame(proj)
            assert in_frame.sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
