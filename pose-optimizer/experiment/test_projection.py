"""Projection tests: camera model, synthetic heatmaps, affine transforms.

Uses real CMU Panoptic camera parameters where possible to ensure the
projection pipeline matches actual data conditions.
"""

import sys
import os

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera import Camera
from scoring import generate_synthetic_heatmaps
from skeleton import coco19_to_skeleton, EVAL_JOINTS, NUM_JOINTS
from cmu_data import (
    load_calibration,
    load_ground_truth_sequence,
    get_sequence_dir,
    DEFAULT_DATA_ROOT,
)

# Real Panoptic-style camera parameters (typical HD camera)
_PANOPTIC_FX = 1395.0
_PANOPTIC_FY = 1395.0
_PANOPTIC_CX = 960.0
_PANOPTIC_CY = 540.0
_PANOPTIC_IMAGE_SIZE = (1080, 1920)  # (height, width)


def _make_panoptic_camera() -> Camera:
    """Create a camera with typical Panoptic HD intrinsics (no extrinsics)."""
    return Camera(
        fx=_PANOPTIC_FX, fy=_PANOPTIC_FY,
        cx=_PANOPTIC_CX, cy=_PANOPTIC_CY,
        image_size=_PANOPTIC_IMAGE_SIZE,
    )


# ---------------------------------------------------------------------------
# Test 1: camera_to_image correctness
# ---------------------------------------------------------------------------

class TestCameraToImage:
    def test_center_point(self):
        """A point at (0, 0, Z) should project to (cx, cy)."""
        cam = _make_panoptic_camera()
        pt = np.array([0.0, 0.0, 3.0])
        proj = cam.camera_to_image(pt)
        np.testing.assert_allclose(proj, [_PANOPTIC_CX, _PANOPTIC_CY], atol=1e-6)

    def test_manual_calculation(self):
        """Verify projection matches manual u = fx*X/Z + cx."""
        cam = _make_panoptic_camera()
        X, Y, Z = 0.5, -0.3, 2.0
        pt = np.array([X, Y, Z])
        proj = cam.camera_to_image(pt)
        expected_u = _PANOPTIC_FX * X / Z + _PANOPTIC_CX
        expected_v = _PANOPTIC_FY * Y / Z + _PANOPTIC_CY
        np.testing.assert_allclose(proj, [expected_u, expected_v], atol=1e-6)

    def test_behind_camera_clamped(self):
        """Points with Z < 0 should be clamped (Z=0.01), not crash."""
        cam = _make_panoptic_camera()
        pt = np.array([0.0, 0.0, -1.0])
        proj = cam.camera_to_image(pt)
        # Should project using Z=0.01 (clamped)
        assert np.all(np.isfinite(proj))

    def test_batch_shape(self):
        """Batch of points should produce correct output shape."""
        cam = _make_panoptic_camera()
        pts = np.random.randn(10, 3)
        pts[:, 2] = np.abs(pts[:, 2]) + 1.0  # positive Z
        proj = cam.camera_to_image(pts)
        assert proj.shape == (10, 2)

    def test_skeleton_shape(self):
        """16-joint skeleton should produce (16, 2) output."""
        cam = _make_panoptic_camera()
        pts = np.random.randn(NUM_JOINTS, 3)
        pts[:, 2] = 2.5
        proj = cam.camera_to_image(pts)
        assert proj.shape == (NUM_JOINTS, 2)


# ---------------------------------------------------------------------------
# Test 2: torch matches numpy
# ---------------------------------------------------------------------------

class TestTorchMatchesNumpy:
    def test_single_point(self):
        cam = _make_panoptic_camera()
        pt_np = np.array([0.3, -0.2, 2.5])
        pt_t = torch.tensor(pt_np, dtype=torch.float32)
        proj_np = cam.camera_to_image(pt_np)
        proj_t = cam.camera_to_image_torch(pt_t).detach().numpy()
        np.testing.assert_allclose(proj_t, proj_np, atol=1e-4)

    def test_batch(self):
        cam = _make_panoptic_camera()
        pts_np = np.random.randn(20, 3).astype(np.float64)
        pts_np[:, 2] = np.abs(pts_np[:, 2]) + 1.0
        pts_t = torch.tensor(pts_np, dtype=torch.float32)
        proj_np = cam.camera_to_image(pts_np)
        proj_t = cam.camera_to_image_torch(pts_t).detach().numpy()
        np.testing.assert_allclose(proj_t, proj_np, atol=1e-3)

    def test_skeleton_batch(self):
        """Multi-frame skeleton projection should match."""
        cam = _make_panoptic_camera()
        pts_np = np.random.randn(5, NUM_JOINTS, 3).astype(np.float64)
        pts_np[:, :, 2] = np.abs(pts_np[:, :, 2]) + 1.5
        pts_t = torch.tensor(pts_np, dtype=torch.float32)
        proj_np = cam.camera_to_image(pts_np)
        proj_t = cam.camera_to_image_torch(pts_t).detach().numpy()
        np.testing.assert_allclose(proj_t, proj_np, atol=1e-3)


# ---------------------------------------------------------------------------
# Test 3: world_to_camera correctness with real calibration
# ---------------------------------------------------------------------------

class TestWorldToCamera:
    @pytest.fixture(autouse=True)
    def _check_data(self):
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        if not os.path.isdir(seq_dir):
            pytest.skip("CMU Panoptic data not available")

    def test_roundtrip(self):
        """world_to_camera -> camera_to_world should be identity."""
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        cams = load_calibration(seq_dir)
        cal = cams["00_00"]
        cam = Camera.from_panoptic_calibration(
            cal["K"], cal["R"], cal["t"], cal["resolution"],
        )
        pts_world = np.random.randn(10, 3) * 100  # in cm range
        pts_cam = cam.world_to_camera(pts_world)
        pts_back = cam.camera_to_world(pts_cam)
        np.testing.assert_allclose(pts_back, pts_world, atol=1e-8)

    def test_camera_z_positive(self):
        """A point at world origin should have positive camera Z (camera looks at scene)."""
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        cams = load_calibration(seq_dir)
        cal = cams["00_00"]
        cam = Camera.from_panoptic_calibration(
            cal["K"], cal["R"], cal["t"], cal["resolution"],
        )
        origin_cam = cam.world_to_camera(np.array([0.0, 0.0, 0.0]))
        # Camera should be some distance from origin, so Z should be positive and large
        assert origin_cam[2] > 0, f"Camera Z at world origin is {origin_cam[2]}"


# ---------------------------------------------------------------------------
# Test 4: Full pipeline projection with real data
# ---------------------------------------------------------------------------

class TestFullPipelineProjection:
    @pytest.fixture(autouse=True)
    def _check_data(self):
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        if not os.path.isdir(seq_dir):
            pytest.skip("CMU Panoptic data not available")

    def test_gt_projects_within_image(self):
        """GT skeleton projected to 2D should be mostly within image bounds."""
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        cams = load_calibration(seq_dir)
        cal = cams["00_00"]
        cam = Camera.from_panoptic_calibration(
            cal["K"], cal["R"], cal["t"], cal["resolution"],
        )

        gt = load_ground_truth_sequence(seq_dir, [0], person_idx=0)
        if gt[0] is None:
            pytest.skip("GT not available")

        gt_cam = cam.world_to_camera(gt[0])  # (16, 3) in cm
        proj_2d = cam.camera_to_image(gt_cam)  # (16, 2) in pixels
        assert proj_2d.shape == (NUM_JOINTS, 2)

        # All Z values should be positive (in front of camera)
        assert np.all(gt_cam[:, 2] > 0), "Some joints behind camera"

        # Most joints should be within image bounds
        h, w = cam.image_size
        in_frame = cam.is_in_frame(proj_2d)
        n_in = int(in_frame.sum())
        assert n_in >= 10, f"Only {n_in}/16 joints in frame"

    def test_projected_joints_reasonable(self):
        """Projected joints should not be at origin or wildly off-screen."""
        seq_dir = get_sequence_dir(DEFAULT_DATA_ROOT, "171204_pose1_sample")
        cams = load_calibration(seq_dir)
        cal = cams["00_00"]
        cam = Camera.from_panoptic_calibration(
            cal["K"], cal["R"], cal["t"], cal["resolution"],
        )

        gt = load_ground_truth_sequence(seq_dir, [0], person_idx=0)
        if gt[0] is None:
            pytest.skip("GT not available")

        gt_cam = cam.world_to_camera(gt[0])
        proj_2d = cam.camera_to_image(gt_cam)

        # No joint should project exactly to (0, 0)
        norms = np.linalg.norm(proj_2d, axis=-1)
        assert np.all(norms > 1.0), "Some joints projected to origin"

        # No joint should be more than 5x image size away
        h, w = cam.image_size
        assert np.all(np.abs(proj_2d[:, 0]) < 5 * w)
        assert np.all(np.abs(proj_2d[:, 1]) < 5 * h)


# ---------------------------------------------------------------------------
# Test 5: is_in_frame consistency
# ---------------------------------------------------------------------------

class TestIsInFrame:
    def test_just_inside(self):
        cam = _make_panoptic_camera()
        h, w = cam.image_size
        pts = np.array([
            [0.0, 0.0],           # top-left corner -- in
            [w - 0.1, h - 0.1],   # just inside bottom-right -- in
            [w / 2, h / 2],        # center -- in
        ])
        mask = cam.is_in_frame(pts)
        assert np.all(mask)

    def test_just_outside(self):
        cam = _make_panoptic_camera()
        h, w = cam.image_size
        pts = np.array([
            [-0.1, 0.0],          # just left of frame
            [0.0, -0.1],          # just above frame
            [float(w), 0.0],      # at right edge (>= w)
            [0.0, float(h)],      # at bottom edge (>= h)
        ])
        mask = cam.is_in_frame(pts)
        assert not np.any(mask)


# ---------------------------------------------------------------------------
# Test 6: Synthetic heatmap peak values
# ---------------------------------------------------------------------------

class TestSyntheticHeatmapPeaks:
    def test_peak_near_one(self):
        """Heatmap peak at the detection point should be close to 1.0."""
        target_2d = np.array([[[960.0, 540.0]]])  # center of 1920x1080
        hm, affine = generate_synthetic_heatmaps(
            target_2d, image_size=(1080, 1920), heatmap_size=64, sigma=50.0,
        )
        assert hm.shape == (1, 1, 64, 64)
        assert hm[0, 0].max() > 0.95

    def test_multi_joint_peaks(self):
        """Each joint's heatmap should peak near 1.0 when target is in frame."""
        n_joints = 16
        target_2d = np.random.rand(1, n_joints, 2)
        target_2d[0, :, 0] *= 1920  # x in [0, 1920]
        target_2d[0, :, 1] *= 1080  # y in [0, 1080]
        hm, affine = generate_synthetic_heatmaps(
            target_2d, image_size=(1080, 1920), heatmap_size=64, sigma=50.0,
        )
        for j in range(n_joints):
            assert hm[0, j].max() > 0.5, f"Joint {j} peak too low: {hm[0, j].max()}"

    def test_non_square_circular_in_pixel_space(self):
        """On non-square image, Gaussian should be circular in pixel space.

        At equal pixel distances from center, heatmap values should be equal.
        """
        target_2d = np.array([[[960.0, 540.0]]])  # center
        hm, affine = generate_synthetic_heatmaps(
            target_2d, image_size=(1080, 1920), heatmap_size=64, sigma=50.0,
        )
        # The heatmap center is at approximately (32, 32)
        # 5 heatmap pixels right = 5 * (1920/64) = 150 px
        # 5 heatmap pixels down = 5 * (1080/64) = 84.375 px
        # So equal heatmap distance != equal pixel distance.
        # val_right should be < val_down (farther in pixel space)
        center_y, center_x = 32, 32
        val_right = float(hm[0, 0, center_y, center_x + 5])
        val_down = float(hm[0, 0, center_y + 5, center_x])
        assert val_right < val_down, (
            f"Expected val_right ({val_right:.4f}) < val_down ({val_down:.4f}) "
            "for non-square image (wider image means more pixels per heatmap unit in X)"
        )


# ---------------------------------------------------------------------------
# Test 7: Affine transform correctness
# ---------------------------------------------------------------------------

class TestAffineTransform:
    def test_affine_maps_heatmap_to_pixel(self):
        """The affine should map heatmap coords to pixel coords correctly."""
        _, affine = generate_synthetic_heatmaps(
            np.zeros((1, 1, 2)),
            image_size=(1080, 1920),
            heatmap_size=64,
            sigma=50.0,
        )
        # affine maps [0, 63] to [0, image_size-1]
        # sx = 1920 / 64 = 30.0, sy = 1080 / 64 = 16.875
        sx = affine[0, 0]
        sy = affine[1, 1]
        np.testing.assert_allclose(sx, 1920.0 / 64, atol=1e-4)
        np.testing.assert_allclose(sy, 1080.0 / 64, atol=1e-4)
        # tx, ty should be 0 (no offset)
        np.testing.assert_allclose(affine[0, 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(affine[1, 2], 0.0, atol=1e-6)

    def test_project_then_sample_heatmap(self):
        """Project a 3D point to pixels, find heatmap coords, sample -- should get high value."""
        cam = _make_panoptic_camera()
        # A point at (0, 0, 3) projects to image center
        pt_cam = np.array([[0.0, 0.0, 3.0]])
        proj_2d = cam.camera_to_image(pt_cam)  # should be (960, 540)

        target_2d = proj_2d.reshape(1, 1, 2)
        hm, affine = generate_synthetic_heatmaps(
            target_2d, image_size=cam.image_size, heatmap_size=64, sigma=50.0,
        )

        # Convert pixel coords to heatmap coords via inverse affine
        sx, sy = affine[0, 0], affine[1, 1]
        hm_x = proj_2d[0, 0] / sx
        hm_y = proj_2d[0, 1] / sy

        # Sample heatmap at those coords (nearest neighbor)
        hm_xi = int(round(hm_x))
        hm_yi = int(round(hm_y))
        hm_xi = np.clip(hm_xi, 0, 63)
        hm_yi = np.clip(hm_yi, 0, 63)
        val = hm[0, 0, hm_yi, hm_xi]
        assert val > 0.9, f"Heatmap value at projected point is {val:.4f}, expected > 0.9"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
