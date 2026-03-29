"""Evaluation metrics: MPJPE, P-MPJPE, SI-MPJPE, VW variants, velocity metrics.

All 9 metrics:
  1. MPJPE           - Mean Per-Joint Position Error
  2. P-MPJPE         - Procrustes-aligned MPJPE
  3. SI-MPJPE        - Scale-Independent MPJPE
  4. VW-MPJPE        - Visibility-Weighted MPJPE
  5. VW-SI-MPJPE     - Visibility-Weighted Scale-Independent MPJPE
  6. MPJVE           - Mean Per-Joint Velocity Error
  7. SI-MPJVE        - Scale-Independent MPJVE
  8. VW-MPJVE        - Visibility-Weighted MPJVE
  9. VW-SI-MPJVE     - Visibility-Weighted Scale-Independent MPJVE
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from camera import Camera
from skeleton import EVAL_JOINTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def root_relative(positions: np.ndarray) -> np.ndarray:
    """Make positions root-relative (subtract hip position per frame).

    Args:
        positions: (J, 3) or (F, J, 3).

    Returns:
        Same shape, hip-centered.
    """
    if positions.ndim == 2:
        return positions - positions[0:1]
    return positions - positions[:, 0:1, :]


def _velocities(positions: np.ndarray) -> np.ndarray:
    """Compute frame-to-frame velocity.

    Args:
        positions: (F, J, 3).

    Returns:
        (F-1, J, 3) velocity vectors.
    """
    return np.diff(positions, axis=0)


# ---------------------------------------------------------------------------
# Optimal scale
# ---------------------------------------------------------------------------

def optimal_scale(predicted: np.ndarray, target: np.ndarray) -> float:
    """Find optimal scale s minimizing MPJPE(s*pred, target).

    Single global scalar across all frames.

    Args:
        predicted: (F, J, 3) root-relative.
        target: (F, J, 3) root-relative.

    Returns:
        Optimal scale factor.
    """
    if float(np.sum(predicted * predicted)) < 1e-12:
        return 1.0

    def _mpjpe_at_scale(s: float) -> float:
        return float(np.mean(np.linalg.norm(s * predicted - target, axis=-1)))

    result = minimize_scalar(_mpjpe_at_scale, bounds=(0.5, 2.0), method="bounded")
    return float(result.x)


def _optimal_scale_weighted(
    predicted: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Find optimal scale s minimizing visibility-weighted MPJPE.

    Args:
        predicted: (F, J, 3) root-relative.
        target: (F, J, 3) root-relative.
        weights: (F, J) per-joint per-frame weights.

    Returns:
        Optimal scale factor.
    """
    if float(np.sum(predicted * predicted)) < 1e-12:
        return 1.0

    w_sum = float(weights.sum())
    if w_sum < 1e-12:
        return 1.0

    def _weighted_mpjpe(s: float) -> float:
        errors = np.linalg.norm(s * predicted - target, axis=-1)
        return float(np.sum(errors * weights) / w_sum)

    result = minimize_scalar(_weighted_mpjpe, bounds=(0.5, 2.0), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigid alignment (Procrustes) of predicted to target.

    Finds optimal scale, rotation, translation that minimises ||s*R*P + t - T||.

    Args:
        predicted: (J, 3) single frame.
        target: (J, 3) single frame.

    Returns:
        (J, 3) aligned prediction.
    """
    mu_p = predicted.mean(axis=0)
    mu_t = target.mean(axis=0)
    p_c = predicted - mu_p
    t_c = target - mu_t

    H = p_c.T @ t_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    var_p = np.sum(p_c ** 2)
    if var_p < 1e-10:
        return np.tile(mu_t, (predicted.shape[0], 1))

    scale = np.trace(np.diag(S) @ D) / var_p
    aligned = scale * (p_c @ R.T) + mu_t
    return aligned


# ---------------------------------------------------------------------------
# Visibility weights
# ---------------------------------------------------------------------------

def compute_visibility_weights(
    gt_3d_cam: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    """Compute per-joint per-frame visibility weights from frame boundary check.

    Projects GT 3D (camera space) to 2D, checks if within image bounds.
    Binary 0/1 weights.

    Args:
        gt_3d_cam: (F, K, 3) ground truth in camera coordinates.
        camera: Camera for projection and frame bounds.

    Returns:
        (F, K) binary visibility weights.
    """
    proj_2d = camera.camera_to_image(gt_3d_cam)  # (F, K, 2)
    return camera.is_in_frame(proj_2d).astype(np.float64)  # (F, K)


# ---------------------------------------------------------------------------
# Position metrics
# ---------------------------------------------------------------------------

def mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean Per-Joint Position Error.

    Args:
        predicted: (J, 3) or (F, J, 3).
        target: same shape.

    Returns:
        Scalar MPJPE.
    """
    return float(np.mean(np.linalg.norm(predicted - target, axis=-1)))


def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Protocol-2 MPJPE (after per-frame Procrustes alignment).

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        Scalar P-MPJPE.
    """
    errors = []
    for i in range(predicted.shape[0]):
        aligned = procrustes_align(predicted[i], target[i])
        errors.append(np.mean(np.linalg.norm(aligned - target[i], axis=-1)))
    return float(np.mean(errors))


def si_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Scale-Independent MPJPE.

    Single global scale across all frames.

    Args:
        predicted: (F, J, 3) root-relative.
        target: (F, J, 3) root-relative.

    Returns:
        Scalar SI-MPJPE.
    """
    s = optimal_scale(predicted, target)
    return mpjpe(s * predicted, target)


def vw_mpjpe(
    predicted: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Visibility-Weighted MPJPE.

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).
        weights: (F, J) visibility weights.

    Returns:
        Scalar VW-MPJPE.
    """
    errors = np.linalg.norm(predicted - target, axis=-1)  # (F, J)
    w_sum = float(weights.sum())
    if w_sum < 1e-12:
        return float(np.mean(errors))
    return float(np.sum(errors * weights) / w_sum)


def vw_si_mpjpe(
    predicted: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Visibility-Weighted Scale-Independent MPJPE.

    This is the primary evaluation metric.

    Args:
        predicted: (F, J, 3) root-relative.
        target: (F, J, 3) root-relative.
        weights: (F, J) visibility weights.

    Returns:
        Scalar VW-SI-MPJPE.
    """
    s = _optimal_scale_weighted(predicted, target, weights)
    scaled = s * predicted
    errors = np.linalg.norm(scaled - target, axis=-1)
    w_sum = float(weights.sum())
    if w_sum < 1e-12:
        return float(np.mean(errors))
    return float(np.sum(errors * weights) / w_sum)


# ---------------------------------------------------------------------------
# Velocity metrics
# ---------------------------------------------------------------------------

def mpjve(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean Per-Joint Velocity Error.

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        Scalar MPJVE.
    """
    pred_vel = _velocities(predicted)
    tgt_vel = _velocities(target)
    return float(np.mean(np.linalg.norm(pred_vel - tgt_vel, axis=-1)))


def si_mpjve(predicted: np.ndarray, target: np.ndarray) -> float:
    """Scale-Independent MPJVE.

    Finds optimal scale on positions, applies to velocities.

    Args:
        predicted: (F, J, 3) root-relative.
        target: (F, J, 3) root-relative.

    Returns:
        Scalar SI-MPJVE.
    """
    s = optimal_scale(predicted, target)
    return mpjve(s * predicted, target)


def vw_mpjve(
    predicted: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Visibility-Weighted MPJVE.

    Uses min visibility of consecutive frames for each velocity pair.

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).
        weights: (F, J) visibility weights.

    Returns:
        Scalar VW-MPJVE.
    """
    pred_vel = _velocities(predicted)
    tgt_vel = _velocities(target)
    errors = np.linalg.norm(pred_vel - tgt_vel, axis=-1)  # (F-1, J)
    # Velocity weights: both frames must be visible
    vel_weights = np.minimum(weights[:-1], weights[1:])  # (F-1, J)
    w_sum = float(vel_weights.sum())
    if w_sum < 1e-12:
        return float(np.mean(errors))
    return float(np.sum(errors * vel_weights) / w_sum)


def vw_si_mpjve(
    predicted: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Visibility-Weighted Scale-Independent MPJVE.

    Args:
        predicted: (F, J, 3) root-relative.
        target: (F, J, 3) root-relative.
        weights: (F, J) visibility weights.

    Returns:
        Scalar VW-SI-MPJVE.
    """
    s = _optimal_scale_weighted(predicted, target, weights)
    return vw_mpjve(s * predicted, target, weights)


# ---------------------------------------------------------------------------
# Per-joint helpers
# ---------------------------------------------------------------------------

def mpjpe_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-joint position error.

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        (J,) mean error per joint.
    """
    return np.mean(np.linalg.norm(predicted - target, axis=-1), axis=0)


def mpjve_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-joint velocity error.

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        (J,) mean velocity error per joint.
    """
    pred_vel = _velocities(predicted)
    tgt_vel = _velocities(target)
    return np.mean(np.linalg.norm(pred_vel - tgt_vel, axis=-1), axis=0)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    camera: Camera,
    eval_joints: list[int] | None = None,
) -> dict[str, float]:
    """Compute all 9 evaluation metrics.

    Both predicted and ground_truth should be in camera space.

    Args:
        predicted: (N, K, 3) predicted positions.
        ground_truth: (N, K, 3) ground truth positions.
        camera: Camera for VW computation.
        eval_joints: Joint indices to evaluate. Defaults to EVAL_JOINTS.

    Returns:
        Dict with all 9 metrics.
    """
    if eval_joints is None:
        eval_joints = EVAL_JOINTS

    # Root-relative, then slice to eval joints
    pred_rr = root_relative(predicted)[:, eval_joints, :]
    gt_rr = root_relative(ground_truth)[:, eval_joints, :]

    # Visibility weights (on all joints, then slice)
    vis = compute_visibility_weights(ground_truth, camera)[:, eval_joints]

    results: dict[str, float] = {}

    # Position metrics
    results["mpjpe"] = mpjpe(pred_rr, gt_rr)
    results["p_mpjpe"] = p_mpjpe(pred_rr, gt_rr)
    results["si_mpjpe"] = si_mpjpe(pred_rr, gt_rr)
    results["vw_mpjpe"] = vw_mpjpe(pred_rr, gt_rr, vis)
    results["vw_si_mpjpe"] = vw_si_mpjpe(pred_rr, gt_rr, vis)

    # Velocity metrics (need >= 2 frames)
    if predicted.shape[0] >= 2:
        results["mpjve"] = mpjve(pred_rr, gt_rr)
        results["si_mpjve"] = si_mpjve(pred_rr, gt_rr)
        results["vw_mpjve"] = vw_mpjve(pred_rr, gt_rr, vis)
        results["vw_si_mpjve"] = vw_si_mpjve(pred_rr, gt_rr, vis)
    else:
        results["mpjve"] = 0.0
        results["si_mpjve"] = 0.0
        results["vw_mpjve"] = 0.0
        results["vw_si_mpjve"] = 0.0

    return results
