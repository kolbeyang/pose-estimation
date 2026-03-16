"""Evaluation metrics: MPJPE, P-MPJPE, and per-joint error comparison."""

from typing import Any

import numpy as np

from skeleton import NUM_JOINTS, JOINT_NAMES, EVAL_JOINTS, NUM_EVAL_JOINTS, EVAL_JOINT_NAMES

# Eval joints excluding ankles (joints 3=RAnkle, 6=LAnkle)
EVAL_JOINTS_NO_ANKLES: list[int] = [j for j in EVAL_JOINTS if j not in (3, 6)]
NUM_EVAL_JOINTS_NO_ANKLES: int = len(EVAL_JOINTS_NO_ANKLES)
EVAL_JOINT_NAMES_NO_ANKLES: list[str] = [JOINT_NAMES[j] for j in EVAL_JOINTS_NO_ANKLES]


def mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean Per-Joint Position Error.

    Args:
        predicted: (J, 3) or (F, J, 3) predicted positions.
        target: (J, 3) or (F, J, 3) ground truth positions.

    Returns:
        Scalar MPJPE in same units as inputs.
    """
    return float(np.mean(np.linalg.norm(predicted - target, axis=-1)))


def mpjpe_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-joint position error.

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        (J,) mean error per joint.
    """
    return np.mean(np.linalg.norm(predicted - target, axis=-1), axis=0)


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


def procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rigid alignment (Procrustes) of predicted to target.

    Finds optimal scale, rotation, translation that minimises ||s*R*P + t - T||.

    Args:
        predicted: (J, 3) single frame.
        target: (J, 3) single frame.

    Returns:
        (J, 3) aligned prediction.
    """
    mu_p: np.ndarray = predicted.mean(axis=0)
    mu_t: np.ndarray = target.mean(axis=0)
    p_c: np.ndarray = predicted - mu_p
    t_c: np.ndarray = target - mu_t

    H: np.ndarray = p_c.T @ t_c
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    U, S, Vt = np.linalg.svd(H)
    d: float = float(np.linalg.det(Vt.T @ U.T))
    D: np.ndarray = np.diag([1, 1, d])
    R: np.ndarray = Vt.T @ D @ U.T

    var_p: float = float(np.sum(p_c ** 2))
    if var_p < 1e-10:
        return np.tile(mu_t, (predicted.shape[0], 1))

    scale: float = float(np.trace(np.diag(S) @ D) / var_p)
    aligned: np.ndarray = scale * (p_c @ R.T) + mu_t
    return aligned


def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Protocol-2 MPJPE (after per-frame Procrustes alignment).

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        Scalar P-MPJPE.
    """
    errors: list[float] = []
    for i in range(predicted.shape[0]):
        aligned: np.ndarray = procrustes_align(predicted[i], target[i])
        errors.append(float(np.mean(np.linalg.norm(aligned - target[i], axis=-1))))
    return float(np.mean(errors))


def compute_comparison(
    detector_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
) -> dict[str, Any]:
    """Compare detector predictions vs ground truth.

    Phase 1 version: only detector vs GT (no optimized).
    Uses root-relative comparison on 12 eval joints.

    All inputs should be in the same coordinate system (camera coords, meters).

    Args:
        detector_3d: List of (17, 3) detector predictions.
        gt_3d: List of (17, 3) ground truth or None for missing frames.

    Returns:
        Dict with metrics including det_mpjpe, det_p_mpjpe, etc.
    """
    n: int = len(detector_3d)
    results: dict[str, Any] = {"n_frames": n, "n_frames_with_gt": 0}

    # Frames that have ground truth
    gt_indices: list[int] = [i for i in range(n) if gt_3d[i] is not None]
    results["n_frames_with_gt"] = len(gt_indices)

    if not gt_indices:
        return results

    # Stack only frames with GT
    det_arr: np.ndarray = np.array([detector_3d[i] for i in gt_indices])
    gt_arr: np.ndarray = np.array([gt_3d[i] for i in gt_indices])

    # Root-relative: subtract joint 0 (hip) first, then slice to eval joints
    det_rr: np.ndarray = root_relative(det_arr)
    gt_rr: np.ndarray = root_relative(gt_arr)

    # Slice to 12 eval joints
    ej: list[int] = EVAL_JOINTS
    det_eval: np.ndarray = det_rr[:, ej, :]
    gt_eval: np.ndarray = gt_rr[:, ej, :]

    results["det_mpjpe"] = mpjpe(det_eval, gt_eval)
    results["det_p_mpjpe"] = p_mpjpe(det_eval, gt_eval)

    # Per-joint errors (12 eval joints)
    results["det_per_joint"] = mpjpe_per_joint(det_eval, gt_eval).tolist()

    # Per-frame MPJPE (over eval joints only)
    results["det_per_frame_mpjpe"] = [
        float(np.mean(np.linalg.norm(det_eval[i] - gt_eval[i], axis=-1)))
        for i in range(len(gt_indices))
    ]

    # Per-joint P-MPJPE
    det_p_per_joint: np.ndarray = np.zeros(NUM_EVAL_JOINTS)
    for i in range(len(gt_indices)):
        det_aligned: np.ndarray = procrustes_align(det_eval[i], gt_eval[i])
        det_p_per_joint += np.linalg.norm(det_aligned - gt_eval[i], axis=-1)
    det_p_per_joint /= len(gt_indices)
    results["det_p_per_joint"] = det_p_per_joint.tolist()

    # No-ankles MPJPE (diagnostic: how much error comes from ankles)
    ej_na: list[int] = EVAL_JOINTS_NO_ANKLES
    det_eval_na: np.ndarray = det_rr[:, ej_na, :]
    gt_eval_na: np.ndarray = gt_rr[:, ej_na, :]
    results["det_mpjpe_no_ankles"] = mpjpe(det_eval_na, gt_eval_na)
    results["det_p_mpjpe_no_ankles"] = p_mpjpe(det_eval_na, gt_eval_na)

    return results


def compute_comparison_with_optimization(
    detector_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
) -> dict[str, Any]:
    """Compare detector baseline, optimized, and ground truth.

    Same as compute_comparison but also computes opt_mpjpe, opt_p_mpjpe,
    opt_per_joint, opt_per_frame_mpjpe, opt_p_per_joint.

    All positions should be in camera-space meters.
    Uses root-relative comparison on 12 eval joints.

    Args:
        detector_3d: List of (17, 3) detector predictions.
        optimized_3d: List of (17, 3) optimized predictions.
        gt_3d: List of (17, 3) ground truth or None for missing frames.

    Returns:
        Dict with metrics including det_*, opt_*, and improvement.
    """
    # Get detector metrics first
    results: dict[str, Any] = compute_comparison(detector_3d, gt_3d)

    n: int = len(detector_3d)
    gt_indices: list[int] = [i for i in range(n) if gt_3d[i] is not None]

    if not gt_indices:
        return results

    # Stack optimized frames with GT
    opt_arr: np.ndarray = np.array([optimized_3d[i] for i in gt_indices])
    gt_arr: np.ndarray = np.array([gt_3d[i] for i in gt_indices])

    # Root-relative, then slice to eval joints
    opt_rr: np.ndarray = root_relative(opt_arr)
    gt_rr: np.ndarray = root_relative(gt_arr)

    ej: list[int] = EVAL_JOINTS
    opt_eval: np.ndarray = opt_rr[:, ej, :]
    gt_eval: np.ndarray = gt_rr[:, ej, :]

    results["opt_mpjpe"] = mpjpe(opt_eval, gt_eval)
    results["opt_p_mpjpe"] = p_mpjpe(opt_eval, gt_eval)

    # Per-joint errors (12 eval joints)
    results["opt_per_joint"] = mpjpe_per_joint(opt_eval, gt_eval).tolist()

    # Per-frame MPJPE
    results["opt_per_frame_mpjpe"] = [
        float(np.mean(np.linalg.norm(opt_eval[i] - gt_eval[i], axis=-1)))
        for i in range(len(gt_indices))
    ]

    # Per-joint P-MPJPE
    opt_p_per_joint: np.ndarray = np.zeros(NUM_EVAL_JOINTS)
    for i in range(len(gt_indices)):
        opt_aligned: np.ndarray = procrustes_align(opt_eval[i], gt_eval[i])
        opt_p_per_joint += np.linalg.norm(opt_aligned - gt_eval[i], axis=-1)
    opt_p_per_joint /= len(gt_indices)
    results["opt_p_per_joint"] = opt_p_per_joint.tolist()

    # No-ankles MPJPE for optimized
    ej_na: list[int] = EVAL_JOINTS_NO_ANKLES
    opt_eval_na: np.ndarray = opt_rr[:, ej_na, :]
    gt_eval_na: np.ndarray = gt_rr[:, ej_na, :]
    results["opt_mpjpe_no_ankles"] = mpjpe(opt_eval_na, gt_eval_na)
    results["opt_p_mpjpe_no_ankles"] = p_mpjpe(opt_eval_na, gt_eval_na)

    # Improvement (positive = optimized is better)
    if "det_mpjpe" in results:
        results["improvement"] = results["det_mpjpe"] - results["opt_mpjpe"]

    return results
