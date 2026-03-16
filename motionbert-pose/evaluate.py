"""Evaluation metrics: MPJPE, P-MPJPE, MPJVE, and per-joint error comparison."""

from __future__ import annotations

from typing import Any

import numpy as np

from camera import Camera
from skeleton import NUM_JOINTS, JOINT_NAMES, EVAL_JOINTS, NUM_EVAL_JOINTS, EVAL_JOINT_NAMES, PARENTS

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


def mpjve(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean Per-Joint Velocity Error.

    Args:
        predicted: (F, J, 3) root-relative positions.
        target: (F, J, 3) root-relative positions.

    Returns:
        Scalar MPJVE.
    """
    pred_vel: np.ndarray = np.diff(predicted, axis=0)
    tgt_vel: np.ndarray = np.diff(target, axis=0)
    return float(np.mean(np.linalg.norm(pred_vel - tgt_vel, axis=-1)))


def mpjve_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-joint velocity error.

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        (J,) mean velocity error per joint.
    """
    pred_vel: np.ndarray = np.diff(predicted, axis=0)
    tgt_vel: np.ndarray = np.diff(target, axis=0)
    return np.mean(np.linalg.norm(pred_vel - tgt_vel, axis=-1), axis=0)


def mpjve_per_frame(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    """Per-frame velocity error (F-1 values).

    Args:
        predicted: (F, J, 3).
        target: (F, J, 3).

    Returns:
        List of length F-1, mean joint velocity error per frame transition.
    """
    pred_vel: np.ndarray = np.diff(predicted, axis=0)
    tgt_vel: np.ndarray = np.diff(target, axis=0)
    return [float(np.mean(np.linalg.norm(pred_vel[i] - tgt_vel[i], axis=-1)))
            for i in range(pred_vel.shape[0])]


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


def reprojection_error_vs_detections(
    positions_3d: list[np.ndarray],
    detections_2d: list[np.ndarray],
    visibility: list[np.ndarray],
    camera: "Camera",
    visibility_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute 2D reprojection error against 2D SH detections.

    Projects 3D predictions to 2D via the camera model, then measures
    pixel distance to the 2D Stacked Hourglass keypoints (NOT GT projections).

    Args:
        positions_3d: Per-frame (17, 3) camera-space positions.
        detections_2d: Per-frame (17, 2) SH 2D keypoints in pixels.
        visibility: Per-frame (17,) visibility scores.
        camera: Camera for 3D->2D projection.
        visibility_threshold: Only count joints above this threshold.

    Returns:
        Dict with 'mean_px' (mean pixel error across visible joints and frames),
        'per_frame_px' (list of per-frame mean pixel errors).
    """
    per_frame_errors: list[float] = []
    for i in range(len(positions_3d)):
        proj_2d = camera.world_to_image(positions_3d[i])  # (17, 2)
        diffs = np.linalg.norm(proj_2d - detections_2d[i], axis=-1)  # (17,)
        mask = visibility[i] >= visibility_threshold
        if mask.sum() > 0:
            per_frame_errors.append(float(diffs[mask].mean()))
        else:
            per_frame_errors.append(0.0)
    return {
        "mean_px": float(np.mean(per_frame_errors)) if per_frame_errors else 0.0,
        "per_frame_px": per_frame_errors,
    }


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
    camera: Camera | None = None,
    detections_2d: list[np.ndarray] | None = None,
    visibility: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare detector baseline, optimized, and ground truth.

    Same as compute_comparison but also computes opt_mpjpe, opt_p_mpjpe,
    opt_per_joint, opt_per_frame_mpjpe, opt_p_per_joint, MPJVE, bone
    lengths, and optional 2D reprojection error.

    All positions should be in camera-space meters.
    Uses root-relative comparison on 12 eval joints.

    Args:
        detector_3d: List of (17, 3) detector predictions.
        optimized_3d: List of (17, 3) optimized predictions.
        gt_3d: List of (17, 3) ground truth or None for missing frames.
        camera: Optional camera for 2D reprojection error.

    Returns:
        Dict with metrics including det_*, opt_*, and improvement.
    """
    # Get detector metrics first
    results: dict[str, Any] = compute_comparison(detector_3d, gt_3d)

    n: int = len(detector_3d)
    gt_indices: list[int] = [i for i in range(n) if gt_3d[i] is not None]

    if not gt_indices:
        return results

    # Stack arrays for frames with GT
    det_arr: np.ndarray = np.array([detector_3d[i] for i in gt_indices])
    opt_arr: np.ndarray = np.array([optimized_3d[i] for i in gt_indices])
    gt_arr: np.ndarray = np.array([gt_3d[i] for i in gt_indices])

    # Root-relative, then slice to eval joints
    det_rr: np.ndarray = root_relative(det_arr)
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

    # --- Bone lengths from GT and Detector ---
    gt_bl: np.ndarray = np.zeros(NUM_JOINTS)
    det_bl: np.ndarray = np.zeros(NUM_JOINTS)
    for j in range(1, NUM_JOINTS):
        p: int = int(PARENTS[j])
        gt_bl[j] = float(np.mean([
            np.linalg.norm(gt_rr[f, j] - gt_rr[f, p])
            for f in range(len(gt_indices))
        ]))
        det_bl[j] = float(np.mean([
            np.linalg.norm(det_rr[f, j] - det_rr[f, p])
            for f in range(len(gt_indices))
        ]))
    results["gt_bone_lengths"] = gt_bl.tolist()
    results["det_bone_lengths"] = det_bl.tolist()

    # --- MPJVE (velocity error) -- need >= 2 GT frames ---
    det_eval: np.ndarray = det_rr[:, ej, :]
    if len(gt_indices) >= 2:
        results["det_mpjve"] = mpjve(det_eval, gt_eval)
        results["opt_mpjve"] = mpjve(opt_eval, gt_eval)
        results["det_mpjve_per_joint"] = mpjve_per_joint(det_eval, gt_eval).tolist()
        results["opt_mpjve_per_joint"] = mpjve_per_joint(opt_eval, gt_eval).tolist()
        results["det_per_frame_mpjve"] = mpjve_per_frame(det_eval, gt_eval)
        results["opt_per_frame_mpjve"] = mpjve_per_frame(opt_eval, gt_eval)

    # --- 2D reprojection error (requires camera) ---
    if camera is not None:
        det_2d_errors: list[float] = []
        opt_2d_errors: list[float] = []
        for i in range(len(gt_indices)):
            gt_2d: np.ndarray = camera.world_to_image(gt_arr[i])
            det_2d: np.ndarray = camera.world_to_image(det_arr[i])
            opt_2d: np.ndarray = camera.world_to_image(opt_arr[i])
            det_2d_errors.append(float(np.mean(np.linalg.norm(det_2d - gt_2d, axis=-1))))
            opt_2d_errors.append(float(np.mean(np.linalg.norm(opt_2d - gt_2d, axis=-1))))
        results["det_per_frame_2d_mpjpe"] = det_2d_errors
        results["opt_per_frame_2d_mpjpe"] = opt_2d_errors
        results["det_2d_mpjpe"] = float(np.mean(det_2d_errors))
        results["opt_2d_mpjpe"] = float(np.mean(opt_2d_errors))

    # --- 2D reprojection error vs 2D DETECTIONS (not GT) ---
    if camera is not None and detections_2d is not None and visibility is not None:
        det_vs_det = reprojection_error_vs_detections(
            detector_3d, detections_2d, visibility, camera,
        )
        opt_vs_det = reprojection_error_vs_detections(
            optimized_3d, detections_2d, visibility, camera,
        )
        results["det_2d_det_mpjpe_px"] = det_vs_det["mean_px"]
        results["opt_2d_det_mpjpe_px"] = opt_vs_det["mean_px"]
        results["det_2d_det_per_frame_px"] = det_vs_det["per_frame_px"]
        results["opt_2d_det_per_frame_px"] = opt_vs_det["per_frame_px"]

    return results
