"""Evaluation metrics: MPJPE and per-joint error comparison."""

from __future__ import annotations

import numpy as np

from model.camera import Camera
from skeleton import NUM_JOINTS, JOINT_NAMES, PARENTS


def mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean Per-Joint Position Error (mm or same unit as inputs).

    Args:
        predicted: (J, 3) or (F, J, 3) predicted positions.
        target: (J, 3) or (F, J, 3) ground truth positions.

    Returns:
        Scalar MPJPE.
    """
    return float(np.mean(np.linalg.norm(predicted - target, axis=-1)))


def mpjpe_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-joint position error.

    Args:
        predicted: (F, J, 3)
        target: (F, J, 3)

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
    pred_vel = np.diff(predicted, axis=0)
    tgt_vel = np.diff(target, axis=0)
    return float(np.mean(np.linalg.norm(pred_vel - tgt_vel, axis=-1)))


def mpjve_per_joint(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-joint velocity error.

    Args:
        predicted: (F, J, 3)
        target: (F, J, 3)

    Returns:
        (J,) mean velocity error per joint.
    """
    pred_vel = np.diff(predicted, axis=0)
    tgt_vel = np.diff(target, axis=0)
    return np.mean(np.linalg.norm(pred_vel - tgt_vel, axis=-1), axis=0)


def mpjve_per_frame(predicted: np.ndarray, target: np.ndarray) -> list[float]:
    """Per-frame velocity error (F-1 values).

    Args:
        predicted: (F, J, 3)
        target: (F, J, 3)

    Returns:
        List of length F-1, mean joint velocity error per frame transition.
    """
    pred_vel = np.diff(predicted, axis=0)
    tgt_vel = np.diff(target, axis=0)
    return [float(np.mean(np.linalg.norm(pred_vel[i] - tgt_vel[i], axis=-1)))
            for i in range(pred_vel.shape[0])]


def root_relative(positions: np.ndarray) -> np.ndarray:
    """Make positions root-relative (subtract hip position per frame).

    Args:
        positions: (J, 3) or (F, J, 3)

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


def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Protocol-2 MPJPE (after per-frame Procrustes alignment).

    Args:
        predicted: (F, J, 3)
        target: (F, J, 3)

    Returns:
        Scalar P-MPJPE.
    """
    errors = []
    for i in range(predicted.shape[0]):
        aligned = procrustes_align(predicted[i], target[i])
        errors.append(np.mean(np.linalg.norm(aligned - target[i], axis=-1)))
    return float(np.mean(errors))


def compute_comparison(
    mediapipe_3d: list[np.ndarray],
    optimized_3d: list[np.ndarray],
    gt_3d: list[np.ndarray | None],
    camera: Camera | None = None,
) -> dict:
    """Compare MediaPipe vs Optimised vs Ground Truth.

    All inputs should be in the same coordinate system (camera coords, meters).

    Returns:
        Dict with metrics.
    """
    n = len(mediapipe_3d)
    results: dict = {"n_frames": n, "n_frames_with_gt": 0}

    # Frames that have ground truth
    gt_indices = [i for i in range(n) if gt_3d[i] is not None]
    results["n_frames_with_gt"] = len(gt_indices)

    if not gt_indices:
        # No GT available — only report self-consistency
        mp_arr = np.array(mediapipe_3d)    # (F, 17, 3)
        opt_arr = np.array(optimized_3d)   # (F, 17, 3)
        mp_rr = root_relative(mp_arr)
        opt_rr = root_relative(opt_arr)
        results["mp_vs_opt_mpjpe"] = mpjpe(mp_rr, opt_rr)
        return results

    # Stack only frames with GT
    mp_arr = np.array([mediapipe_3d[i] for i in gt_indices])
    opt_arr = np.array([optimized_3d[i] for i in gt_indices])
    gt_arr = np.array([gt_3d[i] for i in gt_indices])

    # Root-relative comparison
    mp_rr = root_relative(mp_arr)
    opt_rr = root_relative(opt_arr)
    gt_rr = root_relative(gt_arr)

    results["mp_mpjpe"] = mpjpe(mp_rr, gt_rr)
    results["opt_mpjpe"] = mpjpe(opt_rr, gt_rr)
    results["mp_p_mpjpe"] = p_mpjpe(mp_rr, gt_rr)
    results["opt_p_mpjpe"] = p_mpjpe(opt_rr, gt_rr)

    # Per-joint errors
    results["mp_per_joint"] = mpjpe_per_joint(mp_rr, gt_rr).tolist()
    results["opt_per_joint"] = mpjpe_per_joint(opt_rr, gt_rr).tolist()

    # Per-frame MPJPE
    results["mp_per_frame_mpjpe"] = [
        float(np.mean(np.linalg.norm(mp_rr[i] - gt_rr[i], axis=-1)))
        for i in range(len(gt_indices))
    ]
    results["opt_per_frame_mpjpe"] = [
        float(np.mean(np.linalg.norm(opt_rr[i] - gt_rr[i], axis=-1)))
        for i in range(len(gt_indices))
    ]

    # Per-joint P-MPJPE (Procrustes-aligned per-joint error)
    mp_p_per_joint = np.zeros(NUM_JOINTS)
    opt_p_per_joint = np.zeros(NUM_JOINTS)
    for i in range(len(gt_indices)):
        mp_aligned = procrustes_align(mp_rr[i], gt_rr[i])
        opt_aligned = procrustes_align(opt_rr[i], gt_rr[i])
        mp_p_per_joint += np.linalg.norm(mp_aligned - gt_rr[i], axis=-1)
        opt_p_per_joint += np.linalg.norm(opt_aligned - gt_rr[i], axis=-1)
    mp_p_per_joint /= len(gt_indices)
    opt_p_per_joint /= len(gt_indices)
    results["mp_p_per_joint"] = mp_p_per_joint.tolist()
    results["opt_p_per_joint"] = opt_p_per_joint.tolist()

    # Mean bone lengths from GT and MediaPipe
    gt_bl = np.zeros(NUM_JOINTS)
    mp_bl = np.zeros(NUM_JOINTS)
    for j in range(1, NUM_JOINTS):
        p = PARENTS[j]
        gt_bl[j] = np.mean([np.linalg.norm(gt_rr[f, j] - gt_rr[f, p]) for f in range(len(gt_indices))])
        mp_bl[j] = np.mean([np.linalg.norm(mp_rr[f, j] - mp_rr[f, p]) for f in range(len(gt_indices))])
    results["gt_bone_lengths"] = gt_bl.tolist()
    results["mp_bone_lengths"] = mp_bl.tolist()

    # MPJVE (velocity error) — need at least 2 GT frames
    if len(gt_indices) >= 2:
        results["mp_mpjve"] = mpjve(mp_rr, gt_rr)
        results["opt_mpjve"] = mpjve(opt_rr, gt_rr)
        results["mp_mpjve_per_joint"] = mpjve_per_joint(mp_rr, gt_rr).tolist()
        results["opt_mpjve_per_joint"] = mpjve_per_joint(opt_rr, gt_rr).tolist()
        results["mp_per_frame_mpjve"] = mpjve_per_frame(mp_rr, gt_rr)
        results["opt_per_frame_mpjve"] = mpjve_per_frame(opt_rr, gt_rr)

    # 2D reprojection error in pixels (requires camera)
    if camera is not None:
        mp_2d_errors = []
        opt_2d_errors = []
        for i in range(len(gt_indices)):
            gt_2d = camera.world_to_image(gt_arr[i])    # (J, 2)
            mp_2d = camera.world_to_image(mp_arr[i])     # (J, 2)
            opt_2d = camera.world_to_image(opt_arr[i])   # (J, 2)
            mp_2d_errors.append(float(np.mean(np.linalg.norm(mp_2d - gt_2d, axis=-1))))
            opt_2d_errors.append(float(np.mean(np.linalg.norm(opt_2d - gt_2d, axis=-1))))
        results["mp_per_frame_2d_mpjpe"] = mp_2d_errors
        results["opt_per_frame_2d_mpjpe"] = opt_2d_errors
        results["mp_2d_mpjpe"] = float(np.mean(mp_2d_errors))
        results["opt_2d_mpjpe"] = float(np.mean(opt_2d_errors))

    return results
