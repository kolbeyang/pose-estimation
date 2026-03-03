"""Evaluation metrics: MPJPE and per-joint error comparison."""

import numpy as np

from skeleton import NUM_JOINTS, JOINT_NAMES


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

    return results
