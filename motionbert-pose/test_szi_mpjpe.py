"""Unit tests and 3D visualization for Scale-Z-Invariant MPJPE.

Run with: uv run python test_szi_mpjpe.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from evaluate import optimal_scale, szi_mpjpe, szi_mpjpe_per_joint, mpjpe, p_mpjpe
from skeleton import BONES, JOINT_NAMES, EVAL_JOINTS, NUM_JOINTS, DEFAULT_BONE_LENGTHS, PARENTS, REST_DIRECTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_t_pose() -> np.ndarray:
    """Build a single-frame skeleton from default bone lengths and rest directions.

    Returns:
        (16, 3) joint positions.
    """
    positions: np.ndarray = np.zeros((NUM_JOINTS, 3))
    for j in range(1, NUM_JOINTS):
        p: int = int(PARENTS[j])
        direction: np.ndarray = REST_DIRECTIONS[j].copy()
        norm: float = float(np.linalg.norm(direction))
        if norm > 1e-8:
            direction /= norm
        positions[j] = positions[p] + direction * DEFAULT_BONE_LENGTHS[j]
    return positions


def _draw_skeleton(
    ax: plt.Axes,
    positions: np.ndarray,
    color: str,
    label: str,
    offset_x: float = 0.0,
) -> None:
    """Draw a skeleton on a 3D axes."""
    pos: np.ndarray = positions.copy()
    pos[:, 0] += offset_x
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, s=20, zorder=5)
    for parent, child in BONES:
        ax.plot(
            [pos[parent, 0], pos[child, 0]],
            [pos[parent, 1], pos[child, 1]],
            [pos[parent, 2], pos[child, 2]],
            c=color, linewidth=2,
        )
    # Invisible point for legend
    ax.plot([], [], [], c=color, linewidth=2, label=label)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

passed: int = 0
failed: int = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg: str = f"  FAIL: {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)


def run_tests() -> None:
    global passed, failed
    print("=" * 60)
    print("  SZI-MPJPE Unit Tests")
    print("=" * 60)

    rng: np.random.Generator = np.random.default_rng(42)

    # --- Test 1: Identity ---
    gt: np.ndarray = rng.standard_normal((10, 12, 3))
    s: float = optimal_scale(gt, gt)
    val: float
    val, s_ret = szi_mpjpe(gt, gt)
    check(np.isclose(s, 1.0, atol=1e-6), "Identity: scale == 1.0", f"got {s:.8f}")
    check(np.isclose(val, 0.0, atol=1e-6), "Identity: SZI-MPJPE == 0.0", f"got {val:.8f}")

    # --- Test 2: Known scale (pred = 2*gt) ---
    pred: np.ndarray = 2.0 * gt
    s = optimal_scale(pred, gt)
    val, _ = szi_mpjpe(pred, gt)
    check(np.isclose(s, 0.5, atol=1e-6), "Known scale 2x: s == 0.5", f"got {s:.8f}")
    check(np.isclose(val, 0.0, atol=1e-6), "Known scale 2x: SZI-MPJPE == 0.0", f"got {val:.8f}")

    # --- Test 3: Known scale + noise ---
    noise: np.ndarray = rng.standard_normal(gt.shape) * 0.01
    pred_noisy: np.ndarray = 2.0 * gt + noise
    s = optimal_scale(pred_noisy, gt)
    val, _ = szi_mpjpe(pred_noisy, gt)
    check(np.isclose(s, 0.5, atol=0.05), "Noisy 2x: s ~= 0.5", f"got {s:.6f}")
    check(val > 0 and val < 0.02, "Noisy 2x: SZI-MPJPE small but nonzero", f"got {val:.6f}")

    # --- Test 4: Asymmetric scale (pred = 0.5*gt) ---
    pred_half: np.ndarray = 0.5 * gt
    s = optimal_scale(pred_half, gt)
    val, _ = szi_mpjpe(pred_half, gt)
    check(np.isclose(s, 2.0, atol=1e-6), "Asymmetric 0.5x: s == 2.0", f"got {s:.8f}")
    check(np.isclose(val, 0.0, atol=1e-6), "Asymmetric 0.5x: SZI-MPJPE == 0.0", f"got {val:.8f}")

    # --- Test 5: SZI-MPJPE <= MPJPE ---
    for trial in range(5):
        pred_rand: np.ndarray = rng.standard_normal((8, 12, 3))
        gt_rand: np.ndarray = rng.standard_normal((8, 12, 3))
        szi_val, _ = szi_mpjpe(pred_rand, gt_rand)
        mpjpe_val: float = mpjpe(pred_rand, gt_rand)
        check(
            szi_val <= mpjpe_val + 1e-8,
            f"SZI-MPJPE <= MPJPE (trial {trial})",
            f"szi={szi_val:.6f} mpjpe={mpjpe_val:.6f}",
        )

    # --- Test 6: SZI-MPJPE >= P-MPJPE ---
    for trial in range(5):
        pred_rand = rng.standard_normal((8, 12, 3))
        gt_rand = rng.standard_normal((8, 12, 3))
        szi_val, _ = szi_mpjpe(pred_rand, gt_rand)
        p_mpjpe_val: float = p_mpjpe(pred_rand, gt_rand)
        check(
            szi_val >= p_mpjpe_val - 1e-6,
            f"SZI-MPJPE >= P-MPJPE (trial {trial})",
            f"szi={szi_val:.6f} p_mpjpe={p_mpjpe_val:.6f}",
        )

    print(f"\n  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 3D Visualization (synthetic)
# ---------------------------------------------------------------------------

def visualize_synthetic(output_dir: str) -> None:
    """Render before-scaling, after-scaling, and GT skeletons side by side."""
    print("\n  Generating synthetic visualization...")
    os.makedirs(output_dir, exist_ok=True)

    gt_skeleton: np.ndarray = _make_t_pose()  # (16, 3)
    # Make root-relative
    gt_rr: np.ndarray = gt_skeleton - gt_skeleton[0:1]

    # Simulate a multi-frame prediction at 1.5x scale
    gt_frames: np.ndarray = np.stack([gt_rr] * 5)  # (5, 16, 3)
    pred_frames: np.ndarray = 1.5 * gt_frames

    # Compute optimal scale
    s: float = optimal_scale(pred_frames, gt_frames)
    scaled_frames: np.ndarray = s * pred_frames
    print(f"    Optimal scale for 1.5x pred: {s:.6f} (expected ~0.667)")

    # Use eval joints for a single frame
    frame_idx: int = 0
    gt_single: np.ndarray = gt_frames[frame_idx]
    pred_single: np.ndarray = pred_frames[frame_idx]
    scaled_single: np.ndarray = scaled_frames[frame_idx]

    fig: plt.Figure = plt.figure(figsize=(14, 6))
    ax: plt.Axes = fig.add_subplot(111, projection="3d")

    _draw_skeleton(ax, pred_single, "green", "Predicted (before scaling)", offset_x=-1.0)
    _draw_skeleton(ax, scaled_single, "red", "Scaled (after scaling)", offset_x=0.0)
    _draw_skeleton(ax, gt_single, "blue", "Ground Truth", offset_x=1.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("SZI-MPJPE: Before Scaling / After Scaling / Ground Truth")
    ax.legend(loc="upper left")
    ax.view_init(elev=20, azim=45)

    # Equal aspect ratio
    all_pts: np.ndarray = np.vstack([
        pred_single + np.array([-1.0, 0, 0]),
        scaled_single,
        gt_single + np.array([1.0, 0, 0]),
    ])
    mid: np.ndarray = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    span: float = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max()) / 2 + 0.1
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    path: str = os.path.join(output_dir, "szi_mpjpe_visualization.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ---------------------------------------------------------------------------
# 3D Visualization (real example)
# ---------------------------------------------------------------------------

def visualize_real_example(output_dir: str) -> None:
    """Load a real prediction JSON and visualize before/after scaling."""
    print("\n  Generating real-example visualization...")

    # Find the latest training run with predictions
    training_dir: str = os.path.join(os.path.dirname(__file__), "training_runs")
    if not os.path.isdir(training_dir):
        print("    No training_runs directory found. Skipping real example.")
        return

    # Find the most recent run with predictions
    runs: list[str] = sorted(
        [d for d in os.listdir(training_dir) if d.startswith("motionbert-run-")],
        reverse=True,
    )
    json_path: str | None = None
    for run_name in runs:
        pred_dir: str = os.path.join(training_dir, run_name, "predictions")
        if os.path.isdir(pred_dir):
            json_files: list[str] = [f for f in os.listdir(pred_dir) if f.endswith(".json")]
            if json_files:
                json_path = os.path.join(pred_dir, json_files[0])
                break

    if json_path is None:
        print("    No prediction JSON found. Skipping real example.")
        return

    print(f"    Loading: {json_path}")
    with open(json_path, "r") as f:
        data: dict = json.load(f)

    # Extract detector 3D and GT 3D
    frames: list[dict] = data["frames"]
    det_list: list[np.ndarray] = []
    gt_list: list[np.ndarray] = []
    for frame in frames:
        if frame.get("ground_truth_3d") is not None:
            det_list.append(np.array(frame["detector_3d"]))
            gt_list.append(np.array(frame["ground_truth_3d"]))

    if len(det_list) < 1:
        print("    No frames with GT found. Skipping real example.")
        return

    det_arr: np.ndarray = np.array(det_list)  # (F, 17, 3)
    gt_arr: np.ndarray = np.array(gt_list)

    # Root-relative, eval joints
    det_rr: np.ndarray = det_arr - det_arr[:, 0:1, :]
    gt_rr: np.ndarray = gt_arr - gt_arr[:, 0:1, :]

    ej: list[int] = EVAL_JOINTS
    det_eval: np.ndarray = det_rr[:, ej, :]
    gt_eval: np.ndarray = gt_rr[:, ej, :]

    s: float = optimal_scale(det_eval, gt_eval)
    scaled_eval: np.ndarray = s * det_eval
    print(f"    Optimal scale: {s:.4f}")
    print(f"    MPJPE before: {mpjpe(det_eval, gt_eval)*100:.2f} cm")
    print(f"    MPJPE after:  {mpjpe(scaled_eval, gt_eval)*100:.2f} cm")

    # Use all 17 joints for visualization (scale the full skeleton)
    mid_frame: int = len(det_list) // 2
    det_single: np.ndarray = det_rr[mid_frame]  # (17, 3)
    gt_single: np.ndarray = gt_rr[mid_frame]
    scaled_single: np.ndarray = s * det_single

    os.makedirs(output_dir, exist_ok=True)
    fig: plt.Figure = plt.figure(figsize=(14, 6))
    ax: plt.Axes = fig.add_subplot(111, projection="3d")

    _draw_skeleton(ax, det_single, "green", f"Detector (s=1.0)", offset_x=-1.0)
    _draw_skeleton(ax, scaled_single, "red", f"Scaled (s={s:.3f})", offset_x=0.0)
    _draw_skeleton(ax, gt_single, "blue", "Ground Truth", offset_x=1.0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"SZI-MPJPE Real Example (frame {mid_frame}, scale={s:.3f})")
    ax.legend(loc="upper left")
    ax.view_init(elev=20, azim=45)

    # Equal aspect ratio
    all_pts: np.ndarray = np.vstack([
        det_single + np.array([-1.0, 0, 0]),
        scaled_single,
        gt_single + np.array([1.0, 0, 0]),
    ])
    mid_pt: np.ndarray = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    span: float = float((all_pts.max(axis=0) - all_pts.min(axis=0)).max()) / 2 + 0.1
    ax.set_xlim(mid_pt[0] - span, mid_pt[0] + span)
    ax.set_ylim(mid_pt[1] - span, mid_pt[1] + span)
    ax.set_zlim(mid_pt[2] - span, mid_pt[2] + span)

    path: str = os.path.join(output_dir, "szi_mpjpe_real_example.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tests()

    output_dir: str = os.path.join(os.path.dirname(__file__), "test_output")
    visualize_synthetic(output_dir)
    visualize_real_example(output_dir)

    if failed > 0:
        print(f"\n  {failed} test(s) FAILED.")
        sys.exit(1)
    else:
        print(f"\n  All {passed} tests PASSED.")
        sys.exit(0)
