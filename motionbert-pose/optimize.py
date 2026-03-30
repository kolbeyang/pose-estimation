"""FK optimization loop.

Optimises FK parameters (root position, root rotation, local rotations,
and shared bone lengths) against 2D detection targets.
"""

import numpy as np
import torch

from camera import Camera
from fk import forward_kinematics, forward_kinematics_batch, positions_to_fk_params
from scoring import compute_total_score, compute_total_score_batch
from skeleton import NUM_JOINTS
import config as cfg


def _get_blur_sigma(step: int, num_steps: int, schedule: list[tuple[float, float]]) -> float:
    """Get heatmap blur sigma for current step from schedule."""
    progress: float = step / max(num_steps - 1, 1)
    for frac, sigma in schedule:
        if progress <= frac:
            return sigma
    return schedule[-1][1]


def _apply_blur_torch(
    heatmaps: list[torch.Tensor], sigma: float
) -> list[torch.Tensor]:
    """Apply Gaussian blur to heatmaps using scipy (called rarely, not per-step)."""
    import scipy.ndimage
    blurred: list[torch.Tensor] = []
    for hm in heatmaps:
        hm_np = hm.numpy()
        blurred_np = np.stack([
            scipy.ndimage.gaussian_filter(hm_np[c], sigma=sigma)
            for c in range(hm_np.shape[0])
        ])
        blurred.append(torch.tensor(blurred_np, dtype=torch.float32))
    return blurred


def run_optimization(
    initial_positions_cam: list[np.ndarray],
    visibility: list[np.ndarray],
    camera: Camera,
    num_steps: int | None = None,
    heatmaps: list[np.ndarray] | None = None,
    affine: np.ndarray | None = None,
    heatmap_blur_schedule: list[tuple[float, float]] | None = None,
    profile: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, list[float], dict | None]:
    """Run FK optimization.

    Args:
        initial_positions_cam: Per-frame (17, 3) camera-space positions from detector.
        visibility: Per-frame (17,) visibility weights.
        camera: Camera for 3D->2D projection.
        num_steps: Override for cfg.NUM_STEPS.
        heatmaps: Per-frame (16, 64, 64) Stacked Hourglass heatmaps (optional).
        affine: (2, 3) affine transform from 256-crop to original pixels (optional).

        profile: If True, collect per-step timing breakdown.

    Returns:
        optimized_3d: list of (17, 3) optimized camera-space positions per frame
        bone_lengths_final: (17,) final shared bone lengths
        loss_history: list of loss values per step
        profile_data: timing breakdown dict (None when profile=False)
    """
    if num_steps is None:
        num_steps = cfg.NUM_STEPS

    n_frames: int = len(initial_positions_cam)

    # Convert initial positions to FK parameters
    all_root_pos: list[np.ndarray] = []
    all_root_rot: list[np.ndarray] = []
    all_local_rots: list[np.ndarray] = []
    all_bone_lengths: list[np.ndarray] = []

    print(f"  Initializing FK parameters for {n_frames} frames...")
    roundtrip_errors: list[float] = []
    for i, positions in enumerate(initial_positions_cam):
        root_pos: np.ndarray
        root_rot: np.ndarray
        local_rots: np.ndarray
        bone_lengths: np.ndarray
        root_pos, root_rot, local_rots, bone_lengths = positions_to_fk_params(positions)

        # Verify roundtrip
        with torch.no_grad():
            reconstructed: torch.Tensor = forward_kinematics(
                torch.tensor(root_pos, dtype=torch.float32),
                torch.tensor(root_rot, dtype=torch.float32),
                torch.tensor(local_rots, dtype=torch.float32),
                torch.tensor(bone_lengths, dtype=torch.float32),
            )
            rt_error: float = float(
                torch.mean(torch.norm(
                    reconstructed - torch.tensor(positions, dtype=torch.float32),
                    dim=-1,
                )).item()
            )
            roundtrip_errors.append(rt_error)

        all_root_pos.append(root_pos)
        all_root_rot.append(root_rot)
        all_local_rots.append(local_rots)
        all_bone_lengths.append(bone_lengths)

    mean_rt_error: float = float(np.mean(roundtrip_errors))
    max_rt_error: float = float(np.max(roundtrip_errors))
    print(f"    FK roundtrip error: mean={mean_rt_error*100:.4f} cm, max={max_rt_error*100:.4f} cm")
    if mean_rt_error > 0.001:
        print(f"    WARNING: Roundtrip error > 0.1 cm -- FK may have bugs!")

    # Create learnable parameters
    param_root_pos: list[torch.Tensor] = [
        torch.tensor(rp, dtype=torch.float32, requires_grad=True) for rp in all_root_pos
    ]
    param_root_rot: list[torch.Tensor] = [
        torch.tensor(rr, dtype=torch.float32, requires_grad=True) for rr in all_root_rot
    ]
    param_local_rots: list[torch.Tensor] = [
        torch.tensor(lr, dtype=torch.float32, requires_grad=True)
        for lr in all_local_rots
    ]

    # Shared bone lengths: median across frames
    median_bone_lengths: np.ndarray = np.median(np.array(all_bone_lengths), axis=0)
    param_bone_lengths: torch.Tensor = torch.tensor(
        median_bone_lengths,
        dtype=torch.float32,
        requires_grad=True,
    )

    # Store initial positions (for anchor penalty)
    initial_positions_t: list[torch.Tensor] = [
        torch.tensor(pos, dtype=torch.float32) for pos in initial_positions_cam
    ]

    # Target tensors (not learnable)
    visibility_t: list[torch.Tensor] = [
        torch.tensor(v, dtype=torch.float32) for v in visibility
    ]

    # Convert heatmaps to torch tensors (once, not per step)
    heatmaps_t: list[torch.Tensor] | None = None
    affine_t: torch.Tensor | None = None
    if heatmaps is not None:
        heatmaps_t = [
            torch.tensor(hm, dtype=torch.float32) for hm in heatmaps
        ]
        if affine is not None:
            affine_t = torch.tensor(affine, dtype=torch.float32)
        print(f"    Using real Stacked Hourglass heatmaps for scoring")

    # Apply fixed blur from config (if no dynamic schedule is provided)
    if heatmaps_t is not None and heatmap_blur_schedule is None and cfg.HEATMAP_BLUR_SIGMA > 0:
        heatmaps_t = _apply_blur_torch(heatmaps_t, cfg.HEATMAP_BLUR_SIGMA)

    # Store originals for re-blurring during coarse-to-fine schedule
    heatmaps_t_orig: list[torch.Tensor] | None = None
    if heatmaps_t is not None and heatmap_blur_schedule is not None:
        heatmaps_t_orig = [hm.clone() for hm in heatmaps_t]
        # Apply initial blur
        initial_blur = _get_blur_sigma(0, num_steps, heatmap_blur_schedule)
        if initial_blur > 0:
            heatmaps_t = _apply_blur_torch(heatmaps_t_orig, initial_blur)
    current_blur_sigma: float = (
        _get_blur_sigma(0, num_steps, heatmap_blur_schedule)
        if heatmap_blur_schedule else 0.0
    )

    # Per-joint rotation penalty weights
    rot_per_joint_weights: torch.Tensor = torch.tensor(
        cfg.ROTATION_PENALTY_PER_JOINT,
        dtype=torch.float32,
    )

    # Optimizer (root_pos gets 1.5x LR, bone lengths get their own LR)
    angle_params: list[torch.Tensor] = param_root_rot + param_local_rots
    optimizer: torch.optim.Adam = torch.optim.Adam([
        {"params": param_root_pos, "lr": cfg.LEARNING_RATE * 1.5},
        {"params": angle_params, "lr": cfg.LEARNING_RATE},
        {"params": [param_bone_lengths], "lr": cfg.BONE_LENGTH_LR},
    ])

    loss_history: list[float] = []

    # Profiling containers
    import time as _time
    profile_fk_ms: list[float] = []
    profile_scoring_ms: list[float] = []
    profile_backward_ms: list[float] = []
    profile_step_ms: list[float] = []

    print(f"  Optimising {n_frames} frames for {num_steps} steps...")

    for step in range(num_steps):
        step_t0: float = _time.perf_counter() if profile else 0.0

        optimizer.zero_grad()

        # Forward pass: FK -> 3D positions -> project to 2D
        fk_t0: float = _time.perf_counter() if profile else 0.0
        all_positions: list[torch.Tensor] = []
        all_projected_2d: list[torch.Tensor] = []
        all_local_rots_current: list[torch.Tensor] = []

        for i in range(n_frames):
            positions_3d: torch.Tensor = forward_kinematics(
                param_root_pos[i],
                param_root_rot[i],
                param_local_rots[i],
                param_bone_lengths,
            )
            projected_2d_frame: torch.Tensor = camera.world_to_image_torch(positions_3d)

            all_positions.append(positions_3d)
            all_projected_2d.append(projected_2d_frame)
            all_local_rots_current.append(param_local_rots[i])

        if profile:
            fk_t1: float = _time.perf_counter()
            profile_fk_ms.append((fk_t1 - fk_t0) * 1000)

        # Update blur if schedule changed
        if heatmaps_t_orig is not None and heatmap_blur_schedule is not None:
            new_blur = _get_blur_sigma(step, num_steps, heatmap_blur_schedule)
            if abs(new_blur - current_blur_sigma) > 1e-6:
                if new_blur > 0:
                    heatmaps_t = _apply_blur_torch(heatmaps_t_orig, new_blur)
                else:
                    heatmaps_t = [hm.clone() for hm in heatmaps_t_orig]
                current_blur_sigma = new_blur
                print(f"    [Step {step}] Heatmap blur sigma changed to {new_blur:.1f}")

        # Compute score
        scoring_t0: float = _time.perf_counter() if profile else 0.0
        total_score: torch.Tensor
        details: dict[str, float]
        total_score, details = compute_total_score(
            all_positions,
            all_projected_2d,
            all_local_rots_current,
            visibility_t,
            cfg.POSITION_PENALTY_WEIGHT,
            rot_per_joint_weights,
            initial_positions_list=initial_positions_t,
            init_anchor_weight=cfg.INIT_ANCHOR_WEIGHT,
            heatmaps_list=heatmaps_t,
            affine=affine_t,
            confidence_epsilon=cfg.CONFIDENCE_EPSILON,
        )

        if profile:
            scoring_t1: float = _time.perf_counter()
            profile_scoring_ms.append((scoring_t1 - scoring_t0) * 1000)

        loss: torch.Tensor = -total_score

        backward_t0: float = _time.perf_counter() if profile else 0.0
        loss.backward()
        optimizer.step()
        if profile:
            backward_t1: float = _time.perf_counter()
            profile_backward_ms.append((backward_t1 - backward_t0) * 1000)

        # Clamp bone lengths to positive
        with torch.no_grad():
            param_bone_lengths.clamp_(min=0.01)

        loss_history.append(float(loss.item()))

        if profile:
            step_t1: float = _time.perf_counter()
            profile_step_ms.append((step_t1 - step_t0) * 1000)

        if step % 20 == 0 or step == num_steps - 1:
            print(
                f"    Step {step:4d}/{num_steps}  "
                f"loss={loss.item():.1f}  "
                f"heatmap={details['heatmap']:.1f}  "
                f"pos_p={details['pos_penalty']:.4f}  "
                f"rot_p={details['rot_penalty']:.4f}"
            )

    # Extract final optimised 3D positions
    optimized_3d: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(n_frames):
            positions_final: torch.Tensor = forward_kinematics(
                param_root_pos[i],
                param_root_rot[i],
                param_local_rots[i],
                param_bone_lengths,
            )
            optimized_3d.append(positions_final.numpy().copy())

    bone_lengths_final: np.ndarray = param_bone_lengths.detach().numpy().copy()

    profile_data: dict | None = None
    if profile:
        profile_data = {
            "fk_projection_ms": profile_fk_ms,
            "scoring_ms": profile_scoring_ms,
            "backward_ms": profile_backward_ms,
            "step_total_ms": profile_step_ms,
        }

    return optimized_3d, bone_lengths_final, loss_history, profile_data


def run_optimization_batched(
    initial_positions_cam: list[np.ndarray],
    visibility: list[np.ndarray],
    camera: Camera,
    num_steps: int | None = None,
    heatmaps: list[np.ndarray] | None = None,
    affine: np.ndarray | None = None,
    heatmap_blur_schedule: list[tuple[float, float]] | None = None,
    profile: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, list[float], dict | None]:
    """Batched FK optimization -- vectorized across frames.

    Same interface as run_optimization but uses batch FK and batch scoring
    for significantly reduced Python overhead and autograd graph size.

    Args:
        Same as run_optimization.

    Returns:
        Same as run_optimization.
    """
    import time as _time

    if num_steps is None:
        num_steps = cfg.NUM_STEPS

    n_frames: int = len(initial_positions_cam)

    # Convert initial positions to FK parameters
    all_root_pos: list[np.ndarray] = []
    all_root_rot: list[np.ndarray] = []
    all_local_rots: list[np.ndarray] = []
    all_bone_lengths: list[np.ndarray] = []

    print(f"  Initializing FK parameters for {n_frames} frames...")
    roundtrip_errors: list[float] = []
    for i, positions in enumerate(initial_positions_cam):
        root_pos, root_rot, local_rots, bone_lengths = positions_to_fk_params(positions)
        with torch.no_grad():
            reconstructed = forward_kinematics(
                torch.tensor(root_pos, dtype=torch.float32),
                torch.tensor(root_rot, dtype=torch.float32),
                torch.tensor(local_rots, dtype=torch.float32),
                torch.tensor(bone_lengths, dtype=torch.float32),
            )
            rt_error = float(torch.mean(torch.norm(
                reconstructed - torch.tensor(positions, dtype=torch.float32), dim=-1,
            )).item())
            roundtrip_errors.append(rt_error)
        all_root_pos.append(root_pos)
        all_root_rot.append(root_rot)
        all_local_rots.append(local_rots)
        all_bone_lengths.append(bone_lengths)

    mean_rt_error = float(np.mean(roundtrip_errors))
    max_rt_error = float(np.max(roundtrip_errors))
    print(f"    FK roundtrip error: mean={mean_rt_error*100:.4f} cm, max={max_rt_error*100:.4f} cm")

    # Create BATCHED learnable parameters: (F, 3), (F, 3), (F, J, 3)
    param_root_pos: torch.Tensor = torch.tensor(
        np.array(all_root_pos), dtype=torch.float32, requires_grad=True,
    )  # (F, 3)
    param_root_rot: torch.Tensor = torch.tensor(
        np.array(all_root_rot), dtype=torch.float32, requires_grad=True,
    )  # (F, 3)
    param_local_rots: torch.Tensor = torch.tensor(
        np.array(all_local_rots), dtype=torch.float32, requires_grad=True,
    )  # (F, J, 3)

    # Shared bone lengths: median across frames
    median_bone_lengths: np.ndarray = np.median(np.array(all_bone_lengths), axis=0)
    param_bone_lengths: torch.Tensor = torch.tensor(
        median_bone_lengths, dtype=torch.float32, requires_grad=True,
    )

    # Non-learnable tensors (stacked)
    visibility_t: torch.Tensor = torch.tensor(
        np.array(visibility), dtype=torch.float32,
    )  # (F, J)

    # Heatmaps: (F, 16, 64, 64)
    heatmaps_t: torch.Tensor | None = None
    affine_t: torch.Tensor | None = None
    if heatmaps is not None:
        heatmaps_t = torch.tensor(np.array(heatmaps), dtype=torch.float32)
        if affine is not None:
            affine_t = torch.tensor(affine, dtype=torch.float32)
        print(f"    Using real Stacked Hourglass heatmaps for scoring (batched)")

    # Apply fixed blur
    if heatmaps_t is not None and heatmap_blur_schedule is None and cfg.HEATMAP_BLUR_SIGMA > 0:
        heatmaps_list_for_blur = [heatmaps_t[i] for i in range(n_frames)]
        heatmaps_list_for_blur = _apply_blur_torch(heatmaps_list_for_blur, cfg.HEATMAP_BLUR_SIGMA)
        heatmaps_t = torch.stack(heatmaps_list_for_blur)

    rot_per_joint_weights: torch.Tensor = torch.tensor(
        cfg.ROTATION_PENALTY_PER_JOINT, dtype=torch.float32,
    )

    # Optimizer
    optimizer: torch.optim.Adam = torch.optim.Adam([
        {"params": [param_root_pos], "lr": cfg.LEARNING_RATE * 1.5},
        {"params": [param_root_rot, param_local_rots], "lr": cfg.LEARNING_RATE},
        {"params": [param_bone_lengths], "lr": cfg.BONE_LENGTH_LR},
    ])

    loss_history: list[float] = []
    profile_fk_ms: list[float] = []
    profile_scoring_ms: list[float] = []
    profile_backward_ms: list[float] = []
    profile_step_ms: list[float] = []

    print(f"  Optimising {n_frames} frames for {num_steps} steps (batched)...")

    for step in range(num_steps):
        step_t0: float = _time.perf_counter() if profile else 0.0

        optimizer.zero_grad()

        # Batched FK: (F, J, 3) positions
        fk_t0: float = _time.perf_counter() if profile else 0.0
        all_positions: torch.Tensor = forward_kinematics_batch(
            param_root_pos, param_root_rot, param_local_rots, param_bone_lengths,
        )  # (F, J, 3)

        # Batched projection
        pos_flat: torch.Tensor = all_positions.reshape(-1, 3)  # (F*J, 3)
        proj_flat: torch.Tensor = camera.world_to_image_torch(pos_flat)  # (F*J, 2)
        all_projected_2d: torch.Tensor = proj_flat.reshape(n_frames, NUM_JOINTS, 2)

        if profile:
            fk_t1 = _time.perf_counter()
            profile_fk_ms.append((fk_t1 - fk_t0) * 1000)

        # Batched scoring
        scoring_t0: float = _time.perf_counter() if profile else 0.0
        total_score, details = compute_total_score_batch(
            all_positions, all_projected_2d, param_local_rots,
            visibility_t,
            cfg.POSITION_PENALTY_WEIGHT, rot_per_joint_weights,
            heatmaps=heatmaps_t,
            affine=affine_t,
            confidence_epsilon=cfg.CONFIDENCE_EPSILON,
        )
        if profile:
            scoring_t1 = _time.perf_counter()
            profile_scoring_ms.append((scoring_t1 - scoring_t0) * 1000)

        loss: torch.Tensor = -total_score

        backward_t0: float = _time.perf_counter() if profile else 0.0
        loss.backward()
        optimizer.step()
        if profile:
            backward_t1 = _time.perf_counter()
            profile_backward_ms.append((backward_t1 - backward_t0) * 1000)

        with torch.no_grad():
            param_bone_lengths.clamp_(min=0.01)

        loss_history.append(float(loss.item()))

        if profile:
            step_t1 = _time.perf_counter()
            profile_step_ms.append((step_t1 - step_t0) * 1000)

        if step % 20 == 0 or step == num_steps - 1:
            print(
                f"    Step {step:4d}/{num_steps}  "
                f"loss={loss.item():.1f}  "
                f"heatmap={details['heatmap']:.1f}  "
                f"pos_p={details['pos_penalty']:.4f}  "
                f"rot_p={details['rot_penalty']:.4f}"
            )

    # Extract final positions
    optimized_3d: list[np.ndarray] = []
    with torch.no_grad():
        final_positions = forward_kinematics_batch(
            param_root_pos, param_root_rot, param_local_rots, param_bone_lengths,
        )
        for i in range(n_frames):
            optimized_3d.append(final_positions[i].numpy().copy())

    bone_lengths_final: np.ndarray = param_bone_lengths.detach().numpy().copy()

    profile_data: dict | None = None
    if profile:
        profile_data = {
            "fk_projection_ms": profile_fk_ms,
            "scoring_ms": profile_scoring_ms,
            "backward_ms": profile_backward_ms,
            "step_total_ms": profile_step_ms,
        }

    return optimized_3d, bone_lengths_final, loss_history, profile_data
