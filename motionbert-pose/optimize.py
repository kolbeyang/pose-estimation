"""FK optimization loop.

Optimises FK parameters (root position, root rotation, local rotations,
and shared bone lengths) against 2D detection targets.
"""

import numpy as np
import torch

from camera import Camera
from fk import forward_kinematics, positions_to_fk_params
from scoring import compute_total_score
from skeleton import NUM_JOINTS
import config as cfg


def _get_sigma(step: int, num_steps: int) -> float:
    """Get sigma for current step from coarse-to-fine schedule."""
    progress: float = step / max(num_steps - 1, 1)
    for frac, sigma in cfg.SIGMA_SCHEDULE:
        if progress <= frac:
            return sigma
    return cfg.SIGMA_SCHEDULE[-1][1]


def run_optimization(
    initial_positions_cam: list[np.ndarray],
    target_2d: list[np.ndarray],
    visibility: list[np.ndarray],
    camera: Camera,
    num_steps: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[float]]:
    """Run FK optimization.

    Args:
        initial_positions_cam: Per-frame (17, 3) camera-space positions from detector.
        target_2d: Per-frame (17, 2) pixel target positions.
        visibility: Per-frame (17,) visibility weights.
        camera: Camera for 3D->2D projection.
        num_steps: Override for cfg.NUM_STEPS.

    Returns:
        optimized_3d: list of (17, 3) optimized camera-space positions per frame
        bone_lengths_final: (17,) final shared bone lengths
        loss_history: list of loss values per step
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
    target_2d_t: list[torch.Tensor] = [
        torch.tensor(t, dtype=torch.float32) for t in target_2d
    ]
    visibility_t: list[torch.Tensor] = [
        torch.tensor(v, dtype=torch.float32) for v in visibility
    ]

    # Apply visibility threshold -- zero out low-confidence joints
    for i in range(len(visibility_t)):
        visibility_t[i] = torch.where(
            visibility_t[i] >= cfg.VISIBILITY_THRESHOLD,
            visibility_t[i],
            torch.zeros_like(visibility_t[i]),
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

    print(f"  Optimising {n_frames} frames for {num_steps} steps...")

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass: FK -> 3D positions -> project to 2D
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

        # Compute score with coarse-to-fine sigma
        sigma: float = _get_sigma(step, num_steps)
        total_score: torch.Tensor
        details: dict[str, float]
        total_score, details = compute_total_score(
            all_positions,
            all_projected_2d,
            all_local_rots_current,
            target_2d_t,
            visibility_t,
            sigma,
            cfg.POSITION_PENALTY_WEIGHT,
            rot_per_joint_weights,
            initial_positions_list=initial_positions_t,
            init_anchor_weight=cfg.INIT_ANCHOR_WEIGHT,
            all_joints_smooth_weight=cfg.ALL_JOINTS_SMOOTH_WEIGHT,
        )

        loss: torch.Tensor = -total_score
        loss.backward()
        optimizer.step()

        # Clamp bone lengths to positive
        with torch.no_grad():
            param_bone_lengths.clamp_(min=0.01)

        loss_history.append(float(loss.item()))

        if step % 20 == 0 or step == num_steps - 1:
            print(
                f"    Step {step:4d}/{num_steps}  "
                f"loss={loss.item():.1f}  "
                f"heatmap={details['heatmap']:.1f}  "
                f"sigma={sigma:.0f}  "
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

    return optimized_3d, bone_lengths_final, loss_history
