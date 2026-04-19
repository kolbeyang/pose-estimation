"""Configuration models with Pydantic and JSON serialization."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class ExampleConfig(BaseModel):
    """Configuration for a single data example."""

    sequence: str = "171204_pose1_sample"
    camera: str = "00_00"
    start_frame: int = 0
    num_frames: int = 100
    person_idx: int = 0


class OptimizationConfig(BaseModel):
    """Optimizer hyperparameters."""

    num_steps: int = 50
    learning_rate: float = 0.002
    bone_length_lr: float = 0.001
    position_penalty_weight: float = 500.0
    rotation_penalty_scalar: float = 10.0
    heatmap_blur_sigma: float = 4.0
    confidence_epsilon: float = 1e-4
    anchor_weight: float = 0.0

    # Per-joint rotation penalty multipliers (relative to rotation_penalty_scalar).
    # Indexed by [SKELETON_16] joint order: Pelvis(0) through RWrist(15).
    rotation_penalty_multipliers: list[float] = Field(default_factory=lambda: [
        3.0,   # 0: Pelvis (root rotation)
        1.0,   # 1: RHip
        0.5,   # 2: RKnee
        0.2,   # 3: RAnkle
        1.0,   # 4: LHip
        0.5,   # 5: LKnee
        0.2,   # 6: LAnkle
        1.0,   # 7: Spine
        1.0,   # 8: Neck (Thorax / shoulder level)
        0.5,   # 9: HeadTop
        0.5,   # 10: LShoulder
        0.3,   # 11: LElbow
        0.1,   # 12: LWrist
        0.5,   # 13: RShoulder
        0.3,   # 14: RElbow
        0.1,   # 15: RWrist
    ])

    @property
    def rotation_penalty_per_joint(self) -> np.ndarray:
        """Compute absolute per-joint rotation penalty weights."""
        return np.array(self.rotation_penalty_multipliers) * self.rotation_penalty_scalar


class PipelineGraphConfig(BaseModel):
    """Per-pipeline graph toggles."""

    all_examples_per_joint_mpjpe: bool = True
    per_example_per_joint_mpjpe: bool = True


class PipelineConfig(BaseModel):
    """Per-pipeline configuration."""

    is_generate_heatmap_videos: bool = True
    graphs: PipelineGraphConfig = Field(default_factory=PipelineGraphConfig)
    # Per-pipeline optimization overrides (merged on top of shared optimization config)
    optimization: dict | None = None


class CrossPipelineGraphConfig(BaseModel):
    """Cross-pipeline comparison graph toggles (only when both pipelines run)."""

    per_example_mediapipe_vs_motionbert: bool = True
    position_error_per_joint_improvement: bool = True
    velocity_error_per_joint_improvement: bool = True
    metrics_comparison: bool = True


class RunConfig(BaseModel):
    """Top-level run configuration."""

    # Pipeline toggles -- if present, that pipeline runs
    mediapipe: PipelineConfig | None = None
    motionbert: PipelineConfig | None = None

    # Cross-pipeline graphs -- only valid if both pipelines present
    graphs: CrossPipelineGraphConfig | None = None

    # Shared settings
    examples: list[ExampleConfig] = Field(default_factory=lambda: [])
    data_root: str = "data/panoptic-toolbox"
    output_dir: str | None = None
    target_fps: float = 10.0
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    # Pipeline-specific
    sh_batch_size: int = 32
    motionbert_conf_threshold: float = 0.0
    per_frame_bbox: bool = False

    # Output control
    generate_graphs: bool = True
    generate_video: bool = True

    model_config = {"json_schema_extra": {"examples": []}}

    def optimization_for_pipeline(self, pipeline_name: str) -> OptimizationConfig:
        """Get optimization config for a specific pipeline.

        Merges pipeline-specific overrides on top of the shared optimization config.

        Args:
            pipeline_name: "motionbert" or "mediapipe".

        Returns:
            OptimizationConfig with any per-pipeline overrides applied.
        """
        pipeline_cfg = getattr(self, pipeline_name, None)
        if pipeline_cfg is None or pipeline_cfg.optimization is None:
            return self.optimization
        # Merge: start from shared config dict, overlay pipeline overrides
        base = self.optimization.model_dump()
        base.update(pipeline_cfg.optimization)
        return OptimizationConfig.model_validate(base)


def load_config(path: str) -> RunConfig:
    """Load a RunConfig from a JSON file.

    Args:
        path: Path to JSON config file.

    Returns:
        RunConfig instance.
    """
    import json
    with open(path) as f:
        data = json.load(f)
    return RunConfig.model_validate(data)


def save_config(config: RunConfig, path: str) -> None:
    """Save a RunConfig to a JSON file.

    Args:
        config: RunConfig instance.
        path: Path to write JSON.
    """
    import json
    with open(path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)


def default_single_config() -> RunConfig:
    """Create a default RunConfig with a single example (pose1_sample)."""
    return RunConfig(
        examples=[ExampleConfig()],
        target_fps=10.0,
    )


def default_full_config() -> RunConfig:
    """Create a default full-evaluation RunConfig with all 25 benchmark examples."""
    from cmu_data import EXAMPLES
    examples = [
        ExampleConfig(
            sequence=seq, camera=cam,
            start_frame=sf, num_frames=nf, person_idx=pi,
        )
        for seq, cam, sf, nf, pi in EXAMPLES
    ]
    return RunConfig(
        examples=examples,
        target_fps=10.0,
    )
