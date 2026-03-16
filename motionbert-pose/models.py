"""Pydantic model definitions for the MotionBERT pose estimation pipeline."""

from pydantic import BaseModel


class CameraParams(BaseModel):
    """Camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int


class ExampleConfig(BaseModel):
    """Configuration for one CMU Panoptic example."""

    sequence_name: str
    camera_name: str
    start_frame: int
    num_frames: int
    person_idx: int


class DetectionResult(BaseModel):
    """Metadata about a detection run. Actual tensor data stored separately."""

    num_frames: int
    num_joints: int  # always 17
    image_height: int
    image_width: int
    bbox: list[float]  # [x1, y1, x2, y2] union bbox


class EvaluationResult(BaseModel):
    """Evaluation metrics for one example."""

    mpjpe: float  # meters
    p_mpjpe: float  # meters
    per_joint_mpjpe: list[float]  # 12 eval joints
    per_frame_mpjpe: list[float]
    num_frames: int
    num_eval_joints: int


class ExampleResult(BaseModel):
    """Summary result for one example."""

    name: str
    sequence: str
    camera: str
    start_frame: int
    num_frames: int
    camera_params: CameraParams
    mpjpe: float | None = None
    p_mpjpe: float | None = None
    mpjpe_cm: float | None = None
    p_mpjpe_cm: float | None = None
