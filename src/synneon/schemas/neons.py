from __future__ import annotations

from pydantic import BaseModel


class NeonConfig(BaseModel):
    client_addr: str = "192.168.1.79:8080"
    client_url: str = "neon.local:8080"


class BlinkConfig(BaseModel):
    double_blink_action: str | None = "capture_world_camera_image_with_gaze_annotation"
    double_blink_delta: float = 0.3
    circle_annotation_radius: int = 75


class RecordingConfig(BaseModel):
    max_duration_s: float = -1.0


class SaliencyConfig(BaseModel):
    max_peaks: int = 5
    min_peak_threshold_pct: float = 0.2
    marker_shape: str = "X"  # "X" or "circle"
    marker_radius: int = 15
    gaussian_sigma: float = 60.0
    colormap: int = 2  # cv2.COLORMAP_JET is 2
    alpha: float = 0.6


class FixationConfig(BaseModel):
    algorithm: str = "I-DT"
    min_duration_ms: float = 100.0
    max_dispersion_pixels: float = 30.0


class GazeSample(BaseModel):
    """
    Standardized gaze sample.
    """

    x: float
    y: float
    timestamp: float
    is_worn: bool


class NeonCapturePayload(BaseModel):
    """
    Payload containing all raw multimodal data and configurations captured by the Neon device.
    """

    scene_image: bytes | None = None
    video: list[tuple[float, bytes]] = []  # List of (timestamp, image_bytes)
    gaze_points: list[tuple[float, float, float]] = []  # List of (timestamp, x, y)
    pupil_data: list[tuple[float, float, float]] = []  # List of (timestamp, left_mm, right_mm)
    audio: bytes | None = None

    blink_config: BlinkConfig
    fixation_config: FixationConfig
    saliency_config: SaliencyConfig
    recording_config: RecordingConfig
