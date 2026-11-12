from enum import StrEnum

from pydantic import BaseModel, Field


class CompassSettings(BaseModel):
    # --- Dwell / direction gesture detection ---
    # `threshold` is interpreted as a percentage of full-range normalized gaze travel
    # in a direction relative to a baseline/center point (e.g., 0.8 => 80% left/right/up/down).
    threshold: float = 0.8
    dwell_time: float = 0.5

    # Calibration bounds (optional, populated by calibration)
    min_x: float | None = -45.1537
    max_x: float | None = 1490.5511
    min_y: float | None = -45.1537
    max_y: float | None = 1129.8431

    # --- Motion / timing fields ---
    velocity_threshold: float = 2.0
    min_displacement: float = 0.15
    cooldown: float = 1.0
    smoothing_factor: float = 0.3
    sample_time_length: float = 1.0


class GesturesSettings(BaseModel):
    compass: CompassSettings = Field(default_factory=CompassSettings)


class GestureType(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class GestureEvent(BaseModel):
    type: GestureType
    timestamp: float
    magnitude: float = 0.0
    duration: float = 0.0
