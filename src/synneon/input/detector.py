from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synneon.schemas.gestures import GestureEvent, GestureType
from synneon.schemas.neons import GazeSample

if TYPE_CHECKING:
    from synneon.schemas.gestures import CompassSettings

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    np = None  # type: ignore


@dataclass
class DetectorState:
    """
    Mutable state for gesture detection.
    """

    # Cooldown state
    last_gesture_time: float = 0.0


def _moving_average_smooth(
    xs: list[float], ys: list[float], ts: list[float], smoothing_factor: float
) -> list[tuple[float, float, float]]:
    """
    Simple per-window moving average smoothing.

    smoothing_factor in [0,1] maps to odd kernel sizes [1,3,5,7,9].
    """
    sf = smoothing_factor
    if sf < 0.0:
        sf = 0.0
    elif sf > 1.0:
        sf = 1.0

    k = 1 + 2 * int(round(sf * 4))  # 1,3,5,7,9
    half = k // 2

    # Prefix sums for O(n)
    px = [0.0]
    py = [0.0]
    for x in xs:
        px.append(px[-1] + x)
    for y in ys:
        py.append(py[-1] + y)

    smoothed: list[tuple[float, float, float]] = []
    n = len(xs)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n - 1, i + half)
        count = hi - lo + 1
        sx = px[hi + 1] - px[lo]
        sy = py[hi + 1] - py[lo]
        smoothed.append((sx / count, sy / count, ts[i]))

    return smoothed


def detect_gestures(
    trajectory: list[GazeSample] | tuple[GazeSample, ...],
    state: DetectorState,
    config: CompassSettings,
) -> GestureEvent | None:
    """
    Detect dwell-based directional gestures (UP/DOWN/LEFT/RIGHT).

    Logic:
    1. Check cooldown.
    2. Filter worn samples.
    3. Smooth trajectory.
    4. Determine bounds (min/max x/y) from config or defaults (0-1).
    5. Define active zones based on `config.threshold`.
       Example: If threshold is 0.8, the zone is the extreme 20% of the range.
       Left Zone: x < min_x + (1 - 0.8) * range_x
    6. Check if the *tail* of the trajectory dwells in a zone for `config.dwell_time`.
    """

    if not trajectory:
        return None

    # Filter worn samples
    worn_samples = [s for s in trajectory if s.is_worn]
    if len(worn_samples) < 4:
        return None

    # Sort by timestamp to be safe
    worn_samples.sort(key=lambda s: s.timestamp)

    # Extract coordinates
    xs = [s.x for s in worn_samples]
    ys = [s.y for s in worn_samples]
    ts = [s.timestamp for s in worn_samples]

    # Smooth
    smoothed = _moving_average_smooth(xs, ys, ts, config.smoothing_factor)
    if len(smoothed) < 4:
        return None

    end_x, end_y, end_t = smoothed[-1]

    # Cooldown gate
    if (end_t - state.last_gesture_time) < config.cooldown:
        return None

    # Determine Range
    # Fallback to 0.0 - 1.0 if config is missing values
    min_x = config.min_x if config.min_x is not None else 0.0
    max_x = config.max_x if config.max_x is not None else 1.0
    min_y = config.min_y if config.min_y is not None else 0.0
    max_y = config.max_y if config.max_y is not None else 1.0

    range_x = max_x - min_x
    range_y = max_y - min_y

    # Prevent division by zero or weird ranges by ensuring non-zero positive range if possible,
    # or just using absolute logic.
    if range_x == 0:
        range_x = 1.0
    if range_y == 0:
        range_y = 1.0

    # Threshold: Percentage of range "deep" into the direction.
    # If threshold = 0.8 (80%), then we must be in the "last 20%" of the distance.
    # Left (towards min_x): x <= min_x + (1 - T) * range_x
    # Right (towards max_x): x >= min_x + T * range_x
    # Up (towards min_y): y <= min_y + (1 - T) * range_y  (Assuming min_y is Top/Up)
    # Down (towards max_y): y >= min_y + T * range_y    (Assuming max_y is Bottom/Down)

    T = config.threshold
    # Clamp T
    if T < 0.0:
        T = 0.0
    elif T > 1.0:
        T = 1.0

    thresh_left_x = min_x + (1.0 - T) * range_x
    thresh_right_x = min_x + T * range_x
    thresh_up_y = min_y + (1.0 - T) * range_y
    thresh_down_y = min_y + T * range_y

    dwell_time = config.dwell_time
    if dwell_time < 0:
        dwell_time = 0

    # Helper to check dwell in a specific direction
    def get_dwell_duration(direction: GestureType) -> float:
        duration = 0.0
        # Iterate backwards
        for i in range(len(smoothed) - 1, -1, -1):
            curr_x, curr_y, curr_t = smoothed[i]

            # Check condition
            in_zone = False
            if direction == GestureType.LEFT:
                in_zone = curr_x <= thresh_left_x
            elif direction == GestureType.RIGHT:
                in_zone = curr_x >= thresh_right_x
            elif direction == GestureType.UP:
                in_zone = curr_y <= thresh_up_y
            elif direction == GestureType.DOWN:
                in_zone = curr_y >= thresh_down_y

            if not in_zone:
                break

            # If we are not the last sample, add time difference with NEXT sample (forward in time)
            # Actually, we are iterating backwards.
            # Time accumulated is from the "start of this continuous block" to "end_t".
            # The simplest way is: start_t of the block.
            start_t = curr_t
            duration = end_t - start_t

            if duration >= dwell_time:
                return duration

        return duration

    # Check all directions
    # Order doesn't strictly matter if regions are disjoint (T > 0.5)
    candidates = []

    # Left
    d_left = get_dwell_duration(GestureType.LEFT)
    if d_left >= dwell_time:
        candidates.append((GestureType.LEFT, d_left))

    # Right
    d_right = get_dwell_duration(GestureType.RIGHT)
    if d_right >= dwell_time:
        candidates.append((GestureType.RIGHT, d_right))

    # Up
    d_up = get_dwell_duration(GestureType.UP)
    if d_up >= dwell_time:
        candidates.append((GestureType.UP, d_up))

    # Down
    d_down = get_dwell_duration(GestureType.DOWN)
    if d_down >= dwell_time:
        candidates.append((GestureType.DOWN, d_down))

    if not candidates:
        return None

    # If multiple candidates (unlikely with high threshold), pick the one with longest dwell?
    # Or just the first one?
    # Let's pick the one with the longest dwell duration.
    best_gesture, best_duration = max(candidates, key=lambda x: x[1])

    # Calculate magnitude (distance from center of range)
    # Center
    center_x = min_x + range_x / 2.0
    center_y = min_y + range_y / 2.0
    dx = end_x - center_x
    dy = end_y - center_y
    magnitude = math.sqrt(dx * dx + dy * dy)

    # Update state
    state.last_gesture_time = end_t

    return GestureEvent(type=best_gesture, timestamp=end_t, magnitude=magnitude, duration=best_duration)
