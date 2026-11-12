import math
from dataclasses import dataclass

import cv2  # type: ignore
import numpy as np  # type: ignore

from synneon.schemas.gestures import CompassSettings, GestureEvent, GestureType
from synneon.schemas.neons import GazeSample


@dataclass
class DetectorState:
    """
    Mutable state for gesture detection.

    This detector is trajectory-window based: it looks at the *shape* of the gaze
    path within the provided window and detects a "center -> direction" gesture.

    We keep very little state:
      - smoothed_x/y: last smoothed gaze position (used to smooth the incoming stream)
      - last_timestamp: last timestamp used for smoothing dt validation
      - initialized: smoothing initialization flag
      - last_gesture_time: last emitted gesture timestamp (cooldown gate)

    Note: legacy fields are retained for backward compatibility with any callers/tests
    that may still reference them, but they are not used by the new algorithm.
    """

    # Smoothing state
    smoothed_x: float = 0.0
    smoothed_y: float = 0.0
    last_timestamp: float = 0.0
    initialized: bool = False

    # Cooldown state
    last_gesture_time: float = 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return math.sqrt(dx * dx + dy * dy)


def detect_gestures(
    trajectory: list[GazeSample] | tuple[GazeSample, ...],
    state: DetectorState,
    config: CompassSettings,
    *,
    visualize: bool = True,
    visualize_window_name: str = "SynNeon - Processed Trajectory",
    visualize_size: tuple[int, int] = (512, 512),
    visualize_stroke: int = 2,
    visualize_pixels_per_unit: float = 512.0,
) -> GestureEvent | None:
    """
    Detect "center -> direction" gestures using the provided trajectory window.

    Intuition / "natural feel":
    - People tend to glance away from their current fixation ("center") toward a direction,
      hold briefly, then often return or stabilize.
    - Instead of looking for high velocity spikes, we look for a trajectory that:
        1) starts near a stable center (earlier portion of the window),
        2) ends significantly away from that center (later portion),
        3) has a dominant direction (horizontal or vertical),
        4) is not too jittery (path efficiency / monotonicity heuristics),
        5) respects cooldown.

    Visualization (live window):
      - If `visualize=True`, this function will render the *processed/smoothed* trajectory
        into a live OpenCV window as a black polyline on a white background.
      - Close the window or press `q` while the window is focused to stop updates (window closes).

    Inputs:
      - `trajectory` should be time-ordered and represent ~last N seconds
        (controlled upstream by `sample_time_length`).
      - Coordinates are assumed normalized-ish (0..1), but thresholds are relative.

    Returns:
      - `GestureEvent` if a gesture is detected, else `None`.
    """

    if not trajectory:
        return None

    # Filter to worn-only samples; if any non-worn sample is present, reset smoothing.
    worn_samples: list[GazeSample] = []
    for s in trajectory:
        if not s.is_worn:
            state.initialized = False
            # We don't early-return: user might have re-worn within this window,
            # but practically we ignore the whole window for safety.
            return None
        worn_samples.append(s)

    if len(worn_samples) < 4:
        return None

    # --- Step 1: smooth the trajectory (simple sliding moving-average over this window) ---
    # This is per-window (no cross-window state), computationally cheap, and stable.
    #
    # Window size heuristic:
    # - Map config.smoothing_factor in [0, 1] to a small odd kernel size.
    #   0.0 -> 1 (no smoothing), 1.0 -> 9 (strong smoothing)
    sf = config.smoothing_factor
    if sf < 0.0:
        sf = 0.0
    elif sf > 1.0:
        sf = 1.0
    k = 1 + 2 * int(round(sf * 4))  # 1,3,5,7,9
    half = k // 2

    xs = [s.x for s in worn_samples]
    ys = [s.y for s in worn_samples]
    ts = [s.timestamp for s in worn_samples]

    # Prefix sums for O(n) moving average
    px = [0.0]
    py = [0.0]
    for x in xs:
        px.append(px[-1] + x)
    for y in ys:
        py.append(py[-1] + y)

    smoothed: list[tuple[float, float, float]] = []  # (x, y, timestamp)
    n_samples = len(worn_samples)
    for i in range(n_samples):
        lo = i - half
        hi = i + half
        if lo < 0:
            lo = 0
        if hi >= n_samples:
            hi = n_samples - 1
        # prefix sums are 1-based
        count = hi - lo + 1
        sx = px[hi + 1] - px[lo]
        sy = py[hi + 1] - py[lo]
        smoothed.append((sx / count, sy / count, ts[i]))

    # Mark smoothing initialized for downstream code paths that might rely on it,
    # but do not use it for window smoothing.
    state.initialized = True
    state.last_timestamp = ts[-1]

    if len(smoothed) < 4:
        return None

    # --- Optional: live visualization of the processed/smoothed trajectory (OpenCV) ---
    if visualize:
        if cv2 is None or np is None:
            raise RuntimeError("visualize=True requires opencv-python and numpy to be installed.")
        width, height = visualize_size
        width = max(1, int(width))
        height = max(1, int(height))
        stroke = max(1, int(visualize_stroke))

        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        cx = width // 2
        cy = height // 2

        # Every window is rendered "starting from center" while preserving a fixed scale
        # across windows (no per-window normalization, no fitting-to-canvas).
        #
        # Interpretation:
        # - We translate the trajectory so the FIRST point is (0,0)
        # - Then draw that translated polyline centered at (cx, cy)
        #
        # This means:
        # - Large movements look large, small movements look small
        # - Different windows are comparable (same pixels-per-unit scale)
        base_x, base_y, _ = smoothed[0]
        rel = [(x - base_x, y - base_y) for (x, y, _t) in smoothed]

        # Fixed mapping (units -> pixels). Keep this constant to preserve scale.
        ppu = float(visualize_pixels_per_unit)
        if ppu <= 0:
            ppu = 512.0

        def to_px(rx: float, ry: float) -> tuple[int, int]:
            px = int(round(cx + rx * ppu))
            py = int(round(cy + ry * ppu))
            # Clip to canvas to avoid exceptions
            if px < 0:
                px = 0
            elif px >= width:
                px = width - 1
            if py < 0:
                py = 0
            elif py >= height:
                py = height - 1
            return px, py

        pts = [to_px(rx, ry) for (rx, ry) in rel]

        # Draw center crosshair for reference
        cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), (200, 200, 200), 1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), (200, 200, 200), 1, lineType=cv2.LINE_AA)

        # Draw the polyline
        for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            cv2.line(canvas, (x0, y0), (x1, y1), (0, 0, 0), thickness=stroke, lineType=cv2.LINE_AA)

        # If the path collapsed to a single point (or everything clipped), draw a dot at center.
        if len(pts) == 1 or all(p == pts[0] for p in pts):
            cv2.circle(canvas, (cx, cy), max(2, stroke), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.imshow(visualize_window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(visualize_window_name)

    end_x, end_y, end_t = smoothed[-1]

    # Cooldown gate (natural: avoid double-triggers while the user is still settling gaze)
    if (end_t - state.last_gesture_time) < config.cooldown:
        return None

    # --- Step 2: estimate a "center" from the early portion of the window ---
    # Use first third of samples (at least 2). Median is robust to outliers.
    n = len(smoothed)
    early_n = max(2, n // 3)
    early_xs = [p[0] for p in smoothed[:early_n]]
    early_ys = [p[1] for p in smoothed[:early_n]]
    center_x = _median(early_xs)
    center_y = _median(early_ys)

    # --- Step 3: compute displacement from center to end, and require meaningful magnitude ---
    dx = end_x - center_x
    dy = end_y - center_y
    displacement = math.sqrt(dx * dx + dy * dy)

    # Natural-feel threshold:
    # reuse min_displacement as the minimum required "glance away from center".
    if displacement < config.min_displacement:
        return None

    # --- Step 4: require a dominant axis (avoid diagonals / noisy transitions) ---
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    # AXIS_BIAS = 1.0 means we just take the max.
    # The user asked for "magnitude... in one of the 4 canonical directions".
    # We allow a small bias to prefer decisive moves, or just 1.0.
    # We'll stick to a small bias (1.2) to avoid ambiguous diagonals triggering random directions.
    AXIS_BIAS = 1.2

    dominant: GestureType | None = None
    if abs_dx >= abs_dy * AXIS_BIAS:
        dominant = GestureType.RIGHT if dx > 0 else GestureType.LEFT
    elif abs_dy >= abs_dx * AXIS_BIAS:
        # Coordinate convention: y increases downward => +dy is DOWN
        dominant = GestureType.DOWN if dy > 0 else GestureType.UP
    else:
        return None

    # --- Step 5: emit gesture ---
    # (Simplified: removed path efficiency, stability, and consistency checks)
    duration = smoothed[-1][2] - smoothed[0][2]
    state.last_gesture_time = end_t

    return GestureEvent(
        type=dominant,
        timestamp=end_t,
        magnitude=displacement,
        duration=max(0.0, duration),
    )
